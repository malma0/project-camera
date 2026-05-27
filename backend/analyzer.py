"""
analyzer.py — детектор занятости столов (v3, переписано с нуля).

Архитектура:
  • 2 RTSP-потока, каждый в своём CaptureThread (drop-old-keep-newest).
  • Каждые ~0.5 сек на каждой камере делается tick:
        - YOLO11s (OpenVINO INT8) ищет людей
        - MOG2 даёт маску движения
        - для каждого стола камеры считается "сигнал занятости" этого тика
  • Сигналы складываются в скользящее окно 60 с на каждый стол.
  • Решение start/stop сессии — гистерезисное, с асимметричными порогами
    (FP дороже FN: высокий порог входа, низкий выхода, в сомнении — пусто).
  • Сессии пишутся в SQLite, активные — в data/active.json (для дашборда).
  • После 22:05 (свет в зале выключают) все сессии принудительно закрываются,
    новые не открываются до утра.

Контракт с дашбордом (app.py) — НЕ менять:
  • data/active.json формата {"<table_id>": {"start": <unix_ts>}, ...}
  • data/snapshots/cam_<id>.jpg — JPEG последний кадр камеры
  • таблица sessions(table_id, start_time, end_time, duration_seconds),
    timestamps формата 'YYYY-MM-DD HH:MM:SS.ffffff'.
"""

import os
import sys
import json
import time
import math
import sqlite3
import threading
import logging
from collections import deque
from datetime import datetime, time as dtime

import cv2
import numpy as np

# Проектные модули
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERAS  # список словарей: {id, name, rtsp, tables: [id,...]}

# ------------------------------------------------------------------ #
#  ПУТИ И КОНСТАНТЫ
# ------------------------------------------------------------------ #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
SNAP_DIR     = os.path.join(DATA_DIR, 'snapshots')
ZONES_FILE   = os.path.join(DATA_DIR, 'zones.json')
HOMOG_FILE   = os.path.join(DATA_DIR, 'homography.json')
RULES_FILE   = os.path.join(DATA_DIR, 'table_rules.json')
ACTIVE_FILE  = os.path.join(DATA_DIR, 'active.json')
DB_FILE      = os.path.join(DATA_DIR, 'stats.db')

# Модели — пытаемся OpenVINO INT8, потом FP32, потом PyTorch
MODEL_DIR        = os.path.join(PROJECT_ROOT, 'models')
MODEL_INT8_PATH  = os.path.join(MODEL_DIR, 'yolo11s_int8_openvino_model')
MODEL_FP32_PATH  = os.path.join(MODEL_DIR, 'yolo11s_openvino_model')
MODEL_PT_PATH    = os.path.join(MODEL_DIR, 'yolo11s.pt')

# Класс "person" в COCO
PERSON_CLASS_ID  = 0

# Геометрия стола (мм) и асимметричное расширение зоны "у стола"
TABLE_LENGTH_MM  = 2740       # длинная сторона, ось far↔near
TABLE_WIDTH_MM   = 1525       # короткая сторона (торец где подают), ось left↔right
EXPAND_LONG_MM   = 800        # расширение за торцы (направление far↔near) — где стоят игроки
EXPAND_SHORT_MM  = 300        # расширение в стороны (направление left↔right) — мало, чтоб не цеплять соседей
# Точечное сужение зон ТОЛЬКО для столов которые стоят плотно к соседям.
# Ключ — table_id (int), значение — {'long': мм, 'short': мм}.
# Эти числа ПЕРЕОПРЕДЕЛЯЮТ EXPAND_LONG_MM / EXPAND_SHORT_MM только для своих столов.
# Остальные столы используют дефолтные 800/300 — их трогать не надо.
#
# 5↔6 на cam1 — близкие соседи по оси x (стол5 кончается x≈7569, стол6 начинается x≈7732).
# 7↔8↔9 на cam2 — плотная цепочка вдоль x (9 кончается x≈6877, 8 начинается x≈7482,
#   8 кончается x≈8733, 7 начинается x≈9592). Запас всего ~600-900мм между торцами.
# Поэтому для них уменьшаем long-expand (вдоль x), short оставляем как у дефолта.
TABLE_EXPAND_OVERRIDES = {
    5: {'long': 400, 'short': 300},
    6: {'long': 400, 'short': 300},
    # Стол 7 — крайний на cam2, за правым торцом (x > 10804) стоит скамейка.
    # expand_long=0, expand_short=0 — зона = ровно реальные границы стола.
    # Это гарантирует что люди на скамейке (x≈10847) не попадают в зону.
    # Занятость определяется motion_as_primary=true в table_rules — только движение мяча.
    7: {'long': 0, 'short': 0},
    8: {'long': 400, 'short': 300},
    9: {'long': 400, 'short': 300},
}

# Тайминг
TICK_INTERVAL_S        = 0.5      # как часто анализируем (2 fps на каждую камеру)
WINDOW_SECONDS         = 60       # скользящее окно агрегации сигналов на стол
WINDOW_SIZE            = int(WINDOW_SECONDS / TICK_INTERVAL_S)  # 120 тиков

# Гистерезис: доля "занятых" тиков в окне для входа/выхода.
# В пустом окне ratio = 0. В полностью занятом = 1.
START_RATIO            = 0.65     # ≥65% подтверждений за 60с → старт
STOP_RATIO             = 0.20     # <20% подтверждений за 60с → стоп
# Между ними — состояние сохраняется (это и есть гистерезис).

# Минимальная длина сессии, иначе не пишем в БД (отсекаем артефакты)
MIN_SESSION_SECONDS    = 60

# YOLO
YOLO_CONF              = 0.35     # порог уверенности детектора людей
YOLO_IMGSZ             = 640

# Свет выключают в этот момент — после этого сессии закрываются принудительно
LIGHTS_OFF             = dtime(22, 5)
LIGHTS_ON              = dtime(7, 0)   # утром раньше этого не открываем

# MOG2
MOG2_HISTORY           = 500
MOG2_VAR_THRESHOLD     = 24
MOG2_LEARNING_RATE     = -1       # авто
# Доля движущихся пикселей в маске стола, ниже которой считаем что движения нет
MOTION_RATIO_THRESHOLD = 0.015    # 1.5% площади маски стола
# Фильтр blob'ов сетки: слишком тонкие/длинные регионы выбрасываем
NET_MAX_ASPECT_RATIO   = 6.0      # h/w > 6 → скорее всего сетка, не игрок
# Если у стола ПОДРЯД столько тиков стоит хотя бы один человек — считаем стол
# занятым даже если motion ниже порога (сетка съела движение / 1 игрок спокойно
# подаёт / противник пошёл за мячом). 6 тиков при 0.5с = 3 секунды стабильного
# присутствия. Это закрывает кейс "один игрок не помечается".
STABLE_PRESENCE_TICKS  = 6
# Сглаживание YOLO-детекций: если человек был замечен в зоне в любой из последних
# YOLO_SMOOTHING_TICKS тиков — считаем что он там и сейчас.
# 2 тика = 1 сек — достаточно чтобы пережить разовый плохой кадр YOLO,
# но НЕ так долго чтобы человек успел пройти через зону соседнего стола
# пока идёт в раздевалку (это вызывало ложную занятость на всех столах).
# Было 12 (6 сек) — это слишком много: пока человек шёл по залу его позиция
# 6 сек держалась в истории → все столы мимо которых он прошёл становились "занятыми".
YOLO_SMOOTHING_TICKS   = 2

# Снапшоты для дашборда — чаще обновляем для плавности
SNAPSHOT_EVERY_S       = 2

# Evidence-снимки для доказательной базы:
# Сохраняются только в рабочее время (между EVIDENCE_HOUR_START и EVIDENCE_HOUR_END).
# Хранятся EVIDENCE_KEEP_DAYS дней.
EVIDENCE_DIR           = os.path.join(DATA_DIR, 'evidence')
EVIDENCE_INTERVAL_S    = 15 * 60    # 15 минут
EVIDENCE_KEEP_DAYS     = 90
EVIDENCE_HOUR_START    = 10   # с 10:00
EVIDENCE_HOUR_END      = 22   # до 22:00

# Склейка сессий: если STOP и следующий START для одного стола разделены
# менее чем SESSION_MERGE_GRACE_S секунд — считаем их одной непрерывной сессией.
# Игрок отошёл попить, шарики собрать, поговорить — это не конец сессии.
# ВАЖНО: пауза между уходом и возвращением в БД НЕ идёт. duration_seconds =
# (время до отхода) + (время после возвращения). end_time - start_time будет
# больше чем duration_seconds — это значит был перерыв, и он не в счёт.
SESSION_MERGE_GRACE_S  = 4 * 60   # 4 минуты

# Калибровочный датасет для OpenVINO INT8 квантизации.
# Включается переменной окружения COLLECT_CALIBRATION=1.
# Сохраняет каждый Nth тик в data/calibration_dataset/cam{id}_{ts}.jpg.
# Цель: 200-300 разнообразных кадров за час пик с обеих камер.
COLLECT_CALIBRATION    = os.environ.get('COLLECT_CALIBRATION', '0') == '1'
CALIB_DIR              = os.path.join(DATA_DIR, 'calibration_dataset')
CALIB_EVERY_N_TICKS    = 5   # каждый 5-й тик при 2 fps = 1 кадр / 2.5 сек / камера
CALIB_MAX_FILES        = 800 # увеличено: ~400 на камеру
CALIB_MAX_PER_CAM      = 400 # максимум на одну камеру

# Диагностический лог — включается через переменную окружения:
#   Windows:  set DEBUG_TABLES=6,7  && python backend/analyzer.py
#   Linux:    DEBUG_TABLES=6,7 python backend/analyzer.py
# По умолчанию ВЫКЛЮЧЕН — в продакшене не нужен, только засоряет лог.
# Включай только когда конкретный стол ведёт себя странно и надо понять почему.
_debug_env = os.environ.get('DEBUG_TABLES', '').strip()
DEBUG_TABLES_SET = set(int(x) for x in _debug_env.split(',') if x.strip().isdigit())
DEBUG_LOG_EVERY_N_TICKS = 20  # при tick=0.5s это раз в 10 секунд

# ------------------------------------------------------------------ #
#  ЛОГИРОВАНИЕ
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('analyzer')

# ------------------------------------------------------------------ #
#  ЗАГРУЗКА КОНФИГОВ
# ------------------------------------------------------------------ #

def load_zones():
    with open(ZONES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_homographies():
    """Возвращает {camera_id (int): {'H': matrix px→floor, 'H_inv': matrix floor→px}}."""
    with open(HOMOG_FILE, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    out = {}
    for cam_id_str, H_list in raw.items():
        H = np.array(H_list, dtype=np.float64)
        H_inv = np.linalg.inv(H)
        out[int(cam_id_str)] = {'H': H, 'H_inv': H_inv}
    return out

def load_rules():
    """Объединяет default-правила с per-table override'ами.
    Игнорирует ключи начинающиеся на '_' (комментарии) и ключи которые не int."""
    with open(RULES_FILE, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    default = raw.get('default', {})
    # выкидываем из default служебные ключи если затесались
    default = {k: v for k, v in default.items() if not k.startswith('_')}
    rules = {}
    for k, v in raw.items():
        if k == 'default' or k.startswith('_'):
            continue
        try:
            tid = int(k)
        except ValueError:
            log.warning(f'Skipping non-integer rule key: {k!r}')
            continue
        merged = dict(default)
        merged.update({kk: vv for kk, vv in v.items() if not kk.startswith('_')})
        rules[tid] = merged
    return default, rules

def rule_for(table_id, default, rules):
    return rules.get(int(table_id), dict(default))

# ------------------------------------------------------------------ #
#  ГЕОМЕТРИЯ: расширение зоны стола в координатах пола
# ------------------------------------------------------------------ #

def expand_floor_polygon(points_floor, expand_long=None, expand_short=None):
    """
    Расширяет 4-угольник стола асимметрично:
      • +expand_long по длинной оси (far↔near, где стоят игроки)
      • +expand_short по короткой оси (left↔right, бока стола)

    Если expand_long/expand_short не заданы — берём глобальные EXPAND_LONG_MM /
    EXPAND_SHORT_MM. Используется чтобы для плотно стоящих столов (5/6, 7/8/9)
    задать меньшие значения через TABLE_EXPAND_OVERRIDES, не трогая остальные.

    Вход — список из 4 точек [far-left, far-right, near-right, near-left] в мм.
    Алгоритм:
      1. Центр стола = среднее 4 точек.
      2. Локальные оси:
         long_axis  = (near_mid - far_mid) нормированный  (вдоль длинной стороны)
         short_axis = (right_mid - left_mid) нормированный (вдоль короткой стороны)
      3. Сдвигаем каждый угол вдоль обеих осей в "наружном" направлении.

    Это устойчиво к произвольной ориентации стола в координатах пола
    (столы 12 и 1 у вас стоят зеркально/повёрнуто, см. zones.json).
    """
    if expand_long is None:
        expand_long = EXPAND_LONG_MM
    if expand_short is None:
        expand_short = EXPAND_SHORT_MM
    pts = np.array(points_floor, dtype=np.float64)
    fl, fr, nr, nl = pts[0], pts[1], pts[2], pts[3]

    far_mid   = (fl + fr) / 2.0
    near_mid  = (nl + nr) / 2.0
    left_mid  = (fl + nl) / 2.0
    right_mid = (fr + nr) / 2.0

    long_vec  = near_mid - far_mid          # far → near
    short_vec = right_mid - left_mid        # left → right

    long_norm  = long_vec  / (np.linalg.norm(long_vec)  + 1e-9)
    short_norm = short_vec / (np.linalg.norm(short_vec) + 1e-9)

    # Для каждого угла: его смещение от центра разложим по long/short и
    # сдвинем "от центра" вдоль каждой оси на соответствующий expand.
    center = pts.mean(axis=0)
    expanded = []
    for p in pts:
        v = p - center
        # знак вдоль каждой оси (далеко/близко, влево/вправо)
        s_long  = np.sign(np.dot(v, long_norm))
        s_short = np.sign(np.dot(v, short_norm))
        if s_long  == 0: s_long  = 1
        if s_short == 0: s_short = 1
        expanded.append(p
                        + s_long  * expand_long  * long_norm
                        + s_short * expand_short * short_norm)
    return np.array(expanded, dtype=np.float64)

def floor_polygon_to_pixels(floor_pts, H_inv):
    """Перевод точек пола (мм) → пиксели через обратную гомографию."""
    src = floor_pts.astype(np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H_inv.astype(np.float32))
    return dst.reshape(-1, 2)

def point_in_polygon(point_xy, polygon_pts):
    """polygon_pts: ndarray (N,2). Возвращает True если внутри/на границе."""
    poly = polygon_pts.astype(np.float32)
    res = cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False)
    return res >= 0

def pixel_to_floor(point_px, H):
    """Один пиксель → пол."""
    src = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H.astype(np.float32))
    return (float(dst[0][0][0]), float(dst[0][0][1]))

# ------------------------------------------------------------------ #
#  ЗАХВАТ RTSP
# ------------------------------------------------------------------ #

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

class RTSPCapture(threading.Thread):
    """
    Читает RTSP в отдельном потоке, всегда отдаёт ПОСЛЕДНИЙ кадр
    (drop-old-keep-newest, чтобы анализатор не отставал от реального времени).
    Автоматический reconnect при обрыве.
    """
    def __init__(self, camera_id, rtsp_url):
        super().__init__(daemon=True, name=f'cap-{camera_id}')
        self.camera_id = camera_id
        self.rtsp_url  = rtsp_url
        self._lock     = threading.Lock()
        self._frame    = None
        self._frame_ts = 0.0
        self._stop     = threading.Event()

    def run(self):
        log.info(f'[cam {self.camera_id}] capture starting')
        backoff = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning(f'[cam {self.camera_id}] open failed, retry in {backoff:.0f}s')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            log.info(f'[cam {self.camera_id}] connected')
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    log.warning(f'[cam {self.camera_id}] read failed, reconnecting')
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_ts = time.time()
            cap.release()
            time.sleep(1.0)
        log.info(f'[cam {self.camera_id}] capture stopped')

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_ts

    def stop(self):
        self._stop.set()

# ------------------------------------------------------------------ #
#  YOLO DETECTOR
# ------------------------------------------------------------------ #

class PersonDetector:
    """
    Обёртка над ultralytics YOLO. Пытается грузить OpenVINO INT8 → FP32 → .pt.
    Возвращает список (x1,y1,x2,y2,conf) для класса 'person'.
    """
    def __init__(self):
        from ultralytics import YOLO

        if os.path.isdir(MODEL_INT8_PATH):
            log.info(f'Loading YOLO from {MODEL_INT8_PATH} (OpenVINO INT8)')
            self.model = YOLO(MODEL_INT8_PATH, task='detect')
        elif os.path.isdir(MODEL_FP32_PATH):
            log.info(f'Loading YOLO from {MODEL_FP32_PATH} (OpenVINO FP32)')
            self.model = YOLO(MODEL_FP32_PATH, task='detect')
        elif os.path.isfile(MODEL_PT_PATH):
            log.warning(f'OpenVINO model not found, falling back to PyTorch: {MODEL_PT_PATH}')
            self.model = YOLO(MODEL_PT_PATH)
        else:
            raise FileNotFoundError(
                f'No model found. Expected one of:\n'
                f'  {MODEL_INT8_PATH}\n  {MODEL_FP32_PATH}\n  {MODEL_PT_PATH}\n'
                f'Run scripts/convert_to_openvino.py'
            )

    def detect_people(self, frame):
        res = self.model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]
        out = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out

# ------------------------------------------------------------------ #
#  MOG2 BACKGROUND SUBTRACTION
# ------------------------------------------------------------------ #

class MotionDetector:
    """Один MOG2 на камеру. Возвращает бинарную маску движения того же размера."""
    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )

    def mask(self, frame):
        m = self.mog.apply(frame, learningRate=MOG2_LEARNING_RATE)
        # морфология: убираем шум, склеиваем
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return m

# ------------------------------------------------------------------ #
#  ЗОНА СТОЛА (расширенная) + быстрая проверка попадания и движения
# ------------------------------------------------------------------ #

class TableZone:
    """
    Гео-данные одного стола в системе конкретной камеры.
      • polygon_floor_expanded — расширенный 4-угольник на полу (мм), для bbox-теста
      • polygon_pixel_expanded — он же спроецированный в пиксели (для маски движения)
      • pixel_mask — заранее построенная бинарная маска (frame_h × frame_w),
        с которой быстро считать движение через bitwise_and.
    """
    def __init__(self, table_id, floor_pts, H, H_inv, frame_shape,
                 expand_long=None, expand_short=None):
        self.table_id = table_id
        self.H = H
        self.H_inv = H_inv

        self.polygon_floor_expanded = expand_floor_polygon(
            floor_pts, expand_long=expand_long, expand_short=expand_short,
        )
        self.polygon_pixel_expanded = floor_polygon_to_pixels(
            self.polygon_floor_expanded, H_inv,
        )

        # Маска того же размера что и кадр (для backsub). Полигон может вылезать
        # за края кадра — это ок, fillPoly клипит.
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts_int = self.polygon_pixel_expanded.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts_int], 255)
        self.pixel_mask = mask
        self.mask_area  = int(np.count_nonzero(mask))  # для нормализации motion ratio

    def contains_floor_point(self, floor_xy):
        return point_in_polygon(floor_xy, self.polygon_floor_expanded)

    def motion_ratio(self, motion_mask):
        """Доля движущихся пикселей в зоне стола, после фильтра сетки."""
        if self.mask_area == 0:
            return 0.0
        local = cv2.bitwise_and(motion_mask, motion_mask, mask=self.pixel_mask)

        # Фильтр сетки: ищем connected components, выбрасываем те у которых
        # bbox имеет очень большое отношение сторон (длинные тонкие вертикальные регионы).
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
        good = 0
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area < 30:
                continue
            ar = max(h, w) / max(1, min(h, w))
            if ar > NET_MAX_ASPECT_RATIO:
                continue  # похоже на сетку — игнор
            good += area
        return good / self.mask_area

# ------------------------------------------------------------------ #
#  СОСТОЯНИЕ СТОЛА: скользящее окно + гистерезисная сессия
# ------------------------------------------------------------------ #

class TableState:
    """
    На каждый тик принимает 0 или 1 (была занятость в этом тике или нет),
    держит окно WINDOW_SIZE последних значений, считает ratio занятости,
    управляет состоянием сессии гистерезисом.
    """
    def __init__(self, table_id, rule):
        self.table_id = table_id
        self.rule = rule
        self.window = deque(maxlen=WINDOW_SIZE)
        self.is_active = False
        self.session_start_ts = None
        # анти-флап: запоминаем время последнего изменения состояния
        self._last_change_ts = 0.0

    def tick(self, occupied_now, now_ts):
        """Возвращает событие: None | ('start', ts) | ('stop', start_ts, end_ts)."""
        self.window.append(1 if occupied_now else 0)

        # ratio считаем только когда окно набралось хотя бы наполовину,
        # иначе при старте процесса будут ложные срабатывания.
        if len(self.window) < WINDOW_SIZE // 2:
            return None

        ratio = sum(self.window) / len(self.window)

        # пороги per-table из правил, если заданы — иначе глобальные
        start_th = self.rule.get('window_start_ratio', START_RATIO)
        stop_th  = self.rule.get('window_stop_ratio',  STOP_RATIO)

        event = None
        if not self.is_active and ratio >= start_th:
            self.is_active = True
            # старт сессии помечаем "задним числом" — на ~окно назад,
            # но не раньше последнего перехода. Это даёт честную длительность.
            self.session_start_ts = max(now_ts - WINDOW_SECONDS * start_th,
                                        self._last_change_ts)
            self._last_change_ts = now_ts
            event = ('start', self.session_start_ts)
        elif self.is_active and ratio <= stop_th:
            start_ts = self.session_start_ts
            # конец сессии — "когда сигнал упал", оцениваем как сейчас минус
            # (1 - stop_th) * window (т.е. сколько уже было пусто)
            end_ts = now_ts - WINDOW_SECONDS * (1 - stop_th) * 0.5
            end_ts = max(end_ts, start_ts + 1)
            self.is_active = False
            self.session_start_ts = None
            self._last_change_ts = now_ts
            event = ('stop', start_ts, end_ts)
        return event

    def force_stop(self, end_ts):
        """Принудительный стоп (свет погас / выключение). Возвращает событие или None."""
        if self.is_active and self.session_start_ts is not None:
            start_ts = self.session_start_ts
            self.is_active = False
            self.session_start_ts = None
            self.window.clear()
            return ('stop', start_ts, max(end_ts, start_ts + 1))
        return None

# ------------------------------------------------------------------ #
#  БД И ACTIVE.JSON
# ------------------------------------------------------------------ #

def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def write_session(table_id, start_ts, end_ts, billed_seconds=None):
    """
    Пишет сессию в БД.
      start_ts — время первого START (начало цепочки)
      end_ts   — время последнего STOP (конец цепочки)
      billed_seconds — реальное игровое время БЕЗ пауз (для склеенных сессий).
                       Если None — равно end_ts - start_ts (нет пауз).
    """
    span = int(end_ts - start_ts)
    duration = int(billed_seconds) if billed_seconds is not None else span
    if duration < MIN_SESSION_SECONDS:
        log.info(f'[t{table_id}] session {duration}s < {MIN_SESSION_SECONDS}s — discard')
        return
    start_str = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S.%f')
    end_str   = datetime.fromtimestamp(end_ts  ).strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        conn = db_connect()
        conn.execute(
            "INSERT INTO sessions (table_id, start_time, end_time, duration_seconds) "
            "VALUES (?, ?, ?, ?)",
            (int(table_id), start_str, end_str, duration),
        )
        conn.commit()
        conn.close()
        gap_info = f' (span={span}s, billed={duration}s, paused={span-duration}s)' if span != duration else ''
        log.info(f'[t{table_id}] session saved: {duration}s{gap_info} ({start_str} → {end_str})')
    except Exception as e:
        log.exception(f'[t{table_id}] DB write failed: {e}')

_active_lock = threading.Lock()

def update_active_file(active_sessions):
    """active_sessions: dict {table_id: start_ts}."""
    with _active_lock:
        payload = {str(tid): {'start': float(ts)} for tid, ts in active_sessions.items()}
        tmp = ACTIVE_FILE + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp, ACTIVE_FILE)
        except Exception as e:
            log.exception(f'active.json write failed: {e}')

# ------------------------------------------------------------------ #
#  ВРЕМЕННОЕ ОКНО РАБОТЫ ЗАЛА
# ------------------------------------------------------------------ #

def lights_are_on(now=None):
    """Считаем что зал работает между LIGHTS_ON и LIGHTS_OFF локального времени."""
    now = now or datetime.now()
    t = now.time()
    return LIGHTS_ON <= t < LIGHTS_OFF

# ------------------------------------------------------------------ #
#  ГЛАВНЫЙ ЦИКЛ НА ОДНУ КАМЕРУ
# ------------------------------------------------------------------ #

class CameraWorker(threading.Thread):
    """
    Один поток на камеру: тикает раз в TICK_INTERVAL_S,
    обновляет окна состояний всех столов этой камеры, пишет события.

    detector передаётся снаружи (общий на весь процесс, потокобезопасен).
    active_sessions — общий dict {table_id: start_ts}, защищён active_lock.
    """
    def __init__(self, camera_cfg, zones, homography, default_rule, rules,
                 detector, active_sessions, active_lock):
        super().__init__(daemon=True, name=f'worker-{camera_cfg["id"]}')
        self.cam_id     = camera_cfg['id']
        self.cam_name   = camera_cfg.get('name', f'cam{self.cam_id}')
        self.rtsp_url   = camera_cfg['rtsp']
        self.tables_ids = camera_cfg.get('tables', [])
        self.detector   = detector
        self.active_sessions = active_sessions
        self.active_lock     = active_lock

        # фильтруем зоны, относящиеся к этой камере
        self.zones_raw = {
            int(tid): z for tid, z in zones.items()
            if int(z['camera_id']) == self.cam_id and int(tid) in self.tables_ids
        }
        self.homog = homography[self.cam_id]
        self.default_rule = default_rule
        self.rules = rules

        # объекты состояния по каждому столу
        self.states = {
            tid: TableState(tid, rule_for(tid, default_rule, rules))
            for tid in self.zones_raw
        }
        # geo-объекты создаём при получении первого кадра (нужен shape)
        self.zone_geo = {}      # {tid: TableZone}
        self.motion   = MotionDetector()

        self.capture = RTSPCapture(self.cam_id, self.rtsp_url)
        self._stop   = threading.Event()
        self._last_snapshot_ts = 0.0
        self._tick_counter = 0  # для калибровочного семплинга
        self._last_evidence_ts = 0.0  # для evidence-снимков
        # Склейка сессий. Формат:
        #   _pending_stops[tid] = (orig_start_ts, last_stop_ts, accumulated_billed_s)
        # orig_start_ts — самый первый старт цепочки склеенных сегментов
        # last_stop_ts — когда последний раз закрылись (для расчёта grace)
        # accumulated_billed_s — реальное игровое время (БЕЗ пауз)
        self._pending_stops = {}
        # активные сессии этой камеры с учётом склейки. Формат:
        #   _active_meta[tid] = (orig_start_ts, billed_before_this_segment_s)
        # billed_before — сколько уже накапало в предыдущих сегментах
        # текущий сегмент = now - <последний START этого сегмента>
        self._active_meta = {}
        # Счётчик стабильного присутствия: сколько ПОДРЯД тиков у стола tid стоял
        # хотя бы один человек. Используется чтобы спасти случай "1 игрок стоит
        # спокойно, медленно подаёт" — motion < threshold, но он точно играет.
        # При STABLE_PRESENCE_TICKS подряд считаем стол занятым даже без motion.
        self._stable_presence = {tid: 0 for tid in self.zones_raw}
        # История последних YOLO_SMOOTHING_TICKS детекций на каждом столе:
        # каждый элемент — кол-во людей в зоне в этом тике. Если хоть в одном из
        # последних N тиков было >=1 человек — считаем что есть и сейчас.
        # Спасает от моргания YOLO на дальних столах (6, 7).
        self._detect_history = {tid: deque(maxlen=YOLO_SMOOTHING_TICKS)
                                for tid in self.zones_raw}

    def stop(self):
        self._stop.set()

    def _ensure_geo(self, frame_shape):
        if self.zone_geo:
            return
        for tid, z in self.zones_raw.items():
            # для столов из TABLE_EXPAND_OVERRIDES берём индивидуальные expand
            override = TABLE_EXPAND_OVERRIDES.get(tid, {})
            self.zone_geo[tid] = TableZone(
                tid, z['points_floor'],
                self.homog['H'], self.homog['H_inv'],
                frame_shape,
                expand_long=override.get('long'),
                expand_short=override.get('short'),
            )
        log.info(f'[cam {self.cam_id}] geo built for tables: {sorted(self.zone_geo.keys())}')

    def _save_snapshot(self, frame):
        path = os.path.join(SNAP_DIR, f'cam_{self.cam_id}.jpg')
        try:
            os.makedirs(SNAP_DIR, exist_ok=True)
            ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                log.warning(f'[cam {self.cam_id}] cv2.imwrite returned False for {path}')
        except Exception as e:
            log.exception(f'[cam {self.cam_id}] snapshot save failed: {e}')

    def _save_evidence_frame(self, frame):
        """Сохраняет кадр в data/evidence/YYYY-MM-DD/cam{N}/HH-MM.jpg с timestamp и занятыми столами."""
        try:
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H-%M')
            full_ts  = now.strftime('%Y-%m-%d %H:%M:%S')

            day_dir = os.path.join(EVIDENCE_DIR, date_str, f'cam{self.cam_id}')
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir, f'{time_str}.jpg')

            annotated = frame.copy()
            h, w = annotated.shape[:2]
            font      = cv2.FONT_HERSHEY_SIMPLEX
            scale     = max(0.6, w / 1920 * 0.9)
            thickness = max(1, int(w / 1920 * 2))
            margin    = int(w / 1920 * 20)

            # --- Timestamp (правый нижний угол) ---
            (tw, th), _ = cv2.getTextSize(full_ts, font, scale, thickness)
            x = w - tw - margin
            y = h - margin
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, full_ts, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # --- Занятые столы (левый нижний угол) ---
            # Берём столы этой камеры у которых active_session активна прямо сейчас
            busy_tables = sorted([
                tid for tid, state in self.states.items()
                if state.is_active
            ])
            if busy_tables:
                # формат: "Столы: 7 / 8 / 11"
                tables_text = 'Столы: ' + ' / '.join(str(t) for t in busy_tables)
                (btw, bth), _ = cv2.getTextSize(tables_text, font, scale, thickness)
                bx = margin
                by = h - margin
                overlay2 = annotated.copy()
                cv2.rectangle(overlay2,
                              (bx - 10, by - bth - 10),
                              (bx + btw + 10, by + 10),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.6, annotated, 0.4, 0, annotated)
                cv2.putText(annotated, tables_text, (bx, by), font, scale,
                            (0, 230, 80), thickness, cv2.LINE_AA)

            ok = cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                log.warning(f'[cam {self.cam_id}] evidence imwrite failed: {path}')
        except Exception as e:
            log.exception(f'[cam {self.cam_id}] evidence save failed: {e}')

    def _save_calibration_frame(self, frame):
        """Сохраняет кадр для INT8-калибровки. Cap на CALIB_MAX_PER_CAM на камеру."""
        try:
            os.makedirs(CALIB_DIR, exist_ok=True)
            prefix = f'cam{self.cam_id}_'
            existing = len([f for f in os.listdir(CALIB_DIR)
                           if f.startswith(prefix) and f.endswith('.jpg')])
            if existing >= CALIB_MAX_PER_CAM:
                return
            fn = f'{prefix}{int(time.time()*1000)}.jpg'
            path = os.path.join(CALIB_DIR, fn)
            ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                log.warning(f'[cam {self.cam_id}] cv2.imwrite returned False for calibration frame')
        except Exception as e:
            log.exception(f'[cam {self.cam_id}] calibration save failed: {e}')

    def _flush_expired_pending(self, now_ts):
        """Pending stops старше grace — пишем в БД как окончательно закрытые сессии."""
        if not self._pending_stops:
            return
        expired = []
        for tid, (orig_s, last_e, billed) in list(self._pending_stops.items()):
            if now_ts - last_e > SESSION_MERGE_GRACE_S:
                expired.append((tid, orig_s, last_e, billed))
        for tid, s, e, billed in expired:
            write_session(tid, s, e, billed_seconds=billed)
            del self._pending_stops[tid]
            log.info(f'[t{tid}] session finalized (grace expired)')

    def _tick(self):
        frame, frame_ts = self.capture.read()
        if frame is None:
            return
        self._ensure_geo(frame.shape)

        now_ts = time.time()
        lights = lights_are_on()

        # сначала — финализируем pending stops старше grace
        self._flush_expired_pending(now_ts)

        # снапшот для дашборда
        if now_ts - self._last_snapshot_ts >= SNAPSHOT_EVERY_S:
            self._save_snapshot(frame)
            self._last_snapshot_ts = now_ts

        # evidence-снимок раз в 15 минут, только в рабочем окне 10:00-22:00
        if (now_ts - self._last_evidence_ts >= EVIDENCE_INTERVAL_S):
            hour = datetime.now().hour
            if EVIDENCE_HOUR_START <= hour < EVIDENCE_HOUR_END:
                self._save_evidence_frame(frame)
                self._last_evidence_ts = now_ts

        # сбор калибровочного датасета (если включено через env)
        self._tick_counter += 1
        if COLLECT_CALIBRATION and (self._tick_counter % CALIB_EVERY_N_TICKS == 0):
            self._save_calibration_frame(frame)

        # Если света нет — никаких новых сигналов. Просто закрываем активные
        # сессии этой камеры если они ещё открыты.
        if not lights:
            for tid, st in self.states.items():
                ev = st.force_stop(now_ts)
                if ev is not None:
                    _, seg_start, seg_end = ev
                    meta = self._active_meta.pop(tid, None)
                    if meta is None:
                        write_session(tid, seg_start, seg_end)
                    else:
                        orig_start, billed_before, real_seg_start = meta
                        seg_duration = max(0.0, seg_end - real_seg_start)
                        write_session(tid, orig_start, seg_end,
                                      billed_seconds=billed_before + seg_duration)
                    with self.active_lock:
                        self.active_sessions.pop(tid, None)
                    update_active_file(self.active_sessions)
            # свет погас — pending больше не имеет смысла, финализируем всё
            for tid, (orig_s, last_e, billed) in list(self._pending_stops.items()):
                write_session(tid, orig_s, last_e, billed_seconds=billed)
            self._pending_stops.clear()
            return

        # --- YOLO: люди в кадре ---
        try:
            people = self.detector.detect_people(frame)
        except Exception as e:
            log.exception(f'[cam {self.cam_id}] YOLO failed: {e}')
            people = []

        # для каждого человека — нижняя точка bbox (между ног), в координатах пола
        people_on_floor = []
        for (x1, y1, x2, y2, conf) in people:
            foot_px = ((x1 + x2) / 2.0, y2)
            try:
                floor_xy = pixel_to_floor(foot_px, self.homog['H'])
            except Exception:
                continue
            people_on_floor.append(floor_xy)

        # --- MOG2: маска движения ---
        try:
            motion_mask = self.motion.mask(frame)
        except Exception as e:
            log.exception(f'[cam {self.cam_id}] MOG2 failed: {e}')
            motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # --- ASSIGN-TO-NEAREST: каждый человек засчитывается только ОДНОМУ столу ---
        # Раньше: если человек попал в расширенные полигоны двух соседних столов
        # (7/8, 8/9, 11/12 — стоят плотно), он считался ОБЕИМ → путаница.
        # Теперь: для каждого человека находим ближайший стол (по расстоянию
        # от точки на полу до центра полигона стола в мм), и считаем его ТОЛЬКО там.
        # Условие: человек ДОЛЖЕН попадать в расширенный полигон этого стола
        # (если стоит далеко от всех — никуда не считается).
        # people_per_table: {table_id: count}
        people_per_table = {tid: 0 for tid in self.zone_geo.keys()}
        for p in people_on_floor:
            best_tid = None
            best_dist = float('inf')
            for tid, geo in self.zone_geo.items():
                if not geo.contains_floor_point(p):
                    continue
                zc = geo.polygon_floor_expanded.mean(axis=0)
                d = (p[0] - zc[0]) ** 2 + (p[1] - zc[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid is not None:
                people_per_table[best_tid] += 1

        # --- для каждого стола: сигнал занятости этого тика ---
        for tid, geo in self.zone_geo.items():
            rule = self.states[tid].rule
            min_people = int(rule.get('min_stable_people', 1))
            require_movement = bool(rule.get('require_movement', True))

            # сколько людей в зоне стола — теперь из assign-to-nearest, не двойной счёт
            n_people_here = people_per_table[tid]
            # пишем в историю детекций (для сглаживания)
            self._detect_history[tid].append(n_people_here)
            # сглаженное значение: МАКСИМУМ за последние YOLO_SMOOTHING_TICKS тиков.
            # Если YOLO моргнул на 1-2 кадра, мы всё ещё видим людей из предыдущего тика.
            n_people_smoothed = max(self._detect_history[tid]) if self._detect_history[tid] else 0
            people_ok = n_people_smoothed >= min_people

            # доля движения в маске стола
            motion = geo.motion_ratio(motion_mask)
            # порог движения — может быть переопределён в table_rules per-table
            motion_threshold = float(rule.get('motion_threshold', MOTION_RATIO_THRESHOLD))
            motion_ok = motion >= motion_threshold
            # "хоть что-то шевелится" — порог в 5x ниже motion_threshold.
            # Используется в stable_presence_ok чтобы отличить "игрок стоит без движения"
            # от "игрок ушёл, motion=0". Без этой проверки stable_presence накапливается
            # до 300+/6 и держит стол занятым долго после реального ухода (баг столов 8,9,10).
            motion_any = motion >= (motion_threshold * 0.2)

            # обновляем счётчик стабильного присутствия (people_ok = есть кто стоять)
            if people_ok:
                self._stable_presence[tid] += 1
            else:
                self._stable_presence[tid] = 0

            # stable_presence_ok: человек стабильно стоит у стола И есть хоть какое-то
            # движение. Без motion_any условие — если игрок ушёл и motion=0, stable
            # держится от предыдущих тиков и стол остаётся "занятым" → ложный сигнал.
            stable_presence_ok = (
                self._stable_presence[tid] >= STABLE_PRESENCE_TICKS and motion_any
            )

            # motion_as_primary: для столов где YOLO почти никогда не видит игроков
            # (стол очень далеко / ракурс плохой — например стол 6 на cam1 где видна
            # только нога иногда). Motion даёт стабильный сигнал даже без детекции людей —
            # мяч летит, игрок двигается, пиксели меняются.
            # Задаётся в table_rules: "motion_as_primary": true
            motion_as_primary = bool(rule.get('motion_as_primary', False))

            # ПРАВИЛО ТИКА:
            # — motion_as_primary: занято ТОЛЬКО если есть motion на столе.
            #   stable_presence намеренно не используется: YOLO не видит игроков
            #   на этих столах, поэтому stable всегда 0. Зато motion чёткий сигнал —
            #   мяч летит = motion есть. Игра закончилась = motion=0 = свободно.
            # — require_movement: нужны И люди И motion, ИЛИ стабильное присутствие+motion_any
            # — иначе: достаточно людей (дальние столы без хорошего обзора)
            if motion_as_primary:
                occupied = motion_ok
            elif require_movement:
                occupied = (people_ok and motion_ok) or stable_presence_ok
            else:
                occupied = people_ok or (n_people_smoothed >= 1 and motion_ok)

            # диагностический лог: для столов из DEBUG_TABLES_SET пишем что видим
            if tid in DEBUG_TABLES_SET and (self._tick_counter % DEBUG_LOG_EVERY_N_TICKS == 0):
                window = self.states[tid].window
                window_ratio = sum(window) / max(1, len(window))
                is_active = self.states[tid].is_active
                log.info(f'[DEBUG t{tid}] people_in_zone={n_people_here} '
                         f'(smoothed={n_people_smoothed}) '
                         f'motion={motion:.4f}(thr={motion_threshold:.4f}) '
                         f'occupied_now={occupied} '
                         f'window={sum(window)}/{len(window)}={window_ratio:.2f} '
                         f'active={is_active} '
                         f'req_mov={require_movement} '
                         f'stable={self._stable_presence[tid]}/{STABLE_PRESENCE_TICKS} '
                         f'stable_ok={stable_presence_ok}')
                # если людей в зоне нет — покажем где они вообще и где зона
                if n_people_here == 0 and people_on_floor:
                    # центр зоны стола в координатах пола
                    zc = geo.polygon_floor_expanded.mean(axis=0)
                    z_min = geo.polygon_floor_expanded.min(axis=0)
                    z_max = geo.polygon_floor_expanded.max(axis=0)
                    log.info(f'[DEBUG t{tid}] zone_center=({zc[0]:.0f},{zc[1]:.0f}) '
                             f'zone_bbox=x[{z_min[0]:.0f}..{z_max[0]:.0f}] '
                             f'y[{z_min[1]:.0f}..{z_max[1]:.0f}]')
                    for i, p in enumerate(people_on_floor[:5]):  # не больше 5 людей в логе
                        dx = p[0] - zc[0]
                        dy = p[1] - zc[1]
                        log.info(f'[DEBUG t{tid}]   person{i}: floor=({p[0]:.0f},{p[1]:.0f}) '
                                 f'dist_from_zone_center=({dx:+.0f},{dy:+.0f})mm')

            event = self.states[tid].tick(occupied, now_ts)
            if event is None:
                continue
            if event[0] == 'start':
                _, start_ts = event
                # СКЛЕЙКА: есть ли в pending свежий STOP для этого стола?
                pending = self._pending_stops.get(tid)
                if pending is not None:
                    p_orig_start, p_stop_ts, p_billed = pending
                    gap = start_ts - p_stop_ts
                    if 0 <= gap <= SESSION_MERGE_GRACE_S:
                        # склеиваем: orig_start сохраняется, billed_before тоже,
                        # новый сегмент начнётся от start_ts
                        self._active_meta[tid] = (p_orig_start, p_billed, start_ts)
                        del self._pending_stops[tid]
                        log.info(f'[t{tid}] START (merged after {int(gap)}s pause, '
                                 f'billed_so_far={int(p_billed)}s, '
                                 f'people={n_people_here}, motion={motion:.3f})')
                    else:
                        # gap слишком большой — финализируем pending, стартуем новую
                        write_session(tid, p_orig_start, p_stop_ts, billed_seconds=p_billed)
                        del self._pending_stops[tid]
                        self._active_meta[tid] = (start_ts, 0.0, start_ts)
                        log.info(f'[t{tid}] START (people={n_people_here}, motion={motion:.3f})')
                else:
                    self._active_meta[tid] = (start_ts, 0.0, start_ts)
                    log.info(f'[t{tid}] START (people={n_people_here}, motion={motion:.3f})')

                with self.active_lock:
                    self.active_sessions[tid] = start_ts
                update_active_file(self.active_sessions)
                self.states[tid].session_start_ts = start_ts
            else:
                _, segment_start_ts, end_ts = event
                meta = self._active_meta.pop(tid, None)
                if meta is None:
                    # на всякий случай: если meta нет, обходимся без склейки
                    orig_start = segment_start_ts
                    billed_before = 0.0
                    seg_start = segment_start_ts
                else:
                    orig_start, billed_before, seg_start = meta
                # длина закрывающегося сегмента
                seg_duration = max(0.0, end_ts - seg_start)
                accumulated_billed = billed_before + seg_duration
                # кладём в pending — может прийти склейка
                self._pending_stops[tid] = (orig_start, end_ts, accumulated_billed)
                with self.active_lock:
                    self.active_sessions.pop(tid, None)
                update_active_file(self.active_sessions)
                log.info(f'[t{tid}] STOP  (segment={int(seg_duration)}s, '
                         f'billed_total={int(accumulated_billed)}s, '
                         f'pending merge for {SESSION_MERGE_GRACE_S}s, '
                         f'people={n_people_here}, motion={motion:.3f})')

    def run(self):
        self.capture.start()
        # дать капчуру шанс получить первый кадр
        time.sleep(2.0)
        log.info(f'[cam {self.cam_id}] worker started, tables={sorted(self.tables_ids)}')
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                log.exception(f'[cam {self.cam_id}] tick failed: {e}')
            next_tick += TICK_INTERVAL_S
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # отстаём — пересинхронизируемся
                next_tick = time.monotonic()
        self.capture.stop()
        log.info(f'[cam {self.cam_id}] worker stopped')

# ------------------------------------------------------------------ #
#  ENTRY POINT
# ------------------------------------------------------------------ #

def cleanup_old_evidence():
    """Удаляет evidence-папки старше EVIDENCE_KEEP_DAYS дней."""
    if not os.path.isdir(EVIDENCE_DIR):
        return
    cutoff = datetime.now().date()
    removed = 0
    for name in os.listdir(EVIDENCE_DIR):
        day_path = os.path.join(EVIDENCE_DIR, name)
        if not os.path.isdir(day_path):
            continue
        try:
            day_date = datetime.strptime(name, '%Y-%m-%d').date()
        except ValueError:
            continue
        age_days = (cutoff - day_date).days
        if age_days > EVIDENCE_KEEP_DAYS:
            try:
                import shutil
                shutil.rmtree(day_path)
                removed += 1
            except Exception as e:
                log.warning(f'failed to remove {day_path}: {e}')
    if removed:
        log.info(f'evidence cleanup: removed {removed} old day folders')


def main():
    os.makedirs(SNAP_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    cleanup_old_evidence()
    if COLLECT_CALIBRATION:
        os.makedirs(CALIB_DIR, exist_ok=True)
        log.info(f'CALIBRATION MODE: saving frames to {CALIB_DIR} '
                 f'(every {CALIB_EVERY_N_TICKS} ticks, cap {CALIB_MAX_FILES})')

    zones      = load_zones()
    homography = load_homographies()
    default_rule, rules = load_rules()

    log.info(f'Zones: {len(zones)} tables, homographies: {sorted(homography.keys())}')
    log.info(f'Rules: default={default_rule}, per-table={sorted(rules.keys())}')

    detector = PersonDetector()  # один на процесс — Ultralytics+OpenVINO потокобезопасен на инференс

    active_sessions = {}    # {table_id: start_ts}
    active_lock     = threading.Lock()
    update_active_file(active_sessions)  # пустой стартовый файл

    workers = [
        CameraWorker(cam, zones, homography, default_rule, rules,
                     detector, active_sessions, active_lock)
        for cam in CAMERAS
    ]
    for w in workers:
        w.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info('Shutdown requested')
    finally:
        for w in workers:
            w.stop()
        # дать тредам корректно закрыть RTSP
        for w in workers:
            w.join(timeout=5.0)
        # закрыть все ещё-активные сессии и финализировать pending
        now_ts = time.time()
        for w in workers:
            for tid, st in w.states.items():
                ev = st.force_stop(now_ts)
                if ev is not None:
                    _, seg_start, seg_end = ev
                    meta = w._active_meta.pop(tid, None)
                    if meta is None:
                        write_session(tid, seg_start, seg_end)
                    else:
                        orig_start, billed_before, real_seg_start = meta
                        seg_duration = max(0.0, seg_end - real_seg_start)
                        write_session(tid, orig_start, seg_end,
                                      billed_seconds=billed_before + seg_duration)
            for tid, (orig_s, last_e, billed) in list(w._pending_stops.items()):
                write_session(tid, orig_s, last_e, billed_seconds=billed)
            w._pending_stops.clear()
        update_active_file({})
        log.info('Bye')


if __name__ == '__main__':
    main()
