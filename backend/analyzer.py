import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import sys, os

TABLE_RULES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "table_rules.json"
)

def load_table_rules():
    if not os.path.exists(TABLE_RULES_FILE):
        return {"default": {}}
    with open(TABLE_RULES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_rules_for_table(all_rules, table_id):
    default = all_rules.get("default", {})
    specific = all_rules.get(str(table_id), {})
    # Merge: специфичные правила переопределяют default
    merged = {**default, **specific}
    # Дефолтные значения если ничего не задано
    merged.setdefault("min_stable_people", 2)
    merged.setdefault("score_threshold", 6)
    merged.setdefault("exit_threshold", 4)
    merged.setdefault("score_bonus", 0)
    merged.setdefault("ignore_ball", False)
    merged.setdefault("require_movement", True)
    return merged

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAMERAS, ZONES_FILE, FRAME_INTERVAL, STABLE_DURATION, TRACK_BUFFER,
    SCORE_THRESHOLD, CONFIRMATION_SECONDS, ABSENCE_TOLERANCE, MIN_GAME_DURATION,
    MOVEMENT_HISTORY, MOVEMENT_THRESHOLD_HIGH, DETECTION_CONFIDENCE, TABLE_EXPAND_PX
)
from database import init_db, save_session

CLASS_PERSON = 0
CLASS_RACKET = 38

MOVEMENT_THRESHOLD_LOW = 5

ACTIVE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "active.json"
)

active_sessions_lock = threading.Lock()


def save_active_sessions(camera_id, sessions):
    try:
        with active_sessions_lock:
            existing = {}
            if os.path.exists(ACTIVE_FILE):
                try:
                    with open(ACTIVE_FILE, 'r') as f:
                        existing = json.load(f)
                except:
                    existing = {}

            keys_to_remove = [k for k, v in existing.items()
                              if v.get('camera_id') == camera_id]
            for k in keys_to_remove:
                del existing[k]

            for table_id, start_ts in sessions.items():
                existing[str(table_id)] = {
                    'start': start_ts,
                    'camera_id': camera_id
                }

            with open(ACTIVE_FILE, 'w') as f:
                json.dump(existing, f)
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения active.json: {e}", flush=True)


def create_ball_detector():
    """
    Детектор мяча для настольного тенниса.
    Ищет маленький белый/светлый круглый объект на столе.
    Используем SimpleBlobDetector — без нейросетей, быстро.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Фильтр по цвету (ищем светлые объекты)
    params.filterByColor = True
    params.blobColor = 255  # белый

    # Размер мяча в пикселях (зависит от расстояния камеры)
    # При высоте ~4м и 1920px мяч ~8-20px в диаметре
    params.filterByArea = True
    params.minArea = 20    # минимум ~5px диаметр
    params.maxArea = 800   # максимум ~32px диаметр

    # Форма — круглый
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Инерция (не вытянутый)
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Выпуклость
    params.filterByConvexity = True
    params.minConvexity = 0.7

    return cv2.SimpleBlobDetector_create(params)


def detect_ball_on_table(frame, zone_mask, prev_frame=None):
    """
    Ищет мяч для пинг-понга в зоне стола.

    Стратегия:
    1. Берём зону стола (синяя поверхность)
    2. Ищем маленький белый круглый объект
    3. Если есть предыдущий кадр — проверяем что объект движется (не просто пятно)

    Возвращает список точек где найден мяч.
    """
    balls = []

    try:
        # Применяем маску зоны
        masked = cv2.bitwise_and(frame, frame, mask=zone_mask)

        # Конвертируем в HSV для лучшего выделения белого мяча
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # Маска белого/светло-жёлтого цвета (мяч для пинг-понга)
        # Белый: низкая насыщенность, высокая яркость
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Светло-оранжевый/жёлтый мяч
        lower_orange = np.array([10, 100, 150])
        upper_orange = np.array([30, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        ball_mask = cv2.bitwise_or(white_mask, orange_mask)

        # Убираем шум
        kernel = np.ones((3, 3), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        # Ищем контуры (альтернатива blob detector — надёжнее)
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 800:
                continue

            # Проверяем что это круглый объект
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue

            # Центр объекта
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            balls.append((cx, cy))

    except Exception as e:
        pass  # Тихо игнорируем ошибки детектора мяча

    return balls


class BallTracker:
    """Отслеживает мяч между кадрами чтобы подтвердить что он движется."""

    def __init__(self, table_id, zone_points, frame_w, frame_h, orig_w, orig_h, rules):
        self.table_id = table_id
        self.rules = rules
    # ... остальное без изменений

    def update(self, balls):
        """Принимает список найденных позиций мяча, возвращает True если мяч движется."""
        if not balls:
            return False

        # Берём первый найденный мяч
        pos = balls[0]
        now = time.time()

        if self.last_seen and (now - self.last_seen) > 3:
            # Давно не видели — сбрасываем трек
            self.positions.clear()

        self.positions.append(pos)
        self.last_seen = now

        # Мяч подтверждается только если:
        # 1. Видели его 3+ кадра подряд (исключает разовые блики)
        # 2. Сместился минимум на 15px (исключает неподвижные белые пятна)
        if len(self.positions) >= 3:
            dx = self.positions[-1][0] - self.positions[-3][0]
            dy = self.positions[-1][1] - self.positions[-3][1]
            dist = (dx*dx + dy*dy) ** 0.5
            return dist > 15

        return False

    def is_active(self):
        """Мяч активен если видели его в последние 2 секунды."""
        if self.last_seen is None:
            return False
        return (time.time() - self.last_seen) < 2


class PersonTrack:
    def __init__(self, track_id):
        self.id = track_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.positions = deque(maxlen=MOVEMENT_HISTORY)

    def update(self, center):
        self.last_seen = time.time()
        self.positions.append(center)

    def is_stable(self):
        return (self.last_seen - self.first_seen) >= STABLE_DURATION

    def is_alive(self):
        return (time.time() - self.last_seen) < TRACK_BUFFER

    def movement_intensity(self):
        if len(self.positions) < 2:
            return 0
        distances = []
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            distances.append((dx*dx + dy*dy) ** 0.5)
        return sum(distances) / len(distances) if distances else 0


class TableTracker:
    def __init__(self, table_id, zone_points, frame_w, frame_h, orig_w, orig_h):
        self.table_id = table_id

        scale_x = frame_w / orig_w if orig_w else 1.0
        scale_y = frame_h / orig_h if orig_h else 1.0

        scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in zone_points]
        self.zone_points = np.array(scaled, np.int32)
        self.expanded_zone = self._expand_zone(scaled, TABLE_EXPAND_PX)

        # Маска зоны стола (для детектора мяча)
        self.zone_mask = None  # инициализируется при первом кадре

        self.session_start = None
        self.last_active = None
        self.pending_confirmations = 0

        self.ball_tracker = BallTracker()

    def init_mask(self, frame_h, frame_w):
        """Создаём маску зоны для детектора мяча (один раз)."""
        self.zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(self.zone_mask, [self.zone_points], 255)

    def _expand_zone(self, points, padding):
        pts = np.array(points, np.float32)
        center = pts.mean(axis=0)
        expanded = []
        for pt in pts:
            direction = pt - center
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                expanded.append((pt + unit * padding).astype(int).tolist())
            else:
                expanded.append(pt.astype(int).tolist())
        return np.array(expanded, np.int32)

    def point_in_expanded(self, point):
        return cv2.pointPolygonTest(self.expanded_zone, (float(point[0]), float(point[1])), False) >= 0

    def calculate_score(self, all_tracks, rackets, frame=None):
        nearby_tracks = []
        stable_count = 0
        moving_count = 0
        max_movement = 0
        max_stable_duration = 0

        for track in all_tracks.values():
            if not track.is_alive() or not track.positions:
                continue
            last_pos = track.positions[-1]
            if self.point_in_expanded(last_pos):
                nearby_tracks.append(track)
                intensity = track.movement_intensity()
                max_movement = max(max_movement, intensity)

                if track.is_stable():
                    stable_count += 1
                    duration = track.last_seen - track.first_seen
                    max_stable_duration = max(max_stable_duration, duration)

                if intensity > MOVEMENT_THRESHOLD_LOW:
                    moving_count += 1

        rackets_near = sum(1 for r in rackets if self.point_in_expanded(r))

    # ГЛАВНЫЙ ФИЛЬТР: применяется только при СТАРТЕ новой сессии.
    # Если сессия уже идёт — не блокируем (игроки могут отойти за мячом).
        if self.session_start is None and stable_count < self.rules["min_stable_people"]:
            if len(nearby_tracks) > 0:
                print(f"  [Стол {self.table_id}] фильтр старта: "
                    f"стабильных {stable_count} < {self.rules['min_stable_people']}", flush=True)
            return 0

        racket_near_stable = False
        for racket_pt in rackets:
            if not self.point_in_expanded(racket_pt):
                continue
            for track in nearby_tracks:
                if not track.is_stable() or not track.positions:
                    continue
                last_pos = track.positions[-1]
                dist = ((racket_pt[0]-last_pos[0])**2 + (racket_pt[1]-last_pos[1])**2) ** 0.5
                if dist < 200:
                    racket_near_stable = True
                    break

    # --- Детектор мяча ---
        ball_moving = False
        if frame is not None and self.zone_mask is not None:
            balls = detect_ball_on_table(frame, self.zone_mask)
            ball_moving = self.ball_tracker.update(balls)

        score = 0

    # --- Стабильные игроки ---
        if stable_count >= 2:
            score += 5
        elif stable_count == 1:
            score += 2

    # --- Движущиеся игроки ---
        if moving_count >= 2:
            score += 4
        elif moving_count == 1:
            score += 2

    # --- Ракетки ---
        if racket_near_stable:
            score += 4
        elif rackets_near > 0:
            score += 2

    # --- Интенсивность движения ---
        if max_movement > MOVEMENT_THRESHOLD_HIGH:
            score += 2

    # --- Долго стоят у стола ---
        if max_stable_duration > 60:
            score += 2

    # --- Просто люди рядом ---
        if len(nearby_tracks) >= 2:
            score += 1

    # --- МЯЧ ДВИЖЕТСЯ (только если детектор не отключён правилами и есть люди) ---
        if not self.rules["ignore_ball"] and ball_moving and len(nearby_tracks) >= 1:
            score += 5
            print(f"  [Стол {self.table_id}] 🏓 МЯЧ ДВИЖЕТСЯ! +5 к score", flush=True)

    # --- Бонус из правил стола ---
        score += self.rules["score_bonus"]

        if len(nearby_tracks) > 0 or rackets_near > 0 or ball_moving:
            print(f"  [Стол {self.table_id}] рядом={len(nearby_tracks)} "
                f"стабильных={stable_count} "
                f"движущихся={moving_count} "
                f"мяч={'да' if ball_moving else 'нет'} "
                f"ракетки={rackets_near} "
                f"движение={int(max_movement)} "
                f"score={score}", flush=True)

        return score

    def update(self, score):
        now = time.time()
        is_active = score >= SCORE_THRESHOLD

        if self.session_start is None:
            is_active = score >= self.rules["score_threshold"]
        else:
            is_active = score >= self.rules["exit_threshold"]

        if is_active:
            self.pending_confirmations += 1
            if self.pending_confirmations >= CONFIRMATION_SECONDS and self.session_start is None:
                self.session_start = now - (CONFIRMATION_SECONDS * FRAME_INTERVAL)
                print(f"[Стол {self.table_id}] ▶ Сессия началась (score={score})", flush=True)
            self.last_active = now
        else:

            if is_active:
                self.pending_confirmations += 1
                if self.pending_confirmations >= CONFIRMATION_SECONDS and self.session_start is None:
                    self.session_start = now - (CONFIRMATION_SECONDS * FRAME_INTERVAL)
                    print(f"[Стол {self.table_id}] ▶ Сессия началась (score={score})", flush=True)
                self.last_active = now
            else:
                self.pending_confirmations = 0
                if self.session_start is not None and self.last_active is not None:
                    absence = now - self.last_active
                    if absence > ABSENCE_TOLERANCE:
                        duration = self.last_active - self.session_start
                        if duration >= MIN_GAME_DURATION:
                            start_dt = datetime.fromtimestamp(self.session_start)
                            end_dt = datetime.fromtimestamp(self.last_active)
                            save_session(self.table_id, start_dt, end_dt)
                            print(f"[Стол {self.table_id}] ⏹ Записано: {int(duration)} сек", flush=True)
                        else:
                            print(f"[Стол {self.table_id}] ⏹ Слишком коротко ({int(duration)} сек)", flush=True)
                        self.session_start = None
                        self.last_active = None


def get_box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def process_camera(camera, zones):
    # Каждый поток создаёт свою модель — YOLO не thread-safe!
    print(f"[{camera['name']}] Загрузка YOLO...", flush=True)
    model = YOLO('yolov8n.pt')
    print(f"[{camera['name']}] Подключение...", flush=True)

    # Загружаем правила для столов
    table_rules = load_table_rules()

    # Определяем размер кадра
    cap_test = cv2.VideoCapture(camera['rtsp'], cv2.CAP_FFMPEG)
    frame_w, frame_h = 1920, 1080
    if cap_test.isOpened():
        ret, frame_test = cap_test.read()
        if ret:
            frame_h, frame_w = frame_test.shape[:2]
    cap_test.release()
    print(f"[{camera['name']}] Разрешение: {frame_w}x{frame_h}", flush=True)

    # Создаём трекеры для столов этой камеры
    table_trackers = {}
    for table_id_str, zone in zones.items():
        if zone.get('camera_id') == camera['id']:
            table_id = int(table_id_str)
            rules = get_rules_for_table(table_rules, table_id)
            tracker = TableTracker(
                table_id, zone['points'],
                frame_w, frame_h, frame_w, frame_h,
                rules
            )
            tracker.init_mask(frame_h, frame_w)
            table_trackers[table_id] = tracker
            print(f"[{camera['name']}] Стол {table_id} правила: "
                  f"min_stable={rules['min_stable_people']}, "
                  f"threshold={rules['score_threshold']}, "
                  f"bonus={rules['score_bonus']}, "
                  f"ignore_ball={rules['ignore_ball']}", flush=True)

    if not table_trackers:
        print(f"[{camera['name']}] ⚠ Нет размеченных зон", flush=True)
        return

    print(f"[{camera['name']}] Столы: {sorted(table_trackers.keys())}", flush=True)

    person_tracks = {}

    while True:
        cap = cv2.VideoCapture(camera['rtsp'], cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{camera['name']}] Не удалось подключиться, повтор через 10 сек", flush=True)
            time.sleep(10)
            continue

        print(f"[{camera['name']}] ✅ Подключено", flush=True)
        last_process_time = 0
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{camera['name']}] Поток прерван", flush=True)
                    break

                if time.time() - last_process_time < FRAME_INTERVAL:
                    continue
                last_process_time = time.time()
                frame_count += 1

                # YOLO детекция
                try:
                    results = model.track(
                        frame,
                        classes=[CLASS_PERSON, CLASS_RACKET],
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=DETECTION_CONFIDENCE,
                        verbose=False
                    )
                except Exception as e:
                    print(f"[{camera['name']}] Ошибка YOLO: {e}", flush=True)
                    continue

                rackets = []

                # Парсим результаты детекции
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        xyxy = box.xyxy[0].tolist()
                        center = get_box_center(xyxy)

                        if cls == CLASS_RACKET:
                            rackets.append(center)
                        elif cls == CLASS_PERSON:
                            if box.id is None:
                                continue
                            track_id = int(box.id[0])
                            if track_id not in person_tracks:
                                person_tracks[track_id] = PersonTrack(track_id)
                            person_tracks[track_id].update(center)

                # Удаляем мёртвые треки
                dead = [tid for tid, t in person_tracks.items() if not t.is_alive()]
                for tid in dead:
                    del person_tracks[tid]

                alive_count = len([t for t in person_tracks.values() if t.is_alive()])

                if frame_count % 30 == 0:
                    print(f"[{camera['name']}] Кадр #{frame_count} | "
                          f"Людей: {alive_count} | Ракеток: {len(rackets)}", flush=True)

                # === ПРИВЯЗКА К БЛИЖАЙШЕМУ СТОЛУ ===
                # Каждый человек привязан к ОДНОМУ столу — ближайшему к его позиции
                table_persons = {tid: {} for tid in table_trackers}

                for track_id, track in person_tracks.items():
                    if not track.is_alive() or not track.positions:
                        continue
                    pos = track.positions[-1]

                    best_table = None
                    best_dist = float('inf')
                    for tid, tracker in table_trackers.items():
                        if not tracker.point_in_expanded(pos):
                            continue
                        center = tracker.zone_points.mean(axis=0)
                        dx = pos[0] - center[0]
                        dy = pos[1] - center[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_table = tid

                    if best_table is not None:
                        table_persons[best_table][track_id] = track

                # Ракетки тоже к ближайшему столу
                table_rackets = {tid: [] for tid in table_trackers}
                for racket_pt in rackets:
                    best_table = None
                    best_dist = float('inf')
                    for tid, tracker in table_trackers.items():
                        if not tracker.point_in_expanded(racket_pt):
                            continue
                        center = tracker.zone_points.mean(axis=0)
                        dx = racket_pt[0] - center[0]
                        dy = racket_pt[1] - center[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_table = tid
                    if best_table is not None:
                        table_rackets[best_table].append(racket_pt)

                # === ОБСЧЁТ СТОЛОВ ===
                for tid, tracker in table_trackers.items():
                    score = tracker.calculate_score(
                        table_persons[tid],
                        table_rackets[tid],
                        frame
                    )
                    tracker.update(score)

                # Сохраняем активные сессии каждые 5 кадров
                if frame_count % 5 == 0:
                    active = {}
                    for tid, tracker in table_trackers.items():
                        if tracker.session_start is not None:
                            active[str(tid)] = tracker.session_start
                    save_active_sessions(camera['id'], active)

        except Exception as e:
            import traceback
            print(f"[{camera['name']}] Критическая ошибка: {e}", flush=True)
            traceback.print_exc()
            time.sleep(5)
        finally:
            cap.release()

def main():
    init_db()

    if not os.path.exists(ZONES_FILE):
        print("❌ Сначала разметьте столы: python backend/marker.py")
        return

    with open(ZONES_FILE, 'r') as f:
        zones = json.load(f)
    print(f"Загружено зон: {len(zones)}")

    threads = []
    for camera in CAMERAS:
        t = threading.Thread(
            target=process_camera,
            args=(camera, zones),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(3)

    print("\n🟢 Анализ запущен. Ctrl+C для остановки\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔴 Остановка...")


if __name__ == "__main__":
    main()
