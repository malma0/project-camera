"""
Auto-calibration of table positions.

Runs daily at 22:00. For each camera:
1. Captures a frame
2. Finds all blue rectangles (table surfaces)
3. Matches them to known tables by proximity
4. Updates zones.json with new positions
5. Logs all changes

Gray-out conditions:
- If a table is not found, keep old coordinates and log warning
- If displacement > 3 meters, suspicious - log alert
"""

import cv2
import json
import numpy as np
import os
import sys
import time
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERAS, ZONES_FILE

HOMOGRAPHY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "homography.json"
)

CALIBRATION_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration_log.txt"
)

# Стол ITTF: 2740 x 1525 мм. Соотношение ~1.8:1
TABLE_LENGTH_MM = 2740
TABLE_WIDTH_MM = 1525
TABLE_ASPECT = TABLE_LENGTH_MM / TABLE_WIDTH_MM  # 1.797

# Допуск на размеры и соотношение сторон при поиске стола
ASPECT_TOLERANCE = 0.4   # стол считается столом если соотношение 1.4-2.2
SIZE_TOLERANCE_MM = 600  # размер стола может отличаться на 600мм от номинала

# Максимальный сдвиг стола относительно прежней позиции (мм)
# Больше этого = подозрение что нашли НЕ ТОТ стол
MAX_DISPLACEMENT_MM = 3000

# Порог "сильного сдвига" для алерта
ALERT_DISPLACEMENT_MM = 1500


def log(msg):
    """Пишет в файл и в консоль."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(CALIBRATION_LOG, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass


def capture_frame(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def find_blue_rectangles(frame):
    """Находит синие прямоугольники (столы) на кадре.
    Возвращает список из 4-углов каждого найденного прямоугольника."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска синего цвета (столы для пинг-понга обычно тёмно-синие)
    lower_blue = np.array([95, 80, 50])
    upper_blue = np.array([130, 255, 200])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Морфология — убираем шум и заполняем дыры
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Ищем контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    frame_area = frame.shape[0] * frame.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Слишком маленькие или слишком большие контуры пропускаем
        if area < frame_area * 0.005 or area > frame_area * 0.3:
            continue

        # Аппроксимируем контур многоугольником
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Хотим именно 4 угла
        if len(approx) != 4:
            # Попробуем minAreaRect — он всегда даёт 4 угла
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            approx = box.astype(np.int32).reshape(-1, 1, 2)

        if len(approx) != 4:
            continue

        corners = approx.reshape(4, 2).astype(np.float32)
        rectangles.append(corners)

    return rectangles, mask


def pixel_to_floor(point, H):
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return [float(dst[0][0][0]), float(dst[0][0][1])]


def rect_pixels_to_floor(corners, H):
    """4 угла прямоугольника на картинке -> 4 угла на полу в мм."""
    return [pixel_to_floor(c, H) for c in corners]


def validate_floor_rect(floor_corners):
    """Проверяет что найденный прямоугольник похож на стол по размеру и пропорциям.
    Возвращает (is_valid, длина, ширина)."""
    pts = np.array(floor_corners)
    # Все 4 расстояния между соседними углами
    dists = []
    for i in range(4):
        j = (i + 1) % 4
        d = np.linalg.norm(pts[j] - pts[i])
        dists.append(d)

    dists.sort()
    # 2 короткие стороны и 2 длинные
    width = (dists[0] + dists[1]) / 2
    length = (dists[2] + dists[3]) / 2

    if length == 0:
        return False, 0, 0

    aspect = length / width

    # Проверка размеров и пропорций
    if abs(aspect - TABLE_ASPECT) > ASPECT_TOLERANCE:
        return False, length, width
    if abs(length - TABLE_LENGTH_MM) > SIZE_TOLERANCE_MM:
        return False, length, width
    if abs(width - TABLE_WIDTH_MM) > SIZE_TOLERANCE_MM:
        return False, length, width

    return True, length, width


def reorder_corners(floor_corners):
    """Упорядочивает 4 угла стола в каноническом порядке:
    [far-left, far-right, near-right, near-left]
    Где far = минимальный Y, near = максимальный Y (или наоборот - не важно
    пока порядок согласован)."""
    pts = np.array(floor_corners)
    # Сортируем по Y - сначала маленький Y (дальние), потом большой (ближние)
    sorted_by_y = pts[pts[:, 1].argsort()]
    far_pts = sorted_by_y[:2]
    near_pts = sorted_by_y[2:]

    # Среди дальних: меньший X = left, больший X = right
    far_left = far_pts[far_pts[:, 0].argmin()]
    far_right = far_pts[far_pts[:, 0].argmax()]
    near_left = near_pts[near_pts[:, 0].argmin()]
    near_right = near_pts[near_pts[:, 0].argmax()]

    return [far_left.tolist(), far_right.tolist(),
            near_right.tolist(), near_left.tolist()]


def floor_to_pixel(point, H):
    """Обратное преобразование - для записи points_pixel."""
    H_inv = np.linalg.inv(H)
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H_inv)
    return [int(dst[0][0][0]), int(dst[0][0][1])]


def zone_center(floor_points):
    pts = np.array(floor_points)
    return pts.mean(axis=0)


def calibrate_camera(camera, zones, H, frame=None):
    """Пытается обновить координаты столов для одной камеры.
    Возвращает обновлённый словарь зон и список изменений."""
    cam_id = camera['id']

    if H is None:
        log(f"Камера {cam_id}: гомография отсутствует - пропуск")
        return zones, []

    if frame is None:
        frame = capture_frame(camera['rtsp'])
    if frame is None:
        log(f"Камера {cam_id}: не удалось получить кадр - пропуск")
        return zones, []

    # Находим все синие прямоугольники
    rectangles, mask = find_blue_rectangles(frame)
    log(f"Камера {cam_id}: найдено синих прямоугольников: {len(rectangles)}")

    # Переводим каждый в координаты пола, валидируем размер
    candidates = []  # [(floor_corners_ordered, center, length, width)]
    for rect_px in rectangles:
        floor_corners = rect_pixels_to_floor(rect_px, H)
        is_valid, length, width = validate_floor_rect(floor_corners)
        if not is_valid:
            continue
        ordered = reorder_corners(floor_corners)
        center = zone_center(ordered)
        candidates.append({
            'floor': ordered,
            'pixel_raw': rect_px,
            'center': center,
            'length': length,
            'width': width,
        })

    log(f"Камера {cam_id}: валидных кандидатов (по размеру): {len(candidates)}")

    # Получаем столы этой камеры из zones
    cam_tables = {int(tid): z for tid, z in zones.items()
                  if z.get('camera_id') == cam_id}

    changes = []
    updated_zones = dict(zones)
    used_candidates = set()

    # Для каждого известного стола ищем ближайшего НЕ ИСПОЛЬЗОВАННОГО кандидата
    for table_id, zone in cam_tables.items():
        old_center = zone_center(zone['points_floor'])

        best_idx = None
        best_dist = float('inf')
        for i, cand in enumerate(candidates):
            if i in used_candidates:
                continue
            dist = np.linalg.norm(cand['center'] - old_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            log(f"  Стол {table_id}: кандидатов нет, оставляю старые координаты")
            continue

        if best_dist > MAX_DISPLACEMENT_MM:
            log(f"  Стол {table_id}: ближайший кандидат в {int(best_dist)}мм "
                f"(больше {MAX_DISPLACEMENT_MM}мм) - не обновляю, оставляю старые")
            continue

        used_candidates.add(best_idx)
        cand = candidates[best_idx]

        # Записываем новые координаты
        new_floor = cand['floor']
        new_pixel = [floor_to_pixel(p, H) for p in new_floor]

        if best_dist < 50:
            # Сдвиг меньше 5см - считаем что стол не двигался
            log(f"  Стол {table_id}: на месте (сдвиг {int(best_dist)}мм)")
            continue

        updated_zones[str(table_id)] = {
            'camera_id': cam_id,
            'points_floor': new_floor,
            'points_pixel': new_pixel,
        }

        change = {
            'table_id': table_id,
            'displacement_mm': float(best_dist),
            'old_center': old_center.tolist(),
            'new_center': cand['center'].tolist(),
        }
        changes.append(change)

        if best_dist > ALERT_DISPLACEMENT_MM:
            log(f"  ⚠ Стол {table_id}: СИЛЬНЫЙ сдвиг {int(best_dist)}мм")
        else:
            log(f"  Стол {table_id}: сдвиг {int(best_dist)}мм")

    # Кандидаты которые не привязаны ни к одному столу
    unused = len(candidates) - len(used_candidates)
    if unused > 0:
        log(f"Камера {cam_id}: {unused} кандидатов не привязаны "
            f"(возможно посторонние синие объекты или новые столы)")

    return updated_zones, changes


def run_calibration():
    """Основная функция - выполняет автокалибровку всех камер."""
    log("=" * 60)
    log("АВТОКАЛИБРОВКА СТОЛОВ - НАЧАЛО")
    log("=" * 60)

    # Загружаем зоны и гомографии
    if not os.path.exists(ZONES_FILE):
        log("zones.json не найден - сначала запусти marker_v2.py")
        return

    with open(ZONES_FILE, 'r', encoding='utf-8') as f:
        zones = json.load(f)

    if not os.path.exists(HOMOGRAPHY_FILE):
        log("homography.json не найден - калибровка невозможна")
        return

    with open(HOMOGRAPHY_FILE, 'r') as f:
        raw_hom = json.load(f)
    homographies = {int(k): np.array(v, dtype=np.float32) for k, v in raw_hom.items()}

    all_changes = []
    current_zones = zones

    for camera in CAMERAS:
        H = homographies.get(camera['id'])
        current_zones, changes = calibrate_camera(camera, current_zones, H)
        all_changes.extend(changes)

    if not all_changes:
        log("Изменений не обнаружено - все столы на местах")
        log("=" * 60)
        return

    # Бэкапим старый zones.json
    backup = f"{ZONES_FILE}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, 'r', encoding='utf-8') as f:
            old_data = f.read()
        with open(backup, 'w', encoding='utf-8') as f:
            f.write(old_data)
        log(f"Бэкап старых зон: {backup}")

    # Сохраняем обновлённые зоны
    with open(ZONES_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_zones, f, indent=2, ensure_ascii=False)

    log(f"✅ Обновлено столов: {len(all_changes)}")
    log("=" * 60)


def run_scheduler():
    """Запускает run_calibration() каждый день в 22:00."""
    log("Планировщик автокалибровки запущен. Срабатывает в 22:00 каждый день.")
    last_run_date = None

    while True:
        now = datetime.now()
        # Срабатываем один раз в день в 22:00
        if now.hour == 22 and now.minute == 0:
            today = now.date()
            if last_run_date != today:
                last_run_date = today
                try:
                    run_calibration()
                except Exception as e:
                    log(f"ОШИБКА автокалибровки: {e}")
                    import traceback
                    log(traceback.format_exc())
        time.sleep(30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'now':
        # Запуск немедленно для теста: python auto_calibrate.py now
        run_calibration()
    elif len(sys.argv) > 1 and sys.argv[1] == 'schedule':
        # Запуск планировщика: python auto_calibrate.py schedule
        run_scheduler()
    else:
        print("Использование:")
        print("  python auto_calibrate.py now       - запустить калибровку прямо сейчас")
        print("  python auto_calibrate.py schedule  - запустить планировщик (каждый день в 22:00)")
