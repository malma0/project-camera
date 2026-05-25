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

HOMOGRAPHY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "homography.json"
)


def load_table_rules():
    if not os.path.exists(TABLE_RULES_FILE):
        return {"default": {}}
    with open(TABLE_RULES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_rules_for_table(all_rules, table_id):
    default = all_rules.get("default", {})
    specific = all_rules.get(str(table_id), {})
    merged = {**default, **specific}
    merged.setdefault("min_stable_people", 2)
    merged.setdefault("score_threshold", 6)
    merged.setdefault("exit_threshold", 4)
    merged.setdefault("score_bonus", 0)
    merged.setdefault("ignore_ball", False)
    merged.setdefault("require_movement", True)
    return merged


def load_homographies():
    """Загружает матрицы гомографии {camera_id: np.array 3x3}."""
    if not os.path.exists(HOMOGRAPHY_FILE):
        print("[WARN] homography.json not found - run marker_v2.py first!", flush=True)
        return {}
    with open(HOMOGRAPHY_FILE, 'r') as f:
        raw = json.load(f)
    return {int(k): np.array(v, dtype=np.float32) for k, v in raw.items()}


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

# Расширение зоны стола НА ПОЛУ в миллиметрах
# Игроки стоят примерно в 80-100 см от края стола
FLOOR_EXPAND_MM = 1000

ACTIVE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "active.json"
)

SNAPSHOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "snapshots"
)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

snapshot_lock = threading.Lock()
last_snapshot_time = {}

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
        print(f"[ERROR] save active.json: {e}", flush=True)


def detect_ball_on_table(frame, zone_mask, prev_frame=None):
    balls = []
    try:
        masked = cv2.bitwise_and(frame, frame, mask=zone_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        lower_orange = np.array([10, 100, 150])
        upper_orange = np.array([30, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        ball_mask = cv2.bitwise_or(white_mask, orange_mask)

        kernel = np.ones((3, 3), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 800:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            balls.append((cx, cy))

    except Exception:
        pass

    return balls


class BallTracker:
    def __init__(self):
        self.positions = deque(maxlen=10)
        self.last_seen = None

    def update(self, balls):
        if not balls:
            return False

        pos = balls[0]
        now = time.time()

        if self.last_seen and (now - self.last_seen) > 3:
            self.positions.clear()

        self.positions.append(pos)
        self.last_seen = now

        if len(self.positions) >= 3:
            dx = self.positions[-1][0] - self.positions[-3][0]
            dy = self.positions[-1][1] - self.positions[-3][1]
            dist = (dx*dx + dy*dy) ** 0.5
            return dist > 15
        return False

    def is_active(self):
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
    """Трекер стола с поддержкой гомографии.
    Зоны хранятся в координатах ПОЛА (миллиметры).
    Точки игроков переводятся в координаты пола через гомографию."""

    def __init__(self, table_id, zone_data, homography, rules):
        self.table_id = table_id
        self.rules = rules
        self.homography = homography

        # Зона на полу (миллиметры)
        floor_pts = zone_data.get('points_floor')
        pixel_pts = zone_data.get('points_pixel') or zone_data.get('points')

        if floor_pts is not None and homography is not None:
            self.zone_floor = np.array(floor_pts, dtype=np.float32)
            self.expanded_floor = self._expand_floor_zone(self.zone_floor, FLOOR_EXPAND_MM)
            self.use_homography = True
        else:
            self.zone_floor = None
            self.expanded_floor = None
            self.use_homography = False

        # Зона в пикселях — нужна для маски детектора мяча
        self.zone_pixel = np.array(pixel_pts, np.int32)

        # Расширенная пиксельная зона (fallback если нет гомографии)
        self.expanded_pixel = self._expand_pixel_zone(pixel_pts, TABLE_EXPAND_PX)

        self.zone_mask = None
        self.session_start = None
        self.last_active = None
        self.pending_confirmations = 0
        self.ball_tracker = BallTracker()

    def init_mask(self, frame_h, frame_w):
        self.zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(self.zone_mask, [self.zone_pixel], 255)

    def _expand_floor_zone(self, points, padding_mm):
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        expanded = []
        for pt in pts:
            direction = pt - center
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                expanded.append((pt + unit * padding_mm).tolist())
            else:
                expanded.append(pt.tolist())
        return np.array(expanded, dtype=np.float32)

    def _expand_pixel_zone(self, points, padding_px):
        pts = np.array(points, np.float32)
        center = pts.mean(axis=0)
        expanded = []
        for pt in pts:
            direction = pt - center
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                expanded.append((pt + unit * padding_px).astype(int).tolist())
            else:
                expanded.append(pt.astype(int).tolist())
        return np.array(expanded, np.int32)

    def pixel_to_floor(self, point):
        if self.homography is None:
            return None
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography)
        return (float(dst[0][0][0]), float(dst[0][0][1]))

    def point_in_expanded(self, point):
        """Главная проверка — попадает ли точка в зону стола.
        Если есть гомография — проверяем на полу. Иначе — на картинке."""
        if self.use_homography:
            floor_pt = self.pixel_to_floor(point)
            if floor_pt is None:
                return False
            return cv2.pointPolygonTest(self.expanded_floor,
                                        (float(floor_pt[0]), float(floor_pt[1])), False) >= 0
        else:
            return cv2.pointPolygonTest(self.expanded_pixel,
                                        (float(point[0]), float(point[1])), False) >= 0

    @property
    def zone_points(self):
        """Для обратной совместимости с кодом process_camera (для расчёта центра)."""
        return self.zone_pixel

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

        if self.session_start is None and stable_count < self.rules["min_stable_people"]:
            if len(nearby_tracks) > 0:
                print(f"  [Table {self.table_id}] start filter: "
                      f"stable {stable_count} < {self.rules['min_stable_people']}", flush=True)
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

        ball_moving = False
        if frame is not None and self.zone_mask is not None:
            balls = detect_ball_on_table(frame, self.zone_mask)
            ball_moving = self.ball_tracker.update(balls)

        score = 0

        if stable_count >= 2:
            score += 5
        elif stable_count == 1:
            score += 2

        if moving_count >= 2:
            score += 4
        elif moving_count == 1:
            score += 2

        if racket_near_stable:
            score += 4
        elif rackets_near > 0:
            score += 2

        if max_movement > MOVEMENT_THRESHOLD_HIGH:
            score += 2

        if max_stable_duration > 60:
            score += 2

        if len(nearby_tracks) >= 2:
            score += 1

        if not self.rules["ignore_ball"] and ball_moving and len(nearby_tracks) >= 1:
            score += 5
            print(f"  [Table {self.table_id}] BALL MOVING +5", flush=True)

        score += self.rules["score_bonus"]

        if len(nearby_tracks) > 0 or rackets_near > 0 or ball_moving:
            print(f"  [Table {self.table_id}] near={len(nearby_tracks)} "
                  f"stable={stable_count} moving={moving_count} "
                  f"ball={'y' if ball_moving else 'n'} "
                  f"rackets={rackets_near} mov={int(max_movement)} "
                  f"score={score}", flush=True)

        return score

    def update(self, score):
        now = time.time()

        if self.session_start is None:
            is_active = score >= self.rules["score_threshold"]
        else:
            is_active = score >= self.rules["exit_threshold"]

        if is_active:
            self.pending_confirmations += 1
            if self.pending_confirmations >= CONFIRMATION_SECONDS and self.session_start is None:
                self.session_start = now - (CONFIRMATION_SECONDS * FRAME_INTERVAL)
                print(f"[Table {self.table_id}] > Session started (score={score})", flush=True)
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
                        print(f"[Table {self.table_id}] < Saved {int(duration)} sec", flush=True)
                    else:
                        print(f"[Table {self.table_id}] < Too short ({int(duration)} sec)", flush=True)
                    self.session_start = None
                    self.last_active = None


def get_box_center(xyxy):
    """Берём точку близко к НОГАМ — для гомографии важна точка касания пола."""
    x1, y1, x2, y2 = xyxy
    cx = int((x1 + x2) / 2)
    cy = int(y1 + (y2 - y1) * 0.95)
    return (cx, cy)


def process_camera(camera, zones):
    print(f"[{camera['name']}] Loading YOLO...", flush=True)
    model = YOLO('yolov8n.pt')
    print(f"[{camera['name']}] Connecting...", flush=True)

    table_rules = load_table_rules()
    homographies = load_homographies()
    H = homographies.get(camera['id'])

    if H is None:
        print(f"[{camera['name']}] WARNING: no homography for this camera, working in pixel mode", flush=True)
    else:
        print(f"[{camera['name']}] Homography loaded", flush=True)

    cap_test = cv2.VideoCapture(camera['rtsp'], cv2.CAP_FFMPEG)
    frame_w, frame_h = 1920, 1080
    if cap_test.isOpened():
        ret, frame_test = cap_test.read()
        if ret:
            frame_h, frame_w = frame_test.shape[:2]
    cap_test.release()
    print(f"[{camera['name']}] Resolution: {frame_w}x{frame_h}", flush=True)

    table_trackers = {}
    for table_id_str, zone in zones.items():
        if zone.get('camera_id') == camera['id']:
            table_id = int(table_id_str)
            rules = get_rules_for_table(table_rules, table_id)
            tracker = TableTracker(table_id, zone, H, rules)
            tracker.init_mask(frame_h, frame_w)
            table_trackers[table_id] = tracker
            mode = 'homography' if tracker.use_homography else 'pixel'
            print(f"[{camera['name']}] Table {table_id} ({mode}) "
                  f"min_stable={rules['min_stable_people']}, "
                  f"threshold={rules['score_threshold']}, "
                  f"bonus={rules['score_bonus']}", flush=True)

    if not table_trackers:
        print(f"[{camera['name']}] No zones marked", flush=True)
        return

    print(f"[{camera['name']}] Tables: {sorted(table_trackers.keys())}", flush=True)

    person_tracks = {}

    while True:
        cap = cv2.VideoCapture(camera['rtsp'], cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{camera['name']}] Cannot connect, retry in 10s", flush=True)
            time.sleep(10)
            continue

        print(f"[{camera['name']}] Connected", flush=True)
        last_process_time = 0
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{camera['name']}] Stream broken", flush=True)
                    break

                if time.time() - last_process_time < FRAME_INTERVAL:
                    continue
                last_process_time = time.time()
                frame_count += 1

                # Сохраняем снимок каждые 2 секунды
                cam_id = camera['id']
                now_snap = time.time()
                if now_snap - last_snapshot_time.get(cam_id, 0) > 2:
                    try:
                        small = cv2.resize(frame, (960, 540))
                        tmp = os.path.join(SNAPSHOTS_DIR, f'cam_{cam_id}_tmp.jpg')
                        final = os.path.join(SNAPSHOTS_DIR, f'cam_{cam_id}.jpg')
                        cv2.imwrite(tmp, small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        os.replace(tmp, final)
                        last_snapshot_time[cam_id] = now_snap
                    except Exception:
                        pass

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
                    print(f"[{camera['name']}] YOLO error: {e}", flush=True)
                    continue

                rackets = []

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

                dead = [tid for tid, t in person_tracks.items() if not t.is_alive()]
                for tid in dead:
                    del person_tracks[tid]

                alive_count = len([t for t in person_tracks.values() if t.is_alive()])

                if frame_count % 30 == 0:
                    print(f"[{camera['name']}] Frame #{frame_count} | "
                          f"People: {alive_count} | Rackets: {len(rackets)}", flush=True)

                # Привязка людей к ближайшему столу
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
                        # Дистанция считается на ПОЛУ если есть гомография
                        if tracker.use_homography:
                            floor_pt = tracker.pixel_to_floor(pos)
                            if floor_pt is None:
                                continue
                            center_floor = tracker.zone_floor.mean(axis=0)
                            dx = floor_pt[0] - center_floor[0]
                            dy = floor_pt[1] - center_floor[1]
                        else:
                            center = tracker.zone_pixel.mean(axis=0)
                            dx = pos[0] - center[0]
                            dy = pos[1] - center[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_table = tid

                    if best_table is not None:
                        table_persons[best_table][track_id] = track

                # Ракетки к ближайшему столу
                table_rackets = {tid: [] for tid in table_trackers}
                for racket_pt in rackets:
                    best_table = None
                    best_dist = float('inf')
                    for tid, tracker in table_trackers.items():
                        if not tracker.point_in_expanded(racket_pt):
                            continue
                        if tracker.use_homography:
                            floor_pt = tracker.pixel_to_floor(racket_pt)
                            if floor_pt is None:
                                continue
                            center_floor = tracker.zone_floor.mean(axis=0)
                            dx = floor_pt[0] - center_floor[0]
                            dy = floor_pt[1] - center_floor[1]
                        else:
                            center = tracker.zone_pixel.mean(axis=0)
                            dx = racket_pt[0] - center[0]
                            dy = racket_pt[1] - center[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_table = tid
                    if best_table is not None:
                        table_rackets[best_table].append(racket_pt)

                for tid, tracker in table_trackers.items():
                    score = tracker.calculate_score(
                        table_persons[tid],
                        table_rackets[tid],
                        frame
                    )
                    tracker.update(score)

                if frame_count % 5 == 0:
                    active = {}
                    for tid, tracker in table_trackers.items():
                        if tracker.session_start is not None:
                            active[str(tid)] = tracker.session_start
                    save_active_sessions(camera['id'], active)

        except Exception as e:
            import traceback
            print(f"[{camera['name']}] Critical error: {e}", flush=True)
            traceback.print_exc()
            time.sleep(5)
        finally:
            cap.release()


def main():
    init_db()

    if not os.path.exists(ZONES_FILE):
        print("Mark tables first: python backend/marker_v2.py")
        return

    with open(ZONES_FILE, 'r') as f:
        zones = json.load(f)
    print(f"Loaded zones: {len(zones)}")

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

    print("\nAnalysis running. Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
