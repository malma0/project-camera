"""
Marker with homography (v2.1 - with padding for off-frame clicks).

Workflow:
1. For each camera:
   - STAGE 1 (CALIBRATION): click 4 corners of ONE reference table
   - Press SPACE to go to stage 2
   - STAGE 2 (MARKING): click 4 corners of each table, after 4 clicks
     enter table number directly in the window
   - Press S to save and move to next camera
   - Press ESC to skip camera

Corner click order (ALWAYS):
   1. Far-left
   2. Far-right
   3. Near-right
   4. Near-left

PADDING: 200px gray border is added around the frame so you can click
on points that are outside the visible camera area. Homography handles
off-frame coordinates correctly.
"""

import cv2
import json
import numpy as np
import os
import sys

TABLE_LENGTH_MM = 2740
TABLE_WIDTH_MM = 1525

# Размер серой рамки вокруг кадра в пикселях — чтобы можно было кликать "за кадром"
PADDING = 200

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERAS, ZONES_FILE


CALIB_REAL_CORNERS = np.array([
    [0,                0],
    [TABLE_WIDTH_MM,   0],
    [TABLE_WIDTH_MM,   TABLE_LENGTH_MM],
    [0,                TABLE_LENGTH_MM],
], dtype=np.float32)


class MarkerState:
    def __init__(self):
        self.mode = 'calibration'
        self.current_points = []   # хранит координаты В СИСТЕМЕ КАДРА (без padding)
        self.calib_points = []
        self.homography = None
        self.tables = {}
        self.original_frame = None
        self.camera_id = None
        self.entering_id = False
        self.id_buffer = ''


def pixel_to_floor(point, H):
    src = np.array([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return [float(dst[0][0][0]), float(dst[0][0][1])]


def add_padding(frame, padding):
    """Создаёт серую рамку вокруг кадра."""
    h, w = frame.shape[:2]
    padded = np.full((h + 2 * padding, w + 2 * padding, 3), 60, dtype=np.uint8)
    padded[padding:padding + h, padding:padding + w] = frame
    # Тонкая рамка вокруг кадра
    cv2.rectangle(padded, (padding - 1, padding - 1),
                  (padding + w, padding + h), (120, 120, 120), 1)
    return padded


def draw_overlay(state):
    """Рисует на padded-кадре. Все координаты точек в системе кадра, отображаем со сдвигом PADDING."""
    padded = add_padding(state.original_frame, PADDING)
    h, w = padded.shape[:2]
    P = PADDING

    if state.mode == 'calibration':
        text_lines = [
            f"CAMERA {state.camera_id} - STAGE 1: CALIBRATION",
            f"Click 4 corners of calibration table ({len(state.calib_points)}/4)",
            "Order: FAR-LEFT -> FAR-RIGHT -> NEAR-RIGHT -> NEAR-LEFT",
            "Gray border = off-frame area (can click here too)",
            "[R] reset  [ESC] skip camera",
        ]
        color_bg = (40, 40, 120)
    elif state.entering_id:
        text_lines = [
            f"ENTER TABLE NUMBER: {state.id_buffer}_",
            "Type digits, [ENTER] confirm, [BACKSPACE] erase, [ESC] cancel",
        ]
        color_bg = (120, 80, 0)
    else:
        text_lines = [
            f"CAMERA {state.camera_id} - STAGE 2: MARK TABLES",
            f"Marked: {len(state.tables)} | Current: {len(state.current_points)}/4",
            "Click 4 corners (same order). Gray border = off-frame OK.",
            "After 4 clicks - enter table number.",
            "[R] reset  [S] save and next camera  [ESC] skip",
        ]
        color_bg = (40, 120, 40)

    overlay = padded.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), color_bg, -1)
    cv2.addWeighted(overlay, 0.75, padded, 0.25, 0, padded)

    for i, line in enumerate(text_lines):
        cv2.putText(padded, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Калибровочные точки (рисуем с padding)
    for i, pt in enumerate(state.calib_points):
        disp = (pt[0] + P, pt[1] + P)
        cv2.circle(padded, disp, 8, (0, 255, 255), -1)
        cv2.putText(padded, str(i + 1), (disp[0] + 12, disp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(state.calib_points) == 4:
        pts = np.array([[p[0] + P, p[1] + P] for p in state.calib_points], np.int32)
        cv2.polylines(padded, [pts], True, (0, 255, 255), 2)

    # Готовые столы
    for table_id, t in state.tables.items():
        pts = np.array([[p[0] + P, p[1] + P] for p in t['points_pixel']], np.int32)
        cv2.polylines(padded, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(padded, str(table_id), tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Текущие точки
    for i, pt in enumerate(state.current_points):
        disp = (pt[0] + P, pt[1] + P)
        cv2.circle(padded, disp, 6, (0, 165, 255), -1)
        cv2.putText(padded, str(i + 1), (disp[0] + 10, disp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if len(state.current_points) > 1:
        pts = np.array([[p[0] + P, p[1] + P] for p in state.current_points], np.int32)
        cv2.polylines(padded, [pts], False, (0, 165, 255), 2)

    return padded


def on_mouse(event, x, y, flags, state):
    """Координаты приходят в системе padded окна. Переводим в систему кадра."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.entering_id:
        return

    # Снимаем padding — получаем координаты в системе кадра (могут быть отрицательные или > w/h)
    fx = x - PADDING
    fy = y - PADDING

    if state.mode == 'calibration':
        if len(state.calib_points) < 4:
            state.calib_points.append([fx, fy])
            if len(state.calib_points) == 4:
                src = np.array(state.calib_points, dtype=np.float32)
                state.homography = cv2.getPerspectiveTransform(src, CALIB_REAL_CORNERS)
                print(f"[Camera {state.camera_id}] Homography calculated. Press SPACE to continue.")
    else:
        if len(state.current_points) < 4:
            state.current_points.append([fx, fy])
            if len(state.current_points) == 4:
                state.entering_id = True
                state.id_buffer = ''


def capture_frame(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def process_camera(camera):
    print(f"\n{'=' * 60}")
    print(f"CAMERA {camera['id']} - {camera['name']}")
    print('=' * 60)
    print("Connecting...")

    frame = capture_frame(camera['rtsp'])
    if frame is None:
        print(f"[ERROR] Cannot connect to camera {camera['id']}")
        return {}, None

    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")

    state = MarkerState()
    state.original_frame = frame
    state.camera_id = camera['id']

    win_name = f"Camera {camera['id']} - Marker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    padded_w = w + 2 * PADDING
    padded_h = h + 2 * PADDING
    scale = min(1.0, 1500 / padded_w, 900 / padded_h)
    cv2.resizeWindow(win_name, int(padded_w * scale), int(padded_h * scale))
    cv2.setMouseCallback(win_name, on_mouse, state)

    while True:
        display = draw_overlay(state)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        if state.entering_id:
            if key == 27:
                state.entering_id = False
                state.id_buffer = ''
                state.current_points = []
                print("Table input cancelled")
            elif key == 8 or key == 127:
                state.id_buffer = state.id_buffer[:-1]
            elif key == 13 or key == 10:
                if state.id_buffer.isdigit():
                    table_id = int(state.id_buffer)
                    floor_points = [pixel_to_floor(pt, state.homography)
                                    for pt in state.current_points]
                    state.tables[table_id] = {
                        'points_floor': floor_points,
                        'points_pixel': list(state.current_points),
                    }
                    print(f"[Camera {state.camera_id}] Table {table_id} saved")
                state.current_points = []
                state.entering_id = False
                state.id_buffer = ''
            elif ord('0') <= key <= ord('9'):
                state.id_buffer += chr(key)
            continue

        if key == 27:
            print(f"[Camera {camera['id']}] Skipped")
            break

        if key == ord('r'):
            if state.mode == 'calibration':
                state.calib_points = []
                state.homography = None
                print("Calibration reset")
            else:
                state.current_points = []
                print("Current table reset")

        if state.mode == 'calibration' and state.homography is not None and key == 32:
            state.mode = 'tables'
            print(f"[Camera {camera['id']}] Now mark tables")

        if state.mode == 'tables' and key == ord('s'):
            print(f"[Camera {camera['id']}] Saved {len(state.tables)} tables")
            break

    cv2.destroyWindow(win_name)

    result = {}
    for table_id, t in state.tables.items():
        result[table_id] = {
            'camera_id': camera['id'],
            'points_floor': t['points_floor'],
            'points_pixel': t['points_pixel'],
        }

    homography_list = state.homography.tolist() if state.homography is not None else None
    return result, homography_list


def main():
    print("=" * 60)
    print("MARKER v2.1 - WITH HOMOGRAPHY AND PADDING")
    print("=" * 60)
    print("Gray border around frame = can click here for off-frame points")
    print("Stage 1: click 4 corners of ONE reference table per camera")
    print("Stage 2: click 4 corners of each table, then type its number")
    print("Order: far-left -> far-right -> near-right -> near-left")
    print()

    all_zones = {}
    homographies = {}

    for camera in CAMERAS:
        zones, H = process_camera(camera)
        all_zones.update({str(tid): z for tid, z in zones.items()})
        if H is not None:
            homographies[str(camera['id'])] = H

    if not all_zones:
        print("\n[ERROR] Nothing marked, not saving")
        return

    if os.path.exists(ZONES_FILE):
        backup = ZONES_FILE + '.backup'
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(ZONES_FILE, backup)
        print(f"\nBackup: {backup}")

    with open(ZONES_FILE, 'w') as f:
        json.dump(all_zones, f, indent=2, ensure_ascii=False)

    hom_file = os.path.join(os.path.dirname(ZONES_FILE), 'homography.json')
    with open(hom_file, 'w') as f:
        json.dump(homographies, f, indent=2)

    print(f"\nDONE: {len(all_zones)} tables marked")
    print(f"  {ZONES_FILE}")
    print(f"  {hom_file}")


if __name__ == "__main__":
    main()
