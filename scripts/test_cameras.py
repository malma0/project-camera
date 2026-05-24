import cv2
import sys, os
import time

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from config import CAMERAS, DATA_DIR


def test_camera(camera):
    print(f"\n[{camera['name']}] Подключение...")
    cap = cv2.VideoCapture(camera['rtsp'], cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"[{camera['name']}] ❌ Не удалось открыть поток")
        return False
    
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"[{camera['name']}] ❌ Не удалось получить кадр")
        return False
    
    output_path = os.path.join(DATA_DIR, f"test_camera_{camera['id']}.jpg")
    cv2.imwrite(output_path, frame)
    
    h, w = frame.shape[:2]
    print(f"[{camera['name']}] ✅ Кадр {w}x{h} сохранён: {output_path}")
    return True


if __name__ == "__main__":
    for cam in CAMERAS:
        test_camera(cam)
    print("\nГотово! Открой файлы в data/ чтобы посмотреть")