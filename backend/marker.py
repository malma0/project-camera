import cv2
import json
import sys, os
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERAS, ZONES_FILE


def get_frame_from_camera(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть поток")
    
    # Пропускаем первые кадры (часто битые)
    for _ in range(5):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось получить кадр")
    return frame


def mark_camera(camera, existing_zones):
    print(f"\n=== Разметка для {camera['name']} ===")
    print(f"Столы: {camera['tables']}")
    print("Размечайте САМ СТОЛ (его поверхность сверху), достаточно 4 точек по углам")
    print("Управление:")
    print("  ЛКМ — добавить точку")
    print("  ПКМ — удалить последнюю точку")
    print("  n   — следующий стол")
    print("  b   — назад к предыдущему столу")
    print("  s   — сохранить и выйти")
    print("  q   — выйти без сохранения\n")
    
    frame = get_frame_from_camera(camera['rtsp'])
    
    # Загружаем уже существующие зоны для этой камеры
    zones = {}
    for table_id_str, zone_data in existing_zones.items():
        if zone_data.get('camera_id') == camera['id']:
            zones[int(table_id_str)] = zone_data['points']
    
    current_table_idx = 0
    current_points = []
    
    # Если эта зона уже была размечена — пропускаем дальше
    while (current_table_idx < len(camera['tables']) 
           and camera['tables'][current_table_idx] in zones):
        current_table_idx += 1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current_points:
            current_points.pop()
    
    window_name = f"Разметка - {camera['name']}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = frame.copy()
        
        # Уже размеченные зоны
        for table_id, points in zones.items():
            pts = np.array(points, np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(display, f"#{table_id}", (cx-15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Текущая зона в процессе разметки
        if current_table_idx < len(camera['tables']):
            current_table = camera['tables'][current_table_idx]
            for pt in current_points:
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
            if len(current_points) > 1:
                pts = np.array(current_points, np.int32)
                cv2.polylines(display, [pts], False, (0, 0, 255), 2)
            
            cv2.putText(display, f"Стол #{current_table} ({len(current_points)} точек)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Все столы размечены. Нажмите 's' чтобы сохранить",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('n') and current_table_idx < len(camera['tables']):
            if len(current_points) >= 3:
                table_id = camera['tables'][current_table_idx]
                zones[table_id] = current_points.copy()
                print(f"✓ Стол #{table_id} размечен ({len(current_points)} точек)")
                current_points = []
                current_table_idx += 1
            else:
                print("⚠ Нужно минимум 3 точки!")
        elif key == ord('b') and current_table_idx > 0:
            current_table_idx -= 1
            table_id = camera['tables'][current_table_idx]
            current_points = zones.pop(table_id, [])
            print(f"← Возврат к столу #{table_id}")
        elif key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return zones


def main():
    all_zones = {}
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, 'r') as f:
            all_zones = json.load(f)
    
    for camera in CAMERAS:
        try:
            zones = mark_camera(camera, all_zones)
        except Exception as e:
            print(f"❌ Ошибка с {camera['name']}: {e}")
            continue
        
        if zones is None:
            print(f"Разметка {camera['name']} отменена")
            continue
        
        # Удаляем старые зоны для этой камеры
        all_zones = {k: v for k, v in all_zones.items() 
                     if v.get('camera_id') != camera['id']}
        
        # Добавляем новые
        for table_id, points in zones.items():
            all_zones[str(table_id)] = {
                "camera_id": camera['id'],
                "points": points
            }
    
    with open(ZONES_FILE, 'w') as f:
        json.dump(all_zones, f, indent=2)
    print(f"\n✅ Зоны сохранены в {ZONES_FILE}")


if __name__ == "__main__":
    main()