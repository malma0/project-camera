"""
download_model.py — скачивает yolo11s.pt в models/.

Ultralytics при первом вызове YOLO('yolo11s.pt') сам скачает веса, если их нет.
Этот скрипт просто триггерит скачивание заранее (чтобы не ждать при старте analyzer.py)
и кладёт файл в правильное место.

Запуск:  python scripts/download_model.py
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TARGET = os.path.join(MODELS_DIR, 'yolo11s.pt')


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(TARGET):
        size_mb = os.path.getsize(TARGET) / 1024 / 1024
        print(f'Already exists: {TARGET} ({size_mb:.1f} MB)')
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print('ERROR: ultralytics not installed. Run: pip install ultralytics')
        sys.exit(1)

    # YOLO() сам скачает в текущую рабочую директорию или кеш
    print('Downloading yolo11s.pt via ultralytics...')
    model = YOLO('yolo11s.pt')

    # Ищем где он лёг и копируем в models/
    candidate_paths = [
        'yolo11s.pt',
        os.path.expanduser('~/.cache/ultralytics/yolo11s.pt'),
    ]
    # Сам объект тоже знает свой путь
    try:
        src = str(model.ckpt_path) if hasattr(model, 'ckpt_path') else None
        if src and os.path.exists(src):
            candidate_paths.insert(0, src)
    except Exception:
        pass

    src_found = None
    for p in candidate_paths:
        if os.path.exists(p):
            src_found = p
            break

    if src_found is None:
        print('ERROR: downloaded yolo11s.pt not found in expected locations')
        print('Tried:', candidate_paths)
        sys.exit(1)

    if os.path.abspath(src_found) != os.path.abspath(TARGET):
        shutil.copy2(src_found, TARGET)
        print(f'Copied {src_found} → {TARGET}')
    else:
        print(f'Already in place: {TARGET}')

    size_mb = os.path.getsize(TARGET) / 1024 / 1024
    print(f'Done. Size: {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
