"""
convert_to_openvino.py — конвертирует yolo11s.pt → OpenVINO INT8 с калибровкой
на реальных кадрах из data/calibration_dataset/.

Зачем:
  • Без квантизации YOLO11s на CPU 4-6 vCPU даёт ~1-2 fps на 1 камеру.
  • INT8 + OpenVINO даёт x3-4 ускорение → ~5-8 fps на 1 камеру.
  • С калибровкой на ваших кадрах потеря mAP < 0.5% vs FP32.
  • Без калибровки — потеря может быть 1-3%, особенно на мелких объектах
    (а у вас столы 6 и 11 — именно мелкие/перекрытые).

Запуск:
  1. Собрать ≥100, желательно 200-300, репрезентативных кадров в data/calibration_dataset/
     (запустить analyzer.py с COLLECT_CALIBRATION=1 на 15-20 минут в час пик).
  2. python scripts/convert_to_openvino.py
  3. Перезапустить analyzer.py — он автоматически подхватит OpenVINO модель.

Выход:
  models/yolo11s_int8_openvino_model/   (директория, не файл)
"""

import os
import sys
import glob
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
CALIB_DIR    = os.path.join(DATA_DIR, 'calibration_dataset')

PT_PATH      = os.path.join(MODELS_DIR, 'yolo11s.pt')
PT_FALLBACK  = os.path.join(MODELS_DIR, 'yolov8s.pt')

OUT_INT8_DIR = os.path.join(MODELS_DIR, 'yolo11s_int8_openvino_model')
OUT_FP32_DIR = os.path.join(MODELS_DIR, 'yolo11s_openvino_model')


# ------------------------------------------------------------------ #
#  Проверки
# ------------------------------------------------------------------ #

def check_deps():
    missing = []
    try:
        import ultralytics  # noqa
    except ImportError:
        missing.append('ultralytics')
    try:
        import openvino  # noqa
    except ImportError:
        missing.append('openvino')
    try:
        import nncf  # noqa
    except ImportError:
        missing.append('nncf')

    if missing:
        print('ERROR: missing packages. Install:')
        print(f'   pip install {" ".join(missing)}')
        sys.exit(1)


def find_model():
    """Возвращает (model_path, is_yolo11). Сначала пытаемся yolo11s.pt, потом yolov8s.pt."""
    if os.path.exists(PT_PATH):
        return PT_PATH, True
    if os.path.exists(PT_FALLBACK):
        print(f'NOTE: yolo11s.pt not found, falling back to yolov8s.pt')
        return PT_FALLBACK, False
    print('ERROR: no source model found in models/')
    print(f'  Run: python scripts/download_model.py')
    sys.exit(1)


def check_calibration_dataset():
    """Проверяет что есть достаточно калибровочных кадров."""
    if not os.path.isdir(CALIB_DIR):
        return []
    images = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')) +
                    glob.glob(os.path.join(CALIB_DIR, '*.png')))
    return images


# ------------------------------------------------------------------ #
#  Конвертация в OpenVINO FP32
# ------------------------------------------------------------------ #

def export_fp32(pt_path):
    """
    Ultralytics export → OpenVINO FP32. Это промежуточный шаг,
    из которого мы потом будем квантовать в INT8.
    """
    from ultralytics import YOLO

    print(f'\n[1/3] Exporting {os.path.basename(pt_path)} → OpenVINO FP32...')

    # Если уже экспортировано — используем существующую папку
    candidate = pt_path.replace('.pt', '_openvino_model')
    if os.path.isdir(candidate):
        xml_files = glob.glob(os.path.join(candidate, '*.xml'))
        if xml_files:
            print(f'      Found existing OpenVINO FP32 model: {candidate}')
            if os.path.abspath(candidate) != os.path.abspath(OUT_FP32_DIR):
                if os.path.exists(OUT_FP32_DIR):
                    shutil.rmtree(OUT_FP32_DIR)
                shutil.copytree(candidate, OUT_FP32_DIR)
            print(f'      OpenVINO FP32 model: {OUT_FP32_DIR}')
            return OUT_FP32_DIR

    model = YOLO(pt_path)
    exported = model.export(format='openvino', imgsz=640, half=False, dynamic=False)
    print(f'      Exported: {exported}')

    src_dir = exported if os.path.isdir(exported) else os.path.dirname(exported)
    if not os.path.isdir(src_dir):
        cand = pt_path.replace('.pt', '_openvino_model')
        if os.path.isdir(cand):
            src_dir = cand
    if not os.path.isdir(src_dir):
        print(f'ERROR: cannot locate exported OpenVINO directory; got {exported}')
        sys.exit(1)

    if os.path.exists(OUT_FP32_DIR):
        shutil.rmtree(OUT_FP32_DIR)
    if os.path.abspath(src_dir) != os.path.abspath(OUT_FP32_DIR):
        shutil.copytree(src_dir, OUT_FP32_DIR)
    print(f'      OpenVINO FP32 model: {OUT_FP32_DIR}')
    return OUT_FP32_DIR


# ------------------------------------------------------------------ #
#  Квантизация в INT8 через NNCF
# ------------------------------------------------------------------ #

def quantize_int8(fp32_dir, calib_images, imgsz=640):
    """
    Использует NNCF Post-Training Quantization с реальными калибровочными кадрами.
    Сохраняет результат в OUT_INT8_DIR.
    """
    import numpy as np
    import cv2
    import openvino as ov
    import nncf

    print(f'\n[2/3] Quantizing to INT8 with {len(calib_images)} calibration images...')

    # Загружаем FP32 модель
    xml_files = glob.glob(os.path.join(fp32_dir, '*.xml'))
    if not xml_files:
        print(f'ERROR: no .xml found in {fp32_dir}')
        sys.exit(1)
    xml_path = xml_files[0]

    core = ov.Core()
    model = core.read_model(xml_path)

    # Препроцессинг калибровочных кадров: точно такой же как у Ultralytics
    # (letterbox 640x640, BGR→RGB, /255, HWC→CHW, batch=1).
    def preprocess(img_bgr, size=imgsz):
        h, w = img_bgr.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # letterbox padding
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top  = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        # BGR→RGB, normalize, HWC→CHW, add batch
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)[None, ...]
        return chw

    # NNCF Dataset из генератора
    def calib_generator():
        for path in calib_images:
            img = cv2.imread(path)
            if img is None:
                continue
            yield preprocess(img)

    calibration_dataset = nncf.Dataset(calib_generator())

    quantized = nncf.quantize(
        model,
        calibration_dataset,
        # consensus-настройки для object detection
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=min(len(calib_images), 300),
    )

    # Сохраняем
    if os.path.exists(OUT_INT8_DIR):
        shutil.rmtree(OUT_INT8_DIR)
    os.makedirs(OUT_INT8_DIR)

    # OpenVINO 2024+: save_model. На старых версиях — serialize.
    int8_xml = os.path.join(OUT_INT8_DIR, os.path.basename(xml_path))
    try:
        ov.save_model(quantized, int8_xml, compress_to_fp16=False)
    except AttributeError:
        from openvino.runtime import serialize
        serialize(quantized, int8_xml)

    # Копируем метаданные (Ultralytics кладёт metadata.yaml рядом — без него YOLO не найдёт classes)
    for fn in os.listdir(fp32_dir):
        if fn.endswith('.yaml') or fn.endswith('.json'):
            shutil.copy2(os.path.join(fp32_dir, fn), os.path.join(OUT_INT8_DIR, fn))

    print(f'      INT8 model: {OUT_INT8_DIR}')
    return OUT_INT8_DIR


# ------------------------------------------------------------------ #
#  Проверка работоспособности
# ------------------------------------------------------------------ #

def smoke_test(int8_dir, calib_images):
    """Загружает INT8 модель через ultralytics и делает inference на 1 кадре."""
    from ultralytics import YOLO
    print(f'\n[3/3] Smoke test: loading {int8_dir} via Ultralytics...')
    try:
        model = YOLO(int8_dir, task='detect')
        test_img = calib_images[0] if calib_images else None
        if test_img:
            res = model.predict(test_img, imgsz=640, conf=0.35, classes=[0], verbose=False)
            n = 0 if res[0].boxes is None else len(res[0].boxes)
            print(f'      OK. Detected {n} people on test image.')
        else:
            print(f'      OK (loaded, no test image to predict on).')
    except Exception as e:
        print(f'ERROR: smoke test failed: {e}')
        print('       Model files exist but cannot be loaded by Ultralytics.')
        print('       Check OpenVINO version compatibility.')
        sys.exit(1)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-images', type=int, default=100,
                        help='Минимум калибровочных кадров (по умолчанию 100)')
    parser.add_argument('--skip-int8', action='store_true',
                        help='Только FP32, без квантизации (если NNCF падает)')
    args = parser.parse_args()

    check_deps()

    pt_path, is_yolo11 = find_model()
    print(f'Source model: {pt_path}')

    calib_images = check_calibration_dataset()
    print(f'Calibration images: {len(calib_images)} in {CALIB_DIR}')

    if not args.skip_int8 and len(calib_images) < args.min_images:
        print(f'\nERROR: need at least {args.min_images} calibration images, have {len(calib_images)}.')
        print(f'  Run analyzer.py with COLLECT_CALIBRATION=1 for 15-20 min in busy hours.')
        print(f'  Or run with --skip-int8 to export FP32 only (slower at runtime).')
        sys.exit(1)

    # Шаг 1: экспорт в OpenVINO FP32 (всегда)
    fp32_dir = export_fp32(pt_path)

    # Шаг 2: INT8 квантизация (если есть калибровочные кадры)
    if args.skip_int8:
        print('\nSkipping INT8 quantization (--skip-int8).')
        print(f'FP32 model ready at: {fp32_dir}')
        print('analyzer.py will use it automatically (FP32 fallback).')
        return

    int8_dir = quantize_int8(fp32_dir, calib_images)

    # Шаг 3: smoke-test
    smoke_test(int8_dir, calib_images)

    print('\n' + '=' * 60)
    print('DONE.')
    print(f'  INT8 model: {int8_dir}')
    print(f'  FP32 fallback: {fp32_dir}')
    print('Restart analyzer.py — it will pick up INT8 automatically.')
    print('=' * 60)


if __name__ == '__main__':
    main()
