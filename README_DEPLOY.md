# Развёртывание: новый analyzer.py + OpenVINO INT8

Пошаговая инструкция. Идите по порядку, не пропускайте шаги.

## 0. Установка файлов

Скопируйте файлы из этой поставки в проект:

| Откуда (из поставки) | Куда (в репозитории) |
|---|---|
| `analyzer.py` | `backend/analyzer.py` (старый — в бэкап!) |
| `download_model.py` | `scripts/download_model.py` |
| `convert_to_openvino.py` | `scripts/convert_to_openvino.py` |
| `table_rules.json` | `data/table_rules.json` (старый — в бэкап!) |
| `requirements.txt` | в корень, при желании смержите со своим |

```bash
mv backend/analyzer.py backend/analyzer.py.bak
mv data/table_rules.json data/table_rules.json.bak
# затем кладите новые файлы на их места
mkdir -p scripts models data/snapshots data/calibration_dataset
```

## 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

На CPU-сервере без GPU это поставит ultralytics в CPU-only режиме (torch+cpu). Если pip ругается на torch — поставьте его явно:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Скачивание модели

```bash
python scripts/download_model.py
```

Должно появиться `models/yolo11s.pt` (~18 МБ).

## 3. Первый запуск analyzer.py с PyTorch (для проверки логики)

```bash
python backend/analyzer.py
```

Что должно произойти:
- В логах: `Loading YOLO from .../yolo11s.pt` (PyTorch fallback — это нормально для первого запуска).
- В логах для каждой камеры: `connected`, `geo built for tables: [...]`.
- Через 30-60 секунд: окна прогреваются, начинают появляться события `START`/`STOP`.
- Файл `data/active.json` обновляется когда столы становятся активны/свободны.
- Файлы `data/snapshots/cam_1.jpg` и `cam_2.jpg` обновляются раз в 5 секунд (для дашборда).

Если что-то падает — внимательно читайте traceback. Самые вероятные проблемы:
- `ModuleNotFoundError: config` → запускаете не из корня проекта. Запускайте именно `python backend/analyzer.py` из корня.
- RTSP не подключается → проверьте URL в `config.py`, попробуйте `ffmpeg -i <rtsp_url>` с того же сервера.
- Очень медленный (>2 сек/тик) → ожидаемо для PyTorch на CPU. Это исправит шаг 5.

Дайте поработать 2-3 минуты, проверьте дашборд — статистика должна обновляться. **Если всё работает корректно, переходите к шагу 4.**

## 4. Сбор калибровочного датасета

Включите режим сбора и запустите analyzer.py в час пик (когда играют на разных столах одновременно — стол 6, 11, разные сценарии: люди играют / зрители / пустой стол):

```bash
COLLECT_CALIBRATION=1 python backend/analyzer.py
```

В логах увидите: `CALIBRATION MODE: saving frames to .../calibration_dataset`.

Дайте поработать **15-20 минут**. За это время насчёт ~400 кадров суммарно с обеих камер (4 fps × 2 камеры × 60 секунд × 20 минут / 5 семплинг ≈ 384 кадра, но cap=400, так что не больше). Проверьте:

```bash
ls data/calibration_dataset/ | wc -l    # должно быть 100-400
ls data/calibration_dataset/ | head     # должны быть cam1_*.jpg и cam2_*.jpg
```

Остановите analyzer (Ctrl+C). Если по логам видно мало `cam2_*` — значит вторая камера отваливалась, проверьте подключение.

## 5. Конвертация в OpenVINO INT8

```bash
python scripts/convert_to_openvino.py
```

Что произойдёт:
- `[1/3]` Экспорт PyTorch → OpenVINO FP32 (1-2 мин).
- `[2/3]` Квантизация в INT8 с вашим калибровочным датасетом (2-5 мин). NNCF прогонит ~300 кадров через сеть для подбора оптимальных диапазонов.
- `[3/3]` Smoke-test: загрузка INT8 модели через Ultralytics и инференс на 1 кадре.

Если квантизация падает (`nncf.quantize` чем-то недоволен) — запустите с `--skip-int8`. Получите FP32 модель: чуть медленнее INT8, но всё равно в 2 раза быстрее PyTorch.

```bash
python scripts/convert_to_openvino.py --skip-int8
```

После успешной конвертации появится `models/yolo11s_int8_openvino_model/` (или `..._openvino_model/` для FP32).

## 6. Финальный запуск

```bash
python backend/analyzer.py
```

В логах должно быть: `Loading YOLO from .../yolo11s_int8_openvino_model (OpenVINO INT8)`.

Ожидаемый fps: 4-8 на каждую камеру (зависит от vCPU). Это значительно больше нужных 2 fps — есть запас.

## 7. Что ожидать по точности

После 1-2 часов работы посмотрите в БД:

```bash
sqlite3 data/stats.db "SELECT table_id, COUNT(*), SUM(duration_seconds)/60 as minutes \
  FROM sessions WHERE date(start_time)=date('now') GROUP BY table_id ORDER BY table_id;"
```

И сравните с реальной картиной. **Реалистичная цель: 91-93% точности.** Если по конкретному столу видите сильное расхождение:

- **Завышает (FP)** — стол отмечается занятым когда пуст. Увеличьте `window_start_ratio` этого стола в `table_rules.json` (например с 0.65 → 0.75). Или верните `require_movement: true`.
- **Занижает (FN)** — реальная игра пропускается. Уменьшите `window_start_ratio` (например с 0.65 → 0.55). Или включите `min_stable_people: 1` (вместо 2).

После правки `table_rules.json` достаточно перезапустить analyzer.py — модель не трогается.

## 8. Дашборд и БД

Не трогали. `backend/app.py`, `frontend/`, схема `sessions` — всё работает как раньше. Контракт сохранён:
- `data/active.json` пишется в формате `{"<table_id>": {"start": <unix_ts>}}`.
- `data/snapshots/cam_<id>.jpg` обновляется каждые 5 сек.
- Сессии пишутся в SQLite в том же формате (timestamp с миллисекундами).

## Troubleshooting

| Симптом | Что делать |
|---|---|
| После 22:05 продолжаются сессии | Проверьте локальное время на сервере (`date`). Константы `LIGHTS_ON`/`LIGHTS_OFF` в `analyzer.py`. |
| Камера регулярно reconnect'ится | Network/CPU перегружен. Уменьшите fps RTSP-стрима на самой камере. |
| Стол 11 всё равно плохо детектится | Закройте `require_movement: false` уже стоит. Попробуйте ещё снизить `window_start_ratio` до 0.40. Если не помогает — это потолок без GPU, физически больше выжать сложно. |
| Все столы внезапно перестали детектиться | Калибровочная INT8 модель просела. Удалите `models/yolo11s_int8_openvino_model/`, analyzer.py упадёт на FP32 (`yolo11s_openvino_model/`). Перезапустите. |
