import os
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "stats.db")
ZONES_FILE = os.path.join(DATA_DIR, "zones.json")

os.makedirs(DATA_DIR, exist_ok=True)

CAM1_USER = quote(os.getenv("CAM1_USER", ""), safe='')
CAM1_PASS = quote(os.getenv("CAM1_PASS", ""), safe='')
CAM2_USER = quote(os.getenv("CAM2_USER", ""), safe='')
CAM2_PASS = quote(os.getenv("CAM2_PASS", ""), safe='')

CAMERAS = [
    {
        "id": 1,
        "name": "Камера 1",
        "rtsp": f"rtsp://{CAM1_USER}:{CAM1_PASS}@93.91.163.5:8554/Streaming/Channels/101",
        "tables": [1, 2, 3, 4, 5, 6]
    },
    {
        "id": 2,
        "name": "Камера 2",
        "rtsp": f"rtsp://{CAM2_USER}:{CAM2_PASS}@93.91.163.5:554/profile1",
        "tables": [7, 8, 9, 10, 11, 12]
    }
]

# Кадры
FRAME_INTERVAL = 1.0

# Tracking
STABLE_DURATION = 5            # уменьшили с 20 до 8 сек
TRACK_BUFFER = 30

# Скоринг
SCORE_THRESHOLD = 4               # уменьшили с 6 до 4
CONFIRMATION_SECONDS = 3
ABSENCE_TOLERANCE = 60
MIN_GAME_DURATION = 300

# Движение
MOVEMENT_HISTORY = 5
MOVEMENT_THRESHOLD_HIGH = 60      # уменьшили с 80 до 60

# Детекция
DETECTION_CONFIDENCE = 0.15       # уменьшили с 0.35 до 0.25

# Зона вокруг стола
TABLE_EXPAND_PX = 250             # увеличили с 60 до 250 — игроки стоят далеко