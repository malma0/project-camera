import sys, os
import random
from datetime import datetime, timedelta

# добавляем корень проекта И папку backend в пути
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'backend'))

from database import init_db, save_session
from config import CAMERAS


def generate_fake_sessions():
    init_db()
    
    all_tables = []
    for cam in CAMERAS:
        all_tables.extend(cam['tables'])
    
    print(f"Генерируем данные для {len(all_tables)} столов за 7 дней...")
    
    for days_ago in range(7):
        target_date = datetime.now() - timedelta(days=days_ago)
        
        for table_id in all_tables:
            sessions_count = random.randint(0, 5)
            
            for _ in range(sessions_count):
                hour = random.randint(10, 21)
                minute = random.randint(0, 59)
                start = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                duration_minutes = random.randint(20, 90)
                end = start + timedelta(minutes=duration_minutes)
                
                save_session(table_id, start, end)
    
    print("✅ Фейковые данные созданы")


if __name__ == "__main__":
    generate_fake_sessions()