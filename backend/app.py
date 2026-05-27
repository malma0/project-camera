from flask import Flask, jsonify, request, send_from_directory, redirect, url_for, render_template_string
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import date
import sys, os, json, time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_stats_for_date, get_stats_for_range, init_db
from config import CAMERAS

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-change-this")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"

DASHBOARD_USER = os.getenv("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.getenv("DASHBOARD_PASS", "admin")


class User(UserMixin):
    def __init__(self, id):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


LOGIN_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Вход — Аналитика столов</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .top-bar {
            background: #2563eb;
            height: 64px;
            display: flex;
            align-items: center;
            padding: 0 32px;
            box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        }
        .top-bar h1 { color: #fff; font-size: 17px; font-weight: 600; }
        .body {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px 16px;
        }
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 380px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }
        h2 { font-size: 20px; color: #0f172a; margin-bottom: 6px; }
        .subtitle { color: #64748b; font-size: 14px; margin-bottom: 28px; }
        label { display: block; font-size: 13px; font-weight: 500; color: #374151; margin-bottom: 6px; }
        input {
            width: 100%;
            padding: 10px 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            color: #0f172a;
            font-size: 15px;
            margin-bottom: 16px;
            outline: none;
            transition: border-color 0.15s;
        }
        input:focus { border-color: #2563eb; background: #fff; }
        button {
            width: 100%;
            padding: 11px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            margin-top: 4px;
            transition: background 0.15s;
        }
        button:hover { background: #1d4ed8; }
        .error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <div class="top-bar">
        <h1>Аналитика игровых столов</h1>
    </div>
    <div class="body">
        <div class="card">
            <h2>Вход в систему</h2>
            <p class="subtitle">Введите данные для доступа к аналитике</p>
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            <form method="POST">
                <label>Логин</label>
                <input type="text" name="username" autofocus autocomplete="username">
                <label>Пароль</label>
                <input type="password" name="password" autocomplete="current-password">
                <button type="submit">Войти</button>
            </form>
        </div>
    </div>
</body>
</html>
"""


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == DASHBOARD_USER and password == DASHBOARD_PASS:
            login_user(User(username), remember=True)
            return redirect(url_for('index'))
        else:
            error = "Неверный логин или пароль"
    return render_template_string(LOGIN_HTML, error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login_page'))


@app.route('/')
@login_required
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/stats/today')
@login_required
def stats_today():
    import sqlite3
    stats = get_stats_for_date(date.today())
    db_path = os.path.join(DATA_DIR, 'stats.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT table_id,
               SUM(CAST(strftime('%s', end_time) AS INTEGER) - CAST(strftime('%s', start_time) AS INTEGER)) as total
        FROM sessions
        WHERE start_time >= date('now', '-7 days')
        GROUP BY table_id
    """)
    week_raw = [{'table_id': r[0], 'total_seconds': r[1] or 0} for r in c.fetchall()]
    conn.close()
    return jsonify(format_stats(stats, week_raw))


@app.route('/api/stats/date/<date_str>')
@login_required
def stats_by_date(date_str):
    import sqlite3
    target = date.fromisoformat(date_str)
    stats = get_stats_for_date(target)
    db_path = os.path.join(DATA_DIR, 'stats.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT table_id,
               SUM(CAST(strftime('%s', end_time) AS INTEGER) - CAST(strftime('%s', start_time) AS INTEGER)) as total
        FROM sessions
        WHERE start_time >= date(?, '-7 days') AND start_time <= date(?, '+1 days')
        GROUP BY table_id
    """, (date_str, date_str))
    week_raw = [{'table_id': r[0], 'total_seconds': r[1] or 0} for r in c.fetchall()]
    conn.close()
    return jsonify(format_stats(stats, week_raw))


@app.route('/api/stats/range')
@login_required
def stats_range():
    date_from = date.fromisoformat(request.args.get('from'))
    date_to = date.fromisoformat(request.args.get('to'))
    stats = get_stats_for_range(date_from, date_to)
    return jsonify(format_stats(stats))


@app.route('/api/stats/live')
@login_required
def stats_live():
    # Возвращает только текущие активные сессии (сколько секунд идёт игра прямо сейчас).
    # Фронт отображает это ОТДЕЛЬНО от исторических данных — не складывает вместе.
    active_file = os.path.join(DATA_DIR, "active.json")
    now = time.time()
    result = {}
    try:
        with open(active_file, 'r') as f:
            active = json.load(f)
        for k, v in active.items():
            start_ts = v['start'] if isinstance(v, dict) else v
            duration = int(now - start_ts)
            if duration > 0:
                result[int(k)] = duration
    except:
        pass
    return jsonify(result)

@app.route('/api/cameras/<int:cam_id>/snapshot')
@login_required
def camera_snapshot(cam_id):
    snapshots_dir = os.path.join(DATA_DIR, 'snapshots')
    filename = f'cam_{cam_id}.jpg'
    path = os.path.join(snapshots_dir, filename)
    if not os.path.exists(path):
        return '', 404
    response = send_from_directory(snapshots_dir, filename)
    response.headers['Cache-Control'] = 'no-store'
    return response


def format_stats(stats, week_stats=None):
    all_tables = []
    for camera in CAMERAS:
        all_tables.extend(camera['tables'])

    stats_dict = {row['table_id']: row for row in stats}
    week_dict = {row['table_id']: row for row in (week_stats or [])}

    result = []
    for table_id in sorted(all_tables):
        if table_id in stats_dict:
            total = stats_dict[table_id]['total_seconds']
            sessions = stats_dict[table_id]['sessions_count']
        else:
            total = 0
            sessions = 0

        week_total = week_dict.get(table_id, {}).get('total_seconds', 0)

        hours = total // 3600
        minutes = (total % 3600) // 60
        result.append({
            "table_id": table_id,
            "hours": hours,
            "minutes": minutes,
            "total_seconds": total,
            "week_seconds": week_total,
            "sessions": sessions,
            "formatted": f"{hours}ч {minutes}мин"
        })

    return result

@app.route('/api/stats/week')
@login_required
def stats_week():
    import sqlite3
    db_path = os.path.join(DATA_DIR, 'stats.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT date(start_time) as d,
               SUM(CAST(strftime('%s', end_time) AS INTEGER) - CAST(strftime('%s', start_time) AS INTEGER)) as total
        FROM sessions
        WHERE start_time >= date('now', '-7 days')
        GROUP BY d ORDER BY d
    """)
    result = [{"date": row[0], "total_seconds": row[1] or 0} for row in c.fetchall()]
    conn.close()
    return jsonify(result)


@app.route('/api/stats/hourly')
@login_required
def stats_hourly():
    import sqlite3
    date_str = request.args.get('date') or date.today().isoformat()
    db_path = os.path.join(DATA_DIR, 'stats.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT CAST(strftime('%H', start_time) AS INTEGER) as h,
               SUM(CAST(strftime('%s', end_time) AS INTEGER) - CAST(strftime('%s', start_time) AS INTEGER)) as total
        FROM sessions
        WHERE date(start_time) = ?
        GROUP BY h
    """, (date_str,))
    by_hour = {row[0]: row[1] for row in c.fetchall()}
    result = [{"hour": h, "total_seconds": by_hour.get(h, 0)} for h in range(8, 23)]
    conn.close()
    return jsonify(result)


@app.route('/api/stats/weekly')
@login_required
def stats_weekly():
    import sqlite3
    db_path = os.path.join(DATA_DIR, 'stats.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT ((CAST(strftime('%w', start_time) AS INTEGER) + 6) % 7) as wd,
               SUM(CAST(strftime('%s', end_time) AS INTEGER) - CAST(strftime('%s', start_time) AS INTEGER)) as total
        FROM sessions
        WHERE start_time >= date('now', '-7 days')
        GROUP BY wd
    """)
    by_day = {row[0]: row[1] for row in c.fetchall()}
    result = [{"weekday": d, "total_seconds": by_day.get(d, 0)} for d in range(7)]
    conn.close()
    return jsonify(result)

EVIDENCE_DIR = os.path.join(DATA_DIR, 'evidence')


@app.route('/evidence')
@login_required
def evidence_page():
    """Страница просмотра evidence-снимков."""
    return send_from_directory(FRONTEND_DIR, 'evidence.html')


@app.route('/api/evidence/days')
@login_required
def evidence_days():
    """Возвращает список доступных дней (отсортированный, новые сверху)."""
    if not os.path.isdir(EVIDENCE_DIR):
        return jsonify([])
    days = []
    for name in os.listdir(EVIDENCE_DIR):
        path = os.path.join(EVIDENCE_DIR, name)
        if not os.path.isdir(path):
            continue
        # валидируем формат YYYY-MM-DD
        try:
            from datetime import datetime as _dt
            _dt.strptime(name, '%Y-%m-%d')
            days.append(name)
        except ValueError:
            continue
    days.sort(reverse=True)
    return jsonify(days)


@app.route('/api/evidence/<date_str>')
@login_required
def evidence_for_day(date_str):
    """
    Возвращает для указанного дня список снимков по камерам:
    {
      "cam1": ["08-00.jpg", "08-15.jpg", ...],
      "cam2": [...]
    }
    """
    # защита от path traversal
    if not all(c.isdigit() or c == '-' for c in date_str) or len(date_str) != 10:
        return jsonify({"error": "bad date"}), 400
    day_dir = os.path.join(EVIDENCE_DIR, date_str)
    if not os.path.isdir(day_dir):
        return jsonify({})
    result = {}
    for cam_name in sorted(os.listdir(day_dir)):
        cam_path = os.path.join(day_dir, cam_name)
        if not os.path.isdir(cam_path):
            continue
        files = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
        result[cam_name] = files
    return jsonify(result)


@app.route('/api/evidence/<date_str>/<cam>/<filename>')
@login_required
def evidence_image(date_str, cam, filename):
    """Отдаёт сам JPEG-файл."""
    # path traversal protection
    if '..' in date_str or '..' in cam or '..' in filename or '/' in filename or '\\' in filename:
        return '', 403
    if not filename.endswith('.jpg'):
        return '', 403
    path = os.path.join(EVIDENCE_DIR, date_str, cam, filename)
    if not os.path.isfile(path):
        return '', 404
    return send_from_directory(os.path.join(EVIDENCE_DIR, date_str, cam), filename)


if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
