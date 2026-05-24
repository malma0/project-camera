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
    stats = get_stats_for_date(date.today())
    return jsonify(format_stats(stats))


@app.route('/api/stats/date/<date_str>')
@login_required
def stats_by_date(date_str):
    target = date.fromisoformat(date_str)
    stats = get_stats_for_date(target)
    return jsonify(format_stats(stats))


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


def format_stats(stats):
    all_tables = []
    for camera in CAMERAS:
        all_tables.extend(camera['tables'])

    stats_dict = {row['table_id']: row for row in stats}

    result = []
    for table_id in sorted(all_tables):
        if table_id in stats_dict:
            total = stats_dict[table_id]['total_seconds']
            sessions = stats_dict[table_id]['sessions_count']
        else:
            total = 0
            sessions = 0

        hours = total // 3600
        minutes = (total % 3600) // 60
        result.append({
            "table_id": table_id,
            "hours": hours,
            "minutes": minutes,
            "total_seconds": total,
            "sessions": sessions,
            "formatted": f"{hours}ч {minutes}мин"
        })

    return result


if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
