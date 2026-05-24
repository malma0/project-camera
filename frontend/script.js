const datePicker = document.getElementById('datePicker');
const todayBtn = document.getElementById('todayBtn');
const refreshBtn = document.getElementById('refreshBtn');
const tablesContainer = document.getElementById('tables');
const summaryContainer = document.getElementById('summary');
const loadingEl = document.getElementById('loading');

datePicker.valueAsDate = new Date();

let liveData = {};

async function fetchLive() {
    try {
        const r = await fetch('/api/stats/live');
        liveData = await r.json();
    } catch(e) {}
}

function isToday(dateStr) {
    return !dateStr || dateStr === new Date().toISOString().split('T')[0];
}

async function loadStats(dateStr = null) {
    loadingEl.style.display = 'block';
    tablesContainer.innerHTML = '';
    summaryContainer.innerHTML = '';

    const url = dateStr ? `/api/stats/date/${dateStr}` : '/api/stats/today';
    const today = isToday(dateStr);

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (today) {
            await fetchLive();

            data.forEach(t => {
                const liveSec = liveData[t.table_id];
                if (liveSec !== undefined) {
                    t.is_live = true;
                    t.live_seconds = liveSec;
                    // Показываем время live сессии ОТДЕЛЬНО — не добавляем к total_seconds!
                    const h = Math.floor(liveSec / 3600);
                    const m = Math.floor((liveSec % 3600) / 60);
                    const s = liveSec % 60;
                    t.live_formatted = h > 0 ? `${h}ч ${m}мин` : `${m}мин ${s}сек`;
                }
            });
        }

        renderSummary(data, today);
        renderTables(data, today);
    } catch (err) {
        tablesContainer.innerHTML = `<p style="color:#ef4444">Ошибка: ${err.message}</p>`;
    } finally {
        loadingEl.style.display = 'none';
    }
}

function renderSummary(data, today) {
    const historySec = data.reduce((s, t) => s + t.total_seconds, 0);
    const liveSec = today ? data.reduce((s, t) => s + (t.live_seconds || 0), 0) : 0;
    const totalSec = historySec + liveSec;

    const activeTables = data.filter(t => t.total_seconds > 0 || t.is_live).length;
    const liveCount = data.filter(t => t.is_live).length;
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);

    summaryContainer.innerHTML = `
        <div class="summary-card">
            <div class="label">Всего часов</div>
            <div class="value">${h}<span>ч ${m}мин</span></div>
        </div>
        <div class="summary-card">
            <div class="label">Активных столов</div>
            <div class="value">${activeTables}<span>из ${data.length}</span></div>
        </div>
        <div class="summary-card">
            <div class="label">Играют сейчас</div>
            <div class="value" style="color: #22c55e">${liveCount}</div>
        </div>
    `;
}

function renderTables(data, today) {
    tablesContainer.innerHTML = data.map(t => {
        // Если стол сейчас активен — показываем время текущей сессии
        // Если нет — показываем итоговое историческое время за день
        const displayTime = t.is_live ? t.live_formatted : t.formatted;
        const isEmpty = t.total_seconds === 0 && !t.is_live;

        return `
            <div class="table-card ${isEmpty ? 'empty' : ''} ${t.is_live ? 'live' : ''}">
                ${t.is_live ? '<div class="live-badge">● LIVE</div>' : ''}
                <div class="table-num">Стол №${t.table_id}</div>
                <div class="table-time">${isEmpty ? '—' : displayTime}</div>
                <div class="table-sessions">
                    ${isEmpty ? 'Нет активности' : `Сессий: ${t.sessions}`}
                </div>
            </div>
        `;
    }).join('');
}

todayBtn.addEventListener('click', () => {
    datePicker.valueAsDate = new Date();
    loadStats();
});

refreshBtn.addEventListener('click', () => loadStats(datePicker.value));
datePicker.addEventListener('change', e => loadStats(e.target.value));

// Обновляем каждые 5 сек только если смотрим сегодня
setInterval(() => {
    if (isToday(datePicker.value)) {
        loadStats(datePicker.value);
    }
}, 5000);

loadStats();
