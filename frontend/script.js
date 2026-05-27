const datePicker = document.getElementById('datePicker');
const todayBtn = document.getElementById('todayBtn');
const refreshBtn = document.getElementById('refreshBtn');
const updatedAt = document.getElementById('updatedAt');

const todayStr = new Date().toLocaleDateString('en-CA'); // даёт YYYY-MM-DD по локальному времени
datePicker.value = todayStr;

let liveData = {};
let hoursChart = null;
let weekChart = null;

const DAYS = ['Пн','Вт','Ср','Чт','Пт','Сб','Вс'];

function isToday(dateStr) {
    return !dateStr || dateStr === new Date().toISOString().split('T')[0];
}

function fmtHours(seconds) {
    const h = seconds / 3600;
    if (h < 0.1) return '0';
    if (h < 10) return h.toFixed(1);
    return Math.round(h).toString();
}

async function fetchJson(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(r.status);
        return await r.json();
    } catch (e) {
        console.error(`Fetch failed: ${url}`, e);
        return null;
    }
}

async function loadAll(dateStr) {
    const today = isToday(dateStr);
    const todayUrl = dateStr ? `/api/stats/date/${dateStr}` : '/api/stats/today';

    // Параллельно грузим всё что нужно
    const [todayData, weekData, hourlyData, weeklyData, live] = await Promise.all([
        fetchJson(todayUrl),
        fetchJson('/api/stats/week'),
        fetchJson('/api/stats/hourly' + (dateStr ? `?date=${dateStr}` : '')),
        fetchJson('/api/stats/weekly'),
        today ? fetchJson('/api/stats/live') : Promise.resolve({}),
    ]);

    liveData = live || {};

    if (todayData) {
        // Прикручиваем live к today
        if (today) {
            todayData.forEach(t => {
                const s = liveData[t.table_id];
                if (s !== undefined) {
                    t.is_live = true;
                    t.live_seconds = s;
                }
            });
        }

        renderKPIs(todayData, weekData, today);
        renderTopTables(todayData);
        renderTablesList(todayData);
    }

    if (hourlyData) renderHoursChart(hourlyData);
    if (weeklyData) renderWeekChart(weeklyData);

    const displayDate = new Date(dateStr || new Date());
    displayDate.setHours(12, 0, 0, 0);
    const formattedDate = displayDate.toLocaleDateString('ru-RU', {day: 'numeric', month: '2-digit', year: 'numeric'});
    updatedAt.textContent = `обновлено ${formattedDate} ${new Date().toLocaleTimeString('ru-RU').slice(0, 5)} · автообновление каждые 60 сек`;
}

function renderKPIs(today, week, isCurrentDay) {
    const historySec = today.reduce((s,t) => s + (t.total_seconds || 0), 0);
    const liveSec = isCurrentDay ? today.reduce((s,t) => s + (t.live_seconds || 0), 0) : 0;
    const totalSec = historySec + liveSec;

    document.getElementById('kpiHoursToday').innerHTML = `${fmtHours(totalSec)} <span>ч</span>`;

    const busy = today.filter(t => t.is_live).length;
    const total = today.length;
    document.getElementById('kpiBusy').innerHTML = `${busy} <span>/ ${total}</span>`;
    const pct = total > 0 ? Math.round(busy / total * 100) : 0;
    document.getElementById('kpiBusySub').textContent = `${pct}% загрузка`;

    if (week) {
        const weekSec = week.reduce((s,d) => s + (d.total_seconds || 0), 0);
        document.getElementById('kpiWeek').innerHTML = `${fmtHours(weekSec)} <span>ч</span>`;
    }

    // Среднее время сессии
    const totalSessions = today.reduce((s,t) => s + (t.sessions || 0), 0);
    const avgMin = totalSessions > 0 ? Math.round(historySec / totalSessions / 60) : 0;
    document.getElementById('kpiAvg').innerHTML = `${avgMin} <span>мин</span>`;
}

function renderTopTables(data) {
    const sorted = [...data]
        .map(t => ({...t, hours: (t.total_seconds || 0) + (t.live_seconds || 0)}))
        .sort((a,b) => b.hours - a.hours)
        .slice(0, 3);

    const grid = document.getElementById('topGrid');
    grid.innerHTML = sorted.map((t, i) => `
        <div class="top-item">
            <div class="top-rank">№${i+1}</div>
            <div class="top-name">Стол ${t.table_id}</div>
            <div class="top-stats">${fmtHours(t.hours)} ч сегодня · сессий: ${t.sessions || 0}</div>
        </div>
    `).join('');
}

function renderTablesList(data) {
    const body = document.getElementById('tablesBody');
    body.innerHTML = data.map(t => {
        const todaySec = (t.total_seconds || 0) + (t.live_seconds || 0);
        const isBusy = t.is_live;
        return `
            <tr>
                <td><strong>Стол ${t.table_id}</strong></td>
                <td>${fmtHours(todaySec)} ч</td>
                <td>${fmtHours((t.week_seconds || 0))} ч</td>
                <td>${t.sessions || 0}</td>
                <td><span class="status-pill ${isBusy ? 'busy' : 'free'}">${isBusy ? 'Занят' : 'Свободен'}</span></td>
            </tr>
        `;
    }).join('');
}

function renderHoursChart(data) {
    const labels = data.map(d => d.hour.toString().padStart(2,'0'));
    const values = data.map(d => Math.round((d.total_seconds || 0) / 3600 * 10) / 10);

    const ctx = document.getElementById('hoursChart');
    if (hoursChart) { hoursChart.destroy(); }
    hoursChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: '#2563eb',
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, ticks: { stepSize: 1 } }
            }
        }
    });
}

function renderWeekChart(data) {
    const labels = data.map(d => DAYS[d.weekday] || d.weekday);
    const values = data.map(d => Math.round((d.total_seconds || 0) / 3600));

    const ctx = document.getElementById('weekChart');
    if (weekChart) { weekChart.destroy(); }
    weekChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34,197,94,0.15)',
                fill: true,
                tension: 0.35,
                pointRadius: 4,
                pointBackgroundColor: '#22c55e',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true }
            }
        }
    });
}

// Обновление снимков с камер каждые 2 секунды (это видеопоток "псевдо-live")
function refreshCameras() {
    const t = Date.now();
    document.getElementById('cam1').src = `/api/cameras/1/snapshot?t=${t}`;
    document.getElementById('cam2').src = `/api/cameras/2/snapshot?t=${t}`;
}
setInterval(refreshCameras, 2000);

// === События ===
todayBtn.addEventListener('click', () => {
    datePicker.value = new Date().toLocaleDateString('en-CA');
    loadAll();
});
refreshBtn.addEventListener('click', () => loadAll(datePicker.value));
datePicker.addEventListener('change', e => loadAll(e.target.value));

// Автообновление данных каждые 60 секунд
setInterval(() => {
    if (isToday(datePicker.value)) {
        loadAll(datePicker.value);
    }
}, 60000);

loadAll();

(function setupSmoothCameraRefresh() {
    const CAMERA_REFRESH_MS = 2000;

    function refreshCamera(id) {
        const img = document.getElementById(id);
        if (!img) return;

        // Создаём скрытое изображение, грузим в него новый кадр.
        // Когда загрузится — подменяем видимое одним движением через fade.
        const next = new Image();
        const camNum = id.replace('cam', '');
        const url = `/api/cameras/${camNum}/snapshot?t=${Date.now()}`;

        next.onload = () => {
            img.style.transition = 'opacity 0.25s ease';
            img.style.opacity = '0.6';
            setTimeout(() => {
                img.src = next.src;
                img.style.opacity = '1';
            }, 120);
        };
        next.onerror = () => { /* RTSP моргнул — просто пропустим тик */ };
        next.src = url;
    }

    function loop() {
        refreshCamera('cam1');
        refreshCamera('cam2');
    }

    // Запускаем сразу и потом каждые N мс
    loop();
    setInterval(loop, CAMERA_REFRESH_MS);
})();
