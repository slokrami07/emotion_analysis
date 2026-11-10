// ─── DOM refs ────────────────────────────────────────────
const textarea      = document.getElementById('user-text');
const charCurrent   = document.getElementById('char-current');
const analyzeBtn    = document.getElementById('analyze-btn');
const btnLabel      = analyzeBtn.querySelector('.btn-label');
const btnIcon       = analyzeBtn.querySelector('.btn-icon');
const btnLoader     = analyzeBtn.querySelector('.btn-loader');
const errorMsg      = document.getElementById('error-msg');
const errorText     = errorMsg.querySelector('.error-text');
const resultsEl     = document.getElementById('results');
const resultEmoji   = document.getElementById('result-emoji');
const resultLabel   = document.getElementById('result-label');
const confValue     = document.getElementById('confidence-value');
const ringFill      = document.getElementById('ring-fill');
const emotionGlow   = document.getElementById('emotion-glow');
const barsContainer = document.getElementById('bars-container');
const analyzedText  = document.getElementById('analyzed-text-value');
const statusPill    = document.getElementById('server-status');
const statusLabel   = statusPill.querySelector('.status-label');

// ─── Add SVG gradient defs for the ring ─────────────────
(function injectSvgGradient() {
  const svg = document.querySelector('.confidence-ring');
  if (!svg) return;
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <linearGradient id="ring-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#8b5cf6" />
      <stop offset="100%" stop-color="#6366f1" />
    </linearGradient>
  `;
  svg.insertBefore(defs, svg.firstChild);
})();

// Emoji map (mirrors main.py)
const EMOJIS = {
  sadness:  '😢',
  joy:      '😄',
  love:     '❤️',
  anger:    '😠',
  fear:     '😨',
  surprise: '😲',
};

// Glow colors per emotion
const GLOW_COLORS = {
  sadness:  '#60a5fa',
  joy:      '#fbbf24',
  love:     '#f472b6',
  anger:    '#ef4444',
  fear:     '#a78bfa',
  surprise: '#34d399',
};

// ─── Character counter ──────────────────────────────────
textarea.addEventListener('input', () => {
  charCurrent.textContent = textarea.value.length;
});

// ─── Health check ───────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error();
    const data = await res.json();
    const ok = data.status && data.model_laoded;
    statusPill.className = 'status-pill ' + (ok ? 'online' : 'offline');
    statusLabel.textContent = ok ? 'Model ready' : 'Model loading…';
  } catch {
    statusPill.className = 'status-pill offline';
    statusLabel.textContent = 'Server offline';
  }
}

checkHealth();
setInterval(checkHealth, 15000);

// ─── UI helpers ─────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  btnLabel.hidden     = on;
  btnIcon.hidden      = on;
  btnLoader.hidden    = !on;
  textarea.disabled   = on;
}

function showError(msg) {
  errorText.textContent = msg;
  errorMsg.hidden = false;
  // re-trigger shake
  errorMsg.style.animation = 'none';
  void errorMsg.offsetHeight;
  errorMsg.style.animation = '';
  resultsEl.hidden = true;
}

function hideError() {
  errorMsg.hidden = true;
}

// ─── Confidence ring ────────────────────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 52; // r=52

function setRing(pct) {
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
  ringFill.style.strokeDashoffset = offset;
}

// ─── Render probability bars ────────────────────────────
function renderBars(probs, topEmotion) {
  barsContainer.innerHTML = '';

  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([emotion, prob], i) => {
    const pct = (prob * 100).toFixed(1);
    const isTop = emotion === topEmotion;

    const row = document.createElement('div');
    row.className = 'bar-row' + (isTop ? ' top' : '');
    row.style.setProperty('--delay', `${i * .06}s`);

    row.innerHTML = `
      <span class="bar-label">
        <span class="bar-label-emoji">${EMOJIS[emotion] || ''}</span>
        ${emotion}
      </span>
      <div class="bar-track">
        <div class="bar-fill" data-emotion="${emotion}" style="width: 0%"></div>
      </div>
      <span class="bar-value">${pct}%</span>
    `;

    barsContainer.appendChild(row);

    // Animate bar fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector('.bar-fill').style.width = pct + '%';
      });
    });
  });
}

// ─── Analyze (POST /predict) ────────────────────────────
async function analyze() {
  hideError();

  const text = textarea.value.trim();
  if (!text) {
    showError('Please enter a sentence to analyze.');
    return;
  }

  setLoading(true);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    const emotion   = data.predicted_emotion;
    const confPct   = (data.confidence * 100).toFixed(1);

    // Emoji + label
    resultEmoji.textContent = EMOJIS[emotion] || '🤔';
    // Re-trigger pop animation
    resultEmoji.style.animation = 'none';
    void resultEmoji.offsetHeight;
    resultEmoji.style.animation = '';

    resultLabel.textContent = emotion;

    // Confidence ring
    ringFill.style.strokeDashoffset = CIRCUMFERENCE; // reset
    confValue.textContent = confPct + '%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setRing(parseFloat(confPct)));
    });

    // Glow
    emotionGlow.style.background = GLOW_COLORS[emotion] || '#8b5cf6';

    // Bars
    renderBars(data.all_probabilities, emotion);

    // Analyzed text
    analyzedText.textContent = `"${data.text}"`;

    // Show results, re-trigger stagger animations
    resultsEl.hidden = false;
    resultsEl.querySelectorAll('.glass-card').forEach(card => {
      card.style.animation = 'none';
      void card.offsetHeight;
      card.style.animation = '';
    });

  } catch (err) {
    showError(err.message || 'Something went wrong. Please try again.');
  } finally {
    setLoading(false);
  }
}

// ─── Events ─────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyze);

textarea.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    analyze();
  }
});

// Auto-focus textarea on load
window.addEventListener('DOMContentLoaded', () => textarea.focus());
