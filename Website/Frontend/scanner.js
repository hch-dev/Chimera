const API_URL = "http://127.0.0.1:8000/scan";

document.addEventListener('DOMContentLoaded', () => {
    const urlFromSession = sessionStorage.getItem('scanUrl');
    if (urlFromSession) {
        document.getElementById('urlInput').value = urlFromSession;
        sessionStorage.removeItem('scanUrl');
        setTimeout(() => performScan(), 500);
    }

    const scanBtn = document.getElementById('scanBtn');
    const urlInput = document.getElementById('urlInput');

    if (scanBtn) scanBtn.addEventListener('click', performScan);
    if (urlInput) {
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performScan();
        });
    }
});

function setExampleUrl(url) {
    document.getElementById('urlInput').value = url;
    document.getElementById('urlInput').focus();
}

async function performScan() {
    const urlInput = document.getElementById('urlInput');
    const url = urlInput.value.trim();

    if (!url || !isValidURL(url)) {
        alert('Please enter a valid URL');
        return;
    }

    // UI State: Loading
    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) throw new Error("Server Error");

        const backendData = await response.json();

        // Simulate tiny delay for smoother transition
        setTimeout(() => {
            const uiData = mapBackendToUI(backendData, url);
            showResults(uiData);
            document.getElementById('loadingSection').style.display = 'none';
        }, 800);

    } catch (error) {
        console.error(error);
        document.querySelector('.url-input-section').style.display = 'block';
        document.getElementById('loadingSection').style.display = 'none';
        alert('Backend not reachable. Ensure server.py is running.');
    }
}

// FORMATTER
function formatText(text) {
    if (!text) return "Unknown";
    return text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    .replace(/Ssl/g, "SSL").replace(/Url/g, "URL");
}

function mapBackendToUI(data, originalUrl) {
    const isPhishing = data.verdict === "PHISHING";
    const isSuspicious = data.verdict === "SUSPICIOUS";

    let riskLevel = 'Low';
    let riskClass = 'safe';

    if (isPhishing) {
        riskLevel = 'Critical';
        riskClass = 'danger';
    } else if (isSuspicious) {
        riskLevel = 'Medium';
        riskClass = 'warning';
    }

    const d = data.details || {};
    const getMsg = (key) => d[key]?.message || 'Clean';

    let finalScore = Math.round(data.confidence) || 0;
    let safetyScore = 100 - finalScore;
    if (safetyScore < 0) safetyScore = 0;

    return {
        url: originalUrl,
        riskScore: safetyScore,
        rawRisk: finalScore,
        riskLevel: riskLevel,
        riskClass: riskClass,

        ssl: formatText(getMsg('ssl_presence_and_validity')),
        domainAge: formatText(getMsg('domain_age_analysis')),
        suspiciousPatterns: formatText(getMsg('homoglyph_impersonation')),
        maliciousKeywords: formatText(getMsg('obfuscation_analysis')),
        redirectBehavior: formatText(getMsg('open_redirect_detection')),
        blacklistStatus: formatText(getMsg('threat_intelligence'))
    };
}

function showResults(results) {
    // 1. Risk Badge
    const riskBadge = document.getElementById('riskBadge');
    const icon = results.riskClass === 'safe' ? 'fa-check-circle' : 'fa-exclamation-triangle';
    riskBadge.className = `risk-badge ${results.riskClass}`;
    riskBadge.innerHTML = `<i class="fas ${icon}"></i> ${results.riskLevel} Risk`;

    // 2. Score Circle
    document.getElementById('scoreValue').textContent = results.riskScore;
    document.getElementById('threatLevel').textContent = results.riskLevel;

    // Removed Confidence Level assignment here as requested

    const circle = document.querySelector('.score-circle');
    let color = '#28a745'; // Green
    if (results.riskClass === 'warning') color = '#ffc107';
    if (results.riskClass === 'danger') color = '#dc3545';
    if (circle) circle.style.background = `conic-gradient(${color} ${results.riskScore}%, var(--bg-tertiary) 0)`;

    // 3. Analysis Details (With Filtering Logic)
    updateAnalysisGrid(results);

    // 4. Recommendations
    updateRecommendations(results);

    document.getElementById('resultsSection').style.display = 'block';
}

function updateAnalysisGrid(results) {
    // Helper to determine if a specific text indicates a problem
    const isBad = (text) => {
        const t = text.toLowerCase();
        return t.includes('invalid') || t.includes('missing') || t.includes('detected') ||
        t.includes('found') || t.includes('mismatch') || t.includes('high');
    };

    // Helper to update text and visibility
    const updateCard = (cardId, textId, text) => {
        const card = document.getElementById(cardId);
        const p = document.getElementById(textId);

        if (!card || !p) return;

        p.textContent = text;

        // Color logic
        const bad = isBad(text);
        p.className = bad ? 'text-danger' : 'text-success';

        // VISIBILITY LOGIC (Change #4)
        // If site is SAFE: Show everything (so user sees why it's safe)
        // If site is PHISHING/WARNING: Hide the "Safe" cards, only show the "Bad" cards.
        if (results.riskClass !== 'safe' && !bad) {
            card.style.display = 'none'; // Hide good news on a bad site
        } else {
            card.style.display = 'flex'; // Show otherwise
        }
    };

    updateCard('card-ssl', 'sslStatus', results.ssl);
    updateCard('card-age', 'domainAge', results.domainAge);
    updateCard('card-patterns', 'suspiciousPatterns', results.suspiciousPatterns);
    updateCard('card-keywords', 'maliciousKeywords', results.maliciousKeywords);
    updateCard('card-redirects', 'redirectBehavior', results.redirectBehavior);
    updateCard('card-blacklist', 'blacklistStatus', results.blacklistStatus);
}

function updateRecommendations(results) {
    const list = document.getElementById('recommendationsList');
    let items = [];

    if (results.riskClass === 'safe') {
        items = [
            { type: 'safe', icon: 'fa-check-circle', text: 'This URL appears safe.' },
            { type: 'info', icon: 'fa-lock', text: 'SSL is valid and domain is established.' }
        ];
    } else {
        items = [
            { type: 'danger', icon: 'fa-ban', text: 'CRITICAL THREAT: Do not visit this link.' },
            { type: 'danger', icon: 'fa-user-secret', text: 'Phishing indicators found.' }
        ];
    }

    list.innerHTML = items.map(item => `
    <div class="recommendation-item ${item.type}">
    <i class="fas ${item.icon}"></i> <span>${item.text}</span>
    </div>
    `).join('');
}

function scanAnother() {
    document.querySelector('.url-input-section').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('urlInput').value = '';
}

function reportSuspicious() {
    window.location.href = 'report.html';
}

function saveScanToHistory(data) {} // Implemented in reports page
function isValidURL(str) { try { new URL(str); return true; } catch{ return false; } }
