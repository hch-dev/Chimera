// 1. Set your Ngrok URL here (Do not include /scan at the end)
const SERVER_URL = "https://adelia-commonsense-soaked.ngrok-free.dev";
// The endpoint we want to hit
const API_URL = `${SERVER_URL}/scan`;

document.addEventListener('DOMContentLoaded', () => {
    const sessionUrl = sessionStorage.getItem('scanUrl');
    if (sessionUrl) {
        document.getElementById('urlInput').value = sessionUrl;
        sessionStorage.removeItem('scanUrl');
        setTimeout(performScan, 500);
    }
    document.getElementById('scanBtn').addEventListener('click', performScan);
    document.getElementById('urlInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performScan();
    });
});

function setExampleUrl(url) {
    document.getElementById('urlInput').value = url;
    document.getElementById('urlInput').focus();
}

async function performScan() {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) return alert('Please enter a URL');

    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'flex';
    document.getElementById('resultsSection').style.display = 'none';

    const bar = document.querySelector('.progress-fill');
    bar.style.width = '10%';
    let w = 10;
    const timer = setInterval(() => { if(w < 90) bar.style.width = (w += 5) + '%'; }, 200);

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url })
        });

        if (!res.ok) throw new Error("Backend Error");
        const data = await res.json();

        clearInterval(timer);
        bar.style.width = '100%';

        setTimeout(() => {
            renderResults(data, url);
            document.getElementById('loadingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'block';
        }, 500);

    } catch (err) {
        clearInterval(timer);
        document.querySelector('.url-input-section').style.display = 'block';
        document.getElementById('loadingSection').style.display = 'none';
        alert('Connection Failed. Is python server.py running?');
    }
}

function formatText(text) {
    if (!text) return "Unknown";
    return text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    .replace(/Ssl/g, "SSL").replace(/Url/g, "URL");
}

function renderResults(data, url) {
    // 1. SCORING LOGIC FIX
    // Backend 'final_score' is Risk (0=Safe, 100=Bad)
    const riskScore = Math.round(data.confidence || data.final_score || 0);
    const safetyScore = Math.max(0, 100 - riskScore);

    // 2. VERDICT LOGIC
    let riskLevel = 'Safe';
    let riskClass = 'safe';

    if (riskScore >= 80) {
        riskLevel = 'Critical';
        riskClass = 'danger';
    } else if (riskScore >= 40) {
        riskLevel = 'Suspicious';
        riskClass = 'warning';
    }

    // 3. HEADER UPDATES
    const badge = document.getElementById('riskBadge');
    badge.className = `risk-badge ${riskClass}`;

    let icon = 'fa-check-circle';
    if(riskClass === 'warning') icon = 'fa-exclamation-triangle';
    if(riskClass === 'danger') icon = 'fa-ban';

    badge.innerHTML = `<i class="fas ${icon}"></i> ${riskLevel}`;

    document.getElementById('scoreValue').textContent = safetyScore;

    const circle = document.querySelector('.score-circle');
    let color = '#28a745';
    if(riskClass === 'warning') color = '#ffc107';
    if(riskClass === 'danger') color = '#dc3545';
    circle.style.background = `conic-gradient(${color} ${safetyScore}%, var(--bg-tertiary) 0)`;

    // 4. CARD FILTERING
    const d = data.details || {};
    const isSiteBad = riskClass !== 'safe';

    updateCard('card-ssl', 'sslStatus', d.ssl_presence_and_validity, isSiteBad);
    updateCard('card-age', 'domainAge', d.domain_age_analysis, isSiteBad);
    updateCard('card-redirects', 'redirectBehavior', d.open_redirect_detection, isSiteBad);
    updateCard('card-blacklist', 'blacklistStatus', d.threat_intelligence, isSiteBad);
    updateCard('card-homoglyph', 'homoglyphStatus', d.homoglyph_impersonation, isSiteBad);
    updateCard('card-favicon', 'faviconStatus', d.favicon_mismatch, isSiteBad);
    updateCard('card-abuse', 'abuseStatus', d.domain_abuse_detection, isSiteBad);
    updateCard('card-obfuscation', 'obfuscationStatus', d.obfuscation_analysis, isSiteBad);
    updateCard('card-flux', 'fluxStatus', d.fast_flux_dns, isSiteBad);
    updateCard('card-datauri', 'dataUriStatus', d.data_uri_scheme, isSiteBad);
    updateCard('card-random', 'randomStatus', d.random_domain_detection, isSiteBad);
    updateCard('card-structure', 'structureStatus', d.url_structure_analysis, isSiteBad);
    updateCard('card-path', 'pathStatus', d.path_anomaly_detection, isSiteBad);

    // 5. RECOMMENDATIONS (In Header)
    const list = document.getElementById('recommendationsList');
    if (riskClass === 'danger') {
        list.innerHTML = `
        <div class="rec-item-compact danger"><i class="fas fa-ban"></i> <span><b>CRITICAL THREAT:</b> Do not visit this website.</span></div>
        <div class="rec-item-compact warning"><i class="fas fa-key"></i> <span>Change passwords immediately if visited.</span></div>`;
    } else if (riskClass === 'warning') {
        list.innerHTML = `
        <div class="rec-item-compact warning"><i class="fas fa-exclamation-triangle"></i> <span><b>CAUTION:</b> Suspicious elements found.</span></div>
        <div class="rec-item-compact"><i class="fas fa-search"></i> <span>Check URL spelling carefully.</span></div>`;
    } else {
        list.innerHTML = `
        <div class="rec-item-compact safe"><i class="fas fa-check-circle"></i> <span><b>SAFE TO VISIT:</b> No threats detected.</span></div>
        <div class="rec-item-compact"><i class="fas fa-shield-alt"></i> <span>Standard security checks passed.</span></div>`;
    }
}

function updateCard(cardId, textId, feature, isSiteBad) {
    const card = document.getElementById(cardId);
    const p = document.getElementById(textId);
    if (!feature || !card) {
        if(card) card.classList.add('d-none'); // Hide if backend data missing
        return;
    }

    const msg = formatText(feature.message);
    p.textContent = msg;
    const isFeatureBad = feature.score > 0;
    p.className = isFeatureBad ? 'text-danger' : 'text-success';

    // SMART FILTER:
    // If Phishing -> Show only RED cards.
    // If Safe -> Show only GREEN cards.
    if (isSiteBad) {
        isFeatureBad ? card.classList.remove('d-none') : card.classList.add('d-none');
    } else {
        !isFeatureBad ? card.classList.remove('d-none') : card.classList.add('d-none');
    }
}

function scanAnother() { window.location.reload(); }
function reportSuspicious() { window.location.href = "report.html"; }
