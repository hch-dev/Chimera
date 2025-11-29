// Scanner Page Logic - Final Logic Fix

const API_URL = "http://127.0.0.1:8000/scan";

document.addEventListener('DOMContentLoaded', () => {
    // 1. Handle URL passed from Home Page
    const sessionUrl = sessionStorage.getItem('scanUrl');
    if (sessionUrl) {
        document.getElementById('urlInput').value = sessionUrl;
        sessionStorage.removeItem('scanUrl');
        setTimeout(performScan, 500);
    }

    // 2. Event Listeners
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

    // UI: Loading State
    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'flex';
    document.getElementById('resultsSection').style.display = 'none';

    // Animation Bar
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

        // Render Results
        setTimeout(() => {
            renderResults(data, url);
            document.getElementById('loadingSection').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'block';
        }, 500);

    } catch (err) {
        clearInterval(timer);
        console.error(err);
        document.querySelector('.url-input-section').style.display = 'block';
        document.getElementById('loadingSection').style.display = 'none';

        if(typeof showNotification === 'function') {
            showNotification('Backend not reachable. Is server.py running?', 'error');
        } else {
            alert('Backend connection failed. Is Python running?');
        }
    }
}

// FORMATTER: Cleans text (e.g. "no_ssl_present" -> "No SSL Present")
function formatText(text) {
    if (!text) return "Unknown";
    return text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    .replace(/Ssl/g, "SSL").replace(/Url/g, "URL");
}

function renderResults(data, url) {
    // 1. Calculate Scores
    // Backend gives "final_score" (Risk: 0=Safe, 100=Bad)
    // We display "Safety Score" (100 - Risk)
    const riskScore = Math.round(data.confidence || data.final_score || 0);
    const safetyScore = Math.max(0, 100 - riskScore);

    // 2. Determine Verdict Category
    let riskLevel = 'Safe';
    let riskClass = 'safe';

    if (riskScore >= 80) {
        riskLevel = 'Critical';
        riskClass = 'danger';
    } else if (riskScore >= 40) {
        riskLevel = 'Suspicious';
        riskClass = 'warning';
    }

    // 3. Update Header Badge
    const badge = document.getElementById('riskBadge');
    badge.className = `risk-badge ${riskClass}`;
    // Icon Logic
    let icon = 'fa-check-circle';
    if (riskClass === 'warning') icon = 'fa-exclamation-triangle';
    if (riskClass === 'danger') icon = 'fa-radiation'; // Distinct icon for critical

    badge.innerHTML = `<i class="fas ${icon}"></i> ${riskLevel}`;

    // 4. Update Circle Chart
    document.getElementById('scoreValue').textContent = safetyScore;
    document.getElementById('threatLevel').textContent = riskLevel;

    const circle = document.querySelector('.score-circle');
    let color = '#28a745'; // Green
    if (riskClass === 'warning') color = '#ffc107'; // Yellow
    if (riskClass === 'danger') color = '#dc3545'; // Red

    // Conic Gradient for Donut Chart
    circle.style.background = `conic-gradient(${color} ${safetyScore}%, var(--bg-tertiary) 0)`;

    // 5. Populate Cards (All 13 Features)
    const d = data.details || {};
    const isSiteBad = riskClass !== 'safe';

    // Map ALL features
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

    // 6. Recommendations
    updateRecommendations(riskClass);
}

// SMART CARD LOGIC:
// - If Site is BAD: Show ONLY the cards that are Red/Warning (Score > 0). Hide the Green ones.
// - If Site is SAFE: Show ONLY the Green cards (Score == 0) to reassure user.
function updateCard(cardId, textId, feature, isSiteBad) {
    const card = document.getElementById(cardId);
    const p = document.getElementById(textId);

    // Safety check if elements exist
    if (!card || !p) return;

    // Handle missing data from backend gracefully
    if (!feature) {
        card.classList.add('d-none'); // Hide if no data
        return;
    }

    const msg = formatText(feature.message);
    p.textContent = msg;

    // Is this specific feature indicating a threat?
    // In your backend, score > 0 means SOME risk was found.
    const isFeatureBad = feature.score > 0;

    // Text Color
    p.className = isFeatureBad ? 'text-danger' : 'text-success';

    // VISIBILITY RULE
    if (isSiteBad) {
        // Site is Phishing/Suspicious
        // SHOW: Only features that flagged a problem (Red cards)
        if (isFeatureBad) {
            card.classList.remove('d-none');
        } else {
            card.classList.add('d-none');
        }
    } else {
        // Site is Safe
        // SHOW: Only features that passed successfully (Green cards)
        if (!isFeatureBad) {
            card.classList.remove('d-none');
        } else {
            card.classList.add('d-none');
        }
    }
}

function updateRecommendations(riskClass) {
    const list = document.getElementById('recommendationsList');
    if (riskClass === 'danger') {
        list.innerHTML = `
        <div class="recommendation-item danger">
        <i class="fas fa-ban"></i>
        <span><b>CRITICAL THREAT:</b> Do not visit this link. It is likely a phishing attack.</span>
        </div>
        <div class="recommendation-item warning">
        <i class="fas fa-key"></i>
        <span>If you entered credentials here, change your passwords immediately.</span>
        </div>`;
    } else if (riskClass === 'warning') {
        list.innerHTML = `
        <div class="recommendation-item warning">
        <i class="fas fa-exclamation-triangle"></i>
        <span><b>PROCEED WITH CAUTION:</b> Suspicious patterns detected.</span>
        </div>
        <div class="recommendation-item">
        <i class="fas fa-search"></i>
        <span>Verify the URL spelling carefully before trusting this site.</span>
        </div>`;
    } else {
        list.innerHTML = `
        <div class="recommendation-item safe">
        <i class="fas fa-check-circle"></i>
        <span><b>SAFE TO VISIT:</b> No known threats detected.</span>
        </div>
        <div class="recommendation-item">
        <i class="fas fa-shield-alt"></i>
        <span>Connection is secure and the domain is established.</span>
        </div>`;
    }
}

function scanAnother() {
    document.getElementById('urlInput').value = '';
    document.querySelector('.url-input-section').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function reportSuspicious() {
    window.location.href = "report.html";
}
