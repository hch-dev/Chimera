// Scanner Page Logic
const API_URL = "http://127.0.0.1:8000/scan";

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

    // Show Loading
    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'flex';
    document.getElementById('resultsSection').style.display = 'none';

    // Fake Animation
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
        console.error(err);
        document.querySelector('.url-input-section').style.display = 'block';
        document.getElementById('loadingSection').style.display = 'none';

        // Notification Fallback
        if(typeof showNotification === 'function') {
            showNotification('Backend not reachable. Is server.py running?', 'error');
        } else {
            alert('Backend not reachable. Is Python running?');
        }
    }
}

// FORMATTER
function formatText(text) {
    if (!text) return "Unknown";
    return text.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    .replace(/Ssl/g, "SSL").replace(/Url/g, "URL");
}

function renderResults(data, url) {
    // 1. Risk Logic
    const isBad = data.verdict !== "SAFE";
    const riskClass = isBad ? (data.final_score > 80 ? 'danger' : 'warning') : 'safe';
    const riskLabel = isBad ? (data.final_score > 80 ? 'CRITICAL' : 'SUSPICIOUS') : 'SAFE';

    // Safety Score (Invert Risk Score)
    const safetyScore = Math.max(0, 100 - Math.round(data.confidence || data.final_score));

    // 2. Header & Circle
    const badge = document.getElementById('riskBadge');
    badge.className = `risk-badge ${riskClass}`;
    badge.innerHTML = `<i class="fas ${isBad ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i> ${riskLabel}`;

    document.getElementById('scoreValue').textContent = safetyScore;
    document.getElementById('threatLevel').textContent = riskLabel;

    const circle = document.querySelector('.score-circle');
    let color = riskClass === 'safe' ? '#28a745' : (riskClass === 'warning' ? '#ffc107' : '#dc3545');
    circle.style.background = `conic-gradient(${color} ${safetyScore}%, var(--bg-tertiary) 0)`;

    // 3. Update Analysis Grid (With Filtering)
    const d = data.details || {};

    // Pass 'isBad' to the update function to trigger hiding logic
    updateCard('card-ssl', 'sslStatus', d.ssl_presence_and_validity, isBad);
    updateCard('card-age', 'domainAge', d.domain_age_analysis, isBad);
    updateCard('card-redirects', 'redirectBehavior', d.open_redirect_detection, isBad);
    updateCard('card-blacklist', 'blacklistStatus', d.threat_intelligence, isBad);
    updateCard('card-homoglyph', 'homoglyphStatus', d.homoglyph_impersonation, isBad);
    updateCard('card-favicon', 'faviconStatus', d.favicon_mismatch, isBad);
    updateCard('card-abuse', 'abuseStatus', d.domain_abuse_detection, isBad);
    updateCard('card-obfuscation', 'obfuscationStatus', d.obfuscation_analysis, isBad);
    updateCard('card-flux', 'fluxStatus', d.fast_flux_dns, isBad);
    updateCard('card-datauri', 'dataUriStatus', d.data_uri_scheme, isBad);
    updateCard('card-random', 'randomStatus', d.random_domain_detection, isBad);
    updateCard('card-structure', 'structureStatus', d.url_structure_analysis, isBad);
    updateCard('card-path', 'pathStatus', d.path_anomaly_detection, isBad);

    // 4. Recommendations
    const list = document.getElementById('recommendationsList');
    if (isBad) {
        list.innerHTML = `
        <div class="recommendation-item danger">
        <i class="fas fa-ban"></i>
        <span><b>DO NOT VISIT:</b> High probability of phishing detected.</span>
        </div>
        <div class="recommendation-item warning">
        <i class="fas fa-user-shield"></i>
        <span>If you entered data here recently, change your passwords immediately.</span>
        </div>`;
    } else {
        list.innerHTML = `
        <div class="recommendation-item safe">
        <i class="fas fa-check-circle"></i>
        <span>This website appears to be safe for browsing.</span>
        </div>
        <div class="recommendation-item">
        <i class="fas fa-info-circle"></i>
        <span>Always verify the URL before entering sensitive information.</span>
        </div>`;
    }
}

// Logic: Hide Safe Cards if Site is BAD
function updateCard(cardId, textId, feature, isSiteBad) {
    const card = document.getElementById(cardId);
    const p = document.getElementById(textId);

    if (!feature || !card) return;

    const msg = formatText(feature.message);
    p.textContent = msg;

    // Detect if this specific feature flagged an error
    const isFeatureBad = feature.score > 0;

    // Color text
    p.className = isFeatureBad ? 'text-danger' : 'text-success';

    // FILTERING LOGIC:
    // 1. If Site is Phishing (isSiteBad = true):
    //    - Show ONLY features that are bad (isFeatureBad = true).
    //    - Hide everything else.
    // 2. If Site is Safe (isSiteBad = false):
    //    - Show everything (or at least the Green ones) to reassure user.

    if (isSiteBad) {
        if (isFeatureBad) {
            card.classList.remove('d-none'); // Show the Red Flag
        } else {
            card.classList.add('d-none');    // Hide the Green Flag (Irrelevant info for a threat)
        }
    } else {
        // Site is safe -> Show all cards so user sees what was checked
        card.classList.remove('d-none');
    }
}

function scanAnother() {
    window.location.reload();
}

function reportSuspicious() {
    window.location.href = "report.html";
}
