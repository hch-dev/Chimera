// Scanner Page Specific JavaScript

// CONFIGURATION
const API_URL = "http://127.0.0.1:8000/scan"; // Your FastAPI Backend

document.addEventListener('DOMContentLoaded', () => {
    // Check for URL from homepage session
    const urlFromSession = sessionStorage.getItem('scanUrl');
    if (urlFromSession) {
        document.getElementById('urlInput').value = urlFromSession;
        sessionStorage.removeItem('scanUrl');
        setTimeout(() => performScan(), 500);
    }

    // Setup event listeners
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
    const urlInput = document.getElementById('urlInput');
    const url = urlInput.value.trim();

    if (!url) {
        showNotification('Please enter a URL to scan', 'warning');
        return;
    }

    // Basic Validation
    if (!isValidURL(url)) {
        showNotification('Please enter a valid URL (e.g., https://example.com)', 'error');
        return;
    }

    // UI: Switch to Loading State
    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    // Progress Bar Animation (Visual feedback)
    const progressFill = document.querySelector('.progress-fill');
    progressFill.style.width = '0%';

    let progress = 0;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }
    }, 200);

    try {
        // --- REAL BACKEND CALL ---
        console.log("🚀 Sending request to Chimera Backend:", url);

        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);

        const backendData = await response.json();
        console.log("✅ Backend Response:", backendData);

        // Finish Animation
        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        // Process and Show Results
        setTimeout(() => {
            const uiData = mapBackendToUI(backendData, url);
            saveScanToHistory(uiData); // Save for Reports page
            showResults(uiData);
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        console.error("Scan Failed:", error);

        // Revert UI on error
        document.querySelector('.url-input-section').style.display = 'block';
        document.getElementById('loadingSection').style.display = 'none';

        showNotification('Error connecting to Chimera Engine. Is the backend running?', 'error');
    }
}

// ADAPTER: Convert Python JSON -> Frontend UI Object
function mapBackendToUI(data, originalUrl) {
    // Backend returns: { verdict: "PHISHING", confidence: 95.0, details: {...} }

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

    // Extract key details for display
    const details = data.details || {};
    const sslMsg = details.ssl_presence_and_validity?.message || 'Valid';
    const ageMsg = details.domain_age_analysis?.message || 'Established';

    // Find the "Smoking Gun" (highest risk feature)
    let topThreat = "None detected";
    if (isPhishing || isSuspicious) {
        for (const key in details) {
            if (details[key].score > 50) {
                topThreat = details[key].message;
                break;
            }
        }
    }

    return {
        url: originalUrl,
        domain: new URL(originalUrl).hostname,
        riskScore: Math.round(data.confidence), // Ensure integer
        riskLevel: riskLevel,
        riskClass: riskClass,
        confidence: Math.round(data.confidence),

        // Map specific features from backend details
        ssl: sslMsg,
        domainAge: ageMsg,
        suspiciousPatterns: topThreat,
        maliciousKeywords: details.obfuscation_analysis?.message || 'None',
        redirectBehavior: details.open_redirect_detection?.message || 'Clean',
        blacklistStatus: details.threat_intelligence?.message || 'Clean'
    };
}

// --- UI UPDATE FUNCTIONS ---

function showResults(results) {
    updateRiskDisplay(results);
    updateAnalysisDetails(results);
    updateRecommendations(results);
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function updateRiskDisplay(results) {
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.className = `risk-badge ${results.riskClass}`;
    document.getElementById('riskLevel').textContent = results.riskLevel;

    // Badge Icon
    const iconClass = results.riskClass === 'safe' ? 'fa-shield-check' :
    results.riskClass === 'warning' ? 'fa-exclamation-triangle' :
    'fa-shield-virus';
    riskBadge.querySelector('i').className = `fas ${iconClass}`;

    // Score Circle
    document.getElementById('scoreValue').textContent = results.riskScore;
    document.getElementById('threatLevel').textContent = results.riskLevel;
    document.getElementById('confidenceLevel').textContent = results.confidence + '%';

    // Color the circle ring
    const circle = document.querySelector('.score-circle');
    if (results.riskClass === 'danger') {
        circle.style.background = 'linear-gradient(135deg, #dc3545, #c82333)';
    } else if (results.riskClass === 'warning') {
        circle.style.background = 'linear-gradient(135deg, #ffc107, #fd7e14)';
    } else {
        circle.style.background = 'var(--gradient-primary)';
    }
}

function updateAnalysisDetails(results) {
    const format = (text, elementId) => {
        const el = document.getElementById(elementId);
        el.textContent = text || 'Not Available';
        if (text && (text.includes('detected') || text.includes('MISMATCH') || text.includes('CRITICAL') || text.includes('invalid'))) {
            el.style.color = 'var(--danger-color)';
        } else {
            el.style.color = 'var(--text-secondary)';
        }
    };

    format(results.ssl, 'sslStatus');
    format(results.domainAge, 'domainAge');
    format(results.suspiciousPatterns, 'suspiciousPatterns');
    format(results.maliciousKeywords, 'maliciousKeywords');
    format(results.redirectBehavior, 'redirectBehavior');
    format(results.blacklistStatus, 'blacklistStatus');

    document.querySelectorAll('.status-indicator').forEach(dot => {
        dot.className = `status-indicator ${results.riskClass}`;
    });
}

function updateRecommendations(results) {
    const list = document.getElementById('recommendationsList');
    let items = [];

    if (results.riskClass === 'safe') {
        items = [
            { type: 'safe', icon: 'fa-check-circle', text: 'This URL appears safe based on our heuristic analysis.' },
            { type: 'info', icon: 'fa-lock', text: 'SSL Certificate is valid and domain is established.' }
        ];
    } else if (results.riskClass === 'warning') {
        items = [
            { type: 'warning', icon: 'fa-exclamation-triangle', text: 'Proceed with caution. Suspicious patterns detected.' },
            { type: 'info', icon: 'fa-search', text: 'Check the URL carefully for typos or hidden redirects.' }
        ];
    } else {
        items = [
            { type: 'danger', icon: 'fa-ban', text: 'CRITICAL THREAT: Do not visit this link.' },
            { type: 'danger', icon: 'fa-user-secret', text: 'High confidence of Phishing or Malware.' },
            { type: 'warning', icon: 'fa-flag', text: 'We recommend reporting this URL immediately.' }
        ];
    }

    list.innerHTML = items.map(item => `
    <div class="recommendation-item ${item.type}">
    <i class="fas ${item.icon}"></i>
    <span>${item.text}</span>
    </div>
    `).join('');
}

// Save scan to LocalStorage
function saveScanToHistory(analysis) {
    const historyItem = {
        id: Date.now(),
        date: new Date().toISOString(),
        type: 'URL',
        target: analysis.url,
        riskLevel: analysis.riskLevel,
        riskClass: analysis.riskClass,
        score: analysis.riskScore,
        details: `${analysis.suspiciousPatterns} | ${analysis.ssl}`
    };

    const existing = JSON.parse(localStorage.getItem('chimera_scans') || '[]');
    existing.unshift(historyItem);
    localStorage.setItem('chimera_scans', JSON.stringify(existing.slice(0, 50)));
}

function scanAnother() {
    document.getElementById('urlInput').value = '';
    document.querySelector('.url-input-section').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function reportSuspicious() {
    const url = document.getElementById('urlInput').value;
    sessionStorage.setItem('reportUrl', url);
    window.location.href = 'report.html';
}

function exportReport() {
    showNotification('Export feature coming soon for live scans!', 'info');
}

// Utility
function isValidURL(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}
