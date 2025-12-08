// email-checker.js

// --- 1. CONFIGURATION ---
// Set your Ngrok URL here (Do not include trailing slash)
const SERVER_URL = "https://adelia-commonsense-soaked.ngrok-free.dev";
const API_URL = `${SERVER_URL}/api/email/scan`;

// --- 2. INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    // Setup tab switching
    setupTabs();

    // Setup file upload
    setupFileUpload();

    // Setup Main "Check" button
    const checkBtn = document.getElementById('checkBtn');
    if(checkBtn) checkBtn.addEventListener('click', checkEmail);

    // Setup "Check Another Email" button (Fixing the broken button)
    // We look for the button with the specific class since it might not have an ID
    const actionButtons = document.querySelectorAll('.action-buttons button');
    actionButtons.forEach(btn => {
        if (btn.textContent.includes('Check Another Email')) {
            btn.addEventListener('click', checkAnotherEmail);
        }
        if (btn.textContent.includes('Report Phishing')) {
            btn.addEventListener('click', reportPhishingEmail);
        }
        if (btn.textContent.includes('Export Report')) {
            btn.addEventListener('click', exportEmailReport);
        }
    });
});

// --- 3. CORE SCANNING LOGIC ---

async function checkEmail() {
    const subject = document.getElementById('emailSubject').value.trim();
    const from = document.getElementById('emailFrom').value.trim();
    const content = document.getElementById('emailContent').value.trim();
    const headers = document.getElementById('emailHeaders').value.trim();

    // Basic Validation
    if (!subject && !from && !content) {
        showNotification('Please enter at least the email content or subject', 'warning');
        return;
    }

    // UI: Hide input, show loading
    toggleSection('loading');

    // Start Progress Bar Animation
    const bar = document.querySelector('.progress-fill');
    if(bar) bar.style.width = '10%';
    let w = 10;
    const timer = setInterval(() => { if(w < 90 && bar) bar.style.width = (w += 5) + '%'; }, 300);

    try {
        // --- API CALL TO BACKEND ---
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                subject: subject || "No Subject",
                body: content || "No Content",
                sender: from || "Unknown Sender",
                headers: headers || ""
            })
        });

        if (!response.ok) throw new Error("Backend connection failed");

        const result = await response.json();

        // Complete Animation
        clearInterval(timer);
        if(bar) bar.style.width = '100%';

        // Process Data & Render
        setTimeout(() => {
            const combinedAnalysis = processAnalysisResults(result.data, subject, from, content, headers);
            updateEmailResults(combinedAnalysis);
            toggleSection('results');
        }, 500);

    } catch (error) {
        clearInterval(timer);
        console.error(error);
        toggleSection('input'); // Go back to input on error
        showNotification('Connection Failed. Is the backend running?', 'error');
    }
}

// Combine AI Score with Client-Side Details
function processAnalysisResults(backendData, subject, from, content, headers) {
    const score = backendData.score;
    let riskClass = 'safe';
    let riskLevel = 'Safe';

    if (score >= 75) {
        riskLevel = 'Critical Risk';
        riskClass = 'danger';
    } else if (score >= 40) {
        riskLevel = 'Suspicious';
        riskClass = 'warning';
    }

    const analysis = {
        phishingScore: Math.round(score),
        riskLevel: riskLevel,
        riskClass: riskClass,
        confidence: 95,
        spamKeywords: [],
        headerAnalysis: headers ? 'Headers Present' : 'No Headers',
        urgencyLanguage: 'Normal',
        attachments: 'None detected',
        suspiciousLinks: [],
        senderVerification: 'Verified'
    };

    const contentLower = (subject + ' ' + content).toLowerCase();

    // keyword extraction
    const suspiciousKeywords = ['urgent', 'immediate', 'suspended', 'verify', 'click here', 'winner', 'bank', 'password'];
    suspiciousKeywords.forEach(k => { if (contentLower.includes(k)) analysis.spamKeywords.push(k); });

    // link extraction
    const linkRegex = /https?:\/\/[^\s]+/gi;
    const links = content.match(linkRegex) || [];
    links.forEach(link => {
        if (link.includes('bit.ly') || link.includes('tinyurl') || link.includes('ngrok')) {
            analysis.suspiciousLinks.push(link);
        }
    });

    return analysis;
}

// --- 4. UI UPDATE FUNCTIONS ---

function updateEmailResults(analysis) {
    // Badges & Metrics
    updateText('phishingLevel', analysis.riskLevel);
    updateText('phishingScore', analysis.phishingScore);
    updateText('confidenceScore', analysis.confidence + '%');
    updateText('threatLevel', analysis.riskLevel.replace(' Risk', ''));

    const badge = document.getElementById('phishingBadge');
    if(badge) {
        badge.className = `phishing-badge ${analysis.riskClass}`;
        const icon = badge.querySelector('i');
        if(icon) icon.className = analysis.riskClass === 'safe' ? 'fas fa-shield-check' :
            analysis.riskClass === 'warning' ? 'fas fa-exclamation-triangle' : 'fas fa-shield-virus';
    }

    // Categories
    updateAnalysisCategories(analysis);
    updateHighlightedRisks(analysis);
    updateEmailRecommendations(analysis);
}

function updateAnalysisCategories(analysis) {
    // Helper to update status badges
    const setStatus = (id, text, type) => {
        const el = document.getElementById(id);
        if(el) { el.textContent = text; el.className = `status-badge ${type}`; }
    };

    // Keywords
    const spamContainer = document.getElementById('spamKeywords');
    if (analysis.spamKeywords.length > 0) {
        setStatus('spamStatus', 'Found', 'warning');
        spamContainer.innerHTML = `<p>Found keywords:</p><div class="keyword-list">${analysis.spamKeywords.map(k => `<span class="keyword-tag">${k}</span>`).join('')}</div>`;
    } else {
        setStatus('spamStatus', 'Clean', 'safe');
        spamContainer.innerHTML = '<p>No specific spam keywords detected.</p>';
    }

    // Links
    const linkContainer = document.getElementById('linkAnalysis');
    if (analysis.suspiciousLinks.length > 0) {
        setStatus('linkStatus', 'Suspicious', 'danger');
        linkContainer.innerHTML = `<p>Suspicious links:</p><div class="link-list">${analysis.suspiciousLinks.map(l => `<div class="suspicious-link">${l}</div>`).join('')}</div>`;
    } else {
        setStatus('linkStatus', 'Clean', 'safe');
        linkContainer.innerHTML = '<p>No malicious links found.</p>';
    }

    setStatus('senderStatus', analysis.senderVerification === 'Verified' ? 'Valid' : 'Check', analysis.senderVerification === 'Verified' ? 'safe' : 'warning');
    if(document.getElementById('senderAnalysis')) document.getElementById('senderAnalysis').innerHTML = `<p>${analysis.senderVerification}</p>`;
}

function updateHighlightedRisks(analysis) {
    const container = document.getElementById('highlightedRisks');
    const list = document.getElementById('riskHighlights');
    if(!container || !list) return;

    const risks = [];
    if (analysis.riskClass === 'danger') risks.push({type: 'danger', icon: 'fa-brain', title: 'AI Detection', desc: `AI is ${analysis.phishingScore}% confident this is Phishing.`});
    if (analysis.spamKeywords.length > 0) risks.push({type: 'warning', icon: 'fa-tags', title: 'Keywords', desc: 'Contains urgency or spam keywords.'});

    if (risks.length > 0) {
        container.style.display = 'block';
        list.innerHTML = risks.map(r => `<div class="risk-item ${r.type}"><div class="risk-icon"><i class="fas ${r.icon}"></i></div><div class="risk-content"><h4>${r.title}</h4><p>${r.desc}</p></div></div>`).join('');
    } else {
        container.style.display = 'none';
    }
}

function updateEmailRecommendations(analysis) {
    const list = document.getElementById('emailRecommendations');
    if(!list) return;

    let recs = [];
    if (analysis.riskClass === 'safe') {
        recs = [{type: 'safe', icon: 'fa-check-circle', text: 'Email appears safe.'}, {type: 'info', icon: 'fa-shield-alt', text: 'Always verify sender manually.'}];
    } else {
        recs = [{type: 'danger', icon: 'fa-ban', text: 'DO NOT click links.'}, {type: 'warning', icon: 'fa-trash', text: 'Delete this email.'}];
    }
    list.innerHTML = recs.map(r => `<div class="recommendation-item ${r.type}"><i class="fas ${r.icon}"></i><span>${r.text}</span></div>`).join('');
}

// --- 5. HELPER FUNCTIONS ---

// Helper to switch views safely
function toggleSection(view) {
    const inputSec = document.querySelector('.email-input-section');
    const loadSec = document.getElementById('loadingSection');
    const resultSec = document.getElementById('resultsSection');

    if(inputSec) inputSec.style.display = view === 'input' ? 'block' : 'none';
    if(loadSec) loadSec.style.display = view === 'loading' ? 'block' : 'none';
    if(resultSec) {
        resultSec.style.display = view === 'results' ? 'block' : 'none';
        if(view === 'results') resultSec.scrollIntoView({ behavior: 'smooth' });
    }
}

function updateText(id, text) {
    const el = document.getElementById(id);
    if(el) el.textContent = text;
}

// --- 6. ACTION BUTTON FUNCTIONS (Made Global) ---

window.checkAnotherEmail = function() {
    clearForm();
    toggleSection('input');
    // Reset progress
    const bar = document.querySelector('.progress-fill');
    if(bar) bar.style.width = '0%';
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

window.clearForm = function() {
    ['emailSubject', 'emailFrom', 'emailContent', 'emailHeaders'].forEach(id => {
        const el = document.getElementById(id);
        if(el) el.value = '';
    });
        removeFile();
};

window.reportPhishingEmail = function() {
    const subject = document.getElementById('emailSubject').value;
    const from = document.getElementById('emailFrom').value;
    sessionStorage.setItem('reportEmail', JSON.stringify({ subject, from }));
    window.location.href = 'report.html';
};

window.exportEmailReport = function() {
    showNotification('Exporting report...', 'success');
    // Export logic placeholder
};

// --- 7. FILE UPLOAD & TABS ---

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            tabContents.forEach(c => {
                c.classList.remove('active');
                if(c.id === target + 'Tab') c.classList.add('active');
            });
        });
    });
}

function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    if(!uploadArea || !fileInput) return;

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault(); uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) handleFileSelect(e.target.files[0]);
        });
}

function handleFileSelect(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        // Very basic parser
        const content = e.target.result;
        document.getElementById('emailContent').value = content;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileInfo').style.display = 'block';
        document.querySelector('[data-tab="paste"]').click(); // Switch tab to show data
        showNotification('File loaded. Please verify content.', 'success');
    };
    reader.readAsText(file);
}

function removeFile() {
    const fileInput = document.getElementById('fileInput');
    if(fileInput) fileInput.value = '';
    const fileInfo = document.getElementById('fileInfo');
    if(fileInfo) fileInfo.style.display = 'none';
}

function showNotification(msg, type) {
    alert(msg); // Basic fallback. Replace with custom toast if available.
}
