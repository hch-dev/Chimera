// Email Checker Page Specific JavaScript

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Setup tab switching
    setupTabs();
    
    // Setup file upload
    setupFileUpload();
    
    // Setup check button
    document.getElementById('checkBtn').addEventListener('click', checkEmail);
});

// Setup tab functionality
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Update button states
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content visibility
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetTab + 'Tab') {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Setup file upload functionality
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Handle file selection
function handleFileSelect(file) {
    const validTypes = ['.eml', '.msg', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(fileExtension)) {
        showNotification('Please select a valid email file (.eml, .msg, .txt)', 'error');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        showNotification('File size must be less than 10MB', 'error');
        return;
    }
    
    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'block';
    
    // Read file content
    const reader = new FileReader();
    reader.onload = (e) => {
        const content = e.target.result;
        parseEmailContent(content);
    };
    reader.readAsText(file);
}

// Remove uploaded file
function removeFile() {
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    clearForm();
}

// Parse email content from file
function parseEmailContent(content) {
    // Simple parsing - in a real app, this would be more sophisticated
    const lines = content.split('\n');
    let subject = '';
    let from = '';
    let headers = '';
    let body = '';
    let inHeaders = true;
    
    lines.forEach(line => {
        if (inHeaders && line.trim() === '') {
            inHeaders = false;
            return;
        }
        
        if (inHeaders) {
            headers += line + '\n';
            
            if (line.toLowerCase().startsWith('subject:')) {
                subject = line.replace(/^subject:\s*/i, '');
            } else if (line.toLowerCase().startsWith('from:')) {
                from = line.replace(/^from:\s*/i, '');
            }
        } else {
            body += line + '\n';
        }
    });
    
    // Fill form fields
    document.getElementById('emailSubject').value = subject;
    document.getElementById('emailFrom').value = from;
    document.getElementById('emailContent').value = body;
    document.getElementById('emailHeaders').value = headers;
    
    // Switch to paste tab
    document.querySelector('[data-tab="paste"]').click();
    
    showNotification('Email file loaded successfully', 'success');
}

// Check email for phishing
function checkEmail() {
    const subject = document.getElementById('emailSubject').value.trim();
    const from = document.getElementById('emailFrom').value.trim();
    const content = document.getElementById('emailContent').value.trim();
    
    if (!subject && !from && !content) {
        showNotification('Please enter at least the email content', 'warning');
        return;
    }
    
    // Hide input section, show loading
    document.querySelector('.email-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Simulate analysis
    simulateEmailAnalysis();
}

// Simulate email analysis
function simulateEmailAnalysis() {
    const progressFill = document.querySelector('.progress-fill');
    let progress = 0;
    
    const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 100) progress = 100;
        
        progressFill.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            setTimeout(() => showEmailResults(), 500);
        }
    }, 200);
}

// Show email analysis results
function showEmailResults() {
    const subject = document.getElementById('emailSubject').value;
    const from = document.getElementById('emailFrom').value;
    const content = document.getElementById('emailContent').value;
    const headers = document.getElementById('emailHeaders').value;
    
    const analysis = analyzeEmail(subject, from, content, headers);
    
    // Hide loading, show results
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update results
    updateEmailResults(analysis);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

// Analyze email for phishing (mock analysis)
function analyzeEmail(subject, from, content, headers) {
    // This is a mock analysis - in a real implementation, this would use ML models
    let phishingScore = 15; // Start with low score
    let riskLevel = 'Low Risk';
    let riskClass = 'safe';
    
    const analysis = {
        phishingScore: phishingScore,
        riskLevel: riskLevel,
        riskClass: riskClass,
        confidence: 92,
        spamKeywords: [],
        headerAnalysis: 'Valid',
        urgencyLanguage: 'Normal',
        attachments: 'None detected',
        suspiciousLinks: [],
        senderVerification: 'Verified'
    };
    
    // Check for suspicious patterns
    const suspiciousKeywords = [
        'urgent', 'immediate action required', 'account suspended', 
        'verify your account', 'click here', 'limited time',
        'congratulations', 'winner', 'claim now', 'free gift',
        'security alert', 'unusual activity', 'confirm identity'
    ];
    
    const contentLower = (subject + ' ' + content).toLowerCase();
    
    // Check for spam keywords
    suspiciousKeywords.forEach(keyword => {
        if (contentLower.includes(keyword)) {
            analysis.spamKeywords.push(keyword);
            phishingScore += 5;
        }
    });
    
    // Check for suspicious sender
    if (from) {
        const suspiciousDomains = ['gmail.com', 'yahoo.com', 'hotmail.com'];
        const fromDomain = from.split('@')[1];
        
        if (fromDomain && suspiciousDomains.includes(fromDomain)) {
            // Check if it claims to be from a company but uses free email
            if (contentLower.includes('bank') || contentLower.includes('paypal') || 
                contentLower.includes('amazon') || contentLower.includes('microsoft')) {
                analysis.senderVerification = 'Suspicious - Free email for business';
                phishingScore += 15;
            }
        }
    }
    
    // Check for suspicious links
    const linkRegex = /https?:\/\/[^\s]+/gi;
    const links = content.match(linkRegex) || [];
    
    links.forEach(link => {
        if (link.includes('bit.ly') || link.includes('tinyurl') || 
            link.includes('short.link') || link.includes('t.co')) {
            analysis.suspiciousLinks.push(link);
            phishingScore += 10;
        }
    });
    
    // Check for urgency language
    const urgencyKeywords = ['act now', 'immediately', 'urgent', 'hurry', 'don\'t wait'];
    let urgencyCount = 0;
    
    urgencyKeywords.forEach(keyword => {
        if (contentLower.includes(keyword)) {
            urgencyCount++;
        }
    });
    
    if (urgencyCount > 2) {
        analysis.urgencyLanguage = 'High urgency detected';
        phishingScore += urgencyCount * 3;
    }
    
    // Determine risk level based on score
    if (phishingScore >= 50) {
        analysis.riskLevel = 'High Risk';
        analysis.riskClass = 'danger';
    } else if (phishingScore >= 25) {
        analysis.riskLevel = 'Medium Risk';
        analysis.riskClass = 'warning';
    }
    
    analysis.phishingScore = phishingScore;
    
    return analysis;
}

// Update email results display
function updateEmailResults(analysis) {
    // Update risk badge and metrics
    const phishingBadge = document.getElementById('phishingBadge');
    const phishingLevel = document.getElementById('phishingLevel');
    const phishingScore = document.getElementById('phishingScore');
    const confidenceScore = document.getElementById('confidenceScore');
    const threatLevel = document.getElementById('threatLevel');
    
    // Update badge
    phishingBadge.className = `phishing-badge ${analysis.riskClass}`;
    phishingLevel.textContent = analysis.riskLevel;
    
    // Update metrics
    phishingScore.textContent = analysis.phishingScore;
    confidenceScore.textContent = analysis.confidence + '%';
    threatLevel.textContent = analysis.riskLevel.replace(' Risk', '');
    
    // Update badge icon
    const icon = phishingBadge.querySelector('i');
    icon.className = analysis.riskClass === 'safe' ? 'fas fa-shield-check' : 
                    analysis.riskClass === 'warning' ? 'fas fa-exclamation-triangle' : 
                    'fas fa-shield-virus';
    
    // Update analysis categories
    updateAnalysisCategories(analysis);
    
    // Update highlighted risks
    updateHighlightedRisks(analysis);
    
    // Update recommendations
    updateEmailRecommendations(analysis);
}

// Update analysis categories
function updateAnalysisCategories(analysis) {
    // Spam Keywords
    const spamStatus = document.getElementById('spamStatus');
    const spamKeywords = document.getElementById('spamKeywords');
    
    if (analysis.spamKeywords.length > 0) {
        spamStatus.className = 'status-badge warning';
        spamStatus.textContent = 'Suspicious';
        spamKeywords.innerHTML = `
            <p>Found ${analysis.spamKeywords.length} suspicious keywords:</p>
            <div class="keyword-list">
                ${analysis.spamKeywords.map(keyword => `<span class="keyword-tag">${keyword}</span>`).join('')}
            </div>
        `;
    } else {
        spamStatus.className = 'status-badge safe';
        spamStatus.textContent = 'Clean';
        spamKeywords.innerHTML = '<p>No suspicious keywords detected</p>';
    }
    
    // Email Headers
    const headerStatus = document.getElementById('headerStatus');
    const headerAnalysis = document.getElementById('headerAnalysis');
    
    headerStatus.className = `status-badge ${analysis.riskClass}`;
    headerStatus.textContent = analysis.headerAnalysis;
    headerAnalysis.innerHTML = `<p>${analysis.headerAnalysis}</p>`;
    
    // Urgency Language
    const urgencyStatus = document.getElementById('urgencyStatus');
    const urgencyAnalysis = document.getElementById('urgencyAnalysis');
    
    urgencyStatus.className = `status-badge ${analysis.urgencyLanguage === 'Normal' ? 'safe' : 'warning'}`;
    urgencyStatus.textContent = analysis.urgencyLanguage;
    urgencyAnalysis.innerHTML = `<p>${analysis.urgencyLanguage}</p>`;
    
    // Attachments
    const attachmentStatus = document.getElementById('attachmentStatus');
    const attachmentAnalysis = document.getElementById('attachmentAnalysis');
    
    attachmentStatus.className = 'status-badge safe';
    attachmentStatus.textContent = 'Safe';
    attachmentAnalysis.innerHTML = `<p>${analysis.attachments}</p>`;
    
    // Suspicious Links
    const linkStatus = document.getElementById('linkStatus');
    const linkAnalysis = document.getElementById('linkAnalysis');
    
    if (analysis.suspiciousLinks.length > 0) {
        linkStatus.className = 'status-badge danger';
        linkStatus.textContent = 'Suspicious';
        linkAnalysis.innerHTML = `
            <p>Found ${analysis.suspiciousLinks.length} suspicious links:</p>
            <div class="link-list">
                ${analysis.suspiciousLinks.map(link => `<div class="suspicious-link">${link}</div>`).join('')}
            </div>
        `;
    } else {
        linkStatus.className = 'status-badge safe';
        linkStatus.textContent = 'Clean';
        linkAnalysis.innerHTML = '<p>No malicious links found</p>';
    }
    
    // Sender Verification
    const senderStatus = document.getElementById('senderStatus');
    const senderAnalysis = document.getElementById('senderAnalysis');
    
    senderStatus.className = `status-badge ${analysis.senderVerification === 'Verified' ? 'safe' : 'warning'}`;
    senderStatus.textContent = analysis.senderVerification.split(' - ')[0];
    senderAnalysis.innerHTML = `<p>${analysis.senderVerification}</p>`;
}

// Update highlighted risks
function updateHighlightedRisks(analysis) {
    const highlightedRisks = document.getElementById('highlightedRisks');
    const riskHighlights = document.getElementById('riskHighlights');
    
    const risks = [];
    
    if (analysis.spamKeywords.length > 0) {
        risks.push({
            type: 'warning',
            icon: 'fa-tags',
            title: 'Suspicious Keywords',
            description: `Found ${analysis.spamKeywords.length} keywords commonly used in phishing emails`
        });
    }
    
    if (analysis.suspiciousLinks.length > 0) {
        risks.push({
            type: 'danger',
            icon: 'fa-link',
            title: 'Suspicious Links',
            description: `Detected ${analysis.suspiciousLinks.length} potentially malicious links`
        });
    }
    
    if (analysis.senderVerification !== 'Verified') {
        risks.push({
            type: 'warning',
            icon: 'fa-user',
            title: 'Sender Verification',
            description: 'Sender identity could not be verified'
        });
    }
    
    if (risks.length > 0) {
        highlightedRisks.style.display = 'block';
        riskHighlights.innerHTML = risks.map(risk => `
            <div class="risk-item ${risk.type}">
                <div class="risk-icon">
                    <i class="fas ${risk.icon}"></i>
                </div>
                <div class="risk-content">
                    <h4>${risk.title}</h4>
                    <p>${risk.description}</p>
                </div>
            </div>
        `).join('');
    } else {
        highlightedRisks.style.display = 'none';
    }
}

// Update email recommendations
function updateEmailRecommendations(analysis) {
    const recommendationsList = document.getElementById('emailRecommendations');
    let recommendations = [];
    
    if (analysis.riskClass === 'safe') {
        recommendations = [
            { type: 'safe', icon: 'fa-check-circle', text: 'This email appears to be legitimate' },
            { type: 'info', icon: 'fa-info-circle', text: 'Always verify sender identity for sensitive requests' },
            { type: 'info', icon: 'fa-shield-alt', text: 'Continue to monitor for suspicious activity' }
        ];
    } else if (analysis.riskClass === 'warning') {
        recommendations = [
            { type: 'warning', icon: 'fa-exclamation-triangle', text: 'Exercise caution with this email' },
            { type: 'danger', icon: 'fa-times-circle', text: 'Do not click on any links or download attachments' },
            { type: 'info', icon: 'fa-phone', text: 'Contact the sender through a different channel to verify' },
            { type: 'warning', icon: 'fa-flag', text: 'Consider reporting this email' }
        ];
    } else {
        recommendations = [
            { type: 'danger', icon: 'fa-shield-virus', text: 'This email shows strong signs of phishing' },
            { type: 'danger', icon: 'fa-ban', text: 'DO NOT click any links or provide any information' },
            { type: 'danger', icon: 'fa-trash', text: 'Delete this email immediately' },
            { type: 'warning', icon: 'fa-flag', text: 'Report this email to help protect others' },
            { type: 'info', icon: 'fa-key', text: 'Change your passwords if you clicked any links' }
        ];
    }
    
    recommendationsList.innerHTML = recommendations.map(rec => `
        <div class="recommendation-item ${rec.type}">
            <i class="fas ${rec.icon}"></i>
            <span>${rec.text}</span>
        </div>
    `).join('');
}

// Check another email
function checkAnotherEmail() {
    clearForm();
    document.querySelector('.email-input-section').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Reset progress bar
    document.querySelector('.progress-fill').style.width = '0%';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Clear form
function clearForm() {
    document.getElementById('emailSubject').value = '';
    document.getElementById('emailFrom').value = '';
    document.getElementById('emailContent').value = '';
    document.getElementById('emailHeaders').value = '';
    removeFile();
}

// Report phishing email
function reportPhishingEmail() {
    const subject = document.getElementById('emailSubject').value;
    const from = document.getElementById('emailFrom').value;
    
    if (subject || from) {
        sessionStorage.setItem('reportEmail', JSON.stringify({ subject, from }));
        window.location.href = 'report.html';
    } else {
        showNotification('No email to report', 'warning');
    }
}

// Export email report
function exportEmailReport() {
    const subject = document.getElementById('emailSubject').value;
    const from = document.getElementById('emailFrom').value;
    const content = document.getElementById('emailContent').value;
    const analysis = analyzeEmail(subject, from, content, '');
    
    const report = {
        emailAnalysis: {
            subject: subject,
            from: from,
            scanDate: new Date().toISOString(),
            phishingScore: analysis.phishingScore,
            riskLevel: analysis.riskLevel,
            analysis: {
                spamKeywords: analysis.spamKeywords,
                suspiciousLinks: analysis.suspiciousLinks,
                senderVerification: analysis.senderVerification,
                urgencyLanguage: analysis.urgencyLanguage
            },
            confidence: analysis.confidence
        }
    };
    
    // Create and download JSON file
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `chimera-email-analysis-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showNotification('Email analysis report exported successfully', 'success');
}

// Add email checker specific CSS
const emailCheckerCSS = `
.email-checker-main {
    padding: 2rem 0 4rem;
    min-height: calc(100vh - 200px);
}

.checker-header {
    text-align: center;
    margin-bottom: 3rem;
}

.checker-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.checker-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
}

.checker-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    max-width: 900px;
    margin: 0 auto;
}

.method-tabs {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid var(--border-color);
}

.tab-btn {
    background: none;
    border: none;
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    position: relative;
    transition: var(--transition);
}

.tab-btn.active {
    color: var(--primary-color);
}

.tab-btn.active::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--primary-color);
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.input-group {
    margin-bottom: 1.5rem;
}

.input-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.email-input, .email-textarea {
    width: 100%;
    padding: 1rem 1.5rem;
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    font-size: 1rem;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    transition: var(--transition);
    font-family: var(--font-family);
}

.email-input:focus, .email-textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.email-textarea {
    resize: vertical;
    min-height: 120px;
}

.input-help {
    display: block;
    margin-top: 0.5rem;
    color: var(--text-muted);
    font-size: 0.85rem;
}

.upload-area {
    border: 2px dashed var(--border-color);
    border-radius: var(--border-radius);
    padding: 3rem;
    text-align: center;
    transition: var(--transition);
    cursor: pointer;
}

.upload-area:hover, .upload-area.drag-over {
    border-color: var(--primary-color);
    background: rgba(0, 102, 204, 0.05);
}

.upload-icon {
    font-size: 3rem;
    color: var(--primary-color);
    margin-bottom: 1rem;
}

.upload-area h3 {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.upload-area p {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

.file-info {
    margin-top: 1rem;
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
}

.file-details {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.remove-file {
    background: var(--danger-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: auto;
}

.check-actions {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color);
}

.email-scanner-animation {
    position: relative;
    width: 150px;
    height: 150px;
    margin: 0 auto 2rem;
    display: flex;
    justify-content: center;
    align-items: center;
}

.envelope-icon {
    font-size: 3rem;
    color: var(--primary-color);
    z-index: 2;
}

.scan-waves {
    position: absolute;
    width: 100%;
    height: 100%;
}

.wave {
    position: absolute;
    width: 100%;
    height: 100%;
    border: 2px solid var(--primary-color);
    border-radius: 50%;
    opacity: 0;
    animation: scanWave 2s infinite;
}

.wave:nth-child(2) {
    animation-delay: 0.5s;
}

.wave:nth-child(3) {
    animation-delay: 1s;
}

@keyframes scanWave {
    0% {
        transform: scale(0.8);
        opacity: 0.8;
    }
    100% {
        transform: scale(1.5);
        opacity: 0;
    }
}

.phishing-risk-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.risk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
}

.phishing-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
}

.phishing-badge.safe {
    background: rgba(40, 167, 69, 0.1);
    color: var(--success-color);
    border: 1px solid var(--success-color);
}

.phishing-badge.warning {
    background: rgba(255, 193, 7, 0.1);
    color: var(--warning-color);
    border: 1px solid var(--warning-color);
}

.phishing-badge.danger {
    background: rgba(220, 53, 69, 0.1);
    color: var(--danger-color);
    border: 1px solid var(--danger-color);
}

.risk-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1.5rem;
}

.metric-card {
    text-align: center;
    padding: 1.5rem;
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.metric-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.analysis-breakdown {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.analysis-breakdown h3 {
    margin-bottom: 2rem;
    color: var(--text-primary);
}

.analysis-categories {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
}

.category-card {
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
    overflow: hidden;
}

.category-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-color);
}

.category-icon {
    width: 40px;
    height: 40px;
    background: var(--gradient-primary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    flex-shrink: 0;
}

.category-header h4 {
    flex: 1;
    margin: 0;
    color: var(--text-primary);
}

.status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-badge.safe {
    background: rgba(40, 167, 69, 0.1);
    color: var(--success-color);
}

.status-badge.warning {
    background: rgba(255, 193, 7, 0.1);
    color: var(--warning-color);
}

.status-badge.danger {
    background: rgba(220, 53, 69, 0.1);
    color: var(--danger-color);
}

.category-content {
    padding: 1.5rem;
}

.keyword-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.keyword-tag {
    background: var(--warning-color);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}

.link-list {
    margin-top: 1rem;
}

.suspicious-link {
    background: var(--danger-color);
    color: white;
    padding: 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    word-break: break-all;
}

.highlighted-risks {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.highlighted-risks h3 {
    margin-bottom: 2rem;
    color: var(--text-primary);
}

.risk-highlights {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.risk-item {
    display: flex;
    gap: 1rem;
    padding: 1.5rem;
    border-radius: var(--border-radius);
    border-left: 4px solid;
}

.risk-item.warning {
    background: rgba(255, 193, 7, 0.1);
    border-left-color: var(--warning-color);
}

.risk-item.danger {
    background: rgba(220, 53, 69, 0.1);
    border-left-color: var(--danger-color);
}

.risk-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-shrink: 0;
}

.risk-item.warning .risk-icon {
    background: var(--warning-color);
    color: white;
}

.risk-item.danger .risk-icon {
    background: var(--danger-color);
    color: white;
}

.risk-content h4 {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.risk-content p {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    .checker-header h1 {
        font-size: 2rem;
    }
    
    .method-tabs {
        flex-direction: column;
        border-bottom: none;
    }
    
    .tab-btn {
        border-bottom: 1px solid var(--border-color);
    }
    
    .tab-btn.active::after {
        display: none;
    }
    
    .risk-metrics {
        grid-template-columns: 1fr;
    }
    
    .analysis-categories {
        grid-template-columns: 1fr;
    }
    
    .check-actions {
        flex-direction: column;
    }
    
    .check-actions .btn {
        width: 100%;
    }
}
`;

// Inject email checker CSS
const emailCheckerStyleSheet = document.createElement('style');
emailCheckerStyleSheet.textContent = emailCheckerCSS;
document.head.appendChild(emailCheckerStyleSheet);
