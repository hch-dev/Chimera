// Scanner Page Specific JavaScript

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Check for URL from homepage
    const urlFromSession = sessionStorage.getItem('scanUrl');
    if (urlFromSession) {
        document.getElementById('urlInput').value = urlFromSession;
        sessionStorage.removeItem('scanUrl');
        // Auto-scan if URL was passed from homepage
        setTimeout(() => performScan(), 500);
    }
    
    // Setup event listeners
    document.getElementById('scanBtn').addEventListener('click', performScan);
    document.getElementById('urlInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performScan();
        }
    });
});

// Set example URL
function setExampleUrl(url) {
    document.getElementById('urlInput').value = url;
    document.getElementById('urlInput').focus();
}

// Perform URL scan
function performScan() {
    const urlInput = document.getElementById('urlInput');
    const url = urlInput.value.trim();
    
    if (!url) {
        showNotification('Please enter a URL to scan', 'warning');
        return;
    }
    
    if (!isValidURL(url)) {
        showNotification('Please enter a valid URL (e.g., https://example.com)', 'error');
        return;
    }
    
    // Hide input section, show loading
    document.querySelector('.url-input-section').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Simulate scanning process with progress
    simulateScanningProcess();
}

// Simulate scanning process
function simulateScanningProcess() {
    const progressFill = document.querySelector('.progress-fill');
    let progress = 0;
    
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        progressFill.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            setTimeout(() => showResults(), 500);
        }
    }, 300);
}

// Show scan results
function showResults() {
    const url = document.getElementById('urlInput').value;
    const results = analyzeURL(url);
    
    // Hide loading, show results
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update risk badge and score
    updateRiskDisplay(results);
    
    // Update analysis details
    updateAnalysisDetails(results);
    
    // Update recommendations
    updateRecommendations(results);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

// Analyze URL (mock analysis)
function analyzeURL(url) {
    // This is a mock analysis - in a real implementation, this would call backend APIs
    const urlObj = new URL(url);
    const domain = urlObj.hostname;
    
    // Mock risk calculation based on URL characteristics
    let riskScore = 85; // Start with safe score
    let riskLevel = 'Low';
    let riskClass = 'safe';
    
    const analysis = {
        url: url,
        domain: domain,
        riskScore: riskScore,
        riskLevel: riskLevel,
        riskClass: riskClass,
        ssl: 'Valid',
        domainAge: '5+ years',
        suspiciousPatterns: 'None detected',
        maliciousKeywords: 'None found',
        redirectBehavior: 'No redirects',
        blacklistStatus: 'Clean',
        confidence: 95
    };
    
    // Simulate different results for different URLs
    if (domain.includes('suspicious') || domain.includes('fake') || domain.includes('phishing')) {
        analysis.riskScore = 25;
        analysis.riskLevel = 'High';
        analysis.riskClass = 'danger';
        analysis.ssl = 'Invalid or Missing';
        analysis.suspiciousPatterns = 'Suspicious domain patterns detected';
        analysis.maliciousKeywords = 'Phishing-related keywords found';
        analysis.blacklistStatus = 'Found on threat databases';
        analysis.confidence = 88;
    } else if (domain.includes('123') || url.includes('http://')) {
        analysis.riskScore = 60;
        analysis.riskLevel = 'Medium';
        analysis.riskClass = 'warning';
        analysis.ssl = 'Self-signed';
        analysis.domainAge = 'Less than 1 year';
        analysis.suspiciousPatterns = 'Numeric domain detected';
        analysis.blacklistStatus = 'Not blacklisted but suspicious';
        analysis.confidence = 75;
    }
    
    return analysis;
}

// Update risk display
function updateRiskDisplay(results) {
    const riskBadge = document.getElementById('riskBadge');
    const riskLevel = document.getElementById('riskLevel');
    const scoreValue = document.getElementById('scoreValue');
    const threatLevel = document.getElementById('threatLevel');
    const confidenceLevel = document.getElementById('confidenceLevel');
    
    // Update badge
    riskBadge.className = `risk-badge ${results.riskClass}`;
    riskLevel.textContent = results.riskLevel;
    
    // Update score
    scoreValue.textContent = results.riskScore;
    
    // Update details
    threatLevel.textContent = results.riskLevel;
    confidenceLevel.textContent = results.confidence + '%';
    
    // Update badge icon
    const icon = riskBadge.querySelector('i');
    icon.className = results.riskClass === 'safe' ? 'fas fa-shield-check' : 
                    results.riskClass === 'warning' ? 'fas fa-exclamation-triangle' : 
                    'fas fa-shield-virus';
}

// Update analysis details
function updateAnalysisDetails(results) {
    document.getElementById('sslStatus').textContent = `${results.ssl} SSL certificate`;
    document.getElementById('domainAge').textContent = `Domain registered ${results.domainAge} ago`;
    document.getElementById('suspiciousPatterns').textContent = results.suspiciousPatterns;
    document.getElementById('maliciousKeywords').textContent = results.maliciousKeywords;
    document.getElementById('redirectBehavior').textContent = results.redirectBehavior;
    document.getElementById('blacklistStatus').textContent = results.blacklistStatus;
    
    // Update status indicators
    const indicators = document.querySelectorAll('.status-indicator');
    indicators.forEach((indicator, index) => {
        indicator.className = `status-indicator ${results.riskClass}`;
    });
}

// Update recommendations
function updateRecommendations(results) {
    const recommendationsList = document.getElementById('recommendationsList');
    let recommendations = [];
    
    if (results.riskClass === 'safe') {
        recommendations = [
            { type: 'safe', icon: 'fa-check-circle', text: 'This website appears to be safe for browsing' },
            { type: 'info', icon: 'fa-info-circle', text: 'Always verify the URL before entering sensitive information' },
            { type: 'info', icon: 'fa-lock', text: 'Ensure the site uses HTTPS encryption' }
        ];
    } else if (results.riskClass === 'warning') {
        recommendations = [
            { type: 'warning', icon: 'fa-exclamation-triangle', text: 'Proceed with caution - some risks detected' },
            { type: 'info', icon: 'fa-eye', text: 'Carefully inspect the website before entering any data' },
            { type: 'danger', icon: 'fa-times-circle', text: 'Avoid entering passwords or financial information' },
            { type: 'info', icon: 'fa-search', text: 'Consider using an alternative, well-known website' }
        ];
    } else {
        recommendations = [
            { type: 'danger', icon: 'fa-shield-virus', text: 'This website shows strong signs of being malicious' },
            { type: 'danger', icon: 'fa-ban', text: 'DO NOT proceed - high risk of phishing detected' },
            { type: 'warning', icon: 'fa-flag', text: 'Report this website to help protect others' },
            { type: 'info', icon: 'fa-shield-alt', text: 'Scan your device for malware if you visited this site' }
        ];
    }
    
    recommendationsList.innerHTML = recommendations.map(rec => `
        <div class="recommendation-item ${rec.type}">
            <i class="fas ${rec.icon}"></i>
            <span>${rec.text}</span>
        </div>
    `).join('');
}

// Scan another URL
function scanAnother() {
    // Reset form
    document.getElementById('urlInput').value = '';
    document.querySelector('.url-input-section').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Reset progress bar
    document.querySelector('.progress-fill').style.width = '0%';
    
    // Focus on input
    document.getElementById('urlInput').focus();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Report suspicious website
function reportSuspicious() {
    const url = document.getElementById('urlInput').value;
    if (url) {
        sessionStorage.setItem('reportUrl', url);
        window.location.href = 'report.html';
    } else {
        showNotification('No URL to report', 'warning');
    }
}

// Export report
function exportReport() {
    const url = document.getElementById('urlInput').value;
    const results = analyzeURL(url);
    
    const report = {
        url: url,
        scanDate: new Date().toISOString(),
        riskScore: results.riskScore,
        riskLevel: results.riskLevel,
        analysis: {
            ssl: results.ssl,
            domainAge: results.domainAge,
            suspiciousPatterns: results.suspiciousPatterns,
            maliciousKeywords: results.maliciousKeywords,
            redirectBehavior: results.redirectBehavior,
            blacklistStatus: results.blacklistStatus
        },
        confidence: results.confidence
    };
    
    // Create and download JSON file
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `chimera-scan-report-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showNotification('Report exported successfully', 'success');
}

// Add scanner-specific CSS
const scannerCSS = `
.scanner-main {
    padding: 2rem 0 4rem;
    min-height: calc(100vh - 200px);
}

.scanner-header {
    text-align: center;
    margin-bottom: 3rem;
}

.scanner-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.scanner-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
}

.scanner-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    max-width: 800px;
    margin: 0 auto;
}

.input-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.url-input-wrapper {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.url-input {
    flex: 1;
    padding: 1rem 1.5rem;
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    font-size: 1rem;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    transition: var(--transition);
}

.url-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.scan-button {
    padding: 1rem 2rem;
    white-space: nowrap;
}

.input-suggestions {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.suggestion-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.suggestion-btn {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 0.5rem 1rem;
    color: var(--text-secondary);
    font-size: 0.85rem;
    cursor: pointer;
    transition: var(--transition);
}

.suggestion-btn:hover {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.loading-section {
    text-align: center;
    padding: 4rem 0;
}

.loading-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 3rem;
    box-shadow: var(--shadow-md);
    max-width: 600px;
    margin: 0 auto;
}

.scanner-animation-large {
    position: relative;
    width: 200px;
    height: 200px;
    margin: 0 auto 2rem;
    display: flex;
    justify-content: center;
    align-items: center;
}

.scanner-circle-large {
    position: absolute;
    width: 150px;
    height: 150px;
    border: 3px solid var(--primary-color);
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.scanner-wave-large {
    position: absolute;
    width: 180px;
    height: 180px;
    border: 2px solid var(--secondary-color);
    border-radius: 50%;
    animation: wave 3s infinite;
}

.scanner-icon-large {
    font-size: 3rem;
    color: var(--primary-color);
    z-index: 2;
    animation: float 3s ease-in-out infinite;
}

.progress-bar {
    width: 100%;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 2rem;
}

.progress-fill {
    height: 100%;
    background: var(--gradient-primary);
    border-radius: 3px;
    transition: width 0.3s ease;
}

.results-section {
    animation: fadeInUp 0.6s ease-out;
}

.risk-score-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.risk-score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
}

.risk-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    font-weight: 600;
}

.risk-badge.safe {
    background: rgba(40, 167, 69, 0.1);
    color: var(--success-color);
    border: 1px solid var(--success-color);
}

.risk-badge.warning {
    background: rgba(255, 193, 7, 0.1);
    color: var(--warning-color);
    border: 1px solid var(--warning-color);
}

.risk-badge.danger {
    background: rgba(220, 53, 69, 0.1);
    color: var(--danger-color);
    border: 1px solid var(--danger-color);
}

.risk-score-display {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 2rem;
    align-items: center;
}

.score-circle {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: var(--gradient-primary);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
    position: relative;
}

.score-circle span:first-child {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

.score-details {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.score-item {
    display: flex;
    justify-content: space-between;
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: 8px;
}

.analysis-details {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.analysis-details h3 {
    margin-bottom: 2rem;
    color: var(--text-primary);
}

.analysis-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

.analysis-item {
    display: flex;
    gap: 1rem;
    padding: 1.5rem;
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
}

.analysis-icon {
    width: 50px;
    height: 50px;
    background: var(--gradient-primary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    flex-shrink: 0;
}

.analysis-content {
    flex: 1;
}

.analysis-content h4 {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.analysis-content p {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-top: 0.5rem;
}

.status-indicator.safe {
    background: var(--success-color);
}

.status-indicator.warning {
    background: var(--warning-color);
}

.status-indicator.danger {
    background: var(--danger-color);
}

.recommendations-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 2rem;
}

.recommendations-card h3 {
    margin-bottom: 2rem;
    color: var(--text-primary);
}

.recommendations-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.recommendation-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid;
}

.recommendation-item.safe {
    background: rgba(40, 167, 69, 0.1);
    border-left-color: var(--success-color);
    color: var(--success-color);
}

.recommendation-item.warning {
    background: rgba(255, 193, 7, 0.1);
    border-left-color: var(--warning-color);
    color: var(--warning-color);
}

.recommendation-item.danger {
    background: rgba(220, 53, 69, 0.1);
    border-left-color: var(--danger-color);
    color: var(--danger-color);
}

.recommendation-item.info {
    background: rgba(0, 102, 204, 0.1);
    border-left-color: var(--primary-color);
    color: var(--primary-color);
}

.action-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
}

@media (max-width: 768px) {
    .scanner-header h1 {
        font-size: 2rem;
    }
    
    .url-input-wrapper {
        flex-direction: column;
    }
    
    .risk-score-display {
        grid-template-columns: 1fr;
        text-align: center;
    }
    
    .score-circle {
        margin: 0 auto;
    }
    
    .analysis-grid {
        grid-template-columns: 1fr;
    }
    
    .action-buttons {
        flex-direction: column;
    }
    
    .action-buttons .btn {
        width: 100%;
    }
}
`;

// Inject scanner CSS
const scannerStyleSheet = document.createElement('style');
scannerStyleSheet.textContent = scannerCSS;
document.head.appendChild(scannerStyleSheet);
