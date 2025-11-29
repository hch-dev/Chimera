// Report Page Specific JavaScript

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Load any pre-filled data from sessionStorage
    loadPrefilledData();
    
    // Setup form validation
    setupFormValidation();
    
    // Setup form submission
    setupFormSubmission();
});

// Load pre-filled data
function loadPrefilledData() {
    const reportUrl = sessionStorage.getItem('reportUrl');
    const reportEmail = sessionStorage.getItem('reportEmail');
    
    if (reportUrl) {
        document.getElementById('reportUrl').value = reportUrl;
        sessionStorage.removeItem('reportUrl');
    }
    
    if (reportEmail) {
        try {
            const emailData = JSON.parse(reportEmail);
            if (emailData.subject) {
                document.getElementById('reportUrl').value = 'Email phishing: ' + emailData.subject;
                document.getElementById('reportType').value = 'phishing';
                document.getElementById('description').value = `Phishing email from: ${emailData.from}\nSubject: ${emailData.subject}`;
            }
            sessionStorage.removeItem('reportEmail');
        } catch (e) {
            console.error('Error parsing email data:', e);
        }
    }
}

// Setup form validation
function setupFormValidation() {
    const form = document.getElementById('reportForm');
    const inputs = form.querySelectorAll('input, textarea, select');
    
    // Real-time validation
    inputs.forEach(input => {
        input.addEventListener('blur', () => validateField(input));
        input.addEventListener('input', () => {
            if (input.classList.contains('error')) {
                validateField(input);
            }
        });
    });
    
    // URL validation
    const urlInput = document.getElementById('reportUrl');
    urlInput.addEventListener('input', () => {
        const value = urlInput.value.trim();
        if (value && !isValidURL(value)) {
            showFieldError(urlInput, 'Please enter a valid URL (e.g., https://example.com)');
        } else {
            clearFieldError(urlInput);
        }
    });
}

// Validate individual field
function validateField(field) {
    const value = field.value.trim();
    const fieldName = field.name;
    
    // Clear previous errors
    clearFieldError(field);
    
    // Required field validation
    if (field.hasAttribute('required') && !value) {
        showFieldError(field, 'This field is required');
        return false;
    }
    
    // Specific field validations
    switch (fieldName) {
        case 'url':
            if (value && !isValidURL(value)) {
                showFieldError(field, 'Please enter a valid URL (e.g., https://example.com)');
                return false;
            }
            break;
            
        case 'email':
            if (value && !isValidEmail(value)) {
                showFieldError(field, 'Please enter a valid email address');
                return false;
            }
            break;
            
        case 'description':
            if (value.length < 10) {
                showFieldError(field, 'Please provide at least 10 characters of description');
                return false;
            }
            break;
    }
    
    return true;
}

// Show field error
function showFieldError(field, message) {
    field.classList.add('error');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

// Clear field error
function clearFieldError(field) {
    field.classList.remove('error');
    const errorMessage = field.parentNode.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
}

// Setup form submission
function setupFormSubmission() {
    const form = document.getElementById('reportForm');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate all fields
        const inputs = form.querySelectorAll('input, textarea, select');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!validateField(input)) {
                isValid = false;
            }
        });
        
        if (!isValid) {
            showNotification('Please correct the errors in the form', 'error');
            return;
        }
        
        // Show loading state
        const submitBtn = form.querySelector('.submit-btn');
        const originalContent = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Submitting...';
        submitBtn.disabled = true;
        
        // Collect form data
        const formData = new FormData(form);
        const reportData = {
            url: formData.get('url'),
            type: formData.get('type'),
            threatLevel: formData.get('threatLevel'),
            description: formData.get('description'),
            email: formData.get('email'),
            evidence: formData.get('evidence'),
            discoveryMethod: formData.get('discoveryMethod'),
            timestamp: new Date().toISOString(),
            id: Date.now()
        };
        
        // Simulate submission
        setTimeout(() => {
            // Save report to localStorage
            saveReport(reportData);
            
            // Show success message
            showSuccessMessage(reportData);
            
            // Reset form
            resetForm();
            
            // Restore button
            submitBtn.innerHTML = originalContent;
            submitBtn.disabled = false;
        }, 2000);
    });
}

// Save report to localStorage
function saveReport(reportData) {
    const existingReports = JSON.parse(localStorage.getItem('chimera_reports') || '[]');
    existingReports.unshift(reportData);
    
    // Keep only last 100 reports
    if (existingReports.length > 100) {
        existingReports.splice(100);
    }
    
    localStorage.setItem('chimera_reports', JSON.stringify(existingReports));
}

// Show success message
function showSuccessMessage(reportData) {
    // Hide form
    document.querySelector('.report-form-section').style.display = 'none';
    
    // Show success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = `
        <div class="success-content">
            <div class="success-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h2>Report Submitted Successfully!</h2>
            <p>Thank you for helping keep the internet safe. Your report has been received and will be reviewed by our team.</p>
            <div class="report-summary">
                <h3>Report Summary:</h3>
                <div class="summary-item">
                    <strong>URL:</strong> ${reportData.url}
                </div>
                <div class="summary-item">
                    <strong>Type:</strong> ${reportData.type}
                </div>
                <div class="summary-item">
                    <strong>Threat Level:</strong> ${reportData.threatLevel}
                </div>
                <div class="summary-item">
                    <strong>Report ID:</strong> #${reportData.id}
                </div>
            </div>
            <div class="success-actions">
                <button onclick="submitAnotherReport()" class="btn btn-primary">
                    <i class="fas fa-plus"></i>
                    Submit Another Report
                </button>
                <a href="reports.html" class="btn btn-secondary">
                    <i class="fas fa-chart-bar"></i>
                    View Dashboard
                </a>
                <a href="index.html" class="btn btn-outline">
                    <i class="fas fa-home"></i>
                    Return Home
                </a>
            </div>
        </div>
    `;
    
    // Insert after header
    const header = document.querySelector('.report-header');
    header.parentNode.insertBefore(successDiv, header.nextSibling);
    
    // Scroll to success message
    successDiv.scrollIntoView({ behavior: 'smooth' });
}

// Submit another report
function submitAnotherReport() {
    // Remove success message
    const successMessage = document.querySelector('.success-message');
    if (successMessage) {
        successMessage.remove();
    }
    
    // Show form
    document.querySelector('.report-form-section').style.display = 'block';
    
    // Scroll to form
    document.querySelector('.report-form-section').scrollIntoView({ behavior: 'smooth' });
}

// Reset form
function resetForm() {
    const form = document.getElementById('reportForm');
    form.reset();
    
    // Clear all field errors
    const inputs = form.querySelectorAll('input, textarea, select');
    inputs.forEach(input => clearFieldError(input));
}

// Utility functions
function isValidURL(string) {
    try {
        const url = new URL(string);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (_) {
        return false;
    }
}

function isValidEmail(string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(string);
}

// Add report page specific CSS
const reportCSS = `
.report-main {
    padding: 2rem 0 4rem;
    min-height: calc(100vh - 200px);
}

.report-header {
    text-align: center;
    margin-bottom: 3rem;
}

.report-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.report-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
}

.report-form-section {
    margin-bottom: 4rem;
}

.form-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 3rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    max-width: 800px;
    margin: 0 auto;
}

.form-group {
    margin-bottom: 2rem;
}

.form-group label {
    display: block;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--text-primary);
}

.form-input, .form-select, .form-textarea {
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

.form-input:focus, .form-select:focus, .form-textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.form-input.error, .form-select.error, .form-textarea.error {
    border-color: var(--danger-color);
}

.form-textarea {
    resize: vertical;
    min-height: 120px;
}

.form-help {
    display: block;
    margin-top: 0.5rem;
    color: var(--text-muted);
    font-size: 0.85rem;
}

.error-message {
    color: var(--danger-color);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.error-message::before {
    content: '⚠';
    font-weight: bold;
}

.form-section {
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid var(--border-color);
}

.form-section h3 {
    margin-bottom: 1.5rem;
    color: var(--text-primary);
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 0.5rem;
}

.threat-levels {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.threat-option {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    background: var(--bg-primary);
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    cursor: pointer;
    transition: var(--transition);
}

.threat-option:hover {
    border-color: var(--primary-color);
    background: rgba(0, 102, 204, 0.05);
}

.threat-option input[type="radio"] {
    margin: 0;
}

.threat-indicator {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
}

.threat-indicator.low {
    background: var(--success-color);
}

.threat-indicator.medium {
    background: var(--warning-color);
}

.threat-indicator.high {
    background: var(--danger-color);
}

.threat-label {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 100px;
}

.threat-desc {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.checkbox-label {
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    color: var(--text-primary);
}

.checkbox-label input[type="checkbox"] {
    margin: 0;
}

.checkmark {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-color);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: var(--transition);
}

.checkbox-label input[type="checkbox"]:checked + .checkmark {
    background: var(--primary-color);
    border-color: var(--primary-color);
}

.checkbox-label input[type="checkbox"]:checked + .checkmark::after {
    content: '✓';
    color: white;
    font-weight: bold;
}

.form-actions {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border-color);
}

.success-message {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 3rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
    max-width: 800px;
    margin: 0 auto 3rem;
    text-align: center;
}

.success-content {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.success-icon {
    width: 80px;
    height: 80px;
    background: var(--success-color);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 2rem;
    color: white;
    font-size: 2.5rem;
}

.success-content h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.success-content p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
    max-width: 600px;
}

.report-summary {
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    padding: 2rem;
    margin: 2rem 0;
    text-align: left;
    border: 1px solid var(--border-color);
}

.report-summary h3 {
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.summary-item {
    margin-bottom: 0.75rem;
    color: var(--text-secondary);
}

.summary-item strong {
    color: var(--text-primary);
}

.success-actions {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
}

.recent-reports {
    margin-bottom: 4rem;
}

.section-header {
    text-align: center;
    margin-bottom: 3rem;
}

.section-header h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.section-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.reports-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
}

.report-item {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
    transition: var(--transition);
}

.report-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.report-type {
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
}

.report-type.phishing {
    background: var(--danger-color);
}

.report-type.scam {
    background: var(--warning-color);
}

.report-type.malware {
    background: #8b0000;
}

.report-type.fake {
    background: var(--primary-color);
}

.report-type.spam {
    background: var(--text-muted);
}

.report-date {
    color: var(--text-muted);
    font-size: 0.85rem;
}

.report-content h4 {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.report-url {
    font-family: monospace;
    font-size: 0.9rem;
    color: var(--primary-color);
    margin-bottom: 0.75rem;
    word-break: break-all;
}

.report-desc {
    color: var(--text-secondary);
    margin-bottom: 1rem;
    line-height: 1.6;
}

.report-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.threat-level {
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
}

.threat-level.low {
    background: var(--success-color);
}

.threat-level.medium {
    background: var(--warning-color);
}

.threat-level.high {
    background: var(--danger-color);
}

.report-status {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-weight: 500;
}

.impact-stats {
    margin-bottom: 3rem;
}

.stats-header {
    text-align: center;
    margin-bottom: 3rem;
}

.stats-header h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.stats-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.impact-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
}

.impact-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    border: 1px solid var(--border-color);
    transition: var(--transition);
}

.impact-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.impact-icon {
    width: 60px;
    height: 60px;
    background: var(--gradient-primary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto 1.5rem;
    color: white;
    font-size: 1.5rem;
}

.impact-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.impact-label {
    color: var(--text-secondary);
    font-weight: 500;
}

@media (max-width: 768px) {
    .report-header h1 {
        font-size: 2rem;
    }
    
    .form-card {
        padding: 2rem 1.5rem;
    }
    
    .threat-option {
        flex-direction: column;
        text-align: center;
        gap: 0.75rem;
    }
    
    .threat-label {
        min-width: auto;
    }
    
    .form-actions {
        flex-direction: column;
    }
    
    .form-actions .btn {
        width: 100%;
    }
    
    .reports-grid {
        grid-template-columns: 1fr;
    }
    
    .impact-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .success-actions {
        flex-direction: column;
    }
    
    .success-actions .btn {
        width: 100%;
    }
}
`;

// Inject report page CSS
const reportStyleSheet = document.createElement('style');
reportStyleSheet.textContent = reportCSS;
document.head.appendChild(reportStyleSheet);
