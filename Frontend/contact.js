// Contact Page Specific JavaScript

// Contact form configurations
const contactTypes = {
    support: {
        title: 'Technical Support',
        subtitle: 'Get help with technical issues, bugs, or feature requests',
        fields: ['name', 'email', 'subject', 'message'],
        priority: 'medium'
    },
    security: {
        title: 'Security Concern',
        subtitle: 'Report security vulnerabilities or urgent issues',
        fields: ['name', 'email', 'subject', 'message'],
        priority: 'urgent'
    },
    partnership: {
        title: 'Partnership Inquiry',
        subtitle: 'Explore collaboration opportunities',
        fields: ['name', 'email', 'organization', 'subject', 'message'],
        priority: 'medium'
    },
    feedback: {
        title: 'General Feedback',
        subtitle: 'Share your thoughts and suggestions',
        fields: ['name', 'email', 'subject', 'message'],
        priority: 'low'
    }
};

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    setupFileUpload();
    setupContactForm();
    initializeFAQ();
});

// Show contact form for specific type
function showContactForm(type) {
    const config = contactTypes[type];
    if (!config) return;
    
    // Update form header
    document.getElementById('formTitle').textContent = config.title;
    document.getElementById('formSubtitle').textContent = config.subtitle;
    document.getElementById('contactType').value = type;
    
    // Set default priority
    document.getElementById('priority').value = config.priority;
    
    // Show form section
    document.getElementById('contactFormSection').style.display = 'block';
    
    // Hide options section
    document.querySelector('.contact-options').style.display = 'none';
    
    // Scroll to form
    document.getElementById('contactFormSection').scrollIntoView({ behavior: 'smooth' });
    
    // Focus on first field
    setTimeout(() => {
        document.getElementById('name').focus();
    }, 500);
}

// Hide contact form
function hideContactForm() {
    document.getElementById('contactFormSection').style.display = 'none';
    document.querySelector('.contact-options').style.display = 'block';
    
    // Reset form
    resetContactForm();
}

// Setup file upload functionality
function setupFileUpload() {
    const uploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    
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
        handleFiles(files);
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });
}

// Handle file selection
function handleFiles(files) {
    const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'image/png', 'image/jpeg'];
    const maxSize = 5 * 1024 * 1024; // 5MB
    const fileList = document.getElementById('fileList');
    let validFiles = [];
    
    Array.from(files).forEach(file => {
        // Check file type
        if (!allowedTypes.includes(file.type)) {
            showNotification(`Invalid file type: ${file.name}. Only PDF, DOC, DOCX, PNG, and JPG files are allowed.`, 'error');
            return;
        }
        
        // Check file size
        if (file.size > maxSize) {
            showNotification(`File too large: ${file.name}. Maximum size is 5MB.`, 'error');
            return;
        }
        
        validFiles.push(file);
    });
    
    if (validFiles.length > 0) {
        validFiles.forEach(file => addFileToList(file));
        showNotification(`${validFiles.length} file(s) added successfully`, 'success');
    }
}

// Add file to list
function addFileToList(file) {
    const fileList = document.getElementById('fileList');
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const fileId = Date.now() + Math.random();
    fileItem.dataset.fileId = fileId;
    
    // Create file icon based on type
    let icon = 'fa-file';
    if (file.type.includes('pdf')) icon = 'fa-file-pdf';
    else if (file.type.includes('word')) icon = 'fa-file-word';
    else if (file.type.includes('image')) icon = 'fa-file-image';
    
    fileItem.innerHTML = `
        <div class="file-info">
            <i class="fas ${icon}"></i>
            <div class="file-details">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
        </div>
        <button class="remove-file" onclick="removeFile('${fileId}')">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Store file data
    fileItem.dataset.fileName = file.name;
    fileItem.dataset.fileSize = file.size;
    fileItem.dataset.fileType = file.type;
    
    fileList.appendChild(fileItem);
}

// Remove file from list
function removeFile(fileId) {
    const fileItem = document.querySelector(`[data-file-id="${fileId}"]`);
    if (fileItem) {
        fileItem.remove();
        showNotification('File removed', 'info');
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Setup contact form submission
function setupContactForm() {
    const form = document.getElementById('contactForm');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate form
        if (!validateContactForm()) {
            return;
        }
        
        // Show loading state
        const submitBtn = form.querySelector('.submit-btn');
        const originalContent = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
        submitBtn.disabled = true;
        
        // Collect form data
        const formData = new FormData(form);
        const contactData = {
            type: formData.get('type'),
            name: formData.get('name'),
            email: formData.get('email'),
            organization: formData.get('organization'),
            subject: formData.get('subject'),
            priority: formData.get('priority'),
            message: formData.get('message'),
            timestamp: new Date().toISOString(),
            id: Date.now()
        };
        
        // Collect attached files
        const fileItems = document.querySelectorAll('.file-item');
        contactData.attachments = Array.from(fileItems).map(item => ({
            name: item.dataset.fileName,
            size: item.dataset.fileSize,
            type: item.dataset.fileType
        }));
        
        // Simulate submission
        setTimeout(() => {
            // Save contact request
            saveContactRequest(contactData);
            
            // Show success message
            showContactSuccess(contactData);
            
            // Reset form
            resetContactForm();
            
            // Restore button
            submitBtn.innerHTML = originalContent;
            submitBtn.disabled = false;
        }, 2000);
    });
}

// Validate contact form
function validateContactForm() {
    const form = document.getElementById('contactForm');
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    // Validate email
    const emailField = document.getElementById('email');
    if (emailField.value && !isValidEmail(emailField.value)) {
        showFieldError(emailField, 'Please enter a valid email address');
        isValid = false;
    }
    
    // Validate message length
    const messageField = document.getElementById('message');
    if (messageField.value.length < 10) {
        showFieldError(messageField, 'Please provide at least 10 characters');
        isValid = false;
    }
    
    return isValid;
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

// Save contact request
function saveContactRequest(contactData) {
    const existingRequests = JSON.parse(localStorage.getItem('chimera_contact_requests') || '[]');
    existingRequests.unshift(contactData);
    
    // Keep only last 100 requests
    if (existingRequests.length > 100) {
        existingRequests.splice(100);
    }
    
    localStorage.setItem('chimera_contact_requests', JSON.stringify(existingRequests));
}

// Show contact success message
function showContactSuccess(contactData) {
    // Hide form
    document.getElementById('contactFormSection').style.display = 'none';
    
    // Show success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = `
        <div class="success-content">
            <div class="success-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h2>Message Sent Successfully!</h2>
            <p>Thank you for contacting us. We've received your message and will get back to you within 24 hours.</p>
            <div class="contact-summary">
                <h3>Contact Information:</h3>
                <div class="summary-item">
                    <strong>Ticket ID:</strong> #${contactData.id}
                </div>
                <div class="summary-item">
                    <strong>Type:</strong> ${contactTypes[contactData.type].title}
                </div>
                <div class="summary-item">
                    <strong>Priority:</strong> ${contactData.priority}
                </div>
                <div class="summary-item">
                    <strong>Email:</strong> ${contactData.email}
                </div>
            </div>
            <div class="success-actions">
                <button onclick="submitAnotherMessage()" class="btn btn-primary">
                    <i class="fas fa-plus"></i>
                    Send Another Message
                </button>
                <a href="index.html" class="btn btn-secondary">
                    <i class="fas fa-home"></i>
                    Return Home
                </a>
            </div>
        </div>
    `;
    
    // Insert after header
    const header = document.querySelector('.contact-header');
    header.parentNode.insertBefore(successDiv, header.nextSibling);
    
    // Scroll to success message
    successDiv.scrollIntoView({ behavior: 'smooth' });
}

// Submit another message
function submitAnotherMessage() {
    // Remove success message
    const successMessage = document.querySelector('.success-message');
    if (successMessage) {
        successMessage.remove();
    }
    
    // Show contact options
    document.querySelector('.contact-options').style.display = 'block';
    
    // Scroll to options
    document.querySelector('.contact-options').scrollIntoView({ behavior: 'smooth' });
}

// Reset contact form
function resetContactForm() {
    const form = document.getElementById('contactForm');
    form.reset();
    
    // Clear all field errors
    const inputs = form.querySelectorAll('input, textarea, select');
    inputs.forEach(input => clearFieldError(input));
    
    // Clear file list
    document.getElementById('fileList').innerHTML = '';
}

// Initialize FAQ
function initializeFAQ() {
    // Set up FAQ categories
    const categories = ['general', 'scanner', 'security', 'technical'];
    categories.forEach(category => {
        const faqItems = document.querySelectorAll(`#${category}FAQ .faq-item`);
        faqItems.forEach((item, index) => {
            // Auto-expand first item in each category
            if (index === 0) {
                item.classList.add('expanded');
            }
        });
    });
}

// Show FAQ category
function showFAQCategory(category) {
    // Update tab buttons
    document.querySelectorAll('.category-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update FAQ content
    document.querySelectorAll('.faq-category').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(category + 'FAQ').classList.add('active');
}

// Toggle FAQ item
function toggleFAQ(element) {
    const faqItem = element.parentElement;
    const isExpanded = faqItem.classList.contains('expanded');
    
    // Close all other items in the same category
    const category = faqItem.closest('.faq-category');
    category.querySelectorAll('.faq-item').forEach(item => {
        item.classList.remove('expanded');
    });
    
    // Toggle current item
    if (!isExpanded) {
        faqItem.classList.add('expanded');
    }
}

// Show privacy policy
function showPrivacyPolicy() {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'privacy-modal';
    modal.innerHTML = `
        <div class="modal-overlay" onclick="closePrivacyModal()"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h2>Privacy Policy</h2>
                <button class="close-btn" onclick="closePrivacyModal()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <h3>Data Collection and Use</h3>
                <p>We collect and process personal data only when necessary to provide our services. This includes contact information you provide when reaching out to us, and scan data you submit for analysis.</p>
                
                <h3>Data Protection</h3>
                <p>All personal data is encrypted and stored securely. We implement appropriate technical and organizational measures to protect your information from unauthorized access, alteration, disclosure, or destruction.</p>
                
                <h3>Data Sharing</h3>
                <p>We do not sell, rent, or share your personal data with third parties for marketing purposes. We may share data with service providers only as necessary to provide our services, and only under strict confidentiality agreements.</p>
                
                <h3>Your Rights</h3>
                <p>You have the right to access, correct, or delete your personal data. You may also object to or restrict processing of your data in certain circumstances.</p>
                
                <h3>Contact Information</h3>
                <p>If you have questions about this privacy policy or how we handle your data, please contact us at privacy@chimera.com.</p>
            </div>
            <div class="modal-footer">
                <button class="btn btn-primary" onclick="closePrivacyModal()">I Understand</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';
}

// Close privacy modal
function closePrivacyModal() {
    const modal = document.querySelector('.privacy-modal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}

// Show coming soon message
function showComingSoon() {
    showNotification('This feature is coming soon! Stay tuned for updates.', 'info');
}

// Utility function
function isValidEmail(string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(string);
}

// Add contact page specific CSS
const contactCSS = `
.contact-main {
    padding: 2rem 0 4rem;
    min-height: calc(100vh - 200px);
}

.contact-header {
    text-align: center;
    margin-bottom: 3rem;
}

.contact-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.contact-header p {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
}

.contact-options {
    margin-bottom: 4rem;
}

.options-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
}

.option-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2.5rem;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: var(--transition);
    text-align: center;
}

.option-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-color);
}

.option-icon {
    width: 80px;
    height: 80px;
    background: var(--gradient-primary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto 2rem;
    color: white;
    font-size: 2rem;
}

.option-card h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.option-card p {
    color: var(--text-secondary);
    margin-bottom: 2rem;
    line-height: 1.6;
}

.option-features {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    text-align: left;
}

.option-features span {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.option-features i {
    color: var(--success-color);
    font-size: 0.8rem;
}

.contact-form-section {
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

.form-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border-color);
}

.back-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 4px;
    transition: var(--transition);
}

.back-btn:hover {
    background: var(--bg-tertiary);
    color: var(--primary-color);
}

.form-header h2 {
    font-size: 2rem;
    color: var(--text-primary);
    margin: 0;
}

.form-header p {
    color: var(--text-secondary);
    margin: 0;
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
    min-height: 150px;
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

.file-upload-area {
    border: 2px dashed var(--border-color);
    border-radius: var(--border-radius);
    padding: 3rem;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
}

.file-upload-area:hover, .file-upload-area.drag-over {
    border-color: var(--primary-color);
    background: rgba(0, 102, 204, 0.05);
}

.upload-content i {
    font-size: 3rem;
    color: var(--primary-color);
    margin-bottom: 1rem;
}

.upload-content p {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
    font-weight: 500;
}

.upload-content small {
    color: var(--text-muted);
}

.file-list {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.file-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
}

.file-info {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.file-info i {
    color: var(--primary-color);
    font-size: 1.5rem;
}

.file-details {
    display: flex;
    flex-direction: column;
}

.file-name {
    font-weight: 500;
    color: var(--text-primary);
}

.file-size {
    font-size: 0.85rem;
    color: var(--text-muted);
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
    transition: var(--transition);
}

.remove-file:hover {
    background: #c82333;
}

.checkbox-label {
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    color: var(--text-primary);
    line-height: 1.5;
}

.checkbox-label input[type="checkbox"] {
    margin: 0;
}

.checkbox-label a {
    color: var(--primary-color);
    text-decoration: none;
}

.checkbox-label a:hover {
    text-decoration: underline;
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
    flex-shrink: 0;
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

.faq-section {
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

.faq-categories {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
}

.category-tabs {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid var(--border-color);
    overflow-x: auto;
}

.category-tab {
    background: none;
    border: none;
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
    position: relative;
    transition: var(--transition);
    white-space: nowrap;
}

.category-tab.active {
    color: var(--primary-color);
}

.category-tab.active::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--primary-color);
}

.faq-content {
    min-height: 300px;
}

.faq-category {
    display: none;
}

.faq-category.active {
    display: block;
}

.faq-item {
    border-bottom: 1px solid var(--border-color);
    transition: var(--transition);
}

.faq-item:last-child {
    border-bottom: none;
}

.faq-item.expanded {
    background: var(--bg-tertiary);
}

.faq-question {
    padding: 1.5rem;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: var(--transition);
}

.faq-question:hover {
    background: var(--bg-tertiary);
}

.faq-question h4 {
    margin: 0;
    color: var(--text-primary);
    font-weight: 600;
}

.faq-question i {
    color: var(--text-secondary);
    transition: var(--transition);
}

.faq-item.expanded .faq-question i {
    transform: rotate(180deg);
}

.faq-answer {
    padding: 0 1.5rem;
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease, padding 0.3s ease;
}

.faq-item.expanded .faq-answer {
    padding: 0 1.5rem 1.5rem;
    max-height: 500px;
}

.faq-answer p {
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
}

.quick-help {
    margin-bottom: 4rem;
}

.help-header {
    text-align: center;
    margin-bottom: 3rem;
}

.help-header h2 {
    font-size: 2rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.help-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.help-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

.help-card {
    background: var(--bg-secondary);
    border-radius: var(--border-radius);
    padding: 2rem;
    border: 1px solid var(--border-color);
    text-align: center;
    transition: var(--transition);
}

.help-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-md);
}

.help-icon {
    width: 60px;
    height: 60px;
    background: var(--gradient-secondary);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto 1.5rem;
    color: white;
    font-size: 1.5rem;
}

.help-card h3 {
    font-size: 1.25rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}

.help-card p {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

.help-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--primary-color);
    text-decoration: none;
    font-weight: 600;
    transition: var(--transition);
}

.help-link:hover {
    gap: 0.75rem;
}

.emergency-contact {
    margin-bottom: 3rem;
}

.emergency-card {
    background: linear-gradient(135deg, #dc3545, #c82333);
    border-radius: var(--border-radius);
    padding: 3rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 2rem;
    box-shadow: var(--shadow-lg);
}

.emergency-icon {
    width: 80px;
    height: 80px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 2.5rem;
    flex-shrink: 0;
}

.emergency-content h3 {
    font-size: 1.75rem;
    margin-bottom: 1rem;
}

.emergency-content p {
    margin-bottom: 2rem;
    opacity: 0.9;
    line-height: 1.6;
}

.emergency-actions {
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.btn-danger {
    background: white;
    color: var(--danger-color);
    border: none;
    padding: 1rem 2rem;
    border-radius: var(--border-radius);
    font-weight: 600;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    transition: var(--transition);
}

.btn-danger:hover {
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-2px);
}

.emergency-time {
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
}

.privacy-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
}

.modal-content {
    background: var(--bg-primary);
    border-radius: var(--border-radius);
    max-width: 600px;
    max-height: 80vh;
    width: 90%;
    overflow: hidden;
    position: relative;
    z-index: 1;
    box-shadow: var(--shadow-lg);
}

.modal-header {
    padding: 2rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h2 {
    margin: 0;
    color: var(--text-primary);
}

.close-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: 4px;
    transition: var(--transition);
}

.close-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.modal-body {
    padding: 2rem;
    max-height: 60vh;
    overflow-y: auto;
}

.modal-body h3 {
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
}

.modal-body h3:first-child {
    margin-top: 0;
}

.modal-body p {
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 1rem;
}

.modal-footer {
    padding: 2rem;
    border-top: 1px solid var(--border-color);
    text-align: right;
}

@media (max-width: 768px) {
    .contact-header h1 {
        font-size: 2rem;
    }
    
    .options-grid {
        grid-template-columns: 1fr;
    }
    
    .form-card {
        padding: 2rem 1.5rem;
    }
    
    .form-header {
        flex-direction: column;
        text-align: center;
        gap: 1rem;
    }
    
    .category-tabs {
        flex-wrap: nowrap;
        overflow-x: auto;
    }
    
    .help-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .emergency-card {
        flex-direction: column;
        text-align: center;
    }
    
    .emergency-actions {
        flex-direction: column;
        align-items: center;
    }
    
    .form-actions {
        flex-direction: column;
    }
    
    .form-actions .btn {
        width: 100%;
    }
    
    .modal-content {
        width: 95%;
        margin: 1rem;
    }
    
    .modal-header,
    .modal-body,
    .modal-footer {
        padding: 1.5rem;
    }
}
`;

// Inject contact page CSS
const contactStyleSheet = document.createElement('style');
contactStyleSheet.textContent = contactCSS;
document.head.appendChild(contactStyleSheet);
