// Chimera Phishing Detector - Main JavaScript

// Theme Toggle
const themeToggle = document.getElementById('themeToggle');
const htmlElement = document.documentElement;

// Check for saved theme preference or default to light
const currentTheme = localStorage.getItem('theme') || 'light';
htmlElement.setAttribute('data-theme', currentTheme);

// Update theme toggle icon
function updateThemeIcon() {
    const icon = themeToggle.querySelector('i');
    if (htmlElement.getAttribute('data-theme') === 'dark') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// Quick Scan Dropdown Toggle
function toggleQuickScan() {
    const content = document.getElementById('quickScanContent');
    const icon = document.getElementById('dropdownIcon');
    
    content.classList.toggle('expanded');
    icon.classList.toggle('rotated');
}

// Initialize theme icon
updateThemeIcon();

// Theme toggle event listener
themeToggle.addEventListener('click', () => {
    const currentTheme = htmlElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    htmlElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon();
});

// Mobile Menu Toggle
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const navLinks = document.querySelector('.nav-links');

mobileMenuToggle.addEventListener('click', () => {
    navLinks.classList.toggle('active');
    const icon = mobileMenuToggle.querySelector('i');
    icon.classList.toggle('fa-bars');
    icon.classList.toggle('fa-times');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', () => {
        navLinks.classList.remove('active');
        const icon = mobileMenuToggle.querySelector('i');
        icon.classList.add('fa-bars');
        icon.classList.remove('fa-times');
    });
});

// Quick Scan Function
function quickScan() {
    const input = document.getElementById('quickScanInput');
    const url = input.value.trim();
    
    if (!url) {
        showNotification('Please enter a URL to scan', 'warning');
        return;
    }
    
    // Basic URL validation
    if (!isValidURL(url)) {
        showNotification('Please enter a valid URL', 'error');
        return;
    }
    
    // Show loading state
    const scanBtn = document.querySelector('.scan-btn');
    const originalContent = scanBtn.innerHTML;
    scanBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
    scanBtn.disabled = true;
    
    // Simulate scanning process
    setTimeout(() => {
        // Store the URL in sessionStorage for the scan page
        sessionStorage.setItem('scanUrl', url);
        
        // Redirect to scan page
        window.location.href = 'scan.html';
    }, 1500);
}

// URL Validation
function isValidURL(string) {
    try {
        const url = new URL(string);
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (_) {
        return false;
    }
}

// Notification System
function showNotification(message, type = 'info') {
    // Remove existing notification if any
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove functionality disabled - notification stays until manually closed
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add scroll effect to header
window.addEventListener('scroll', () => {
    const header = document.querySelector('.header');
    if (window.scrollY > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// Add CSS for header scroll effect
const headerScrollCSS = `
.header.scrolled {
    box-shadow: var(--shadow-md);
}
`;

// Add CSS for notifications
const notificationCSS = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    max-width: 400px;
    animation: slideIn 0.3s ease-out;
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
}

.notification-success .notification-content {
    background: rgba(40, 167, 69, 0.9);
    color: white;
}

.notification-error .notification-content {
    background: rgba(220, 53, 69, 0.9);
    color: white;
}

.notification-warning .notification-content {
    background: rgba(255, 193, 7, 0.9);
    color: #212529;
}

.notification-info .notification-content {
    background: rgba(0, 102, 204, 0.9);
    color: white;
}

.notification-close {
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    opacity: 0.7;
    transition: opacity 0.2s;
}

.notification-close:hover {
    opacity: 1;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
`;

// Inject notification CSS
const styleSheet = document.createElement('style');
styleSheet.textContent = headerScrollCSS + notificationCSS;
document.head.appendChild(styleSheet);

// Page load animations
document.addEventListener('DOMContentLoaded', () => {
    // Add fade-in animation to elements
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe feature cards and other elements
    document.querySelectorAll('.feature-card, .step, .awareness-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Format date utility
function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

// Copy to clipboard utility
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

// Initialize tooltips
function initTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', (e) => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = e.target.getAttribute('data-tooltip');
            document.body.appendChild(tooltip);
            
            const rect = e.target.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
        });
        
        element.addEventListener('mouseleave', () => {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) tooltip.remove();
        });
    });
}

// Call tooltips initialization
initTooltips();

// Smooth page loading animations
function setupPageTransitions() {
    // Add click handlers to navigation links
    const navLinks = document.querySelectorAll('a[href$=".html"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Skip if it's the current page or external link
            if (this.getAttribute('href') === '#' || this.classList.contains('external')) {
                return;
            }
            
            e.preventDefault();
            
            // Show loading overlay
            showPageLoader();
            
            // Navigate to new page after a short delay
            setTimeout(() => {
                window.location.href = this.getAttribute('href');
            }, 300);
        });
    });
}

// Show page loading overlay
function showPageLoader() {
    const loader = document.createElement('div');
    loader.className = 'page-loader';
    loader.innerHTML = `
        <div class="loader-overlay">
            <div class="loader-content">
                <div class="loader-logo">
                    <span>Chimera</span>
                </div>
                <div class="loader-spinner">
                    <div class="spinner"></div>
                </div>
                <div class="loader-text">
                    <span class="loading-dot">.</span>
                    <span class="loading-dot">.</span>
                    <span class="loading-dot">.</span>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(loader);
    
    // Add fade-in animation
    setTimeout(() => {
        loader.classList.add('active');
    }, 10);
}

// Initialize page transitions
document.addEventListener('DOMContentLoaded', () => {
    setupPageTransitions();
    
    // Hide loader when page is fully loaded
    window.addEventListener('load', () => {
        const existingLoader = document.querySelector('.page-loader');
        if (existingLoader) {
            existingLoader.classList.add('fade-out');
            setTimeout(() => {
                existingLoader.remove();
            }, 500);
        }
    });
});

// Add page loader CSS
const pageLoaderCSS = `
.page-loader {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 99999;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s ease, visibility 0.3s ease;
}

.page-loader.active {
    opacity: 1;
    visibility: visible;
}

.page-loader.fade-out {
    opacity: 0;
    visibility: hidden;
}

.loader-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
}

.loader-content {
    text-align: center;
    color: white;
}

.loader-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 2rem;
    animation: pulse 2s infinite;
}

.loader-logo i {
    font-size: 3rem;
    animation: spin 2s linear infinite;
}

.loader-logo span {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
}

.loader-spinner {
    margin-bottom: 2rem;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-top: 4px solid white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

.loader-text {
    font-size: 1.2rem;
    font-weight: 500;
    letter-spacing: 2px;
}

.loading-dot {
    display: inline-block;
    animation: loadingDot 1.5s infinite;
}

.loading-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.loading-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
}

@keyframes loadingDot {
    0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-10px); }
}

/* Dark theme loader */
[data-theme="dark"] .loader-overlay {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}
`;

// Inject page loader CSS
const pageLoaderStyleSheet = document.createElement('style');
pageLoaderStyleSheet.textContent = pageLoaderCSS;
document.head.appendChild(pageLoaderStyleSheet);
