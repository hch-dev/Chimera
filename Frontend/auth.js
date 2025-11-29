// Authentication JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Login Form Handler
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');

    // Password Toggle Functions
    function setupPasswordToggle(toggleId, inputId) {
        const toggle = document.getElementById(toggleId);
        const input = document.getElementById(inputId);

        if (toggle && input) {
            toggle.addEventListener('click', function() {
                const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
                input.setAttribute('type', type);

                const icon = toggle.querySelector('i');
                icon.classList.toggle('fa-eye');
                icon.classList.toggle('fa-eye-slash');
            });
        }
    }

    // Setup password toggles
    setupPasswordToggle('passwordToggle', 'password');
    setupPasswordToggle('confirmPasswordToggle', 'confirmPassword');

    // Login Form Validation
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();

            if (validateLoginForm()) {
                // Simulate login
                const email = document.getElementById('email').value;
                const rememberMe = document.getElementById('rememberMe').checked;

                // Show loading state
                const submitBtn = loginForm.querySelector('.auth-btn');
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...';
                submitBtn.disabled = true;

                // Simulate API call
                setTimeout(() => {
                    // Store login session
                    const userData = {
                        email: email,
                        loginTime: new Date().toISOString(),
                           rememberMe: rememberMe
                    };

                    if (rememberMe) {
                        localStorage.setItem('chimeraUser', JSON.stringify(userData));
                    } else {
                        sessionStorage.setItem('chimeraUser', JSON.stringify(userData));
                    }

                    showNotification('Login successful! Redirecting to dashboard...', 'success');

                    setTimeout(() => {
                        window.location.href = 'index.html';
                    }, 2000);
                }, 1500);
            }
        });
    }

    // Signup Form Validation
    if (signupForm) {
        // Password strength checker
        const passwordInput = document.getElementById('password');

        if (passwordInput) {
            passwordInput.addEventListener('input', function() {
                const password = this.value;
                const strength = calculatePasswordStrength(password);
                updatePasswordStrength(strength, password); // Pass password to check if empty
            });
        }

        signupForm.addEventListener('submit', function(e) {
            e.preventDefault();

            if (validateSignupForm()) {
                // Show loading state
                const submitBtn = signupForm.querySelector('.auth-btn');
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
                submitBtn.disabled = true;

                // Get form data
                const formData = {
                    fullName: document.getElementById('fullName').value,
                                    email: document.getElementById('email').value,
                                    marketing: document.getElementById('terms').checked, // Assuming terms checkbox used for consent
                                    signupTime: new Date().toISOString()
                };

                // Simulate API call
                setTimeout(() => {
                    showNotification('Account created successfully! Please check your email to verify.', 'success');

                    // Store user data
                    sessionStorage.setItem('chimeraNewUser', JSON.stringify(formData));

                    setTimeout(() => {
                        window.location.href = 'login.html';
                    }, 3000);
                }, 2000);
            }
        });
    }

    // Social Login Handlers
    document.querySelectorAll('.social-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const provider = this.classList.contains('google') ? 'Google' : 'Microsoft';
            showNotification(`Redirecting to ${provider} login...`, 'info');

            // Simulate OAuth flow
            setTimeout(() => {
                showNotification(`${provider} authentication would be implemented here`, 'info');
            }, 1500);
        });
    });

    // Forgot Password Handler
    const forgotPasswordLink = document.querySelector('.forgot-password');
    if (forgotPasswordLink) {
        forgotPasswordLink.addEventListener('click', function(e) {
            e.preventDefault();

            const email = document.getElementById('email').value;
            if (!email) {
                showError('emailError', 'Please enter your email address first');
                return;
            }

            showNotification('Password reset link sent to your email!', 'success');
        });
    }
});

// Login Form Validation
function validateLoginForm() {
    let isValid = true;

    // Clear previous errors
    clearAllErrors();

    // Email validation
    const email = document.getElementById('email').value;
    if (!email) {
        showError('emailError', 'Email is required');
        isValid = false;
    } else if (!isValidEmail(email)) {
        showError('emailError', 'Please enter a valid email address');
        isValid = false;
    }

    // Password validation
    const password = document.getElementById('password').value;
    if (!password) {
        showError('passwordError', 'Password is required');
        isValid = false;
    } else if (password.length < 6) {
        showError('passwordError', 'Password must be at least 6 characters');
        isValid = false;
    }

    return isValid;
}

// Signup Form Validation
function validateSignupForm() {
    let isValid = true;

    // Clear previous errors
    clearAllErrors();

    // Full Name validation
    const fullName = document.getElementById('fullName').value;
    if (!fullName) {
        showError('fullNameError', 'Full name is required');
        isValid = false;
    } else if (fullName.length < 2) {
        showError('fullNameError', 'Name must be at least 2 characters');
        isValid = false;
    }

    // Email validation
    const email = document.getElementById('email').value;
    if (!email) {
        showError('emailError', 'Email is required');
        isValid = false;
    } else if (!isValidEmail(email)) {
        showError('emailError', 'Please enter a valid email address');
        isValid = false;
    }

    // Password validation
    const password = document.getElementById('password').value;
    if (!password) {
        showError('passwordError', 'Password is required');
        isValid = false;
    } else if (password.length < 8) {
        showError('passwordError', 'Password must be at least 8 characters');
        isValid = false;
    } else if (!isStrongPassword(password)) {
        showError('passwordError', 'Password must contain uppercase, lowercase, and number');
        isValid = false;
    }

    // Confirm Password validation
    const confirmPassword = document.getElementById('confirmPassword').value;
    if (!confirmPassword) {
        showError('confirmPasswordError', 'Please confirm your password');
        isValid = false;
    } else if (password !== confirmPassword) {
        showError('confirmPasswordError', 'Passwords do not match');
        isValid = false;
    }

    // Terms validation
    const terms = document.getElementById('terms').checked;
    if (!terms) {
        showError('termsError', 'You must agree to the terms and conditions');
        isValid = false;
    }

    return isValid;
}

// Password Strength Calculator
function calculatePasswordStrength(password) {
    if (!password) return 0; // Explicitly return 0 if empty

    let strength = 0;

    if (password.length >= 8) strength++;
    if (password.length >= 12) strength++;
    if (/[a-z]/.test(password)) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^a-zA-Z0-9]/.test(password)) strength++;

    // Normalize score to 1-4 range for non-empty passwords
    if (strength < 2) return 1;
    if (strength < 4) return 2;
    if (strength < 6) return 3;
    return 4;
}

// Update Password Strength Display
function updatePasswordStrength(strength, password) {
    const strengthBar = document.querySelector('.strength-bar');
    const strengthText = document.querySelector('.strength-text');

    if (!strengthBar || !strengthText) return;

    // If password is empty, reset everything
    if (!password || password.length === 0) {
        strengthBar.style.width = '0%';
        strengthBar.style.backgroundColor = 'transparent';
        strengthText.textContent = 'Password strength';
        strengthText.style.color = 'var(--text-muted)'; // Reset color
        return;
    }

    const strengthLevels = [
        { width: '25%', color: '#dc3545', text: 'Weak' },
        { width: '50%', color: '#ffc107', text: 'Fair' },
        { width: '75%', color: '#fd7e14', text: 'Good' },
        { width: '100%', color: '#28a745', text: 'Strong' }
    ];

    // Strength is 1-based index in the array
    const level = strengthLevels[strength - 1] || strengthLevels[0];

    strengthBar.style.width = level.width;
    strengthBar.style.backgroundColor = level.color;
    strengthText.textContent = `Password strength: ${level.text}`;
    strengthText.style.color = level.color;
}

// Utility Functions
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function isStrongPassword(password) {
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);

    return hasUpperCase && hasLowerCase && hasNumbers;
}

function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

function clearAllErrors() {
    document.querySelectorAll('.form-error').forEach(error => {
        error.textContent = '';
        error.style.display = 'none';
    });
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
    <div class="notification-content">
    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
    <span>${message}</span>
    <button class="notification-close">&times;</button>
    </div>
    `;

    // Add to page
    document.body.appendChild(notification);

    // Show notification
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    // Close button handler
    const closeBtn = notification.querySelector('.notification-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }
    }, 5000);
}s
