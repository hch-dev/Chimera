document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generateBtn');
    const emailCountInput = document.getElementById('emailCount');
    const errorMessage = document.getElementById('errorMessage');
    const resultsSection = document.getElementById('resultsSection');
    const emailResults = document.getElementById('emailResults');

    // Phishing email templates
    const phishingTemplates = [
        {
            subject: "Urgent: Account Verification Required",
            body: "Dear valued customer, your account will be suspended unless you verify your identity immediately. Click here: http://fake-verification.com"
        },
        {
            subject: "You've Won $1,000,000!!!",
            body: "Congratulations! You are our lucky winner. Claim your prize now by providing your banking details at: http://winner-claim.com"
        },
        {
            subject: "PayPal Security Alert",
            body: "We detected suspicious activity on your PayPal account. Please login to secure your account: http://paypal-security-update.com"
        },
        {
            subject: "Netflix Subscription Failed",
            body: "Your Netflix subscription payment failed. Update your payment information to continue service: http://netflix-billing.com"
        },
        {
            subject: "Amazon Order Confirmation",
            body: "Your order #12345 has been shipped. Track your package here: http://amazon-tracking-delivery.com"
        },
        {
            subject: "Microsoft Account Recovery",
            body: "Someone tried to access your Microsoft account. Verify your identity to protect your account: http://microsoft-secure.com"
        },
        {
            subject: "Bank of America Fraud Alert",
            body: "We've detected fraudulent charges on your account. Please review and confirm transactions: http://bank-america-secure.com"
        },
        {
            subject: "LinkedIn Job Offer",
            body: "You've been selected for a high-paying position! Accept your offer now: http://linkedin-careers-premium.com"
        },
        {
            subject: "COVID-19 Relief Payment",
            body: "You are eligible for government COVID-19 relief funds. Claim your payment: http://covid-relief-gov.com"
        },
        {
            subject: "Dropbox Storage Full",
            body: "Your Dropbox storage is full. Upgrade now to avoid losing files: http://dropbox-premium-upgrade.com"
        },
        {
            subject: "Apple ID Locked",
            body: "Your Apple ID has been locked for security reasons. Unlock your account: http://apple-id-recovery.com"
        },
        {
            subject: "Facebook Password Reset",
            body: "Someone requested a password reset for your Facebook account. Cancel or confirm: http://facebook-security.com"
        },
        {
            subject: "Google Drive Shared Document",
            body: "John shared an important document with you. View it now: http://google-drive-shared.com"
        },
        {
            subject: "Instagram Account Verification",
            body: "Your account needs verification to avoid suspension. Verify now: http://instagram-verify.com"
        },
        {
            subject: "WhatsApp Security Update",
            body: "Update your WhatsApp security settings to protect your messages: http://whatsapp-secure-update.com"
        },
        {
            subject: "Twitter Account Suspension",
            body: "Your Twitter account has been flagged for violation. Appeal the decision: http://twitter-appeal.com"
        },
        {
            subject: "YouTube Monetization Available",
            body: "Your channel is now eligible for monetization! Claim your earnings: http://youtube-partner.com"
        },
        {
            subject: "Gmail Storage Limit Reached",
            body: "Your Gmail storage is full. Upgrade to Google Workspace: http://gmail-premium.com"
        },
        {
            subject: "Zoom Meeting Invitation",
            body: "You're invited to an important meeting. Join now: http://zoom-meeting-urgent.com"
        }
    ];

    // Shuffle array to get random templates
    function shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // Generate phishing emails
    function generatePhishingEmails(count) {
        const shuffled = shuffleArray(phishingTemplates);
        return shuffled.slice(0, count);
    }

    // Display results
    function displayResults(emails) {
        emailResults.innerHTML = '';
        
        emails.forEach((email, index) => {
            const emailDiv = document.createElement('div');
            emailDiv.className = 'email-item';
            emailDiv.innerHTML = `
                <strong>Email ${index + 1}:</strong> [Subject: ${email.subject}] [Body: ${email.body}]
            `;
            emailResults.appendChild(emailDiv);
            
            // Add separator except for last email
            if (index < emails.length - 1) {
                const separator = document.createElement('div');
                separator.className = 'email-separator';
                emailResults.appendChild(separator);
            }
        });
        
        resultsSection.style.display = 'block';
    }

    // Clear results
    function clearResults() {
        resultsSection.style.display = 'none';
        emailResults.innerHTML = '';
        errorMessage.style.display = 'none';
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    // Generate button click handler
    generateBtn.addEventListener('click', function() {
        const count = parseInt(emailCountInput.value);
        
        // Clear previous results and errors
        clearResults();
        
        // Validation
        if (isNaN(count) || count < 1) {
            showError('Please enter a valid number greater than 0');
            return;
        }
        
        if (count > 10) {
            showError('Maximum limit is 10');
            return;
        }
        
        // Generate and display emails
        const emails = generatePhishingEmails(count);
        displayResults(emails);
    });

    // Input field enter key handler
    emailCountInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            generateBtn.click();
        }
    });

    // Clear error when user starts typing
    emailCountInput.addEventListener('input', function() {
        if (errorMessage.style.display !== 'none') {
            errorMessage.style.display = 'none';
        }
    });
});
