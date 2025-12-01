// content-script.js

import { analyze } from "../api/detector.js"; // Adjust path accordingly

let passiveUrlScanEnabled = false;
let passiveEmailScanEnabled = false;

async function passiveScanUrl() {
    if (!passiveUrlScanEnabled) return;

    const url = window.location.href;
    const result = await analyze({ type: "url", text: url, isActive: false });
    console.log("Passive URL Scan Result:", result);
    // Optionally send message back to popup or background
}

async function passiveScanEmail() {
    if (!passiveEmailScanEnabled) return;

    // Naive extraction: look for email-like text on the page (improve as needed)
    const bodyText = document.body.innerText || "";
    const emails = bodyText.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}/g) || [];

    // If no emails found, you might want to analyze the whole body or specific areas
    const textToAnalyze = emails.length > 0 ? emails.join("\n") : bodyText;

    const result = await analyze({ type: "email", text: textToAnalyze, isActive: false });
    console.log("Passive Email Scan Result:", result);
    // Optionally send message back to popup or background
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "enable_passive_url_scan") {
        passiveUrlScanEnabled = true;
        passiveScanUrl();
        sendResponse({ status: "Passive URL scan enabled" });
    } 
    else if (message.action === "disable_passive_url_scan") {
        passiveUrlScanEnabled = false;
        sendResponse({ status: "Passive URL scan disabled" });
    }
    else if (message.action === "enable_passive_email_scan") {
        passiveEmailScanEnabled = true;
        passiveScanEmail();
        sendResponse({ status: "Passive Email scan enabled" });
    }
    else if (message.action === "disable_passive_email_scan") {
        passiveEmailScanEnabled = false;
        sendResponse({ status: "Passive Email scan disabled" });
    }
});

// Automatically trigger scans on page load if enabled
window.addEventListener('load', () => {
    if (passiveUrlScanEnabled) {
        passiveScanUrl();
    }
    if (passiveEmailScanEnabled) {
        passiveScanEmail();
    }
});
