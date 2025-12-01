// 1. MODULE IMPORT (MUST BE FIRST)
import { analyze } from "../api/detector.js";

// --- DOM READY ---
document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const statusToggle = document.getElementById('statusToggle');
    const extensionUI = document.getElementById('extension-ui');

    const themeToggle = document.getElementById('themeToggle');
    const themeStatusText = document.getElementById('themeStatusText');

    // URL scan mode toggle & related UI
    const urlModeToggle = document.getElementById('urlModeToggle');
    const urlModeText = document.getElementById('urlModeText');
    const urlActiveUI = document.getElementById('urlActiveUI');
    const scanActiveUrlBtn = document.getElementById('scanActiveUrlBtn');
    const activeUrlInput = document.getElementById('activeUrlInput');
    const urlActiveResult = document.getElementById('urlActiveResult');

    // Email scan mode toggle & UI
    const emailModeToggle = document.getElementById('emailModeToggle');
    const emailModeText = document.getElementById('emailModeText');
    const emailActiveUI = document.getElementById('emailActiveUI');
    const scanEmailActiveBtn = document.getElementById('scanEmailActiveBtn');
    const emailActiveInput = document.getElementById('emailActiveInput');
    const emailActiveResult = document.getElementById('emailActiveResult');

    // Notification area for messages (create a div in your popup.html with this ID for inline messages)
    const notificationArea = document.getElementById('notificationArea');

    function showNotification(msg) {
        if (notificationArea) {
            notificationArea.textContent = msg;
            notificationArea.style.display = 'block';
        } else {
            alert(msg);
        }
    }

    function clearNotification() {
        if (notificationArea) {
            notificationArea.textContent = '';
            notificationArea.style.display = 'none';
        }
    }

    // --- Extension Enable/Disable ---
    function updateExtensionUIVisibility() {
        extensionUI.style.display = statusToggle.checked ? 'block' : 'none';
        if (!statusToggle.checked) clearNotification();
    }

    chrome.storage.sync.get('isEnabled', (data) => {
        statusToggle.checked = data.isEnabled !== undefined ? data.isEnabled : true;
        updateExtensionUIVisibility();
    });

    statusToggle.addEventListener('change', () => {
        const enabled = statusToggle.checked;
        chrome.storage.sync.set({ isEnabled: enabled }, () => {
            console.log(`Extension ${enabled ? "enabled" : "disabled"}`);
            updateExtensionUIVisibility();
        });
    });

    // --- Theme Initialization ---
    function applyTheme(isDark) {
        if (isDark) {
            document.body.classList.add('dark-theme');
            themeStatusText.textContent = "Dark";
        } else {
            document.body.classList.remove('dark-theme');
            themeStatusText.textContent = "Light";
        }
    }

    chrome.storage.sync.get('theme', data => {
        let isDark;
        if (data.theme === 'dark') isDark = true;
        else if (data.theme === 'light') isDark = false;
        else isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

        applyTheme(isDark);
        themeToggle.checked = isDark;
    });

    themeToggle.addEventListener('change', () => {
        const isDark = themeToggle.checked;
        applyTheme(isDark);
        chrome.storage.sync.set({ theme: isDark ? 'dark' : 'light' });
    });

    // --- URL Mode Toggle ---
    function updateUrlModeUI() {
        const isActive = urlModeToggle.checked;
        urlModeText.textContent = isActive ? "Active" : "Passive";
        if (isActive) {
            urlActiveUI.style.display = "block";
            urlActiveResult.textContent = "";
        } else {
            urlActiveUI.style.display = "none";
            urlActiveResult.textContent = "Passive scanning active: scans occur automatically on page change.";
        }
        clearNotification();
    }

    urlModeToggle.addEventListener('change', updateUrlModeUI);
    updateUrlModeUI();

    // --- Email Mode Toggle ---
    function updateEmailModeUI() {
        const isActive = emailModeToggle.checked;
        emailModeText.textContent = isActive ? "Active" : "Passive";
        if (isActive) {
            emailActiveUI.style.display = "block";
            emailActiveResult.textContent = "";
        } else {
            emailActiveUI.style.display = "none";
            emailActiveResult.textContent = "Passive scanning active: scans occur automatically on page change.";
        }
        clearNotification();
    }

    emailModeToggle.addEventListener('change', updateEmailModeUI);
    updateEmailModeUI();

    // --- Active URL Scan ---
    scanActiveUrlBtn.addEventListener('click', async () => {
        clearNotification();
        const url = activeUrlInput.value.trim();
        if (!url) {
            showNotification("Please enter a URL for active scan.");
            return;
        }
        urlActiveResult.textContent = "Scanning...";
        try {
            const result = await analyze({ type: "url", text: url, isActive: true });
            urlActiveResult.textContent = `Active URL Scan Verdict: ${result.verdict.toUpperCase()} (confidence: ${result.confidence})`;
        } catch (err) {
            urlActiveResult.textContent = "Error during scan.";
            console.error(err);
        }
    });

    // --- Active Email Scan ---
    scanEmailActiveBtn.addEventListener('click', async () => {
        clearNotification();
        const text = emailActiveInput.value.trim();
        if (!text) {
            showNotification("Please enter some email text.");
            return;
        }
        emailActiveResult.textContent = "Scanning...";
        try {
            const result = await analyze({ type: "email", text: text, isActive: true });
            emailActiveResult.textContent = `Active Email Scan Verdict: ${result.verdict.toUpperCase()} (confidence: ${result.confidence})`;
        } catch (err) {
            emailActiveResult.textContent = "Error during scan.";
            console.error(err);
        }
    });

    /**
     * Robustly send message to content script in active tab.
     * Injects content script if not already injected, then retries.
     * Notifies user if page is restricted (like chrome://).
     */
    async function sendMessageToActiveTab(message) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab?.id) {
            console.warn("No active tab found.");
            showNotification("No active tab found.");
            return;
        }

        const url = tab.url || "";

        // Check for restricted/internal URLs and notify user
        if (url.startsWith("chrome://") || url.startsWith("chrome-extension://") || url.startsWith("file://")) {
            console.warn(`Cannot inject content script into restricted or internal pages: ${url}`);
            showNotification(`Extension cannot operate on restricted/internal page:\n${url}`);
            return;
        }

        function trySend() {
            return new Promise((resolve) => {
                chrome.tabs.sendMessage(tab.id, message, (response) => {
                    if (chrome.runtime.lastError) {
                        resolve({ error: chrome.runtime.lastError.message });
                    } else {
                        resolve({ response });
                    }
                });
            });
        }

        let result = await trySend();
        if (!result.error) {
            clearNotification();
            return result.response;
        }

        if (result.error.includes("Receiving end does not exist")) {
            try {
                await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ["content-script.js"]
                });
            } catch (e) {
                console.error("Content script injection failed:", e);
                showNotification("Failed to inject content script on this page:\n" + e.message);
                return null;
            }

            result = await trySend();
            if (!result.error) {
                clearNotification();
                return result.response;
            } else {
                console.error("Retry failed:", result.error);
                showNotification("Failed to communicate with content script after injection:\n" + result.error);
                return null;
            }
        }

        console.error("Message send failed:", result.error);
        showNotification("Unknown error communicating with active tab:\n" + result.error);
        return null;
    }

    // Update passive URL scan state based on toggle
    async function updatePassiveUrlScanState() {
        if (!urlModeToggle.checked) { // Passive mode active when toggle is unchecked
            await sendMessageToActiveTab({ action: "enable_passive_url_scan" });
        } else {
            await sendMessageToActiveTab({ action: "disable_passive_url_scan" });
        }
    }

    // Update passive Email scan state based on toggle
    async function updatePassiveEmailScanState() {
        if (!emailModeToggle.checked) {  // Passive mode active when toggle is unchecked
            await sendMessageToActiveTab({ action: "enable_passive_email_scan" });
        } else {
            await sendMessageToActiveTab({ action: "disable_passive_email_scan" });
        }
    }

    // Call these functions on toggle change:
    urlModeToggle.addEventListener('change', async () => {
        updateUrlModeUI();
        await updatePassiveUrlScanState();
    });

    emailModeToggle.addEventListener('change', async () => {
        updateEmailModeUI();
        await updatePassiveEmailScanState();
    });

    // Also call on popup load to sync state:
    updatePassiveUrlScanState();
    updatePassiveEmailScanState();
});
