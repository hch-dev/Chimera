chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "scanSelection") {
        analyzeSelectedText(msg.text);
    }
});
