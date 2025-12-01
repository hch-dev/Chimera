import * as messaging from "../utils/messaging.js"; 
import { detectURL, detectText, analyze } from "../api/detector.js";

chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {  
  if (msg.action === "analyze_url") {
    const result = await detectURL(msg.url);
    sendResponse({ result });
  }

  if (msg.action === "analyze_text") {
    const result = await detectText(msg.text);
    sendResponse({ result });
  }

  return true; 
});

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "chimera-scan-selection",
        title: "Scan selected text",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "chimera-scan-selection") {
        const selectedText = info.selectionText;
        await analyzeSelectedText(selectedText); 
    }
});

async function analyzeSelectedText(text) {
    const result = await analyze({ type: "selection", text }); 

    chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon128.png", 
        title: "PhishingAI Scan Result",
        message: `Verdict: ${result.verdict ? result.verdict.toUpperCase() : "Analysis Error"}`
    });
}