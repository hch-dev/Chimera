document.getElementById("scan").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { action: "scan_page" });
});

import { analyze } from "../api/detector.js";

document.getElementById("scanEmailBtn").addEventListener("click", async () => {
    const text = document.getElementById("emailInput").value.trim();
    if (!text) {
        document.getElementById("emailResult").innerText = "Please enter some text.";
        return;
    }

    document.getElementById("emailResult").innerText = "Scanning...";

    const result = await analyze({ type: "email", text });

    document.getElementById("emailResult").innerText =
        `Verdict: ${result.verdict.toUpperCase()} (confidence: ${result.confidence})`;
});
