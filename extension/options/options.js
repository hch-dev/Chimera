document.addEventListener("DOMContentLoaded", () => {

    const autoScan = document.getElementById("autoScan");
    const notifyThreats = document.getElementById("notifyThreats");
    const saveBtn = document.getElementById("save");

    // Load saved settings
    chrome.storage.sync.get(["autoScan", "notifyThreats"], (data) => {
        autoScan.checked = data.autoScan ?? true;
        notifyThreats.checked = data.notifyThreats ?? true;
    });

    // Save settings
    saveBtn.addEventListener("click", () => {
        chrome.storage.sync.set({
            autoScan: autoScan.checked,
            notifyThreats: notifyThreats.checked
        }, () => {
            saveBtn.innerText = "Saved!";
            setTimeout(() => saveBtn.innerText = "Save Settings", 1200);
        });
    });

});
