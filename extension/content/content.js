chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
  if (msg.action === "scan_page") {
    
    const urls = Array.from(document.querySelectorAll("a"))
                      .map(a => a.href);

    const text = document.body.innerText;

    chrome.runtime.sendMessage(
      { action: "analyze_text", text },
      response => {
        console.log("Email/Page scan result:", response.result);
      }
    );

    chrome.runtime.sendMessage(
      { action: "analyze_url", url: window.location.href },
      response => {
        console.log("URL scan:", response.result);
      }
    );
  }
});
