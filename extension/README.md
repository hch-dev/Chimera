# **README.md**

“This repository provides the UI, file structure, and communication framework for the Chimera phishing detection extension. The integration team is responsible for implementing backend connectivity (GAN model API), wiring the logic in detector.js, and finalizing the extension for production.”

# 🛡️ Chimera Browser Extension

A phishing-detection browser extension powered by your custom GAN-based URL & email detector.

This extension scans URLs, page content, and (optionally) email text to identify potential phishing attempts. It communicates with your backend AI model through the `api/detector.js` file.

---

## 📁 **Project Structure**

```
extension/
│
├── manifest.json
│
├── popup/
│   ├── popup.html
│   ├── popup.js
│   └── popup.css
│
├── options/
│   ├── options.html
│   ├── options.js
│   └── options.css
│
├── background/
│   └── service_worker.js
│
├── content/
│   └── content.js
│
├── api/
│   └── detector.js
│
├── utils/
│   └── messaging.js
│
└── icons/
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

---

## 🧠 **Where to Integrate the AI Model**

### 🔥 **Your GAN phishing detection API must be integrated inside:**

```
api/detector.js
```

This is the **only file** where the integration team should write the code that communicates with the backend model.

### detector.js Responsibilities

* Accept URLs, page text, or email text.
* Send the input to your GAN inference API.
* Receive the prediction (e.g., `"safe"` / `"phishing"` / probability scores).
* Return that result to the popup, background, or content scripts.

---

## 📘 **File-By-File Explanation**

### 📄 `manifest.json`

The extension's core configuration:

* Declares permissions (`tabs`, `storage`, `activeTab`).
* Registers popup, background service worker, and content scripts.
* Loads extension icons and declares action behavior.

---

## 📁 popup/

This is what the user sees when they click the extension icon.

### **popup.html**

The UI layout (HTML structure).

### **popup.js**

* Sends URLs to the detector.
* Displays results (e.g., safe / warning).
* Handles UI state updates.

### **popup.css**

Styled to match your Chimera website theme (the Tumblr-like template you provided).

---

## 📁 options/

Extension settings page (chrome://extensions → "Options")

### **options.html**

UI for configuration (API URL, toggle features, developer options).

### **options.js**

* Saves & loads settings using Chrome Storage.
* Allows integration team to insert backend API URL.

### **options.css**

Styling for the options page, consistent with your brand.

---

## 📁 background/

### **service_worker.js**

Runs in the background:

* Intercepts tab updates.
* Auto-scans visited URLs (if enabled).
* Sends results to content.js or popup.
* Uses `utils/messaging.js` to talk to other scripts.

---

## 📁 content/

### **content.js**

Injected into webpages:

* Extracts page text or suspicious elements.
* Sends data to background or directly to detector.
* Displays on-page warnings (banners or highlights).

---

## 📁 api/

### **detector.js**

**The MOST important file for your AI integration.**

This acts as the bridge between the extension and your GAN model.

Implementation notes for the integration team:

* Add your fetch request or WebSocket client here.
* Expose a function like:

```js
export async function analyze(input) {
    // send input to AI
    // return model result
}
```

* Ensure it returns JSON in a consistent format, e.g.:

```json
{
  "verdict": "phishing",
  "confidence": 0.92
}
```

Popup, background, and content scripts depend on this.

---

## 📁 utils/

### **messaging.js**

A unified wrapper around Chrome message passing:

* `sendToBackground()`
* `sendToContent()`
* `sendToPopup()`

Ensures stable communication between components.

---

## 📁 icons/

Place the extension icons here:

| File            | Purpose                          |
| --------------- | -------------------------------- |
| **icon16.png**  | Toolbar                          |
| **icon48.png**  | Chrome settings / extension list |
| **icon128.png** | Store listing, high-resolution   |

---

## 🚀 How to Build & Load the Extension

1. Open **Chrome**
2. Go to **chrome://extensions/**
3. Enable **Developer mode**
4. Click **Load unpacked**
5. Select the **extension/** folder

---

## 🧩 Integration Checklist

* [ ] Add your API URL to **options.js** (or hard-code inside detector.js).
* [ ] Write the AI request logic inside **api/detector.js**.
* [ ] Ensure correct return format.
* [ ] Verify popup → background → API messaging works.
* [ ] Test detection on real URLs.

---
