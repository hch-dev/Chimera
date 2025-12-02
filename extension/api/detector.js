// api/detector.js

// Example placeholder: replace with your actual backend URL if different
const API_URL = "https://adelia-commonsense-soaked.ngrok-free.dev";

/**
 * Detects phishing in a specific URL
 */
export async function detectURL(url) {
  try {
    const res = await fetch(`${API_URL}/scan  `, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    return await res.json();
  } catch (error) {
    console.error("Error in detectURL:", error);
    // Return a fallback object so the extension doesn't crash
    return { verdict: "error", error: error.message };
  }
}

/**
 * Detects phishing in text (emails, selections)
 */
export async function detectText(text) {
  try {
    const res = await fetch(`${API_URL}/email`, { // Assuming /email is the endpoint for text
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    return await res.json();
  } catch (error) {
    console.error("Error in detectText:", error);
    return { verdict: "error", error: error.message };
  }
}

/**
 * Main wrapper function used by the Service Worker.
 * Routes the request to the correct function based on 'type'.
 */
export async function analyze({ type, text }) {
  // Route "selection" (from context menu) to detectText
  if (type === "selection" || type === "email") {
    return await detectText(text);
  }

  // Route "url" to detectURL
  if (type === "url") {
    return await detectURL(text);
  }

  return { verdict: "unknown_type" };
}
