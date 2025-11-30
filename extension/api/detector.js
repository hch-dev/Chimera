// Example placeholder: replace with your GAN endpoint
const API_URL = "http://localhost:5000/detect";

async function detectURL(url) {
  const res = await fetch(API_URL + "/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });

  return await res.json();
}

async function detectText(text) {
  const res = await fetch(API_URL + "/email", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  return await res.json();
}
