/* -----------------------------------------------------------
   app.js – front‑end logic for Delish Recipe Chatbot landing
   • Handles chat interaction with /query API (LLM + RAG)
   • Smooth‑scroll helpers & simple nav toggle (mobile)
   • Lightweight (no external JS deps except what HTML loads)
   ----------------------------------------------------------- */

// Wait until DOM is fully parsed
window.addEventListener("DOMContentLoaded", () => {
  /* ──────────────────────────────
     1. Chatbot interaction logic
     ────────────────────────────── */
  const chatWindow = document.querySelector("#chat-window");
  const chatForm   = document.querySelector("#chat-form");
  const userInput  = document.querySelector("#user-input");

  // Add chat bubble
  function addBubble(text, sender = "bot") {
    const wrapper = document.createElement("div");
    wrapper.className =
      sender === "user"
        ? "ml-auto mb-3 max-w-[80%] bg-brand-gold text-gray-900 px-4 py-2 rounded-2xl shadow" // ← 글자색을 검정으로 변경
        : "mr-auto mb-3 max-w-[80%] bg-gray-100 text-gray-900 px-4 py-2 rounded-2xl shadow";
    wrapper.textContent = text;
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return wrapper;
  }

  // Loading bubble (dots)
  function addLoading() {
    const loader = document.createElement("div");
    loader.className = "mr-auto mb-3 flex items-center gap-1";
    loader.innerHTML = `<span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full"></span>
                       <span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full" style="animation-delay:.15s"></span>
                       <span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full" style="animation-delay:.3s"></span>`;
    chatWindow.appendChild(loader);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return loader;
  }

  async function sendQuery(text) {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = userInput.value.trim();
      if (!text) return;

      addBubble(text, "user");
      userInput.value = "";
      userInput.focus();

      const loader = addLoading();
      try {
        const data = await sendQuery(text);
        loader.remove();
        addBubble(data.answer || "죄송합니다, 답변을 생성하지 못했어요.");
      } catch (err) {
        console.error(err);
        loader.remove();
        addBubble("오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
      }
    });
  }

  /* ──────────────────────────────
     2. Smooth scroll for anchor links
     ────────────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const target = document.querySelector(anchor.getAttribute("href"));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  /* ──────────────────────────────
     3. Mobile nav toggle (if exists)
     ────────────────────────────── */
  const navToggle = document.querySelector("#nav-toggle");
  const navMenu   = document.querySelector("#nav-menu");
  if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
      navMenu.classList.toggle("hidden");
    });
  }
});
