/* -----------------------------------------------------------
   app.js - Enhanced Recipe Chatbot with Session Memory
   - 세션 관리(localStorage)
   - 결정(저신뢰) 버튼 렌더
   - 이미지 갤러리
   - 출처 섹션 렌더
   ----------------------------------------------------------- */

window.addEventListener("DOMContentLoaded", () => {
  // 1) DOM Elements
  const chatWindow = document.querySelector("#chat-window");
  const chatForm = document.querySelector("#chat-form");
  const userInput = document.querySelector("#user-input");
  const resetBtn = document.querySelector("#reset-chat");
  const sessionInfo = document.querySelector("#session-info");
  // Image controls
  const toggleImages = document.querySelector('#toggle-images');
  const policySelect = document.querySelector('#image-policy');
  const maxImagesInput = document.querySelector('#max-images');

  // 2) Session Management
  let sessionId = localStorage.getItem("recipe_rag_session_id");
  let conversationTurns = 0;
  let lastUserQuery = "";

  function updateSessionInfo() {
    if (!sessionInfo) return;
    sessionInfo.textContent = sessionId
      ? `대화 ${conversationTurns} | 세션: ${sessionId.slice(0, 8)}...`
      : "새 대화를 시작해 보세요";
  }

  function resetSession() {
    if (!confirm("채팅을 초기화할까요?")) return;
    if (sessionId) {
      fetch(`/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
    }
    localStorage.removeItem("recipe_rag_session_id");
    sessionId = null;
    conversationTurns = 0;
    if (chatWindow) {
      chatWindow.innerHTML = `
        <div class="self-start rounded-lg bg-gray-200 px-4 py-2 text-sm text-gray-800">
          안녕하세요! 요리 레시피, 조리법, 보관법 등 궁금한 점을 물어보세요.
        </div>
      `;
    }
    updateSessionInfo();
  }

  // 3) Chat UI helpers
  function addBubble(text, sender = "bot") {
    const wrapper = document.createElement("div");
    wrapper.className =
      sender === "user"
        ? "ml-auto mb-3 max-w-[80%] bg-[#c59d5f] text-white px-4 py-3 rounded-2xl shadow-md"
        : "mr-auto mb-3 max-w-[80%] bg-gray-100 text-gray-900 px-4 py-3 rounded-2xl shadow-md";
    wrapper.innerHTML = formatMessage(text || "");
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return wrapper;
  }

  function addImageGallery(urls, parentBubble = null) {
    try {
      if (!urls || !urls.length) return null;
      const valid = urls.filter(u => typeof u === 'string' && u.startsWith('http'));
      if (!valid.length) return null;
      const container = document.createElement('div');
      container.className = 'mt-3 grid grid-cols-2 gap-2';
      valid.slice(0, 6).forEach((u) => {
        const a = document.createElement('a');
        a.href = u; a.target = '_blank'; a.rel = 'noopener noreferrer';
        a.className = 'block overflow-hidden rounded-lg hover:opacity-90 transition';
        const img = document.createElement('img');
        img.src = u; img.alt = 'recipe image'; img.loading = 'lazy';
        img.style.maxHeight = '160px'; img.style.objectFit = 'cover';
        a.appendChild(img); container.appendChild(a);
      });
      if (parentBubble) { parentBubble.appendChild(container); }
      else { const b = addBubble('', 'bot'); b.appendChild(container); }
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (e) { console.warn('Image gallery render failed', e); }
  }

  function formatMessage(text) {
    return String(text)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }

  function addLoading() {
    const loader = document.createElement("div");
    loader.className = "mr-auto mb-3 flex items-center gap-1";
    loader.innerHTML = `
      <span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full"></span>
      <span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full" style="animation-delay:.15s"></span>
      <span class="animate-pulse w-2 h-2 bg-gray-500 rounded-full" style="animation-delay:.3s"></span>
    `;
    chatWindow.appendChild(loader);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return loader;
  }

  // 4) API
  async function sendQuery(text, extra = {}) {
    const body = {
      query: text,
      session_id: sessionId,
      k: 10,
      enable_rewrite: true,
      include_images: toggleImages ? !!toggleImages.checked : true,
      image_policy: policySelect ? policySelect.value : 'strict',
      max_images: maxImagesInput ? Math.max(0, Math.min(12, parseInt(maxImagesInput.value || '5', 10))) : 5,
      ...extra,
    };
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // 5) Form handler
  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = userInput.value.trim();
      if (!text) return;
      lastUserQuery = text;
      addBubble(text, "user");
      userInput.value = ""; userInput.focus();
      const loader = addLoading();
      try {
        const data = await sendQuery(text);
        loader.remove();
        if (data.session_id) { sessionId = data.session_id; localStorage.setItem("recipe_rag_session_id", sessionId); }
        conversationTurns = data.conversation_turns || 0; updateSessionInfo();
        const botBubble = addBubble(data.answer || "죄송해요. 답변을 생성하지 못했어요.");
        try { if (data && data.decision_required) renderDecisionControls(botBubble, data); } catch {}
        try { if (Array.isArray(data.image_urls) && data.image_urls.length > 0) addImageGallery(data.image_urls); } catch {}
        try { if (Array.isArray(data.sources) && data.sources.length > 0) renderSources(botBubble, data.sources); } catch {}
      } catch (err) {
        console.error(err); loader.remove(); addBubble("오류가 발생했어요. 잠시 후 다시 시도해 주세요.");
      }
    });
  }

  // 6) Decision controls
  async function sendDecision(decision) {
    const loader = addLoading();
    try {
      const extra = decision === 'proceed'
        ? { decision: 'proceed', allow_low_confidence: true }
        : { decision: 'clarify' };
      const data = await sendQuery(lastUserQuery, extra);
      loader.remove();
      if (data.session_id) { sessionId = data.session_id; localStorage.setItem('recipe_rag_session_id', sessionId); }
      conversationTurns = data.conversation_turns || 0; updateSessionInfo();
      addBubble(data.answer || (decision === 'proceed' ? '진행을 선택했지만 답변을 생성하지 못했어요.' : '질문 다듬기 안내를 생성하지 못했어요.'));
      try { if (Array.isArray(data.image_urls) && data.image_urls.length > 0) addImageGallery(data.image_urls); } catch {}
      try { if (Array.isArray(data.sources) && data.sources.length > 0) renderSources(null, data.sources); } catch {}
    } catch (e) {
      loader.remove(); console.error(e); addBubble('요청 처리 중 오류가 발생했어요.');
    }
  }

  function renderDecisionControls(parentBubble, data) {
    const box = document.createElement('div');
    box.className = 'mt-3 flex gap-2';
    const proceedBtn = document.createElement('button');
    proceedBtn.className = 'px-3 py-2 rounded bg-[var(--brand-gold)] text-white text-sm hover:bg-[var(--brand-gold-dark)]';
    proceedBtn.textContent = '그대로 진행 (정확도 낮음 감수)';
    proceedBtn.addEventListener('click', () => sendDecision('proceed'));
    const clarifyBtn = document.createElement('button');
    clarifyBtn.className = 'px-3 py-2 rounded bg-gray-200 text-gray-800 text-sm hover:bg-gray-300';
    clarifyBtn.textContent = '질문 다듬기';
    clarifyBtn.addEventListener('click', () => {
      try {
        // Prefill the input with the last user query for easy editing
        if (typeof lastUserQuery === 'string' && lastUserQuery.trim()) {
          userInput.value = lastUserQuery;
          userInput.focus();
          // select all so the user can start typing to replace
          try { userInput.setSelectionRange(0, userInput.value.length); } catch (_) {}
        } else {
          userInput.focus();
        }
        // Optionally remove the decision controls after action
        try { box.remove(); } catch (_) {}
      } catch (_) {
        // Fallback: just focus the input
        userInput.focus();
      }
    });
    box.appendChild(proceedBtn); box.appendChild(clarifyBtn);
    if (parentBubble) parentBubble.appendChild(box); else addBubble('', 'bot').appendChild(box);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function renderSources(parentBubble, sources) {
    try {
      const box = document.createElement('div');
      box.className = 'mt-3 p-3 rounded-lg bg-white border border-gray-200';
      const title = document.createElement('div');
      title.className = 'text-sm font-semibold text-gray-700 mb-2';
      title.textContent = '출처';
      box.appendChild(title);
      const list = document.createElement('ul');
      list.className = 'space-y-1';
      (sources || []).slice(0, 5).forEach((s) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        const url = (s && s.url) || '';
        const ttl = (s && s.title) || '';
        let label = ttl && ttl.trim() ? ttl.trim() : (url ? (new URL(url)).hostname : '링크');
        a.textContent = label;
        if (url && url.startsWith('http')) {
          a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
          a.className = 'text-sm text-[#c59d5f] hover:underline';
        } else {
          a.href = '#'; a.className = 'text-sm text-gray-500 pointer-events-none';
        }
        li.appendChild(a); list.appendChild(li);
      });
      box.appendChild(list);
      if (parentBubble) parentBubble.appendChild(box); else addBubble('', 'bot').appendChild(box);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (e) { console.warn('Sources render failed', e); }
  }

  // 7) Misc
  if (resetBtn) resetBtn.addEventListener("click", resetSession);
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const target = document.querySelector(a.getAttribute("href"));
      if (target) { e.preventDefault(); target.scrollIntoView({ behavior: "smooth", block: "start" }); }
    });
  });
  updateSessionInfo();
});
