(function() {
  const CHATGPT_SELECTORS = {
    INPUT_BOX: "#prompt-textarea",
    FORM: 'form[data-type="unified-composer"]',
    SUBMIT_BUTTON: 'button[data-testid="send-button"]'
  };
  const checkMemoryLimitAndUpdateButton = async (settingsButton) => {
    try {
      const memoryInfo = await backgroundAPI.getMemoryLimitInfo();
      if (memoryInfo.status === "success" && !memoryInfo.canAdd) {
        addRedDotToButton(settingsButton);
      }
    } catch (error) {
      console.error("Error checking memory limit for settings button:", error);
    }
  };
  const addRedDotToButton = (button) => {
    if (button.querySelector(".memory-limit-indicator")) {
      return;
    }
    const redDot = document.createElement("div");
    redDot.className = "memory-limit-indicator";
    button.appendChild(redDot);
    button.title = "Memory limit reached! Click to manage your memories and sign in for unlimited storage";
  };
  const uiBlueprints = {
    getMemorySuggestionsContainer: (messageId, suggestionsCount, detectedMode = null, headerText = null, bulkActionLabel = "Undo all") => `
            <div class="memory-suggestions-container" data-message-id="${messageId}">
                <div class="memory-suggestions-header">
                    <div class="memory-suggestions-header-icon">${getMemoriesSVG("#6c757d", 14)}</div>
                    <span class="memory-suggestions-title">${headerText || `Saved ${suggestionsCount} ${suggestionsCount === 1 ? "memory" : "memories"}`}</span>
                    ${detectedMode ? `<span class="detected-mode-label">${detectedMode}</span>` : ""}
                </div>
                <div class="memory-suggestions-list"></div>
                ${bulkActionLabel ? `<button class="discard-all-button">${bulkActionLabel}</button>` : ""}
            </div>
        `,
    getMemorySuggestionItem: (suggestion, index) => {
      const memoryText = typeof suggestion === "string" ? suggestion : suggestion.memory || "";
      const tagText = typeof suggestion === "object" ? suggestion.tag || "" : "";
      return `
                <div class="memory-suggestion-item" data-index="${index}">
                    <div class="suggestion-content-wrapper" style="display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; padding-right: 8px;">
                        <div class="suggestion-text">${memoryText}</div>
                        ${tagText ? `<div class="suggestion-tag-badge">${tagText}</div>` : ""}
                    </div>
                    <div class="suggestion-buttons">
                        <button class="approve-button" title="Approve and save this memory">✓</button>
                        <button class="edit-button" title="Edit this memory">${getEditSVG("#6c757d", 12)}</button>
                    </div>
                </div>
            `;
    },
    getAutoSavedMemoryItem: (savedMemory, index) => {
      const memoryText = (savedMemory == null ? void 0 : savedMemory.text) || "";
      const tagText = (savedMemory == null ? void 0 : savedMemory.tag) || "";
      return `
                <div class="memory-suggestion-item memory-suggestion-item--saved" data-index="${index}" data-memory-id="${(savedMemory == null ? void 0 : savedMemory.id) || ""}">
                    <div class="suggestion-content-wrapper" style="display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; padding-right: 8px;">
                        <div class="suggestion-text">${memoryText}</div>
                        ${tagText ? `<div class="suggestion-tag-badge">${tagText}</div>` : ""}
                    </div>
                    <div class="suggestion-buttons">
                        <button class="undo-button" title="Delete this memory and undo the auto-save">Undo</button>
                    </div>
                </div>
            `;
    },
    getReadOnlyMemoryItem: (memory, index, itemClass = "") => {
      const memoryText = typeof memory === "string" ? memory : memory.memory || memory.text || "";
      const tagText = typeof memory === "object" ? memory.tag || "" : "";
      return `
                <div class="memory-suggestion-item ${itemClass}" data-index="${index}">
                    <div class="suggestion-content-wrapper" style="display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; padding-right: 8px;">
                        <div class="suggestion-text">${memoryText}</div>
                        ${tagText ? `<div class="suggestion-tag-badge">${tagText}</div>` : ""}
                    </div>
                </div>
            `;
    },
    getMemoryEditField: (originalText) => `
            <textarea class="memory-edit-field" placeholder="Edit memory...">${originalText}</textarea>
        `,
    getExtractedMemoryNotification: (memories) => `
            <div class="extracted-memory-notification">
                <div class="memory-prefix">
                    <div class="memory-prefix-icon">${getMemoriesSVG("#d1d5db", 12)}</div>
                    <span class="memory-prefix-text">memory saved:</span>
                </div>
                <div class="extracted-memory-text">${memories.join(" • ")}</div>
            </div>
        `,
    getLimitBlockedWarning: (limitType) => {
      const isGuestLimit = limitType === "guest";
      const warningClass = `memory-limit-warning ${isGuestLimit ? "memory-limit-warning--guest" : "memory-limit-warning--logged-in"}`;
      const buttonClass = `memory-warning-button ${isGuestLimit ? "memory-warning-button--guest" : "memory-warning-button--logged-in"}`;
      const warningText = isGuestLimit ? "We extracted this memory, but couldn't save it because you've reached the guest limit. Sign in to unlock 100 free memories." : "We extracted this memory, but couldn't save it because you've reached your free limit. Upgrade to keep saving automatically.";
      return `
                <div class="${warningClass}">
                    <div class="memory-warning-icon">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke-linecap="round" stroke-linejoin="round"/>
                            <line x1="12" y1="9" x2="12" y2="13" stroke-linecap="round" stroke-linejoin="round"/>
                            <line x1="12" y1="17" x2="12.01" y2="17" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="memory-warning-text">${warningText}</div>
                    <button class="${buttonClass}">${isGuestLimit ? "Sign in" : "Upgrade"}</button>
                </div>
            `;
    },
    getMainContainer: () => `
            <div class="maxmemory-main-container">
                <div class="maxmemory-brand">
                    <div class="maxmemory-logo">${getMaxMemoryLogoSVG("#6b7280", 12)}</div>
                    <span class="maxmemory-brand-text">MaxMemory</span>
                    <div class="maxmemory-toggle-container">
                        <label class="maxmemory-toggle-switch">
                            <input type="checkbox" id="maxmemory-toggle" checked>
                            <span class="maxmemory-toggle-slider"></span>
                        </label>
                    </div>
                    <button class="maxmemory-settings-button" title="Open MaxMemory settings and memories">
                        ${getSettingsSVG("#888", 14)}
                    </button>
                </div>
                <button class="get-memories-button" id="get-memories-button" style="display: none;">
                    ${getMemoriesSVG("#40414f")}<span>Submit</span>
                </button>
            </div>
        `,
    getBrandText: () => `
            <div class="maxmemory-brand">
                <div class="maxmemory-logo">${getMaxMemoryLogoSVG("#6b7280", 12)}</div>
                <span class="maxmemory-brand-text">MaxMemory</span>
                <div class="maxmemory-toggle-container">
                    <label class="maxmemory-toggle-switch">
                        <input type="checkbox" id="maxmemory-toggle" checked>
                        <span class="maxmemory-toggle-slider"></span>
                    </label>
                </div>
            </div>
        `,
    getSettingsButton: () => `
            <button class="maxmemory-settings-button" title="Open MaxMemory settings and memories">
                ${getSettingsSVG("#888", 14)}
            </button>
        `,
    getSubmitButton: () => `
            <button class="get-memories-button" id="get-memories-button" style="display: none;">
                ${getMemoriesSVG("#40414f")}<span>Submit</span>
            </button>
        `,
    getMemoryLimitWarning: (limitType, current, limit) => {
      const warningText = limitType === "guest" ? `Guest limit reached (${current}/${limit}). Sign In with Google for 100 free memories.` : `Free limit reached (${current}/${limit}). Upgrade to Pro for unlimited memories.`;
      const actionLabel = limitType === "guest" ? "Sign in" : "Upgrade";
      const warningClass = limitType === "guest" ? "memory-limit-warning memory-limit-warning--guest" : "memory-limit-warning memory-limit-warning--logged-in";
      const buttonClass = limitType === "guest" ? "memory-warning-button memory-warning-button--guest" : "memory-warning-button memory-warning-button--logged-in";
      return `
                <div class="${warningClass}">
                    <div class="memory-warning-icon">
                        ${getWarningIconSVG("#92400e", 14)}
                    </div>
                    <div class="memory-warning-text">${warningText}</div>
                    <button class="${buttonClass}">${actionLabel}</button>
                </div>
            `;
    },
    getWarningButton: () => `
            <button class="memory-warning-button">Sign in</button>
        `
  };
  const backgroundAPI = {
    async searchMemories(query) {
      return await chrome.runtime.sendMessage({
        type: "SEARCH_MEMORIES",
        query
      });
    },
    trackError(errorData) {
      chrome.runtime.sendMessage({
        type: "TRACK_ERROR",
        errorData
      }).catch(() => {
      });
    },
    trackPopupOpened(source = "content_script") {
      chrome.runtime.sendMessage({
        type: "TRACK_POPUP_OPENED",
        source
      }).catch(() => {
      });
    },
    openPopupInTab() {
      chrome.runtime.sendMessage({
        type: "OPEN_POPUP_IN_TAB"
      }).catch(() => {
      });
    },
    trackMaxMemoryToggled(enabled) {
      chrome.runtime.sendMessage({
        type: "TRACK_MAX_MEMORY_TOGGLED",
        enabled
      }).catch(() => {
      });
    },
    openPopup() {
      chrome.runtime.sendMessage({
        type: "OPEN_POPUP"
      }).catch(() => {
      });
    },
    async getMemoryLimitInfo() {
      try {
        return await chrome.runtime.sendMessage({
          type: "GET_MEMORY_LIMIT_INFO"
        });
      } catch (error) {
        console.error("Error getting memory limit info:", error);
        return { status: "error" };
      }
    },
    trackMemorySuggestionDiscarded(discardedCount, content, mode = null) {
      chrome.runtime.sendMessage({
        type: "TRACK_MEMORY_SUGGESTION_DISCARDED",
        discardedCount,
        content,
        mode
      }).catch(() => {
      });
    },
    async deleteMemory(id, text = "") {
      return await chrome.runtime.sendMessage({
        type: "DELETE_MEMORY",
        id,
        text
      });
    }
  };
  const createMemoriesIcon = () => {
    const parser = new DOMParser();
    return parser.parseFromString(getMemoriesSVG("#d1d5db"), "image/svg+xml").documentElement;
  };
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const year = date.getFullYear();
    const month = `0${date.getMonth() + 1}`.slice(-2);
    const day = `0${date.getDate()}`.slice(-2);
    return `${year}-${month}-${day}`;
  };
  const MEMORY_MARKERS = {
    start: "[RELEVANT_PAST_MEMORIES_START]",
    end: "[RELEVANT_PAST_MEMORIES_END]"
  };
  const containsMemoryMarkers = (value) => {
    const text = typeof value === "string" ? value : (value == null ? void 0 : value.textContent) || "";
    return text.includes(MEMORY_MARKERS.start) && text.includes(MEMORY_MARKERS.end);
  };
  const getConversationMessageContainers = (root = document) => {
    const roleBasedMessages = Array.from(root.querySelectorAll("[data-message-author-role]"));
    if (roleBasedMessages.length) {
      return roleBasedMessages;
    }
    return Array.from(root.querySelectorAll("article"));
  };
  const getUserMessageContainers = (root = document) => {
    const roleBasedUserMessages = Array.from(root.querySelectorAll('[data-message-author-role="user"]'));
    if (roleBasedUserMessages.length) {
      return roleBasedUserMessages;
    }
    return Array.from(root.querySelectorAll("article"));
  };
  const getClosestUserMessageContainer = (node) => {
    var _a;
    if (!node) return null;
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return ((_a = element == null ? void 0 : element.closest) == null ? void 0 : _a.call(element, '[data-message-author-role="user"], article')) || null;
  };
  const getMessageContentElement = (messageContainer) => {
    var _a;
    if (!messageContainer) return null;
    const preferredSelectors = '.whitespace-pre-wrap, [data-testid="user-message"], [dir="auto"]';
    const preferredCandidates = [];
    if ((_a = messageContainer.matches) == null ? void 0 : _a.call(messageContainer, preferredSelectors)) {
      preferredCandidates.push(messageContainer);
    }
    preferredCandidates.push(...messageContainer.querySelectorAll(preferredSelectors));
    let deepestMatchingPreferred = null;
    for (const candidate of preferredCandidates) {
      if (candidate instanceof HTMLElement && containsMemoryMarkers(candidate)) {
        deepestMatchingPreferred = candidate;
      }
    }
    if (deepestMatchingPreferred) {
      return deepestMatchingPreferred;
    }
    let deepestMatchingElement = null;
    const walker = document.createTreeWalker(messageContainer, NodeFilter.SHOW_ELEMENT);
    while (walker.nextNode()) {
      const candidate = walker.currentNode;
      if (!(candidate instanceof HTMLElement)) continue;
      if (candidate.closest(".memory-section")) continue;
      if (containsMemoryMarkers(candidate)) {
        deepestMatchingElement = candidate;
      }
    }
    return deepestMatchingElement;
  };
  const getInputBox = () => {
    const inputBox = document.querySelector(CHATGPT_SELECTORS.INPUT_BOX);
    return inputBox;
  };
  const styleMemoriesInChat = () => {
    getUserMessageContainers().forEach(handleMessageStyling);
  };
  let styleMemoriesInChatScheduled = false;
  const scheduleStyleMemoriesInChat = () => {
    if (styleMemoriesInChatScheduled) {
      return;
    }
    styleMemoriesInChatScheduled = true;
    requestAnimationFrame(() => {
      styleMemoriesInChatScheduled = false;
      styleMemoriesInChat();
    });
  };
  let pendingMemoryStyleRetryTimer = null;
  let pendingMemoryStyleRetryAttempts = 0;
  let pendingMemoryStyleTarget = null;
  const clearPendingMemoryStylingWatch = () => {
    if (pendingMemoryStyleRetryTimer) {
      clearTimeout(pendingMemoryStyleRetryTimer);
      pendingMemoryStyleRetryTimer = null;
    }
    pendingMemoryStyleRetryAttempts = 0;
    pendingMemoryStyleTarget = null;
  };
  const watchForPendingMemoryStyledMessage = () => {
    if (!pendingMemoryStyleTarget) {
      return;
    }
    scheduleStyleMemoriesInChat();
    const userMessages = getUserMessageContainers();
    const matchingMessage = Array.from(userMessages).reverse().find((messageContainer) => {
      const messageDiv = getMessageContentElement(messageContainer);
      if (!messageDiv) return false;
      const messageText = messageDiv.textContent || "";
      return messageText.includes(MEMORY_MARKERS.start) && messageText.includes(MEMORY_MARKERS.end) && messageText.includes(pendingMemoryStyleTarget.memoriesSnippet);
    });
    if (matchingMessage) {
      handleMessageStyling(matchingMessage);
      const styledMessageDiv = getMessageContentElement(matchingMessage);
      if (styledMessageDiv == null ? void 0 : styledMessageDiv.querySelector(".memory-section")) {
        clearPendingMemoryStylingWatch();
        return;
      }
    }
    pendingMemoryStyleRetryAttempts += 1;
    if (pendingMemoryStyleRetryAttempts >= 30) {
      clearPendingMemoryStylingWatch();
      return;
    }
    pendingMemoryStyleRetryTimer = setTimeout(watchForPendingMemoryStyledMessage, 250);
  };
  const beginPendingMemoryStylingWatch = (memoriesText) => {
    if (!memoriesText) {
      return;
    }
    clearPendingMemoryStylingWatch();
    pendingMemoryStyleTarget = {
      memoriesSnippet: memoriesText.slice(0, 160)
    };
    watchForPendingMemoryStyledMessage();
  };
  const handleMessageStyling = (messageContainer) => {
    var _a;
    const messageDiv = getMessageContentElement(messageContainer);
    if (!messageDiv) return;
    const match = messageDiv.textContent.match(/\[RELEVANT_PAST_MEMORIES_START\]([\s\S]*?)\[RELEVANT_PAST_MEMORIES_END\]/);
    if (!match) return;
    const [fullMatch, memoriesContent] = match;
    const [before, after] = messageDiv.textContent.split(fullMatch);
    const trimmedMemoriesContent = memoriesContent.trim();
    const contentSignature = `${before.trim()}::${trimmedMemoriesContent}::${after.trim()}`;
    if (messageDiv.dataset.maxmemoryProcessed === "true" && messageDiv.dataset.maxmemoryProcessedSignature === contentSignature && messageDiv.querySelector(".memory-section")) {
      return;
    }
    messageDiv.setAttribute("data-full-memories", trimmedMemoriesContent);
    const truncatedContent = memoriesContent.length > 280 ? `${memoriesContent.slice(0, 280)}... <span class="show-more-memories" style="color: #666; font-weight: 600; cursor: pointer; user-select: text; pointer-events: auto;">Show more</span>` : memoriesContent;
    messageDiv.innerHTML = `${before.trim()}<div class="memory-section">${createMemoriesIcon().outerHTML} <span class="memories-content">${truncatedContent}</span></div>${after.trim()}`;
    messageDiv.dataset.maxmemoryProcessed = "true";
    messageDiv.dataset.maxmemoryProcessedSignature = contentSignature;
    (_a = messageDiv.querySelector(".show-more-memories")) == null ? void 0 : _a.addEventListener("click", (e) => {
      e.stopPropagation();
      e.target.closest(".memories-content").innerHTML = memoriesContent;
    });
  };
  const ObserverManager = {
    _mainObserver: null,
    _messageListObserver: null,
    _inputFormObserver: null,
    _submitButtonObserver: null,
    _callbacks: {},
    // NEW: Flags to prevent re-deploying observers
    _inputObserversDeployed: false,
    _messageObserverDeployed: false,
    init(callbacks) {
      this._callbacks = callbacks;
    },
    start() {
      this.stop();
      console.log("[ObserverManager] Starting observers.");
      this._inputObserversDeployed = false;
      this._messageObserverDeployed = false;
      const mainContainer = document.querySelector("main");
      if (!mainContainer) {
        setTimeout(() => this.start(), 250);
        return;
      }
      this._mainObserver = new MutationObserver(() => {
        const form = document.querySelector(CHATGPT_SELECTORS.FORM);
        const messageContainer = this._getMessageListContainer();
        if (form && !this._inputObserversDeployed) {
          console.log("[ObserverManager] Input form found. Deploying input-related observers.");
          this._deployInputObservers(form);
          this._inputObserversDeployed = true;
        }
        if (messageContainer && !this._messageObserverDeployed) {
          console.log("[ObserverManager] Message container found. Deploying message observer.");
          this._deployMessageObserver(messageContainer);
          this._messageObserverDeployed = true;
        }
        if (this._inputObserversDeployed && this._messageObserverDeployed) {
          console.log("[ObserverManager] All targeted observers deployed. Disconnecting main observer.");
          this._mainObserver.disconnect();
          this._mainObserver = null;
        }
      });
      this._mainObserver.observe(mainContainer, {
        childList: true,
        subtree: true
      });
      const initialForm = document.querySelector(CHATGPT_SELECTORS.FORM);
      const initialMessageContainer = this._getMessageListContainer();
      if (initialForm) {
        this._deployInputObservers(initialForm);
        this._inputObserversDeployed = true;
      }
      if (initialMessageContainer) {
        this._deployMessageObserver(initialMessageContainer);
        this._messageObserverDeployed = true;
      }
      if (this._inputObserversDeployed && this._messageObserverDeployed) {
        this._mainObserver.disconnect();
        this._mainObserver = null;
      }
    },
    // NEW: Function to deploy only input-related observers
    _deployInputObservers(formEl) {
      this._inputFormObserver = new MutationObserver((mutations) => {
        if (this._callbacks.onInputAreaChanged) {
          this._callbacks.onInputAreaChanged(mutations);
        }
      });
      this._inputFormObserver.observe(formEl.parentNode, { childList: true });
      this._submitButtonObserver = new MutationObserver((mutations) => {
        if (this._callbacks.onSubmitButtonChanged) {
          this._callbacks.onSubmitButtonChanged(mutations);
        }
      });
      this._submitButtonObserver.observe(formEl, { childList: true, subtree: true });
      if (this._callbacks.onUIReady) {
        console.log("[ObserverManager] Input UI ready, triggering onUIReady callback");
        this._callbacks.onUIReady();
      }
    },
    // NEW: Function to deploy only the message list observer
    _deployMessageObserver(messageContainerEl) {
      console.log("[ObserverManager] Scanning for pre-existing messages to style...");
      scheduleStyleMemoriesInChat();
      this._messageListObserver = new MutationObserver((mutations) => {
        if (this._callbacks.onMessagesAdded) {
          this._callbacks.onMessagesAdded(mutations);
        }
      });
      this._messageListObserver.observe(messageContainerEl, {
        childList: true,
        subtree: true,
        characterData: true
      });
    },
    stop() {
      if (this._mainObserver) this._mainObserver.disconnect();
      if (this._messageListObserver) this._messageListObserver.disconnect();
      if (this._inputFormObserver) this._inputFormObserver.disconnect();
      if (this._submitButtonObserver) this._submitButtonObserver.disconnect();
      this._mainObserver = null;
      this._messageListObserver = null;
      this._inputFormObserver = null;
      this._submitButtonObserver = null;
      this._inputObserversDeployed = false;
      this._messageObserverDeployed = false;
      console.log("[ObserverManager] All observers stopped.");
    },
    _getMessageListContainer() {
      const mainEl = document.querySelector("main");
      if (!mainEl) return null;
      const messageContainers = getConversationMessageContainers(mainEl);
      if (!messageContainers.length) return null;
      const containsAll = (el) => messageContainers.every((messageContainer) => el && el.contains(messageContainer));
      let candidate = messageContainers[0].parentElement;
      while (candidate && candidate !== mainEl && containsAll(candidate.parentElement)) {
        candidate = candidate.parentElement;
      }
      return candidate || mainEl;
    }
  };
  function setupInputListeners() {
    const inputBox = getInputBox();
    if (!inputBox || inputBox.__maxMemoryBound) return;
    const updateVisibility = () => {
      const submitButton = document.querySelector(CHATGPT_SELECTORS.SUBMIT_BUTTON);
      if (!submitButton) return;
      const hasContent = getInputContent(inputBox).length > 0;
      if (hasContent) {
        submitButton.style.visibility = "hidden";
        submitButton.style.opacity = "0";
      } else {
        submitButton.style.visibility = "visible";
        submitButton.style.opacity = "1";
      }
    };
    inputBox.addEventListener("input", updateVisibility);
    inputBox.addEventListener("keyup", updateVisibility);
    inputBox.__maxMemoryBound = true;
    updateVisibility();
  }
  async function getAndInsertMemories(button) {
    try {
      const toggleResponse = await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: "GET_MAXMEMORY_ENABLED" }, resolve);
      });
      if (toggleResponse && toggleResponse.status === "success" && !toggleResponse.enabled) {
        console.log("MaxMemory is disabled, skipping memory processing");
        return;
      }
      button.disabled = true;
      button.classList.add("loading");
      const inputBox = getInputBox();
      if (!inputBox) {
        console.error("Input box not found.");
        backgroundAPI.trackError({
          error_type: "input_box_not_found",
          error_message: "ChatGPT input box not found",
          context: "content_script",
          function: "getAndInsertMemories",
          url: window.location.href
        });
        return;
      }
      let userInput2 = "";
      console.log("[ContentScript] Getting input from:", inputBox.tagName);
      if (inputBox.tagName === "TEXTAREA") {
        userInput2 = inputBox.value;
        console.log("[ContentScript] Textarea input length:", userInput2.length);
      } else {
        const paragraphs = inputBox.querySelectorAll("p");
        userInput2 = Array.from(paragraphs).map((p) => p.textContent).join("\n");
        console.log("[ContentScript] ContentEditable input length:", userInput2.length);
      }
      const response = await backgroundAPI.searchMemories(userInput2.trim());
      if ((response == null ? void 0 : response.status) === "success" && response.results.length) {
        const limitedResults = response.results.slice(0, 10);
        const memoriesText = limitedResults.map((memory) => `[${formatDate(memory.timestamp)}] ${memory.memory_text}`).join(" ");
        console.log("[ContentScript] Injecting memories:", memoriesText.substring(0, 50) + "...");
        console.log("[ContentScript] Injecting memories into input box");
        let newContent;
        if (inputBox.tagName === "TEXTAREA") {
          newContent = `[RELEVANT_PAST_MEMORIES_START] ${memoriesText} [RELEVANT_PAST_MEMORIES_END]

${userInput2}`;
        } else {
          const lines = userInput2.split("\n");
          newContent = `[RELEVANT_PAST_MEMORIES_START] ${memoriesText} [RELEVANT_PAST_MEMORIES_END]

${lines.join("\n")}`;
        }
        setInputContent(inputBox, newContent);
        beginPendingMemoryStylingWatch(memoriesText);
        console.log("[ContentScript] Memories injected, content length:", newContent.length);
        inputBox.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(inputBox);
        range.collapse(false);
        selection.removeAllRanges();
        selection.addRange(range);
      }
    } catch (error) {
      console.error("Error fetching memories:", error);
      backgroundAPI.trackError({
        error_type: "memory_fetch_error",
        error_message: error.message,
        error_stack: error.stack,
        context: "content_script",
        function: "getAndInsertMemories",
        user_input_length: userInput ? userInput.length : 0
      });
    } finally {
      button.disabled = false;
      button.classList.remove("loading");
      console.log("[ContentScript] getAndInsertMemories completed");
    }
  }
  const getChatId = () => {
    const url = window.location.href;
    const match = url.match(/\/c\/([a-f0-9-]+)/);
    return match ? match[1] : "default";
  };
  const setNativeSubmitButtonVisibility = (isVisible) => {
    const submitButton = document.querySelector(CHATGPT_SELECTORS.SUBMIT_BUTTON);
    if (!submitButton) return;
    submitButton.style.visibility = isVisible ? "visible" : "hidden";
    submitButton.style.opacity = isVisible ? "1" : "0";
  };
  const syncMaxMemoryToggleUI = (enabled) => {
    const toggleSwitch = document.querySelector("#maxmemory-toggle");
    const button = document.getElementById("get-memories-button");
    const inputBox = getInputBox();
    const hasContent = inputBox ? getInputContent(inputBox).length > 0 : false;
    if (toggleSwitch) {
      toggleSwitch.checked = enabled;
    }
    if (button && enabled && hasContent) {
      button.style.display = "flex";
      button.style.transition = "opacity 0.2s ease-in-out, transform 0.2s ease-in-out";
      button.style.visibility = "visible";
      button.style.opacity = "1";
      button.style.transform = "translateY(0)";
    } else if (button) {
      button.style.visibility = "hidden";
      button.style.opacity = "0";
      button.style.transform = "translateY(10px)";
      setTimeout(() => {
        if (button.style.opacity === "0") {
          button.style.display = "none";
        }
      }, 200);
    }
    setNativeSubmitButtonVisibility(!(enabled && hasContent));
  };
  const createMaxMemoryInterface = async () => {
    const container = document.createElement("div");
    container.innerHTML = uiBlueprints.getMainContainer();
    const settingsButton = container.querySelector(".maxmemory-settings-button");
    const button = container.querySelector("#get-memories-button");
    const toggleSwitch = container.querySelector("#maxmemory-toggle");
    try {
      const response = await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: "GET_MAXMEMORY_ENABLED" }, resolve);
      });
      if (response && response.status === "success") {
        toggleSwitch.checked = response.enabled;
      }
    } catch (error) {
      console.error("Error getting MaxMemory enabled state:", error);
      toggleSwitch.checked = true;
    }
    toggleSwitch.addEventListener("change", async (e) => {
      const enabled = e.target.checked;
      try {
        const response = await new Promise((resolve) => {
          chrome.runtime.sendMessage({
            type: "SET_MAXMEMORY_ENABLED",
            enabled
          }, resolve);
        });
        if (response && response.status === "success") {
          console.log("MaxMemory toggle state updated:", enabled);
          syncMaxMemoryToggleUI(enabled);
        } else {
          console.error("Failed to update MaxMemory toggle state");
          e.target.checked = !enabled;
        }
      } catch (error) {
        console.error("Error updating MaxMemory toggle state:", error);
        e.target.checked = !enabled;
      }
    });
    settingsButton.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      backgroundAPI.trackPopupOpened("settings_button");
      backgroundAPI.openPopupInTab();
    });
    checkMemoryLimitAndUpdateButton(settingsButton);
    const updateButtonVisibility = (hasContent = null) => {
      if (hasContent === null) {
        const inputBox = getInputBox();
        hasContent = inputBox ? getInputContent(inputBox).length > 0 : false;
      }
      syncMaxMemoryToggleUI(toggleSwitch.checked);
    };
    const monitorInputChanges = async () => {
      const inputBox = getInputBox();
      if (!inputBox) return;
      const content = getInputContent(inputBox);
      const hasContent = content && content.length > 0;
      updateButtonVisibility(hasContent);
    };
    const setupInputMonitoring = () => {
      const inputBox = getInputBox();
      if (inputBox) {
        inputBox.addEventListener("input", monitorInputChanges);
        inputBox.addEventListener("keyup", monitorInputChanges);
        inputBox.addEventListener("paste", () => {
          setTimeout(monitorInputChanges, 10);
        });
        const observer = new MutationObserver(monitorInputChanges);
        observer.observe(inputBox, {
          childList: true,
          subtree: true,
          characterData: true
        });
      }
    };
    updateButtonVisibility(false);
    setupInputMonitoring();
    button.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      await getAndInsertMemories(button);
      setTimeout(() => {
        const submitButton = document.querySelector(CHATGPT_SELECTORS.SUBMIT_BUTTON);
        if (submitButton && !submitButton.disabled) {
          const originalVisibility = submitButton.style.visibility;
          const originalOpacity = submitButton.style.opacity;
          submitButton.style.visibility = "visible";
          submitButton.style.opacity = "1";
          submitButton.click();
          setTimeout(() => {
            submitButton.style.visibility = originalVisibility;
            submitButton.style.opacity = originalOpacity;
          }, 50);
        }
      }, 100);
    });
    return container.firstElementChild;
  };
  function addGetMemoriesButton() {
    if (window.memoryVaultButtonTimer) {
      clearTimeout(window.memoryVaultButtonTimer);
    }
    if (document.getElementById("maxmemory-container")) {
      console.log("Memory vault container already exists, skipping creation");
      return;
    }
    console.log("Adding MaxMemory button to page");
    const inputBox = getInputBox();
    if (!inputBox) {
      window.memoryVaultButtonTimer = setTimeout(addGetMemoriesButton, 500);
      return;
    }
    window.memoryVaultButtonTimer = setTimeout(() => {
      if (document.getElementById("maxmemory-container")) {
        console.log("Memory vault container already exists (second check), skipping creation");
        return;
      }
      console.log("Creating memory vault container");
      const container = document.createElement("div");
      container.id = "maxmemory-container";
      container.style.display = "flex";
      container.style.marginBottom = "12px";
      createMaxMemoryInterface().then((memoriesButtonContainer) => {
        if (document.getElementById("maxmemory-container")) {
          console.log("Memory vault container already exists (final check), skipping creation");
          return;
        }
        container.appendChild(memoriesButtonContainer);
        const target = document.querySelector(CHATGPT_SELECTORS.FORM);
        if (target) {
          target.parentNode.insertBefore(container, target);
        } else {
          setTimeout(addGetMemoriesButton, 500);
        }
      });
    }, 100);
  }
  function handleEnterKey(event) {
    if (event.key !== "Enter" && event.key !== "NumpadEnter" || event.shiftKey || event.isComposing) {
      return true;
    }
    event.preventDefault();
    event.stopPropagation();
    console.log("[ContentScript] Processing Enter key with MaxMemory");
    (async () => {
      const inputBox = getInputBox();
      const inputContent = getInputContent(inputBox);
      if (!inputContent || !inputBox) {
        console.log("[ContentScript] No input content or input box found");
        return;
      }
      const memoriesButton = document.getElementById("get-memories-button");
      if (memoriesButton && memoriesButton.style.display !== "none" && memoriesButton.style.visibility !== "hidden") {
        memoriesButton.click();
      } else {
        await getAndInsertMemories(memoriesButton || { disabled: false, classList: { add: () => {
        }, remove: () => {
        } } });
        setTimeout(() => {
          console.log("[ContentScript] Submitting after memory injection");
          const submitButton = document.querySelector(CHATGPT_SELECTORS.SUBMIT_BUTTON);
          if (submitButton && !submitButton.disabled) {
            submitButton.style.visibility = "visible";
            submitButton.style.opacity = "1";
            submitButton.click();
            console.log("[ContentScript] Submit button clicked");
          } else {
            console.log("[ContentScript] Submit button not found or disabled");
          }
        }, 300);
      }
    })();
    return false;
  }
  const setupEnterKeyPrevention = () => {
    const proseMirrorEditor = document.querySelector(".ProseMirror");
    const fieldset = document.querySelector("fieldset.flex");
    const contentEditableDiv = document.querySelector('[contenteditable="true"]');
    const promptTextarea = document.querySelector(CHATGPT_SELECTORS.INPUT_BOX);
    console.log("[ContentScript] Setting up Enter key prevention:", {
      proseMirrorEditor: !!proseMirrorEditor,
      fieldset: !!fieldset,
      contentEditableDiv: !!contentEditableDiv,
      promptTextarea: !!promptTextarea
    });
    [window, proseMirrorEditor, fieldset, contentEditableDiv, promptTextarea].forEach((element, index) => {
      if (element) {
        console.log(`[ContentScript] Adding Enter key listeners to element ${index}:`, element.tagName || "window");
        element.addEventListener("keydown", handleEnterKey, { capture: true });
        element.addEventListener("keypress", handleEnterKey, { capture: true });
      }
    });
    if (!promptTextarea) {
      console.log("[ContentScript] Prompt textarea not found, attempting to find it again...");
      const el = document.querySelector(CHATGPT_SELECTORS.INPUT_BOX);
      if (el) {
        console.log("[ContentScript] Prompt textarea found — attaching Enter key listeners");
        el.addEventListener("keydown", handleEnterKey, { capture: true });
        el.addEventListener("keypress", handleEnterKey, { capture: true });
      } else {
        console.log("[ContentScript] Prompt textarea still not found - this should not happen when called from onUIReady");
      }
    }
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, listener, options) {
      var _a;
      if ((type === "keypress" || type === "keydown") && ((_a = this.classList) == null ? void 0 : _a.contains("ProseMirror"))) {
        const wrappedListener = function(event) {
          if ((event.key === "Enter" || event.key === "NumpadEnter") && !event.shiftKey && !event.isComposing) {
            return handleEnterKey(event);
          }
          return listener.apply(this, arguments);
        };
        return originalAddEventListener.call(this, type, wrappedListener, options);
      }
      return originalAddEventListener.apply(this, arguments);
    };
    document.addEventListener("submit", (e) => {
      console.log("[ContentScript] Form submit event intercepted:", e.target);
      e.preventDefault();
      e.stopPropagation();
      return false;
    }, true);
    const form = document.querySelector(CHATGPT_SELECTORS.FORM);
    if (form) {
      console.log("[ContentScript] Adding submit listener to ChatGPT form");
      form.addEventListener("submit", (e) => {
        console.log("[ContentScript] ChatGPT form submit intercepted");
        e.preventDefault();
        e.stopPropagation();
        return false;
      }, true);
    }
  };
  const handleSubmitButtonVisibility = async (mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === "childList") {
        const toggleSwitch = document.querySelector("#maxmemory-toggle");
        syncMaxMemoryToggleUI(toggleSwitch ? toggleSwitch.checked : true);
      }
    });
  };
  async function init() {
    if (window.memoryVaultInitialized) {
      console.log("MaxMemory already initialized on this page, skipping");
      return;
    }
    window.memoryVaultInitialized = true;
    console.log("Initializing MaxMemory extension");
    let currentObservedChatId = getChatId();
    ObserverManager.init({
      onMessagesAdded: (mutations) => {
        scheduleStyleMemoriesInChat();
        for (const mutation of mutations) {
          const targetMessageContainer = getClosestUserMessageContainer(mutation.target);
          if (targetMessageContainer) {
            handleMessageStyling(targetMessageContainer);
          }
          for (const node of mutation.addedNodes) {
            if (!node || node.nodeType !== Node.ELEMENT_NODE) continue;
            if (node.matches && node.matches('article, [data-message-author-role="user"]')) {
              handleMessageStyling(node);
            } else if (node.querySelectorAll) {
              getUserMessageContainers(node).forEach(handleMessageStyling);
              const closestMessageContainer = getClosestUserMessageContainer(node);
              if (closestMessageContainer) {
                handleMessageStyling(closestMessageContainer);
              }
            }
          }
        }
      },
      onInputAreaChanged: (mutations) => {
        for (const mutation of mutations) {
          if (mutation.type === "childList" && (mutation.addedNodes.length || mutation.removedNodes.length)) {
            addGetMemoriesButton();
            setupInputListeners();
          }
        }
      },
      onSubmitButtonChanged: handleSubmitButtonVisibility,
      onUIReady: () => {
        console.log("[ObserverManager] UI is ready. Setting up listeners and components.");
        scheduleStyleMemoriesInChat();
        addGetMemoriesButton();
        setupInputListeners();
        setupEnterKeyPrevention();
      }
    });
    ObserverManager.start();
    let lastUrl = window.location.href;
    const checkForNavigation = async () => {
      const currentUrl = window.location.href;
      if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;
        const chatId = getChatId();
        if (chatId !== currentObservedChatId) {
          currentObservedChatId = chatId;
          console.log("Navigation detected, reinitializing UI components");
          const existingContainer = document.getElementById("maxmemory-container");
          if (existingContainer) {
            existingContainer.remove();
          }
          ObserverManager.stop();
          ObserverManager.start();
        }
      }
    };
    setInterval(checkForNavigation, 1e3);
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.type === "TAB_READY") {
        console.log("Tab is ready");
        sendResponse({ ready: true });
      }
      return true;
    });
  }
  const cleanup = () => {
    console.log("[ContentScript] Cleaning up observers before page unload");
    ObserverManager.stop();
    window.memoryVaultObserversInitialized = false;
    window.memoryVaultInitialized = false;
  };
  window.addEventListener("beforeunload", cleanup);
  window.addEventListener("pagehide", cleanup);
  init();
  document.addEventListener("DOMContentLoaded", init);
  console.log("[ContentScript] MaxMemory initialized with efficient ObserverManager - no more expensive document.body observers!");
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "DISPLAY_EXTRACTED_MEMORIES") {
      displayExtractedMemories(request.memories, request.savedToDatabase, request.limitType);
      sendResponse({ status: "success" });
    } else if (request.type === "DISPLAY_MEMORY_SUGGESTIONS") {
      displayMemorySuggestions(
        request.suggestions,
        request.detectedMode,
        request.extractedWhileAtLimit || false,
        request.limitType || null
      );
      sendResponse({ status: "success" });
    } else if (request.type === "DISPLAY_MEMORY_LIMIT_WARNING") {
      displayMemoryLimitWarning(request.limitType, request.current, request.limit);
      sendResponse({ status: "success" });
    } else if (request.type === "GET_CONVERSATION_HISTORY") {
      const history = scrapeConversationHistory(request.count);
      sendResponse({ status: "success", history });
    } else if (request.type === "MAXMEMORY_ENABLED_STATE_CHANGED") {
      syncMaxMemoryToggleUI(request.enabled);
      sendResponse({ status: "success" });
    }
    return true;
  });
  const displayMemoryLimitWarning = (limitType, current, limit) => {
    const messages = getUserMessageContainers();
    const latestUserMessage = Array.from(messages).reverse().find((msg) => {
      const messageDiv = getMessageContentElement(msg);
      return messageDiv && !messageDiv.textContent.includes("[RELEVANT_PAST_MEMORIES_START]");
    });
    if (!latestUserMessage) {
      console.log("Could not find latest user message to display warning");
      return;
    }
    if (latestUserMessage.querySelector(".memory-limit-warning")) {
      return;
    }
    const warningDiv = document.createElement("div");
    warningDiv.innerHTML = uiBlueprints.getMemoryLimitWarning(limitType, current, limit);
    const warningElement = warningDiv.firstElementChild;
    const signInButton = warningElement.querySelector(".memory-warning-button");
    signInButton.addEventListener("click", (e) => {
      e.stopPropagation();
      backgroundAPI.trackPopupOpened("memory_limit_warning");
      backgroundAPI.openPopupInTab();
    });
    latestUserMessage.appendChild(warningElement);
  };
  const displayExtractedMemories = (memories, savedToDatabase = true, limitType = null) => {
    console.log("Displaying extracted memories:", memories, "savedToDatabase:", savedToDatabase, "limitType:", limitType);
    const userMessages = document.querySelectorAll('[data-message-author-role="user"]');
    if (userMessages.length === 0) {
      console.log("No user messages found to attach memories to");
      return;
    }
    const latestUserMessage = userMessages[userMessages.length - 1];
    const messageId = latestUserMessage.getAttribute("data-message-id");
    const existingNotification = latestUserMessage.querySelector(".extracted-memory-notification");
    if (existingNotification) {
      console.log("Memory notification already exists for this message");
      return;
    }
    const notificationHTML = uiBlueprints.getExtractedMemoryNotification(memories);
    latestUserMessage.insertAdjacentHTML("beforeend", notificationHTML);
    const memoryNotification = latestUserMessage.querySelector(".extracted-memory-notification");
    memoryNotification.setAttribute("data-message-id", messageId);
    const prefixText = memoryNotification.querySelector(".memory-prefix-text");
    const memoryText = memoryNotification.querySelector(".extracted-memory-text");
    if (memoryText) {
      const normalizedMemories = memories.map((memory) => typeof memory === "string" ? memory : memory.memory || memory.text || "").filter(Boolean);
      memoryText.textContent = normalizedMemories.join(" • ");
    }
    if (savedToDatabase) {
      prefixText.addEventListener("click", (e) => {
        e.stopPropagation();
        backgroundAPI.openPopup();
      });
      return;
    }
    prefixText.textContent = "memory extracted:";
    prefixText.style.cursor = "default";
    prefixText.classList.remove("memory-prefix-text");
    const warningSection = document.createElement("div");
    const isGuestLimit = limitType === "guest";
    warningSection.className = `memory-limit-warning ${isGuestLimit ? "memory-limit-warning--guest" : "memory-limit-warning--logged-in"}`;
    const warningIcon = document.createElement("div");
    warningIcon.className = "memory-warning-icon";
    warningIcon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="12" y1="9" x2="12" y2="13" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="12" y1="17" x2="12.01" y2="17" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;
    const warningText = document.createElement("div");
    warningText.className = "memory-warning-text";
    warningText.textContent = isGuestLimit ? "We extracted this memory, but couldn't save it because you've reached the guest limit. Sign in to unlock 100 free memories." : "We extracted this memory, but couldn't save it because you've reached your free limit. Upgrade to keep saving automatically.";
    const warningButton = document.createElement("button");
    warningButton.className = `memory-warning-button ${isGuestLimit ? "memory-warning-button--guest" : "memory-warning-button--logged-in"}`;
    warningButton.textContent = isGuestLimit ? "Sign in" : "Upgrade";
    warningButton.addEventListener("click", (e) => {
      e.stopPropagation();
      backgroundAPI.trackPopupOpened("memory_limit_warning");
      backgroundAPI.openPopupInTab();
    });
    warningSection.appendChild(warningIcon);
    warningSection.appendChild(warningText);
    warningSection.appendChild(warningButton);
    memoryNotification.appendChild(warningSection);
  };
  const updateAutoSavedContainerState = (suggestionsContainer, savedCount, failedCount = 0) => {
    const titleElement = suggestionsContainer.querySelector(".memory-suggestions-title");
    if (titleElement) {
      let title = `Saved ${savedCount} ${savedCount === 1 ? "memory" : "memories"}`;
      if (failedCount > 0) {
        title += ` (${failedCount} failed)`;
      }
      titleElement.textContent = title;
    }
    const bulkUndoButton = suggestionsContainer.querySelector(".discard-all-button");
    if (bulkUndoButton) {
      bulkUndoButton.style.display = savedCount > 1 ? "block" : "none";
    }
  };
  const removeSuggestionContainerIfEmpty = (suggestionsContainer) => {
    if (!suggestionsContainer) return;
    const remainingItems = suggestionsContainer.querySelectorAll(".memory-suggestion-item");
    if (remainingItems.length === 0) {
      suggestionsContainer.remove();
    } else {
      updateAutoSavedContainerState(suggestionsContainer, remainingItems.length);
    }
  };
  const handleUndoMemory = async (savedMemory, suggestionItem, suggestionsContainer) => {
    try {
      const undoButton = suggestionItem.querySelector(".undo-button");
      if (undoButton) {
        undoButton.disabled = true;
        undoButton.textContent = "Undoing...";
      }
      suggestionItem.style.opacity = "0.7";
      const response = await backgroundAPI.deleteMemory(savedMemory.id, savedMemory.text);
      if (response.status !== "success") {
        throw new Error(response.message || "Failed to undo memory");
      }
      suggestionItem.style.transition = "all 0.2s ease";
      suggestionItem.style.opacity = "0";
      suggestionItem.style.transform = "translateY(-6px)";
      setTimeout(() => {
        suggestionItem.remove();
        removeSuggestionContainerIfEmpty(suggestionsContainer);
      }, 200);
      console.log("Auto-saved memory undone:", savedMemory.text);
    } catch (error) {
      console.error("Error undoing auto-saved memory:", error);
      suggestionItem.classList.add("memory-suggestion-item--failed");
      suggestionItem.style.opacity = "1";
      const undoButton = suggestionItem.querySelector(".undo-button");
      if (undoButton) {
        undoButton.disabled = false;
        undoButton.textContent = "Undo";
      }
    }
  };
  const handleUndoAll = async (suggestionsContainer) => {
    var _a, _b;
    const suggestionItems = Array.from(suggestionsContainer.querySelectorAll(".memory-suggestion-item"));
    for (const suggestionItem of suggestionItems) {
      const savedMemory = {
        id: suggestionItem.getAttribute("data-memory-id"),
        text: ((_b = (_a = suggestionItem.querySelector(".suggestion-text")) == null ? void 0 : _a.textContent) == null ? void 0 : _b.replace(/\s*\(saved\)\s*$/i, "").trim()) || ""
      };
      if (!savedMemory.id) {
        continue;
      }
      await handleUndoMemory(savedMemory, suggestionItem, suggestionsContainer);
    }
  };
  const displayMemorySuggestions = async (suggestions, detectedMode = null, extractedWhileAtLimit = false, limitType = null) => {
    console.log("Displaying memory suggestions:", suggestions, "with detected mode:", detectedMode, "extractedWhileAtLimit:", extractedWhileAtLimit, "limitType:", limitType);
    if (!suggestions || suggestions.length === 0) {
      console.log("No memory suggestions to display");
      return;
    }
    if (extractedWhileAtLimit) {
      displayExtractedMemories(suggestions, false, limitType);
      return;
    }
    const userMessages = document.querySelectorAll('[data-message-author-role="user"]');
    if (userMessages.length === 0) {
      console.log("No user messages found to attach suggestions to");
      return;
    }
    const latestUserMessage = userMessages[userMessages.length - 1];
    const messageId = latestUserMessage.getAttribute("data-message-id");
    const existingSuggestions = latestUserMessage.querySelector(".memory-suggestions-container");
    if (existingSuggestions) {
      console.log("Memory suggestions already exist for this message");
      return;
    }
    const containerHTML = uiBlueprints.getMemorySuggestionsContainer(
      messageId,
      suggestions.length,
      detectedMode,
      `Saving ${suggestions.length} ${suggestions.length === 1 ? "memory" : "memories"}...`,
      "Undo all"
    );
    latestUserMessage.insertAdjacentHTML("beforeend", containerHTML);
    const suggestionsContainer = latestUserMessage.querySelector('.memory-suggestions-container[data-message-id="' + messageId + '"]');
    const suggestionsList = suggestionsContainer.querySelector(".memory-suggestions-list");
    const memoriesToSave = suggestions.map((suggestion) => ({
      text: typeof suggestion === "string" ? suggestion : suggestion.memory || "",
      tag: typeof suggestion === "object" ? suggestion.tag || null : null,
      wasEdited: false,
      originalContent: typeof suggestion === "string" ? suggestion : suggestion.memory || "",
      isAutoApplied: true
    })).filter((memory) => memory.text);
    try {
      const response = await chrome.runtime.sendMessage({
        type: "SAVE_APPROVED_MEMORIES",
        memories: memoriesToSave,
        mode: detectedMode
      });
      if (response.status !== "success") {
        throw new Error(response.message || "Failed to auto-save memories");
      }
      const savedMemories = response.saved || [];
      const failedMemories = response.failed || [];
      if (savedMemories.length === 0) {
        suggestionsContainer.remove();
        return;
      }
      suggestionsList.innerHTML = "";
      savedMemories.forEach((savedMemory, index) => {
        const itemHTML = uiBlueprints.getAutoSavedMemoryItem(savedMemory, index);
        suggestionsList.insertAdjacentHTML("beforeend", itemHTML);
        const suggestionItem = suggestionsList.querySelector(`[data-index="${index}"]`);
        const undoButton = suggestionItem.querySelector(".undo-button");
        undoButton.addEventListener("click", async (e) => {
          e.stopPropagation();
          await handleUndoMemory(savedMemory, suggestionItem, suggestionsContainer);
        });
      });
      updateAutoSavedContainerState(suggestionsContainer, savedMemories.length, failedMemories.length);
      const discardAllButton = suggestionsContainer.querySelector(".discard-all-button");
      discardAllButton.addEventListener("click", async (e) => {
        e.stopPropagation();
        await handleUndoAll(suggestionsContainer);
      });
    } catch (error) {
      console.error("Error auto-saving memories:", error);
      suggestionsContainer.remove();
      if (/memory limit/i.test(error.message || "")) {
        displayExtractedMemories(memoriesToSave.map((memory) => memory.text), false, limitType);
      }
    }
  };
  function scrapeConversationHistory(numMessages = 4) {
    console.log(`[Scraper] Scraping last ${numMessages} messages.`);
    const messages = [];
    const messageNodes = document.querySelectorAll("[data-message-author-role]");
    const recentNodes = Array.from(messageNodes).slice(-numMessages);
    for (const node of recentNodes) {
      const role = node.getAttribute("data-message-author-role");
      if (!role) continue;
      const clonedNode = node.cloneNode(true);
      let retrievedMemoriesText = null;
      const messageDiv = getMessageContentElement(clonedNode);
      if (messageDiv && messageDiv.hasAttribute("data-full-memories")) {
        retrievedMemoriesText = messageDiv.getAttribute("data-full-memories");
        console.log(`[Scraper] Success: Found and extracted full memories from data-attribute.`);
        console.log({ retrievedMemoriesText });
      } else {
        const memorySectionNode = clonedNode.querySelector(".memory-section");
        if (memorySectionNode) {
          retrievedMemoriesText = memorySectionNode.textContent.trim();
          console.warn(`[Scraper] Fallback: Scraping from textContent. Data may be truncated.`);
          console.log({ retrievedMemoriesText });
        }
      }
      clonedNode.querySelectorAll(".memory-section, .memory-suggestions-container, .extracted-memory-notification, .memory-limit-warning").forEach((el) => el.remove());
      const originalUserText = clonedNode.textContent.trim();
      if (originalUserText || retrievedMemoriesText) {
        const messageObject = {
          role,
          text: originalUserText
        };
        if (retrievedMemoriesText) {
          messageObject.retrievedMemories = retrievedMemoriesText;
        }
        messages.push(messageObject);
      }
    }
    console.log(`[Scraper] Scraped ${messages.length} valid messages.`);
    console.log({ messages });
    return messages;
  }
  const getInputContent = (inputBox) => {
    return inputBox.tagName === "TEXTAREA" ? inputBox.value.trim() : Array.from(inputBox.querySelectorAll("p")).map((p) => p.textContent.trim()).join("\n");
  };
  const setInputContent = (inputBox, content) => {
    console.log("[ContentScript] Setting input content, type:", inputBox.tagName);
    if (inputBox.tagName === "TEXTAREA") {
      inputBox.value = content;
      const inputEvent = new Event("input", { bubbles: true });
      inputBox.dispatchEvent(inputEvent);
    } else {
      inputBox.innerHTML = `<p>${content}</p>`;
      const inputEvent = new Event("input", { bubbles: true });
      inputBox.dispatchEvent(inputEvent);
    }
    inputBox.focus();
    const range = document.createRange();
    range.selectNodeContents(inputBox);
    range.collapse(false);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    console.log("[ContentScript] Input content set, length:", content.length);
  };
})();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udGVudFNjcmlwdC5qcyIsInNvdXJjZXMiOlsiLi4vLi4vanMvY29udGVudFNjcmlwdC5qcyJdLCJzb3VyY2VzQ29udGVudCI6WyIvLyBjb250ZW50U2NyaXB0LmpzXHJcblxyXG4oZnVuY3Rpb24oKSB7XHJcbiAgICAvLyBTZWxlY3RvciBjb25zdGFudHMgLSBlYXN5IHRvIGNvbmZpZ3VyZVxyXG4gICAgY29uc3QgQ0hBVEdQVF9TRUxFQ1RPUlMgPSB7XHJcbiAgICAgICAgSU5QVVRfQk9YOiAnI3Byb21wdC10ZXh0YXJlYScsXHJcbiAgICAgICAgRk9STTogJ2Zvcm1bZGF0YS10eXBlPVwidW5pZmllZC1jb21wb3NlclwiXScsXHJcbiAgICAgICAgU1VCTUlUX0JVVFRPTjogJ2J1dHRvbltkYXRhLXRlc3RpZD1cInNlbmQtYnV0dG9uXCJdJ1xyXG4gICAgfTtcclxuXHJcbiAgICAvLyBGdW5jdGlvbiB0byBjaGVjayBtZW1vcnkgbGltaXQgYW5kIHVwZGF0ZSBzZXR0aW5ncyBidXR0b24gd2l0aCByZWQgZG90IGluZGljYXRvclxyXG4gICAgY29uc3QgY2hlY2tNZW1vcnlMaW1pdEFuZFVwZGF0ZUJ1dHRvbiA9IGFzeW5jIChzZXR0aW5nc0J1dHRvbikgPT4ge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGNvbnN0IG1lbW9yeUluZm8gPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldE1lbW9yeUxpbWl0SW5mbygpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgaWYgKG1lbW9yeUluZm8uc3RhdHVzID09PSAnc3VjY2VzcycgJiYgIW1lbW9yeUluZm8uY2FuQWRkKSB7XHJcbiAgICAgICAgICAgICAgICAvLyBNZW1vcnkgbGltaXQgcmVhY2hlZCwgYWRkIHJlZCBkb3QgaW5kaWNhdG9yXHJcbiAgICAgICAgICAgICAgICBhZGRSZWREb3RUb0J1dHRvbihzZXR0aW5nc0J1dHRvbik7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBjaGVja2luZyBtZW1vcnkgbGltaXQgZm9yIHNldHRpbmdzIGJ1dHRvbjonLCBlcnJvcik7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuXHJcbiAgICAvLyBGdW5jdGlvbiB0byBhZGQgcmVkIGRvdCBpbmRpY2F0b3IgdG8gc2V0dGluZ3MgYnV0dG9uXHJcbiAgICBjb25zdCBhZGRSZWREb3RUb0J1dHRvbiA9IChidXR0b24pID0+IHtcclxuICAgICAgICAvLyBDaGVjayBpZiByZWQgZG90IGFscmVhZHkgZXhpc3RzXHJcbiAgICAgICAgaWYgKGJ1dHRvbi5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LWxpbWl0LWluZGljYXRvcicpKSB7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGNvbnN0IHJlZERvdCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xyXG4gICAgICAgIHJlZERvdC5jbGFzc05hbWUgPSAnbWVtb3J5LWxpbWl0LWluZGljYXRvcic7XHJcbiAgICAgICAgXHJcbiAgICAgICAgYnV0dG9uLmFwcGVuZENoaWxkKHJlZERvdCk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVXBkYXRlIGJ1dHRvbiB0aXRsZSB0byBpbmRpY2F0ZSBtZW1vcnkgbGltaXRcclxuICAgICAgICBidXR0b24udGl0bGUgPSAnTWVtb3J5IGxpbWl0IHJlYWNoZWQhIENsaWNrIHRvIG1hbmFnZSB5b3VyIG1lbW9yaWVzIGFuZCBzaWduIGluIGZvciB1bmxpbWl0ZWQgc3RvcmFnZSc7XHJcbiAgICB9O1xyXG5cclxuXHJcblxyXG4gICAgLy8gVUkgQmx1ZXByaW50cyBNb2R1bGUgLSBTaW5nbGUgc291cmNlIG9mIHRydXRoIGZvciBIVE1MIHN0cnVjdHVyZXNcclxuICAgIGNvbnN0IHVpQmx1ZXByaW50cyA9IHtcclxuICAgICAgICBnZXRNZW1vcnlTdWdnZXN0aW9uc0NvbnRhaW5lcjogKG1lc3NhZ2VJZCwgc3VnZ2VzdGlvbnNDb3VudCwgZGV0ZWN0ZWRNb2RlID0gbnVsbCwgaGVhZGVyVGV4dCA9IG51bGwsIGJ1bGtBY3Rpb25MYWJlbCA9ICdVbmRvIGFsbCcpID0+IGBcclxuICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1lbW9yeS1zdWdnZXN0aW9ucy1jb250YWluZXJcIiBkYXRhLW1lc3NhZ2UtaWQ9XCIke21lc3NhZ2VJZH1cIj5cclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbnMtaGVhZGVyXCI+XHJcbiAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1lbW9yeS1zdWdnZXN0aW9ucy1oZWFkZXItaWNvblwiPiR7Z2V0TWVtb3JpZXNTVkcoJyM2Yzc1N2QnLCAxNCl9PC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbnMtdGl0bGVcIj4ke2hlYWRlclRleHQgfHwgYFNhdmVkICR7c3VnZ2VzdGlvbnNDb3VudH0gJHtzdWdnZXN0aW9uc0NvdW50ID09PSAxID8gJ21lbW9yeScgOiAnbWVtb3JpZXMnfWB9PC9zcGFuPlxyXG4gICAgICAgICAgICAgICAgICAgICR7ZGV0ZWN0ZWRNb2RlID8gYDxzcGFuIGNsYXNzPVwiZGV0ZWN0ZWQtbW9kZS1sYWJlbFwiPiR7ZGV0ZWN0ZWRNb2RlfTwvc3Bhbj5gIDogJyd9XHJcbiAgICAgICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbnMtbGlzdFwiPjwvZGl2PlxyXG4gICAgICAgICAgICAgICAgJHtidWxrQWN0aW9uTGFiZWwgPyBgPGJ1dHRvbiBjbGFzcz1cImRpc2NhcmQtYWxsLWJ1dHRvblwiPiR7YnVsa0FjdGlvbkxhYmVsfTwvYnV0dG9uPmAgOiAnJ31cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgYCxcclxuICAgICAgICBcclxuICAgICAgICBnZXRNZW1vcnlTdWdnZXN0aW9uSXRlbTogKHN1Z2dlc3Rpb24sIGluZGV4KSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IG1lbW9yeVRleHQgPSB0eXBlb2Ygc3VnZ2VzdGlvbiA9PT0gJ3N0cmluZycgPyBzdWdnZXN0aW9uIDogKHN1Z2dlc3Rpb24ubWVtb3J5IHx8ICcnKTtcclxuICAgICAgICAgICAgY29uc3QgdGFnVGV4dCA9IHR5cGVvZiBzdWdnZXN0aW9uID09PSAnb2JqZWN0JyA/IChzdWdnZXN0aW9uLnRhZyB8fCAnJykgOiAnJztcclxuICAgICAgICAgICAgcmV0dXJuIGBcclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbi1pdGVtXCIgZGF0YS1pbmRleD1cIiR7aW5kZXh9XCI+XHJcbiAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cInN1Z2dlc3Rpb24tY29udGVudC13cmFwcGVyXCIgc3R5bGU9XCJkaXNwbGF5OiBmbGV4OyBmbGV4LWRpcmVjdGlvbjogY29sdW1uOyBnYXA6IDJweDsgZmxleDogMTsgbWluLXdpZHRoOiAwOyBwYWRkaW5nLXJpZ2h0OiA4cHg7XCI+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJzdWdnZXN0aW9uLXRleHRcIj4ke21lbW9yeVRleHR9PC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICR7dGFnVGV4dCA/IGA8ZGl2IGNsYXNzPVwic3VnZ2VzdGlvbi10YWctYmFkZ2VcIj4ke3RhZ1RleHR9PC9kaXY+YCA6ICcnfVxyXG4gICAgICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJzdWdnZXN0aW9uLWJ1dHRvbnNcIj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz1cImFwcHJvdmUtYnV0dG9uXCIgdGl0bGU9XCJBcHByb3ZlIGFuZCBzYXZlIHRoaXMgbWVtb3J5XCI+4pyTPC9idXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIDxidXR0b24gY2xhc3M9XCJlZGl0LWJ1dHRvblwiIHRpdGxlPVwiRWRpdCB0aGlzIG1lbW9yeVwiPiR7Z2V0RWRpdFNWRygnIzZjNzU3ZCcsIDEyKX08L2J1dHRvbj5cclxuICAgICAgICAgICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICBgO1xyXG4gICAgICAgIH0sXHJcblxyXG4gICAgICAgIGdldEF1dG9TYXZlZE1lbW9yeUl0ZW06IChzYXZlZE1lbW9yeSwgaW5kZXgpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgbWVtb3J5VGV4dCA9IHNhdmVkTWVtb3J5Py50ZXh0IHx8ICcnO1xyXG4gICAgICAgICAgICBjb25zdCB0YWdUZXh0ID0gc2F2ZWRNZW1vcnk/LnRhZyB8fCAnJztcclxuICAgICAgICAgICAgcmV0dXJuIGBcclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbi1pdGVtIG1lbW9yeS1zdWdnZXN0aW9uLWl0ZW0tLXNhdmVkXCIgZGF0YS1pbmRleD1cIiR7aW5kZXh9XCIgZGF0YS1tZW1vcnktaWQ9XCIke3NhdmVkTWVtb3J5Py5pZCB8fCAnJ31cIj5cclxuICAgICAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPVwic3VnZ2VzdGlvbi1jb250ZW50LXdyYXBwZXJcIiBzdHlsZT1cImRpc3BsYXk6IGZsZXg7IGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47IGdhcDogMnB4OyBmbGV4OiAxOyBtaW4td2lkdGg6IDA7IHBhZGRpbmctcmlnaHQ6IDhweDtcIj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cInN1Z2dlc3Rpb24tdGV4dFwiPiR7bWVtb3J5VGV4dH08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgJHt0YWdUZXh0ID8gYDxkaXYgY2xhc3M9XCJzdWdnZXN0aW9uLXRhZy1iYWRnZVwiPiR7dGFnVGV4dH08L2Rpdj5gIDogJyd9XHJcbiAgICAgICAgICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cInN1Z2dlc3Rpb24tYnV0dG9uc1wiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8YnV0dG9uIGNsYXNzPVwidW5kby1idXR0b25cIiB0aXRsZT1cIkRlbGV0ZSB0aGlzIG1lbW9yeSBhbmQgdW5kbyB0aGUgYXV0by1zYXZlXCI+VW5kbzwvYnV0dG9uPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICAgIGA7XHJcbiAgICAgICAgfSxcclxuXHJcbiAgICAgICAgZ2V0UmVhZE9ubHlNZW1vcnlJdGVtOiAobWVtb3J5LCBpbmRleCwgaXRlbUNsYXNzID0gJycpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgbWVtb3J5VGV4dCA9IHR5cGVvZiBtZW1vcnkgPT09ICdzdHJpbmcnID8gbWVtb3J5IDogKG1lbW9yeS5tZW1vcnkgfHwgbWVtb3J5LnRleHQgfHwgJycpO1xyXG4gICAgICAgICAgICBjb25zdCB0YWdUZXh0ID0gdHlwZW9mIG1lbW9yeSA9PT0gJ29iamVjdCcgPyAobWVtb3J5LnRhZyB8fCAnJykgOiAnJztcclxuICAgICAgICAgICAgcmV0dXJuIGBcclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktc3VnZ2VzdGlvbi1pdGVtICR7aXRlbUNsYXNzfVwiIGRhdGEtaW5kZXg9XCIke2luZGV4fVwiPlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJzdWdnZXN0aW9uLWNvbnRlbnQtd3JhcHBlclwiIHN0eWxlPVwiZGlzcGxheTogZmxleDsgZmxleC1kaXJlY3Rpb246IGNvbHVtbjsgZ2FwOiAycHg7IGZsZXg6IDE7IG1pbi13aWR0aDogMDsgcGFkZGluZy1yaWdodDogOHB4O1wiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPVwic3VnZ2VzdGlvbi10ZXh0XCI+JHttZW1vcnlUZXh0fTwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAke3RhZ1RleHQgPyBgPGRpdiBjbGFzcz1cInN1Z2dlc3Rpb24tdGFnLWJhZGdlXCI+JHt0YWdUZXh0fTwvZGl2PmAgOiAnJ31cclxuICAgICAgICAgICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICBgO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgXHJcbiAgICAgICAgZ2V0TWVtb3J5RWRpdEZpZWxkOiAob3JpZ2luYWxUZXh0KSA9PiBgXHJcbiAgICAgICAgICAgIDx0ZXh0YXJlYSBjbGFzcz1cIm1lbW9yeS1lZGl0LWZpZWxkXCIgcGxhY2Vob2xkZXI9XCJFZGl0IG1lbW9yeS4uLlwiPiR7b3JpZ2luYWxUZXh0fTwvdGV4dGFyZWE+XHJcbiAgICAgICAgYCxcclxuICAgICAgICBcclxuICAgICAgICBnZXRFeHRyYWN0ZWRNZW1vcnlOb3RpZmljYXRpb246IChtZW1vcmllcykgPT4gYFxyXG4gICAgICAgICAgICA8ZGl2IGNsYXNzPVwiZXh0cmFjdGVkLW1lbW9yeS1ub3RpZmljYXRpb25cIj5cclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktcHJlZml4XCI+XHJcbiAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1lbW9yeS1wcmVmaXgtaWNvblwiPiR7Z2V0TWVtb3JpZXNTVkcoJyNkMWQ1ZGInLCAxMil9PC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9XCJtZW1vcnktcHJlZml4LXRleHRcIj5tZW1vcnkgc2F2ZWQ6PC9zcGFuPlxyXG4gICAgICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPVwiZXh0cmFjdGVkLW1lbW9yeS10ZXh0XCI+JHttZW1vcmllcy5qb2luKCcg4oCiICcpfTwvZGl2PlxyXG4gICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICBgLFxyXG5cclxuICAgICAgICBnZXRMaW1pdEJsb2NrZWRXYXJuaW5nOiAobGltaXRUeXBlKSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IGlzR3Vlc3RMaW1pdCA9IGxpbWl0VHlwZSA9PT0gJ2d1ZXN0JztcclxuICAgICAgICAgICAgY29uc3Qgd2FybmluZ0NsYXNzID0gYG1lbW9yeS1saW1pdC13YXJuaW5nICR7aXNHdWVzdExpbWl0ID8gJ21lbW9yeS1saW1pdC13YXJuaW5nLS1ndWVzdCcgOiAnbWVtb3J5LWxpbWl0LXdhcm5pbmctLWxvZ2dlZC1pbid9YDtcclxuICAgICAgICAgICAgY29uc3QgYnV0dG9uQ2xhc3MgPSBgbWVtb3J5LXdhcm5pbmctYnV0dG9uICR7aXNHdWVzdExpbWl0ID8gJ21lbW9yeS13YXJuaW5nLWJ1dHRvbi0tZ3Vlc3QnIDogJ21lbW9yeS13YXJuaW5nLWJ1dHRvbi0tbG9nZ2VkLWluJ31gO1xyXG4gICAgICAgICAgICBjb25zdCB3YXJuaW5nVGV4dCA9IGlzR3Vlc3RMaW1pdFxyXG4gICAgICAgICAgICAgICAgPyBcIldlIGV4dHJhY3RlZCB0aGlzIG1lbW9yeSwgYnV0IGNvdWxkbid0IHNhdmUgaXQgYmVjYXVzZSB5b3UndmUgcmVhY2hlZCB0aGUgZ3Vlc3QgbGltaXQuIFNpZ24gaW4gdG8gdW5sb2NrIDEwMCBmcmVlIG1lbW9yaWVzLlwiXHJcbiAgICAgICAgICAgICAgICA6IFwiV2UgZXh0cmFjdGVkIHRoaXMgbWVtb3J5LCBidXQgY291bGRuJ3Qgc2F2ZSBpdCBiZWNhdXNlIHlvdSd2ZSByZWFjaGVkIHlvdXIgZnJlZSBsaW1pdC4gVXBncmFkZSB0byBrZWVwIHNhdmluZyBhdXRvbWF0aWNhbGx5LlwiO1xyXG5cclxuICAgICAgICAgICAgcmV0dXJuIGBcclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCIke3dhcm5pbmdDbGFzc31cIj5cclxuICAgICAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPVwibWVtb3J5LXdhcm5pbmctaWNvblwiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8c3ZnIHdpZHRoPVwiMTRcIiBoZWlnaHQ9XCIxNFwiIHZpZXdCb3g9XCIwIDAgMjQgMjRcIiBmaWxsPVwibm9uZVwiIHN0cm9rZT1cImN1cnJlbnRDb2xvclwiIHN0cm9rZS13aWR0aD1cIjJcIj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9XCJNMTAuMjkgMy44NkwxLjgyIDE4YTIgMiAwIDAwMS43MSAzaDE2Ljk0YTIgMiAwIDAwMS43MS0zTDEzLjcxIDMuODZhMiAyIDAgMDAtMy40MiAwelwiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPVwiMTJcIiB5MT1cIjlcIiB4Mj1cIjEyXCIgeTI9XCIxM1wiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPVwiMTJcIiB5MT1cIjE3XCIgeDI9XCIxMi4wMVwiIHkyPVwiMTdcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgc3Ryb2tlLWxpbmVqb2luPVwicm91bmRcIi8+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIDwvc3ZnPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktd2FybmluZy10ZXh0XCI+JHt3YXJuaW5nVGV4dH08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICA8YnV0dG9uIGNsYXNzPVwiJHtidXR0b25DbGFzc31cIj4ke2lzR3Vlc3RMaW1pdCA/ICdTaWduIGluJyA6ICdVcGdyYWRlJ308L2J1dHRvbj5cclxuICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICBgO1xyXG4gICAgICAgIH0sXHJcblxyXG4gICAgICAgIGdldE1haW5Db250YWluZXI6ICgpID0+IGBcclxuICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1heG1lbW9yeS1tYWluLWNvbnRhaW5lclwiPlxyXG4gICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1heG1lbW9yeS1icmFuZFwiPlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtYXhtZW1vcnktbG9nb1wiPiR7Z2V0TWF4TWVtb3J5TG9nb1NWRygnIzZiNzI4MCcsIDEyKX08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz1cIm1heG1lbW9yeS1icmFuZC10ZXh0XCI+TWF4TWVtb3J5PC9zcGFuPlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtYXhtZW1vcnktdG9nZ2xlLWNvbnRhaW5lclwiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8bGFiZWwgY2xhc3M9XCJtYXhtZW1vcnktdG9nZ2xlLXN3aXRjaFwiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGlucHV0IHR5cGU9XCJjaGVja2JveFwiIGlkPVwibWF4bWVtb3J5LXRvZ2dsZVwiIGNoZWNrZWQ+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz1cIm1heG1lbW9yeS10b2dnbGUtc2xpZGVyXCI+PC9zcGFuPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8L2xhYmVsPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgIDxidXR0b24gY2xhc3M9XCJtYXhtZW1vcnktc2V0dGluZ3MtYnV0dG9uXCIgdGl0bGU9XCJPcGVuIE1heE1lbW9yeSBzZXR0aW5ncyBhbmQgbWVtb3JpZXNcIj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgJHtnZXRTZXR0aW5nc1NWRygnIzg4OCcsIDE0KX1cclxuICAgICAgICAgICAgICAgICAgICA8L2J1dHRvbj5cclxuICAgICAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz1cImdldC1tZW1vcmllcy1idXR0b25cIiBpZD1cImdldC1tZW1vcmllcy1idXR0b25cIiBzdHlsZT1cImRpc3BsYXk6IG5vbmU7XCI+XHJcbiAgICAgICAgICAgICAgICAgICAgJHtnZXRNZW1vcmllc1NWRygnIzQwNDE0ZicpfTxzcGFuPlN1Ym1pdDwvc3Bhbj5cclxuICAgICAgICAgICAgICAgIDwvYnV0dG9uPlxyXG4gICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICBgLFxyXG5cclxuICAgICAgICBnZXRCcmFuZFRleHQ6ICgpID0+IGBcclxuICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1heG1lbW9yeS1icmFuZFwiPlxyXG4gICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1heG1lbW9yeS1sb2dvXCI+JHtnZXRNYXhNZW1vcnlMb2dvU1ZHKCcjNmI3MjgwJywgMTIpfTwvZGl2PlxyXG4gICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9XCJtYXhtZW1vcnktYnJhbmQtdGV4dFwiPk1heE1lbW9yeTwvc3Bhbj5cclxuICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtYXhtZW1vcnktdG9nZ2xlLWNvbnRhaW5lclwiPlxyXG4gICAgICAgICAgICAgICAgICAgIDxsYWJlbCBjbGFzcz1cIm1heG1lbW9yeS10b2dnbGUtc3dpdGNoXCI+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIDxpbnB1dCB0eXBlPVwiY2hlY2tib3hcIiBpZD1cIm1heG1lbW9yeS10b2dnbGVcIiBjaGVja2VkPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz1cIm1heG1lbW9yeS10b2dnbGUtc2xpZGVyXCI+PC9zcGFuPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvbGFiZWw+XHJcbiAgICAgICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgYCxcclxuXHJcbiAgICAgICAgZ2V0U2V0dGluZ3NCdXR0b246ICgpID0+IGBcclxuICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz1cIm1heG1lbW9yeS1zZXR0aW5ncy1idXR0b25cIiB0aXRsZT1cIk9wZW4gTWF4TWVtb3J5IHNldHRpbmdzIGFuZCBtZW1vcmllc1wiPlxyXG4gICAgICAgICAgICAgICAgJHtnZXRTZXR0aW5nc1NWRygnIzg4OCcsIDE0KX1cclxuICAgICAgICAgICAgPC9idXR0b24+XHJcbiAgICAgICAgYCxcclxuXHJcbiAgICAgICAgZ2V0U3VibWl0QnV0dG9uOiAoKSA9PiBgXHJcbiAgICAgICAgICAgIDxidXR0b24gY2xhc3M9XCJnZXQtbWVtb3JpZXMtYnV0dG9uXCIgaWQ9XCJnZXQtbWVtb3JpZXMtYnV0dG9uXCIgc3R5bGU9XCJkaXNwbGF5OiBub25lO1wiPlxyXG4gICAgICAgICAgICAgICAgJHtnZXRNZW1vcmllc1NWRygnIzQwNDE0ZicpfTxzcGFuPlN1Ym1pdDwvc3Bhbj5cclxuICAgICAgICAgICAgPC9idXR0b24+XHJcbiAgICAgICAgYCxcclxuXHJcbiAgICAgICAgZ2V0TWVtb3J5TGltaXRXYXJuaW5nOiAobGltaXRUeXBlLCBjdXJyZW50LCBsaW1pdCkgPT4ge1xyXG4gICAgICAgICAgICBjb25zdCB3YXJuaW5nVGV4dCA9IGxpbWl0VHlwZSA9PT0gJ2d1ZXN0J1xyXG4gICAgICAgICAgICAgICAgPyBgR3Vlc3QgbGltaXQgcmVhY2hlZCAoJHtjdXJyZW50fS8ke2xpbWl0fSkuIFNpZ24gSW4gd2l0aCBHb29nbGUgZm9yIDEwMCBmcmVlIG1lbW9yaWVzLmBcclxuICAgICAgICAgICAgICAgIDogYEZyZWUgbGltaXQgcmVhY2hlZCAoJHtjdXJyZW50fS8ke2xpbWl0fSkuIFVwZ3JhZGUgdG8gUHJvIGZvciB1bmxpbWl0ZWQgbWVtb3JpZXMuYDtcclxuICAgICAgICAgICAgY29uc3QgYWN0aW9uTGFiZWwgPSBsaW1pdFR5cGUgPT09ICdndWVzdCcgPyAnU2lnbiBpbicgOiAnVXBncmFkZSc7XHJcbiAgICAgICAgICAgIGNvbnN0IHdhcm5pbmdDbGFzcyA9IGxpbWl0VHlwZSA9PT0gJ2d1ZXN0J1xyXG4gICAgICAgICAgICAgICAgPyAnbWVtb3J5LWxpbWl0LXdhcm5pbmcgbWVtb3J5LWxpbWl0LXdhcm5pbmctLWd1ZXN0J1xyXG4gICAgICAgICAgICAgICAgOiAnbWVtb3J5LWxpbWl0LXdhcm5pbmcgbWVtb3J5LWxpbWl0LXdhcm5pbmctLWxvZ2dlZC1pbic7XHJcbiAgICAgICAgICAgIGNvbnN0IGJ1dHRvbkNsYXNzID0gbGltaXRUeXBlID09PSAnZ3Vlc3QnXHJcbiAgICAgICAgICAgICAgICA/ICdtZW1vcnktd2FybmluZy1idXR0b24gbWVtb3J5LXdhcm5pbmctYnV0dG9uLS1ndWVzdCdcclxuICAgICAgICAgICAgICAgIDogJ21lbW9yeS13YXJuaW5nLWJ1dHRvbiBtZW1vcnktd2FybmluZy1idXR0b24tLWxvZ2dlZC1pbic7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICByZXR1cm4gYFxyXG4gICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIiR7d2FybmluZ0NsYXNzfVwiPlxyXG4gICAgICAgICAgICAgICAgICAgIDxkaXYgY2xhc3M9XCJtZW1vcnktd2FybmluZy1pY29uXCI+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICR7Z2V0V2FybmluZ0ljb25TVkcoJyM5MjQwMGUnLCAxNCl9XHJcbiAgICAgICAgICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgPGRpdiBjbGFzcz1cIm1lbW9yeS13YXJuaW5nLXRleHRcIj4ke3dhcm5pbmdUZXh0fTwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgIDxidXR0b24gY2xhc3M9XCIke2J1dHRvbkNsYXNzfVwiPiR7YWN0aW9uTGFiZWx9PC9idXR0b24+XHJcbiAgICAgICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgICAgYDtcclxuICAgICAgICB9LFxyXG5cclxuICAgICAgICBnZXRXYXJuaW5nQnV0dG9uOiAoKSA9PiBgXHJcbiAgICAgICAgICAgIDxidXR0b24gY2xhc3M9XCJtZW1vcnktd2FybmluZy1idXR0b25cIj5TaWduIGluPC9idXR0b24+XHJcbiAgICAgICAgYFxyXG4gICAgfTtcclxuXHJcbiAgICAvLyBCYWNrZ3JvdW5kIHNjcmlwdCBjb21tdW5pY2F0aW9uIGZ1bmN0aW9uc1xyXG4gICAgY29uc3QgYmFja2dyb3VuZEFQSSA9IHtcclxuICAgICAgICBhc3luYyBzZWFyY2hNZW1vcmllcyhxdWVyeSkge1xyXG4gICAgICAgICAgICByZXR1cm4gYXdhaXQgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogJ1NFQVJDSF9NRU1PUklFUycsXHJcbiAgICAgICAgICAgICAgICBxdWVyeTogcXVlcnlcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfSxcclxuXHJcbiAgICAgICAgdHJhY2tFcnJvcihlcnJvckRhdGEpIHtcclxuICAgICAgICAgICAgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogJ1RSQUNLX0VSUk9SJyxcclxuICAgICAgICAgICAgICAgIGVycm9yRGF0YTogZXJyb3JEYXRhXHJcbiAgICAgICAgICAgIH0pLmNhdGNoKCgpID0+IHt9KTtcclxuICAgICAgICB9LFxyXG5cclxuICAgICAgICB0cmFja1BvcHVwT3BlbmVkKHNvdXJjZSA9ICdjb250ZW50X3NjcmlwdCcpIHtcclxuICAgICAgICAgICAgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogJ1RSQUNLX1BPUFVQX09QRU5FRCcsXHJcbiAgICAgICAgICAgICAgICBzb3VyY2U6IHNvdXJjZVxyXG4gICAgICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICAgICAgfSxcclxuXHJcbiAgICAgICAgb3BlblBvcHVwSW5UYWIoKSB7XHJcbiAgICAgICAgICAgIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHtcclxuICAgICAgICAgICAgICAgIHR5cGU6ICdPUEVOX1BPUFVQX0lOX1RBQidcclxuICAgICAgICAgICAgfSkuY2F0Y2goKCkgPT4ge30pO1xyXG4gICAgICAgIH0sXHJcblxyXG4gICAgICAgIHRyYWNrTWF4TWVtb3J5VG9nZ2xlZChlbmFibGVkKSB7XHJcbiAgICAgICAgICAgIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHtcclxuICAgICAgICAgICAgICAgIHR5cGU6ICdUUkFDS19NQVhfTUVNT1JZX1RPR0dMRUQnLFxyXG4gICAgICAgICAgICAgICAgZW5hYmxlZDogZW5hYmxlZFxyXG4gICAgICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICAgICAgfSxcclxuXHJcbiAgICAgICAgb3BlblBvcHVwKCkge1xyXG4gICAgICAgICAgICBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7XHJcbiAgICAgICAgICAgICAgICB0eXBlOiAnT1BFTl9QT1BVUCdcclxuICAgICAgICAgICAgfSkuY2F0Y2goKCkgPT4ge30pO1xyXG4gICAgICAgIH0sXHJcblxyXG4gICAgICAgIGFzeW5jIGdldE1lbW9yeUxpbWl0SW5mbygpIHtcclxuICAgICAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgICAgIHJldHVybiBhd2FpdCBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7XHJcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogJ0dFVF9NRU1PUllfTElNSVRfSU5GTydcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgZ2V0dGluZyBtZW1vcnkgbGltaXQgaW5mbzonLCBlcnJvcik7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4geyBzdGF0dXM6ICdlcnJvcicgfTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0sXHJcblxyXG4gICAgICAgIHRyYWNrTWVtb3J5U3VnZ2VzdGlvbkRpc2NhcmRlZChkaXNjYXJkZWRDb3VudCwgY29udGVudCwgbW9kZSA9IG51bGwpIHtcclxuICAgICAgICAgICAgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogJ1RSQUNLX01FTU9SWV9TVUdHRVNUSU9OX0RJU0NBUkRFRCcsXHJcbiAgICAgICAgICAgICAgICBkaXNjYXJkZWRDb3VudDogZGlzY2FyZGVkQ291bnQsXHJcbiAgICAgICAgICAgICAgICBjb250ZW50OiBjb250ZW50LFxyXG4gICAgICAgICAgICAgICAgbW9kZTogbW9kZVxyXG4gICAgICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICAgICAgfSxcclxuXHJcbiAgICAgICAgYXN5bmMgZGVsZXRlTWVtb3J5KGlkLCB0ZXh0ID0gJycpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGF3YWl0IGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHtcclxuICAgICAgICAgICAgICAgIHR5cGU6ICdERUxFVEVfTUVNT1JZJyxcclxuICAgICAgICAgICAgICAgIGlkOiBpZCxcclxuICAgICAgICAgICAgICAgIHRleHQ6IHRleHRcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuICAgIFxyXG5cclxuXHJcbiAgICAvLyBTdHlsZXMgYXJlIG5vdyBsb2FkZWQgZnJvbSBjc3MvY29udGVudFNjcmlwdC5jc3MgdmlhIG1hbmlmZXN0Lmpzb25cclxuXHJcbiAgICBjb25zdCBjcmVhdGVNZW1vcmllc0ljb24gPSAoKSA9PiB7XHJcbiAgICAgICAgY29uc3QgcGFyc2VyID0gbmV3IERPTVBhcnNlcigpO1xyXG4gICAgICAgIHJldHVybiBwYXJzZXIucGFyc2VGcm9tU3RyaW5nKGdldE1lbW9yaWVzU1ZHKCcjZDFkNWRiJyksICdpbWFnZS9zdmcreG1sJykuZG9jdW1lbnRFbGVtZW50O1xyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCBmb3JtYXREYXRlID0gKHRpbWVzdGFtcCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGRhdGUgPSBuZXcgRGF0ZSh0aW1lc3RhbXApO1xyXG4gICAgICAgIGNvbnN0IHllYXIgPSBkYXRlLmdldEZ1bGxZZWFyKCk7XHJcbiAgICAgICAgY29uc3QgbW9udGggPSAoYDAke2RhdGUuZ2V0TW9udGgoKSArIDF9YCkuc2xpY2UoLTIpO1xyXG4gICAgICAgIGNvbnN0IGRheSA9IChgMCR7ZGF0ZS5nZXREYXRlKCl9YCkuc2xpY2UoLTIpO1xyXG4gICAgICAgIHJldHVybiBgJHt5ZWFyfS0ke21vbnRofS0ke2RheX1gO1xyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCBNRU1PUllfTUFSS0VSUyA9IHtcclxuICAgICAgICBzdGFydDogJ1tSRUxFVkFOVF9QQVNUX01FTU9SSUVTX1NUQVJUXScsXHJcbiAgICAgICAgZW5kOiAnW1JFTEVWQU5UX1BBU1RfTUVNT1JJRVNfRU5EXSdcclxuICAgIH07XHJcblxyXG4gICAgY29uc3QgY29udGFpbnNNZW1vcnlNYXJrZXJzID0gKHZhbHVlKSA9PiB7XHJcbiAgICAgICAgY29uc3QgdGV4dCA9IHR5cGVvZiB2YWx1ZSA9PT0gJ3N0cmluZycgPyB2YWx1ZSA6ICh2YWx1ZT8udGV4dENvbnRlbnQgfHwgJycpO1xyXG4gICAgICAgIHJldHVybiB0ZXh0LmluY2x1ZGVzKE1FTU9SWV9NQVJLRVJTLnN0YXJ0KSAmJiB0ZXh0LmluY2x1ZGVzKE1FTU9SWV9NQVJLRVJTLmVuZCk7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGdldENvbnZlcnNhdGlvbk1lc3NhZ2VDb250YWluZXJzID0gKHJvb3QgPSBkb2N1bWVudCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IHJvbGVCYXNlZE1lc3NhZ2VzID0gQXJyYXkuZnJvbShyb290LnF1ZXJ5U2VsZWN0b3JBbGwoJ1tkYXRhLW1lc3NhZ2UtYXV0aG9yLXJvbGVdJykpO1xyXG4gICAgICAgIGlmIChyb2xlQmFzZWRNZXNzYWdlcy5sZW5ndGgpIHtcclxuICAgICAgICAgICAgcmV0dXJuIHJvbGVCYXNlZE1lc3NhZ2VzO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgcmV0dXJuIEFycmF5LmZyb20ocm9vdC5xdWVyeVNlbGVjdG9yQWxsKCdhcnRpY2xlJykpO1xyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCBnZXRVc2VyTWVzc2FnZUNvbnRhaW5lcnMgPSAocm9vdCA9IGRvY3VtZW50KSA9PiB7XHJcbiAgICAgICAgY29uc3Qgcm9sZUJhc2VkVXNlck1lc3NhZ2VzID0gQXJyYXkuZnJvbShyb290LnF1ZXJ5U2VsZWN0b3JBbGwoJ1tkYXRhLW1lc3NhZ2UtYXV0aG9yLXJvbGU9XCJ1c2VyXCJdJykpO1xyXG4gICAgICAgIGlmIChyb2xlQmFzZWRVc2VyTWVzc2FnZXMubGVuZ3RoKSB7XHJcbiAgICAgICAgICAgIHJldHVybiByb2xlQmFzZWRVc2VyTWVzc2FnZXM7XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICByZXR1cm4gQXJyYXkuZnJvbShyb290LnF1ZXJ5U2VsZWN0b3JBbGwoJ2FydGljbGUnKSk7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGdldENsb3Nlc3RVc2VyTWVzc2FnZUNvbnRhaW5lciA9IChub2RlKSA9PiB7XHJcbiAgICAgICAgaWYgKCFub2RlKSByZXR1cm4gbnVsbDtcclxuXHJcbiAgICAgICAgY29uc3QgZWxlbWVudCA9IG5vZGUubm9kZVR5cGUgPT09IE5vZGUuRUxFTUVOVF9OT0RFID8gbm9kZSA6IG5vZGUucGFyZW50RWxlbWVudDtcclxuICAgICAgICByZXR1cm4gZWxlbWVudD8uY2xvc2VzdD8uKCdbZGF0YS1tZXNzYWdlLWF1dGhvci1yb2xlPVwidXNlclwiXSwgYXJ0aWNsZScpIHx8IG51bGw7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGdldE1lc3NhZ2VDb250ZW50RWxlbWVudCA9IChtZXNzYWdlQ29udGFpbmVyKSA9PiB7XHJcbiAgICAgICAgaWYgKCFtZXNzYWdlQ29udGFpbmVyKSByZXR1cm4gbnVsbDtcclxuXHJcbiAgICAgICAgY29uc3QgcHJlZmVycmVkU2VsZWN0b3JzID0gJy53aGl0ZXNwYWNlLXByZS13cmFwLCBbZGF0YS10ZXN0aWQ9XCJ1c2VyLW1lc3NhZ2VcIl0sIFtkaXI9XCJhdXRvXCJdJztcclxuICAgICAgICBjb25zdCBwcmVmZXJyZWRDYW5kaWRhdGVzID0gW107XHJcblxyXG4gICAgICAgIGlmIChtZXNzYWdlQ29udGFpbmVyLm1hdGNoZXM/LihwcmVmZXJyZWRTZWxlY3RvcnMpKSB7XHJcbiAgICAgICAgICAgIHByZWZlcnJlZENhbmRpZGF0ZXMucHVzaChtZXNzYWdlQ29udGFpbmVyKTtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIHByZWZlcnJlZENhbmRpZGF0ZXMucHVzaCguLi5tZXNzYWdlQ29udGFpbmVyLnF1ZXJ5U2VsZWN0b3JBbGwocHJlZmVycmVkU2VsZWN0b3JzKSk7XHJcblxyXG4gICAgICAgIGxldCBkZWVwZXN0TWF0Y2hpbmdQcmVmZXJyZWQgPSBudWxsO1xyXG4gICAgICAgIGZvciAoY29uc3QgY2FuZGlkYXRlIG9mIHByZWZlcnJlZENhbmRpZGF0ZXMpIHtcclxuICAgICAgICAgICAgaWYgKGNhbmRpZGF0ZSBpbnN0YW5jZW9mIEhUTUxFbGVtZW50ICYmIGNvbnRhaW5zTWVtb3J5TWFya2VycyhjYW5kaWRhdGUpKSB7XHJcbiAgICAgICAgICAgICAgICBkZWVwZXN0TWF0Y2hpbmdQcmVmZXJyZWQgPSBjYW5kaWRhdGU7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGlmIChkZWVwZXN0TWF0Y2hpbmdQcmVmZXJyZWQpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGRlZXBlc3RNYXRjaGluZ1ByZWZlcnJlZDtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGxldCBkZWVwZXN0TWF0Y2hpbmdFbGVtZW50ID0gbnVsbDtcclxuICAgICAgICBjb25zdCB3YWxrZXIgPSBkb2N1bWVudC5jcmVhdGVUcmVlV2Fsa2VyKG1lc3NhZ2VDb250YWluZXIsIE5vZGVGaWx0ZXIuU0hPV19FTEVNRU5UKTtcclxuICAgICAgICB3aGlsZSAod2Fsa2VyLm5leHROb2RlKCkpIHtcclxuICAgICAgICAgICAgY29uc3QgY2FuZGlkYXRlID0gd2Fsa2VyLmN1cnJlbnROb2RlO1xyXG4gICAgICAgICAgICBpZiAoIShjYW5kaWRhdGUgaW5zdGFuY2VvZiBIVE1MRWxlbWVudCkpIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICBpZiAoY2FuZGlkYXRlLmNsb3Nlc3QoJy5tZW1vcnktc2VjdGlvbicpKSBjb250aW51ZTtcclxuXHJcbiAgICAgICAgICAgIGlmIChjb250YWluc01lbW9yeU1hcmtlcnMoY2FuZGlkYXRlKSkge1xyXG4gICAgICAgICAgICAgICAgZGVlcGVzdE1hdGNoaW5nRWxlbWVudCA9IGNhbmRpZGF0ZTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgcmV0dXJuIGRlZXBlc3RNYXRjaGluZ0VsZW1lbnQ7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGdldElucHV0Qm94ID0gKCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGlucHV0Qm94ID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihDSEFUR1BUX1NFTEVDVE9SUy5JTlBVVF9CT1gpO1xyXG4gICAgICAgLy8gY29uc29sZS5sb2coJ0xvb2tpbmcgZm9yIGlucHV0IGJveDonLCBpbnB1dEJveCk7XHJcbiAgICAgICAgcmV0dXJuIGlucHV0Qm94O1xyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCBzdHlsZU1lbW9yaWVzSW5DaGF0ID0gKCkgPT4ge1xyXG4gICAgICAgIC8vIEhhbmRsZSBDaGF0R1BUIG1lc3NhZ2VzXHJcbiAgICAgICAgZ2V0VXNlck1lc3NhZ2VDb250YWluZXJzKCkuZm9yRWFjaChoYW5kbGVNZXNzYWdlU3R5bGluZyk7XHJcbiAgICAgICAgXHJcblxyXG4gICAgfTtcclxuXHJcbiAgICBsZXQgc3R5bGVNZW1vcmllc0luQ2hhdFNjaGVkdWxlZCA9IGZhbHNlO1xyXG4gICAgY29uc3Qgc2NoZWR1bGVTdHlsZU1lbW9yaWVzSW5DaGF0ID0gKCkgPT4ge1xyXG4gICAgICAgIGlmIChzdHlsZU1lbW9yaWVzSW5DaGF0U2NoZWR1bGVkKSB7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIHN0eWxlTWVtb3JpZXNJbkNoYXRTY2hlZHVsZWQgPSB0cnVlO1xyXG4gICAgICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZSgoKSA9PiB7XHJcbiAgICAgICAgICAgIHN0eWxlTWVtb3JpZXNJbkNoYXRTY2hlZHVsZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgc3R5bGVNZW1vcmllc0luQ2hhdCgpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuXHJcbiAgICBsZXQgcGVuZGluZ01lbW9yeVN0eWxlUmV0cnlUaW1lciA9IG51bGw7XHJcbiAgICBsZXQgcGVuZGluZ01lbW9yeVN0eWxlUmV0cnlBdHRlbXB0cyA9IDA7XHJcbiAgICBsZXQgcGVuZGluZ01lbW9yeVN0eWxlVGFyZ2V0ID0gbnVsbDtcclxuXHJcbiAgICBjb25zdCBjbGVhclBlbmRpbmdNZW1vcnlTdHlsaW5nV2F0Y2ggPSAoKSA9PiB7XHJcbiAgICAgICAgaWYgKHBlbmRpbmdNZW1vcnlTdHlsZVJldHJ5VGltZXIpIHtcclxuICAgICAgICAgICAgY2xlYXJUaW1lb3V0KHBlbmRpbmdNZW1vcnlTdHlsZVJldHJ5VGltZXIpO1xyXG4gICAgICAgICAgICBwZW5kaW5nTWVtb3J5U3R5bGVSZXRyeVRpbWVyID0gbnVsbDtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIHBlbmRpbmdNZW1vcnlTdHlsZVJldHJ5QXR0ZW1wdHMgPSAwO1xyXG4gICAgICAgIHBlbmRpbmdNZW1vcnlTdHlsZVRhcmdldCA9IG51bGw7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IHdhdGNoRm9yUGVuZGluZ01lbW9yeVN0eWxlZE1lc3NhZ2UgPSAoKSA9PiB7XHJcbiAgICAgICAgaWYgKCFwZW5kaW5nTWVtb3J5U3R5bGVUYXJnZXQpIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgc2NoZWR1bGVTdHlsZU1lbW9yaWVzSW5DaGF0KCk7XHJcblxyXG4gICAgICAgIGNvbnN0IHVzZXJNZXNzYWdlcyA9IGdldFVzZXJNZXNzYWdlQ29udGFpbmVycygpO1xyXG4gICAgICAgIGNvbnN0IG1hdGNoaW5nTWVzc2FnZSA9IEFycmF5LmZyb20odXNlck1lc3NhZ2VzKS5yZXZlcnNlKCkuZmluZCgobWVzc2FnZUNvbnRhaW5lcikgPT4ge1xyXG4gICAgICAgICAgICBjb25zdCBtZXNzYWdlRGl2ID0gZ2V0TWVzc2FnZUNvbnRlbnRFbGVtZW50KG1lc3NhZ2VDb250YWluZXIpO1xyXG4gICAgICAgICAgICBpZiAoIW1lc3NhZ2VEaXYpIHJldHVybiBmYWxzZTtcclxuXHJcbiAgICAgICAgICAgIGNvbnN0IG1lc3NhZ2VUZXh0ID0gbWVzc2FnZURpdi50ZXh0Q29udGVudCB8fCAnJztcclxuICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgIG1lc3NhZ2VUZXh0LmluY2x1ZGVzKE1FTU9SWV9NQVJLRVJTLnN0YXJ0KSAmJlxyXG4gICAgICAgICAgICAgICAgbWVzc2FnZVRleHQuaW5jbHVkZXMoTUVNT1JZX01BUktFUlMuZW5kKSAmJlxyXG4gICAgICAgICAgICAgICAgbWVzc2FnZVRleHQuaW5jbHVkZXMocGVuZGluZ01lbW9yeVN0eWxlVGFyZ2V0Lm1lbW9yaWVzU25pcHBldClcclxuICAgICAgICAgICAgKTtcclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgaWYgKG1hdGNoaW5nTWVzc2FnZSkge1xyXG4gICAgICAgICAgICBoYW5kbGVNZXNzYWdlU3R5bGluZyhtYXRjaGluZ01lc3NhZ2UpO1xyXG5cclxuICAgICAgICAgICAgY29uc3Qgc3R5bGVkTWVzc2FnZURpdiA9IGdldE1lc3NhZ2VDb250ZW50RWxlbWVudChtYXRjaGluZ01lc3NhZ2UpO1xyXG4gICAgICAgICAgICBpZiAoc3R5bGVkTWVzc2FnZURpdj8ucXVlcnlTZWxlY3RvcignLm1lbW9yeS1zZWN0aW9uJykpIHtcclxuICAgICAgICAgICAgICAgIGNsZWFyUGVuZGluZ01lbW9yeVN0eWxpbmdXYXRjaCgpO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBwZW5kaW5nTWVtb3J5U3R5bGVSZXRyeUF0dGVtcHRzICs9IDE7XHJcbiAgICAgICAgaWYgKHBlbmRpbmdNZW1vcnlTdHlsZVJldHJ5QXR0ZW1wdHMgPj0gMzApIHtcclxuICAgICAgICAgICAgY2xlYXJQZW5kaW5nTWVtb3J5U3R5bGluZ1dhdGNoKCk7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIHBlbmRpbmdNZW1vcnlTdHlsZVJldHJ5VGltZXIgPSBzZXRUaW1lb3V0KHdhdGNoRm9yUGVuZGluZ01lbW9yeVN0eWxlZE1lc3NhZ2UsIDI1MCk7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGJlZ2luUGVuZGluZ01lbW9yeVN0eWxpbmdXYXRjaCA9IChtZW1vcmllc1RleHQpID0+IHtcclxuICAgICAgICBpZiAoIW1lbW9yaWVzVGV4dCkge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBjbGVhclBlbmRpbmdNZW1vcnlTdHlsaW5nV2F0Y2goKTtcclxuICAgICAgICBwZW5kaW5nTWVtb3J5U3R5bGVUYXJnZXQgPSB7XHJcbiAgICAgICAgICAgIG1lbW9yaWVzU25pcHBldDogbWVtb3JpZXNUZXh0LnNsaWNlKDAsIDE2MClcclxuICAgICAgICB9O1xyXG4gICAgICAgIHdhdGNoRm9yUGVuZGluZ01lbW9yeVN0eWxlZE1lc3NhZ2UoKTtcclxuICAgIH07XHJcblxyXG5jb25zdCBoYW5kbGVNZXNzYWdlU3R5bGluZyA9IChtZXNzYWdlQ29udGFpbmVyKSA9PiB7XHJcbiAgICAgICAgY29uc3QgbWVzc2FnZURpdiA9IGdldE1lc3NhZ2VDb250ZW50RWxlbWVudChtZXNzYWdlQ29udGFpbmVyKTtcclxuICAgICAgICBpZiAoIW1lc3NhZ2VEaXYpIHJldHVybjtcclxuXHJcbiAgICAgICAgY29uc3QgbWF0Y2ggPSBtZXNzYWdlRGl2LnRleHRDb250ZW50Lm1hdGNoKC9cXFtSRUxFVkFOVF9QQVNUX01FTU9SSUVTX1NUQVJUXFxdKFtcXHNcXFNdKj8pXFxbUkVMRVZBTlRfUEFTVF9NRU1PUklFU19FTkRcXF0vKTtcclxuICAgICAgICBpZiAoIW1hdGNoKSByZXR1cm47XHJcblxyXG4gICAgICAgIGNvbnN0IFtmdWxsTWF0Y2gsIG1lbW9yaWVzQ29udGVudF0gPSBtYXRjaDtcclxuICAgICAgICBjb25zdCBbYmVmb3JlLCBhZnRlcl0gPSBtZXNzYWdlRGl2LnRleHRDb250ZW50LnNwbGl0KGZ1bGxNYXRjaCk7XHJcbiAgICAgICAgY29uc3QgdHJpbW1lZE1lbW9yaWVzQ29udGVudCA9IG1lbW9yaWVzQ29udGVudC50cmltKCk7XHJcbiAgICAgICAgY29uc3QgY29udGVudFNpZ25hdHVyZSA9IGAke2JlZm9yZS50cmltKCl9Ojoke3RyaW1tZWRNZW1vcmllc0NvbnRlbnR9Ojoke2FmdGVyLnRyaW0oKX1gO1xyXG5cclxuICAgICAgICAvLyBDaGF0R1BUIG1heSByZXVzZSB0aGUgc2FtZSBET00gbm9kZSBmb3IgbGF0ZXIgdXNlciBtZXNzYWdlcywgc28gb25seSBza2lwXHJcbiAgICAgICAgLy8gd2hlbiB3ZSd2ZSBhbHJlYWR5IHN0eWxlZCB0aGlzIGV4YWN0IG1lc3NhZ2UgY29udGVudC5cclxuICAgICAgICBpZiAoXHJcbiAgICAgICAgICAgIG1lc3NhZ2VEaXYuZGF0YXNldC5tYXhtZW1vcnlQcm9jZXNzZWQgPT09ICd0cnVlJyAmJlxyXG4gICAgICAgICAgICBtZXNzYWdlRGl2LmRhdGFzZXQubWF4bWVtb3J5UHJvY2Vzc2VkU2lnbmF0dXJlID09PSBjb250ZW50U2lnbmF0dXJlICYmXHJcbiAgICAgICAgICAgIG1lc3NhZ2VEaXYucXVlcnlTZWxlY3RvcignLm1lbW9yeS1zZWN0aW9uJylcclxuICAgICAgICApIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICAvLyAtLS0gVEhFIEZJWCAtLS1cclxuICAgICAgICAvLyBTdG9yZSB0aGUgZnVsbCwgdW50cnVuY2F0ZWQgbWVtb3JpZXMgc3RyaW5nIGluIGEgZGF0YSBhdHRyaWJ1dGVcclxuICAgICAgICAvLyBXZSdsbCBwbGFjZSBpdCBvbiB0aGUgbWVzc2FnZURpdiBpdHNlbGYgZm9yIGVhc3kgc2NyYXBpbmcgbGF0ZXIuXHJcbiAgICAgICAgbWVzc2FnZURpdi5zZXRBdHRyaWJ1dGUoJ2RhdGEtZnVsbC1tZW1vcmllcycsIHRyaW1tZWRNZW1vcmllc0NvbnRlbnQpO1xyXG4gICAgICAgIC8vIC0tLSBFTkQgRklYIC0tLVxyXG5cclxuICAgICAgICAvLyBDcmVhdGUgdGhlIHRydW5jYXRlZCB2ZXJzaW9uIGZvciBkaXNwbGF5IHB1cnBvc2VzIG9ubHlcclxuICAgICAgICBjb25zdCB0cnVuY2F0ZWRDb250ZW50ID0gbWVtb3JpZXNDb250ZW50Lmxlbmd0aCA+IDI4MCA/IFxyXG4gICAgICAgICAgICBgJHttZW1vcmllc0NvbnRlbnQuc2xpY2UoMCwgMjgwKX0uLi4gPHNwYW4gY2xhc3M9XCJzaG93LW1vcmUtbWVtb3JpZXNcIiBzdHlsZT1cImNvbG9yOiAjNjY2OyBmb250LXdlaWdodDogNjAwOyBjdXJzb3I6IHBvaW50ZXI7IHVzZXItc2VsZWN0OiB0ZXh0OyBwb2ludGVyLWV2ZW50czogYXV0bztcIj5TaG93IG1vcmU8L3NwYW4+YCA6IFxyXG4gICAgICAgICAgICBtZW1vcmllc0NvbnRlbnQ7XHJcblxyXG4gICAgICAgIC8vIFJlYnVpbGQgdGhlIGlubmVyIEhUTUwgd2l0aCB0aGUgdHJ1bmNhdGVkIGNvbnRlbnRcclxuICAgICAgICBtZXNzYWdlRGl2LmlubmVySFRNTCA9IGAke2JlZm9yZS50cmltKCl9PGRpdiBjbGFzcz1cIm1lbW9yeS1zZWN0aW9uXCI+JHtjcmVhdGVNZW1vcmllc0ljb24oKS5vdXRlckhUTUx9IDxzcGFuIGNsYXNzPVwibWVtb3JpZXMtY29udGVudFwiPiR7dHJ1bmNhdGVkQ29udGVudH08L3NwYW4+PC9kaXY+JHthZnRlci50cmltKCl9YDtcclxuXHJcbiAgICAgICAgLy8gQWRkIGEgZmxhZyB0byBpbmRpY2F0ZSB0aGF0IHdlJ3ZlIHByb2Nlc3NlZCB0aGlzIGVsZW1lbnRcclxuICAgICAgICBtZXNzYWdlRGl2LmRhdGFzZXQubWF4bWVtb3J5UHJvY2Vzc2VkID0gJ3RydWUnO1xyXG4gICAgICAgIG1lc3NhZ2VEaXYuZGF0YXNldC5tYXhtZW1vcnlQcm9jZXNzZWRTaWduYXR1cmUgPSBjb250ZW50U2lnbmF0dXJlO1xyXG5cclxuICAgICAgICAvLyBSZS1hdHRhY2ggdGhlIFwiU2hvdyBtb3JlXCIgY2xpY2sgbGlzdGVuZXJcclxuICAgICAgICBtZXNzYWdlRGl2LnF1ZXJ5U2VsZWN0b3IoJy5zaG93LW1vcmUtbWVtb3JpZXMnKT8uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBlID0+IHtcclxuICAgICAgICAgICAgZS5zdG9wUHJvcGFnYXRpb24oKTtcclxuICAgICAgICAgICAgLy8gVGhlIG9yaWdpbmFsLCBmdWxsIGNvbnRlbnQgaXMgc3RpbGwgYXZhaWxhYmxlIGluIHRoZSBjbG9zdXJlXHJcbiAgICAgICAgICAgIGUudGFyZ2V0LmNsb3Nlc3QoJy5tZW1vcmllcy1jb250ZW50JykuaW5uZXJIVE1MID0gbWVtb3JpZXNDb250ZW50O1xyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuICAgIC8vIFNpbXBsZSBwb2xsaW5nIGZvciBmb3JtIGVsZW1lbnQgLSBubyBnbG9iYWwgb2JzZXJ2ZXJzXHJcbiAgICBjb25zdCB3YWl0Rm9yRm9ybSA9ICgpID0+IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7XHJcbiAgICAgICAgY29uc3QgY2hlY2tGb3JtID0gKCkgPT4ge1xyXG4gICAgICAgICAgICBjb25zdCBmb3JtID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihDSEFUR1BUX1NFTEVDVE9SUy5GT1JNKTtcclxuICAgICAgICAgICAgaWYgKGZvcm0pIHtcclxuICAgICAgICAgICAgICAgIHJlc29sdmUoZm9ybSk7XHJcbiAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBzZXRUaW1lb3V0KGNoZWNrRm9ybSwgMjUwKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH07XHJcbiAgICAgICAgY2hlY2tGb3JtKCk7XHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBPYnNlcnZlck1hbmFnZXIgLSBFZmZpY2llbnQgb2JzZXJ2ZXIgbWFuYWdlbWVudCBzeXN0ZW1cclxuICAgIC8vIFJFUExBQ0UgeW91ciBleGlzdGluZyBPYnNlcnZlck1hbmFnZXIgb2JqZWN0IHdpdGggdGhpcyBvbmVcclxuXHJcbmNvbnN0IE9ic2VydmVyTWFuYWdlciA9IHtcclxuICAgIF9tYWluT2JzZXJ2ZXI6IG51bGwsXHJcbiAgICBfbWVzc2FnZUxpc3RPYnNlcnZlcjogbnVsbCxcclxuICAgIF9pbnB1dEZvcm1PYnNlcnZlcjogbnVsbCxcclxuICAgIF9zdWJtaXRCdXR0b25PYnNlcnZlcjogbnVsbCxcclxuICAgIFxyXG4gICAgX2NhbGxiYWNrczoge30sXHJcbiAgICBcclxuICAgIC8vIE5FVzogRmxhZ3MgdG8gcHJldmVudCByZS1kZXBsb3lpbmcgb2JzZXJ2ZXJzXHJcbiAgICBfaW5wdXRPYnNlcnZlcnNEZXBsb3llZDogZmFsc2UsXHJcbiAgICBfbWVzc2FnZU9ic2VydmVyRGVwbG95ZWQ6IGZhbHNlLFxyXG5cclxuICAgIGluaXQoY2FsbGJhY2tzKSB7XHJcbiAgICAgICAgdGhpcy5fY2FsbGJhY2tzID0gY2FsbGJhY2tzO1xyXG4gICAgfSxcclxuXHJcbiAgICBzdGFydCgpIHtcclxuICAgICAgICB0aGlzLnN0b3AoKTsgLy8gU3RvcCBhbnkgcHJldmlvdXMgb2JzZXJ2ZXJzIHRvIHByZXZlbnQgZHVwbGljYXRlc1xyXG4gICAgICAgIGNvbnNvbGUubG9nKCdbT2JzZXJ2ZXJNYW5hZ2VyXSBTdGFydGluZyBvYnNlcnZlcnMuJyk7XHJcblxyXG4gICAgICAgIHRoaXMuX2lucHV0T2JzZXJ2ZXJzRGVwbG95ZWQgPSBmYWxzZTtcclxuICAgICAgICB0aGlzLl9tZXNzYWdlT2JzZXJ2ZXJEZXBsb3llZCA9IGZhbHNlO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGNvbnN0IG1haW5Db250YWluZXIgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKCdtYWluJyk7XHJcbiAgICAgICAgaWYgKCFtYWluQ29udGFpbmVyKSB7XHJcbiAgICAgICAgICAgIHNldFRpbWVvdXQoKCkgPT4gdGhpcy5zdGFydCgpLCAyNTApO1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIHRoaXMuX21haW5PYnNlcnZlciA9IG5ldyBNdXRhdGlvbk9ic2VydmVyKCgpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgZm9ybSA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoQ0hBVEdQVF9TRUxFQ1RPUlMuRk9STSk7XHJcbiAgICAgICAgICAgIGNvbnN0IG1lc3NhZ2VDb250YWluZXIgPSB0aGlzLl9nZXRNZXNzYWdlTGlzdENvbnRhaW5lcigpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gKipDSEFOR0VEIExPR0lDKio6IERlcGxveSBpbnB1dCBvYnNlcnZlcnMgYXMgc29vbiBhcyB0aGUgZm9ybSBpcyBmb3VuZFxyXG4gICAgICAgICAgICBpZiAoZm9ybSAmJiAhdGhpcy5faW5wdXRPYnNlcnZlcnNEZXBsb3llZCkge1xyXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ1tPYnNlcnZlck1hbmFnZXJdIElucHV0IGZvcm0gZm91bmQuIERlcGxveWluZyBpbnB1dC1yZWxhdGVkIG9ic2VydmVycy4nKTtcclxuICAgICAgICAgICAgICAgIHRoaXMuX2RlcGxveUlucHV0T2JzZXJ2ZXJzKGZvcm0pO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5faW5wdXRPYnNlcnZlcnNEZXBsb3llZCA9IHRydWU7XHJcbiAgICAgICAgICAgIH1cclxuXHJcbiAgICAgICAgICAgIC8vICoqQ0hBTkdFRCBMT0dJQyoqOiBEZXBsb3kgbWVzc2FnZSBvYnNlcnZlciB3aGVuIHRoZSBtZXNzYWdlIGNvbnRhaW5lciBpcyBmb3VuZFxyXG4gICAgICAgICAgICBpZiAobWVzc2FnZUNvbnRhaW5lciAmJiAhdGhpcy5fbWVzc2FnZU9ic2VydmVyRGVwbG95ZWQpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbT2JzZXJ2ZXJNYW5hZ2VyXSBNZXNzYWdlIGNvbnRhaW5lciBmb3VuZC4gRGVwbG95aW5nIG1lc3NhZ2Ugb2JzZXJ2ZXIuJyk7XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9kZXBsb3lNZXNzYWdlT2JzZXJ2ZXIobWVzc2FnZUNvbnRhaW5lcik7XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9tZXNzYWdlT2JzZXJ2ZXJEZXBsb3llZCA9IHRydWU7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIElmIGJvdGggYXJlIGRlcGxveWVkLCB3ZSBjYW4gc3RvcCB0aGUgbWFpbiBcInNjb3V0XCIgb2JzZXJ2ZXJcclxuICAgICAgICAgICAgaWYgKHRoaXMuX2lucHV0T2JzZXJ2ZXJzRGVwbG95ZWQgJiYgdGhpcy5fbWVzc2FnZU9ic2VydmVyRGVwbG95ZWQpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbT2JzZXJ2ZXJNYW5hZ2VyXSBBbGwgdGFyZ2V0ZWQgb2JzZXJ2ZXJzIGRlcGxveWVkLiBEaXNjb25uZWN0aW5nIG1haW4gb2JzZXJ2ZXIuJyk7XHJcbiAgICAgICAgICAgICAgICB0aGlzLl9tYWluT2JzZXJ2ZXIuZGlzY29ubmVjdCgpO1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fbWFpbk9ic2VydmVyID0gbnVsbDtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIFxyXG4gICAgICAgIHRoaXMuX21haW5PYnNlcnZlci5vYnNlcnZlKG1haW5Db250YWluZXIsIHtcclxuICAgICAgICAgICAgY2hpbGRMaXN0OiB0cnVlLFxyXG4gICAgICAgICAgICBzdWJ0cmVlOiB0cnVlXHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gQWxzbyBjaGVjayBpZiBlbGVtZW50cyBhcmUgYWxyZWFkeSB0aGVyZSBvbiBzdGFydHVwXHJcbiAgICAgICAgY29uc3QgaW5pdGlhbEZvcm0gPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKENIQVRHUFRfU0VMRUNUT1JTLkZPUk0pO1xyXG4gICAgICAgIGNvbnN0IGluaXRpYWxNZXNzYWdlQ29udGFpbmVyID0gdGhpcy5fZ2V0TWVzc2FnZUxpc3RDb250YWluZXIoKTtcclxuICAgICAgICBpZiAoaW5pdGlhbEZvcm0pIHtcclxuICAgICAgICAgICAgdGhpcy5fZGVwbG95SW5wdXRPYnNlcnZlcnMoaW5pdGlhbEZvcm0pO1xyXG4gICAgICAgICAgICB0aGlzLl9pbnB1dE9ic2VydmVyc0RlcGxveWVkID0gdHJ1ZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGluaXRpYWxNZXNzYWdlQ29udGFpbmVyKSB7XHJcbiAgICAgICAgICAgIHRoaXMuX2RlcGxveU1lc3NhZ2VPYnNlcnZlcihpbml0aWFsTWVzc2FnZUNvbnRhaW5lcik7XHJcbiAgICAgICAgICAgIHRoaXMuX21lc3NhZ2VPYnNlcnZlckRlcGxveWVkID0gdHJ1ZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKHRoaXMuX2lucHV0T2JzZXJ2ZXJzRGVwbG95ZWQgJiYgdGhpcy5fbWVzc2FnZU9ic2VydmVyRGVwbG95ZWQpIHtcclxuICAgICAgICAgICAgIHRoaXMuX21haW5PYnNlcnZlci5kaXNjb25uZWN0KCk7XHJcbiAgICAgICAgICAgICB0aGlzLl9tYWluT2JzZXJ2ZXIgPSBudWxsO1xyXG4gICAgICAgIH1cclxuICAgIH0sXHJcbiAgICBcclxuICAgIC8vIE5FVzogRnVuY3Rpb24gdG8gZGVwbG95IG9ubHkgaW5wdXQtcmVsYXRlZCBvYnNlcnZlcnNcclxuICAgIF9kZXBsb3lJbnB1dE9ic2VydmVycyhmb3JtRWwpIHtcclxuICAgICAgICAvLyBPYnNlcnZlciBmb3IgY2hhbmdlcyBpbiB0aGUgaW5wdXQgYXJlYVxyXG4gICAgICAgIHRoaXMuX2lucHV0Rm9ybU9ic2VydmVyID0gbmV3IE11dGF0aW9uT2JzZXJ2ZXIoKG11dGF0aW9ucykgPT4ge1xyXG4gICAgICAgICAgICBpZiAodGhpcy5fY2FsbGJhY2tzLm9uSW5wdXRBcmVhQ2hhbmdlZCkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fY2FsbGJhY2tzLm9uSW5wdXRBcmVhQ2hhbmdlZChtdXRhdGlvbnMpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdGhpcy5faW5wdXRGb3JtT2JzZXJ2ZXIub2JzZXJ2ZShmb3JtRWwucGFyZW50Tm9kZSwgeyBjaGlsZExpc3Q6IHRydWUgfSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gT2JzZXJ2ZXIgZm9yIHN1Ym1pdCBidXR0b24gY2hhbmdlc1xyXG4gICAgICAgIHRoaXMuX3N1Ym1pdEJ1dHRvbk9ic2VydmVyID0gbmV3IE11dGF0aW9uT2JzZXJ2ZXIoKG11dGF0aW9ucykgPT4ge1xyXG4gICAgICAgICAgICBpZiAodGhpcy5fY2FsbGJhY2tzLm9uU3VibWl0QnV0dG9uQ2hhbmdlZCkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fY2FsbGJhY2tzLm9uU3VibWl0QnV0dG9uQ2hhbmdlZChtdXRhdGlvbnMpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdGhpcy5fc3VibWl0QnV0dG9uT2JzZXJ2ZXIub2JzZXJ2ZShmb3JtRWwsIHsgY2hpbGRMaXN0OiB0cnVlLCBzdWJ0cmVlOiB0cnVlIH0pO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFRyaWdnZXIgdGhlIFwiVUkgaXMgcmVhZHlcIiBjYWxsYmFjayB0byBpbmplY3QgYnV0dG9ucywgZXRjLlxyXG4gICAgICAgIGlmICh0aGlzLl9jYWxsYmFja3Mub25VSVJlYWR5KSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbT2JzZXJ2ZXJNYW5hZ2VyXSBJbnB1dCBVSSByZWFkeSwgdHJpZ2dlcmluZyBvblVJUmVhZHkgY2FsbGJhY2snKTtcclxuICAgICAgICAgICAgdGhpcy5fY2FsbGJhY2tzLm9uVUlSZWFkeSgpO1xyXG4gICAgICAgIH1cclxuICAgIH0sXHJcblxyXG4gICAgLy8gTkVXOiBGdW5jdGlvbiB0byBkZXBsb3kgb25seSB0aGUgbWVzc2FnZSBsaXN0IG9ic2VydmVyXHJcbiAgICBfZGVwbG95TWVzc2FnZU9ic2VydmVyKG1lc3NhZ2VDb250YWluZXJFbCkge1xyXG4gICAgICAgIC8vIC0tLSBTVEFSVCBPRiBGSVggLS0tXHJcbiAgICAgICAgLy8gMS4gU2NhbiBmb3IgYW55IG1lc3NhZ2VzIHRoYXQgYWxyZWFkeSBleGlzdCBSSUdIVCBOT1cuXHJcbiAgICAgICAgLy8gVGhpcyBjbG9zZXMgdGhlIHJhY2UgY29uZGl0aW9uIG9uIGluaXRpYWwgcGFnZSBsb2FkLlxyXG4gICAgICAgIGNvbnNvbGUubG9nKCdbT2JzZXJ2ZXJNYW5hZ2VyXSBTY2FubmluZyBmb3IgcHJlLWV4aXN0aW5nIG1lc3NhZ2VzIHRvIHN0eWxlLi4uJyk7XHJcbiAgICAgICAgc2NoZWR1bGVTdHlsZU1lbW9yaWVzSW5DaGF0KCk7XHJcbiAgICAgICAgLy8gLS0tIEVORCBPRiBGSVggLS0tXHJcblxyXG4gICAgICAgIC8vIDIuIE5vdywgc2V0IHVwIHRoZSBvYnNlcnZlciBmb3IgYW55IE5FVyBtZXNzYWdlcyB0aGF0IGFwcGVhciBsYXRlci5cclxuICAgICAgICB0aGlzLl9tZXNzYWdlTGlzdE9ic2VydmVyID0gbmV3IE11dGF0aW9uT2JzZXJ2ZXIoKG11dGF0aW9ucykgPT4ge1xyXG4gICAgICAgICAgICBpZiAodGhpcy5fY2FsbGJhY2tzLm9uTWVzc2FnZXNBZGRlZCkge1xyXG4gICAgICAgICAgICAgICAgdGhpcy5fY2FsbGJhY2tzLm9uTWVzc2FnZXNBZGRlZChtdXRhdGlvbnMpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgdGhpcy5fbWVzc2FnZUxpc3RPYnNlcnZlci5vYnNlcnZlKG1lc3NhZ2VDb250YWluZXJFbCwge1xyXG4gICAgICAgICAgICBjaGlsZExpc3Q6IHRydWUsXHJcbiAgICAgICAgICAgIHN1YnRyZWU6IHRydWUsXHJcbiAgICAgICAgICAgIGNoYXJhY3RlckRhdGE6IHRydWVcclxuICAgICAgICB9KTtcclxuICAgIH0sXHJcblxyXG4gICAgc3RvcCgpIHtcclxuICAgICAgICBpZiAodGhpcy5fbWFpbk9ic2VydmVyKSB0aGlzLl9tYWluT2JzZXJ2ZXIuZGlzY29ubmVjdCgpO1xyXG4gICAgICAgIGlmICh0aGlzLl9tZXNzYWdlTGlzdE9ic2VydmVyKSB0aGlzLl9tZXNzYWdlTGlzdE9ic2VydmVyLmRpc2Nvbm5lY3QoKTtcclxuICAgICAgICBpZiAodGhpcy5faW5wdXRGb3JtT2JzZXJ2ZXIpIHRoaXMuX2lucHV0Rm9ybU9ic2VydmVyLmRpc2Nvbm5lY3QoKTtcclxuICAgICAgICBpZiAodGhpcy5fc3VibWl0QnV0dG9uT2JzZXJ2ZXIpIHRoaXMuX3N1Ym1pdEJ1dHRvbk9ic2VydmVyLmRpc2Nvbm5lY3QoKTtcclxuICAgICAgICB0aGlzLl9tYWluT2JzZXJ2ZXIgPSBudWxsO1xyXG4gICAgICAgIHRoaXMuX21lc3NhZ2VMaXN0T2JzZXJ2ZXIgPSBudWxsO1xyXG4gICAgICAgIHRoaXMuX2lucHV0Rm9ybU9ic2VydmVyID0gbnVsbDtcclxuICAgICAgICB0aGlzLl9zdWJtaXRCdXR0b25PYnNlcnZlciA9IG51bGw7XHJcbiAgICAgICAgdGhpcy5faW5wdXRPYnNlcnZlcnNEZXBsb3llZCA9IGZhbHNlOyAvLyBSZXNldCBmbGFnc1xyXG4gICAgICAgIHRoaXMuX21lc3NhZ2VPYnNlcnZlckRlcGxveWVkID0gZmFsc2U7XHJcbiAgICAgICAgY29uc29sZS5sb2coJ1tPYnNlcnZlck1hbmFnZXJdIEFsbCBvYnNlcnZlcnMgc3RvcHBlZC4nKTtcclxuICAgIH0sXHJcbiAgICBcclxuICAgIF9nZXRNZXNzYWdlTGlzdENvbnRhaW5lcigpIHtcclxuICAgICAgICAvLyAuLi4gKHRoaXMgZnVuY3Rpb24gcmVtYWlucyB1bmNoYW5nZWQpXHJcbiAgICAgICAgY29uc3QgbWFpbkVsID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcignbWFpbicpO1xyXG4gICAgICAgIGlmICghbWFpbkVsKSByZXR1cm4gbnVsbDtcclxuICAgICAgICBjb25zdCBtZXNzYWdlQ29udGFpbmVycyA9IGdldENvbnZlcnNhdGlvbk1lc3NhZ2VDb250YWluZXJzKG1haW5FbCk7XHJcbiAgICAgICAgaWYgKCFtZXNzYWdlQ29udGFpbmVycy5sZW5ndGgpIHJldHVybiBudWxsO1xyXG4gICAgICAgIGNvbnN0IGNvbnRhaW5zQWxsID0gKGVsKSA9PiBtZXNzYWdlQ29udGFpbmVycy5ldmVyeSgobWVzc2FnZUNvbnRhaW5lcikgPT4gZWwgJiYgZWwuY29udGFpbnMobWVzc2FnZUNvbnRhaW5lcikpO1xyXG4gICAgICAgIGxldCBjYW5kaWRhdGUgPSBtZXNzYWdlQ29udGFpbmVyc1swXS5wYXJlbnRFbGVtZW50O1xyXG4gICAgICAgIHdoaWxlIChjYW5kaWRhdGUgJiYgY2FuZGlkYXRlICE9PSBtYWluRWwgJiYgY29udGFpbnNBbGwoY2FuZGlkYXRlLnBhcmVudEVsZW1lbnQpKSB7XHJcbiAgICAgICAgICAgIGNhbmRpZGF0ZSA9IGNhbmRpZGF0ZS5wYXJlbnRFbGVtZW50O1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gY2FuZGlkYXRlIHx8IG1haW5FbDtcclxuICAgIH1cclxufTtcclxuICAgIFxyXG4gICAgZnVuY3Rpb24gc2V0dXBJbnB1dExpc3RlbmVycygpIHtcclxuICAgICAgICBjb25zdCBpbnB1dEJveCA9IGdldElucHV0Qm94KCk7XHJcbiAgICAgICAgaWYgKCFpbnB1dEJveCB8fCBpbnB1dEJveC5fX21heE1lbW9yeUJvdW5kKSByZXR1cm47XHJcbiAgICAgICAgY29uc3QgdXBkYXRlVmlzaWJpbGl0eSA9ICgpID0+IHtcclxuICAgICAgICAgICAgY29uc3Qgc3VibWl0QnV0dG9uID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihDSEFUR1BUX1NFTEVDVE9SUy5TVUJNSVRfQlVUVE9OKTtcclxuICAgICAgICAgICAgaWYgKCFzdWJtaXRCdXR0b24pIHJldHVybjtcclxuICAgICAgICAgICAgY29uc3QgaGFzQ29udGVudCA9IGdldElucHV0Q29udGVudChpbnB1dEJveCkubGVuZ3RoID4gMDtcclxuICAgICAgICAgICAgaWYgKGhhc0NvbnRlbnQpIHtcclxuICAgICAgICAgICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gJ2hpZGRlbic7XHJcbiAgICAgICAgICAgICAgICBzdWJtaXRCdXR0b24uc3R5bGUub3BhY2l0eSA9ICcwJztcclxuICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gJ3Zpc2libGUnO1xyXG4gICAgICAgICAgICAgICAgc3VibWl0QnV0dG9uLnN0eWxlLm9wYWNpdHkgPSAnMSc7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgICAgIGlucHV0Qm94LmFkZEV2ZW50TGlzdGVuZXIoJ2lucHV0JywgdXBkYXRlVmlzaWJpbGl0eSk7XHJcbiAgICAgICAgaW5wdXRCb3guYWRkRXZlbnRMaXN0ZW5lcigna2V5dXAnLCB1cGRhdGVWaXNpYmlsaXR5KTtcclxuICAgICAgICBpbnB1dEJveC5fX21heE1lbW9yeUJvdW5kID0gdHJ1ZTtcclxuICAgICAgICAvLyBSdW4gb25jZSBpbml0aWFsbHlcclxuICAgICAgICB1cGRhdGVWaXNpYmlsaXR5KCk7XHJcbiAgICB9O1xyXG5cclxuICAgIGFzeW5jIGZ1bmN0aW9uIGdldEFuZEluc2VydE1lbW9yaWVzKGJ1dHRvbikge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIC8vIENoZWNrIGlmIE1heE1lbW9yeSBpcyBlbmFibGVkIGJlZm9yZSBwcm9jZXNzaW5nXHJcbiAgICAgICAgICAgIGNvbnN0IHRvZ2dsZVJlc3BvbnNlID0gYXdhaXQgbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHtcclxuICAgICAgICAgICAgICAgIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHsgdHlwZTogJ0dFVF9NQVhNRU1PUllfRU5BQkxFRCcgfSwgcmVzb2x2ZSk7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgaWYgKHRvZ2dsZVJlc3BvbnNlICYmIHRvZ2dsZVJlc3BvbnNlLnN0YXR1cyA9PT0gJ3N1Y2Nlc3MnICYmICF0b2dnbGVSZXNwb25zZS5lbmFibGVkKSB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnTWF4TWVtb3J5IGlzIGRpc2FibGVkLCBza2lwcGluZyBtZW1vcnkgcHJvY2Vzc2luZycpO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBidXR0b24uZGlzYWJsZWQgPSB0cnVlO1xyXG4gICAgICAgICAgICBidXR0b24uY2xhc3NMaXN0LmFkZCgnbG9hZGluZycpO1xyXG5cclxuICAgICAgICAgICAgLy8gQmFja2VuZCBoYW5kbGVzIGFsbCBBUEkgcmVxdWVzdHMgLSBubyBBUEkga2V5IG9yIGFjY291bnQgcmVxdWlyZWQgZnJvbSB1c2VyXHJcblxyXG4gICAgICAgICAgICBjb25zdCBpbnB1dEJveCA9IGdldElucHV0Qm94KCk7XHJcbiAgICAgICAgICAgIGlmICghaW5wdXRCb3gpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0lucHV0IGJveCBub3QgZm91bmQuJyk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIFRyYWNrIGlucHV0IGJveCBub3QgZm91bmQgZXJyb3JcclxuICAgICAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tFcnJvcih7XHJcbiAgICAgICAgICAgICAgICAgICAgZXJyb3JfdHlwZTogJ2lucHV0X2JveF9ub3RfZm91bmQnLFxyXG4gICAgICAgICAgICAgICAgICAgIGVycm9yX21lc3NhZ2U6ICdDaGF0R1BUIGlucHV0IGJveCBub3QgZm91bmQnLFxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnRleHQ6ICdjb250ZW50X3NjcmlwdCcsXHJcbiAgICAgICAgICAgICAgICAgICAgZnVuY3Rpb246ICdnZXRBbmRJbnNlcnRNZW1vcmllcycsXHJcbiAgICAgICAgICAgICAgICAgICAgdXJsOiB3aW5kb3cubG9jYXRpb24uaHJlZlxyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgLy8gR2V0IHRoZSBjdXJyZW50IHVzZXIgaW5wdXQgd2hpbGUgcHJlc2VydmluZyBsaW5lIGJyZWFrc1xyXG4gICAgICAgICAgICBsZXQgdXNlcklucHV0ID0gJyc7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gR2V0dGluZyBpbnB1dCBmcm9tOicsIGlucHV0Qm94LnRhZ05hbWUpO1xyXG4gICAgICAgICAgICBpZiAoaW5wdXRCb3gudGFnTmFtZSA9PT0gJ1RFWFRBUkVBJykge1xyXG4gICAgICAgICAgICAgICAgdXNlcklucHV0ID0gaW5wdXRCb3gudmFsdWU7ICAvLyBSZW1vdmUgdHJpbSgpIHRvIHByZXNlcnZlIGxpbmUgYnJlYWtzXHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIFRleHRhcmVhIGlucHV0IGxlbmd0aDonLCB1c2VySW5wdXQubGVuZ3RoKTtcclxuICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgIC8vIEZvciBjb250ZW50ZWRpdGFibGUsIHByZXNlcnZlIGxpbmUgYnJlYWtzIGJldHdlZW4gcGFyYWdyYXBoc1xyXG4gICAgICAgICAgICAgICAgY29uc3QgcGFyYWdyYXBocyA9IGlucHV0Qm94LnF1ZXJ5U2VsZWN0b3JBbGwoJ3AnKTtcclxuICAgICAgICAgICAgICAgIHVzZXJJbnB1dCA9IEFycmF5LmZyb20ocGFyYWdyYXBocylcclxuICAgICAgICAgICAgICAgICAgICAubWFwKHAgPT4gcC50ZXh0Q29udGVudCkgIC8vIFJlbW92ZSB0cmltKCkgdG8gcHJlc2VydmUgc3BhY2luZ1xyXG4gICAgICAgICAgICAgICAgICAgIC5qb2luKCdcXG4nKTtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gQ29udGVudEVkaXRhYmxlIGlucHV0IGxlbmd0aDonLCB1c2VySW5wdXQubGVuZ3RoKTtcclxuICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLnNlYXJjaE1lbW9yaWVzKHVzZXJJbnB1dC50cmltKCkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgaWYgKHJlc3BvbnNlPy5zdGF0dXMgPT09ICdzdWNjZXNzJyAmJiByZXNwb25zZS5yZXN1bHRzLmxlbmd0aCkge1xyXG4gICAgICAgICAgICAgICAgY29uc3QgbGltaXRlZFJlc3VsdHMgPSByZXNwb25zZS5yZXN1bHRzLnNsaWNlKDAsIDEwKTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IG1lbW9yaWVzVGV4dCA9IGxpbWl0ZWRSZXN1bHRzXHJcbiAgICAgICAgICAgICAgICAgICAgLm1hcChtZW1vcnkgPT4gYFske2Zvcm1hdERhdGUobWVtb3J5LnRpbWVzdGFtcCl9XSAke21lbW9yeS5tZW1vcnlfdGV4dH1gKVxyXG4gICAgICAgICAgICAgICAgICAgIC5qb2luKCcgJyk7XHJcblxyXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ1tDb250ZW50U2NyaXB0XSBJbmplY3RpbmcgbWVtb3JpZXM6JywgbWVtb3JpZXNUZXh0LnN1YnN0cmluZygwLCA1MCkgKyAnLi4uJyk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gSW5qZWN0aW5nIG1lbW9yaWVzIGludG8gaW5wdXQgYm94Jyk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIFByZXBhcmUgdGhlIG5ldyBjb250ZW50IHdpdGggbWVtb3JpZXNcclxuICAgICAgICAgICAgICAgIGxldCBuZXdDb250ZW50O1xyXG4gICAgICAgICAgICAgICAgaWYgKGlucHV0Qm94LnRhZ05hbWUgPT09ICdURVhUQVJFQScpIHtcclxuICAgICAgICAgICAgICAgICAgICBuZXdDb250ZW50ID0gYFtSRUxFVkFOVF9QQVNUX01FTU9SSUVTX1NUQVJUXSAke21lbW9yaWVzVGV4dH0gW1JFTEVWQU5UX1BBU1RfTUVNT1JJRVNfRU5EXVxcblxcbiR7dXNlcklucHV0fWA7XHJcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIC8vIEZvciBjb250ZW50ZWRpdGFibGUsIHdlJ2xsIGhhbmRsZSB0aGUgSFRNTCBzdHJ1Y3R1cmUgaW4gc2V0SW5wdXRDb250ZW50XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc3QgbGluZXMgPSB1c2VySW5wdXQuc3BsaXQoJ1xcbicpO1xyXG4gICAgICAgICAgICAgICAgICAgIG5ld0NvbnRlbnQgPSBgW1JFTEVWQU5UX1BBU1RfTUVNT1JJRVNfU1RBUlRdICR7bWVtb3JpZXNUZXh0fSBbUkVMRVZBTlRfUEFTVF9NRU1PUklFU19FTkRdXFxuXFxuJHtsaW5lcy5qb2luKCdcXG4nKX1gO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAvLyBVc2UgdGhlIGhlbHBlciBmdW5jdGlvbiB0byBzZXQgY29udGVudCBhbmQgdHJpZ2dlciBuZWNlc3NhcnkgZXZlbnRzXHJcbiAgICAgICAgICAgICAgICBzZXRJbnB1dENvbnRlbnQoaW5wdXRCb3gsIG5ld0NvbnRlbnQpO1xyXG4gICAgICAgICAgICAgICAgYmVnaW5QZW5kaW5nTWVtb3J5U3R5bGluZ1dhdGNoKG1lbW9yaWVzVGV4dCk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gTWVtb3JpZXMgaW5qZWN0ZWQsIGNvbnRlbnQgbGVuZ3RoOicsIG5ld0NvbnRlbnQubGVuZ3RoKTtcclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgLy8gRm9jdXMgdGhlIGlucHV0IGFuZCBtb3ZlIGN1cnNvciB0byBlbmRcclxuICAgICAgICAgICAgICAgIGlucHV0Qm94LmZvY3VzKCk7XHJcbiAgICAgICAgICAgICAgICBjb25zdCBzZWxlY3Rpb24gPSB3aW5kb3cuZ2V0U2VsZWN0aW9uKCk7XHJcbiAgICAgICAgICAgICAgICBjb25zdCByYW5nZSA9IGRvY3VtZW50LmNyZWF0ZVJhbmdlKCk7XHJcbiAgICAgICAgICAgICAgICByYW5nZS5zZWxlY3ROb2RlQ29udGVudHMoaW5wdXRCb3gpO1xyXG4gICAgICAgICAgICAgICAgcmFuZ2UuY29sbGFwc2UoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgc2VsZWN0aW9uLnJlbW92ZUFsbFJhbmdlcygpO1xyXG4gICAgICAgICAgICAgICAgc2VsZWN0aW9uLmFkZFJhbmdlKHJhbmdlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGZldGNoaW5nIG1lbW9yaWVzOicsIGVycm9yKTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFRyYWNrIGVycm9yIGluIE1peHBhbmVsXHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tFcnJvcih7XHJcbiAgICAgICAgICAgICAgICBlcnJvcl90eXBlOiAnbWVtb3J5X2ZldGNoX2Vycm9yJyxcclxuICAgICAgICAgICAgICAgIGVycm9yX21lc3NhZ2U6IGVycm9yLm1lc3NhZ2UsXHJcbiAgICAgICAgICAgICAgICBlcnJvcl9zdGFjazogZXJyb3Iuc3RhY2ssXHJcbiAgICAgICAgICAgICAgICBjb250ZXh0OiAnY29udGVudF9zY3JpcHQnLFxyXG4gICAgICAgICAgICAgICAgZnVuY3Rpb246ICdnZXRBbmRJbnNlcnRNZW1vcmllcycsXHJcbiAgICAgICAgICAgICAgICB1c2VyX2lucHV0X2xlbmd0aDogdXNlcklucHV0ID8gdXNlcklucHV0Lmxlbmd0aCA6IDBcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgIH0gZmluYWxseSB7XHJcbiAgICAgICAgICAgIGJ1dHRvbi5kaXNhYmxlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICBidXR0b24uY2xhc3NMaXN0LnJlbW92ZSgnbG9hZGluZycpO1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIGdldEFuZEluc2VydE1lbW9yaWVzIGNvbXBsZXRlZCcpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcblxyXG4gICAgLy8gSGVscGVyIGZ1bmN0aW9uIHRvIGdldCBjaGF0IElEIGZyb20gVVJMXHJcbiAgICBjb25zdCBnZXRDaGF0SWQgPSAoKSA9PiB7XHJcbiAgICAgICAgY29uc3QgdXJsID0gd2luZG93LmxvY2F0aW9uLmhyZWY7XHJcbiAgICAgICAgY29uc3QgbWF0Y2ggPSB1cmwubWF0Y2goL1xcL2NcXC8oW2EtZjAtOS1dKykvKTtcclxuICAgICAgICByZXR1cm4gbWF0Y2ggPyBtYXRjaFsxXSA6ICdkZWZhdWx0JztcclxuICAgIH07XHJcblxyXG4gICAgY29uc3Qgc2V0TmF0aXZlU3VibWl0QnV0dG9uVmlzaWJpbGl0eSA9IChpc1Zpc2libGUpID0+IHtcclxuICAgICAgICBjb25zdCBzdWJtaXRCdXR0b24gPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKENIQVRHUFRfU0VMRUNUT1JTLlNVQk1JVF9CVVRUT04pO1xyXG4gICAgICAgIGlmICghc3VibWl0QnV0dG9uKSByZXR1cm47XHJcblxyXG4gICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gaXNWaXNpYmxlID8gJ3Zpc2libGUnIDogJ2hpZGRlbic7XHJcbiAgICAgICAgc3VibWl0QnV0dG9uLnN0eWxlLm9wYWNpdHkgPSBpc1Zpc2libGUgPyAnMScgOiAnMCc7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IHN5bmNNYXhNZW1vcnlUb2dnbGVVSSA9IChlbmFibGVkKSA9PiB7XHJcbiAgICAgICAgY29uc3QgdG9nZ2xlU3dpdGNoID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcignI21heG1lbW9yeS10b2dnbGUnKTtcclxuICAgICAgICBjb25zdCBidXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZ2V0LW1lbW9yaWVzLWJ1dHRvbicpO1xyXG4gICAgICAgIGNvbnN0IGlucHV0Qm94ID0gZ2V0SW5wdXRCb3goKTtcclxuICAgICAgICBjb25zdCBoYXNDb250ZW50ID0gaW5wdXRCb3ggPyAoZ2V0SW5wdXRDb250ZW50KGlucHV0Qm94KS5sZW5ndGggPiAwKSA6IGZhbHNlO1xyXG5cclxuICAgICAgICBpZiAodG9nZ2xlU3dpdGNoKSB7XHJcbiAgICAgICAgICAgIHRvZ2dsZVN3aXRjaC5jaGVja2VkID0gZW5hYmxlZDtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGlmIChidXR0b24gJiYgZW5hYmxlZCAmJiBoYXNDb250ZW50KSB7XHJcbiAgICAgICAgICAgIGJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ2ZsZXgnO1xyXG4gICAgICAgICAgICBidXR0b24uc3R5bGUudHJhbnNpdGlvbiA9ICdvcGFjaXR5IDAuMnMgZWFzZS1pbi1vdXQsIHRyYW5zZm9ybSAwLjJzIGVhc2UtaW4tb3V0JztcclxuICAgICAgICAgICAgYnV0dG9uLnN0eWxlLnZpc2liaWxpdHkgPSAndmlzaWJsZSc7XHJcbiAgICAgICAgICAgIGJ1dHRvbi5zdHlsZS5vcGFjaXR5ID0gJzEnO1xyXG4gICAgICAgICAgICBidXR0b24uc3R5bGUudHJhbnNmb3JtID0gJ3RyYW5zbGF0ZVkoMCknO1xyXG4gICAgICAgIH0gZWxzZSBpZiAoYnV0dG9uKSB7XHJcbiAgICAgICAgICAgIGJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gJ2hpZGRlbic7XHJcbiAgICAgICAgICAgIGJ1dHRvbi5zdHlsZS5vcGFjaXR5ID0gJzAnO1xyXG4gICAgICAgICAgICBidXR0b24uc3R5bGUudHJhbnNmb3JtID0gJ3RyYW5zbGF0ZVkoMTBweCknO1xyXG5cclxuICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBpZiAoYnV0dG9uLnN0eWxlLm9wYWNpdHkgPT09ICcwJykge1xyXG4gICAgICAgICAgICAgICAgICAgIGJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9LCAyMDApO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgc2V0TmF0aXZlU3VibWl0QnV0dG9uVmlzaWJpbGl0eSghKGVuYWJsZWQgJiYgaGFzQ29udGVudCkpO1xyXG4gICAgfTtcclxuXHJcblxyXG5cclxuICAgIGNvbnN0IGNyZWF0ZU1heE1lbW9yeUludGVyZmFjZSA9IGFzeW5jICgpID0+IHtcclxuICAgICAgICAvLyBDcmVhdGUgbWFpbiBjb250YWluZXIgdXNpbmcgYmx1ZXByaW50XHJcbiAgICAgICAgY29uc3QgY29udGFpbmVyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XHJcbiAgICAgICAgY29udGFpbmVyLmlubmVySFRNTCA9IHVpQmx1ZXByaW50cy5nZXRNYWluQ29udGFpbmVyKCk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gR2V0IHJlZmVyZW5jZXMgdG8gdGhlIGNyZWF0ZWQgZWxlbWVudHNcclxuICAgICAgICBjb25zdCBzZXR0aW5nc0J1dHRvbiA9IGNvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcubWF4bWVtb3J5LXNldHRpbmdzLWJ1dHRvbicpO1xyXG4gICAgICAgIGNvbnN0IGJ1dHRvbiA9IGNvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcjZ2V0LW1lbW9yaWVzLWJ1dHRvbicpO1xyXG4gICAgICAgIGNvbnN0IHRvZ2dsZVN3aXRjaCA9IGNvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcjbWF4bWVtb3J5LXRvZ2dsZScpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEluaXRpYWxpemUgdG9nZ2xlIHN0YXRlIGZyb20gc3RvcmFnZS5cclxuICAgICAgICAvLyBJTVBPUlRBTlQ6IFdlIHNldCB0b2dnbGVTd2l0Y2guY2hlY2tlZCBkaXJlY3RseSBoZXJlIGJlY2F1c2UgdGhlIGVsZW1lbnRcclxuICAgICAgICAvLyBpcyBub3QgeWV0IGluIHRoZSBET00sIHNvIHN5bmNNYXhNZW1vcnlUb2dnbGVVSSAod2hpY2ggcXVlcmllcyBkb2N1bWVudClcclxuICAgICAgICAvLyB3b3VsZCBmaW5kIG5vdGhpbmcgYW5kIHNpbGVudGx5IGRvIG5vdGhpbmcuXHJcbiAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyB0eXBlOiAnR0VUX01BWE1FTU9SWV9FTkFCTEVEJyB9LCByZXNvbHZlKTtcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBpZiAocmVzcG9uc2UgJiYgcmVzcG9uc2Uuc3RhdHVzID09PSAnc3VjY2VzcycpIHtcclxuICAgICAgICAgICAgICAgIHRvZ2dsZVN3aXRjaC5jaGVja2VkID0gcmVzcG9uc2UuZW5hYmxlZDtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGdldHRpbmcgTWF4TWVtb3J5IGVuYWJsZWQgc3RhdGU6JywgZXJyb3IpO1xyXG4gICAgICAgICAgICB0b2dnbGVTd2l0Y2guY2hlY2tlZCA9IHRydWU7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEFkZCB0b2dnbGUgZXZlbnQgbGlzdGVuZXJcclxuICAgICAgICB0b2dnbGVTd2l0Y2guYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgYXN5bmMgKGUpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgZW5hYmxlZCA9IGUudGFyZ2V0LmNoZWNrZWQ7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICB0cnkge1xyXG4gICAgICAgICAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHsgXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHR5cGU6ICdTRVRfTUFYTUVNT1JZX0VOQUJMRUQnLCBcclxuICAgICAgICAgICAgICAgICAgICAgICAgZW5hYmxlZDogZW5hYmxlZCBcclxuICAgICAgICAgICAgICAgICAgICB9LCByZXNvbHZlKTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICBpZiAocmVzcG9uc2UgJiYgcmVzcG9uc2Uuc3RhdHVzID09PSAnc3VjY2VzcycpIHtcclxuICAgICAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnTWF4TWVtb3J5IHRvZ2dsZSBzdGF0ZSB1cGRhdGVkOicsIGVuYWJsZWQpO1xyXG4gICAgICAgICAgICAgICAgICAgIHN5bmNNYXhNZW1vcnlUb2dnbGVVSShlbmFibGVkKTtcclxuICAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcignRmFpbGVkIHRvIHVwZGF0ZSBNYXhNZW1vcnkgdG9nZ2xlIHN0YXRlJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gUmV2ZXJ0IHRvZ2dsZSBzdGF0ZSBvbiBlcnJvclxyXG4gICAgICAgICAgICAgICAgICAgIGUudGFyZ2V0LmNoZWNrZWQgPSAhZW5hYmxlZDtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIHVwZGF0aW5nIE1heE1lbW9yeSB0b2dnbGUgc3RhdGU6JywgZXJyb3IpO1xyXG4gICAgICAgICAgICAgICAgLy8gUmV2ZXJ0IHRvZ2dsZSBzdGF0ZSBvbiBlcnJvclxyXG4gICAgICAgICAgICAgICAgZS50YXJnZXQuY2hlY2tlZCA9ICFlbmFibGVkO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gQWRkIHNldHRpbmdzIGJ1dHRvbiBldmVudCBsaXN0ZW5lclxyXG4gICAgICAgIHNldHRpbmdzQnV0dG9uLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKGUpID0+IHtcclxuICAgICAgICAgICAgZS5wcmV2ZW50RGVmYXVsdCgpO1xyXG4gICAgICAgICAgICBlLnN0b3BQcm9wYWdhdGlvbigpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gVHJhY2sgcG9wdXAgb3BlbmVkIGZyb20gc2V0dGluZ3MgYnV0dG9uXHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tQb3B1cE9wZW5lZCgnc2V0dGluZ3NfYnV0dG9uJyk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBPcGVuIGV4dGVuc2lvbiBwb3B1cCBpbiBhIG5ldyB0YWJcclxuICAgICAgICAgICAgYmFja2dyb3VuZEFQSS5vcGVuUG9wdXBJblRhYigpO1xyXG4gICAgICAgIH0pO1xyXG5cclxuICAgICAgICAvLyBDaGVjayBtZW1vcnkgbGltaXQgYW5kIGFkZCByZWQgZG90IGlmIG5lZWRlZFxyXG4gICAgICAgIGNoZWNrTWVtb3J5TGltaXRBbmRVcGRhdGVCdXR0b24oc2V0dGluZ3NCdXR0b24pO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFN1Ym1pdCBidXR0b24gaXMgYWxyZWFkeSBjcmVhdGVkIGJ5IHRoZSBibHVlcHJpbnRcclxuXHJcbiAgICAgICAgLy8gRnVuY3Rpb24gdG8gdXBkYXRlIGJ1dHRvbiB2aXNpYmlsaXR5IGJhc2VkIG9uIGlucHV0IGNvbnRlbnRcclxuICAgICAgICBjb25zdCB1cGRhdGVCdXR0b25WaXNpYmlsaXR5ID0gKGhhc0NvbnRlbnQgPSBudWxsKSA9PiB7XHJcbiAgICAgICAgICAgIC8vIENoZWNrIGlmIHRoZXJlJ3MgY29udGVudCBpbiB0aGUgaW5wdXQgaWYgbm90IHByb3ZpZGVkXHJcbiAgICAgICAgICAgIGlmIChoYXNDb250ZW50ID09PSBudWxsKSB7XHJcbiAgICAgICAgICAgICAgICBjb25zdCBpbnB1dEJveCA9IGdldElucHV0Qm94KCk7XHJcbiAgICAgICAgICAgICAgICBoYXNDb250ZW50ID0gaW5wdXRCb3ggPyAoZ2V0SW5wdXRDb250ZW50KGlucHV0Qm94KS5sZW5ndGggPiAwKSA6IGZhbHNlO1xyXG4gICAgICAgICAgICB9XHJcblxyXG4gICAgICAgICAgICBzeW5jTWF4TWVtb3J5VG9nZ2xlVUkodG9nZ2xlU3dpdGNoLmNoZWNrZWQpO1xyXG4gICAgICAgIH07XHJcblxyXG4gICAgICAgIC8vIEZ1bmN0aW9uIHRvIG1vbml0b3IgaW5wdXQgY2hhbmdlcyBhbmQgdXBkYXRlIGJ1dHRvbiB2aXNpYmlsaXR5XHJcbiAgICAgICAgY29uc3QgbW9uaXRvcklucHV0Q2hhbmdlcyA9IGFzeW5jICgpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgaW5wdXRCb3ggPSBnZXRJbnB1dEJveCgpO1xyXG4gICAgICAgICAgICBpZiAoIWlucHV0Qm94KSByZXR1cm47XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBjb25zdCBjb250ZW50ID0gZ2V0SW5wdXRDb250ZW50KGlucHV0Qm94KTtcclxuICAgICAgICAgICAgY29uc3QgaGFzQ29udGVudCA9IGNvbnRlbnQgJiYgY29udGVudC5sZW5ndGggPiAwO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgdXBkYXRlQnV0dG9uVmlzaWJpbGl0eShoYXNDb250ZW50KTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEFkZCBpbnB1dCBldmVudCBsaXN0ZW5lcnMgdG8gbW9uaXRvciB0ZXh0IGNoYW5nZXNcclxuICAgICAgICBjb25zdCBzZXR1cElucHV0TW9uaXRvcmluZyA9ICgpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgaW5wdXRCb3ggPSBnZXRJbnB1dEJveCgpO1xyXG4gICAgICAgICAgICBpZiAoaW5wdXRCb3gpIHtcclxuICAgICAgICAgICAgICAgIC8vIEFkZCBldmVudCBsaXN0ZW5lcnMgZm9yIGlucHV0IGNoYW5nZXNcclxuICAgICAgICAgICAgICAgIGlucHV0Qm94LmFkZEV2ZW50TGlzdGVuZXIoJ2lucHV0JywgbW9uaXRvcklucHV0Q2hhbmdlcyk7XHJcbiAgICAgICAgICAgICAgICBpbnB1dEJveC5hZGRFdmVudExpc3RlbmVyKCdrZXl1cCcsIG1vbml0b3JJbnB1dENoYW5nZXMpO1xyXG4gICAgICAgICAgICAgICAgaW5wdXRCb3guYWRkRXZlbnRMaXN0ZW5lcigncGFzdGUnLCAoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgc2V0VGltZW91dChtb25pdG9ySW5wdXRDaGFuZ2VzLCAxMCk7IC8vIFNtYWxsIGRlbGF5IGZvciBwYXN0ZSB0byBjb21wbGV0ZVxyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIEFsc28gbW9uaXRvciBmb3IgcHJvZ3JhbW1hdGljIGNoYW5nZXNcclxuICAgICAgICAgICAgICAgIGNvbnN0IG9ic2VydmVyID0gbmV3IE11dGF0aW9uT2JzZXJ2ZXIobW9uaXRvcklucHV0Q2hhbmdlcyk7XHJcbiAgICAgICAgICAgICAgICBvYnNlcnZlci5vYnNlcnZlKGlucHV0Qm94LCB7XHJcbiAgICAgICAgICAgICAgICAgICAgY2hpbGRMaXN0OiB0cnVlLFxyXG4gICAgICAgICAgICAgICAgICAgIHN1YnRyZWU6IHRydWUsXHJcbiAgICAgICAgICAgICAgICAgICAgY2hhcmFjdGVyRGF0YTogdHJ1ZVxyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEluaXRpYWxpemUgVUlcclxuICAgICAgICB1cGRhdGVCdXR0b25WaXNpYmlsaXR5KGZhbHNlKTsgLy8gU3RhcnQgd2l0aCBidXR0b24gaGlkZGVuXHJcbiAgICAgICAgc2V0dXBJbnB1dE1vbml0b3JpbmcoKTtcclxuXHJcbiAgICAgICAgYnV0dG9uLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgYXN5bmMgKGUpID0+IHtcclxuICAgICAgICAgICAgZS5wcmV2ZW50RGVmYXVsdCgpO1xyXG4gICAgICAgICAgICBlLnN0b3BQcm9wYWdhdGlvbigpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgYXdhaXQgZ2V0QW5kSW5zZXJ0TWVtb3JpZXMoYnV0dG9uKTsgLy8gQmFja2VuZCBhdXRvLWRldGVjdHMgbW9kZVxyXG5cclxuICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gRGlyZWN0bHkgY2xpY2sgdGhlIG9yaWdpbmFsIHN1Ym1pdCBidXR0b24gdG8gYXZvaWQgcmVjdXJzaXZlIEVudGVyIGtleSBoYW5kbGluZ1xyXG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IHN1Ym1pdEJ1dHRvbiA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoQ0hBVEdQVF9TRUxFQ1RPUlMuU1VCTUlUX0JVVFRPTik7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHN1Ym1pdEJ1dHRvbiAmJiAhc3VibWl0QnV0dG9uLmRpc2FibGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIFRlbXBvcmFyaWx5IHNob3cgdGhlIHN1Ym1pdCBidXR0b24gdG8gZW5zdXJlIGl0IGNhbiBiZSBjbGlja2VkXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnN0IG9yaWdpbmFsVmlzaWJpbGl0eSA9IHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjb25zdCBvcmlnaW5hbE9wYWNpdHkgPSBzdWJtaXRCdXR0b24uc3R5bGUub3BhY2l0eTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgc3VibWl0QnV0dG9uLnN0eWxlLnZpc2liaWxpdHkgPSAndmlzaWJsZSc7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS5vcGFjaXR5ID0gJzEnO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICAgICAgc3VibWl0QnV0dG9uLmNsaWNrKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBSZXN0b3JlIG9yaWdpbmFsIHZpc2liaWxpdHkgYWZ0ZXIgYSBzaG9ydCBkZWxheVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gb3JpZ2luYWxWaXNpYmlsaXR5O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgc3VibWl0QnV0dG9uLnN0eWxlLm9wYWNpdHkgPSBvcmlnaW5hbE9wYWNpdHk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH0sIDUwKTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9LCAxMDApO1xyXG4gICAgICAgIH0pO1xyXG5cclxuICAgICAgICAvLyBBbGwgY29tcG9uZW50cyBhcmUgYWxyZWFkeSBhc3NlbWJsZWQgYnkgdGhlIGJsdWVwcmludFxyXG4gICAgICAgIHJldHVybiBjb250YWluZXIuZmlyc3RFbGVtZW50Q2hpbGQ7IC8vIFJldHVybiB0aGUgYWN0dWFsIGNvbnRhaW5lciBkaXYsIG5vdCB0aGUgd3JhcHBlclxyXG4gICAgfTtcclxuXHJcbiAgICBmdW5jdGlvbiBhZGRHZXRNZW1vcmllc0J1dHRvbigpIHtcclxuICAgICAgICAvLyBDbGVhciBhbnkgcHJldmlvdXMgdGltZXIgdG8gYXZvaWQgbXVsdGlwbGUgYXR0ZW1wdHNcclxuICAgICAgICBpZiAod2luZG93Lm1lbW9yeVZhdWx0QnV0dG9uVGltZXIpIHtcclxuICAgICAgICAgICAgY2xlYXJUaW1lb3V0KHdpbmRvdy5tZW1vcnlWYXVsdEJ1dHRvblRpbWVyKTtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIC8vIFVzZSBhIG1vcmUgc3BlY2lmaWMgY29udGFpbmVyIElEIHRvIGNoZWNrIGlmIHRoZSBidXR0b24gYWxyZWFkeSBleGlzdHNcclxuICAgICAgICBpZiAoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21heG1lbW9yeS1jb250YWluZXInKSkge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnTWVtb3J5IHZhdWx0IGNvbnRhaW5lciBhbHJlYWR5IGV4aXN0cywgc2tpcHBpbmcgY3JlYXRpb24nKTtcclxuICAgICAgICAgICAgcmV0dXJuOyAvLyBVSSBjb21wb25lbnRzIGFscmVhZHkgYWRkZWRcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGNvbnNvbGUubG9nKCdBZGRpbmcgTWF4TWVtb3J5IGJ1dHRvbiB0byBwYWdlJyk7XHJcblxyXG4gICAgICAgIC8vIENoZWNrIGlmIGlucHV0IGJveCBleGlzdHNcclxuICAgICAgICBjb25zdCBpbnB1dEJveCA9IGdldElucHV0Qm94KCk7XHJcbiAgICAgICAgaWYgKCFpbnB1dEJveCkge1xyXG4gICAgICAgICAgICAvLyBJZiBpbnB1dCBib3ggaXNuJ3QgcmVhZHksIHRyeSBhZ2FpbiBpbiBhIHNob3J0IHdoaWxlXHJcbiAgICAgICAgICAgIHdpbmRvdy5tZW1vcnlWYXVsdEJ1dHRvblRpbWVyID0gc2V0VGltZW91dChhZGRHZXRNZW1vcmllc0J1dHRvbiwgNTAwKTtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgLy8gQWRkIGEgc2xpZ2h0IGRlbGF5IHRvIGF2b2lkIHJhY2UgY29uZGl0aW9ucyBiZXR3ZWVuIHRhYnNcclxuICAgICAgICB3aW5kb3cubWVtb3J5VmF1bHRCdXR0b25UaW1lciA9IHNldFRpbWVvdXQoKCkgPT4ge1xyXG4gICAgICAgICAgICAvLyBEb3VibGUtY2hlY2sgYWdhaW4gYWZ0ZXIgdGhlIGRlbGF5IHRvIG1ha2UgYWJzb2x1dGVseSBzdXJlXHJcbiAgICAgICAgICAgIGlmIChkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWF4bWVtb3J5LWNvbnRhaW5lcicpKSB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnTWVtb3J5IHZhdWx0IGNvbnRhaW5lciBhbHJlYWR5IGV4aXN0cyAoc2Vjb25kIGNoZWNrKSwgc2tpcHBpbmcgY3JlYXRpb24nKTtcclxuICAgICAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgY29uc29sZS5sb2coJ0NyZWF0aW5nIG1lbW9yeSB2YXVsdCBjb250YWluZXInKTtcclxuICAgICAgICAgICAgY29uc3QgY29udGFpbmVyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XHJcbiAgICAgICAgICAgIGNvbnRhaW5lci5pZCA9ICdtYXhtZW1vcnktY29udGFpbmVyJzsgLy8gQWRkIGEgc3BlY2lmaWMgSUQgdG8gdGhlIGNvbnRhaW5lclxyXG4gICAgICAgICAgICBjb250YWluZXIuc3R5bGUuZGlzcGxheSA9ICdmbGV4JztcclxuICAgICAgICAgICAgY29udGFpbmVyLnN0eWxlLm1hcmdpbkJvdHRvbSA9ICcxMnB4JztcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIGNyZWF0ZU1heE1lbW9yeUludGVyZmFjZSgpLnRoZW4obWVtb3JpZXNCdXR0b25Db250YWluZXIgPT4ge1xyXG4gICAgICAgICAgICAgICAgLy8gRmluYWwgY2hlY2sgYmVmb3JlIGFwcGVuZGluZyB0byBET01cclxuICAgICAgICAgICAgICAgIGlmIChkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWF4bWVtb3J5LWNvbnRhaW5lcicpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ01lbW9yeSB2YXVsdCBjb250YWluZXIgYWxyZWFkeSBleGlzdHMgKGZpbmFsIGNoZWNrKSwgc2tpcHBpbmcgY3JlYXRpb24nKTtcclxuICAgICAgICAgICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGNvbnRhaW5lci5hcHBlbmRDaGlsZChtZW1vcmllc0J1dHRvbkNvbnRhaW5lcik7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGNvbnN0IHRhcmdldCA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoQ0hBVEdQVF9TRUxFQ1RPUlMuRk9STSk7XHJcblxyXG4gICAgICAgICAgICAgICAgaWYgKHRhcmdldCkge1xyXG4gICAgICAgICAgICAgICAgICAgIHRhcmdldC5wYXJlbnROb2RlLmluc2VydEJlZm9yZShjb250YWluZXIsIHRhcmdldCk7XHJcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIC8vIElmIHRhcmdldCBjb250YWluZXIgaXNuJ3QgcmVhZHksIHRyeSBhZ2FpblxyXG4gICAgICAgICAgICAgICAgICAgIHNldFRpbWVvdXQoYWRkR2V0TWVtb3JpZXNCdXR0b24sIDUwMCk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH0sIDEwMCk7IC8vIFNob3J0IGRlbGF5IHRvIGF2b2lkIHJhY2UgY29uZGl0aW9uc1xyXG4gICAgfTtcclxuXHJcbiAgICBmdW5jdGlvbiBoYW5kbGVFbnRlcktleShldmVudCkge1xyXG4gICAgICAgIC8vIFRoaXMgaXMgbm93IGFuIElOU1RBTlQsIFNZTkNIUk9OT1VTIGNoZWNrLlxyXG4gICAgICAgIGlmIChldmVudC5rZXkgIT09ICdFbnRlcicgJiYgZXZlbnQua2V5ICE9PSAnTnVtcGFkRW50ZXInIHx8IGV2ZW50LnNoaWZ0S2V5IHx8IGV2ZW50LmlzQ29tcG9zaW5nKSB7XHJcbiAgICAgICAgICAgIHJldHVybiB0cnVlOyAvLyBMZXQgdGhlIG5hdGl2ZSBoYW5kbGVyIHByb2NlZWQuXHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICAvLyBJZiB3ZSBnZXQgaGVyZSwgd2UgYXJlIHByb2Nlc3NpbmcgRW50ZXIga2V5LiBQUkVWRU5UIEZJUlNULlxyXG4gICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XHJcbiAgICAgICAgZXZlbnQuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgY29uc29sZS5sb2coJ1tDb250ZW50U2NyaXB0XSBQcm9jZXNzaW5nIEVudGVyIGtleSB3aXRoIE1heE1lbW9yeScpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIE5vdywgYW5kIG9ubHkgbm93LCBkbyB3ZSBwcm9jZWVkIHdpdGggdGhlIGFzeW5jIGxvZ2ljLlxyXG4gICAgICAgIC8vIFdlIGNhbiBtYWtlIHRoaXMgYW4gYXN5bmMgSUlGRSB0byBrZWVwIHRoZSBoYW5kbGVyIGNsZWFuLlxyXG4gICAgICAgIChhc3luYyAoKSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IGlucHV0Qm94ID0gZ2V0SW5wdXRCb3goKTtcclxuICAgICAgICAgICAgY29uc3QgaW5wdXRDb250ZW50ID0gZ2V0SW5wdXRDb250ZW50KGlucHV0Qm94KTtcclxuICAgICAgICAgICAgaWYgKCFpbnB1dENvbnRlbnQgfHwgIWlucHV0Qm94KSB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIE5vIGlucHV0IGNvbnRlbnQgb3IgaW5wdXQgYm94IGZvdW5kJyk7XHJcbiAgICAgICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgICAgIH1cclxuXHJcbiAgICAgICAgICAgIGNvbnN0IG1lbW9yaWVzQnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2dldC1tZW1vcmllcy1idXR0b24nKTtcclxuICAgICAgICAgICAgaWYgKG1lbW9yaWVzQnV0dG9uICYmIG1lbW9yaWVzQnV0dG9uLnN0eWxlLmRpc3BsYXkgIT09ICdub25lJyAmJiBtZW1vcmllc0J1dHRvbi5zdHlsZS52aXNpYmlsaXR5ICE9PSAnaGlkZGVuJykge1xyXG4gICAgICAgICAgICAgICAgLy8gSWYgTWF4TWVtb3J5IGJ1dHRvbiBpcyB2aXNpYmxlLCB0cmlnZ2VyIGl0IGRpcmVjdGx5XHJcbiAgICAgICAgICAgICAgICBtZW1vcmllc0J1dHRvbi5jbGljaygpO1xyXG4gICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgLy8gRmFsbGJhY2s6IHByb2Nlc3MgbWVtb3JpZXMgYW5kIHN1Ym1pdCBtYW51YWxseVxyXG4gICAgICAgICAgICAgICAgYXdhaXQgZ2V0QW5kSW5zZXJ0TWVtb3JpZXMobWVtb3JpZXNCdXR0b24gfHwgeyBkaXNhYmxlZDogZmFsc2UsIGNsYXNzTGlzdDogeyBhZGQ6ICgpID0+IHt9LCByZW1vdmU6ICgpID0+IHt9IH0gfSk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIEFkZCBhIGxvbmdlciBkZWxheSB0byBlbnN1cmUgbWVtb3JpZXMgYXJlIHByb3Blcmx5IGluamVjdGVkIGJlZm9yZSBzdWJtaXR0aW5nXHJcbiAgICAgICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIFN1Ym1pdHRpbmcgYWZ0ZXIgbWVtb3J5IGluamVjdGlvbicpO1xyXG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IHN1Ym1pdEJ1dHRvbiA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoQ0hBVEdQVF9TRUxFQ1RPUlMuU1VCTUlUX0JVVFRPTik7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHN1Ym1pdEJ1dHRvbiAmJiAhc3VibWl0QnV0dG9uLmRpc2FibGVkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIE1ha2Ugc3VyZSB0aGUgYnV0dG9uIGlzIHZpc2libGUgYmVmb3JlIGNsaWNraW5nXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHN1Ym1pdEJ1dHRvbi5zdHlsZS52aXNpYmlsaXR5ID0gJ3Zpc2libGUnO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBzdWJtaXRCdXR0b24uc3R5bGUub3BhY2l0eSA9ICcxJztcclxuICAgICAgICAgICAgICAgICAgICAgICAgc3VibWl0QnV0dG9uLmNsaWNrKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gU3VibWl0IGJ1dHRvbiBjbGlja2VkJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ1tDb250ZW50U2NyaXB0XSBTdWJtaXQgYnV0dG9uIG5vdCBmb3VuZCBvciBkaXNhYmxlZCcpO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH0sIDMwMCk7IC8vIEluY3JlYXNlZCBkZWxheSB0byAzMDBtc1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSkoKTtcclxuICAgICAgICBcclxuICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IHNldHVwRW50ZXJLZXlQcmV2ZW50aW9uID0gKCkgPT4ge1xyXG4gICAgICAgIC8vIFRhcmdldCB0aGUgc3BlY2lmaWMgZWxlbWVudHNcclxuICAgICAgICBjb25zdCBwcm9zZU1pcnJvckVkaXRvciA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJy5Qcm9zZU1pcnJvcicpO1xyXG4gICAgICAgIGNvbnN0IGZpZWxkc2V0ID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcignZmllbGRzZXQuZmxleCcpO1xyXG4gICAgICAgIGNvbnN0IGNvbnRlbnRFZGl0YWJsZURpdiA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJ1tjb250ZW50ZWRpdGFibGU9XCJ0cnVlXCJdJyk7XHJcbiAgICAgICAgY29uc3QgcHJvbXB0VGV4dGFyZWEgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKENIQVRHUFRfU0VMRUNUT1JTLklOUFVUX0JPWCk7XHJcblxyXG4gICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gU2V0dGluZyB1cCBFbnRlciBrZXkgcHJldmVudGlvbjonLCB7XHJcbiAgICAgICAgICAgIHByb3NlTWlycm9yRWRpdG9yOiAhIXByb3NlTWlycm9yRWRpdG9yLFxyXG4gICAgICAgICAgICBmaWVsZHNldDogISFmaWVsZHNldCxcclxuICAgICAgICAgICAgY29udGVudEVkaXRhYmxlRGl2OiAhIWNvbnRlbnRFZGl0YWJsZURpdixcclxuICAgICAgICAgICAgcHJvbXB0VGV4dGFyZWE6ICEhcHJvbXB0VGV4dGFyZWFcclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgLy8gQWx3YXlzIGF0dGFjaCB0byB3aW5kb3cgc28gd2UgY2FuIGludGVyY2VwdCBldmVuIGlmIGVsZW1lbnRzIGFyZSBtaXNzaW5nXHJcbiAgICAgICAgW3dpbmRvdywgcHJvc2VNaXJyb3JFZGl0b3IsIGZpZWxkc2V0LCBjb250ZW50RWRpdGFibGVEaXYsIHByb21wdFRleHRhcmVhXS5mb3JFYWNoKChlbGVtZW50LCBpbmRleCkgPT4ge1xyXG4gICAgICAgICAgICBpZiAoZWxlbWVudCkge1xyXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coYFtDb250ZW50U2NyaXB0XSBBZGRpbmcgRW50ZXIga2V5IGxpc3RlbmVycyB0byBlbGVtZW50ICR7aW5kZXh9OmAsIGVsZW1lbnQudGFnTmFtZSB8fCAnd2luZG93Jyk7XHJcbiAgICAgICAgICAgICAgICBlbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCBoYW5kbGVFbnRlcktleSwgeyBjYXB0dXJlOiB0cnVlIH0pO1xyXG4gICAgICAgICAgICAgICAgZWxlbWVudC5hZGRFdmVudExpc3RlbmVyKCdrZXlwcmVzcycsIGhhbmRsZUVudGVyS2V5LCB7IGNhcHR1cmU6IHRydWUgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgLy8gSWYgdGhlIHByb21wdCB0ZXh0YXJlYSBpc24ndCBhdmFpbGFibGUgeWV0LCB0cnkgdG8gZmluZCBpdCBhZ2FpblxyXG4gICAgICAgIC8vIE5vdGU6IFRoaXMgZnVuY3Rpb24gaXMgbm93IGNhbGxlZCBmcm9tIG9uVUlSZWFkeSB3aGVuIGVsZW1lbnRzIGFyZSBndWFyYW50ZWVkIHRvIGJlIHByZXNlbnRcclxuICAgICAgICBpZiAoIXByb21wdFRleHRhcmVhKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gUHJvbXB0IHRleHRhcmVhIG5vdCBmb3VuZCwgYXR0ZW1wdGluZyB0byBmaW5kIGl0IGFnYWluLi4uJyk7XHJcbiAgICAgICAgICAgIGNvbnN0IGVsID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihDSEFUR1BUX1NFTEVDVE9SUy5JTlBVVF9CT1gpO1xyXG4gICAgICAgICAgICBpZiAoZWwpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gUHJvbXB0IHRleHRhcmVhIGZvdW5kIOKAlCBhdHRhY2hpbmcgRW50ZXIga2V5IGxpc3RlbmVycycpO1xyXG4gICAgICAgICAgICAgICAgZWwuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIGhhbmRsZUVudGVyS2V5LCB7IGNhcHR1cmU6IHRydWUgfSk7XHJcbiAgICAgICAgICAgICAgICBlbC5hZGRFdmVudExpc3RlbmVyKCdrZXlwcmVzcycsIGhhbmRsZUVudGVyS2V5LCB7IGNhcHR1cmU6IHRydWUgfSk7XHJcbiAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIFByb21wdCB0ZXh0YXJlYSBzdGlsbCBub3QgZm91bmQgLSB0aGlzIHNob3VsZCBub3QgaGFwcGVuIHdoZW4gY2FsbGVkIGZyb20gb25VSVJlYWR5Jyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIC8vIE92ZXJyaWRlIHRoZWlyIGV2ZW50IGhhbmRsZXIgc2V0dXAgZm9yIFByb3NlTWlycm9yIGlmIHByZXNlbnRcclxuICAgICAgICBjb25zdCBvcmlnaW5hbEFkZEV2ZW50TGlzdGVuZXIgPSBFdmVudFRhcmdldC5wcm90b3R5cGUuYWRkRXZlbnRMaXN0ZW5lcjtcclxuICAgICAgICBFdmVudFRhcmdldC5wcm90b3R5cGUuYWRkRXZlbnRMaXN0ZW5lciA9IGZ1bmN0aW9uKHR5cGUsIGxpc3RlbmVyLCBvcHRpb25zKSB7XHJcbiAgICAgICAgICAgIGlmICgodHlwZSA9PT0gJ2tleXByZXNzJyB8fCB0eXBlID09PSAna2V5ZG93bicpICYmIHRoaXMuY2xhc3NMaXN0Py5jb250YWlucygnUHJvc2VNaXJyb3InKSkge1xyXG4gICAgICAgICAgICAgICAgY29uc3Qgd3JhcHBlZExpc3RlbmVyID0gZnVuY3Rpb24oZXZlbnQpIHtcclxuICAgICAgICAgICAgICAgICAgICBpZiAoKGV2ZW50LmtleSA9PT0gJ0VudGVyJyB8fCBldmVudC5rZXkgPT09ICdOdW1wYWRFbnRlcicpICYmICFldmVudC5zaGlmdEtleSAmJiAhZXZlbnQuaXNDb21wb3NpbmcpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGhhbmRsZUVudGVyS2V5KGV2ZW50KTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIGxpc3RlbmVyLmFwcGx5KHRoaXMsIGFyZ3VtZW50cyk7XHJcbiAgICAgICAgICAgICAgICB9O1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuIG9yaWdpbmFsQWRkRXZlbnRMaXN0ZW5lci5jYWxsKHRoaXMsIHR5cGUsIHdyYXBwZWRMaXN0ZW5lciwgb3B0aW9ucyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcmV0dXJuIG9yaWdpbmFsQWRkRXZlbnRMaXN0ZW5lci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xyXG4gICAgICAgIH07XHJcblxyXG4gICAgICAgIC8vIFByZXZlbnQgZm9ybSBzdWJtaXNzaW9uIGlmIGFueVxyXG4gICAgICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ3N1Ym1pdCcsIChlKSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gRm9ybSBzdWJtaXQgZXZlbnQgaW50ZXJjZXB0ZWQ6JywgZS50YXJnZXQpO1xyXG4gICAgICAgICAgICBlLnByZXZlbnREZWZhdWx0KCk7XHJcbiAgICAgICAgICAgIGUuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcclxuICAgICAgICB9LCB0cnVlKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBBbHNvIHByZXZlbnQgZm9ybSBzdWJtaXNzaW9uIG9uIHRoZSBzcGVjaWZpYyBmb3JtXHJcbiAgICAgICAgY29uc3QgZm9ybSA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoQ0hBVEdQVF9TRUxFQ1RPUlMuRk9STSk7XHJcbiAgICAgICAgaWYgKGZvcm0pIHtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coJ1tDb250ZW50U2NyaXB0XSBBZGRpbmcgc3VibWl0IGxpc3RlbmVyIHRvIENoYXRHUFQgZm9ybScpO1xyXG4gICAgICAgICAgICBmb3JtLmFkZEV2ZW50TGlzdGVuZXIoJ3N1Ym1pdCcsIChlKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnW0NvbnRlbnRTY3JpcHRdIENoYXRHUFQgZm9ybSBzdWJtaXQgaW50ZXJjZXB0ZWQnKTtcclxuICAgICAgICAgICAgICAgIGUucHJldmVudERlZmF1bHQoKTtcclxuICAgICAgICAgICAgICAgIGUuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgICAgICAgICByZXR1cm4gZmFsc2U7XHJcbiAgICAgICAgICAgIH0sIHRydWUpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcblxyXG5cclxuICAgIC8vIFN1Ym1pdCBidXR0b24gdmlzaWJpbGl0eSBoYW5kbGVyIC0gbm93IGludGVncmF0ZWQgd2l0aCBPYnNlcnZlck1hbmFnZXJcclxuICAgIGNvbnN0IGhhbmRsZVN1Ym1pdEJ1dHRvblZpc2liaWxpdHkgPSBhc3luYyAobXV0YXRpb25zKSA9PiB7XHJcbiAgICAgICAgbXV0YXRpb25zLmZvckVhY2gobXV0YXRpb24gPT4ge1xyXG4gICAgICAgICAgICBpZiAobXV0YXRpb24udHlwZSA9PT0gJ2NoaWxkTGlzdCcpIHtcclxuICAgICAgICAgICAgICAgIGNvbnN0IHRvZ2dsZVN3aXRjaCA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJyNtYXhtZW1vcnktdG9nZ2xlJyk7XHJcbiAgICAgICAgICAgICAgICBzeW5jTWF4TWVtb3J5VG9nZ2xlVUkodG9nZ2xlU3dpdGNoID8gdG9nZ2xlU3dpdGNoLmNoZWNrZWQgOiB0cnVlKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgfTtcclxuXHJcbiAgICAvLyBSRVBMQUNFIHlvdXIgZXhpc3RpbmcgaW5pdCgpIGZ1bmN0aW9uIHdpdGggdGhpcyBvbmVcclxuXHJcbmFzeW5jIGZ1bmN0aW9uIGluaXQoKSB7XHJcbiAgICAvLyBQcmV2ZW50IG11bHRpcGxlIGluaXRpYWxpemF0aW9uc1xyXG4gICAgaWYgKHdpbmRvdy5tZW1vcnlWYXVsdEluaXRpYWxpemVkKSB7XHJcbiAgICAgICAgY29uc29sZS5sb2coJ01heE1lbW9yeSBhbHJlYWR5IGluaXRpYWxpemVkIG9uIHRoaXMgcGFnZSwgc2tpcHBpbmcnKTtcclxuICAgICAgICByZXR1cm47XHJcbiAgICB9XHJcbiAgICBcclxuICAgIC8vIEZsYWcgdG8gdHJhY2sgaW5pdGlhbGl6YXRpb24gc3RhdGVcclxuICAgIHdpbmRvdy5tZW1vcnlWYXVsdEluaXRpYWxpemVkID0gdHJ1ZTtcclxuICAgIGNvbnNvbGUubG9nKCdJbml0aWFsaXppbmcgTWF4TWVtb3J5IGV4dGVuc2lvbicpO1xyXG5cclxuICAgIGxldCBjdXJyZW50T2JzZXJ2ZWRDaGF0SWQgPSBnZXRDaGF0SWQoKTtcclxuXHJcbiAgICAvLyBJbml0aWFsaXplIE9ic2VydmVyTWFuYWdlciB3aXRoIGNhbGxiYWNrc1xyXG4gICAgT2JzZXJ2ZXJNYW5hZ2VyLmluaXQoe1xyXG4gICAgICAgIG9uTWVzc2FnZXNBZGRlZDogKG11dGF0aW9ucykgPT4ge1xyXG4gICAgICAgICAgICBzY2hlZHVsZVN0eWxlTWVtb3JpZXNJbkNoYXQoKTtcclxuXHJcbiAgICAgICAgICAgIGZvciAoY29uc3QgbXV0YXRpb24gb2YgbXV0YXRpb25zKSB7XHJcbiAgICAgICAgICAgICAgICBjb25zdCB0YXJnZXRNZXNzYWdlQ29udGFpbmVyID0gZ2V0Q2xvc2VzdFVzZXJNZXNzYWdlQ29udGFpbmVyKG11dGF0aW9uLnRhcmdldCk7XHJcbiAgICAgICAgICAgICAgICBpZiAodGFyZ2V0TWVzc2FnZUNvbnRhaW5lcikge1xyXG4gICAgICAgICAgICAgICAgICAgIGhhbmRsZU1lc3NhZ2VTdHlsaW5nKHRhcmdldE1lc3NhZ2VDb250YWluZXIpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgICAgIGZvciAoY29uc3Qgbm9kZSBvZiBtdXRhdGlvbi5hZGRlZE5vZGVzKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKCFub2RlIHx8IG5vZGUubm9kZVR5cGUgIT09IE5vZGUuRUxFTUVOVF9OT0RFKSBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgICAgICBpZiAobm9kZS5tYXRjaGVzICYmIG5vZGUubWF0Y2hlcygnYXJ0aWNsZSwgW2RhdGEtbWVzc2FnZS1hdXRob3Itcm9sZT1cInVzZXJcIl0nKSkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBoYW5kbGVNZXNzYWdlU3R5bGluZyhub2RlKTtcclxuICAgICAgICAgICAgICAgICAgICB9IGVsc2UgaWYgKG5vZGUucXVlcnlTZWxlY3RvckFsbCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBnZXRVc2VyTWVzc2FnZUNvbnRhaW5lcnMobm9kZSkuZm9yRWFjaChoYW5kbGVNZXNzYWdlU3R5bGluZyk7XHJcblxyXG4gICAgICAgICAgICAgICAgICAgICAgICBjb25zdCBjbG9zZXN0TWVzc2FnZUNvbnRhaW5lciA9IGdldENsb3Nlc3RVc2VyTWVzc2FnZUNvbnRhaW5lcihub2RlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGNsb3Nlc3RNZXNzYWdlQ29udGFpbmVyKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBoYW5kbGVNZXNzYWdlU3R5bGluZyhjbG9zZXN0TWVzc2FnZUNvbnRhaW5lcik7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9LFxyXG4gICAgICAgIG9uSW5wdXRBcmVhQ2hhbmdlZDogKG11dGF0aW9ucykgPT4ge1xyXG4gICAgICAgICAgICAvLyBJZiB0aGUgaW5wdXQgYXJlYSBpcyByZS1yZW5kZXJlZCwgcmUtaW5qZWN0IGJ1dHRvbiBhbmQgbGlzdGVuZXJzXHJcbiAgICAgICAgICAgIGZvciAoY29uc3QgbXV0YXRpb24gb2YgbXV0YXRpb25zKSB7XHJcbiAgICAgICAgICAgICAgICBpZiAobXV0YXRpb24udHlwZSA9PT0gJ2NoaWxkTGlzdCcgJiYgKG11dGF0aW9uLmFkZGVkTm9kZXMubGVuZ3RoIHx8IG11dGF0aW9uLnJlbW92ZWROb2Rlcy5sZW5ndGgpKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgYWRkR2V0TWVtb3JpZXNCdXR0b24oKTtcclxuICAgICAgICAgICAgICAgICAgICBzZXR1cElucHV0TGlzdGVuZXJzKCk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9LFxyXG4gICAgICAgIG9uU3VibWl0QnV0dG9uQ2hhbmdlZDogaGFuZGxlU3VibWl0QnV0dG9uVmlzaWJpbGl0eSxcclxuICAgICAgICBvblVJUmVhZHk6ICgpID0+IHtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coJ1tPYnNlcnZlck1hbmFnZXJdIFVJIGlzIHJlYWR5LiBTZXR0aW5nIHVwIGxpc3RlbmVycyBhbmQgY29tcG9uZW50cy4nKTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFN0eWxlIG1lc3NhZ2VzIHRoYXQgYXJlIEFMUkVBRFkgb24gdGhlIHBhZ2Ugb24gaW5pdGlhbCBsb2FkXHJcbiAgICAgICAgICAgIHNjaGVkdWxlU3R5bGVNZW1vcmllc0luQ2hhdCgpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gU2V0dXAgdGhlIHJlc3Qgb2YgdGhlIFVJXHJcbiAgICAgICAgICAgIGFkZEdldE1lbW9yaWVzQnV0dG9uKCk7XHJcbiAgICAgICAgICAgIHNldHVwSW5wdXRMaXN0ZW5lcnMoKTtcclxuICAgICAgICAgICAgc2V0dXBFbnRlcktleVByZXZlbnRpb24oKTtcclxuICAgICAgICB9XHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBTdGFydCB0aGUgZWZmaWNpZW50IG9ic2VydmVyIHN5c3RlbS4gSXQgd2lsbCB3YWl0IGZvciB0aGUgbmVjZXNzYXJ5IGVsZW1lbnRzLlxyXG4gICAgT2JzZXJ2ZXJNYW5hZ2VyLnN0YXJ0KCk7XHJcblxyXG4gICAgLy8gVVJMIG1vbml0b3JpbmcgZm9yIENoYXRHUFQgbmF2aWdhdGlvbiBkZXRlY3Rpb25cclxuICAgIGxldCBsYXN0VXJsID0gd2luZG93LmxvY2F0aW9uLmhyZWY7XHJcbiAgICBjb25zdCBjaGVja0Zvck5hdmlnYXRpb24gPSBhc3luYyAoKSA9PiB7XHJcbiAgICAgICAgY29uc3QgY3VycmVudFVybCA9IHdpbmRvdy5sb2NhdGlvbi5ocmVmO1xyXG4gICAgICAgIGlmIChjdXJyZW50VXJsICE9PSBsYXN0VXJsKSB7XHJcbiAgICAgICAgICAgIGxhc3RVcmwgPSBjdXJyZW50VXJsO1xyXG4gICAgICAgICAgICBjb25zdCBjaGF0SWQgPSBnZXRDaGF0SWQoKTtcclxuICAgICAgICAgICAgaWYgKGNoYXRJZCAhPT0gY3VycmVudE9ic2VydmVkQ2hhdElkKSB7XHJcbiAgICAgICAgICAgICAgICBjdXJyZW50T2JzZXJ2ZWRDaGF0SWQgPSBjaGF0SWQ7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZygnTmF2aWdhdGlvbiBkZXRlY3RlZCwgcmVpbml0aWFsaXppbmcgVUkgY29tcG9uZW50cycpO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAvLyBSZW1vdmUgZXhpc3RpbmcgVUkgY29tcG9uZW50c1xyXG4gICAgICAgICAgICAgICAgY29uc3QgZXhpc3RpbmdDb250YWluZXIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWF4bWVtb3J5LWNvbnRhaW5lcicpO1xyXG4gICAgICAgICAgICAgICAgaWYgKGV4aXN0aW5nQ29udGFpbmVyKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgZXhpc3RpbmdDb250YWluZXIucmVtb3ZlKCk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIFN0b3Agb2xkIG9ic2VydmVycyBhbmQgcmVzdGFydCB3aXRoIG5ldyBwYWdlXHJcbiAgICAgICAgICAgICAgICBPYnNlcnZlck1hbmFnZXIuc3RvcCgpO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAvLyBSZXN0YXJ0IG9ic2VydmVyIHN5c3RlbSBmb3IgbmV3IHBhZ2VcclxuICAgICAgICAgICAgICAgIE9ic2VydmVyTWFuYWdlci5zdGFydCgpO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAvLyBUaGUgb25VSVJlYWR5IGNhbGxiYWNrIHdpbGwgaGFuZGxlIHJlLWluaXRpYWxpemluZyBVSSBjb21wb25lbnRzXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG5cclxuICAgIC8vIENoZWNrIGZvciBuYXZpZ2F0aW9uIGV2ZXJ5IDEwMDBtcyAobGlnaHR3ZWlnaHQgcG9sbGluZylcclxuICAgIHNldEludGVydmFsKGNoZWNrRm9yTmF2aWdhdGlvbiwgMTAwMCk7XHJcblxyXG4gICAgLy8gQWRkIG1lc3NhZ2UgbGlzdGVuZXIgZm9yIHRhYiByZWFkeSByZXF1ZXN0c1xyXG4gICAgY2hyb21lLnJ1bnRpbWUub25NZXNzYWdlLmFkZExpc3RlbmVyKChyZXF1ZXN0LCBzZW5kZXIsIHNlbmRSZXNwb25zZSkgPT4ge1xyXG4gICAgICAgIGlmIChyZXF1ZXN0LnR5cGUgPT09ICdUQUJfUkVBRFknKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdUYWIgaXMgcmVhZHknKTtcclxuICAgICAgICAgICAgc2VuZFJlc3BvbnNlKHtyZWFkeTogdHJ1ZX0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICByZXR1cm4gdHJ1ZTsgLy8gS2VlcCBtZXNzYWdlIGNoYW5uZWwgb3BlbiBmb3IgYXN5bmMgcmVzcG9uc2VcclxuICAgIH0pO1xyXG59O1xyXG5cclxuICAgIC8vIENsZWFudXAgZnVuY3Rpb24gdG8gc3RvcCBhbGwgb2JzZXJ2ZXJzIHdoZW4gcGFnZSB1bmxvYWRzXHJcbiAgICBjb25zdCBjbGVhbnVwID0gKCkgPT4ge1xyXG4gICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gQ2xlYW5pbmcgdXAgb2JzZXJ2ZXJzIGJlZm9yZSBwYWdlIHVubG9hZCcpO1xyXG4gICAgICAgIE9ic2VydmVyTWFuYWdlci5zdG9wKCk7XHJcbiAgICAgICAgd2luZG93Lm1lbW9yeVZhdWx0T2JzZXJ2ZXJzSW5pdGlhbGl6ZWQgPSBmYWxzZTtcclxuICAgICAgICB3aW5kb3cubWVtb3J5VmF1bHRJbml0aWFsaXplZCA9IGZhbHNlO1xyXG4gICAgfTtcclxuICAgIFxyXG4gICAgLy8gQWRkIGNsZWFudXAgbGlzdGVuZXJzXHJcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcignYmVmb3JldW5sb2FkJywgY2xlYW51cCk7XHJcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcigncGFnZWhpZGUnLCBjbGVhbnVwKTtcclxuICAgIFxyXG4gICAgLy8gRW5zdXJlIHRoZSBjb250ZW50IHNjcmlwdCBpcyBpbml0aWFsaXplZCBhcyBzb29uIGFzIHBvc3NpYmxlXHJcbiAgICBpbml0KCk7XHJcbiAgICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgaW5pdCk7XHJcbiAgICBcclxuICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gTWF4TWVtb3J5IGluaXRpYWxpemVkIHdpdGggZWZmaWNpZW50IE9ic2VydmVyTWFuYWdlciAtIG5vIG1vcmUgZXhwZW5zaXZlIGRvY3VtZW50LmJvZHkgb2JzZXJ2ZXJzIScpO1xyXG4vLyBMaXN0ZW4gZm9yIG1lc3NhZ2VzIGZyb20gYmFja2dyb3VuZCBzY3JpcHRcclxuICAgIGNocm9tZS5ydW50aW1lLm9uTWVzc2FnZS5hZGRMaXN0ZW5lcigocmVxdWVzdCwgc2VuZGVyLCBzZW5kUmVzcG9uc2UpID0+IHtcclxuICAgICAgICBpZiAocmVxdWVzdC50eXBlID09PSAnRElTUExBWV9FWFRSQUNURURfTUVNT1JJRVMnKSB7XHJcbiAgICAgICAgICAgIGRpc3BsYXlFeHRyYWN0ZWRNZW1vcmllcyhyZXF1ZXN0Lm1lbW9yaWVzLCByZXF1ZXN0LnNhdmVkVG9EYXRhYmFzZSwgcmVxdWVzdC5saW1pdFR5cGUpO1xyXG4gICAgICAgICAgICBzZW5kUmVzcG9uc2Uoe3N0YXR1czogJ3N1Y2Nlc3MnfSk7XHJcbiAgICAgICAgfSBlbHNlIGlmIChyZXF1ZXN0LnR5cGUgPT09ICdESVNQTEFZX01FTU9SWV9TVUdHRVNUSU9OUycpIHtcclxuICAgICAgICAgICAgZGlzcGxheU1lbW9yeVN1Z2dlc3Rpb25zKFxyXG4gICAgICAgICAgICAgICAgcmVxdWVzdC5zdWdnZXN0aW9ucyxcclxuICAgICAgICAgICAgICAgIHJlcXVlc3QuZGV0ZWN0ZWRNb2RlLFxyXG4gICAgICAgICAgICAgICAgcmVxdWVzdC5leHRyYWN0ZWRXaGlsZUF0TGltaXQgfHwgZmFsc2UsXHJcbiAgICAgICAgICAgICAgICByZXF1ZXN0LmxpbWl0VHlwZSB8fCBudWxsXHJcbiAgICAgICAgICAgICk7XHJcbiAgICAgICAgICAgIHNlbmRSZXNwb25zZSh7c3RhdHVzOiAnc3VjY2Vzcyd9KTtcclxuICAgICAgICB9IGVsc2UgaWYgKHJlcXVlc3QudHlwZSA9PT0gJ0RJU1BMQVlfTUVNT1JZX0xJTUlUX1dBUk5JTkcnKSB7XHJcbiAgICAgICAgICAgIGRpc3BsYXlNZW1vcnlMaW1pdFdhcm5pbmcocmVxdWVzdC5saW1pdFR5cGUsIHJlcXVlc3QuY3VycmVudCwgcmVxdWVzdC5saW1pdCk7XHJcbiAgICAgICAgICAgIHNlbmRSZXNwb25zZSh7c3RhdHVzOiAnc3VjY2Vzcyd9KTtcclxuICAgICAgICB9IGVsc2UgaWYgKHJlcXVlc3QudHlwZSA9PT0gJ0dFVF9DT05WRVJTQVRJT05fSElTVE9SWScpIHtcclxuICAgICAgICAgICAgY29uc3QgaGlzdG9yeSA9IHNjcmFwZUNvbnZlcnNhdGlvbkhpc3RvcnkocmVxdWVzdC5jb3VudCk7XHJcbiAgICAgICAgICAgIHNlbmRSZXNwb25zZSh7IHN0YXR1czogJ3N1Y2Nlc3MnLCBoaXN0b3J5OiBoaXN0b3J5IH0pO1xyXG4gICAgICAgIH0gZWxzZSBpZiAocmVxdWVzdC50eXBlID09PSAnTUFYTUVNT1JZX0VOQUJMRURfU1RBVEVfQ0hBTkdFRCcpIHtcclxuICAgICAgICAgICAgc3luY01heE1lbW9yeVRvZ2dsZVVJKHJlcXVlc3QuZW5hYmxlZCk7XHJcbiAgICAgICAgICAgIHNlbmRSZXNwb25zZSh7IHN0YXR1czogJ3N1Y2Nlc3MnIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICAvLyBLZWVwIHRoZSBtZXNzYWdlIGNoYW5uZWwgb3BlbiBmb3IgYXN5bmMgcmVzcG9uc2VzXHJcbiAgICAgICAgcmV0dXJuIHRydWU7XHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBGdW5jdGlvbiB0byBkaXNwbGF5IG1lbW9yeSBsaW1pdCB3YXJuaW5nXHJcbiAgICBjb25zdCBkaXNwbGF5TWVtb3J5TGltaXRXYXJuaW5nID0gKGxpbWl0VHlwZSwgY3VycmVudCwgbGltaXQpID0+IHtcclxuICAgICAgICAvLyBGaW5kIHRoZSBsYXRlc3QgdXNlciBtZXNzYWdlIGNvbnRhaW5lclxyXG4gICAgICAgIGNvbnN0IG1lc3NhZ2VzID0gZ2V0VXNlck1lc3NhZ2VDb250YWluZXJzKCk7XHJcbiAgICAgICAgY29uc3QgbGF0ZXN0VXNlck1lc3NhZ2UgPSBBcnJheS5mcm9tKG1lc3NhZ2VzKS5yZXZlcnNlKCkuZmluZChtc2cgPT4ge1xyXG4gICAgICAgICAgICBjb25zdCBtZXNzYWdlRGl2ID0gZ2V0TWVzc2FnZUNvbnRlbnRFbGVtZW50KG1zZyk7XHJcbiAgICAgICAgICAgIHJldHVybiBtZXNzYWdlRGl2ICYmICFtZXNzYWdlRGl2LnRleHRDb250ZW50LmluY2x1ZGVzKCdbUkVMRVZBTlRfUEFTVF9NRU1PUklFU19TVEFSVF0nKTtcclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgaWYgKCFsYXRlc3RVc2VyTWVzc2FnZSkge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnQ291bGQgbm90IGZpbmQgbGF0ZXN0IHVzZXIgbWVzc2FnZSB0byBkaXNwbGF5IHdhcm5pbmcnKTtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgLy8gQ2hlY2sgaWYgd2FybmluZyBhbHJlYWR5IGV4aXN0c1xyXG4gICAgICAgIGlmIChsYXRlc3RVc2VyTWVzc2FnZS5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LWxpbWl0LXdhcm5pbmcnKSkge1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICAvLyBDcmVhdGUgd2FybmluZyBlbGVtZW50IHVzaW5nIGJsdWVwcmludFxyXG4gICAgICAgIGNvbnN0IHdhcm5pbmdEaXYgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcclxuICAgICAgICB3YXJuaW5nRGl2LmlubmVySFRNTCA9IHVpQmx1ZXByaW50cy5nZXRNZW1vcnlMaW1pdFdhcm5pbmcobGltaXRUeXBlLCBjdXJyZW50LCBsaW1pdCk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gR2V0IHRoZSBhY3R1YWwgd2FybmluZyBlbGVtZW50IChmaXJzdCBjaGlsZClcclxuICAgICAgICBjb25zdCB3YXJuaW5nRWxlbWVudCA9IHdhcm5pbmdEaXYuZmlyc3RFbGVtZW50Q2hpbGQ7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gQWRkIGV2ZW50IGxpc3RlbmVyIHRvIHRoZSBzaWduIGluIGJ1dHRvblxyXG4gICAgICAgIGNvbnN0IHNpZ25JbkJ1dHRvbiA9IHdhcm5pbmdFbGVtZW50LnF1ZXJ5U2VsZWN0b3IoJy5tZW1vcnktd2FybmluZy1idXR0b24nKTtcclxuICAgICAgICBzaWduSW5CdXR0b24uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4ge1xyXG4gICAgICAgICAgICBlLnN0b3BQcm9wYWdhdGlvbigpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gVHJhY2sgcG9wdXAgb3BlbmVkIGZyb20gbWVtb3J5IGxpbWl0IHdhcm5pbmdcclxuICAgICAgICAgICAgYmFja2dyb3VuZEFQSS50cmFja1BvcHVwT3BlbmVkKCdtZW1vcnlfbGltaXRfd2FybmluZycpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gT3BlbiBleHRlbnNpb24gcG9wdXAgaW4gYSBuZXcgdGFiXHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkub3BlblBvcHVwSW5UYWIoKTtcclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgLy8gSW5zZXJ0IHdhcm5pbmcgYWZ0ZXIgdGhlIHVzZXIgbWVzc2FnZVxyXG4gICAgICAgIGxhdGVzdFVzZXJNZXNzYWdlLmFwcGVuZENoaWxkKHdhcm5pbmdFbGVtZW50KTtcclxuICAgIH07XHJcblxyXG4gICAgLy8gRnVuY3Rpb24gdG8gZGlzcGxheSBleHRyYWN0ZWQgbWVtb3JpZXMgbmV4dCB0byB0aGUgbGF0ZXN0IHVzZXIgbWVzc2FnZVxyXG4gICAgY29uc3QgZGlzcGxheUV4dHJhY3RlZE1lbW9yaWVzID0gKG1lbW9yaWVzLCBzYXZlZFRvRGF0YWJhc2UgPSB0cnVlLCBsaW1pdFR5cGUgPSBudWxsKSA9PiB7XHJcbiAgICAgICAgY29uc29sZS5sb2coJ0Rpc3BsYXlpbmcgZXh0cmFjdGVkIG1lbW9yaWVzOicsIG1lbW9yaWVzLCAnc2F2ZWRUb0RhdGFiYXNlOicsIHNhdmVkVG9EYXRhYmFzZSwgJ2xpbWl0VHlwZTonLCBsaW1pdFR5cGUpO1xyXG5cclxuICAgICAgICAvLyBGaW5kIHRoZSBtb3N0IHJlY2VudCB1c2VyIG1lc3NhZ2VcclxuICAgICAgICBjb25zdCB1c2VyTWVzc2FnZXMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS1tZXNzYWdlLWF1dGhvci1yb2xlPVwidXNlclwiXScpO1xyXG4gICAgICAgIGlmICh1c2VyTWVzc2FnZXMubGVuZ3RoID09PSAwKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdObyB1c2VyIG1lc3NhZ2VzIGZvdW5kIHRvIGF0dGFjaCBtZW1vcmllcyB0bycpO1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIGNvbnN0IGxhdGVzdFVzZXJNZXNzYWdlID0gdXNlck1lc3NhZ2VzW3VzZXJNZXNzYWdlcy5sZW5ndGggLSAxXTtcclxuICAgICAgICBjb25zdCBtZXNzYWdlSWQgPSBsYXRlc3RVc2VyTWVzc2FnZS5nZXRBdHRyaWJ1dGUoJ2RhdGEtbWVzc2FnZS1pZCcpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIENoZWNrIGlmIHdlIGFscmVhZHkgaGF2ZSBhIG1lbW9yeSBub3RpZmljYXRpb24gZm9yIHRoaXMgbWVzc2FnZVxyXG4gICAgICAgIGNvbnN0IGV4aXN0aW5nTm90aWZpY2F0aW9uID0gbGF0ZXN0VXNlck1lc3NhZ2UucXVlcnlTZWxlY3RvcignLmV4dHJhY3RlZC1tZW1vcnktbm90aWZpY2F0aW9uJyk7XHJcbiAgICAgICAgaWYgKGV4aXN0aW5nTm90aWZpY2F0aW9uKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdNZW1vcnkgbm90aWZpY2F0aW9uIGFscmVhZHkgZXhpc3RzIGZvciB0aGlzIG1lc3NhZ2UnKTtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICAvLyBDcmVhdGUgbWVtb3J5IG5vdGlmaWNhdGlvbiB1c2luZyBibHVlcHJpbnRcclxuICAgICAgICBjb25zdCBub3RpZmljYXRpb25IVE1MID0gdWlCbHVlcHJpbnRzLmdldEV4dHJhY3RlZE1lbW9yeU5vdGlmaWNhdGlvbihtZW1vcmllcyk7XHJcbiAgICAgICAgbGF0ZXN0VXNlck1lc3NhZ2UuaW5zZXJ0QWRqYWNlbnRIVE1MKCdiZWZvcmVlbmQnLCBub3RpZmljYXRpb25IVE1MKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBBZGQgZXZlbnQgbGlzdGVuZXJzIHRvIHRoZSBjcmVhdGVkIGVsZW1lbnRzXHJcbiAgICAgICAgY29uc3QgbWVtb3J5Tm90aWZpY2F0aW9uID0gbGF0ZXN0VXNlck1lc3NhZ2UucXVlcnlTZWxlY3RvcignLmV4dHJhY3RlZC1tZW1vcnktbm90aWZpY2F0aW9uJyk7XHJcbiAgICAgICAgbWVtb3J5Tm90aWZpY2F0aW9uLnNldEF0dHJpYnV0ZSgnZGF0YS1tZXNzYWdlLWlkJywgbWVzc2FnZUlkKTtcclxuICAgICAgICBcclxuICAgICAgICBjb25zdCBwcmVmaXhUZXh0ID0gbWVtb3J5Tm90aWZpY2F0aW9uLnF1ZXJ5U2VsZWN0b3IoJy5tZW1vcnktcHJlZml4LXRleHQnKTtcclxuICAgICAgICBcclxuICAgICAgICBjb25zdCBtZW1vcnlUZXh0ID0gbWVtb3J5Tm90aWZpY2F0aW9uLnF1ZXJ5U2VsZWN0b3IoJy5leHRyYWN0ZWQtbWVtb3J5LXRleHQnKTtcclxuICAgICAgICBpZiAobWVtb3J5VGV4dCkge1xyXG4gICAgICAgICAgICBjb25zdCBub3JtYWxpemVkTWVtb3JpZXMgPSBtZW1vcmllcy5tYXAoKG1lbW9yeSkgPT4gdHlwZW9mIG1lbW9yeSA9PT0gJ3N0cmluZycgPyBtZW1vcnkgOiAobWVtb3J5Lm1lbW9yeSB8fCBtZW1vcnkudGV4dCB8fCAnJykpLmZpbHRlcihCb29sZWFuKTtcclxuICAgICAgICAgICAgbWVtb3J5VGV4dC50ZXh0Q29udGVudCA9IG5vcm1hbGl6ZWRNZW1vcmllcy5qb2luKCcg4oCiICcpO1xyXG4gICAgICAgIH1cclxuICAgICAgICAvLyBBZGQgY2xpY2sgaGFuZGxlciB0byBvcGVuIHBvcHVwXHJcbiAgICAgICAgaWYgKHNhdmVkVG9EYXRhYmFzZSkge1xyXG4gICAgICAgICAgICAvLyBBZGQgY2xpY2sgaGFuZGxlciB0byBvcGVuIHBvcHVwXHJcbiAgICAgICAgICAgIHByZWZpeFRleHQuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgZS5zdG9wUHJvcGFnYXRpb24oKTtcclxuICAgICAgICAgICAgICAgIGJhY2tncm91bmRBUEkub3BlblBvcHVwKCk7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICByZXR1cm47XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBwcmVmaXhUZXh0LnRleHRDb250ZW50ID0gJ21lbW9yeSBleHRyYWN0ZWQ6JztcclxuICAgICAgICBwcmVmaXhUZXh0LnN0eWxlLmN1cnNvciA9ICdkZWZhdWx0JztcclxuICAgICAgICBwcmVmaXhUZXh0LmNsYXNzTGlzdC5yZW1vdmUoJ21lbW9yeS1wcmVmaXgtdGV4dCcpO1xyXG5cclxuICAgICAgICBjb25zdCB3YXJuaW5nU2VjdGlvbiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xyXG4gICAgICAgIGNvbnN0IGlzR3Vlc3RMaW1pdCA9IGxpbWl0VHlwZSA9PT0gJ2d1ZXN0JztcclxuICAgICAgICB3YXJuaW5nU2VjdGlvbi5jbGFzc05hbWUgPSBgbWVtb3J5LWxpbWl0LXdhcm5pbmcgJHtpc0d1ZXN0TGltaXQgPyAnbWVtb3J5LWxpbWl0LXdhcm5pbmctLWd1ZXN0JyA6ICdtZW1vcnktbGltaXQtd2FybmluZy0tbG9nZ2VkLWluJ31gO1xyXG5cclxuICAgICAgICBjb25zdCB3YXJuaW5nSWNvbiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xyXG4gICAgICAgIHdhcm5pbmdJY29uLmNsYXNzTmFtZSA9ICdtZW1vcnktd2FybmluZy1pY29uJztcclxuICAgICAgICB3YXJuaW5nSWNvbi5pbm5lckhUTUwgPSBgPHN2ZyB3aWR0aD1cIjE0XCIgaGVpZ2h0PVwiMTRcIiB2aWV3Qm94PVwiMCAwIDI0IDI0XCIgZmlsbD1cIm5vbmVcIiBzdHJva2U9XCJjdXJyZW50Q29sb3JcIiBzdHJva2Utd2lkdGg9XCIyXCI+XHJcbiAgICAgICAgICAgIDxwYXRoIGQ9XCJNMTAuMjkgMy44NkwxLjgyIDE4YTIgMiAwIDAwMS43MSAzaDE2Ljk0YTIgMiAwIDAwMS43MS0zTDEzLjcxIDMuODZhMiAyIDAgMDAtMy40MiAwelwiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgPGxpbmUgeDE9XCIxMlwiIHkxPVwiOVwiIHgyPVwiMTJcIiB5Mj1cIjEzXCIgc3Ryb2tlLWxpbmVjYXA9XCJyb3VuZFwiIHN0cm9rZS1saW5lam9pbj1cInJvdW5kXCIvPlxyXG4gICAgICAgICAgICA8bGluZSB4MT1cIjEyXCIgeTE9XCIxN1wiIHgyPVwiMTIuMDFcIiB5Mj1cIjE3XCIgc3Ryb2tlLWxpbmVjYXA9XCJyb3VuZFwiIHN0cm9rZS1saW5lam9pbj1cInJvdW5kXCIvPlxyXG4gICAgICAgIDwvc3ZnPmA7XHJcblxyXG4gICAgICAgIGNvbnN0IHdhcm5pbmdUZXh0ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XHJcbiAgICAgICAgd2FybmluZ1RleHQuY2xhc3NOYW1lID0gJ21lbW9yeS13YXJuaW5nLXRleHQnO1xyXG4gICAgICAgIHdhcm5pbmdUZXh0LnRleHRDb250ZW50ID0gaXNHdWVzdExpbWl0XHJcbiAgICAgICAgICAgID8gXCJXZSBleHRyYWN0ZWQgdGhpcyBtZW1vcnksIGJ1dCBjb3VsZG4ndCBzYXZlIGl0IGJlY2F1c2UgeW91J3ZlIHJlYWNoZWQgdGhlIGd1ZXN0IGxpbWl0LiBTaWduIGluIHRvIHVubG9jayAxMDAgZnJlZSBtZW1vcmllcy5cIlxyXG4gICAgICAgICAgICA6IFwiV2UgZXh0cmFjdGVkIHRoaXMgbWVtb3J5LCBidXQgY291bGRuJ3Qgc2F2ZSBpdCBiZWNhdXNlIHlvdSd2ZSByZWFjaGVkIHlvdXIgZnJlZSBsaW1pdC4gVXBncmFkZSB0byBrZWVwIHNhdmluZyBhdXRvbWF0aWNhbGx5LlwiO1xyXG5cclxuICAgICAgICBjb25zdCB3YXJuaW5nQnV0dG9uID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7XHJcbiAgICAgICAgd2FybmluZ0J1dHRvbi5jbGFzc05hbWUgPSBgbWVtb3J5LXdhcm5pbmctYnV0dG9uICR7aXNHdWVzdExpbWl0ID8gJ21lbW9yeS13YXJuaW5nLWJ1dHRvbi0tZ3Vlc3QnIDogJ21lbW9yeS13YXJuaW5nLWJ1dHRvbi0tbG9nZ2VkLWluJ31gO1xyXG4gICAgICAgIHdhcm5pbmdCdXR0b24udGV4dENvbnRlbnQgPSBpc0d1ZXN0TGltaXQgPyAnU2lnbiBpbicgOiAnVXBncmFkZSc7XHJcbiAgICAgICAgd2FybmluZ0J1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIChlKSA9PiB7XHJcbiAgICAgICAgICAgIGUuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tQb3B1cE9wZW5lZCgnbWVtb3J5X2xpbWl0X3dhcm5pbmcnKTtcclxuICAgICAgICAgICAgYmFja2dyb3VuZEFQSS5vcGVuUG9wdXBJblRhYigpO1xyXG4gICAgICAgIH0pO1xyXG5cclxuICAgICAgICB3YXJuaW5nU2VjdGlvbi5hcHBlbmRDaGlsZCh3YXJuaW5nSWNvbik7XHJcbiAgICAgICAgd2FybmluZ1NlY3Rpb24uYXBwZW5kQ2hpbGQod2FybmluZ1RleHQpO1xyXG4gICAgICAgIHdhcm5pbmdTZWN0aW9uLmFwcGVuZENoaWxkKHdhcm5pbmdCdXR0b24pO1xyXG4gICAgICAgIG1lbW9yeU5vdGlmaWNhdGlvbi5hcHBlbmRDaGlsZCh3YXJuaW5nU2VjdGlvbik7XHJcbiAgICB9O1xyXG5cclxuXHJcbiAgICBjb25zdCB1cGRhdGVBdXRvU2F2ZWRDb250YWluZXJTdGF0ZSA9IChzdWdnZXN0aW9uc0NvbnRhaW5lciwgc2F2ZWRDb3VudCwgZmFpbGVkQ291bnQgPSAwKSA9PiB7XHJcbiAgICAgICAgY29uc3QgdGl0bGVFbGVtZW50ID0gc3VnZ2VzdGlvbnNDb250YWluZXIucXVlcnlTZWxlY3RvcignLm1lbW9yeS1zdWdnZXN0aW9ucy10aXRsZScpO1xyXG4gICAgICAgIGlmICh0aXRsZUVsZW1lbnQpIHtcclxuICAgICAgICAgICAgbGV0IHRpdGxlID0gYFNhdmVkICR7c2F2ZWRDb3VudH0gJHtzYXZlZENvdW50ID09PSAxID8gJ21lbW9yeScgOiAnbWVtb3JpZXMnfWA7XHJcbiAgICAgICAgICAgIGlmIChmYWlsZWRDb3VudCA+IDApIHtcclxuICAgICAgICAgICAgICAgIHRpdGxlICs9IGAgKCR7ZmFpbGVkQ291bnR9IGZhaWxlZClgO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHRpdGxlRWxlbWVudC50ZXh0Q29udGVudCA9IHRpdGxlO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgY29uc3QgYnVsa1VuZG9CdXR0b24gPSBzdWdnZXN0aW9uc0NvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcuZGlzY2FyZC1hbGwtYnV0dG9uJyk7XHJcbiAgICAgICAgaWYgKGJ1bGtVbmRvQnV0dG9uKSB7XHJcbiAgICAgICAgICAgIGJ1bGtVbmRvQnV0dG9uLnN0eWxlLmRpc3BsYXkgPSBzYXZlZENvdW50ID4gMSA/ICdibG9jaycgOiAnbm9uZSc7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCByZW1vdmVTdWdnZXN0aW9uQ29udGFpbmVySWZFbXB0eSA9IChzdWdnZXN0aW9uc0NvbnRhaW5lcikgPT4ge1xyXG4gICAgICAgIGlmICghc3VnZ2VzdGlvbnNDb250YWluZXIpIHJldHVybjtcclxuXHJcbiAgICAgICAgY29uc3QgcmVtYWluaW5nSXRlbXMgPSBzdWdnZXN0aW9uc0NvbnRhaW5lci5xdWVyeVNlbGVjdG9yQWxsKCcubWVtb3J5LXN1Z2dlc3Rpb24taXRlbScpO1xyXG4gICAgICAgIGlmIChyZW1haW5pbmdJdGVtcy5sZW5ndGggPT09IDApIHtcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbnNDb250YWluZXIucmVtb3ZlKCk7XHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgdXBkYXRlQXV0b1NhdmVkQ29udGFpbmVyU3RhdGUoc3VnZ2VzdGlvbnNDb250YWluZXIsIHJlbWFpbmluZ0l0ZW1zLmxlbmd0aCk7XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuXHJcbiAgICBjb25zdCBoYW5kbGVVbmRvTWVtb3J5ID0gYXN5bmMgKHNhdmVkTWVtb3J5LCBzdWdnZXN0aW9uSXRlbSwgc3VnZ2VzdGlvbnNDb250YWluZXIpID0+IHtcclxuICAgICAgICB0cnkge1xyXG4gICAgICAgICAgICBjb25zdCB1bmRvQnV0dG9uID0gc3VnZ2VzdGlvbkl0ZW0ucXVlcnlTZWxlY3RvcignLnVuZG8tYnV0dG9uJyk7XHJcbiAgICAgICAgICAgIGlmICh1bmRvQnV0dG9uKSB7XHJcbiAgICAgICAgICAgICAgICB1bmRvQnV0dG9uLmRpc2FibGVkID0gdHJ1ZTtcclxuICAgICAgICAgICAgICAgIHVuZG9CdXR0b24udGV4dENvbnRlbnQgPSAnVW5kb2luZy4uLic7XHJcbiAgICAgICAgICAgIH1cclxuXHJcbiAgICAgICAgICAgIHN1Z2dlc3Rpb25JdGVtLnN0eWxlLm9wYWNpdHkgPSAnMC43JztcclxuXHJcbiAgICAgICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgYmFja2dyb3VuZEFQSS5kZWxldGVNZW1vcnkoc2F2ZWRNZW1vcnkuaWQsIHNhdmVkTWVtb3J5LnRleHQpO1xyXG4gICAgICAgICAgICBpZiAocmVzcG9uc2Uuc3RhdHVzICE9PSAnc3VjY2VzcycpIHtcclxuICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihyZXNwb25zZS5tZXNzYWdlIHx8ICdGYWlsZWQgdG8gdW5kbyBtZW1vcnknKTtcclxuICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUudHJhbnNpdGlvbiA9ICdhbGwgMC4ycyBlYXNlJztcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUub3BhY2l0eSA9ICcwJztcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUudHJhbnNmb3JtID0gJ3RyYW5zbGF0ZVkoLTZweCknO1xyXG5cclxuICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5yZW1vdmUoKTtcclxuICAgICAgICAgICAgICAgIHJlbW92ZVN1Z2dlc3Rpb25Db250YWluZXJJZkVtcHR5KHN1Z2dlc3Rpb25zQ29udGFpbmVyKTtcclxuICAgICAgICAgICAgfSwgMjAwKTtcclxuXHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdBdXRvLXNhdmVkIG1lbW9yeSB1bmRvbmU6Jywgc2F2ZWRNZW1vcnkudGV4dCk7XHJcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgdW5kb2luZyBhdXRvLXNhdmVkIG1lbW9yeTonLCBlcnJvcik7XHJcblxyXG4gICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5jbGFzc0xpc3QuYWRkKCdtZW1vcnktc3VnZ2VzdGlvbi1pdGVtLS1mYWlsZWQnKTtcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUub3BhY2l0eSA9ICcxJztcclxuXHJcbiAgICAgICAgICAgIGNvbnN0IHVuZG9CdXR0b24gPSBzdWdnZXN0aW9uSXRlbS5xdWVyeVNlbGVjdG9yKCcudW5kby1idXR0b24nKTtcclxuICAgICAgICAgICAgaWYgKHVuZG9CdXR0b24pIHtcclxuICAgICAgICAgICAgICAgIHVuZG9CdXR0b24uZGlzYWJsZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgICAgIHVuZG9CdXR0b24udGV4dENvbnRlbnQgPSAnVW5kbyc7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IGhhbmRsZVVuZG9BbGwgPSBhc3luYyAoc3VnZ2VzdGlvbnNDb250YWluZXIpID0+IHtcclxuICAgICAgICBjb25zdCBzdWdnZXN0aW9uSXRlbXMgPSBBcnJheS5mcm9tKHN1Z2dlc3Rpb25zQ29udGFpbmVyLnF1ZXJ5U2VsZWN0b3JBbGwoJy5tZW1vcnktc3VnZ2VzdGlvbi1pdGVtJykpO1xyXG4gICAgICAgIGZvciAoY29uc3Qgc3VnZ2VzdGlvbkl0ZW0gb2Ygc3VnZ2VzdGlvbkl0ZW1zKSB7XHJcbiAgICAgICAgICAgIGNvbnN0IHNhdmVkTWVtb3J5ID0ge1xyXG4gICAgICAgICAgICAgICAgaWQ6IHN1Z2dlc3Rpb25JdGVtLmdldEF0dHJpYnV0ZSgnZGF0YS1tZW1vcnktaWQnKSxcclxuICAgICAgICAgICAgICAgIHRleHQ6IHN1Z2dlc3Rpb25JdGVtLnF1ZXJ5U2VsZWN0b3IoJy5zdWdnZXN0aW9uLXRleHQnKT8udGV4dENvbnRlbnQ/LnJlcGxhY2UoL1xccypcXChzYXZlZFxcKVxccyokL2ksICcnKS50cmltKCkgfHwgJydcclxuICAgICAgICAgICAgfTtcclxuXHJcbiAgICAgICAgICAgIGlmICghc2F2ZWRNZW1vcnkuaWQpIHtcclxuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xyXG4gICAgICAgICAgICB9XHJcblxyXG4gICAgICAgICAgICBhd2FpdCBoYW5kbGVVbmRvTWVtb3J5KHNhdmVkTWVtb3J5LCBzdWdnZXN0aW9uSXRlbSwgc3VnZ2VzdGlvbnNDb250YWluZXIpO1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcblxyXG4gICAgLy8gRnVuY3Rpb24gdG8gZGlzcGxheSBhdXRvLXNhdmVkIG1lbW9yaWVzIHdpdGggdW5kbyBidXR0b25zXHJcbiAgICBjb25zdCBkaXNwbGF5TWVtb3J5U3VnZ2VzdGlvbnMgPSBhc3luYyAoc3VnZ2VzdGlvbnMsIGRldGVjdGVkTW9kZSA9IG51bGwsIGV4dHJhY3RlZFdoaWxlQXRMaW1pdCA9IGZhbHNlLCBsaW1pdFR5cGUgPSBudWxsKSA9PiB7XHJcbiAgICAgICAgY29uc29sZS5sb2coJ0Rpc3BsYXlpbmcgbWVtb3J5IHN1Z2dlc3Rpb25zOicsIHN1Z2dlc3Rpb25zLCAnd2l0aCBkZXRlY3RlZCBtb2RlOicsIGRldGVjdGVkTW9kZSwgJ2V4dHJhY3RlZFdoaWxlQXRMaW1pdDonLCBleHRyYWN0ZWRXaGlsZUF0TGltaXQsICdsaW1pdFR5cGU6JywgbGltaXRUeXBlKTtcclxuXHJcbiAgICAgICAgaWYgKCFzdWdnZXN0aW9ucyB8fCBzdWdnZXN0aW9ucy5sZW5ndGggPT09IDApIHtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coJ05vIG1lbW9yeSBzdWdnZXN0aW9ucyB0byBkaXNwbGF5Jyk7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcblxyXG4gICAgICAgIGlmIChleHRyYWN0ZWRXaGlsZUF0TGltaXQpIHtcclxuICAgICAgICAgICAgZGlzcGxheUV4dHJhY3RlZE1lbW9yaWVzKHN1Z2dlc3Rpb25zLCBmYWxzZSwgbGltaXRUeXBlKTtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgLy8gRmluZCB0aGUgbW9zdCByZWNlbnQgdXNlciBtZXNzYWdlXHJcbiAgICAgICAgY29uc3QgdXNlck1lc3NhZ2VzID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnW2RhdGEtbWVzc2FnZS1hdXRob3Itcm9sZT1cInVzZXJcIl0nKTtcclxuICAgICAgICBpZiAodXNlck1lc3NhZ2VzLmxlbmd0aCA9PT0gMCkge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnTm8gdXNlciBtZXNzYWdlcyBmb3VuZCB0byBhdHRhY2ggc3VnZ2VzdGlvbnMgdG8nKTtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBjb25zdCBsYXRlc3RVc2VyTWVzc2FnZSA9IHVzZXJNZXNzYWdlc1t1c2VyTWVzc2FnZXMubGVuZ3RoIC0gMV07XHJcbiAgICAgICAgY29uc3QgbWVzc2FnZUlkID0gbGF0ZXN0VXNlck1lc3NhZ2UuZ2V0QXR0cmlidXRlKCdkYXRhLW1lc3NhZ2UtaWQnKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBDaGVjayBpZiB3ZSBhbHJlYWR5IGhhdmUgc3VnZ2VzdGlvbnMgZm9yIHRoaXMgbWVzc2FnZVxyXG4gICAgICAgIGNvbnN0IGV4aXN0aW5nU3VnZ2VzdGlvbnMgPSBsYXRlc3RVc2VyTWVzc2FnZS5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LXN1Z2dlc3Rpb25zLWNvbnRhaW5lcicpO1xyXG4gICAgICAgIGlmIChleGlzdGluZ1N1Z2dlc3Rpb25zKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdNZW1vcnkgc3VnZ2VzdGlvbnMgYWxyZWFkeSBleGlzdCBmb3IgdGhpcyBtZXNzYWdlJyk7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gQ3JlYXRlIHRoZSBjb250YWluZXIgaW1tZWRpYXRlbHkgc28gdGhlIHVzZXIgc2VlcyB0aGF0IGF1dG8tc2F2ZSBpcyBoYXBwZW5pbmcuXHJcbiAgICAgICAgY29uc3QgY29udGFpbmVySFRNTCA9IHVpQmx1ZXByaW50cy5nZXRNZW1vcnlTdWdnZXN0aW9uc0NvbnRhaW5lcihcclxuICAgICAgICAgICAgbWVzc2FnZUlkLFxyXG4gICAgICAgICAgICBzdWdnZXN0aW9ucy5sZW5ndGgsXHJcbiAgICAgICAgICAgIGRldGVjdGVkTW9kZSxcclxuICAgICAgICAgICAgYFNhdmluZyAke3N1Z2dlc3Rpb25zLmxlbmd0aH0gJHtzdWdnZXN0aW9ucy5sZW5ndGggPT09IDEgPyAnbWVtb3J5JyA6ICdtZW1vcmllcyd9Li4uYCxcclxuICAgICAgICAgICAgJ1VuZG8gYWxsJ1xyXG4gICAgICAgICk7XHJcbiAgICAgICAgbGF0ZXN0VXNlck1lc3NhZ2UuaW5zZXJ0QWRqYWNlbnRIVE1MKCdiZWZvcmVlbmQnLCBjb250YWluZXJIVE1MKTtcclxuICAgICAgICBcclxuICAgICAgICBjb25zdCBzdWdnZXN0aW9uc0NvbnRhaW5lciA9IGxhdGVzdFVzZXJNZXNzYWdlLnF1ZXJ5U2VsZWN0b3IoJy5tZW1vcnktc3VnZ2VzdGlvbnMtY29udGFpbmVyW2RhdGEtbWVzc2FnZS1pZD1cIicgKyBtZXNzYWdlSWQgKyAnXCJdJyk7XHJcbiAgICAgICAgY29uc3Qgc3VnZ2VzdGlvbnNMaXN0ID0gc3VnZ2VzdGlvbnNDb250YWluZXIucXVlcnlTZWxlY3RvcignLm1lbW9yeS1zdWdnZXN0aW9ucy1saXN0Jyk7XHJcblxyXG4gICAgICAgIGNvbnN0IG1lbW9yaWVzVG9TYXZlID0gc3VnZ2VzdGlvbnMubWFwKChzdWdnZXN0aW9uKSA9PiAoe1xyXG4gICAgICAgICAgICB0ZXh0OiB0eXBlb2Ygc3VnZ2VzdGlvbiA9PT0gJ3N0cmluZycgPyBzdWdnZXN0aW9uIDogKHN1Z2dlc3Rpb24ubWVtb3J5IHx8ICcnKSxcclxuICAgICAgICAgICAgdGFnOiB0eXBlb2Ygc3VnZ2VzdGlvbiA9PT0gJ29iamVjdCcgPyAoc3VnZ2VzdGlvbi50YWcgfHwgbnVsbCkgOiBudWxsLFxyXG4gICAgICAgICAgICB3YXNFZGl0ZWQ6IGZhbHNlLFxyXG4gICAgICAgICAgICBvcmlnaW5hbENvbnRlbnQ6IHR5cGVvZiBzdWdnZXN0aW9uID09PSAnc3RyaW5nJyA/IHN1Z2dlc3Rpb24gOiAoc3VnZ2VzdGlvbi5tZW1vcnkgfHwgJycpLFxyXG4gICAgICAgICAgICBpc0F1dG9BcHBsaWVkOiB0cnVlXHJcbiAgICAgICAgfSkpLmZpbHRlcigobWVtb3J5KSA9PiBtZW1vcnkudGV4dCk7XHJcblxyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogJ1NBVkVfQVBQUk9WRURfTUVNT1JJRVMnLFxyXG4gICAgICAgICAgICAgICAgbWVtb3JpZXM6IG1lbW9yaWVzVG9TYXZlLFxyXG4gICAgICAgICAgICAgICAgbW9kZTogZGV0ZWN0ZWRNb2RlXHJcbiAgICAgICAgICAgIH0pO1xyXG5cclxuICAgICAgICAgICAgaWYgKHJlc3BvbnNlLnN0YXR1cyAhPT0gJ3N1Y2Nlc3MnKSB7XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IocmVzcG9uc2UubWVzc2FnZSB8fCAnRmFpbGVkIHRvIGF1dG8tc2F2ZSBtZW1vcmllcycpO1xyXG4gICAgICAgICAgICB9XHJcblxyXG4gICAgICAgICAgICBjb25zdCBzYXZlZE1lbW9yaWVzID0gcmVzcG9uc2Uuc2F2ZWQgfHwgW107XHJcbiAgICAgICAgICAgIGNvbnN0IGZhaWxlZE1lbW9yaWVzID0gcmVzcG9uc2UuZmFpbGVkIHx8IFtdO1xyXG5cclxuICAgICAgICAgICAgaWYgKHNhdmVkTWVtb3JpZXMubGVuZ3RoID09PSAwKSB7XHJcbiAgICAgICAgICAgICAgICBzdWdnZXN0aW9uc0NvbnRhaW5lci5yZW1vdmUoKTtcclxuICAgICAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICAgICAgfVxyXG5cclxuICAgICAgICAgICAgc3VnZ2VzdGlvbnNMaXN0LmlubmVySFRNTCA9ICcnO1xyXG4gICAgICAgICAgICBzYXZlZE1lbW9yaWVzLmZvckVhY2goKHNhdmVkTWVtb3J5LCBpbmRleCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgY29uc3QgaXRlbUhUTUwgPSB1aUJsdWVwcmludHMuZ2V0QXV0b1NhdmVkTWVtb3J5SXRlbShzYXZlZE1lbW9yeSwgaW5kZXgpO1xyXG4gICAgICAgICAgICAgICAgc3VnZ2VzdGlvbnNMaXN0Lmluc2VydEFkamFjZW50SFRNTCgnYmVmb3JlZW5kJywgaXRlbUhUTUwpO1xyXG5cclxuICAgICAgICAgICAgICAgIGNvbnN0IHN1Z2dlc3Rpb25JdGVtID0gc3VnZ2VzdGlvbnNMaXN0LnF1ZXJ5U2VsZWN0b3IoYFtkYXRhLWluZGV4PVwiJHtpbmRleH1cIl1gKTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IHVuZG9CdXR0b24gPSBzdWdnZXN0aW9uSXRlbS5xdWVyeVNlbGVjdG9yKCcudW5kby1idXR0b24nKTtcclxuICAgICAgICAgICAgICAgIHVuZG9CdXR0b24uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBhc3luYyAoZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIGUuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgYXdhaXQgaGFuZGxlVW5kb01lbW9yeShzYXZlZE1lbW9yeSwgc3VnZ2VzdGlvbkl0ZW0sIHN1Z2dlc3Rpb25zQ29udGFpbmVyKTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgICAgIHVwZGF0ZUF1dG9TYXZlZENvbnRhaW5lclN0YXRlKHN1Z2dlc3Rpb25zQ29udGFpbmVyLCBzYXZlZE1lbW9yaWVzLmxlbmd0aCwgZmFpbGVkTWVtb3JpZXMubGVuZ3RoKTtcclxuXHJcbiAgICAgICAgICAgIGNvbnN0IGRpc2NhcmRBbGxCdXR0b24gPSBzdWdnZXN0aW9uc0NvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcuZGlzY2FyZC1hbGwtYnV0dG9uJyk7XHJcbiAgICAgICAgICAgIGRpc2NhcmRBbGxCdXR0b24uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBhc3luYyAoZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgZS5zdG9wUHJvcGFnYXRpb24oKTtcclxuICAgICAgICAgICAgICAgIGF3YWl0IGhhbmRsZVVuZG9BbGwoc3VnZ2VzdGlvbnNDb250YWluZXIpO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBhdXRvLXNhdmluZyBtZW1vcmllczonLCBlcnJvcik7XHJcbiAgICAgICAgICAgIHN1Z2dlc3Rpb25zQ29udGFpbmVyLnJlbW92ZSgpO1xyXG5cclxuICAgICAgICAgICAgaWYgKC9tZW1vcnkgbGltaXQvaS50ZXN0KGVycm9yLm1lc3NhZ2UgfHwgJycpKSB7XHJcbiAgICAgICAgICAgICAgICBkaXNwbGF5RXh0cmFjdGVkTWVtb3JpZXMobWVtb3JpZXNUb1NhdmUubWFwKG1lbW9yeSA9PiBtZW1vcnkudGV4dCksIGZhbHNlLCBsaW1pdFR5cGUpO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG4gICAgfTtcclxuXHJcbiAgICAvLyBIYW5kbGUgYXBwcm92ZSBtZW1vcnkgYWN0aW9uXHJcbiAgICBjb25zdCBoYW5kbGVBcHByb3ZlTWVtb3J5ID0gYXN5bmMgKG1lbW9yeSwgdGFnLCBzdWdnZXN0aW9uSXRlbSkgPT4ge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIC8vIENoZWNrIGlmIHRoaXMgc3VnZ2VzdGlvbiBpcyBjdXJyZW50bHkgYmVpbmcgZWRpdGVkXHJcbiAgICAgICAgICAgIGlmIChzdWdnZXN0aW9uSXRlbS5nZXRBdHRyaWJ1dGUoJ2RhdGEtZWRpdGluZycpID09PSAndHJ1ZScpIHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdTdWdnZXN0aW9uIGlzIGJlaW5nIGVkaXRlZCwgc2tpcHBpbmcgYXBwcm92ZScpO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBHZXQgdGhlIGRldGVjdGVkIG1vZGUgZnJvbSB0aGUgc3VnZ2VzdGlvbnMgY29udGFpbmVyXHJcbiAgICAgICAgICAgIGNvbnN0IHN1Z2dlc3Rpb25zQ29udGFpbmVyID0gc3VnZ2VzdGlvbkl0ZW0uY2xvc2VzdCgnLm1lbW9yeS1zdWdnZXN0aW9ucy1jb250YWluZXInKTtcclxuICAgICAgICAgICAgY29uc3QgZGV0ZWN0ZWRNb2RlTGFiZWwgPSBzdWdnZXN0aW9uc0NvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcuZGV0ZWN0ZWQtbW9kZS1sYWJlbCcpO1xyXG4gICAgICAgICAgICBjb25zdCBkZXRlY3RlZE1vZGUgPSBkZXRlY3RlZE1vZGVMYWJlbCA/IGRldGVjdGVkTW9kZUxhYmVsLnRleHRDb250ZW50IDogbnVsbDtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIERpc2FibGUgYnV0dG9ucyB0byBwcmV2ZW50IGRvdWJsZS1jbGlja2luZ1xyXG4gICAgICAgICAgICBjb25zdCBidXR0b25zID0gc3VnZ2VzdGlvbkl0ZW0ucXVlcnlTZWxlY3RvckFsbCgnYnV0dG9uJyk7XHJcbiAgICAgICAgICAgIGJ1dHRvbnMuZm9yRWFjaChidG4gPT4gYnRuLmRpc2FibGVkID0gdHJ1ZSk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTaG93IGxvYWRpbmcgc3RhdGVcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUub3BhY2l0eSA9ICcwLjcnO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gU2F2ZSB0aGUgYXBwcm92ZWQgbWVtb3J5IChub3QgZWRpdGVkKVxyXG4gICAgICAgICAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHtcclxuICAgICAgICAgICAgICAgIHR5cGU6ICdTQVZFX0FQUFJPVkVEX01FTU9SSUVTJyxcclxuICAgICAgICAgICAgICAgIG1lbW9yaWVzOiBbe1xyXG4gICAgICAgICAgICAgICAgICAgIHRleHQ6IG1lbW9yeSxcclxuICAgICAgICAgICAgICAgICAgICB0YWc6IHRhZyxcclxuICAgICAgICAgICAgICAgICAgICB3YXNFZGl0ZWQ6IGZhbHNlLFxyXG4gICAgICAgICAgICAgICAgICAgIG9yaWdpbmFsQ29udGVudDogbWVtb3J5XHJcbiAgICAgICAgICAgICAgICB9XSxcclxuICAgICAgICAgICAgICAgIG1vZGU6IGRldGVjdGVkTW9kZVxyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIGlmIChyZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJykge1xyXG4gICAgICAgICAgICAgICAgLy8gU2hvdyBzdWNjZXNzIGZlZWRiYWNrXHJcbiAgICAgICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5zdHlsZS5jc3NUZXh0ICs9IGBcclxuICAgICAgICAgICAgICAgICAgICBiYWNrZ3JvdW5kOiAjZDRlZGRhO1xyXG4gICAgICAgICAgICAgICAgICAgIGJvcmRlci1jb2xvcjogI2MzZTZjYjtcclxuICAgICAgICAgICAgICAgICAgICB0cmFuc2Zvcm06IHNjYWxlKDAuOTgpO1xyXG4gICAgICAgICAgICAgICAgYDtcclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgY29uc3QgbWVtb3J5VGV4dCA9IHN1Z2dlc3Rpb25JdGVtLnF1ZXJ5U2VsZWN0b3IoJy5zdWdnZXN0aW9uLXRleHQnKTtcclxuICAgICAgICAgICAgICAgIG1lbW9yeVRleHQuc3R5bGUuY29sb3IgPSAnIzE1NTcyNCc7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlUZXh0LmlubmVySFRNTCA9IGDinJMgJHttZW1vcnl9IDxlbSBzdHlsZT1cImZvbnQtc2l6ZTogMTFweDsgb3BhY2l0eTogMC44O1wiPihzYXZlZCk8L2VtPmA7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIFJlbW92ZSBidXR0b25zXHJcbiAgICAgICAgICAgICAgICBjb25zdCBidXR0b25zQ29udGFpbmVyID0gc3VnZ2VzdGlvbkl0ZW0ucXVlcnlTZWxlY3RvcignZGl2Omxhc3QtY2hpbGQnKTtcclxuICAgICAgICAgICAgICAgIGlmIChidXR0b25zQ29udGFpbmVyKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgYnV0dG9uc0NvbnRhaW5lci5yZW1vdmUoKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIFxyXG5cclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ01lbW9yeSBhcHByb3ZlZCBhbmQgc2F2ZWQ6JywgbWVtb3J5KTtcclxuICAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihyZXNwb25zZS5tZXNzYWdlIHx8ICdGYWlsZWQgdG8gc2F2ZSBtZW1vcnknKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGFwcHJvdmluZyBtZW1vcnk6JywgZXJyb3IpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gU2hvdyBlcnJvciBmZWVkYmFja1xyXG4gICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5zdHlsZS5jc3NUZXh0ICs9IGBcclxuICAgICAgICAgICAgICAgIGJhY2tncm91bmQ6ICNmOGQ3ZGE7XHJcbiAgICAgICAgICAgICAgICBib3JkZXItY29sb3I6ICNmNWM2Y2I7XHJcbiAgICAgICAgICAgIGA7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBjb25zdCBtZW1vcnlUZXh0ID0gc3VnZ2VzdGlvbkl0ZW0ucXVlcnlTZWxlY3RvcignLnN1Z2dlc3Rpb24tdGV4dCcpO1xyXG4gICAgICAgICAgICBtZW1vcnlUZXh0LmlubmVySFRNTCA9IGAke21lbW9yeX0gPGVtIHN0eWxlPVwiY29sb3I6ICM3MjFjMjQ7IGZvbnQtc2l6ZTogMTFweDtcIj4oZmFpbGVkIHRvIHNhdmUpPC9lbT5gO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gUmUtZW5hYmxlIGJ1dHRvbnNcclxuICAgICAgICAgICAgY29uc3QgYnV0dG9ucyA9IHN1Z2dlc3Rpb25JdGVtLnF1ZXJ5U2VsZWN0b3JBbGwoJ2J1dHRvbicpO1xyXG4gICAgICAgICAgICBidXR0b25zLmZvckVhY2goYnRuID0+IGJ0bi5kaXNhYmxlZCA9IGZhbHNlKTtcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUub3BhY2l0eSA9ICcxJztcclxuICAgICAgICB9XHJcbiAgICB9O1xyXG5cclxuICAgIC8vIEhhbmRsZSBlZGl0IG1lbW9yeSBhY3Rpb25cclxuICAgIGNvbnN0IGhhbmRsZUVkaXRNZW1vcnkgPSAobWVtb3J5VGV4dEVsZW1lbnQsIGJ1dHRvbnNDb250YWluZXIsIG9yaWdpbmFsVGV4dCkgPT4ge1xyXG4gICAgICAgIC8vIE1hcmsgdGhpcyBzdWdnZXN0aW9uIGFzIGJlaW5nIGVkaXRlZCB0byBwcmV2ZW50IGR1cGxpY2F0ZSBzYXZpbmdcclxuICAgICAgICBjb25zdCBzdWdnZXN0aW9uSXRlbSA9IG1lbW9yeVRleHRFbGVtZW50LmNsb3Nlc3QoJy5tZW1vcnktc3VnZ2VzdGlvbi1pdGVtJyk7XHJcbiAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc2V0QXR0cmlidXRlKCdkYXRhLWVkaXRpbmcnLCAndHJ1ZScpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIENyZWF0ZSBlZGl0IGZpZWxkIHVzaW5nIGJsdWVwcmludFxyXG4gICAgICAgIGNvbnN0IGVkaXRGaWVsZEhUTUwgPSB1aUJsdWVwcmludHMuZ2V0TWVtb3J5RWRpdEZpZWxkKG9yaWdpbmFsVGV4dCk7XHJcbiAgICAgICAgbWVtb3J5VGV4dEVsZW1lbnQuaW5zZXJ0QWRqYWNlbnRIVE1MKCdhZnRlcmVuZCcsIGVkaXRGaWVsZEhUTUwpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGNvbnN0IGlucHV0RmllbGQgPSBzdWdnZXN0aW9uSXRlbS5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LWVkaXQtZmllbGQnKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBHZXQgZXhpc3RpbmcgYnV0dG9uc1xyXG4gICAgICAgIGNvbnN0IGFwcHJvdmVCdXR0b24gPSBidXR0b25zQ29udGFpbmVyLnF1ZXJ5U2VsZWN0b3IoJy5hcHByb3ZlLWJ1dHRvbicpO1xyXG4gICAgICAgIGNvbnN0IGVkaXRCdXR0b24gPSBidXR0b25zQ29udGFpbmVyLnF1ZXJ5U2VsZWN0b3IoJy5lZGl0LWJ1dHRvbicpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEhpZGUgZWRpdCBidXR0b24gZHVyaW5nIGVkaXQgbW9kZVxyXG4gICAgICAgIGVkaXRCdXR0b24uc3R5bGUuZGlzcGxheSA9ICdub25lJztcclxuICAgICAgICBcclxuICAgICAgICAvLyBIYW5kbGUgc2F2ZVxyXG4gICAgICAgIGNvbnN0IGhhbmRsZVNhdmUgPSBhc3luYyAoKSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IG5ld1RleHQgPSBpbnB1dEZpZWxkLnZhbHVlLnRyaW0oKTtcclxuICAgICAgICAgICAgaWYgKG5ld1RleHQgJiYgbmV3VGV4dCAhPT0gb3JpZ2luYWxUZXh0KSB7XHJcbiAgICAgICAgICAgICAgICB0cnkge1xyXG4gICAgICAgICAgICAgICAgICAgIC8vIEdldCB0aGUgZGV0ZWN0ZWQgbW9kZSBmcm9tIHRoZSBzdWdnZXN0aW9ucyBjb250YWluZXJcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCBzdWdnZXN0aW9uSXRlbSA9IG1lbW9yeVRleHRFbGVtZW50LmNsb3Nlc3QoJy5tZW1vcnktc3VnZ2VzdGlvbi1pdGVtJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc3Qgc3VnZ2VzdGlvbnNDb250YWluZXIgPSBzdWdnZXN0aW9uSXRlbS5jbG9zZXN0KCcubWVtb3J5LXN1Z2dlc3Rpb25zLWNvbnRhaW5lcicpO1xyXG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IGRldGVjdGVkTW9kZUxhYmVsID0gc3VnZ2VzdGlvbnNDb250YWluZXIucXVlcnlTZWxlY3RvcignLmRldGVjdGVkLW1vZGUtbGFiZWwnKTtcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCBkZXRlY3RlZE1vZGUgPSBkZXRlY3RlZE1vZGVMYWJlbCA/IGRldGVjdGVkTW9kZUxhYmVsLnRleHRDb250ZW50IDogbnVsbDtcclxuICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCB0YWdCYWRnZSA9IHN1Z2dlc3Rpb25JdGVtLnF1ZXJ5U2VsZWN0b3IoJy5zdWdnZXN0aW9uLXRhZy1iYWRnZScpO1xyXG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IGN1cnJlbnRUYWcgPSB0YWdCYWRnZSA/IHRhZ0JhZGdlLnRleHRDb250ZW50LnRyaW0oKSA6IG51bGw7XHJcbiAgICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAgLy8gRGlzYWJsZSBidXR0b25zIHRvIHByZXZlbnQgZG91YmxlLWNsaWNraW5nXHJcbiAgICAgICAgICAgICAgICAgICAgY29uc3QgYnV0dG9ucyA9IGJ1dHRvbnNDb250YWluZXIucXVlcnlTZWxlY3RvckFsbCgnYnV0dG9uJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgYnV0dG9ucy5mb3JFYWNoKGJ0biA9PiBidG4uZGlzYWJsZWQgPSB0cnVlKTtcclxuICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICAvLyBTaG93IGxvYWRpbmcgc3RhdGVcclxuICAgICAgICAgICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5zdHlsZS5vcGFjaXR5ID0gJzAuNyc7XHJcbiAgICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAgLy8gU2F2ZSB0aGUgZWRpdGVkIG1lbW9yeVxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2Uoe1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB0eXBlOiAnU0FWRV9BUFBST1ZFRF9NRU1PUklFUycsXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG1lbW9yaWVzOiBbe1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGV4dDogbmV3VGV4dCxcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRhZzogY3VycmVudFRhZyxcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdhc0VkaXRlZDogdHJ1ZSxcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9yaWdpbmFsQ29udGVudDogb3JpZ2luYWxUZXh0XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH1dLFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBtb2RlOiBkZXRlY3RlZE1vZGVcclxuICAgICAgICAgICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICBpZiAocmVzcG9uc2Uuc3RhdHVzID09PSAnc3VjY2VzcycpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gU2hvdyBzdWNjZXNzIGZlZWRiYWNrIGFuZCByZXR1cm4gdG8gbm9ybWFsIHRleHQgZGlzcGxheVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBzdWdnZXN0aW9uSXRlbS5zdHlsZS5jc3NUZXh0ICs9IGBcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJhY2tncm91bmQ6ICNkNGVkZGE7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBib3JkZXItY29sb3I6ICNjM2U2Y2I7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0cmFuc2Zvcm06IHNjYWxlKDAuOTgpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgb3BhY2l0eTogMTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgYDtcclxuICAgICAgICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIC8vIFJlbW92ZSBpbnB1dCBmaWVsZCBhbmQgcmVzdG9yZSB0ZXh0IGRpc3BsYXlcclxuICAgICAgICAgICAgICAgICAgICAgICAgaW5wdXRGaWVsZC5yZW1vdmUoKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgbWVtb3J5VGV4dEVsZW1lbnQudGV4dENvbnRlbnQgPSBuZXdUZXh0O1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBtZW1vcnlUZXh0RWxlbWVudC5zdHlsZS5kaXNwbGF5ID0gJ2Jsb2NrJztcclxuICAgICAgICAgICAgICAgICAgICAgICAgbWVtb3J5VGV4dEVsZW1lbnQuc3R5bGUuY29sb3IgPSAnIzE1NTcyNCc7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG1lbW9yeVRleHRFbGVtZW50LmlubmVySFRNTCA9IGDinJMgJHtuZXdUZXh0fSA8ZW0gc3R5bGU9XCJmb250LXNpemU6IDExcHg7IG9wYWNpdHk6IDAuODtcIj4oc2F2ZWQpPC9lbT5gO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gUmVtb3ZlIGJ1dHRvbnNcclxuICAgICAgICAgICAgICAgICAgICAgICAgYnV0dG9uc0NvbnRhaW5lci5yZW1vdmUoKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgXHJcblxyXG4gICAgICAgICAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ01lbW9yeSBlZGl0ZWQgYW5kIHNhdmVkOicsIG5ld1RleHQpO1xyXG4gICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihyZXNwb25zZS5tZXNzYWdlIHx8ICdGYWlsZWQgdG8gc2F2ZSBlZGl0ZWQgbWVtb3J5Jyk7XHJcbiAgICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICAgICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBzYXZpbmcgZWRpdGVkIG1lbW9yeTonLCBlcnJvcik7XHJcbiAgICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAgLy8gU2hvdyBlcnJvciBmZWVkYmFjayBhbmQgcmVzdG9yZSBlZGl0IG1vZGVcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCBzdWdnZXN0aW9uSXRlbSA9IG1lbW9yeVRleHRFbGVtZW50LmNsb3Nlc3QoJy5tZW1vcnktc3VnZ2VzdGlvbi1pdGVtJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgc3VnZ2VzdGlvbkl0ZW0uc3R5bGUuY3NzVGV4dCArPSBgXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGJhY2tncm91bmQ6ICNmOGQ3ZGE7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGJvcmRlci1jb2xvcjogI2Y1YzZjYjtcclxuICAgICAgICAgICAgICAgICAgICAgICAgb3BhY2l0eTogMTtcclxuICAgICAgICAgICAgICAgICAgICBgO1xyXG4gICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgIC8vIFJlLWVuYWJsZSBidXR0b25zXHJcbiAgICAgICAgICAgICAgICAgICAgY29uc3QgYnV0dG9ucyA9IGJ1dHRvbnNDb250YWluZXIucXVlcnlTZWxlY3RvckFsbCgnYnV0dG9uJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgYnV0dG9ucy5mb3JFYWNoKGJ0biA9PiBidG4uZGlzYWJsZWQgPSBmYWxzZSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH0gZWxzZSBpZiAobmV3VGV4dCA9PT0gb3JpZ2luYWxUZXh0KSB7XHJcbiAgICAgICAgICAgICAgICAvLyBObyBjaGFuZ2VzIG1hZGUsIGp1c3QgcmVzdG9yZSBvcmlnaW5hbCBkaXNwbGF5XHJcbiAgICAgICAgICAgICAgICBpbnB1dEZpZWxkLnJlbW92ZSgpO1xyXG4gICAgICAgICAgICAgICAgbWVtb3J5VGV4dEVsZW1lbnQudGV4dENvbnRlbnQgPSBvcmlnaW5hbFRleHQ7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlUZXh0RWxlbWVudC5zdHlsZS5kaXNwbGF5ID0gJ2Jsb2NrJztcclxuICAgICAgICAgICAgICAgIGVkaXRCdXR0b24uc3R5bGUuZGlzcGxheSA9ICdmbGV4JztcclxuICAgICAgICAgICAgICAgIC8vIFJlbW92ZSBlZGl0aW5nIGZsYWcgc2luY2Ugbm8gY2hhbmdlcyB3ZXJlIG1hZGVcclxuICAgICAgICAgICAgICAgIGNvbnN0IHN1Z2dlc3Rpb25JdGVtID0gbWVtb3J5VGV4dEVsZW1lbnQuY2xvc2VzdCgnLm1lbW9yeS1zdWdnZXN0aW9uLWl0ZW0nKTtcclxuICAgICAgICAgICAgICAgIHN1Z2dlc3Rpb25JdGVtLnJlbW92ZUF0dHJpYnV0ZSgnZGF0YS1lZGl0aW5nJyk7XHJcbiAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAvLyBFbXB0eSB0ZXh0LCByZXN0b3JlIG9yaWdpbmFsIHRleHQgYW5kIGJ1dHRvbnNcclxuICAgICAgICAgICAgICAgIGlucHV0RmllbGQucmVtb3ZlKCk7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlUZXh0RWxlbWVudC50ZXh0Q29udGVudCA9IG9yaWdpbmFsVGV4dDtcclxuICAgICAgICAgICAgICAgIG1lbW9yeVRleHRFbGVtZW50LnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xyXG4gICAgICAgICAgICAgICAgZWRpdEJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ2ZsZXgnO1xyXG4gICAgICAgICAgICAgICAgLy8gUmVtb3ZlIGVkaXRpbmcgZmxhZyBzaW5jZSBlZGl0IHdhcyBjYW5jZWxsZWRcclxuICAgICAgICAgICAgICAgIGNvbnN0IHN1Z2dlc3Rpb25JdGVtID0gbWVtb3J5VGV4dEVsZW1lbnQuY2xvc2VzdCgnLm1lbW9yeS1zdWdnZXN0aW9uLWl0ZW0nKTtcclxuICAgICAgICAgICAgICAgIHN1Z2dlc3Rpb25JdGVtLnJlbW92ZUF0dHJpYnV0ZSgnZGF0YS1lZGl0aW5nJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9O1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFVwZGF0ZSBhcHByb3ZlIGJ1dHRvbiB0byBhY3QgYXMgc2F2ZSBidXR0b24gZHVyaW5nIGVkaXRcclxuICAgICAgICBjb25zdCBvcmlnaW5hbEFwcHJvdmVIYW5kbGVyID0gYXBwcm92ZUJ1dHRvbi5vbmNsaWNrO1xyXG4gICAgICAgIGFwcHJvdmVCdXR0b24ub25jbGljayA9IGFzeW5jIChlKSA9PiB7XHJcbiAgICAgICAgICAgIGUuc3RvcFByb3BhZ2F0aW9uKCk7XHJcbiAgICAgICAgICAgIGF3YWl0IGhhbmRsZVNhdmUoKTtcclxuICAgICAgICB9O1xyXG4gICAgICAgIFxyXG4gICAgICAgIGlucHV0RmllbGQuYWRkRXZlbnRMaXN0ZW5lcigna2V5cHJlc3MnLCBhc3luYyAoZSkgPT4ge1xyXG4gICAgICAgICAgICBpZiAoZS5rZXkgPT09ICdFbnRlcicgJiYgZS5jdHJsS2V5KSB7XHJcbiAgICAgICAgICAgICAgICBlLnByZXZlbnREZWZhdWx0KCk7XHJcbiAgICAgICAgICAgICAgICBhd2FpdCBoYW5kbGVTYXZlKCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBBZGQgZXNjYXBlIGtleSB0byBjYW5jZWwgZWRpdFxyXG4gICAgICAgIGlucHV0RmllbGQuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIChlKSA9PiB7XHJcbiAgICAgICAgICAgIGlmIChlLmtleSA9PT0gJ0VzY2FwZScpIHtcclxuICAgICAgICAgICAgICAgIC8vIENhbmNlbCBlZGl0IGFuZCByZXN0b3JlIG9yaWdpbmFsIGRpc3BsYXlcclxuICAgICAgICAgICAgICAgIGlucHV0RmllbGQucmVtb3ZlKCk7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlUZXh0RWxlbWVudC50ZXh0Q29udGVudCA9IG9yaWdpbmFsVGV4dDtcclxuICAgICAgICAgICAgICAgIG1lbW9yeVRleHRFbGVtZW50LnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xyXG4gICAgICAgICAgICAgICAgZWRpdEJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ2ZsZXgnO1xyXG4gICAgICAgICAgICAgICAgLy8gUmVtb3ZlIGVkaXRpbmcgZmxhZyBzaW5jZSBlZGl0IHdhcyBjYW5jZWxsZWRcclxuICAgICAgICAgICAgICAgIGNvbnN0IHN1Z2dlc3Rpb25JdGVtID0gbWVtb3J5VGV4dEVsZW1lbnQuY2xvc2VzdCgnLm1lbW9yeS1zdWdnZXN0aW9uLWl0ZW0nKTtcclxuICAgICAgICAgICAgICAgIHN1Z2dlc3Rpb25JdGVtLnJlbW92ZUF0dHJpYnV0ZSgnZGF0YS1lZGl0aW5nJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBIaWRlIG9yaWdpbmFsIHRleHQsIHNob3cgaW5wdXRcclxuICAgICAgICBtZW1vcnlUZXh0RWxlbWVudC5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEluc2VydCBpbnB1dCBmaWVsZFxyXG4gICAgICAgIG1lbW9yeVRleHRFbGVtZW50LnBhcmVudE5vZGUuaW5zZXJ0QmVmb3JlKGlucHV0RmllbGQsIGJ1dHRvbnNDb250YWluZXIpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEZvY3VzIHRoZSBpbnB1dFxyXG4gICAgICAgIGlucHV0RmllbGQuZm9jdXMoKTtcclxuICAgICAgICBpbnB1dEZpZWxkLnNlbGVjdCgpO1xyXG4gICAgfTtcclxuICAgIFxyXG4gICAgLy8gSGFuZGxlIGRpc2NhcmQgYWxsIGFjdGlvblxyXG4gICAgY29uc3QgaGFuZGxlRGlzY2FyZEFsbCA9IChzdWdnZXN0aW9uc0NvbnRhaW5lcikgPT4ge1xyXG4gICAgICAgIC8vIEdldCB0aGUgZGV0ZWN0ZWQgbW9kZSBmcm9tIHRoZSBjb250YWluZXJcclxuICAgICAgICBjb25zdCBkZXRlY3RlZE1vZGVMYWJlbCA9IHN1Z2dlc3Rpb25zQ29udGFpbmVyLnF1ZXJ5U2VsZWN0b3IoJy5kZXRlY3RlZC1tb2RlLWxhYmVsJyk7XHJcbiAgICAgICAgY29uc3QgZGV0ZWN0ZWRNb2RlID0gZGV0ZWN0ZWRNb2RlTGFiZWwgPyBkZXRlY3RlZE1vZGVMYWJlbC50ZXh0Q29udGVudCA6IG51bGw7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gR2V0IGFsbCBzdWdnZXN0aW9uIGl0ZW1zIHRvIHRyYWNrIHdoYXQncyBiZWluZyBkaXNjYXJkZWRcclxuICAgICAgICBjb25zdCBzdWdnZXN0aW9uSXRlbXMgPSBzdWdnZXN0aW9uc0NvbnRhaW5lci5xdWVyeVNlbGVjdG9yQWxsKCcubWVtb3J5LXN1Z2dlc3Rpb24taXRlbScpO1xyXG4gICAgICAgIGNvbnN0IGRpc2NhcmRlZFN1Z2dlc3Rpb25zID0gW107XHJcbiAgICAgICAgXHJcbiAgICAgICAgc3VnZ2VzdGlvbkl0ZW1zLmZvckVhY2goaXRlbSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IHRleHRFbGVtZW50ID0gaXRlbS5xdWVyeVNlbGVjdG9yKCcuc3VnZ2VzdGlvbi10ZXh0Jyk7XHJcbiAgICAgICAgICAgIGlmICh0ZXh0RWxlbWVudCAmJiAhaXRlbS5nZXRBdHRyaWJ1dGUoJ2RhdGEtZWRpdGluZycpKSB7XHJcbiAgICAgICAgICAgICAgICBkaXNjYXJkZWRTdWdnZXN0aW9ucy5wdXNoKHRleHRFbGVtZW50LnRleHRDb250ZW50KTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFRyYWNrIHRoZSBkaXNjYXJkIGV2ZW50XHJcbiAgICAgICAgaWYgKGRpc2NhcmRlZFN1Z2dlc3Rpb25zLmxlbmd0aCA+IDApIHtcclxuICAgICAgICAgICAgLy8gVHJhY2sgZGlzY2FyZGVkIHN1Z2dlc3Rpb25zIHdpdGggZGV0ZWN0ZWQgbW9kZVxyXG4gICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrTWVtb3J5U3VnZ2VzdGlvbkRpc2NhcmRlZChkaXNjYXJkZWRTdWdnZXN0aW9ucy5sZW5ndGgsIGRpc2NhcmRlZFN1Z2dlc3Rpb25zLCBkZXRlY3RlZE1vZGUpO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICAvLyBBZGQgZmFkZS1vdXQgYW5pbWF0aW9uIHRvIGVudGlyZSBjb250YWluZXJcclxuICAgICAgICBzdWdnZXN0aW9uc0NvbnRhaW5lci5zdHlsZS50cmFuc2l0aW9uID0gJ2FsbCAwLjNzIGVhc2UnO1xyXG4gICAgICAgIHN1Z2dlc3Rpb25zQ29udGFpbmVyLnN0eWxlLm9wYWNpdHkgPSAnMCc7XHJcbiAgICAgICAgc3VnZ2VzdGlvbnNDb250YWluZXIuc3R5bGUudHJhbnNmb3JtID0gJ3RyYW5zbGF0ZVkoLTEwcHgpIHNjYWxlKDAuOTgpJztcclxuICAgICAgICBcclxuICAgICAgICAvLyBSZW1vdmUgdGhlIGNvbnRhaW5lciBhZnRlciBhbmltYXRpb25cclxuICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcclxuICAgICAgICAgICAgc3VnZ2VzdGlvbnNDb250YWluZXIucmVtb3ZlKCk7XHJcbiAgICAgICAgfSwgMzAwKTtcclxuICAgICAgICBcclxuICAgICAgICBjb25zb2xlLmxvZygnQWxsIG1lbW9yeSBzdWdnZXN0aW9ucyBkaXNjYXJkZWQ6JywgZGlzY2FyZGVkU3VnZ2VzdGlvbnMubGVuZ3RoKTtcclxuICAgIH07XHJcblxyXG4gLyoqXHJcbiAqIFNjcmFwZXMgdGhlIGxhc3QgZmV3IGNoYXQgbWVzc2FnZXMgZnJvbSB0aGUgcGFnZSB3aXRoIGhpZ2ggcHJlY2lzaW9uLlxyXG4gKiBAcGFyYW0ge251bWJlcn0gbnVtTWVzc2FnZXMgLSBUaGUgbnVtYmVyIG9mIHJlY2VudCBtZXNzYWdlcyB0byByZXRyaWV2ZS5cclxuICogQHJldHVybnMge0FycmF5PHtyb2xlOiBzdHJpbmcsIHRleHQ6IHN0cmluZ30+fSAtIEFuIGFycmF5IG9mIG1lc3NhZ2Ugb2JqZWN0cy5cclxuICovXHJcbi8vIE5FVywgVkVSQk9TRSBERUJVR0dJTkcgVkVSU0lPTiBvZiBzY3JhcGVDb252ZXJzYXRpb25IaXN0b3J5XHJcblxyXG4vLyBGSU5BTCBQUk9EVUNUSU9OIFZFUlNJT04gb2Ygc2NyYXBlQ29udmVyc2F0aW9uSGlzdG9yeVxyXG5cclxuLyoqXHJcbiAqIFNjcmFwZXMgdGhlIGxhc3QgZmV3IGNoYXQgbWVzc2FnZXMgZnJvbSB0aGUgcGFnZSB3aXRoIGhpZ2ggcHJlY2lzaW9uLlxyXG4gKiBAcGFyYW0ge251bWJlcn0gbnVtTWVzc2FnZXMgLSBUaGUgbnVtYmVyIG9mIHJlY2VudCBtZXNzYWdlcyB0byByZXRyaWV2ZS5cclxuICogQHJldHVybnMge0FycmF5PHtyb2xlOiBzdHJpbmcsIHRleHQ6IHN0cmluZ30+fSAtIEFuIGFycmF5IG9mIG1lc3NhZ2Ugb2JqZWN0cy5cclxuICovXHJcbi8vIEZJTkFMIFBST0RVQ1RJT04gVkVSU0lPTiBvZiBzY3JhcGVDb252ZXJzYXRpb25IaXN0b3J5ICh3aXRoIG1lbW9yeSBjYXB0dXJlKVxyXG5cclxuLyoqXHJcbiAqIFNjcmFwZXMgdGhlIGxhc3QgZmV3IGNoYXQgbWVzc2FnZXMgZnJvbSB0aGUgcGFnZSwgaW50ZWxsaWdlbnRseSBzZXBhcmF0aW5nXHJcbiAqIHRoZSB1c2VyJ3Mgb3JpZ2luYWwgdGV4dCBmcm9tIGFueSBpbmplY3RlZCBNYXhNZW1vcnkgVUkgc2VjdGlvbnMuXHJcbiAqIEBwYXJhbSB7bnVtYmVyfSBudW1NZXNzYWdlcyAtIFRoZSBudW1iZXIgb2YgcmVjZW50IG1lc3NhZ2VzIHRvIHJldHJpZXZlLlxyXG4gKiBAcmV0dXJucyB7QXJyYXk8e3JvbGU6IHN0cmluZywgdGV4dDogc3RyaW5nLCByZXRyaWV2ZWRNZW1vcmllcz86IHN0cmluZ30+fSBcclxuICogICAgICAgICAgQW4gYXJyYXkgb2YgbWVzc2FnZSBvYmplY3RzLiBUaGUgJ3JldHJpZXZlZE1lbW9yaWVzJyBrZXkgd2lsbCBvbmx5XHJcbiAqICAgICAgICAgIGV4aXN0IG9uIHVzZXIgbWVzc2FnZXMgd2hlcmUgbWVtb3JpZXMgd2VyZSBpbmplY3RlZC5cclxuICovXHJcbmZ1bmN0aW9uIHNjcmFwZUNvbnZlcnNhdGlvbkhpc3RvcnkobnVtTWVzc2FnZXMgPSA0KSB7XHJcbiAgICBjb25zb2xlLmxvZyhgW1NjcmFwZXJdIFNjcmFwaW5nIGxhc3QgJHtudW1NZXNzYWdlc30gbWVzc2FnZXMuYCk7XHJcbiAgICBjb25zdCBtZXNzYWdlcyA9IFtdO1xyXG4gICAgXHJcbiAgICBjb25zdCBtZXNzYWdlTm9kZXMgPSBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCdbZGF0YS1tZXNzYWdlLWF1dGhvci1yb2xlXScpO1xyXG4gICAgY29uc3QgcmVjZW50Tm9kZXMgPSBBcnJheS5mcm9tKG1lc3NhZ2VOb2Rlcykuc2xpY2UoLW51bU1lc3NhZ2VzKTtcclxuXHJcbiAgICBmb3IgKGNvbnN0IG5vZGUgb2YgcmVjZW50Tm9kZXMpIHtcclxuICAgICAgICBjb25zdCByb2xlID0gbm9kZS5nZXRBdHRyaWJ1dGUoJ2RhdGEtbWVzc2FnZS1hdXRob3Itcm9sZScpO1xyXG4gICAgICAgIGlmICghcm9sZSkgY29udGludWU7XHJcblxyXG4gICAgICAgIGNvbnN0IGNsb25lZE5vZGUgPSBub2RlLmNsb25lTm9kZSh0cnVlKTtcclxuICAgICAgICBsZXQgcmV0cmlldmVkTWVtb3JpZXNUZXh0ID0gbnVsbDtcclxuXHJcbiAgICAgICAgLy8gLS0tIFRIRSBGSVggLS0tXHJcbiAgICAgICAgLy8gMS4gTG9vayBmb3IgdGhlIG1lc3NhZ2UgZGl2IHdpdGhpbiB0aGUgY2xvbmVkIG5vZGUuXHJcbiAgICAgICAgY29uc3QgbWVzc2FnZURpdiA9IGdldE1lc3NhZ2VDb250ZW50RWxlbWVudChjbG9uZWROb2RlKTtcclxuXHJcbiAgICAgICAgLy8gMi4gUHJpb3JpdGl6ZSByZWFkaW5nIHRoZSBmdWxsLCBjbGVhbiBkYXRhIGZyb20gb3VyIG5ldyBhdHRyaWJ1dGUuXHJcbiAgICAgICAgaWYgKG1lc3NhZ2VEaXYgJiYgbWVzc2FnZURpdi5oYXNBdHRyaWJ1dGUoJ2RhdGEtZnVsbC1tZW1vcmllcycpKSB7XHJcbiAgICAgICAgICAgIHJldHJpZXZlZE1lbW9yaWVzVGV4dCA9IG1lc3NhZ2VEaXYuZ2V0QXR0cmlidXRlKCdkYXRhLWZ1bGwtbWVtb3JpZXMnKTtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coYFtTY3JhcGVyXSBTdWNjZXNzOiBGb3VuZCBhbmQgZXh0cmFjdGVkIGZ1bGwgbWVtb3JpZXMgZnJvbSBkYXRhLWF0dHJpYnV0ZS5gKTtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coe3JldHJpZXZlZE1lbW9yaWVzVGV4dH0pXHJcbiAgICAgICAgfSBcclxuICAgICAgICAvLyAzLiBGYWxsYmFjazogSWYgdGhlIGF0dHJpYnV0ZSBkb2Vzbid0IGV4aXN0LCBzY3JhcGUgZnJvbSB0aGUgdGV4dCBjb250ZW50LlxyXG4gICAgICAgIC8vIFRoaXMgaGFuZGxlcyBvbGRlciBtZXNzYWdlcyBvciBhbnkgZWRnZSBjYXNlcy5cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgICAgY29uc3QgbWVtb3J5U2VjdGlvbk5vZGUgPSBjbG9uZWROb2RlLnF1ZXJ5U2VsZWN0b3IoJy5tZW1vcnktc2VjdGlvbicpO1xyXG4gICAgICAgICAgICBpZiAobWVtb3J5U2VjdGlvbk5vZGUpIHtcclxuICAgICAgICAgICAgICAgIC8vIFRoaXMgbWlnaHQgYmUgdHJ1bmNhdGVkIGRhdGEsIGJ1dCBpdCdzIGJldHRlciB0aGFuIG5vdGhpbmcuXHJcbiAgICAgICAgICAgICAgICByZXRyaWV2ZWRNZW1vcmllc1RleHQgPSBtZW1vcnlTZWN0aW9uTm9kZS50ZXh0Q29udGVudC50cmltKCk7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLndhcm4oYFtTY3JhcGVyXSBGYWxsYmFjazogU2NyYXBpbmcgZnJvbSB0ZXh0Q29udGVudC4gRGF0YSBtYXkgYmUgdHJ1bmNhdGVkLmApO1xyXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coe3JldHJpZXZlZE1lbW9yaWVzVGV4dH0pXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgLy8gLS0tIEVORCBGSVggLS0tXHJcblxyXG4gICAgICAgIC8vIFJlbW92ZSBhbnkgb3RoZXIgVUkgYXJ0aWZhY3RzIGZvciBhIGNsZWFuIHVzZXIgbWVzc2FnZSB0ZXh0XHJcbiAgICAgICAgY2xvbmVkTm9kZS5xdWVyeVNlbGVjdG9yQWxsKCcubWVtb3J5LXNlY3Rpb24sIC5tZW1vcnktc3VnZ2VzdGlvbnMtY29udGFpbmVyLCAuZXh0cmFjdGVkLW1lbW9yeS1ub3RpZmljYXRpb24sIC5tZW1vcnktbGltaXQtd2FybmluZycpXHJcbiAgICAgICAgICAgICAgICAgIC5mb3JFYWNoKGVsID0+IGVsLnJlbW92ZSgpKTtcclxuXHJcbiAgICAgICAgY29uc3Qgb3JpZ2luYWxVc2VyVGV4dCA9IGNsb25lZE5vZGUudGV4dENvbnRlbnQudHJpbSgpO1xyXG5cclxuICAgICAgICBpZiAob3JpZ2luYWxVc2VyVGV4dCB8fCByZXRyaWV2ZWRNZW1vcmllc1RleHQpIHsgLy8gRW5zdXJlIHdlIGFkZCB0dXJucyB0aGF0IG9ubHkgY29udGFpbiBtZW1vcmllc1xyXG4gICAgICAgICAgICBjb25zdCBtZXNzYWdlT2JqZWN0ID0ge1xyXG4gICAgICAgICAgICAgICAgcm9sZTogcm9sZSxcclxuICAgICAgICAgICAgICAgIHRleHQ6IG9yaWdpbmFsVXNlclRleHRcclxuICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIGlmIChyZXRyaWV2ZWRNZW1vcmllc1RleHQpIHtcclxuICAgICAgICAgICAgICAgIG1lc3NhZ2VPYmplY3QucmV0cmlldmVkTWVtb3JpZXMgPSByZXRyaWV2ZWRNZW1vcmllc1RleHQ7XHJcbiAgICAgICAgICAgIH1cclxuXHJcbiAgICAgICAgICAgIG1lc3NhZ2VzLnB1c2gobWVzc2FnZU9iamVjdCk7XHJcbiAgICAgICAgfVxyXG4gICAgfVxyXG4gICAgXHJcbiAgICBjb25zb2xlLmxvZyhgW1NjcmFwZXJdIFNjcmFwZWQgJHttZXNzYWdlcy5sZW5ndGh9IHZhbGlkIG1lc3NhZ2VzLmApO1xyXG4gICAgY29uc29sZS5sb2coe21lc3NhZ2VzfSlcclxuICAgIHJldHVybiBtZXNzYWdlcztcclxufVxyXG4gICAgLy8gSGVscGVyIGZ1bmN0aW9ucyBmb3IgaW5wdXQgY29udGVudCBtYW5hZ2VtZW50XHJcbiAgICBjb25zdCBnZXRJbnB1dENvbnRlbnQgPSAoaW5wdXRCb3gpID0+IHtcclxuICAgICAgICByZXR1cm4gaW5wdXRCb3gudGFnTmFtZSA9PT0gJ1RFWFRBUkVBJyA/IFxyXG4gICAgICAgICAgICBpbnB1dEJveC52YWx1ZS50cmltKCkgOiBcclxuICAgICAgICAgICAgQXJyYXkuZnJvbShpbnB1dEJveC5xdWVyeVNlbGVjdG9yQWxsKCdwJykpXHJcbiAgICAgICAgICAgICAgICAubWFwKHAgPT4gcC50ZXh0Q29udGVudC50cmltKCkpXHJcbiAgICAgICAgICAgICAgICAuam9pbignXFxuJyk7XHJcbiAgICB9O1xyXG5cclxuICAgIGNvbnN0IHNldElucHV0Q29udGVudCA9IChpbnB1dEJveCwgY29udGVudCkgPT4ge1xyXG4gICAgICAgIGNvbnNvbGUubG9nKCdbQ29udGVudFNjcmlwdF0gU2V0dGluZyBpbnB1dCBjb250ZW50LCB0eXBlOicsIGlucHV0Qm94LnRhZ05hbWUpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGlmIChpbnB1dEJveC50YWdOYW1lID09PSAnVEVYVEFSRUEnKSB7XHJcbiAgICAgICAgICAgIGlucHV0Qm94LnZhbHVlID0gY29udGVudDtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFRyaWdnZXIgaW5wdXQgZXZlbnQgdG8gZW5zdXJlIENoYXRHUFQncyBsaXN0ZW5lcnMgZGV0ZWN0IHRoZSBjaGFuZ2VcclxuICAgICAgICAgICAgY29uc3QgaW5wdXRFdmVudCA9IG5ldyBFdmVudCgnaW5wdXQnLCB7IGJ1YmJsZXM6IHRydWUgfSk7XHJcbiAgICAgICAgICAgIGlucHV0Qm94LmRpc3BhdGNoRXZlbnQoaW5wdXRFdmVudCk7XHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgaW5wdXRCb3guaW5uZXJIVE1MID0gYDxwPiR7Y29udGVudH08L3A+YDtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFRyaWdnZXIgbXV0YXRpb24gZXZlbnRzIGZvciBjb250ZW50ZWRpdGFibGVcclxuICAgICAgICAgICAgY29uc3QgaW5wdXRFdmVudCA9IG5ldyBFdmVudCgnaW5wdXQnLCB7IGJ1YmJsZXM6IHRydWUgfSk7XHJcbiAgICAgICAgICAgIGlucHV0Qm94LmRpc3BhdGNoRXZlbnQoaW5wdXRFdmVudCk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEZvY3VzIGFuZCBtb3ZlIGN1cnNvciB0byBlbmRcclxuICAgICAgICBpbnB1dEJveC5mb2N1cygpO1xyXG4gICAgICAgIGNvbnN0IHJhbmdlID0gZG9jdW1lbnQuY3JlYXRlUmFuZ2UoKTtcclxuICAgICAgICByYW5nZS5zZWxlY3ROb2RlQ29udGVudHMoaW5wdXRCb3gpO1xyXG4gICAgICAgIHJhbmdlLmNvbGxhcHNlKGZhbHNlKTtcclxuICAgICAgICBjb25zdCBzZWxlY3Rpb24gPSB3aW5kb3cuZ2V0U2VsZWN0aW9uKCk7XHJcbiAgICAgICAgc2VsZWN0aW9uLnJlbW92ZUFsbFJhbmdlcygpO1xyXG4gICAgICAgIHNlbGVjdGlvbi5hZGRSYW5nZShyYW5nZSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgY29uc29sZS5sb2coJ1tDb250ZW50U2NyaXB0XSBJbnB1dCBjb250ZW50IHNldCwgbGVuZ3RoOicsIGNvbnRlbnQubGVuZ3RoKTtcclxuICAgIH07XHJcblxyXG59KSgpO1xyXG4iXSwibmFtZXMiOlsidXNlcklucHV0Il0sIm1hcHBpbmdzIjoiQ0FFQyxXQUFXO0FBRVIsUUFBTSxvQkFBb0I7QUFBQSxJQUN0QixXQUFXO0FBQUEsSUFDWCxNQUFNO0FBQUEsSUFDTixlQUFlO0FBQUEsRUFBQTtBQUluQixRQUFNLGtDQUFrQyxPQUFPLG1CQUFtQjtBQUM5RCxRQUFJO0FBQ0EsWUFBTSxhQUFhLE1BQU0sY0FBYyxtQkFBQTtBQUV2QyxVQUFJLFdBQVcsV0FBVyxhQUFhLENBQUMsV0FBVyxRQUFRO0FBRXZELDBCQUFrQixjQUFjO0FBQUEsTUFDcEM7QUFBQSxJQUNKLFNBQVMsT0FBTztBQUNaLGNBQVEsTUFBTSxvREFBb0QsS0FBSztBQUFBLElBQzNFO0FBQUEsRUFDSjtBQUdBLFFBQU0sb0JBQW9CLENBQUMsV0FBVztBQUVsQyxRQUFJLE9BQU8sY0FBYyx5QkFBeUIsR0FBRztBQUNqRDtBQUFBLElBQ0o7QUFFQSxVQUFNLFNBQVMsU0FBUyxjQUFjLEtBQUs7QUFDM0MsV0FBTyxZQUFZO0FBRW5CLFdBQU8sWUFBWSxNQUFNO0FBR3pCLFdBQU8sUUFBUTtBQUFBLEVBQ25CO0FBS0EsUUFBTSxlQUFlO0FBQUEsSUFDakIsK0JBQStCLENBQUMsV0FBVyxrQkFBa0IsZUFBZSxNQUFNLGFBQWEsTUFBTSxrQkFBa0IsZUFBZTtBQUFBLHlFQUNyRSxTQUFTO0FBQUE7QUFBQSxrRUFFaEIsZUFBZSxXQUFXLEVBQUUsQ0FBQztBQUFBLDZEQUNsQyxjQUFjLFNBQVMsZ0JBQWdCLElBQUkscUJBQXFCLElBQUks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
