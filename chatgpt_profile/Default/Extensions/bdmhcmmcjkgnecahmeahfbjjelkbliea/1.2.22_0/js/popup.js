const backgroundAPI = {
  async trackPopupOpened(source = "popup") {
    return chrome.runtime.sendMessage({
      type: "TRACK_POPUP_OPENED",
      source
    });
  },
  async getDeviceId() {
    try {
      const response = await chrome.runtime.sendMessage({ type: "GET_DEVICE_ID" });
      if (response.status === "success") {
        return response.deviceId;
      } else {
        throw new Error(response.message || "Failed to get device ID");
      }
    } catch (error) {
      console.error("Error getting device ID:", error);
      throw error;
    }
  },
  trackAuthenticationRedirected(destination) {
    chrome.runtime.sendMessage({
      type: "TRACK_AUTHENTICATION_REDIRECTED",
      destination
    }).catch(() => {
    });
  },
  trackSignOut(userId) {
    chrome.runtime.sendMessage({
      type: "TRACK_SIGN_OUT",
      userId
    }).catch(() => {
    });
  },
  trackUpgradeClicked(source) {
    chrome.runtime.sendMessage({
      type: "TRACK_UPGRADE_CLICKED",
      source
    }).catch(() => {
    });
  },
  trackError(errorData) {
    chrome.runtime.sendMessage({
      type: "TRACK_ERROR",
      errorData
    }).catch(() => {
    });
  },
  async getAllMemories() {
    return chrome.runtime.sendMessage({
      type: "GET_ALL_MEMORIES"
    });
  },
  async deleteMemory(id, text = "") {
    return chrome.runtime.sendMessage({
      type: "DELETE_MEMORY",
      id,
      text
    });
  },
  async getMemoryLimitInfo() {
    return chrome.runtime.sendMessage({
      type: "GET_MEMORY_LIMIT_INFO"
    });
  },
  async signOut() {
    return chrome.runtime.sendMessage({
      type: "SIGN_OUT"
    });
  },
  async getCurrentUser(forceRefresh = false) {
    return chrome.runtime.sendMessage({
      type: "GET_CURRENT_USER",
      forceRefresh
    });
  },
  // Legacy methods for backward compatibility (deprecated)
  async getAuthState() {
    console.warn("getAuthState is deprecated, use getCurrentUser instead");
    return this.getCurrentUser();
  },
  async getSubscriptionStatus() {
    console.warn("getSubscriptionStatus is deprecated, use getCurrentUser instead");
    const response = await this.getCurrentUser();
    if (response.status === "success" && response.user) {
      return {
        status: "success",
        isPaid: response.user.isPaid || false,
        subscriptionType: response.user.subscriptionType || null,
        subscriptionExpiry: response.user.subscriptionExpiry || null
      };
    }
    return { status: "error", message: "User not authenticated" };
  },
  async saveMemory(text, tag = null) {
    return chrome.runtime.sendMessage({
      type: "SAVE_MEMORY",
      text,
      tag
    });
  },
  async editMemory(id, newText, originalText = "", tag = null) {
    return chrome.runtime.sendMessage({
      type: "EDIT_MEMORY",
      id,
      text: newText,
      originalText,
      tag
    });
  }
};
if (window.innerWidth <= 600) {
  document.body.classList.add("popup-mode");
}
let currentPage = 1;
const itemsPerPage = 50;
let totalPages = 1;
let allMemoriesData = [];
let fullMemoriesList = [];
let currentUserData = null;
function updateTagFilterDropdown(memories) {
  const filterSelect = document.getElementById("filter-tag");
  if (!filterSelect) return;
  const currentSelection = filterSelect.value;
  filterSelect.innerHTML = '<option value="all">All Tags</option>';
  const tags = /* @__PURE__ */ new Set();
  memories.forEach((m) => {
    if (m.tag) tags.add(m.tag);
  });
  const sortedTags = Array.from(tags).sort((a, b) => a.localeCompare(b));
  sortedTags.forEach((tag) => {
    const option = document.createElement("option");
    option.value = tag;
    option.textContent = tag;
    filterSelect.appendChild(option);
  });
  if (tags.has(currentSelection)) {
    filterSelect.value = currentSelection;
  } else {
    filterSelect.value = "all";
  }
}
function applyFiltersAndSorting() {
  var _a;
  const sortOrder = document.getElementById("sort-memories").value;
  const filterTag = ((_a = document.getElementById("filter-tag")) == null ? void 0 : _a.value) || "all";
  let filtered = [...fullMemoriesList];
  if (filterTag !== "all") {
    filtered = filtered.filter((m) => m.tag === filterTag);
  }
  allMemoriesData = filtered.sort((a, b) => {
    return sortOrder === "newest" ? b.timestamp - a.timestamp : a.timestamp - b.timestamp;
  });
  if (allMemoriesData.length === 0) {
    totalPages = 1;
    currentPage = 1;
    document.getElementById("memory-count").textContent = "No memories found";
  } else {
    totalPages = Math.ceil(allMemoriesData.length / itemsPerPage);
    currentPage = Math.min(currentPage, totalPages);
    document.getElementById("memory-count").textContent = `Total Memories: ${allMemoriesData.length}`;
  }
  updatePaginationControls();
  displayMemoriesPage(currentPage);
}
async function loadAllMemories() {
  console.log("Initiating GET_ALL_MEMORIES request");
  try {
    const response = await backgroundAPI.getAllMemories();
    console.log("Received response for GET_ALL_MEMORIES:", response);
    if (response && response.status === "success" && Array.isArray(response.memories)) {
      fullMemoriesList = response.memories;
      updateTagFilterDropdown(fullMemoriesList);
      applyFiltersAndSorting();
      await updateMemoryLimitBanner(currentUserData);
    } else {
      console.warn("Unexpected response structure:", response);
      throw new Error((response == null ? void 0 : response.message) || "Unknown error.");
    }
  } catch (error) {
    console.error("Error loading memories:", error);
    backgroundAPI.trackError({
      type: error.name || "Error",
      message: error.message || "Unknown error loading memories",
      stack: error.stack,
      context: "popup_load_memories",
      functionName: "loadAllMemories"
    });
    alert(`Error loading memories: ${error.message ?? "Unknown error"}`);
  }
}
async function deleteMemory(id, text) {
  console.log("Attempting to delete memory with ID:", id, "text:", text);
  try {
    const response = await backgroundAPI.deleteMemory(id, text);
    console.log("Received delete response:", response);
    if (response.status === "success") {
      const currentPageBeforeDelete = currentPage;
      await loadAllMemories();
      if (currentPageBeforeDelete <= totalPages) {
        currentPage = currentPageBeforeDelete;
        displayMemoriesPage(currentPage);
        updatePaginationControls();
      }
    }
  } catch (error) {
    console.error("Error deleting memory:", error);
    backgroundAPI.trackError({
      type: error.name || "Error",
      message: error.message || "Unknown error deleting memory",
      stack: error.stack,
      context: "popup_delete_memory",
      functionName: "deleteMemory",
      memoryId: id
    });
  }
}
function formatDate(timestamp) {
  return new Date(timestamp).toLocaleString();
}
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-button").forEach((btn) => btn.classList.remove("active"));
    button.classList.add("active");
    document.querySelectorAll(".tab-content").forEach((tab) => tab.classList.remove("active"));
    document.getElementById(`${button.dataset.tab}-tab`).classList.add("active");
    if (button.dataset.tab === "view") {
      loadAllMemories();
    }
  });
});
async function updateMemoryLimitBanner(user) {
  const memoryLimitBanner = document.getElementById("memory-limit-banner");
  const memoryLimitTitle = document.getElementById("memory-limit-title");
  const memoryLimitText = document.getElementById("memory-limit-text");
  if (!memoryLimitBanner || !memoryLimitTitle || !memoryLimitText) return;
  try {
    const response = await backgroundAPI.getMemoryLimitInfo();
    if (response.status === "success") {
      const { limit, current, userType } = response;
      if (userType === "paid") {
        memoryLimitBanner.classList.remove("hidden");
        const signinButton2 = document.getElementById("memory-limit-signin-button");
        const upgradeButton2 = document.getElementById("upgrade-button");
        const bannerIcon2 = memoryLimitBanner.querySelector("svg");
        const bannerTitle2 = document.getElementById("memory-limit-title");
        const bannerText2 = document.getElementById("memory-limit-text");
        memoryLimitBanner.className = "mb-4 p-4 bg-purple-50 border border-purple-200 rounded-lg";
        if (bannerTitle2) {
          bannerTitle2.className = "text-purple-900 text-sm font-semibold leading-relaxed mb-1";
        }
        if (bannerText2) {
          bannerText2.className = "text-purple-800 text-sm font-medium leading-relaxed";
        }
        if (bannerIcon2) {
          bannerIcon2.setAttribute("stroke", "#7c3aed");
          bannerIcon2.innerHTML = `
                       <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" stroke-linecap="round" stroke-linejoin="round"/>
                   `;
        }
        if (signinButton2) {
          signinButton2.style.display = "none";
        }
        if (upgradeButton2) {
          upgradeButton2.style.display = "none";
        }
        const title2 = "MaxMemory Pro active";
        const message2 = "You have unlimited memories and full access to your memory vault.";
        memoryLimitTitle.textContent = title2;
        memoryLimitText.textContent = message2;
        return;
      }
      memoryLimitBanner.classList.remove("hidden");
      const signinButton = document.getElementById("memory-limit-signin-button");
      const upgradeButton = document.getElementById("upgrade-button");
      const bannerButtonContainer = signinButton.parentElement;
      const bannerIcon = memoryLimitBanner.querySelector("svg");
      const bannerTitle = document.getElementById("memory-limit-title");
      const bannerText = document.getElementById("memory-limit-text");
      let title;
      let message;
      if (userType === "guest") {
        if (current >= limit) {
          message = `You have hit the guest limit at ${current}/${limit} memories. Sign in now to keep saving and unlock the full free tier.`;
        } else {
          message = `${current}/${limit} guest memories used. Sign in to unlock 100 free memories and sync them to your account.`;
        }
        title = "Sign In with Google for 100 free memories";
        memoryLimitBanner.className = "mb-4 p-4 bg-orange-50 border border-orange-200 rounded-lg";
        if (bannerTitle) {
          bannerTitle.className = "text-orange-900 text-sm font-semibold leading-relaxed mb-1";
        }
        if (bannerText) {
          bannerText.className = "text-orange-800 text-sm font-medium leading-relaxed";
        }
        if (bannerIcon) {
          bannerIcon.setAttribute("stroke", "#ea580c");
          bannerIcon.innerHTML = `
                           <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke-linecap="round" stroke-linejoin="round"/>
                           <line x1="12" y1="9" x2="12" y2="13" stroke-linecap="round" stroke-linejoin="round"/>
                           <line x1="12" y1="17" x2="12.01" y2="17" stroke-linecap="round" stroke-linejoin="round"/>
                       `;
        }
        if (signinButton) {
          signinButton.style.display = "flex";
          signinButton.className = "flex items-center gap-2 px-4 py-2 bg-white text-gray-700 text-sm font-medium rounded-md hover:bg-gray-50 border border-gray-300 transition-colors duration-200 shadow-sm";
          signinButton.innerHTML = `
                           <svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="16px" height="16px" viewBox="0 0 48 48"><g><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"></path><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.42-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"></path><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"></path><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"></path><path fill="none" d="M0 0h48v48H0z"></path></g></svg>
                           Sign In with Google
                       `;
        }
        if (upgradeButton && upgradeButton.parentElement !== bannerButtonContainer) {
          upgradeButton.style.display = "none";
        }
      } else if (userType === "logged_in") {
        if (current >= limit) {
          message = `You have hit your free limit at ${current}/${limit} memories. Upgrade to keep saving without interruptions.`;
        } else {
          message = `${current}/${limit} free memories used. Upgrade to Pro to remove the cap and keep your memory vault growing.`;
        }
        title = "Upgrade to Pro for unlimited memories";
        memoryLimitBanner.className = "mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg";
        if (bannerTitle) {
          bannerTitle.className = "text-blue-900 text-sm font-semibold leading-relaxed mb-1";
        }
        if (bannerText) {
          bannerText.className = "text-blue-800 text-sm font-medium leading-relaxed";
        }
        if (bannerIcon) {
          bannerIcon.setAttribute("stroke", "#2563eb");
          bannerIcon.innerHTML = `
                           <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" stroke-linecap="round" stroke-linejoin="round"/>
                       `;
        }
        if (signinButton) {
          signinButton.style.display = "none";
        }
        if (upgradeButton) {
          if (upgradeButton.parentElement !== bannerButtonContainer) {
            bannerButtonContainer.appendChild(upgradeButton);
          }
          upgradeButton.style.display = "flex";
          upgradeButton.className = "flex items-center gap-2 px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-md hover:bg-purple-700 transition-colors duration-200 shadow-sm";
        }
      }
      memoryLimitTitle.textContent = title;
      memoryLimitText.textContent = message;
    }
  } catch (error) {
    console.error("Error getting memory limit info:", error);
    backgroundAPI.trackError({
      type: error.name || "Error",
      message: error.message || "Unknown error getting memory limit info",
      stack: error.stack,
      context: "popup_memory_limit_banner",
      functionName: "updateMemoryLimitBanner"
    });
    memoryLimitBanner.classList.add("hidden");
  }
}
document.getElementById("sort-memories").addEventListener("change", loadAllMemories);
document.addEventListener("DOMContentLoaded", async function() {
  const signinButton = document.getElementById("signin-button");
  const signoutButton = document.getElementById("signout-button");
  const userProfile = document.getElementById("user-profile");
  const userGreeting = document.getElementById("user-greeting");
  if (!signoutButton) {
    console.error("Required DOM element signout-button not found");
    return;
  }
  if (!userProfile) {
    console.error("Required DOM element user-profile not found");
    return;
  }
  if (!userGreeting) {
    console.error("Required DOM element user-greeting not found");
    return;
  }
  document.addEventListener("visibilitychange", function() {
    if (document.visibilityState === "visible" && currentUserData && currentUserData.uid) {
      console.log("Popup became visible, refreshing user data");
      refreshCurrentUserData();
    }
  });
  window.addEventListener("focus", function() {
    if (currentUserData && currentUserData.uid && !isSigningOut) {
      console.log("Popup regained focus, refreshing user data");
      refreshCurrentUserData();
    }
  });
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "AUTH_STATE_CHANGED" && message.user) {
      console.log("Received auth state change:", message.user);
      if (currentUserData && currentUserData.uid === message.user.uid) {
        const wasProBefore = currentUserData.isPaid || false;
        const isProNow = message.user.isPaid || false;
        currentUserData = {
          ...currentUserData,
          isPaid: isProNow,
          subscriptionType: message.user.subscriptionType || null
        };
        if (wasProBefore !== isProNow) {
          console.log("Updating UI due to subscription change");
          updateAuthUI(currentUserData);
        }
      }
    }
  });
  if (signinButton) {
    signinButton.addEventListener("click", async () => {
      backgroundAPI.trackAuthenticationRedirected("webapp");
      const deviceId = await backgroundAPI.getDeviceId();
      chrome.tabs.create({
        url: `https://maxmemory.web.app/auth?source=extension&deviceId=${encodeURIComponent(deviceId)}`
      });
    });
  }
  let isSigningOut = false;
  signoutButton.addEventListener("click", () => {
    if (isSigningOut) return;
    isSigningOut = true;
    signoutButton.disabled = true;
    signoutButton.textContent = "Signing out...";
    if (currentUserData) {
      backgroundAPI.trackSignOut(currentUserData.uid);
    }
    backgroundAPI.signOut().then(async (response) => {
      if (response.status === "success") {
        currentUserData = null;
        await updateAuthUI(null);
      } else {
        console.error("Sign-out failed:", response.message);
      }
    }).catch((error) => {
      console.error(error);
      backgroundAPI.trackError({
        type: error.name || "Error",
        message: error.message || "Unknown error during sign out",
        stack: error.stack,
        context: "popup_sign_out",
        functionName: "signOutHandler"
      });
    }).finally(() => {
      isSigningOut = false;
      signoutButton.disabled = false;
      signoutButton.textContent = "Sign Out";
    });
  });
  function getInitialUserData() {
    if (isSigningOut) return;
    backgroundAPI.getCurrentUser(true).then(async (response) => {
      if (response.status === "success" && !isSigningOut) {
        currentUserData = response.user;
        await updateAuthUI(currentUserData);
      } else {
        console.log("No user data received or error occurred");
      }
    }).catch((error) => {
      console.error("Error getting current user:", error);
      backgroundAPI.trackError({
        type: error.name || "Error",
        message: error.message || "Unknown error getting current user",
        stack: error.stack,
        context: "popup_get_current_user",
        functionName: "getInitialUserData"
      });
    });
  }
  getInitialUserData();
  try {
    await backgroundAPI.trackPopupOpened("popup");
  } catch (error) {
    console.error("Error tracking popup opened:", error);
  }
  chrome.runtime.onMessage.addListener(async (message) => {
    if (message.type === "AUTH_STATE_CHANGED" && !isSigningOut) {
      currentUserData = message.user;
      await updateAuthUI(message.user);
    }
  });
  async function updateAuthUI(user) {
    await updateMemoryLimitBanner();
    updateSigninSection(user);
    updateUpgradeButton(user);
    if (user) {
      const username = user.displayName || user.email.split("@")[0];
      userGreeting.textContent = `Hi, ${username}`;
      userProfile.style.display = "flex";
      if (signinButton) {
        signinButton.style.display = "none";
      }
      const proBadge = document.getElementById("pro-badge");
      if (proBadge) {
        if (user.isPaid) {
          proBadge.classList.remove("hidden");
        } else {
          proBadge.classList.add("hidden");
        }
      }
    } else {
      userProfile.style.display = "none";
      if (signinButton) {
        signinButton.style.display = "flex";
      }
      const proBadge = document.getElementById("pro-badge");
      if (proBadge) {
        proBadge.classList.add("hidden");
      }
    }
  }
  function updateSigninSection(user) {
    const statusMessage = document.getElementById("backend-status-message");
    if (!statusMessage) {
      return;
    }
    if (user && user.uid) {
      statusMessage.classList.remove("hidden");
    } else {
      statusMessage.classList.add("hidden");
    }
  }
  function updateUpgradeButton(user) {
    var _a;
    const upgradeButton2 = document.getElementById("upgrade-button");
    if (!upgradeButton2) {
      return;
    }
    if (user && user.uid && !user.isPaid) {
      const bannerButtonContainer = (_a = document.getElementById("memory-limit-signin-button")) == null ? void 0 : _a.parentElement;
      if (upgradeButton2.parentElement !== bannerButtonContainer) {
        upgradeButton2.style.display = "flex";
      }
    } else {
      upgradeButton2.style.display = "none";
    }
  }
  async function refreshCurrentUserData() {
    if (!currentUserData || !currentUserData.uid || isSigningOut) return;
    try {
      const response = await backgroundAPI.getCurrentUser();
      if (response.status === "success" && response.user && !isSigningOut) {
        const wasProBefore = currentUserData.isPaid || false;
        const isProNow = response.user.isPaid || false;
        currentUserData = response.user;
        console.log("Refreshed user data:", currentUserData);
        if (wasProBefore !== isProNow) {
          console.log("Subscription status changed, updating UI");
          await updateAuthUI(currentUserData);
        }
      }
    } catch (error) {
      console.error("Error refreshing current user data:", error);
      backgroundAPI.trackError({
        type: error.name || "Error",
        message: error.message || "Unknown error refreshing user data",
        stack: error.stack,
        context: "popup_refresh_user_data",
        functionName: "refreshCurrentUserData"
      });
    }
  }
  document.getElementById("all-memories");
  document.getElementById("memory-count");
  const sortSelect = document.getElementById("sort-memories");
  const filterTagSelect = document.getElementById("filter-tag");
  sortSelect.addEventListener("change", () => {
    currentPage = 1;
    applyFiltersAndSorting();
  });
  if (filterTagSelect) {
    filterTagSelect.addEventListener("change", () => {
      currentPage = 1;
      applyFiltersAndSorting();
    });
  }
  const addMemoryButton = document.getElementById("add-memory-button");
  const addMemorySection = document.getElementById("add-memory-section");
  const newMemoryInput = document.getElementById("new-memory-input");
  const cancelAddMemory = document.getElementById("cancel-add-memory");
  const confirmAddMemory = document.getElementById("confirm-add-memory");
  addMemoryButton.addEventListener("click", () => {
    addMemorySection.style.display = "block";
    addMemoryButton.style.display = "none";
    newMemoryInput.focus();
  });
  function hideAddMemorySection() {
    addMemorySection.style.display = "none";
    addMemoryButton.style.display = "flex";
    newMemoryInput.value = "";
    const tagInput = document.getElementById("new-memory-tag");
    if (tagInput) tagInput.value = "";
  }
  cancelAddMemory.addEventListener("click", hideAddMemorySection);
  confirmAddMemory.addEventListener("click", async () => {
    var _a;
    const text = newMemoryInput.value.trim();
    const tag = ((_a = document.getElementById("new-memory-tag")) == null ? void 0 : _a.value.trim()) || null;
    if (!text) return;
    try {
      const limitResponse = await backgroundAPI.getMemoryLimitInfo();
      if (limitResponse.status === "success" && !limitResponse.canAdd) {
        let message;
        if (limitResponse.userType === "guest") {
          message = `You've reached the ${limitResponse.limit} memory limit for guest users. Sign in for free to get 100 memories!`;
        } else if (limitResponse.userType === "logged_in") {
          message = `You've reached the ${limitResponse.limit} memory limit. Upgrade to Pro for unlimited memories!`;
        }
        alert(message);
        return;
      }
    } catch (error) {
      console.error("Error checking memory limit:", error);
      backgroundAPI.trackError({
        type: error.name || "Error",
        message: error.message || "Unknown error checking memory limit",
        stack: error.stack,
        context: "popup_add_memory_limit_check",
        functionName: "confirmAddMemoryHandler"
      });
    }
    confirmAddMemory.disabled = true;
    confirmAddMemory.innerHTML = `
            <svg class="spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 2a10 10 0 0 1 10 10"></path>
            </svg>
        `;
    try {
      const response = await backgroundAPI.saveMemory(text, tag);
      if (response.status === "success") {
        hideAddMemorySection();
        currentPage = 1;
        await loadAllMemories();
      } else {
        throw new Error(response.message || "Failed to save memory");
      }
    } catch (error) {
      console.error("Error saving memory:", error);
      backgroundAPI.trackError({
        type: error.name || "Error",
        message: error.message || "Unknown error saving memory",
        stack: error.stack,
        context: "popup_save_memory",
        functionName: "confirmAddMemoryHandler"
      });
    } finally {
      confirmAddMemory.disabled = false;
      confirmAddMemory.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M5 13l4 4L19 7" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            `;
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && addMemorySection.style.display === "block") {
      hideAddMemorySection();
    }
  });
  const upgradeButton = document.getElementById("upgrade-button");
  if (upgradeButton) {
    upgradeButton.addEventListener("click", () => {
      backgroundAPI.trackUpgradeClicked("popup");
      chrome.tabs.create({
        url: "https://maxmemory.web.app/pricing?source=extension"
      });
    });
  }
  document.getElementById("prev-page").addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      displayMemoriesPage(currentPage);
      updatePaginationControls();
    }
  });
  document.getElementById("next-page").addEventListener("click", () => {
    if (currentPage < totalPages) {
      currentPage++;
      displayMemoriesPage(currentPage);
      updatePaginationControls();
    }
  });
  document.getElementById("sort-memories").addEventListener("change", () => {
    currentPage = 1;
    loadAllMemories();
  });
  const memoryLimitSigninButton = document.getElementById("memory-limit-signin-button");
  if (memoryLimitSigninButton) {
    memoryLimitSigninButton.addEventListener("click", async () => {
      try {
        await backgroundAPI.trackPopupOpened("memory_limit_warning");
        const response = await backgroundAPI.getMemoryLimitInfo();
        if (response.status === "success") {
          if (response.userType === "guest") {
            backgroundAPI.trackAuthenticationRedirected("webapp");
            const deviceId = await backgroundAPI.getDeviceId();
            chrome.tabs.create({
              url: `https://maxmemory.web.app/auth?source=extension&reason=memory_limit&deviceId=${encodeURIComponent(deviceId)}`
            });
          } else if (response.userType === "logged_in") {
            backgroundAPI.trackUpgradeClicked("memory_limit_banner");
            chrome.tabs.create({ url: "https://maxmemory.web.app/pricing?source=extension&reason=memory_limit" });
          }
        }
      } catch (error) {
        console.error("Error handling button click:", error);
        const deviceId = await backgroundAPI.getDeviceId();
        chrome.tabs.create({
          url: `https://maxmemory.web.app/auth?source=extension&reason=memory_limit&deviceId=${encodeURIComponent(deviceId)}`
        });
      }
    });
  }
  loadAllMemories();
});
async function saveEdit(id, textElement, editButton, originalText, newTag = null, tagEditInput = null) {
  const newText = textElement.textContent.trim();
  if (!newText) {
    alert("Memory text cannot be empty.");
    return;
  }
  try {
    const response = await backgroundAPI.editMemory(id, newText, originalText, newTag);
    if (response.status === "success") {
      textElement.setAttribute("contenteditable", "false");
      textElement.classList.remove("bg-gray-50", "border", "border-gray-300", "rounded", "p-2");
      editButton.disabled = false;
      if (tagEditInput) {
        tagEditInput.classList.add("hidden");
      }
      const fullIndex = fullMemoriesList.findIndex((m) => m.id === id);
      if (fullIndex !== -1) {
        fullMemoriesList[fullIndex].memory_text = newText;
        fullMemoriesList[fullIndex].tag = newTag || null;
      }
      const displayIndex = allMemoriesData.findIndex((m) => m.id === id);
      if (displayIndex !== -1) {
        allMemoriesData[displayIndex].memory_text = newText;
        allMemoriesData[displayIndex].tag = newTag || null;
      }
      updateTagFilterDropdown(fullMemoriesList);
      applyFiltersAndSorting();
      editButton.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
            `;
    } else {
      throw new Error(response.message || "Failed to save changes.");
    }
  } catch (error) {
    console.error("Error saving edit:", error);
    backgroundAPI.trackError({
      type: error.name || "Error",
      message: error.message || "Unknown error editing memory",
      stack: error.stack,
      context: "popup_edit_memory",
      functionName: "saveEdit",
      memoryId: id
    });
    editButton.disabled = false;
    editButton.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 13l4 4L19 7"></path>
            </svg>
        `;
  }
}
function displayMemoriesPage(page) {
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const memoriesToDisplay = allMemoriesData.slice(startIndex, endIndex);
  const container = document.getElementById("all-memories");
  if (allMemoriesData.length === 0) {
    const emptyStateTemplate = document.getElementById("empty-state-template");
    if (emptyStateTemplate) {
      container.innerHTML = emptyStateTemplate.innerHTML;
      const emptyStateButton = container.querySelector("#empty-state-add-button");
      if (emptyStateButton) {
        emptyStateButton.addEventListener("click", () => {
          const addMemoryButton = document.getElementById("add-memory-button");
          const addMemorySection = document.getElementById("add-memory-section");
          addMemorySection.style.display = "block";
          addMemoryButton.style.display = "none";
          document.getElementById("new-memory-input").focus();
        });
      }
    }
    return;
  }
  const memoryCardTemplate = document.getElementById("memory-card-template");
  if (!memoryCardTemplate) {
    console.error("Memory card template not found");
    return;
  }
  container.innerHTML = "";
  const groupedMemories = {};
  memoriesToDisplay.forEach((memory) => {
    const tag = memory.tag || "General";
    if (!groupedMemories[tag]) {
      groupedMemories[tag] = [];
    }
    groupedMemories[tag].push(memory);
  });
  const tagNames = Object.keys(groupedMemories).sort((a, b) => {
    if (a === "General") return 1;
    if (b === "General") return -1;
    return a.localeCompare(b);
  });
  tagNames.forEach((tagName) => {
    const memoriesInGroup = groupedMemories[tagName];
    const sectionDiv = document.createElement("div");
    sectionDiv.className = "tag-group-section mb-4 border border-gray-200/50 rounded-xl overflow-hidden bg-gray-50/20 shadow-sm";
    const headerDiv = document.createElement("div");
    headerDiv.className = "tag-group-header flex items-center justify-between px-3 py-2 bg-gray-100/70 hover:bg-gray-100 transition-colors cursor-pointer select-none border-b border-gray-200/30";
    headerDiv.innerHTML = `
            <div class="flex items-center gap-2">
                <svg class="chevron-icon transform transition-transform duration-200 w-3 h-3 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="transform: rotate(90deg);">
                    <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
                <span class="font-bold text-xs text-gray-700">${tagName}</span>
                <span class="text-[10px] text-gray-500 bg-white px-2 py-0.5 rounded-full border border-gray-200/60 font-semibold">${memoriesInGroup.length}</span>
            </div>
        `;
    const contentDiv = document.createElement("div");
    contentDiv.className = "tag-group-content p-3 flex flex-col gap-2 transition-all duration-200";
    headerDiv.addEventListener("click", () => {
      const chevron = headerDiv.querySelector(".chevron-icon");
      if (contentDiv.style.display === "none") {
        contentDiv.style.display = "flex";
        chevron.style.transform = "rotate(90deg)";
      } else {
        contentDiv.style.display = "none";
        chevron.style.transform = "rotate(0deg)";
      }
    });
    memoriesInGroup.forEach((memory) => {
      const cardElement = memoryCardTemplate.cloneNode(true);
      cardElement.id = "";
      cardElement.classList.remove("hidden");
      const memoryText = cardElement.querySelector("span[contenteditable]");
      const memoryDate = cardElement.querySelector(".memory-date");
      const newTagBadge = cardElement.querySelector("#new-tag-template");
      const editButton = cardElement.querySelector('button[title="Edit"]');
      const deleteButton = cardElement.querySelector('button[title="Delete"]');
      const tagBadge = cardElement.querySelector(".memory-tag-badge");
      const tagEditInput = cardElement.querySelector(".memory-tag-edit-input");
      if (memoryText) memoryText.textContent = memory.memory_text;
      if (memoryDate) memoryDate.textContent = formatDate(memory.timestamp);
      if (tagBadge) {
        if (memory.tag) {
          tagBadge.textContent = memory.tag;
          tagBadge.classList.remove("hidden");
        } else {
          tagBadge.classList.add("hidden");
        }
      }
      if (newTagBadge) {
        const uniqueTagId = `new-tag-${memory.id}`;
        newTagBadge.id = uniqueTagId;
        newTagBadge.classList.add("hidden");
        const now = Date.now();
        const timeDiff = now - memory.timestamp;
        const thirtyMinutes = 30 * 60 * 1e3;
        const isRecent = timeDiff < thirtyMinutes;
        if (isRecent) {
          newTagBadge.classList.remove("hidden");
        }
      }
      if (editButton) editButton.dataset.id = memory.id;
      if (deleteButton) deleteButton.dataset.id = memory.id;
      if (editButton) {
        editButton.addEventListener("click", () => {
          const textElement = cardElement.querySelector("span[contenteditable]");
          if (textElement.getAttribute("contenteditable") === "true") {
            const newTag = tagEditInput ? tagEditInput.value.trim() : null;
            saveEdit(memory.id, textElement, editButton, memory.memory_text, newTag, tagEditInput);
          } else {
            textElement.setAttribute("contenteditable", "true");
            textElement.classList.add("bg-gray-50", "border", "border-gray-300", "rounded", "p-2");
            textElement.focus();
            if (tagEditInput) {
              tagEditInput.value = memory.tag || "";
              tagEditInput.classList.remove("hidden");
            }
            editButton.innerHTML = `
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M5 13l4 4L19 7"></path>
                            </svg>
                        `;
          }
        });
      }
      if (deleteButton) {
        deleteButton.addEventListener("click", () => {
          console.log("Delete button clicked for memory ID:", memory.id);
          deleteMemory(memory.id, memory.memory_text);
        });
      }
      contentDiv.appendChild(cardElement);
    });
    sectionDiv.appendChild(headerDiv);
    sectionDiv.appendChild(contentDiv);
    container.appendChild(sectionDiv);
  });
}
function updatePaginationControls() {
  const prevButton = document.getElementById("prev-page");
  const nextButton = document.getElementById("next-page");
  const pageInfo = document.getElementById("page-info");
  const paginationControls = document.getElementById("pagination-controls");
  if (allMemoriesData.length === 0) {
    paginationControls.style.display = "none";
    return;
  } else {
    paginationControls.style.display = "flex";
  }
  prevButton.disabled = currentPage === 1;
  nextButton.disabled = currentPage === totalPages;
  pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
}
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9wdXAuanMiLCJzb3VyY2VzIjpbIi4uLy4uL2pzL3BvcHVwLmpzIl0sInNvdXJjZXNDb250ZW50IjpbIi8vIHBvcHVwLmpzIFxyXG5pbXBvcnQgJy4uL2Nzcy9wb3B1cC5jc3MnO1xyXG5cclxuLy8gQ2VudHJhbGl6ZWQgYmFja2dyb3VuZCBBUEkgZnVuY3Rpb25zIHRoYXQgc2VuZCBtZXNzYWdlcyB0byBiYWNrZ3JvdW5kLmpzXHJcbmNvbnN0IGJhY2tncm91bmRBUEkgPSB7XHJcbiAgICBhc3luYyB0cmFja1BvcHVwT3BlbmVkKHNvdXJjZSA9ICdwb3B1cCcpIHtcclxuICAgICAgICByZXR1cm4gY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyBcclxuICAgICAgICAgICAgdHlwZTogJ1RSQUNLX1BPUFVQX09QRU5FRCcsXHJcbiAgICAgICAgICAgIHNvdXJjZTogc291cmNlXHJcbiAgICAgICAgfSk7XHJcbiAgICB9LFxyXG5cclxuICAgIGFzeW5jIGdldERldmljZUlkKCkge1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyB0eXBlOiAnR0VUX0RFVklDRV9JRCcgfSk7XHJcbiAgICAgICAgICAgIGlmIChyZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJykge1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuIHJlc3BvbnNlLmRldmljZUlkO1xyXG4gICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKHJlc3BvbnNlLm1lc3NhZ2UgfHwgJ0ZhaWxlZCB0byBnZXQgZGV2aWNlIElEJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBnZXR0aW5nIGRldmljZSBJRDonLCBlcnJvcik7XHJcbiAgICAgICAgICAgIHRocm93IGVycm9yO1xyXG4gICAgICAgIH1cclxuICAgIH0sXHJcblxyXG4gICAgdHJhY2tBdXRoZW50aWNhdGlvblJlZGlyZWN0ZWQoZGVzdGluYXRpb24pIHtcclxuICAgICAgICBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7IFxyXG4gICAgICAgICAgICB0eXBlOiAnVFJBQ0tfQVVUSEVOVElDQVRJT05fUkVESVJFQ1RFRCcsIFxyXG4gICAgICAgICAgICBkZXN0aW5hdGlvbiBcclxuICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICB9LFxyXG5cclxuICAgIHRyYWNrU2lnbk91dCh1c2VySWQpIHtcclxuICAgICAgICBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7IFxyXG4gICAgICAgICAgICB0eXBlOiAnVFJBQ0tfU0lHTl9PVVQnLCBcclxuICAgICAgICAgICAgdXNlcklkIFxyXG4gICAgICAgIH0pLmNhdGNoKCgpID0+IHt9KTtcclxuICAgIH0sXHJcblxyXG4gICAgdHJhY2tVcGdyYWRlQ2xpY2tlZChzb3VyY2UpIHtcclxuICAgICAgICBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7IFxyXG4gICAgICAgICAgICB0eXBlOiAnVFJBQ0tfVVBHUkFERV9DTElDS0VEJywgXHJcbiAgICAgICAgICAgIHNvdXJjZSBcclxuICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICB9LFxyXG5cclxuICAgIHRyYWNrRXJyb3IoZXJyb3JEYXRhKSB7XHJcbiAgICAgICAgY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyBcclxuICAgICAgICAgICAgdHlwZTogJ1RSQUNLX0VSUk9SJywgXHJcbiAgICAgICAgICAgIGVycm9yRGF0YSBcclxuICAgICAgICB9KS5jYXRjaCgoKSA9PiB7fSk7XHJcbiAgICB9LFxyXG5cclxuICAgIGFzeW5jIGdldEFsbE1lbW9yaWVzKCkge1xyXG4gICAgICAgIHJldHVybiBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7IFxyXG4gICAgICAgICAgICB0eXBlOiAnR0VUX0FMTF9NRU1PUklFUydcclxuICAgICAgICB9KTtcclxuICAgIH0sXHJcblxyXG4gICAgYXN5bmMgZGVsZXRlTWVtb3J5KGlkLCB0ZXh0ID0gJycpIHtcclxuICAgICAgICByZXR1cm4gY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyBcclxuICAgICAgICAgICAgdHlwZTogJ0RFTEVURV9NRU1PUlknLCBcclxuICAgICAgICAgICAgaWQ6IGlkLFxyXG4gICAgICAgICAgICB0ZXh0OiB0ZXh0XHJcbiAgICAgICAgfSk7XHJcbiAgICB9LFxyXG5cclxuICAgIGFzeW5jIGdldE1lbW9yeUxpbWl0SW5mbygpIHtcclxuICAgICAgICByZXR1cm4gY2hyb21lLnJ1bnRpbWUuc2VuZE1lc3NhZ2UoeyBcclxuICAgICAgICAgICAgdHlwZTogJ0dFVF9NRU1PUllfTElNSVRfSU5GTycgXHJcbiAgICAgICAgfSk7XHJcbiAgICB9LFxyXG5cclxuICAgIGFzeW5jIHNpZ25PdXQoKSB7XHJcbiAgICAgICAgcmV0dXJuIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHsgXHJcbiAgICAgICAgICAgIHR5cGU6ICdTSUdOX09VVCcgXHJcbiAgICAgICAgfSk7XHJcbiAgICB9LFxyXG5cclxuICAgIGFzeW5jIGdldEN1cnJlbnRVc2VyKGZvcmNlUmVmcmVzaCA9IGZhbHNlKSB7XHJcbiAgICAgICAgcmV0dXJuIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHsgXHJcbiAgICAgICAgICAgIHR5cGU6ICdHRVRfQ1VSUkVOVF9VU0VSJyxcclxuICAgICAgICAgICAgZm9yY2VSZWZyZXNoOiBmb3JjZVJlZnJlc2hcclxuICAgICAgICB9KTtcclxuICAgIH0sXHJcblxyXG4gICAgLy8gTGVnYWN5IG1ldGhvZHMgZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgKGRlcHJlY2F0ZWQpXHJcbiAgICBhc3luYyBnZXRBdXRoU3RhdGUoKSB7XHJcbiAgICAgICAgY29uc29sZS53YXJuKCdnZXRBdXRoU3RhdGUgaXMgZGVwcmVjYXRlZCwgdXNlIGdldEN1cnJlbnRVc2VyIGluc3RlYWQnKTtcclxuICAgICAgICByZXR1cm4gdGhpcy5nZXRDdXJyZW50VXNlcigpO1xyXG4gICAgfSxcclxuXHJcbiAgICBhc3luYyBnZXRTdWJzY3JpcHRpb25TdGF0dXMoKSB7XHJcbiAgICAgICAgY29uc29sZS53YXJuKCdnZXRTdWJzY3JpcHRpb25TdGF0dXMgaXMgZGVwcmVjYXRlZCwgdXNlIGdldEN1cnJlbnRVc2VyIGluc3RlYWQnKTtcclxuICAgICAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IHRoaXMuZ2V0Q3VycmVudFVzZXIoKTtcclxuICAgICAgICBpZiAocmVzcG9uc2Uuc3RhdHVzID09PSAnc3VjY2VzcycgJiYgcmVzcG9uc2UudXNlcikge1xyXG4gICAgICAgICAgICByZXR1cm4ge1xyXG4gICAgICAgICAgICAgICAgc3RhdHVzOiAnc3VjY2VzcycsXHJcbiAgICAgICAgICAgICAgICBpc1BhaWQ6IHJlc3BvbnNlLnVzZXIuaXNQYWlkIHx8IGZhbHNlLFxyXG4gICAgICAgICAgICAgICAgc3Vic2NyaXB0aW9uVHlwZTogcmVzcG9uc2UudXNlci5zdWJzY3JpcHRpb25UeXBlIHx8IG51bGwsXHJcbiAgICAgICAgICAgICAgICBzdWJzY3JpcHRpb25FeHBpcnk6IHJlc3BvbnNlLnVzZXIuc3Vic2NyaXB0aW9uRXhwaXJ5IHx8IG51bGxcclxuICAgICAgICAgICAgfTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHsgc3RhdHVzOiAnZXJyb3InLCBtZXNzYWdlOiAnVXNlciBub3QgYXV0aGVudGljYXRlZCcgfTtcclxuICAgIH0sXHJcblxyXG4gICAgYXN5bmMgc2F2ZU1lbW9yeSh0ZXh0LCB0YWcgPSBudWxsKSB7XHJcbiAgICAgICAgcmV0dXJuIGNocm9tZS5ydW50aW1lLnNlbmRNZXNzYWdlKHsgXHJcbiAgICAgICAgICAgIHR5cGU6ICdTQVZFX01FTU9SWScsIFxyXG4gICAgICAgICAgICB0ZXh0OiB0ZXh0LFxyXG4gICAgICAgICAgICB0YWc6IHRhZ1xyXG4gICAgICAgIH0pO1xyXG4gICAgfSxcclxuXHJcbiAgICBhc3luYyBlZGl0TWVtb3J5KGlkLCBuZXdUZXh0LCBvcmlnaW5hbFRleHQgPSAnJywgdGFnID0gbnVsbCkge1xyXG4gICAgICAgIHJldHVybiBjaHJvbWUucnVudGltZS5zZW5kTWVzc2FnZSh7IFxyXG4gICAgICAgICAgICB0eXBlOiAnRURJVF9NRU1PUlknLCBcclxuICAgICAgICAgICAgaWQ6IGlkLFxyXG4gICAgICAgICAgICB0ZXh0OiBuZXdUZXh0LFxyXG4gICAgICAgICAgICBvcmlnaW5hbFRleHQ6IG9yaWdpbmFsVGV4dCxcclxuICAgICAgICAgICAgdGFnOiB0YWdcclxuICAgICAgICB9KTtcclxuICAgIH1cclxufTtcclxuXHJcblxyXG5cclxuLy8gQWRkIHRoaXMgYXQgdGhlIHRvcCBvZiB0aGUgZmlsZVxyXG5pZiAod2luZG93LmlubmVyV2lkdGggPD0gNjAwKSB7IC8vIFBvcHVwIG1vZGUgZGV0ZWN0aW9uXHJcbiAgZG9jdW1lbnQuYm9keS5jbGFzc0xpc3QuYWRkKCdwb3B1cC1tb2RlJyk7XHJcbn1cclxuXHJcbi8vIEFkZCB0aGVzZSB2YXJpYWJsZXMgYXQgdGhlIHRvcCBvZiB0aGUgZmlsZVxyXG5sZXQgY3VycmVudFBhZ2UgPSAxO1xyXG5jb25zdCBpdGVtc1BlclBhZ2UgPSA1MDtcclxubGV0IHRvdGFsUGFnZXMgPSAxO1xyXG5sZXQgYWxsTWVtb3JpZXNEYXRhID0gW107XHJcbmxldCBmdWxsTWVtb3JpZXNMaXN0ID0gW107XHJcbmxldCBjdXJyZW50VXNlckRhdGEgPSBudWxsO1xyXG5cclxuXHJcblxyXG4vLyBGdW5jdGlvbiB0byB1cGRhdGUgdGhlIHRhZyBmaWx0ZXIgc2VsZWN0IGRyb3Bkb3duXHJcbmZ1bmN0aW9uIHVwZGF0ZVRhZ0ZpbHRlckRyb3Bkb3duKG1lbW9yaWVzKSB7XHJcbiAgICBjb25zdCBmaWx0ZXJTZWxlY3QgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmlsdGVyLXRhZycpO1xyXG4gICAgaWYgKCFmaWx0ZXJTZWxlY3QpIHJldHVybjtcclxuICAgIFxyXG4gICAgLy8gU3RvcmUgY3VycmVudCBzZWxlY3Rpb25cclxuICAgIGNvbnN0IGN1cnJlbnRTZWxlY3Rpb24gPSBmaWx0ZXJTZWxlY3QudmFsdWU7XHJcbiAgICBcclxuICAgIC8vIFJlc2V0IHRvIFwiQWxsXCJcclxuICAgIGZpbHRlclNlbGVjdC5pbm5lckhUTUwgPSAnPG9wdGlvbiB2YWx1ZT1cImFsbFwiPkFsbCBUYWdzPC9vcHRpb24+JztcclxuICAgIFxyXG4gICAgLy8gR2V0IHVuaXF1ZSB0YWdzXHJcbiAgICBjb25zdCB0YWdzID0gbmV3IFNldCgpO1xyXG4gICAgbWVtb3JpZXMuZm9yRWFjaChtID0+IHtcclxuICAgICAgICBpZiAobS50YWcpIHRhZ3MuYWRkKG0udGFnKTtcclxuICAgIH0pO1xyXG4gICAgXHJcbiAgICAvLyBTb3J0IGFscGhhYmV0aWNhbGx5XHJcbiAgICBjb25zdCBzb3J0ZWRUYWdzID0gQXJyYXkuZnJvbSh0YWdzKS5zb3J0KChhLCBiKSA9PiBhLmxvY2FsZUNvbXBhcmUoYikpO1xyXG4gICAgXHJcbiAgICBzb3J0ZWRUYWdzLmZvckVhY2godGFnID0+IHtcclxuICAgICAgICBjb25zdCBvcHRpb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdvcHRpb24nKTtcclxuICAgICAgICBvcHRpb24udmFsdWUgPSB0YWc7XHJcbiAgICAgICAgb3B0aW9uLnRleHRDb250ZW50ID0gdGFnO1xyXG4gICAgICAgIGZpbHRlclNlbGVjdC5hcHBlbmRDaGlsZChvcHRpb24pO1xyXG4gICAgfSk7XHJcbiAgICBcclxuICAgIC8vIFJlc3RvcmUgc2VsZWN0aW9uIGlmIGl0IHN0aWxsIGV4aXN0c1xyXG4gICAgaWYgKHRhZ3MuaGFzKGN1cnJlbnRTZWxlY3Rpb24pKSB7XHJcbiAgICAgICAgZmlsdGVyU2VsZWN0LnZhbHVlID0gY3VycmVudFNlbGVjdGlvbjtcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgICAgZmlsdGVyU2VsZWN0LnZhbHVlID0gJ2FsbCc7XHJcbiAgICB9XHJcbn1cclxuXHJcbi8vIEZ1bmN0aW9uIHRvIGFwcGx5IGZpbHRlcnMgYW5kIHNvcnRpbmcgbG9jYWxseSB3aXRob3V0IG5ldHdvcmsgcm91bmR0cmlwXHJcbmZ1bmN0aW9uIGFwcGx5RmlsdGVyc0FuZFNvcnRpbmcoKSB7XHJcbiAgICBjb25zdCBzb3J0T3JkZXIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc29ydC1tZW1vcmllcycpLnZhbHVlO1xyXG4gICAgY29uc3QgZmlsdGVyVGFnID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2ZpbHRlci10YWcnKT8udmFsdWUgfHwgJ2FsbCc7XHJcbiAgICBcclxuICAgIC8vIEZpbHRlclxyXG4gICAgbGV0IGZpbHRlcmVkID0gWy4uLmZ1bGxNZW1vcmllc0xpc3RdO1xyXG4gICAgaWYgKGZpbHRlclRhZyAhPT0gJ2FsbCcpIHtcclxuICAgICAgICBmaWx0ZXJlZCA9IGZpbHRlcmVkLmZpbHRlcihtID0+IG0udGFnID09PSBmaWx0ZXJUYWcpO1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICAvLyBTb3J0XHJcbiAgICBhbGxNZW1vcmllc0RhdGEgPSBmaWx0ZXJlZC5zb3J0KChhLCBiKSA9PiB7XHJcbiAgICAgICAgcmV0dXJuIHNvcnRPcmRlciA9PT0gJ25ld2VzdCdcclxuICAgICAgICAgICAgPyBiLnRpbWVzdGFtcCAtIGEudGltZXN0YW1wXHJcbiAgICAgICAgICAgIDogYS50aW1lc3RhbXAgLSBiLnRpbWVzdGFtcDtcclxuICAgIH0pO1xyXG5cclxuICAgIC8vIEhhbmRsZSBlbXB0eSBtZW1vcmllcyBhcnJheVxyXG4gICAgaWYgKGFsbE1lbW9yaWVzRGF0YS5sZW5ndGggPT09IDApIHtcclxuICAgICAgICB0b3RhbFBhZ2VzID0gMTtcclxuICAgICAgICBjdXJyZW50UGFnZSA9IDE7XHJcbiAgICAgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1jb3VudCcpLnRleHRDb250ZW50ID0gJ05vIG1lbW9yaWVzIGZvdW5kJztcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgICAgdG90YWxQYWdlcyA9IE1hdGguY2VpbChhbGxNZW1vcmllc0RhdGEubGVuZ3RoIC8gaXRlbXNQZXJQYWdlKTtcclxuICAgICAgICBjdXJyZW50UGFnZSA9IE1hdGgubWluKGN1cnJlbnRQYWdlLCB0b3RhbFBhZ2VzKTtcclxuICAgICAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWVtb3J5LWNvdW50JykudGV4dENvbnRlbnQgPSBgVG90YWwgTWVtb3JpZXM6ICR7YWxsTWVtb3JpZXNEYXRhLmxlbmd0aH1gO1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICB1cGRhdGVQYWdpbmF0aW9uQ29udHJvbHMoKTtcclxuICAgIGRpc3BsYXlNZW1vcmllc1BhZ2UoY3VycmVudFBhZ2UpO1xyXG59XHJcblxyXG4vLyBGdW5jdGlvbiB0byBsb2FkIGFuZCBkaXNwbGF5IGFsbCBtZW1vcmllcyBieSBzZW5kaW5nIGEgbWVzc2FnZSB0byBiYWNrZ3JvdW5kLmpzXHJcbmFzeW5jIGZ1bmN0aW9uIGxvYWRBbGxNZW1vcmllcygpIHtcclxuICAgIGNvbnNvbGUubG9nKCdJbml0aWF0aW5nIEdFVF9BTExfTUVNT1JJRVMgcmVxdWVzdCcpO1xyXG5cclxuICAgIHRyeSB7XHJcbiAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldEFsbE1lbW9yaWVzKCk7XHJcblxyXG4gICAgICAgIGNvbnNvbGUubG9nKCdSZWNlaXZlZCByZXNwb25zZSBmb3IgR0VUX0FMTF9NRU1PUklFUzonLCByZXNwb25zZSk7XHJcblxyXG4gICAgICAgIGlmIChyZXNwb25zZSAmJiByZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJyAmJiBBcnJheS5pc0FycmF5KHJlc3BvbnNlLm1lbW9yaWVzKSkge1xyXG4gICAgICAgICAgICBmdWxsTWVtb3JpZXNMaXN0ID0gcmVzcG9uc2UubWVtb3JpZXM7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBQb3B1bGF0ZSB0aGUgZHJvcGRvd25cclxuICAgICAgICAgICAgdXBkYXRlVGFnRmlsdGVyRHJvcGRvd24oZnVsbE1lbW9yaWVzTGlzdCk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBBcHBseSBzb3J0aW5nIGFuZCBmaWx0ZXJpbmdcclxuICAgICAgICAgICAgYXBwbHlGaWx0ZXJzQW5kU29ydGluZygpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gVXBkYXRlIG1lbW9yeSBsaW1pdCBiYW5uZXIgYWZ0ZXIgbG9hZGluZyBtZW1vcmllc1xyXG4gICAgICAgICAgICBhd2FpdCB1cGRhdGVNZW1vcnlMaW1pdEJhbm5lcihjdXJyZW50VXNlckRhdGEpO1xyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUud2FybignVW5leHBlY3RlZCByZXNwb25zZSBzdHJ1Y3R1cmU6JywgcmVzcG9uc2UpO1xyXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IocmVzcG9uc2U/Lm1lc3NhZ2UgfHwgJ1Vua25vd24gZXJyb3IuJyk7XHJcbiAgICAgICAgfVxyXG4gICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBsb2FkaW5nIG1lbW9yaWVzOicsIGVycm9yKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBUcmFjayBlcnJvciBpbiBNaXhwYW5lbFxyXG4gICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tFcnJvcih7XHJcbiAgICAgICAgICAgIHR5cGU6IGVycm9yLm5hbWUgfHwgJ0Vycm9yJyxcclxuICAgICAgICAgICAgbWVzc2FnZTogZXJyb3IubWVzc2FnZSB8fCAnVW5rbm93biBlcnJvciBsb2FkaW5nIG1lbW9yaWVzJyxcclxuICAgICAgICAgICAgc3RhY2s6IGVycm9yLnN0YWNrLFxyXG4gICAgICAgICAgICBjb250ZXh0OiAncG9wdXBfbG9hZF9tZW1vcmllcycsXHJcbiAgICAgICAgICAgIGZ1bmN0aW9uTmFtZTogJ2xvYWRBbGxNZW1vcmllcydcclxuICAgICAgICB9KTtcclxuICAgICAgICBcclxuICAgICAgICBhbGVydChgRXJyb3IgbG9hZGluZyBtZW1vcmllczogJHtlcnJvci5tZXNzYWdlID8/ICdVbmtub3duIGVycm9yJ31gKTtcclxuICAgIH1cclxufVxyXG5cclxuXHJcbi8vIEZ1bmN0aW9uIHRvIGRlbGV0ZSBhIG1lbW9yeVxyXG5hc3luYyBmdW5jdGlvbiBkZWxldGVNZW1vcnkoaWQsIHRleHQpIHtcclxuICAgIGNvbnNvbGUubG9nKCdBdHRlbXB0aW5nIHRvIGRlbGV0ZSBtZW1vcnkgd2l0aCBJRDonLCBpZCwgJ3RleHQ6JywgdGV4dCk7XHJcbiAgICBcclxuICAgIC8vIFRyYWNrIG1lbW9yeSBkZWxldGlvblxyXG4gICAgLy8gTWVtb3J5IGRlbGV0aW9uIHRyYWNraW5nIGhhbmRsZWQgYnkgYmFja2VuZFxyXG4gICAgXHJcbiAgICB0cnkge1xyXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgYmFja2dyb3VuZEFQSS5kZWxldGVNZW1vcnkoaWQsIHRleHQpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGNvbnNvbGUubG9nKCdSZWNlaXZlZCBkZWxldGUgcmVzcG9uc2U6JywgcmVzcG9uc2UpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGlmIChyZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJykge1xyXG4gICAgICAgICAgICAvLyBBZnRlciBzdWNjZXNzZnVsIGRlbGV0aW9uLCByZWxvYWQgbWVtb3JpZXMgYW5kIHN0YXkgb24gY3VycmVudCBwYWdlIGlmIHBvc3NpYmxlXHJcbiAgICAgICAgICAgIGNvbnN0IGN1cnJlbnRQYWdlQmVmb3JlRGVsZXRlID0gY3VycmVudFBhZ2U7XHJcbiAgICAgICAgICAgIGF3YWl0IGxvYWRBbGxNZW1vcmllcygpO1xyXG4gICAgICAgICAgICBpZiAoY3VycmVudFBhZ2VCZWZvcmVEZWxldGUgPD0gdG90YWxQYWdlcykge1xyXG4gICAgICAgICAgICAgICAgY3VycmVudFBhZ2UgPSBjdXJyZW50UGFnZUJlZm9yZURlbGV0ZTtcclxuICAgICAgICAgICAgICAgIGRpc3BsYXlNZW1vcmllc1BhZ2UoY3VycmVudFBhZ2UpO1xyXG4gICAgICAgICAgICAgICAgdXBkYXRlUGFnaW5hdGlvbkNvbnRyb2xzKCk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGRlbGV0aW5nIG1lbW9yeTonLCBlcnJvcik7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVHJhY2sgZXJyb3IgaW4gTWl4cGFuZWxcclxuICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgICB0eXBlOiBlcnJvci5uYW1lIHx8ICdFcnJvcicsXHJcbiAgICAgICAgICAgIG1lc3NhZ2U6IGVycm9yLm1lc3NhZ2UgfHwgJ1Vua25vd24gZXJyb3IgZGVsZXRpbmcgbWVtb3J5JyxcclxuICAgICAgICAgICAgc3RhY2s6IGVycm9yLnN0YWNrLFxyXG4gICAgICAgICAgICBjb250ZXh0OiAncG9wdXBfZGVsZXRlX21lbW9yeScsXHJcbiAgICAgICAgICAgIGZ1bmN0aW9uTmFtZTogJ2RlbGV0ZU1lbW9yeScsXHJcbiAgICAgICAgICAgIG1lbW9yeUlkOiBpZFxyXG4gICAgICAgIH0pO1xyXG4gICAgICAgIFxyXG4gICAgfVxyXG59XHJcblxyXG4vLyBIZWxwZXIgZnVuY3Rpb24gdG8gZm9ybWF0IHRpbWVzdGFtcFxyXG5mdW5jdGlvbiBmb3JtYXREYXRlKHRpbWVzdGFtcCkge1xyXG4gICAgcmV0dXJuIG5ldyBEYXRlKHRpbWVzdGFtcCkudG9Mb2NhbGVTdHJpbmcoKTtcclxufVxyXG5cclxuLy8gVGFiIHN3aXRjaGluZyBmdW5jdGlvbmFsaXR5XHJcbmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy50YWItYnV0dG9uJykuZm9yRWFjaChidXR0b24gPT4ge1xyXG4gICAgYnV0dG9uLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xyXG4gICAgICAgIC8vIFVwZGF0ZSBidXR0b24gc3RhdGVzXHJcbiAgICAgICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnRhYi1idXR0b24nKS5mb3JFYWNoKGJ0biA9PiBidG4uY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJykpO1xyXG4gICAgICAgIGJ1dHRvbi5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTtcclxuXHJcbiAgICAgICAgLy8gVXBkYXRlIHRhYiBjb250ZW50XHJcbiAgICAgICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnLnRhYi1jb250ZW50JykuZm9yRWFjaCh0YWIgPT4gdGFiLmNsYXNzTGlzdC5yZW1vdmUoJ2FjdGl2ZScpKTtcclxuICAgICAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgJHtidXR0b24uZGF0YXNldC50YWJ9LXRhYmApLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpO1xyXG5cclxuICAgICAgICAvLyBMb2FkIGNvbnRlbnQgZm9yIHNwZWNpZmljIHRhYnNcclxuICAgICAgICBpZiAoYnV0dG9uLmRhdGFzZXQudGFiID09PSAndmlldycpIHtcclxuICAgICAgICAgICAgbG9hZEFsbE1lbW9yaWVzKCk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcbn0pO1xyXG5cclxuXHJcblxyXG4vLyBGdW5jdGlvbiB0byB1cGRhdGUgbWVtb3J5IGxpbWl0IGJhbm5lciB2aXNpYmlsaXR5XHJcbmFzeW5jIGZ1bmN0aW9uIHVwZGF0ZU1lbW9yeUxpbWl0QmFubmVyKHVzZXIpIHtcclxuICAgIGNvbnN0IG1lbW9yeUxpbWl0QmFubmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC1iYW5uZXInKTtcclxuICAgIGNvbnN0IG1lbW9yeUxpbWl0VGl0bGUgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWVtb3J5LWxpbWl0LXRpdGxlJyk7XHJcbiAgICBjb25zdCBtZW1vcnlMaW1pdFRleHQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWVtb3J5LWxpbWl0LXRleHQnKTtcclxuICAgIGlmICghbWVtb3J5TGltaXRCYW5uZXIgfHwgIW1lbW9yeUxpbWl0VGl0bGUgfHwgIW1lbW9yeUxpbWl0VGV4dCkgcmV0dXJuO1xyXG4gICAgXHJcbiAgICB0cnkge1xyXG4gICAgICAgIC8vIEdldCBtZW1vcnkgbGltaXQgaW5mbyBmcm9tIGJhY2tncm91bmQgc2NyaXB0XHJcbiAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldE1lbW9yeUxpbWl0SW5mbygpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIGlmIChyZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJykge1xyXG4gICAgICAgICAgICBjb25zdCB7IGxpbWl0LCBjdXJyZW50LCB1c2VyVHlwZSB9ID0gcmVzcG9uc2U7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTaG93IFBybyBiYW5uZXIgZm9yIHBhaWQgdXNlcnNcclxuICAgICAgICAgICAgaWYgKHVzZXJUeXBlID09PSAncGFpZCcpIHtcclxuICAgICAgICAgICAgICAgIG1lbW9yeUxpbWl0QmFubmVyLmNsYXNzTGlzdC5yZW1vdmUoJ2hpZGRlbicpO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICBjb25zdCBzaWduaW5CdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbWVtb3J5LWxpbWl0LXNpZ25pbi1idXR0b24nKTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IHVwZ3JhZGVCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndXBncmFkZS1idXR0b24nKTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IGJhbm5lckljb24gPSBtZW1vcnlMaW1pdEJhbm5lci5xdWVyeVNlbGVjdG9yKCdzdmcnKTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IGJhbm5lclRpdGxlID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC10aXRsZScpO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgYmFubmVyVGV4dCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtZW1vcnktbGltaXQtdGV4dCcpO1xyXG4gICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAvLyBVc2UgcHVycGxlIFBybyBzdHlsZSBmb3IgcGFpZCB1c2Vyc1xyXG4gICAgICAgICAgICAgICAgbWVtb3J5TGltaXRCYW5uZXIuY2xhc3NOYW1lID0gJ21iLTQgcC00IGJnLXB1cnBsZS01MCBib3JkZXIgYm9yZGVyLXB1cnBsZS0yMDAgcm91bmRlZC1sZyc7XHJcbiAgICAgICAgICAgICAgICBpZiAoYmFubmVyVGl0bGUpIHtcclxuICAgICAgICAgICAgICAgICAgICBiYW5uZXJUaXRsZS5jbGFzc05hbWUgPSAndGV4dC1wdXJwbGUtOTAwIHRleHQtc20gZm9udC1zZW1pYm9sZCBsZWFkaW5nLXJlbGF4ZWQgbWItMSc7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAoYmFubmVyVGV4dCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGJhbm5lclRleHQuY2xhc3NOYW1lID0gJ3RleHQtcHVycGxlLTgwMCB0ZXh0LXNtIGZvbnQtbWVkaXVtIGxlYWRpbmctcmVsYXhlZCc7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBpZiAoYmFubmVySWNvbikge1xyXG4gICAgICAgICAgICAgICAgICAgYmFubmVySWNvbi5zZXRBdHRyaWJ1dGUoJ3N0cm9rZScsICcjN2MzYWVkJyk7XHJcbiAgICAgICAgICAgICAgICAgICBiYW5uZXJJY29uLmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPVwiTTEyIDJsMy4wOSA2LjI2TDIyIDkuMjdsLTUgNC44NyAxLjE4IDYuODhMMTIgMTcuNzdsLTYuMTggMy4yNUw3IDE0LjE0IDIgOS4yN2w2LjkxLTEuMDFMMTIgMnpcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgc3Ryb2tlLWxpbmVqb2luPVwicm91bmRcIi8+XHJcbiAgICAgICAgICAgICAgICAgICBgO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAvLyBIaWRlIGJvdGggYnV0dG9ucyBmb3IgUHJvIHVzZXJzXHJcbiAgICAgICAgICAgICAgIGlmIChzaWduaW5CdXR0b24pIHtcclxuICAgICAgICAgICAgICAgICAgIHNpZ25pbkJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIGlmICh1cGdyYWRlQnV0dG9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICB1cGdyYWRlQnV0dG9uLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XHJcbiAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIGNvbnN0IHRpdGxlID0gJ01heE1lbW9yeSBQcm8gYWN0aXZlJztcclxuICAgICAgICAgICAgICAgY29uc3QgbWVzc2FnZSA9ICdZb3UgaGF2ZSB1bmxpbWl0ZWQgbWVtb3JpZXMgYW5kIGZ1bGwgYWNjZXNzIHRvIHlvdXIgbWVtb3J5IHZhdWx0Lic7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlMaW1pdFRpdGxlLnRleHRDb250ZW50ID0gdGl0bGU7XHJcbiAgICAgICAgICAgICAgICBtZW1vcnlMaW1pdFRleHQudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xyXG4gICAgICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTaG93IGJhbm5lciBmb3IgZ3Vlc3QgYW5kIGxvZ2dlZCBpbiB1c2Vyc1xyXG4gICAgICAgICAgICAgICBtZW1vcnlMaW1pdEJhbm5lci5jbGFzc0xpc3QucmVtb3ZlKCdoaWRkZW4nKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgIGNvbnN0IHNpZ25pbkJ1dHRvbiA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtZW1vcnktbGltaXQtc2lnbmluLWJ1dHRvbicpO1xyXG4gICAgICAgICAgICAgICBjb25zdCB1cGdyYWRlQnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3VwZ3JhZGUtYnV0dG9uJyk7XHJcbiAgICAgICAgICAgICAgIGNvbnN0IGJhbm5lckJ1dHRvbkNvbnRhaW5lciA9IHNpZ25pbkJ1dHRvbi5wYXJlbnRFbGVtZW50O1xyXG4gICAgICAgICAgICAgICBjb25zdCBiYW5uZXJJY29uID0gbWVtb3J5TGltaXRCYW5uZXIucXVlcnlTZWxlY3Rvcignc3ZnJyk7XHJcbiAgICAgICAgICAgICAgIGNvbnN0IGJhbm5lclRpdGxlID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC10aXRsZScpO1xyXG4gICAgICAgICAgICAgICBjb25zdCBiYW5uZXJUZXh0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC10ZXh0Jyk7XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICBsZXQgdGl0bGU7XHJcbiAgICAgICAgICAgICAgIGxldCBtZXNzYWdlO1xyXG4gICAgICAgICAgICAgICBpZiAodXNlclR5cGUgPT09ICdndWVzdCcpIHtcclxuICAgICAgICAgICAgICAgICAgIGlmIChjdXJyZW50ID49IGxpbWl0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgbWVzc2FnZSA9IGBZb3UgaGF2ZSBoaXQgdGhlIGd1ZXN0IGxpbWl0IGF0ICR7Y3VycmVudH0vJHtsaW1pdH0gbWVtb3JpZXMuIFNpZ24gaW4gbm93IHRvIGtlZXAgc2F2aW5nIGFuZCB1bmxvY2sgdGhlIGZ1bGwgZnJlZSB0aWVyLmA7XHJcbiAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UgPSBgJHtjdXJyZW50fS8ke2xpbWl0fSBndWVzdCBtZW1vcmllcyB1c2VkLiBTaWduIGluIHRvIHVubG9jayAxMDAgZnJlZSBtZW1vcmllcyBhbmQgc3luYyB0aGVtIHRvIHlvdXIgYWNjb3VudC5gO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgdGl0bGUgPSAnU2lnbiBJbiB3aXRoIEdvb2dsZSBmb3IgMTAwIGZyZWUgbWVtb3JpZXMnO1xyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAvLyBVc2Ugd2FybmluZyBzdHlsZSBmb3IgZ3Vlc3RzIChvcmFuZ2UvcmVkKVxyXG4gICAgICAgICAgICAgICAgICAgbWVtb3J5TGltaXRCYW5uZXIuY2xhc3NOYW1lID0gJ21iLTQgcC00IGJnLW9yYW5nZS01MCBib3JkZXIgYm9yZGVyLW9yYW5nZS0yMDAgcm91bmRlZC1sZyc7XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoYmFubmVyVGl0bGUpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICBiYW5uZXJUaXRsZS5jbGFzc05hbWUgPSAndGV4dC1vcmFuZ2UtOTAwIHRleHQtc20gZm9udC1zZW1pYm9sZCBsZWFkaW5nLXJlbGF4ZWQgbWItMSc7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoYmFubmVyVGV4dCkge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIGJhbm5lclRleHQuY2xhc3NOYW1lID0gJ3RleHQtb3JhbmdlLTgwMCB0ZXh0LXNtIGZvbnQtbWVkaXVtIGxlYWRpbmctcmVsYXhlZCc7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoYmFubmVySWNvbikge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIGJhbm5lckljb24uc2V0QXR0cmlidXRlKCdzdHJva2UnLCAnI2VhNTgwYycpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgIGJhbm5lckljb24uaW5uZXJIVE1MID0gYFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPVwiTTEwLjI5IDMuODZMMS44MiAxOGEyIDIgMCAwMDEuNzEgM2gxNi45NGEyIDIgMCAwMDEuNzEtM0wxMy43MSAzLjg2YTIgMiAwIDAwLTMuNDIgMHpcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgc3Ryb2tlLWxpbmVqb2luPVwicm91bmRcIi8+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPVwiMTJcIiB5MT1cIjlcIiB4Mj1cIjEyXCIgeTI9XCIxM1wiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgPGxpbmUgeDE9XCIxMlwiIHkxPVwiMTdcIiB4Mj1cIjEyLjAxXCIgeTI9XCIxN1wiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgICAgICAgICAgICBgO1xyXG4gICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAvLyBTaG93IHNpZ24taW4gYnV0dG9uIGZvciBndWVzdHMsIGhpZGUgdXBncmFkZSBidXR0b24gZnJvbSB0b3BcclxuICAgICAgICAgICAgICAgICAgIGlmIChzaWduaW5CdXR0b24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICBzaWduaW5CdXR0b24uc3R5bGUuZGlzcGxheSA9ICdmbGV4JztcclxuICAgICAgICAgICAgICAgICAgICAgICBzaWduaW5CdXR0b24uY2xhc3NOYW1lID0gJ2ZsZXggaXRlbXMtY2VudGVyIGdhcC0yIHB4LTQgcHktMiBiZy13aGl0ZSB0ZXh0LWdyYXktNzAwIHRleHQtc20gZm9udC1tZWRpdW0gcm91bmRlZC1tZCBob3ZlcjpiZy1ncmF5LTUwIGJvcmRlciBib3JkZXItZ3JheS0zMDAgdHJhbnNpdGlvbi1jb2xvcnMgZHVyYXRpb24tMjAwIHNoYWRvdy1zbSc7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgc2lnbmluQnV0dG9uLmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgPHN2ZyB2ZXJzaW9uPVwiMS4xXCIgeG1sbnM9XCJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2Z1wiIHdpZHRoPVwiMTZweFwiIGhlaWdodD1cIjE2cHhcIiB2aWV3Qm94PVwiMCAwIDQ4IDQ4XCI+PGc+PHBhdGggZmlsbD1cIiNFQTQzMzVcIiBkPVwiTTI0IDkuNWMzLjU0IDAgNi43MSAxLjIyIDkuMjEgMy42bDYuODUtNi44NUMzNS45IDIuMzggMzAuNDcgMCAyNCAwIDE0LjYyIDAgNi41MSA1LjM4IDIuNTYgMTMuMjJsNy45OCA2LjE5QzEyLjQzIDEzLjcyIDE3Ljc0IDkuNSAyNCA5LjV6XCI+PC9wYXRoPjxwYXRoIGZpbGw9XCIjNDI4NUY0XCIgZD1cIk00Ni45OCAyNC41NWMwLTEuNTctLjE1LTMuMDktLjQyLTQuNTVIMjR2OS4wMmgxMi45NGMtLjU4IDIuOTYtMi4yNiA1LjQ4LTQuNzggNy4xOGw3LjczIDZjNC41MS00LjE4IDcuMDktMTAuMzYgNy4wOS0xNy42NXpcIj48L3BhdGg+PHBhdGggZmlsbD1cIiNGQkJDMDVcIiBkPVwiTTEwLjUzIDI4LjU5Yy0uNDgtMS40NS0uNzYtMi45OS0uNzYtNC41OXMuMjctMy4xNC43Ni00LjU5bC03Ljk4LTYuMTlDLjkyIDE2LjQ2IDAgMjAuMTIgMCAyNGMwIDMuODguOTIgNy41NCAyLjU2IDEwLjc4bDcuOTctNi4xOXpcIj48L3BhdGg+PHBhdGggZmlsbD1cIiMzNEE4NTNcIiBkPVwiTTI0IDQ4YzYuNDggMCAxMS45My0yLjEzIDE1Ljg5LTUuODFsLTcuNzMtNmMtMi4xNSAxLjQ1LTQuOTIgMi4zLTguMTYgMi4zLTYuMjYgMC0xMS41Ny00LjIyLTEzLjQ3LTkuOTFsLTcuOTggNi4xOUM2LjUxIDQyLjYyIDE0LjYyIDQ4IDI0IDQ4elwiPjwvcGF0aD48cGF0aCBmaWxsPVwibm9uZVwiIGQ9XCJNMCAwaDQ4djQ4SDB6XCI+PC9wYXRoPjwvZz48L3N2Zz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgU2lnbiBJbiB3aXRoIEdvb2dsZVxyXG4gICAgICAgICAgICAgICAgICAgICAgIGA7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBpZiAodXBncmFkZUJ1dHRvbiAmJiB1cGdyYWRlQnV0dG9uLnBhcmVudEVsZW1lbnQgIT09IGJhbm5lckJ1dHRvbkNvbnRhaW5lcikge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIHVwZ3JhZGVCdXR0b24uc3R5bGUuZGlzcGxheSA9ICdub25lJztcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgfSBlbHNlIGlmICh1c2VyVHlwZSA9PT0gJ2xvZ2dlZF9pbicpIHtcclxuICAgICAgICAgICAgICAgICAgIGlmIChjdXJyZW50ID49IGxpbWl0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgbWVzc2FnZSA9IGBZb3UgaGF2ZSBoaXQgeW91ciBmcmVlIGxpbWl0IGF0ICR7Y3VycmVudH0vJHtsaW1pdH0gbWVtb3JpZXMuIFVwZ3JhZGUgdG8ga2VlcCBzYXZpbmcgd2l0aG91dCBpbnRlcnJ1cHRpb25zLmA7XHJcbiAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UgPSBgJHtjdXJyZW50fS8ke2xpbWl0fSBmcmVlIG1lbW9yaWVzIHVzZWQuIFVwZ3JhZGUgdG8gUHJvIHRvIHJlbW92ZSB0aGUgY2FwIGFuZCBrZWVwIHlvdXIgbWVtb3J5IHZhdWx0IGdyb3dpbmcuYDtcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIHRpdGxlID0gJ1VwZ3JhZGUgdG8gUHJvIGZvciB1bmxpbWl0ZWQgbWVtb3JpZXMnO1xyXG4gICAgICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgICAgICAgICAvLyBVc2UgZnJpZW5kbHkgYmx1ZSBzdHlsZSBmb3IgbG9nZ2VkLWluIHVzZXJzXHJcbiAgICAgICAgICAgICAgICAgICBtZW1vcnlMaW1pdEJhbm5lci5jbGFzc05hbWUgPSAnbWItNCBwLTQgYmctYmx1ZS01MCBib3JkZXIgYm9yZGVyLWJsdWUtMjAwIHJvdW5kZWQtbGcnO1xyXG4gICAgICAgICAgICAgICAgICAgaWYgKGJhbm5lclRpdGxlKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgYmFubmVyVGl0bGUuY2xhc3NOYW1lID0gJ3RleHQtYmx1ZS05MDAgdGV4dC1zbSBmb250LXNlbWlib2xkIGxlYWRpbmctcmVsYXhlZCBtYi0xJztcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIGlmIChiYW5uZXJUZXh0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgYmFubmVyVGV4dC5jbGFzc05hbWUgPSAndGV4dC1ibHVlLTgwMCB0ZXh0LXNtIGZvbnQtbWVkaXVtIGxlYWRpbmctcmVsYXhlZCc7XHJcbiAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICBpZiAoYmFubmVySWNvbikge1xyXG4gICAgICAgICAgICAgICAgICAgICAgIGJhbm5lckljb24uc2V0QXR0cmlidXRlKCdzdHJva2UnLCAnIzI1NjNlYicpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgIGJhbm5lckljb24uaW5uZXJIVE1MID0gYFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPVwiTTEyIDJsMy4wOSA2LjI2TDIyIDkuMjdsLTUgNC44NyAxLjE4IDYuODhMMTIgMTcuNzdsLTYuMTggMy4yNUw3IDE0LjE0IDIgOS4yN2w2LjkxLTEuMDFMMTIgMnpcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgc3Ryb2tlLWxpbmVqb2luPVwicm91bmRcIi8+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgYDtcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgLy8gTW92ZSBleGlzdGluZyB1cGdyYWRlIGJ1dHRvbiB0byBiYW5uZXIsIGhpZGUgc2lnbi1pbiBidXR0b25cclxuICAgICAgICAgICAgICAgICAgIGlmIChzaWduaW5CdXR0b24pIHtcclxuICAgICAgICAgICAgICAgICAgICAgICBzaWduaW5CdXR0b24uc3R5bGUuZGlzcGxheSA9ICdub25lJztcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgIGlmICh1cGdyYWRlQnV0dG9uKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgLy8gTW92ZSB1cGdyYWRlIGJ1dHRvbiB0byBiYW5uZXIgaWYgbm90IGFscmVhZHkgdGhlcmVcclxuICAgICAgICAgICAgICAgICAgICAgICBpZiAodXBncmFkZUJ1dHRvbi5wYXJlbnRFbGVtZW50ICE9PSBiYW5uZXJCdXR0b25Db250YWluZXIpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgYmFubmVyQnV0dG9uQ29udGFpbmVyLmFwcGVuZENoaWxkKHVwZ3JhZGVCdXR0b24pO1xyXG4gICAgICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgICAgICAgICB1cGdyYWRlQnV0dG9uLnN0eWxlLmRpc3BsYXkgPSAnZmxleCc7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgLy8gS2VlcCB0aGUgYmFubmVyIGJsdWUgd2hpbGUgdXNpbmcgYSBzdHJvbmcgdXBncmFkZSBDVEEgYnV0dG9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgdXBncmFkZUJ1dHRvbi5jbGFzc05hbWUgPSAnZmxleCBpdGVtcy1jZW50ZXIgZ2FwLTIgcHgtNCBweS0yIGJnLXB1cnBsZS02MDAgdGV4dC13aGl0ZSB0ZXh0LXNtIGZvbnQtbWVkaXVtIHJvdW5kZWQtbWQgaG92ZXI6YmctcHVycGxlLTcwMCB0cmFuc2l0aW9uLWNvbG9ycyBkdXJhdGlvbi0yMDAgc2hhZG93LXNtJztcclxuICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgbWVtb3J5TGltaXRUaXRsZS50ZXh0Q29udGVudCA9IHRpdGxlO1xyXG4gICAgICAgICAgICAgICBtZW1vcnlMaW1pdFRleHQudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xyXG4gICAgICAgIH1cclxuICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgZ2V0dGluZyBtZW1vcnkgbGltaXQgaW5mbzonLCBlcnJvcik7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVHJhY2sgZXJyb3IgaW4gTWl4cGFuZWxcclxuICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgICB0eXBlOiBlcnJvci5uYW1lIHx8ICdFcnJvcicsXHJcbiAgICAgICAgICAgIG1lc3NhZ2U6IGVycm9yLm1lc3NhZ2UgfHwgJ1Vua25vd24gZXJyb3IgZ2V0dGluZyBtZW1vcnkgbGltaXQgaW5mbycsXHJcbiAgICAgICAgICAgIHN0YWNrOiBlcnJvci5zdGFjayxcclxuICAgICAgICAgICAgY29udGV4dDogJ3BvcHVwX21lbW9yeV9saW1pdF9iYW5uZXInLFxyXG4gICAgICAgICAgICBmdW5jdGlvbk5hbWU6ICd1cGRhdGVNZW1vcnlMaW1pdEJhbm5lcidcclxuICAgICAgICB9KTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBGYWxsYmFjayB0byBoaWRpbmcgYmFubmVyIG9uIGVycm9yXHJcbiAgICAgICAgbWVtb3J5TGltaXRCYW5uZXIuY2xhc3NMaXN0LmFkZCgnaGlkZGVuJyk7XHJcbiAgICB9XHJcbn1cclxuXHJcbi8vIEV2ZW50IExpc3RlbmVyc1xyXG5kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc29ydC1tZW1vcmllcycpLmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIGxvYWRBbGxNZW1vcmllcyk7XHJcblxyXG4vLyBBdXRoZW50aWNhdGlvbiBpcyBub3cgaGFuZGxlZCB2aWEgVVJMIG1vbml0b3JpbmcgaW4gYmFja2dyb3VuZC5qc1xyXG5cclxuLy8gSW5pdGlhbCBsb2FkIG9mIG1lbW9yaWVzIHdoZW4gdGhlIHBvcHVwIGlzIG9wZW5lZFxyXG5kb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdET01Db250ZW50TG9hZGVkJywgYXN5bmMgZnVuY3Rpb24oKSB7XHJcbiBcclxuICAgIC8vIEF1dGhlbnRpY2F0aW9uIGlzIG5vdyBoYW5kbGVkIGF1dG9tYXRpY2FsbHkgdmlhIFVSTCBtb25pdG9yaW5nIGluIGJhY2tncm91bmQuanNcclxuICAgIFxyXG4gICAgLy8gQXV0aCBlbGVtZW50c1xyXG4gICAgY29uc3Qgc2lnbmluQnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NpZ25pbi1idXR0b24nKTtcclxuICAgIGNvbnN0IHNpZ25vdXRCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnc2lnbm91dC1idXR0b24nKTtcclxuICAgIGNvbnN0IHVzZXJQcm9maWxlID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3VzZXItcHJvZmlsZScpO1xyXG4gICAgY29uc3QgdXNlckdyZWV0aW5nID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3VzZXItZ3JlZXRpbmcnKTtcclxuICAgIFxyXG4gICAgLy8gQ2hlY2sgaWYgcmVxdWlyZWQgZWxlbWVudHMgZXhpc3RcclxuICAgIGlmICghc2lnbm91dEJ1dHRvbikge1xyXG4gICAgICAgIGNvbnNvbGUuZXJyb3IoJ1JlcXVpcmVkIERPTSBlbGVtZW50IHNpZ25vdXQtYnV0dG9uIG5vdCBmb3VuZCcpO1xyXG4gICAgICAgIHJldHVybjtcclxuICAgIH1cclxuICAgIGlmICghdXNlclByb2ZpbGUpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKCdSZXF1aXJlZCBET00gZWxlbWVudCB1c2VyLXByb2ZpbGUgbm90IGZvdW5kJyk7XHJcbiAgICAgICAgcmV0dXJuO1xyXG4gICAgfVxyXG4gICAgaWYgKCF1c2VyR3JlZXRpbmcpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKCdSZXF1aXJlZCBET00gZWxlbWVudCB1c2VyLWdyZWV0aW5nIG5vdCBmb3VuZCcpO1xyXG4gICAgICAgIHJldHVybjtcclxuICAgIH1cclxuXHJcbiAgICAvLyBBZGQgdmlzaWJpbGl0eSBjaGFuZ2UgbGlzdGVuZXIgdG8gcmVmcmVzaCBkYXRhIHdoZW4gcG9wdXAgYmVjb21lcyB2aXNpYmxlIGFnYWluXHJcbiAgICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCd2aXNpYmlsaXR5Y2hhbmdlJywgZnVuY3Rpb24oKSB7XHJcbiAgICAgICAgaWYgKGRvY3VtZW50LnZpc2liaWxpdHlTdGF0ZSA9PT0gJ3Zpc2libGUnICYmIGN1cnJlbnRVc2VyRGF0YSAmJiBjdXJyZW50VXNlckRhdGEudWlkKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdQb3B1cCBiZWNhbWUgdmlzaWJsZSwgcmVmcmVzaGluZyB1c2VyIGRhdGEnKTtcclxuICAgICAgICAgICAgcmVmcmVzaEN1cnJlbnRVc2VyRGF0YSgpO1xyXG4gICAgICAgIH1cclxuICAgIH0pO1xyXG5cclxuICAgIC8vIEFkZCB3aW5kb3cgZm9jdXMgbGlzdGVuZXIgdG8gcmVmcmVzaCBkYXRhIHdoZW4gcG9wdXAgcmVnYWlucyBmb2N1c1xyXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2ZvY3VzJywgZnVuY3Rpb24oKSB7XHJcbiAgICAgICAgaWYgKGN1cnJlbnRVc2VyRGF0YSAmJiBjdXJyZW50VXNlckRhdGEudWlkICYmICFpc1NpZ25pbmdPdXQpIHtcclxuICAgICAgICAgICAgY29uc29sZS5sb2coJ1BvcHVwIHJlZ2FpbmVkIGZvY3VzLCByZWZyZXNoaW5nIHVzZXIgZGF0YScpO1xyXG4gICAgICAgICAgICByZWZyZXNoQ3VycmVudFVzZXJEYXRhKCk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcblxyXG4gICAgLy8gTGlzdGVuIGZvciBhdXRoIHN0YXRlIGNoYW5nZXMgZnJvbSBiYWNrZ3JvdW5kIHNjcmlwdFxyXG4gICAgY2hyb21lLnJ1bnRpbWUub25NZXNzYWdlLmFkZExpc3RlbmVyKChtZXNzYWdlLCBzZW5kZXIsIHNlbmRSZXNwb25zZSkgPT4ge1xyXG4gICAgICAgIGlmIChtZXNzYWdlLnR5cGUgPT09ICdBVVRIX1NUQVRFX0NIQU5HRUQnICYmIG1lc3NhZ2UudXNlcikge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnUmVjZWl2ZWQgYXV0aCBzdGF0ZSBjaGFuZ2U6JywgbWVzc2FnZS51c2VyKTtcclxuICAgICAgICAgICAgLy8gVXBkYXRlIGN1cnJlbnQgdXNlciBkYXRhIGFuZCByZWZyZXNoIFVJXHJcbiAgICAgICAgICAgIGlmIChjdXJyZW50VXNlckRhdGEgJiYgY3VycmVudFVzZXJEYXRhLnVpZCA9PT0gbWVzc2FnZS51c2VyLnVpZCkge1xyXG4gICAgICAgICAgICAgICAgY29uc3Qgd2FzUHJvQmVmb3JlID0gY3VycmVudFVzZXJEYXRhLmlzUGFpZCB8fCBmYWxzZTtcclxuICAgICAgICAgICAgICAgIGNvbnN0IGlzUHJvTm93ID0gbWVzc2FnZS51c2VyLmlzUGFpZCB8fCBmYWxzZTtcclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgY3VycmVudFVzZXJEYXRhID0ge1xyXG4gICAgICAgICAgICAgICAgICAgIC4uLmN1cnJlbnRVc2VyRGF0YSxcclxuICAgICAgICAgICAgICAgICAgICBpc1BhaWQ6IGlzUHJvTm93LFxyXG4gICAgICAgICAgICAgICAgICAgIHN1YnNjcmlwdGlvblR5cGU6IG1lc3NhZ2UudXNlci5zdWJzY3JpcHRpb25UeXBlIHx8IG51bGxcclxuICAgICAgICAgICAgICAgIH07XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGlmICh3YXNQcm9CZWZvcmUgIT09IGlzUHJvTm93KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ1VwZGF0aW5nIFVJIGR1ZSB0byBzdWJzY3JpcHRpb24gY2hhbmdlJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgdXBkYXRlQXV0aFVJKGN1cnJlbnRVc2VyRGF0YSk7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBIYW5kbGUgU2lnbi1JbiAtIFJlZGlyZWN0IHRvIHdlYmFwcFxyXG4gICAgaWYgKHNpZ25pbkJ1dHRvbikge1xyXG4gICAgICAgIHNpZ25pbkJ1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGFzeW5jICgpID0+IHtcclxuICAgICAgICAgICAgLy8gVHJhY2sgYXV0aGVudGljYXRpb24gcmVkaXJlY3QgdG8gd2ViYXBwXHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tBdXRoZW50aWNhdGlvblJlZGlyZWN0ZWQoJ3dlYmFwcCcpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gR2V0IGN1cnJlbnQgZGV2aWNlIElEIHRvIHBhc3MgdG8gd2ViYXBwXHJcbiAgICAgICAgICAgIGNvbnN0IGRldmljZUlkID0gYXdhaXQgYmFja2dyb3VuZEFQSS5nZXREZXZpY2VJZCgpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gT3BlbiB3ZWJhcHAgYXV0aGVudGljYXRpb24gcGFnZSBpbiBhIG5ldyB0YWIgd2l0aCBkZXZpY2UgSURcclxuICAgICAgICAgICAgY2hyb21lLnRhYnMuY3JlYXRlKHtcclxuICAgICAgICAgICAgICAgIHVybDogYF9fRlJPTlRFTkRfVVJMX18vYXV0aD9zb3VyY2U9ZXh0ZW5zaW9uJmRldmljZUlkPSR7ZW5jb2RlVVJJQ29tcG9uZW50KGRldmljZUlkKX1gXHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfVxyXG5cclxuICAgIC8vIEhhbmRsZSBTaWduLU91dFxyXG4gICAgbGV0IGlzU2lnbmluZ091dCA9IGZhbHNlO1xyXG4gICAgc2lnbm91dEJ1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcclxuICAgICAgICBpZiAoaXNTaWduaW5nT3V0KSByZXR1cm47IC8vIFByZXZlbnQgbXVsdGlwbGUgc2lnbi1vdXQgYXR0ZW1wdHNcclxuICAgICAgICBcclxuICAgICAgICBpc1NpZ25pbmdPdXQgPSB0cnVlO1xyXG4gICAgICAgIHNpZ25vdXRCdXR0b24uZGlzYWJsZWQgPSB0cnVlO1xyXG4gICAgICAgIHNpZ25vdXRCdXR0b24udGV4dENvbnRlbnQgPSAnU2lnbmluZyBvdXQuLi4nO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFRyYWNrIHNpZ24gb3V0XHJcbiAgICAgICAgaWYgKGN1cnJlbnRVc2VyRGF0YSkge1xyXG4gICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrU2lnbk91dChjdXJyZW50VXNlckRhdGEudWlkKTtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVXNlIGJhY2tncm91bmQgc2NyaXB0IGZvciBhdXRoZW50aWNhdGlvblxyXG4gICAgICAgIGJhY2tncm91bmRBUEkuc2lnbk91dCgpXHJcbiAgICAgICAgICAgIC50aGVuKGFzeW5jIChyZXNwb25zZSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgaWYgKHJlc3BvbnNlLnN0YXR1cyA9PT0gJ3N1Y2Nlc3MnKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgLy8gQ2xlYXIgbG9jYWwgdXNlciBkYXRhIGFuZCB1cGRhdGUgVUlcclxuICAgICAgICAgICAgICAgICAgICBjdXJyZW50VXNlckRhdGEgPSBudWxsO1xyXG4gICAgICAgICAgICAgICAgICAgIGF3YWl0IHVwZGF0ZUF1dGhVSShudWxsKTtcclxuICAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcignU2lnbi1vdXQgZmFpbGVkOicsIHJlc3BvbnNlLm1lc3NhZ2UpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9KVxyXG4gICAgICAgICAgICAuY2F0Y2goZXJyb3IgPT4ge1xyXG4gICAgICAgICAgICAgICAgY29uc29sZS5lcnJvcihlcnJvcik7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIFRyYWNrIGVycm9yIGluIE1peHBhbmVsXHJcbiAgICAgICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgICAgICAgICAgIHR5cGU6IGVycm9yLm5hbWUgfHwgJ0Vycm9yJyxcclxuICAgICAgICAgICAgICAgICAgICBtZXNzYWdlOiBlcnJvci5tZXNzYWdlIHx8ICdVbmtub3duIGVycm9yIGR1cmluZyBzaWduIG91dCcsXHJcbiAgICAgICAgICAgICAgICAgICAgc3RhY2s6IGVycm9yLnN0YWNrLFxyXG4gICAgICAgICAgICAgICAgICAgIGNvbnRleHQ6ICdwb3B1cF9zaWduX291dCcsXHJcbiAgICAgICAgICAgICAgICAgICAgZnVuY3Rpb25OYW1lOiAnc2lnbk91dEhhbmRsZXInXHJcbiAgICAgICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICB9KVxyXG4gICAgICAgICAgICAuZmluYWxseSgoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBpc1NpZ25pbmdPdXQgPSBmYWxzZTtcclxuICAgICAgICAgICAgICAgIHNpZ25vdXRCdXR0b24uZGlzYWJsZWQgPSBmYWxzZTtcclxuICAgICAgICAgICAgICAgIHNpZ25vdXRCdXR0b24udGV4dENvbnRlbnQgPSAnU2lnbiBPdXQnO1xyXG4gICAgICAgICAgICB9KTtcclxuICAgIH0pO1xyXG5cclxuICAgIC8vIEdldCBpbml0aWFsIHVzZXIgZGF0YSBmcm9tIGJhY2tncm91bmQgc2NyaXB0XHJcbiAgICBmdW5jdGlvbiBnZXRJbml0aWFsVXNlckRhdGEoKSB7XHJcbiAgICAgICAgaWYgKGlzU2lnbmluZ091dCkgcmV0dXJuOyAvLyBEb24ndCBmZXRjaCB1c2VyIGRhdGEgZHVyaW5nIHNpZ24tb3V0XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gRm9yY2UgcmVmcmVzaCB3aGVuIHBvcHVwIG9wZW5zIHRvIGVuc3VyZSBsYXRlc3QgZGF0YVxyXG4gICAgICAgIGJhY2tncm91bmRBUEkuZ2V0Q3VycmVudFVzZXIodHJ1ZSlcclxuICAgICAgICAgICAgLnRoZW4oYXN5bmMgKHJlc3BvbnNlKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBpZiAocmVzcG9uc2Uuc3RhdHVzID09PSAnc3VjY2VzcycgJiYgIWlzU2lnbmluZ091dCkge1xyXG4gICAgICAgICAgICAgICAgICAgIGN1cnJlbnRVc2VyRGF0YSA9IHJlc3BvbnNlLnVzZXI7XHJcbiAgICAgICAgICAgICAgICAgICAgYXdhaXQgdXBkYXRlQXV0aFVJKGN1cnJlbnRVc2VyRGF0YSk7XHJcblxyXG4gICAgICAgICAgICAgICAgICAgIC8vIE5vIG5lZWQgdG8gY2FsbCBmZXRjaExhdGVzdFVzZXJEYXRhIHNpbmNlIGdldEN1cnJlbnRVc2VyIGFscmVhZHkgcHJvdmlkZXMgdGhlIGxhdGVzdCBkYXRhXHJcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdObyB1c2VyIGRhdGEgcmVjZWl2ZWQgb3IgZXJyb3Igb2NjdXJyZWQnKTtcclxuICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgICAgLmNhdGNoKGVycm9yID0+IHtcclxuICAgICAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGdldHRpbmcgY3VycmVudCB1c2VyOicsIGVycm9yKTtcclxuICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgLy8gVHJhY2sgZXJyb3IgaW4gTWl4cGFuZWxcclxuICAgICAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tFcnJvcih7XHJcbiAgICAgICAgICAgICAgICAgICAgdHlwZTogZXJyb3IubmFtZSB8fCAnRXJyb3InLFxyXG4gICAgICAgICAgICAgICAgICAgIG1lc3NhZ2U6IGVycm9yLm1lc3NhZ2UgfHwgJ1Vua25vd24gZXJyb3IgZ2V0dGluZyBjdXJyZW50IHVzZXInLFxyXG4gICAgICAgICAgICAgICAgICAgIHN0YWNrOiBlcnJvci5zdGFjayxcclxuICAgICAgICAgICAgICAgICAgICBjb250ZXh0OiAncG9wdXBfZ2V0X2N1cnJlbnRfdXNlcicsXHJcbiAgICAgICAgICAgICAgICAgICAgZnVuY3Rpb25OYW1lOiAnZ2V0SW5pdGlhbFVzZXJEYXRhJ1xyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICBnZXRJbml0aWFsVXNlckRhdGEoKTtcclxuICAgLy8gVHJhY2sgcG9wdXAgb3BlbmVkXHJcbiAgICB0cnkge1xyXG4gICAgICAgIGF3YWl0IGJhY2tncm91bmRBUEkudHJhY2tQb3B1cE9wZW5lZCgncG9wdXAnKTtcclxuICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgdHJhY2tpbmcgcG9wdXAgb3BlbmVkOicsIGVycm9yKTtcclxuICAgIH1cclxuICAgIFxyXG4gICAgLy8gQmFja2VuZC1vbmx5IGZ1bmN0aW9uYWxpdHlcclxuXHJcbiAgICAvLyBMaXN0ZW4gZm9yIGF1dGggc3RhdGUgY2hhbmdlcyBhbmQgc3luYyBjb21wbGV0aW9uIGZyb20gYmFja2dyb3VuZCBzY3JpcHRcclxuICAgIGNocm9tZS5ydW50aW1lLm9uTWVzc2FnZS5hZGRMaXN0ZW5lcihhc3luYyAobWVzc2FnZSkgPT4ge1xyXG4gICAgICAgIGlmIChtZXNzYWdlLnR5cGUgPT09ICdBVVRIX1NUQVRFX0NIQU5HRUQnICYmICFpc1NpZ25pbmdPdXQpIHtcclxuICAgICAgICAgICAgY3VycmVudFVzZXJEYXRhID0gbWVzc2FnZS51c2VyO1xyXG4gICAgICAgICAgICBhd2FpdCB1cGRhdGVBdXRoVUkobWVzc2FnZS51c2VyKTtcclxuICAgICAgICAgICAgXHJcbiB9XHJcblxyXG4gICAgfSk7XHJcblxyXG4gICAgLy8gRnVuY3Rpb24gdG8gdXBkYXRlIFVJIGJhc2VkIG9uIGF1dGggc3RhdGVcclxuICAgIGFzeW5jIGZ1bmN0aW9uIHVwZGF0ZUF1dGhVSSh1c2VyKSB7XHJcbiAgICAgICAgLy8gVXBkYXRlIG1lbW9yeSBsaW1pdCBiYW5uZXIgdmlzaWJpbGl0eVxyXG4gICAgICAgIGF3YWl0IHVwZGF0ZU1lbW9yeUxpbWl0QmFubmVyKHVzZXIpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFVwZGF0ZSBzaWduaW4gc2VjdGlvbiB2aXNpYmlsaXR5XHJcbiAgICAgICAgdXBkYXRlU2lnbmluU2VjdGlvbih1c2VyKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBVcGRhdGUgdXBncmFkZSBidXR0b24gdmlzaWJpbGl0eVxyXG4gICAgICAgIHVwZGF0ZVVwZ3JhZGVCdXR0b24odXNlcik7XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKHVzZXIpIHtcclxuICAgICAgICAgICAgLy8gVXNlciBpcyBzaWduZWQgaW5cclxuICAgICAgICAgICAgY29uc3QgdXNlcm5hbWUgPSB1c2VyLmRpc3BsYXlOYW1lIHx8IHVzZXIuZW1haWwuc3BsaXQoJ0AnKVswXTtcclxuICAgICAgICAgICAgdXNlckdyZWV0aW5nLnRleHRDb250ZW50ID0gYEhpLCAke3VzZXJuYW1lfWA7XHJcbiAgICAgICAgICAgIHVzZXJQcm9maWxlLnN0eWxlLmRpc3BsYXkgPSAnZmxleCc7XHJcbiAgICAgICAgICAgIGlmIChzaWduaW5CdXR0b24pIHtcclxuICAgICAgICAgICAgICAgIHNpZ25pbkJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTaG93L2hpZGUgUHJvIGJhZGdlIGJhc2VkIG9uIHN1YnNjcmlwdGlvbiBzdGF0dXNcclxuICAgICAgICAgICAgY29uc3QgcHJvQmFkZ2UgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncHJvLWJhZGdlJyk7XHJcbiAgICAgICAgICAgIGlmIChwcm9CYWRnZSkge1xyXG4gICAgICAgICAgICAgICAgaWYgKHVzZXIuaXNQYWlkKSB7XHJcbiAgICAgICAgICAgICAgICAgICAgcHJvQmFkZ2UuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7XHJcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIHByb0JhZGdlLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAvLyBVc2VyIGlzIHNpZ25lZCBvdXRcclxuICAgICAgICAgICAgdXNlclByb2ZpbGUuc3R5bGUuZGlzcGxheSA9ICdub25lJztcclxuICAgICAgICAgICAgaWYgKHNpZ25pbkJ1dHRvbikge1xyXG4gICAgICAgICAgICAgICAgc2lnbmluQnV0dG9uLnN0eWxlLmRpc3BsYXkgPSAnZmxleCc7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIEhpZGUgUHJvIGJhZGdlIHdoZW4gc2lnbmVkIG91dFxyXG4gICAgICAgICAgICBjb25zdCBwcm9CYWRnZSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdwcm8tYmFkZ2UnKTtcclxuICAgICAgICAgICAgaWYgKHByb0JhZGdlKSB7XHJcbiAgICAgICAgICAgICAgICBwcm9CYWRnZS5jbGFzc0xpc3QuYWRkKCdoaWRkZW4nKTtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIFxyXG5cclxuICAgIC8vIEZ1bmN0aW9uIHRvIHVwZGF0ZSBzaWduaW4gc2VjdGlvbiB2aXNpYmlsaXR5XHJcbiAgICBmdW5jdGlvbiB1cGRhdGVTaWduaW5TZWN0aW9uKHVzZXIpIHtcclxuICAgICAgICBjb25zdCBzdGF0dXNNZXNzYWdlID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2JhY2tlbmQtc3RhdHVzLW1lc3NhZ2UnKTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBPbmx5IHByb2NlZWQgaWYgdGhlIGVsZW1lbnQgZXhpc3RzXHJcbiAgICAgICAgaWYgKCFzdGF0dXNNZXNzYWdlKSB7XHJcbiAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICB9XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKHVzZXIgJiYgdXNlci51aWQpIHtcclxuICAgICAgICAgICAgLy8gVXNlciBpcyBzaWduZWQgaW4gLSBzaG93IHN0YXR1cyBtZXNzYWdlXHJcbiAgICAgICAgICAgIHN0YXR1c01lc3NhZ2UuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7XHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgLy8gVXNlciBpcyBub3Qgc2lnbmVkIGluIC0gaGlkZSBzdGF0dXMgbWVzc2FnZVxyXG4gICAgICAgICAgICBzdGF0dXNNZXNzYWdlLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIFxyXG4gICAgLy8gRnVuY3Rpb24gdG8gdXBkYXRlIHVwZ3JhZGUgYnV0dG9uIHZpc2liaWxpdHlcclxuICAgIGZ1bmN0aW9uIHVwZGF0ZVVwZ3JhZGVCdXR0b24odXNlcikge1xyXG4gICAgICAgIGNvbnN0IHVwZ3JhZGVCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndXBncmFkZS1idXR0b24nKTtcclxuICAgICAgICBcclxuICAgICAgICBpZiAoIXVwZ3JhZGVCdXR0b24pIHtcclxuICAgICAgICAgICAgcmV0dXJuO1xyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICAvLyBUaGUgdXBncmFkZSBidXR0b24gdmlzaWJpbGl0eSBpcyBub3cgaGFuZGxlZCBieSB1cGRhdGVNZW1vcnlMaW1pdEJhbm5lclxyXG4gICAgICAgIC8vIFRoaXMgZnVuY3Rpb24gaXMga2VwdCBmb3IgY29tcGF0aWJpbGl0eSBidXQgdGhlIGxvZ2ljIGlzIG1vdmVkXHJcbiAgICAgICAgLy8gU2hvdyB1cGdyYWRlIGJ1dHRvbiBmb3Igc2lnbmVkLWluIHVzZXJzIHdobyBhcmUgbm90IFBybyAoZmFsbGJhY2spXHJcbiAgICAgICAgaWYgKHVzZXIgJiYgdXNlci51aWQgJiYgIXVzZXIuaXNQYWlkKSB7XHJcbiAgICAgICAgICAgIC8vIE9ubHkgc2hvdyBpZiBub3QgYWxyZWFkeSBtb3ZlZCB0byBiYW5uZXJcclxuICAgICAgICAgICAgY29uc3QgYmFubmVyQnV0dG9uQ29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC1zaWduaW4tYnV0dG9uJyk/LnBhcmVudEVsZW1lbnQ7XHJcbiAgICAgICAgICAgIGlmICh1cGdyYWRlQnV0dG9uLnBhcmVudEVsZW1lbnQgIT09IGJhbm5lckJ1dHRvbkNvbnRhaW5lcikge1xyXG4gICAgICAgICAgICAgICAgdXBncmFkZUJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ2ZsZXgnO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgdXBncmFkZUJ1dHRvbi5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xyXG4gICAgICAgIH1cclxuICAgIH1cclxuICAgIC8vIEZ1bmN0aW9uIHRvIHJlZnJlc2ggdGhlIGN1cnJlbnQgdXNlciBkYXRhIHVzaW5nIHRoZSBjb25zb2xpZGF0ZWQgQVBJXHJcbiAgICBhc3luYyBmdW5jdGlvbiByZWZyZXNoQ3VycmVudFVzZXJEYXRhKCkge1xyXG4gICAgICBpZiAoIWN1cnJlbnRVc2VyRGF0YSB8fCAhY3VycmVudFVzZXJEYXRhLnVpZCB8fCBpc1NpZ25pbmdPdXQpIHJldHVybjtcclxuICAgICAgXHJcbiAgICAgIHRyeSB7XHJcbiAgICAgICAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldEN1cnJlbnRVc2VyKCk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKHJlc3BvbnNlLnN0YXR1cyA9PT0gJ3N1Y2Nlc3MnICYmIHJlc3BvbnNlLnVzZXIgJiYgIWlzU2lnbmluZ091dCkge1xyXG4gICAgICAgICAgLy8gQ2hlY2sgaWYgc3Vic2NyaXB0aW9uIHN0YXR1cyBjaGFuZ2VkXHJcbiAgICAgICAgICBjb25zdCB3YXNQcm9CZWZvcmUgPSBjdXJyZW50VXNlckRhdGEuaXNQYWlkIHx8IGZhbHNlO1xyXG4gICAgICAgICAgY29uc3QgaXNQcm9Ob3cgPSByZXNwb25zZS51c2VyLmlzUGFpZCB8fCBmYWxzZTtcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgLy8gVXBkYXRlIHRoZSBjdXJyZW50IHVzZXIgZGF0YSB3aXRoIHRoZSBsYXRlc3QgaW5mb1xyXG4gICAgICAgICAgY3VycmVudFVzZXJEYXRhID0gcmVzcG9uc2UudXNlcjtcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgY29uc29sZS5sb2coJ1JlZnJlc2hlZCB1c2VyIGRhdGE6JywgY3VycmVudFVzZXJEYXRhKTtcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgLy8gSWYgc3Vic2NyaXB0aW9uIHN0YXR1cyBjaGFuZ2VkLCB1cGRhdGUgdGhlIFVJXHJcbiAgICAgICAgICBpZiAod2FzUHJvQmVmb3JlICE9PSBpc1Byb05vdykge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygnU3Vic2NyaXB0aW9uIHN0YXR1cyBjaGFuZ2VkLCB1cGRhdGluZyBVSScpO1xyXG4gICAgICAgICAgICBhd2FpdCB1cGRhdGVBdXRoVUkoY3VycmVudFVzZXJEYXRhKTtcclxuICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgcmVmcmVzaGluZyBjdXJyZW50IHVzZXIgZGF0YTonLCBlcnJvcik7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVHJhY2sgZXJyb3IgaW4gTWl4cGFuZWxcclxuICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgdHlwZTogZXJyb3IubmFtZSB8fCAnRXJyb3InLFxyXG4gICAgICAgICAgbWVzc2FnZTogZXJyb3IubWVzc2FnZSB8fCAnVW5rbm93biBlcnJvciByZWZyZXNoaW5nIHVzZXIgZGF0YScsXHJcbiAgICAgICAgICBzdGFjazogZXJyb3Iuc3RhY2ssXHJcbiAgICAgICAgICBjb250ZXh0OiAncG9wdXBfcmVmcmVzaF91c2VyX2RhdGEnLFxyXG4gICAgICAgICAgZnVuY3Rpb25OYW1lOiAncmVmcmVzaEN1cnJlbnRVc2VyRGF0YSdcclxuICAgICAgICB9KTtcclxuICAgICAgfVxyXG4gICAgfVxyXG5cclxuICAgIGNvbnN0IGFsbE1lbW9yaWVzQ29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2FsbC1tZW1vcmllcycpO1xyXG4gICAgY29uc3QgbWVtb3J5Q291bnRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1jb3VudCcpO1xyXG4gICAgY29uc3Qgc29ydFNlbGVjdCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzb3J0LW1lbW9yaWVzJyk7XHJcbiAgICBjb25zdCBmaWx0ZXJUYWdTZWxlY3QgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZmlsdGVyLXRhZycpO1xyXG4gICAgXHJcbiAgICAvLyBBZGQgZXZlbnQgbGlzdGVuZXJzXHJcbiAgICBzb3J0U2VsZWN0LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsICgpID0+IHtcclxuICAgICAgICBjdXJyZW50UGFnZSA9IDE7XHJcbiAgICAgICAgYXBwbHlGaWx0ZXJzQW5kU29ydGluZygpO1xyXG4gICAgfSk7XHJcblxyXG4gICAgaWYgKGZpbHRlclRhZ1NlbGVjdCkge1xyXG4gICAgICAgIGZpbHRlclRhZ1NlbGVjdC5hZGRFdmVudExpc3RlbmVyKCdjaGFuZ2UnLCAoKSA9PiB7XHJcbiAgICAgICAgICAgIGN1cnJlbnRQYWdlID0gMTtcclxuICAgICAgICAgICAgYXBwbHlGaWx0ZXJzQW5kU29ydGluZygpO1xyXG4gICAgICAgIH0pO1xyXG4gICAgfVxyXG5cclxuICAgIC8vIEJhY2tlbmQtb25seSBmdW5jdGlvbmFsaXR5XHJcblxyXG4gICAgLy8gQWRkIG1lbW9yeSBtb2RhbCBmdW5jdGlvbmFsaXR5XHJcbiAgICBjb25zdCBhZGRNZW1vcnlCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYWRkLW1lbW9yeS1idXR0b24nKTtcclxuICAgIGNvbnN0IGFkZE1lbW9yeVNlY3Rpb24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYWRkLW1lbW9yeS1zZWN0aW9uJyk7XHJcbiAgICBjb25zdCBuZXdNZW1vcnlJbnB1dCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduZXctbWVtb3J5LWlucHV0Jyk7XHJcbiAgICBjb25zdCBjYW5jZWxBZGRNZW1vcnkgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY2FuY2VsLWFkZC1tZW1vcnknKTtcclxuICAgIGNvbnN0IGNvbmZpcm1BZGRNZW1vcnkgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY29uZmlybS1hZGQtbWVtb3J5Jyk7XHJcblxyXG4gICAgYWRkTWVtb3J5QnV0dG9uLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xyXG4gICAgICAgIGFkZE1lbW9yeVNlY3Rpb24uc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XHJcbiAgICAgICAgYWRkTWVtb3J5QnV0dG9uLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XHJcbiAgICAgICAgbmV3TWVtb3J5SW5wdXQuZm9jdXMoKTtcclxuICAgIH0pO1xyXG5cclxuICAgIGZ1bmN0aW9uIGhpZGVBZGRNZW1vcnlTZWN0aW9uKCkge1xyXG4gICAgICAgIGFkZE1lbW9yeVNlY3Rpb24uc3R5bGUuZGlzcGxheSA9ICdub25lJztcclxuICAgICAgICBhZGRNZW1vcnlCdXR0b24uc3R5bGUuZGlzcGxheSA9ICdmbGV4JztcclxuICAgICAgICBuZXdNZW1vcnlJbnB1dC52YWx1ZSA9ICcnO1xyXG4gICAgICAgIGNvbnN0IHRhZ0lucHV0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ25ldy1tZW1vcnktdGFnJyk7XHJcbiAgICAgICAgaWYgKHRhZ0lucHV0KSB0YWdJbnB1dC52YWx1ZSA9ICcnO1xyXG4gICAgfVxyXG5cclxuICAgIGNhbmNlbEFkZE1lbW9yeS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGhpZGVBZGRNZW1vcnlTZWN0aW9uKTtcclxuXHJcbiAgICBjb25maXJtQWRkTWVtb3J5LmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgYXN5bmMgKCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IHRleHQgPSBuZXdNZW1vcnlJbnB1dC52YWx1ZS50cmltKCk7XHJcbiAgICAgICAgY29uc3QgdGFnID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ25ldy1tZW1vcnktdGFnJyk/LnZhbHVlLnRyaW0oKSB8fCBudWxsO1xyXG4gICAgICAgIGlmICghdGV4dCkgcmV0dXJuO1xyXG5cclxuICAgICAgICAvLyBDaGVjayBtZW1vcnkgbGltaXQgYmVmb3JlIGFkZGluZ1xyXG4gICAgICAgIHRyeSB7XHJcbiAgICAgICAgICAgIGNvbnN0IGxpbWl0UmVzcG9uc2UgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldE1lbW9yeUxpbWl0SW5mbygpO1xyXG4gICAgICAgICAgICBpZiAobGltaXRSZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJyAmJiAhbGltaXRSZXNwb25zZS5jYW5BZGQpIHtcclxuICAgICAgICAgICAgICAgIGxldCBtZXNzYWdlO1xyXG4gICAgICAgICAgICAgICAgaWYgKGxpbWl0UmVzcG9uc2UudXNlclR5cGUgPT09ICdndWVzdCcpIHtcclxuICAgICAgICAgICAgICAgICAgICBtZXNzYWdlID0gYFlvdSd2ZSByZWFjaGVkIHRoZSAke2xpbWl0UmVzcG9uc2UubGltaXR9IG1lbW9yeSBsaW1pdCBmb3IgZ3Vlc3QgdXNlcnMuIFNpZ24gaW4gZm9yIGZyZWUgdG8gZ2V0IDEwMCBtZW1vcmllcyFgO1xyXG4gICAgICAgICAgICAgICAgfSBlbHNlIGlmIChsaW1pdFJlc3BvbnNlLnVzZXJUeXBlID09PSAnbG9nZ2VkX2luJykge1xyXG4gICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UgPSBgWW91J3ZlIHJlYWNoZWQgdGhlICR7bGltaXRSZXNwb25zZS5saW1pdH0gbWVtb3J5IGxpbWl0LiBVcGdyYWRlIHRvIFBybyBmb3IgdW5saW1pdGVkIG1lbW9yaWVzIWA7XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICBhbGVydChtZXNzYWdlKTtcclxuICAgICAgICAgICAgICAgIHJldHVybjtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUuZXJyb3IoJ0Vycm9yIGNoZWNraW5nIG1lbW9yeSBsaW1pdDonLCBlcnJvcik7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBUcmFjayBlcnJvciBpbiBNaXhwYW5lbFxyXG4gICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgICAgICAgdHlwZTogZXJyb3IubmFtZSB8fCAnRXJyb3InLFxyXG4gICAgICAgICAgICAgICAgbWVzc2FnZTogZXJyb3IubWVzc2FnZSB8fCAnVW5rbm93biBlcnJvciBjaGVja2luZyBtZW1vcnkgbGltaXQnLFxyXG4gICAgICAgICAgICAgICAgc3RhY2s6IGVycm9yLnN0YWNrLFxyXG4gICAgICAgICAgICAgICAgY29udGV4dDogJ3BvcHVwX2FkZF9tZW1vcnlfbGltaXRfY2hlY2snLFxyXG4gICAgICAgICAgICAgICAgZnVuY3Rpb25OYW1lOiAnY29uZmlybUFkZE1lbW9yeUhhbmRsZXInXHJcbiAgICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuXHJcbiAgICAgICAgY29uZmlybUFkZE1lbW9yeS5kaXNhYmxlZCA9IHRydWU7XHJcbiAgICAgICAgY29uZmlybUFkZE1lbW9yeS5pbm5lckhUTUwgPSBgXHJcbiAgICAgICAgICAgIDxzdmcgY2xhc3M9XCJzcGlubmVyXCIgd2lkdGg9XCIxNFwiIGhlaWdodD1cIjE0XCIgdmlld0JveD1cIjAgMCAyNCAyNFwiIGZpbGw9XCJub25lXCIgc3Ryb2tlPVwiY3VycmVudENvbG9yXCIgc3Ryb2tlLXdpZHRoPVwiMlwiPlxyXG4gICAgICAgICAgICAgICAgPGNpcmNsZSBjeD1cIjEyXCIgY3k9XCIxMlwiIHI9XCIxMFwiPjwvY2lyY2xlPlxyXG4gICAgICAgICAgICAgICAgPHBhdGggZD1cIk0xMiAyYTEwIDEwIDAgMCAxIDEwIDEwXCI+PC9wYXRoPlxyXG4gICAgICAgICAgICA8L3N2Zz5cclxuICAgICAgICBgO1xyXG5cclxuICAgICAgICB0cnkge1xyXG4gICAgICAgICAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IGJhY2tncm91bmRBUEkuc2F2ZU1lbW9yeSh0ZXh0LCB0YWcpO1xyXG5cclxuICAgICAgICAgICAgaWYgKHJlc3BvbnNlLnN0YXR1cyA9PT0gJ3N1Y2Nlc3MnKSB7XHJcbiAgICAgICAgICAgICAgICBoaWRlQWRkTWVtb3J5U2VjdGlvbigpO1xyXG4gICAgICAgICAgICAgICAgY3VycmVudFBhZ2UgPSAxOyAvLyBHbyB0byBmaXJzdCBwYWdlIHRvIHNlZSB0aGUgbmV3IG1lbW9yeVxyXG4gICAgICAgICAgICAgICAgYXdhaXQgbG9hZEFsbE1lbW9yaWVzKCk7XHJcbiAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IocmVzcG9uc2UubWVzc2FnZSB8fCAnRmFpbGVkIHRvIHNhdmUgbWVtb3J5Jyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBzYXZpbmcgbWVtb3J5OicsIGVycm9yKTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFRyYWNrIGVycm9yIGluIE1peHBhbmVsXHJcbiAgICAgICAgICAgIGJhY2tncm91bmRBUEkudHJhY2tFcnJvcih7XHJcbiAgICAgICAgICAgICAgICB0eXBlOiBlcnJvci5uYW1lIHx8ICdFcnJvcicsXHJcbiAgICAgICAgICAgICAgICBtZXNzYWdlOiBlcnJvci5tZXNzYWdlIHx8ICdVbmtub3duIGVycm9yIHNhdmluZyBtZW1vcnknLFxyXG4gICAgICAgICAgICAgICAgc3RhY2s6IGVycm9yLnN0YWNrLFxyXG4gICAgICAgICAgICAgICAgY29udGV4dDogJ3BvcHVwX3NhdmVfbWVtb3J5JyxcclxuICAgICAgICAgICAgICAgIGZ1bmN0aW9uTmFtZTogJ2NvbmZpcm1BZGRNZW1vcnlIYW5kbGVyJ1xyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfSBmaW5hbGx5IHtcclxuICAgICAgICAgICAgY29uZmlybUFkZE1lbW9yeS5kaXNhYmxlZCA9IGZhbHNlO1xyXG4gICAgICAgICAgICBjb25maXJtQWRkTWVtb3J5LmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9XCIxNFwiIGhlaWdodD1cIjE0XCIgdmlld0JveD1cIjAgMCAyNCAyNFwiIGZpbGw9XCJub25lXCIgc3Ryb2tlPVwiY3VycmVudENvbG9yXCIgc3Ryb2tlLXdpZHRoPVwiMlwiPlxyXG4gICAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9XCJNNSAxM2w0IDRMMTkgN1wiIHN0cm9rZS1saW5lY2FwPVwicm91bmRcIiBzdHJva2UtbGluZWpvaW49XCJyb3VuZFwiLz5cclxuICAgICAgICAgICAgICAgIDwvc3ZnPlxyXG4gICAgICAgICAgICBgO1xyXG4gICAgICAgIH1cclxuICAgIH0pO1xyXG5cclxuICAgIC8vIEFkZCBFc2NhcGUga2V5IHN1cHBvcnRcclxuICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCAoZSkgPT4ge1xyXG4gICAgICAgIGlmIChlLmtleSA9PT0gJ0VzY2FwZScgJiYgYWRkTWVtb3J5U2VjdGlvbi5zdHlsZS5kaXNwbGF5ID09PSAnYmxvY2snKSB7XHJcbiAgICAgICAgICAgIGhpZGVBZGRNZW1vcnlTZWN0aW9uKCk7XHJcbiAgICAgICAgfVxyXG4gICAgfSk7XHJcblxyXG4gICAgLy8gQWRkIHVwZ3JhZGUgYnV0dG9uIGxpc3RlbmVyXHJcbiAgICBjb25zdCB1cGdyYWRlQnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3VwZ3JhZGUtYnV0dG9uJyk7XHJcbiAgICBpZiAodXBncmFkZUJ1dHRvbikge1xyXG4gICAgICAgIHVwZ3JhZGVCdXR0b24uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7XHJcbiAgICAgICAgICAgIC8vIFRyYWNrIHVwZ3JhZGUgY2xpY2tlZFxyXG4gICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrVXBncmFkZUNsaWNrZWQoJ3BvcHVwJyk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBPcGVuIHdlYmFwcCB1cGdyYWRlIHBhZ2UgaW4gYSBuZXcgdGFiXHJcbiAgICAgICAgICAgIGNocm9tZS50YWJzLmNyZWF0ZSh7XHJcbiAgICAgICAgICAgICAgICB1cmw6ICdfX0ZST05URU5EX1VSTF9fL3ByaWNpbmc/c291cmNlPWV4dGVuc2lvbidcclxuICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgfSk7XHJcbiAgICB9XHJcblxyXG4gICAgLy8gQWRkIHBhZ2luYXRpb24gYnV0dG9uIGxpc3RlbmVyc1xyXG4gICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3ByZXYtcGFnZScpLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xyXG4gICAgICAgIGlmIChjdXJyZW50UGFnZSA+IDEpIHtcclxuICAgICAgICAgICAgY3VycmVudFBhZ2UtLTtcclxuICAgICAgICAgICAgZGlzcGxheU1lbW9yaWVzUGFnZShjdXJyZW50UGFnZSk7XHJcbiAgICAgICAgICAgIHVwZGF0ZVBhZ2luYXRpb25Db250cm9scygpO1xyXG4gICAgICAgIH1cclxuICAgIH0pO1xyXG5cclxuICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduZXh0LXBhZ2UnKS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcclxuICAgICAgICBpZiAoY3VycmVudFBhZ2UgPCB0b3RhbFBhZ2VzKSB7XHJcbiAgICAgICAgICAgIGN1cnJlbnRQYWdlKys7XHJcbiAgICAgICAgICAgIGRpc3BsYXlNZW1vcmllc1BhZ2UoY3VycmVudFBhZ2UpO1xyXG4gICAgICAgICAgICB1cGRhdGVQYWdpbmF0aW9uQ29udHJvbHMoKTtcclxuICAgICAgICB9XHJcbiAgICB9KTtcclxuXHJcbiAgICAvLyBVcGRhdGUgc29ydCBtZW1vcmllcyBldmVudCBsaXN0ZW5lclxyXG4gICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3NvcnQtbWVtb3JpZXMnKS5hZGRFdmVudExpc3RlbmVyKCdjaGFuZ2UnLCAoKSA9PiB7XHJcbiAgICAgICAgY3VycmVudFBhZ2UgPSAxOyAvLyBSZXNldCB0byBmaXJzdCBwYWdlIHdoZW4gc29ydGluZ1xyXG4gICAgICAgIGxvYWRBbGxNZW1vcmllcygpO1xyXG4gICAgfSk7XHJcblxyXG5cclxuXHJcbiAgICAvLyBBZGQgbWVtb3J5IGxpbWl0IGJhbm5lciBidXR0b24gaGFuZGxlciAod2lsbCBiZSBkeW5hbWljYWxseSB1cGRhdGVkIGJhc2VkIG9uIHVzZXIgdHlwZSlcclxuICAgIGNvbnN0IG1lbW9yeUxpbWl0U2lnbmluQnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ21lbW9yeS1saW1pdC1zaWduaW4tYnV0dG9uJyk7XHJcbiAgICBpZiAobWVtb3J5TGltaXRTaWduaW5CdXR0b24pIHtcclxuICAgICAgICBtZW1vcnlMaW1pdFNpZ25pbkJ1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGFzeW5jICgpID0+IHtcclxuICAgICAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgICAgIC8vIFRyYWNrIHBvcHVwIG9wZW5lZCBmcm9tIG1lbW9yeSBsaW1pdCB3YXJuaW5nXHJcbiAgICAgICAgICAgICAgICBhd2FpdCBiYWNrZ3JvdW5kQVBJLnRyYWNrUG9wdXBPcGVuZWQoJ21lbW9yeV9saW1pdF93YXJuaW5nJyk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIEdldCBjdXJyZW50IHVzZXIgdHlwZSB0byBkZXRlcm1pbmUgYWN0aW9uXHJcbiAgICAgICAgICAgICAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IGJhY2tncm91bmRBUEkuZ2V0TWVtb3J5TGltaXRJbmZvKCk7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGlmIChyZXNwb25zZS5zdGF0dXMgPT09ICdzdWNjZXNzJykge1xyXG4gICAgICAgICAgICAgICAgICAgIGlmIChyZXNwb25zZS51c2VyVHlwZSA9PT0gJ2d1ZXN0Jykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBUcmFjayBhdXRoZW50aWNhdGlvbiByZWRpcmVjdCBmcm9tIG1lbW9yeSBsaW1pdCBiYW5uZXJcclxuICAgICAgICAgICAgICAgICAgICAgICAgYmFja2dyb3VuZEFQSS50cmFja0F1dGhlbnRpY2F0aW9uUmVkaXJlY3RlZCgnd2ViYXBwJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBHZXQgY3VycmVudCBkZXZpY2UgSUQgdG8gcGFzcyB0byB3ZWJhcHBcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgZGV2aWNlSWQgPSBhd2FpdCBiYWNrZ3JvdW5kQVBJLmdldERldmljZUlkKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAvLyBPcGVuIHdlYmFwcCBhdXRoZW50aWNhdGlvbiBwYWdlIGluIGEgbmV3IHRhYiB3aXRoIGRldmljZSBJRFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaHJvbWUudGFicy5jcmVhdGUoe1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdXJsOiBgX19GUk9OVEVORF9VUkxfXy9hdXRoP3NvdXJjZT1leHRlbnNpb24mcmVhc29uPW1lbW9yeV9saW1pdCZkZXZpY2VJZD0ke2VuY29kZVVSSUNvbXBvbmVudChkZXZpY2VJZCl9YFxyXG4gICAgICAgICAgICAgICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgICAgICAgICB9IGVsc2UgaWYgKHJlc3BvbnNlLnVzZXJUeXBlID09PSAnbG9nZ2VkX2luJykge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrVXBncmFkZUNsaWNrZWQoJ21lbW9yeV9saW1pdF9iYW5uZXInKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgLy8gT3BlbiBwcmljaW5nIHBhZ2UgZm9yIGxvZ2dlZC1pbiB1c2Vyc1xyXG4gICAgICAgICAgICAgICAgICAgICAgICBjaHJvbWUudGFicy5jcmVhdGUoeyB1cmw6ICdfX0ZST05URU5EX1VSTF9fL3ByaWNpbmc/c291cmNlPWV4dGVuc2lvbiZyZWFzb249bWVtb3J5X2xpbWl0JyB9KTtcclxuICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBoYW5kbGluZyBidXR0b24gY2xpY2s6JywgZXJyb3IpO1xyXG4gICAgICAgICAgICAgICAgLy8gRmFsbGJhY2sgdG8gYXV0aCBwYWdlXHJcbiAgICAgICAgICAgICAgICBjb25zdCBkZXZpY2VJZCA9IGF3YWl0IGJhY2tncm91bmRBUEkuZ2V0RGV2aWNlSWQoKTtcclxuICAgICAgICAgICAgICAgIGNocm9tZS50YWJzLmNyZWF0ZSh7XHJcbiAgICAgICAgICAgICAgICAgICAgdXJsOiBgX19GUk9OVEVORF9VUkxfXy9hdXRoP3NvdXJjZT1leHRlbnNpb24mcmVhc29uPW1lbW9yeV9saW1pdCZkZXZpY2VJZD0ke2VuY29kZVVSSUNvbXBvbmVudChkZXZpY2VJZCl9YFxyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9KTtcclxuICAgIH1cclxuXHJcbiAgICAvLyBJbml0aWFsIG1lbW9yeSBsb2FkXHJcbiAgICBsb2FkQWxsTWVtb3JpZXMoKTtcclxufSk7XHJcblxyXG4vLyBTaWduLWluIHNlY3Rpb24gdmlzaWJpbGl0eSBtYW5hZ2VkIGJ5IHVwZGF0ZVNpZ25pblNlY3Rpb25cclxuXHJcbi8vIEFkZCB0aGVzZSBuZXcgZnVuY3Rpb25zIGZvciBlZGl0IGZ1bmN0aW9uYWxpdHlcclxuYXN5bmMgZnVuY3Rpb24gc2F2ZUVkaXQoaWQsIHRleHRFbGVtZW50LCBlZGl0QnV0dG9uLCBvcmlnaW5hbFRleHQsIG5ld1RhZyA9IG51bGwsIHRhZ0VkaXRJbnB1dCA9IG51bGwpIHtcclxuICAgIGNvbnN0IG5ld1RleHQgPSB0ZXh0RWxlbWVudC50ZXh0Q29udGVudC50cmltKCk7XHJcbiAgICBcclxuICAgIGlmICghbmV3VGV4dCkge1xyXG4gICAgICAgIGFsZXJ0KCdNZW1vcnkgdGV4dCBjYW5ub3QgYmUgZW1wdHkuJyk7XHJcbiAgICAgICAgcmV0dXJuO1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICB0cnkge1xyXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgYmFja2dyb3VuZEFQSS5lZGl0TWVtb3J5KGlkLCBuZXdUZXh0LCBvcmlnaW5hbFRleHQsIG5ld1RhZyk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgaWYgKHJlc3BvbnNlLnN0YXR1cyA9PT0gJ3N1Y2Nlc3MnKSB7XHJcbiAgICAgICAgICAgIHRleHRFbGVtZW50LnNldEF0dHJpYnV0ZSgnY29udGVudGVkaXRhYmxlJywgJ2ZhbHNlJyk7XHJcbiAgICAgICAgICAgIHRleHRFbGVtZW50LmNsYXNzTGlzdC5yZW1vdmUoJ2JnLWdyYXktNTAnLCAnYm9yZGVyJywgJ2JvcmRlci1ncmF5LTMwMCcsICdyb3VuZGVkJywgJ3AtMicpO1xyXG4gICAgICAgICAgICBlZGl0QnV0dG9uLmRpc2FibGVkID0gZmFsc2U7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBpZiAodGFnRWRpdElucHV0KSB7XHJcbiAgICAgICAgICAgICAgICB0YWdFZGl0SW5wdXQuY2xhc3NMaXN0LmFkZCgnaGlkZGVuJyk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFVwZGF0ZSBsb2NhbCBtZW1vcnkgcmVwcmVzZW50YXRpb25zXHJcbiAgICAgICAgICAgIGNvbnN0IGZ1bGxJbmRleCA9IGZ1bGxNZW1vcmllc0xpc3QuZmluZEluZGV4KG0gPT4gbS5pZCA9PT0gaWQpO1xyXG4gICAgICAgICAgICBpZiAoZnVsbEluZGV4ICE9PSAtMSkge1xyXG4gICAgICAgICAgICAgICAgZnVsbE1lbW9yaWVzTGlzdFtmdWxsSW5kZXhdLm1lbW9yeV90ZXh0ID0gbmV3VGV4dDtcclxuICAgICAgICAgICAgICAgIGZ1bGxNZW1vcmllc0xpc3RbZnVsbEluZGV4XS50YWcgPSBuZXdUYWcgfHwgbnVsbDtcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgY29uc3QgZGlzcGxheUluZGV4ID0gYWxsTWVtb3JpZXNEYXRhLmZpbmRJbmRleChtID0+IG0uaWQgPT09IGlkKTtcclxuICAgICAgICAgICAgaWYgKGRpc3BsYXlJbmRleCAhPT0gLTEpIHtcclxuICAgICAgICAgICAgICAgIGFsbE1lbW9yaWVzRGF0YVtkaXNwbGF5SW5kZXhdLm1lbW9yeV90ZXh0ID0gbmV3VGV4dDtcclxuICAgICAgICAgICAgICAgIGFsbE1lbW9yaWVzRGF0YVtkaXNwbGF5SW5kZXhdLnRhZyA9IG5ld1RhZyB8fCBudWxsO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICB1cGRhdGVUYWdGaWx0ZXJEcm9wZG93bihmdWxsTWVtb3JpZXNMaXN0KTtcclxuICAgICAgICAgICAgYXBwbHlGaWx0ZXJzQW5kU29ydGluZygpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgZWRpdEJ1dHRvbi5pbm5lckhUTUwgPSBgXHJcbiAgICAgICAgICAgICAgICA8c3ZnIHdpZHRoPVwiMTRcIiBoZWlnaHQ9XCIxNFwiIHZpZXdCb3g9XCIwIDAgMjQgMjRcIiBmaWxsPVwibm9uZVwiIHN0cm9rZT1cImN1cnJlbnRDb2xvclwiIHN0cm9rZS13aWR0aD1cIjJcIj5cclxuICAgICAgICAgICAgICAgICAgICA8cGF0aCBkPVwiTTExIDRINGEyIDIgMCAwMC0yIDJ2MTRhMiAyIDAgMDAyIDJoMTRhMiAyIDAgMDAyLTJ2LTdcIj48L3BhdGg+XHJcbiAgICAgICAgICAgICAgICAgICAgPHBhdGggZD1cIk0xOC41IDIuNWEyLjEyMSAyLjEyMSAwIDAxMyAzTDEyIDE1bC00IDEgMS00IDkuNS05LjV6XCI+PC9wYXRoPlxyXG4gICAgICAgICAgICAgICAgPC9zdmc+XHJcbiAgICAgICAgICAgIGA7XHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKHJlc3BvbnNlLm1lc3NhZ2UgfHwgJ0ZhaWxlZCB0byBzYXZlIGNoYW5nZXMuJyk7XHJcbiAgICAgICAgfVxyXG4gICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKCdFcnJvciBzYXZpbmcgZWRpdDonLCBlcnJvcik7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gVHJhY2sgZXJyb3IgaW4gTWl4cGFuZWxcclxuICAgICAgICBiYWNrZ3JvdW5kQVBJLnRyYWNrRXJyb3Ioe1xyXG4gICAgICAgICAgICB0eXBlOiBlcnJvci5uYW1lIHx8ICdFcnJvcicsXHJcbiAgICAgICAgICAgIG1lc3NhZ2U6IGVycm9yLm1lc3NhZ2UgfHwgJ1Vua25vd24gZXJyb3IgZWRpdGluZyBtZW1vcnknLFxyXG4gICAgICAgICAgICBzdGFjazogZXJyb3Iuc3RhY2ssXHJcbiAgICAgICAgICAgIGNvbnRleHQ6ICdwb3B1cF9lZGl0X21lbW9yeScsXHJcbiAgICAgICAgICAgIGZ1bmN0aW9uTmFtZTogJ3NhdmVFZGl0JyxcclxuICAgICAgICAgICAgbWVtb3J5SWQ6IGlkXHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgXHJcblxyXG4gICAgICAgIGVkaXRCdXR0b24uZGlzYWJsZWQgPSBmYWxzZTtcclxuICAgICAgICBlZGl0QnV0dG9uLmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgPHN2ZyB3aWR0aD1cIjE0XCIgaGVpZ2h0PVwiMTRcIiB2aWV3Qm94PVwiMCAwIDI0IDI0XCIgZmlsbD1cIm5vbmVcIiBzdHJva2U9XCJjdXJyZW50Q29sb3JcIiBzdHJva2Utd2lkdGg9XCIyXCI+XHJcbiAgICAgICAgICAgICAgICA8cGF0aCBkPVwiTTUgMTNsNCA0TDE5IDdcIj48L3BhdGg+XHJcbiAgICAgICAgICAgIDwvc3ZnPlxyXG4gICAgICAgIGA7XHJcbiAgICB9XHJcbn1cclxuXHJcbi8vIFVwZGF0ZSB0aGUgZGlzcGxheU1lbW9yaWVzUGFnZSBmdW5jdGlvbiB0byBpbmNsdWRlIGVkaXQvZGVsZXRlIGZ1bmN0aW9uYWxpdHkgYW5kIHRhZyBncm91cGluZ1xyXG5mdW5jdGlvbiBkaXNwbGF5TWVtb3JpZXNQYWdlKHBhZ2UpIHtcclxuICAgIGNvbnN0IHN0YXJ0SW5kZXggPSAocGFnZSAtIDEpICogaXRlbXNQZXJQYWdlO1xyXG4gICAgY29uc3QgZW5kSW5kZXggPSBzdGFydEluZGV4ICsgaXRlbXNQZXJQYWdlO1xyXG4gICAgY29uc3QgbWVtb3JpZXNUb0Rpc3BsYXkgPSBhbGxNZW1vcmllc0RhdGEuc2xpY2Uoc3RhcnRJbmRleCwgZW5kSW5kZXgpO1xyXG5cclxuICAgIGNvbnN0IGNvbnRhaW5lciA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdhbGwtbWVtb3JpZXMnKTtcclxuICAgIFxyXG4gICAgLy8gQ2hlY2sgaWYgdGhlcmUgYXJlIG5vIG1lbW9yaWVzIHRvIGRpc3BsYXlcclxuICAgIGlmIChhbGxNZW1vcmllc0RhdGEubGVuZ3RoID09PSAwKSB7XHJcbiAgICAgICAgLy8gVXNlIHRoZSBlbXB0eSBzdGF0ZSB0ZW1wbGF0ZVxyXG4gICAgICAgIGNvbnN0IGVtcHR5U3RhdGVUZW1wbGF0ZSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdlbXB0eS1zdGF0ZS10ZW1wbGF0ZScpO1xyXG4gICAgICAgIGlmIChlbXB0eVN0YXRlVGVtcGxhdGUpIHtcclxuICAgICAgICAgICAgY29udGFpbmVyLmlubmVySFRNTCA9IGVtcHR5U3RhdGVUZW1wbGF0ZS5pbm5lckhUTUw7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBBZGQgZXZlbnQgbGlzdGVuZXIgdG8gdGhlIGVtcHR5IHN0YXRlIGFkZCBidXR0b25cclxuICAgICAgICAgICAgY29uc3QgZW1wdHlTdGF0ZUJ1dHRvbiA9IGNvbnRhaW5lci5xdWVyeVNlbGVjdG9yKCcjZW1wdHktc3RhdGUtYWRkLWJ1dHRvbicpO1xyXG4gICAgICAgICAgICBpZiAoZW1wdHlTdGF0ZUJ1dHRvbikge1xyXG4gICAgICAgICAgICAgICAgZW1wdHlTdGF0ZUJ1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCBhZGRNZW1vcnlCdXR0b24gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYWRkLW1lbW9yeS1idXR0b24nKTtcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCBhZGRNZW1vcnlTZWN0aW9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2FkZC1tZW1vcnktc2VjdGlvbicpO1xyXG4gICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgIGFkZE1lbW9yeVNlY3Rpb24uc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XHJcbiAgICAgICAgICAgICAgICAgICAgYWRkTWVtb3J5QnV0dG9uLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XHJcbiAgICAgICAgICAgICAgICAgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ25ldy1tZW1vcnktaW5wdXQnKS5mb2N1cygpO1xyXG4gICAgICAgICAgICAgICAgfSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuO1xyXG4gICAgfVxyXG4gICAgXHJcbiAgICAvLyBSZWd1bGFyIGRpc3BsYXkgb2YgbWVtb3JpZXMgdXNpbmcgdGVtcGxhdGVcclxuICAgIGNvbnN0IG1lbW9yeUNhcmRUZW1wbGF0ZSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdtZW1vcnktY2FyZC10ZW1wbGF0ZScpO1xyXG4gICAgaWYgKCFtZW1vcnlDYXJkVGVtcGxhdGUpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKCdNZW1vcnkgY2FyZCB0ZW1wbGF0ZSBub3QgZm91bmQnKTtcclxuICAgICAgICByZXR1cm47XHJcbiAgICB9XHJcbiAgICBcclxuICAgIGNvbnRhaW5lci5pbm5lckhUTUwgPSAnJztcclxuICAgIFxyXG4gICAgLy8gR3JvdXAgbWVtb3JpZXMgYnkgdGFnXHJcbiAgICBjb25zdCBncm91cGVkTWVtb3JpZXMgPSB7fTtcclxuICAgIG1lbW9yaWVzVG9EaXNwbGF5LmZvckVhY2gobWVtb3J5ID0+IHtcclxuICAgICAgICBjb25zdCB0YWcgPSBtZW1vcnkudGFnIHx8ICdHZW5lcmFsJztcclxuICAgICAgICBpZiAoIWdyb3VwZWRNZW1vcmllc1t0YWddKSB7XHJcbiAgICAgICAgICAgIGdyb3VwZWRNZW1vcmllc1t0YWddID0gW107XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGdyb3VwZWRNZW1vcmllc1t0YWddLnB1c2gobWVtb3J5KTtcclxuICAgIH0pO1xyXG4gICAgXHJcbiAgICAvLyBTb3J0IHRhZyBncm91cHM6IHB1dCAnR2VuZXJhbCcgYXQgdGhlIGVuZCwgb3RoZXIgdGFncyBhbHBoYWJldGljYWxseVxyXG4gICAgY29uc3QgdGFnTmFtZXMgPSBPYmplY3Qua2V5cyhncm91cGVkTWVtb3JpZXMpLnNvcnQoKGEsIGIpID0+IHtcclxuICAgICAgICBpZiAoYSA9PT0gJ0dlbmVyYWwnKSByZXR1cm4gMTtcclxuICAgICAgICBpZiAoYiA9PT0gJ0dlbmVyYWwnKSByZXR1cm4gLTE7XHJcbiAgICAgICAgcmV0dXJuIGEubG9jYWxlQ29tcGFyZShiKTtcclxuICAgIH0pO1xyXG4gICAgXHJcbiAgICB0YWdOYW1lcy5mb3JFYWNoKHRhZ05hbWUgPT4ge1xyXG4gICAgICAgIGNvbnN0IG1lbW9yaWVzSW5Hcm91cCA9IGdyb3VwZWRNZW1vcmllc1t0YWdOYW1lXTtcclxuICAgICAgICBcclxuICAgICAgICAvLyBDcmVhdGUgdGFnIHNlY3Rpb24gZWxlbWVudFxyXG4gICAgICAgIGNvbnN0IHNlY3Rpb25EaXYgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcclxuICAgICAgICBzZWN0aW9uRGl2LmNsYXNzTmFtZSA9ICd0YWctZ3JvdXAtc2VjdGlvbiBtYi00IGJvcmRlciBib3JkZXItZ3JheS0yMDAvNTAgcm91bmRlZC14bCBvdmVyZmxvdy1oaWRkZW4gYmctZ3JheS01MC8yMCBzaGFkb3ctc20nO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEhlYWRlclxyXG4gICAgICAgIGNvbnN0IGhlYWRlckRpdiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xyXG4gICAgICAgIGhlYWRlckRpdi5jbGFzc05hbWUgPSAndGFnLWdyb3VwLWhlYWRlciBmbGV4IGl0ZW1zLWNlbnRlciBqdXN0aWZ5LWJldHdlZW4gcHgtMyBweS0yIGJnLWdyYXktMTAwLzcwIGhvdmVyOmJnLWdyYXktMTAwIHRyYW5zaXRpb24tY29sb3JzIGN1cnNvci1wb2ludGVyIHNlbGVjdC1ub25lIGJvcmRlci1iIGJvcmRlci1ncmF5LTIwMC8zMCc7XHJcbiAgICAgICAgaGVhZGVyRGl2LmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgPGRpdiBjbGFzcz1cImZsZXggaXRlbXMtY2VudGVyIGdhcC0yXCI+XHJcbiAgICAgICAgICAgICAgICA8c3ZnIGNsYXNzPVwiY2hldnJvbi1pY29uIHRyYW5zZm9ybSB0cmFuc2l0aW9uLXRyYW5zZm9ybSBkdXJhdGlvbi0yMDAgdy0zIGgtMyB0ZXh0LWdyYXktNTAwXCIgdmlld0JveD1cIjAgMCAyNCAyNFwiIGZpbGw9XCJub25lXCIgc3Ryb2tlPVwiY3VycmVudENvbG9yXCIgc3Ryb2tlLXdpZHRoPVwiMi41XCIgc3R5bGU9XCJ0cmFuc2Zvcm06IHJvdGF0ZSg5MGRlZyk7XCI+XHJcbiAgICAgICAgICAgICAgICAgICAgPHBvbHlsaW5lIHBvaW50cz1cIjkgMTggMTUgMTIgOSA2XCI+PC9wb2x5bGluZT5cclxuICAgICAgICAgICAgICAgIDwvc3ZnPlxyXG4gICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9XCJmb250LWJvbGQgdGV4dC14cyB0ZXh0LWdyYXktNzAwXCI+JHt0YWdOYW1lfTwvc3Bhbj5cclxuICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPVwidGV4dC1bMTBweF0gdGV4dC1ncmF5LTUwMCBiZy13aGl0ZSBweC0yIHB5LTAuNSByb3VuZGVkLWZ1bGwgYm9yZGVyIGJvcmRlci1ncmF5LTIwMC82MCBmb250LXNlbWlib2xkXCI+JHttZW1vcmllc0luR3JvdXAubGVuZ3RofTwvc3Bhbj5cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgYDtcclxuICAgICAgICBcclxuICAgICAgICAvLyBDb250ZW50IGNvbnRhaW5lclxyXG4gICAgICAgIGNvbnN0IGNvbnRlbnREaXYgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcclxuICAgICAgICBjb250ZW50RGl2LmNsYXNzTmFtZSA9ICd0YWctZ3JvdXAtY29udGVudCBwLTMgZmxleCBmbGV4LWNvbCBnYXAtMiB0cmFuc2l0aW9uLWFsbCBkdXJhdGlvbi0yMDAnO1xyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIFRvZ2dsZSBiZWhhdmlvclxyXG4gICAgICAgIGhlYWRlckRpdi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcclxuICAgICAgICAgICAgY29uc3QgY2hldnJvbiA9IGhlYWRlckRpdi5xdWVyeVNlbGVjdG9yKCcuY2hldnJvbi1pY29uJyk7XHJcbiAgICAgICAgICAgIGlmIChjb250ZW50RGl2LnN0eWxlLmRpc3BsYXkgPT09ICdub25lJykge1xyXG4gICAgICAgICAgICAgICAgY29udGVudERpdi5zdHlsZS5kaXNwbGF5ID0gJ2ZsZXgnO1xyXG4gICAgICAgICAgICAgICAgY2hldnJvbi5zdHlsZS50cmFuc2Zvcm0gPSAncm90YXRlKDkwZGVnKSc7XHJcbiAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBjb250ZW50RGl2LnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XHJcbiAgICAgICAgICAgICAgICBjaGV2cm9uLnN0eWxlLnRyYW5zZm9ybSA9ICdyb3RhdGUoMGRlZyknO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgXHJcbiAgICAgICAgLy8gUG9wdWxhdGUgZ3JvdXAgd2l0aCBjYXJkc1xyXG4gICAgICAgIG1lbW9yaWVzSW5Hcm91cC5mb3JFYWNoKG1lbW9yeSA9PiB7XHJcbiAgICAgICAgICAgIC8vIENsb25lIHRoZSB0ZW1wbGF0ZVxyXG4gICAgICAgICAgICBjb25zdCBjYXJkRWxlbWVudCA9IG1lbW9yeUNhcmRUZW1wbGF0ZS5jbG9uZU5vZGUodHJ1ZSk7XHJcbiAgICAgICAgICAgIGNhcmRFbGVtZW50LmlkID0gJyc7IC8vIFJlbW92ZSB0ZW1wbGF0ZSBJRFxyXG4gICAgICAgICAgICBjYXJkRWxlbWVudC5jbGFzc0xpc3QucmVtb3ZlKCdoaWRkZW4nKTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIC8vIFBvcHVsYXRlIHRoZSBjYXJkIHdpdGggbWVtb3J5IGRhdGFcclxuICAgICAgICAgICAgY29uc3QgbWVtb3J5VGV4dCA9IGNhcmRFbGVtZW50LnF1ZXJ5U2VsZWN0b3IoJ3NwYW5bY29udGVudGVkaXRhYmxlXScpO1xyXG4gICAgICAgICAgICBjb25zdCBtZW1vcnlEYXRlID0gY2FyZEVsZW1lbnQucXVlcnlTZWxlY3RvcignLm1lbW9yeS1kYXRlJyk7XHJcbiAgICAgICAgICAgIGNvbnN0IG5ld1RhZ0JhZGdlID0gY2FyZEVsZW1lbnQucXVlcnlTZWxlY3RvcignI25ldy10YWctdGVtcGxhdGUnKTtcclxuICAgICAgICAgICAgY29uc3QgZWRpdEJ1dHRvbiA9IGNhcmRFbGVtZW50LnF1ZXJ5U2VsZWN0b3IoJ2J1dHRvblt0aXRsZT1cIkVkaXRcIl0nKTtcclxuICAgICAgICAgICAgY29uc3QgZGVsZXRlQnV0dG9uID0gY2FyZEVsZW1lbnQucXVlcnlTZWxlY3RvcignYnV0dG9uW3RpdGxlPVwiRGVsZXRlXCJdJyk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBUYWcgYmFkZ2UgaW5zaWRlIGNhcmRcclxuICAgICAgICAgICAgY29uc3QgdGFnQmFkZ2UgPSBjYXJkRWxlbWVudC5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LXRhZy1iYWRnZScpO1xyXG4gICAgICAgICAgICBjb25zdCB0YWdFZGl0SW5wdXQgPSBjYXJkRWxlbWVudC5xdWVyeVNlbGVjdG9yKCcubWVtb3J5LXRhZy1lZGl0LWlucHV0Jyk7XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTZXQgbWVtb3J5IGNvbnRlbnRcclxuICAgICAgICAgICAgaWYgKG1lbW9yeVRleHQpIG1lbW9yeVRleHQudGV4dENvbnRlbnQgPSBtZW1vcnkubWVtb3J5X3RleHQ7XHJcbiAgICAgICAgICAgIGlmIChtZW1vcnlEYXRlKSBtZW1vcnlEYXRlLnRleHRDb250ZW50ID0gZm9ybWF0RGF0ZShtZW1vcnkudGltZXN0YW1wKTtcclxuICAgICAgICAgICAgXHJcbiAgICAgICAgICAgIGlmICh0YWdCYWRnZSkge1xyXG4gICAgICAgICAgICAgICAgaWYgKG1lbW9yeS50YWcpIHtcclxuICAgICAgICAgICAgICAgICAgICB0YWdCYWRnZS50ZXh0Q29udGVudCA9IG1lbW9yeS50YWc7XHJcbiAgICAgICAgICAgICAgICAgICAgdGFnQmFkZ2UuY2xhc3NMaXN0LnJlbW92ZSgnaGlkZGVuJyk7XHJcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICAgICAgICAgIHRhZ0JhZGdlLmNsYXNzTGlzdC5hZGQoJ2hpZGRlbicpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBIYW5kbGUgbmV3IHRhZyB3aXRoIHVuaXF1ZSBJRCBhbmQgcHJvcGVyIHRpbWVzdGFtcCBsb2dpY1xyXG4gICAgICAgICAgICBpZiAobmV3VGFnQmFkZ2UpIHtcclxuICAgICAgICAgICAgICAgIGNvbnN0IHVuaXF1ZVRhZ0lkID0gYG5ldy10YWctJHttZW1vcnkuaWR9YDtcclxuICAgICAgICAgICAgICAgIG5ld1RhZ0JhZGdlLmlkID0gdW5pcXVlVGFnSWQ7XHJcbiAgICAgICAgICAgICAgICBuZXdUYWdCYWRnZS5jbGFzc0xpc3QuYWRkKCdoaWRkZW4nKTsgLy8gRW5zdXJlIGl0J3MgaGlkZGVuIGZpcnN0XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIC8vIE9ubHkgc2hvdyAnTmV3JyB0YWcgZm9yIG1lbW9yaWVzIHRoYXQgd2VyZSBjcmVhdGVkIHJlY2VudGx5IChsZXNzIHRoYW4gMzAgbWludXRlcyBhZ28pXHJcbiAgICAgICAgICAgICAgICBjb25zdCBub3cgPSBEYXRlLm5vdygpO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgdGltZURpZmYgPSBub3cgLSBtZW1vcnkudGltZXN0YW1wO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgdGhpcnR5TWludXRlcyA9IDMwICogNjAgKiAxMDAwO1xyXG4gICAgICAgICAgICAgICAgY29uc3QgaXNSZWNlbnQgPSB0aW1lRGlmZiA8IHRoaXJ0eU1pbnV0ZXM7XHJcbiAgICAgICAgICAgICAgICBcclxuICAgICAgICAgICAgICAgIGlmIChpc1JlY2VudCkge1xyXG4gICAgICAgICAgICAgICAgICAgIG5ld1RhZ0JhZGdlLmNsYXNzTGlzdC5yZW1vdmUoJ2hpZGRlbicpO1xyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAvLyBTZXQgZGF0YSBhdHRyaWJ1dGVzIGZvciBidXR0b25zXHJcbiAgICAgICAgICAgIGlmIChlZGl0QnV0dG9uKSBlZGl0QnV0dG9uLmRhdGFzZXQuaWQgPSBtZW1vcnkuaWQ7XHJcbiAgICAgICAgICAgIGlmIChkZWxldGVCdXR0b24pIGRlbGV0ZUJ1dHRvbi5kYXRhc2V0LmlkID0gbWVtb3J5LmlkO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICAgICAgLy8gQWRkIGV2ZW50IGxpc3RlbmVyc1xyXG4gICAgICAgICAgICBpZiAoZWRpdEJ1dHRvbikge1xyXG4gICAgICAgICAgICAgICAgZWRpdEJ1dHRvbi5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICBjb25zdCB0ZXh0RWxlbWVudCA9IGNhcmRFbGVtZW50LnF1ZXJ5U2VsZWN0b3IoJ3NwYW5bY29udGVudGVkaXRhYmxlXScpO1xyXG4gICAgICAgICAgICAgICAgICAgIGlmICh0ZXh0RWxlbWVudC5nZXRBdHRyaWJ1dGUoJ2NvbnRlbnRlZGl0YWJsZScpID09PSAndHJ1ZScpIHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgY29uc3QgbmV3VGFnID0gdGFnRWRpdElucHV0ID8gdGFnRWRpdElucHV0LnZhbHVlLnRyaW0oKSA6IG51bGw7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHNhdmVFZGl0KG1lbW9yeS5pZCwgdGV4dEVsZW1lbnQsIGVkaXRCdXR0b24sIG1lbW9yeS5tZW1vcnlfdGV4dCwgbmV3VGFnLCB0YWdFZGl0SW5wdXQpO1xyXG4gICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRleHRFbGVtZW50LnNldEF0dHJpYnV0ZSgnY29udGVudGVkaXRhYmxlJywgJ3RydWUnKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgdGV4dEVsZW1lbnQuY2xhc3NMaXN0LmFkZCgnYmctZ3JheS01MCcsICdib3JkZXInLCAnYm9yZGVyLWdyYXktMzAwJywgJ3JvdW5kZWQnLCAncC0yJyk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHRleHRFbGVtZW50LmZvY3VzKCk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodGFnRWRpdElucHV0KSB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB0YWdFZGl0SW5wdXQudmFsdWUgPSBtZW1vcnkudGFnIHx8ICcnO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdGFnRWRpdElucHV0LmNsYXNzTGlzdC5yZW1vdmUoJ2hpZGRlbicpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgICAgICAgICAgICAgICBlZGl0QnV0dG9uLmlubmVySFRNTCA9IGBcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9XCIxNFwiIGhlaWdodD1cIjE0XCIgdmlld0JveD1cIjAgMCAyNCAyNFwiIGZpbGw9XCJub25lXCIgc3Ryb2tlPVwiY3VycmVudENvbG9yXCIgc3Ryb2tlLXdpZHRoPVwiMlwiPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9XCJNNSAxM2w0IDRMMTkgN1wiPjwvcGF0aD5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDwvc3ZnPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICBgO1xyXG4gICAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBpZiAoZGVsZXRlQnV0dG9uKSB7XHJcbiAgICAgICAgICAgICAgICBkZWxldGVCdXR0b24uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgY29uc29sZS5sb2coJ0RlbGV0ZSBidXR0b24gY2xpY2tlZCBmb3IgbWVtb3J5IElEOicsIG1lbW9yeS5pZCk7XHJcbiAgICAgICAgICAgICAgICAgICAgZGVsZXRlTWVtb3J5KG1lbW9yeS5pZCwgbWVtb3J5Lm1lbW9yeV90ZXh0KTtcclxuICAgICAgICAgICAgICAgIH0pO1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIFxyXG4gICAgICAgICAgICBjb250ZW50RGl2LmFwcGVuZENoaWxkKGNhcmRFbGVtZW50KTtcclxuICAgICAgICB9KTtcclxuICAgICAgICBcclxuICAgICAgICBzZWN0aW9uRGl2LmFwcGVuZENoaWxkKGhlYWRlckRpdik7XHJcbiAgICAgICAgc2VjdGlvbkRpdi5hcHBlbmRDaGlsZChjb250ZW50RGl2KTtcclxuICAgICAgICBjb250YWluZXIuYXBwZW5kQ2hpbGQoc2VjdGlvbkRpdik7XHJcbiAgICB9KTtcclxufVxyXG5cclxuXHJcblxyXG5mdW5jdGlvbiB1cGRhdGVQYWdpbmF0aW9uQ29udHJvbHMoKSB7XHJcbiAgICBjb25zdCBwcmV2QnV0dG9uID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3ByZXYtcGFnZScpO1xyXG4gICAgY29uc3QgbmV4dEJ1dHRvbiA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCduZXh0LXBhZ2UnKTtcclxuICAgIGNvbnN0IHBhZ2VJbmZvID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3BhZ2UtaW5mbycpO1xyXG4gICAgY29uc3QgcGFnaW5hdGlvbkNvbnRyb2xzID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3BhZ2luYXRpb24tY29udHJvbHMnKTtcclxuXHJcbiAgICAvLyBIaWRlIHBhZ2luYXRpb24gY29udHJvbHMgd2hlbiB0aGVyZSBhcmUgbm8gbWVtb3JpZXNcclxuICAgIGlmIChhbGxNZW1vcmllc0RhdGEubGVuZ3RoID09PSAwKSB7XHJcbiAgICAgICAgcGFnaW5hdGlvbkNvbnRyb2xzLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XHJcbiAgICAgICAgcmV0dXJuO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgICBwYWdpbmF0aW9uQ29udHJvbHMuc3R5bGUuZGlzcGxheSA9ICdmbGV4JztcclxuICAgIH1cclxuXHJcbiAgICBwcmV2QnV0dG9uLmRpc2FibGVkID0gY3VycmVudFBhZ2UgPT09IDE7XHJcbiAgICBuZXh0QnV0dG9uLmRpc2FibGVkID0gY3VycmVudFBhZ2UgPT09IHRvdGFsUGFnZXM7XHJcbiAgICBwYWdlSW5mby50ZXh0Q29udGVudCA9IGBQYWdlICR7Y3VycmVudFBhZ2V9IG9mICR7dG90YWxQYWdlc31gO1xyXG59XHJcblxyXG4iXSwibmFtZXMiOlsic2lnbmluQnV0dG9uIiwidXBncmFkZUJ1dHRvbiIsImJhbm5lckljb24iLCJiYW5uZXJUaXRsZSIsImJhbm5lclRleHQiLCJ0aXRsZSIsIm1lc3NhZ2UiXSwibWFwcGluZ3MiOiJBQUlBLE1BQU0sZ0JBQWdCO0FBQUEsRUFDbEIsTUFBTSxpQkFBaUIsU0FBUyxTQUFTO0FBQ3JDLFdBQU8sT0FBTyxRQUFRLFlBQVk7QUFBQSxNQUM5QixNQUFNO0FBQUEsTUFDTjtBQUFBLElBQ1osQ0FBUztBQUFBLEVBQ0w7QUFBQSxFQUVBLE1BQU0sY0FBYztBQUNoQixRQUFJO0FBQ0EsWUFBTSxXQUFXLE1BQU0sT0FBTyxRQUFRLFlBQVksRUFBRSxNQUFNLGdCQUFlLENBQUU7QUFDM0UsVUFBSSxTQUFTLFdBQVcsV0FBVztBQUMvQixlQUFPLFNBQVM7QUFBQSxNQUNwQixPQUFPO0FBQ0gsY0FBTSxJQUFJLE1BQU0sU0FBUyxXQUFXLHlCQUF5QjtBQUFBLE1BQ2pFO0FBQUEsSUFDSixTQUFTLE9BQU87QUFDWixjQUFRLE1BQU0sNEJBQTRCLEtBQUs7QUFDL0MsWUFBTTtBQUFBLElBQ1Y7QUFBQSxFQUNKO0FBQUEsRUFFQSw4QkFBOEIsYUFBYTtBQUN2QyxXQUFPLFFBQVEsWUFBWTtBQUFBLE1BQ3ZCLE1BQU07QUFBQSxNQUNOO0FBQUEsSUFDWixDQUFTLEVBQUUsTUFBTSxNQUFNO0FBQUEsSUFBQyxDQUFDO0FBQUEsRUFDckI7QUFBQSxFQUVBLGFBQWEsUUFBUTtBQUNqQixXQUFPLFFBQVEsWUFBWTtBQUFBLE1BQ3ZCLE1BQU07QUFBQSxNQUNOO0FBQUEsSUFDWixDQUFTLEVBQUUsTUFBTSxNQUFNO0FBQUEsSUFBQyxDQUFDO0FBQUEsRUFDckI7QUFBQSxFQUVBLG9CQUFvQixRQUFRO0FBQ3hCLFdBQU8sUUFBUSxZQUFZO0FBQUEsTUFDdkIsTUFBTTtBQUFBLE1BQ047QUFBQSxJQUNaLENBQVMsRUFBRSxNQUFNLE1BQU07QUFBQSxJQUFDLENBQUM7QUFBQSxFQUNyQjtBQUFBLEVB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
