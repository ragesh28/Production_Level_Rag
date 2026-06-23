(function () {
  'use strict';

  /* global chrome, browser */

  // Reads mp_login_timestamp from cookie or localStorage, reports to background
  (function () {
    try {
      const runtime = (typeof chrome !== 'undefined' && chrome.runtime) ||
        (typeof browser !== 'undefined' && browser.runtime);
      if (!runtime || !runtime.sendMessage) return;

      const getCookie = (name) => {
        try {
          const m = document.cookie.split('; ').find(r => r.startsWith(name + '='));
          return m ? m.split('=')[1] : null;
        } catch (_) { return null; }
      };

      const tsFromCookie = getCookie('mp_login_timestamp');
      let ts = tsFromCookie;
      if (!ts) {
        try { ts = localStorage.getItem('mp_login_timestamp'); } catch (_) { /* noop */ }
      } else {
        try { localStorage.setItem('mp_login_timestamp', ts); } catch (_) { /* noop */ }
      }

      if (ts) {
        runtime.sendMessage({ action: 'mp_probe_timestamp', ts });
      }
    } catch (_) {
      // Silent by design
    }
  })();

})();
