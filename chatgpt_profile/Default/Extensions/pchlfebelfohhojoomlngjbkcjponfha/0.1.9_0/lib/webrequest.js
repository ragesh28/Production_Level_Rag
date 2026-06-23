app.webrequest = {
  "on": {
    "headers": {
      "received": {
        "listener": null,
        "callback": function (callback) {
          app.webrequest.on.headers.received.listener = callback;
        },
        "remove": function () {
          if (chrome.webRequest) {
            chrome.webRequest.onHeadersReceived.removeListener(app.webrequest.on.headers.received.listener);
          }
        },
        "add": function (e) {
          const filter = e ? e : {"urls": ["*://*/*"]};
          const options = navigator.userAgent.indexOf("Firefox") !== -1 ? ["responseHeaders"] : ["responseHeaders", "extraHeaders"];
          /*  */
          if (chrome.webRequest) {
            chrome.webRequest.onHeadersReceived.removeListener(app.webrequest.on.headers.received.listener);
            chrome.webRequest.onHeadersReceived.addListener(app.webrequest.on.headers.received.listener, filter, options);
          }
        }
      }
    }
  }
};
