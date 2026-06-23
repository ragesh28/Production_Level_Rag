var core = {
  "start": function () {
    core.load();
  },
  "install": function () {
    core.load();
  },
  "load": function () {
    core.update.addon(false, false);
  },
  "register": {
    "listeners": function () {
      app.webrequest.on.headers.received.remove();
      if (config.addon.state === "active") {
        app.webrequest.on.headers.received.add({"urls" : ["*://*/*"]});
      }
    }
  },
  "update": {
    "popup": function (list) {
      const top = config.session.current.domain;
      /*  */
      app.popup.post("load", {
        "top": top,
        "theme": config.addon.theme,
        "state": config.addon.state,
        "list": top && list ? list[top] : [],
        "metadata": config.session.global.metadata
      });
    },
    "session": {
      "objects": function (tab) {
        const id = tab.id;
        const url = config.session.top.url;
        const title = config.session.top.title;
        const removed = config.session.top.removed;
        /*  */
        if (removed[id] === undefined) {
          if (tab.url) url[id] = tab.url;
          if (tab.title) title[id] = tab.title;
        }
        /*  */
        config.session.top.url = url;
        config.session.top.title = title;
        config.session.current.domain = tab.url ? config.download.item.extract.domain(tab.url) : '';
      }
    },
    "addon": function (notify, popup) {
      app.tab.query.active(function (tab) {
        if (config.addon.state === "active") {
          if (tab) {
            if (tab.url) {
              const list = config.media.list;
              const domain = config.download.item.extract.domain(tab.url);
              const count = list[domain] && list[domain].length ? list[domain].length : 0;
              /*  */
              config.session.current.domain = domain;
              core.update.button(tab.id, count, notify);
              if (popup) {
                core.update.popup(list);
              }
            }
          }
        } else {
          core.update.button(tab.id, 0, notify);
        }
      });
    },
    "button": function (tabId, number, notify) {
      const found = number > 0 && typeof number === "number";
      /*  */
      if (config.addon.state === "active") {
        app.button.icon(tabId, config.addon.state);
        app.button.badge.text(tabId, config.addon.state === "inactive" ? '' : (found ? number : ''));
        app.button.title(tabId, config.addon.state === "inactive" ? "Video & Audio Downloader" : (found ? number + " media found!" : "No media found!"));
        /*  */
        if (notify) {
          if (found === false) {            
            app.notifications.create({
              "title": "Video & Audio Downloader",
              "message": "No media found! please browse to a website with media content(s) and try again."
            });
          }
        }
      } else {
        app.button.badge.text(tabId, '');
        app.button.icon(tabId, config.addon.state);
        app.button.title(tabId, "Video & Audio Downloader is disabled");
        /*  */
        if (notify) {
          app.notifications.create({
            "title": "Video & Audio Downloader",
            "message": "The addon is disabled! please enable the addon and try again."
          });
        }
      }
    }
  },
  "action": {
    "storage": function (changes, namespace) {
      /*  */
    },
    "process": function (info) {
      const id = info.tabId;
      const type = info.type;
      const current = info.url;
      const frameId = info.frameId;
      const docurl = info.documentUrl;
      const initiator = info.initiator;
      const url = config.session.top.url;
      const title = config.session.top.title;
      /*  */
      const tmp = type === "main_frame" ? current : (initiator && frameId === 0 ? initiator : (docurl ? docurl : ''));
      if (tmp) url[id] = tmp;
      /*  */
      if (config.download.has.permission(url[id], current)) {
        const header = config.query.header(info.responseHeaders);
        /*  */
        header.url = current;
        header.rtype = info.type || '';
        header.extension = config.url.get.media.extension(header.url, header.type);
        /*  */
        const flag_1 = header.tag.indexOf("video") !== -1 || header.tag.indexOf("audio") !== -1;
        const flag_2 = header.type.indexOf("video") !== -1 || header.type.indexOf("audio") !== -1;
        const flag_3 = header.rtype.indexOf("video") !== -1 || header.rtype.indexOf("audio") !== -1;
        /*  */
        if (header.extension || flag_1 || flag_2 || flag_3) {
          if (!header.size || parseInt(header.size) > 100000) {
            header.originalsize = header.size || '0';
            header.size = config.url.get.media.size(header.size) || '?';
            header.duration = config.url.get.media.duration(header.time);
            /*  */
            if (config.download.item.is.valid(url[id], header.url, header.size)) {
              const cond = title[id] !== undefined;
              const cond_1 = cond && title[id].indexOf("://") === -1;
              const cond_2 = cond && title[id].indexOf("www.") === -1;
              /*  */
              header.title = cond_1 && cond_2 ? title[id] : '';
              config.download.item.add(url[id], header);
            }
          }
        }
      }
      /*  */
      config.session.top.url = url;
    },
    "popup": {
      "theme": function (e) {
        config.addon.theme = e;
      },
      "load": function () {
        core.update.addon(true, false);
        core.update.popup(config.media.list);
      },
      "store": function (e) {
        const tmp = config.media.list;
        tmp[config.session.current.domain] = e.list;
        config.media.list = tmp;
        /*  */
        core.update.popup(tmp);
      },
      "clear": function () {
        const tmp = config.media.list;
        tmp[config.session.current.domain] = [];
        config.media.list = tmp;
        /*  */
        app.popup.send("reload");
      },
      "toggle": function () {
        if (config.addon.state === "active") {
          config.addon.state = "inactive";
          core.update.popup(null);
        } else {
          const list = config.media.list;
          config.addon.state = "active";
          core.update.popup(list);
        }
        /*  */
        core.register.listeners();
        core.update.addon(true, false);
      },
      "download": {
        "one": function (e) {
          app.downloads.start({
            "url": e.url,
            "filename": e.filename + e.extension
          });
          /*  */
          app.notifications.create({
            "title": "Video & Audio Downloader",
            "message": "Downloading media as: " + e.filename + e.extension
          });
        },
        "all": function (e) {
          app.downloads.start({
            "url": e.url,
            "filename": e.filename
          });
          /*  */
          app.notifications.create({
            "title": "Video & Audio Downloader",
            "message": "Downloading " + e.type + " file as: " + e.filename
          });
        }
      },
      "tab": {
        "activated": function (tab) {
          core.update.session.objects(tab);
          core.update.addon(false, false);
        },
        "removed": function (tabId) {
          const removed = config.session.top.removed;
          delete removed[tabId];
          config.session.top.removed = removed;
        },
        "updated": function (tab) {
          const updated = config.session.top.updated;
          /*  */
          if (tab.url) {
            if (updated[tab.id] !== tab.url) {
              updated[tab.id] = tab.url;
              core.update.addon(false, false);
            }
          }
          /*  */
          core.update.session.objects(tab);
          config.session.top.updated = updated;
        }
      }
    }
  }
};

app.storage.load(core.register.listeners);
app.webrequest.on.headers.received.callback(core.action.process);

app.tab.on.removed(core.action.popup.tab.removed);
app.tab.on.updated(core.action.popup.tab.updated);
app.tab.on.activated(core.action.popup.tab.activated);

app.popup.receive("load", core.action.popup.load);
app.popup.receive("store", core.action.popup.store);
app.popup.receive("clear", core.action.popup.clear);
app.popup.receive("theme", core.action.popup.theme);
app.popup.receive("toggle", core.action.popup.toggle);
app.popup.receive("download-one-item", core.action.popup.download.one);
app.popup.receive("download-all-items", core.action.popup.download.all);
app.popup.receive("support", function () {app.tab.open(app.homepage())});
app.popup.receive("metadata", function (e) {config.session.global.metadata = e});
app.popup.receive("audio-joiner", function () {app.tab.open(config.page["audio-joiner"])});
app.popup.receive("donation", function () {app.tab.open(app.homepage() + "?reason=support")});
app.popup.receive("convert-to-mp3", function () {app.tab.open(config.page["convert-to-mp3"])});
app.popup.receive("video-audio-muxer", function () {app.tab.open(config.page["video-audio-muxer"])});

app.on.startup(core.start);
app.on.installed(core.install);
app.on.storage(core.action.storage);
