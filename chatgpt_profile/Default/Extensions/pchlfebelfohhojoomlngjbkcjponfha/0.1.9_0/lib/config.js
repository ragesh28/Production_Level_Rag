var config = {};

config.page = {
  "audio-joiner": "https://webbrowsertools.com/audio-joiner/",
  "convert-to-mp3": "https://webbrowsertools.com/convert-to-mp3/",
  "video-audio-muxer": "https://webbrowsertools.com/video-audio-muxer/"
};

config.welcome = {
  set lastupdate (val) {app.storage.write("lastupdate", val)},
  get lastupdate () {return app.storage.read("lastupdate") !== undefined ? app.storage.read("lastupdate") : 0}
};

config.addon = {
  set state (val) {app.storage.write("state", val)},
  set theme (val) {app.storage.write("theme", val)},
  get theme () {return app.storage.read("theme") !== undefined ? app.storage.read("theme") : "light"},
  get state () {return app.storage.read("state") !== undefined ? app.storage.read("state") : "active"}
};

config.query = {
  "header": function (headers) {
    const header = {"tag": '', "time": '', "type": '', "size": '', "title": '', "pagetitle": ''};
    /*  */
    if (headers && headers.length) {
      for (let i = 0; i < headers.length; ++i) {
        const name = headers[i].name.toLowerCase();
        /*  */
        if (name.indexOf("-tg") !== -1) header.tag = headers[i].value;
        if (name.indexOf("type") !== -1) header.type = headers[i].value;
        if (name.indexOf("length") !== -1) header.size = headers[i].value;
        if (name.indexOf("timestamp") !== -1) header.time = headers[i].value;
        if (name.indexOf("disposition") !== -1) header.title = headers[i].value;
      }
    }
    /*  */
    return header;
  }
};

config.session = {
  "current": {
    set domain (val) {app.session.write("current-domain", val)},
    get domain () {return app.session.read("current-domain") !== undefined ? app.session.read("current-domain") : ''}
  },
  "global": {
    set metadata (val) {app.session.write("global-metadata", val)},
    set medialist (val) {app.session.write("global-medialist", val)},
    get metadata () {return app.session.read("global-metadata") !== undefined ? app.session.read("global-metadata") : {}},
    get medialist () {return app.session.read("global-medialist") !== undefined ? app.session.read("global-medialist") : {}}
  },
  "top": {
    set url (val) {app.session.write("top-url", val)},
    set title (val) {app.session.write("top-title", val)},
    set removed (val) {app.session.write("top-removed", val)},
    set updated (val) {app.session.write("top-updated", val)},
    get url () {return app.session.read("top-url") !== undefined ? app.session.read("top-url") : {}},
    get title () {return app.session.read("top-title") !== undefined ? app.session.read("top-title") : {}},
    get removed () {return app.session.read("top-removed") !== undefined ? app.session.read("top-removed") : {}},
    get updated () {return app.session.read("top-updated") !== undefined ? app.session.read("top-updated") : {}}
  }
};

config.media = {
  "extension": [
    "\\.mov", "\\.qt", "\\.wmv", "\\.yuv", "\\.rm", "\\.rmvb", "\\.m3u", "\\.mp4",
    "\\.mmf", "\\.mp3", "\\.mpc", "\\.ogg", "\\.oga", "\\.tta", "\\.wav", "\\.wma",
    "\\.3gp", "\\.3g2", "\\.mxf", "\\.roq", "\\.nsv", "\\.flv", "\\.f4v", "\\.f4p",
    "\\.webm", "\\.mkv", "\\.vob", "\\.ogv", "\\.ogg", "\\.drc", "\\.mng", "\\.avi",
    "\\.mp2", "\\.mpeg", "\\.mpe", "\\.mpv", "\\.mpg", "\\.m2v", "\\.m4v", "\\.svi",
    "\\.3gp", "\\.aac", "\\.aax", "\\.aiff", "\\.flac", "\\.m4a", "\\.m4b", "\\.m4p",
    "\\.f4a", "\\.f4b", "\\.m4p"
  ],
  get list () {
    return config.session.global.medialist;
  },
  set list (e) {
    delete e["undefined"];
    /*  */
    let size = 0, maxSize = 1000;
    for (let m in e) if (e.hasOwnProperty(m)) size++;
    if (size > maxSize) {
      size = 0;
      let rand = Math.round(Math.random() * maxSize);
      for (let m in e) {
        if (e.hasOwnProperty(m)) {
          let item = size++;
          if (item === rand) {
            delete e[m];
            break;
          }
        }
      }
    }
    /*  */
    config.session.global.medialist = e;
    core.update.addon(true, true);
  }
};

config.url = {
  "check": {
    "ad": new RegExp(
      [
        "[\\=\\&\\_\\-\\.\\/\\?\\s]ad[\\=\\&\\_\\-\\.\\/\\?\\s]",
        "[\\=\\&\\_\\-\\.\\/\\?\\s]ads[\\=\\&\\_\\-\\.\\/\\?\\s]",
        "[\\=\\&\\_\\-\\.\\/\\?\\s]pagead[\\=\\&\\_\\-\\.\\/\\?\\s\\d]",
        "\\.google\\-analytics\\.", "[\\.]php[\\=\\&\\_\\-\\.\\/\\?\\s\\%]",
        "\\_adam\\-", "\\-adam\\_", "\\-adam\\-", "\\&adid\\=", "\\.2mdn\\.", "\\&adfmt\\=",
        "\\.js", "\\.css", "\\.png", "\\.jpg", "\\.jpeg", "\\.woff", "\\%22ad", "\\/adam\\/",
        "\\/adServer\\/", "\\.doubleclick\\.", "\\.serving\\-sys.\\", "\\.googlesyndication\\.",
        "\\.atdmt\\.", "watch7ad\\_", "\\/adunit\\/", "\\=adhost\\&", "\\.innovid\\.", "\\/adsales\\/"
      ].join('|'), 'i'
    )
  },
  "get": {
    "media": {
      "duration": function (s) {
        const date = new Date();
        date.setSeconds(s / 1000);
        /*  */
        return date.toISOString().slice(11, 19) || '';
      },
      "extension": function (url, type) {
        const regexp = new RegExp(config.media.extension.join('|'), 'i');
        const ext = regexp.exec(url) || regexp.exec(type) || null;
        /*  */
        return ext && ext.length ? ext[0] : null;
      },
      "size": function (s) {
        if (s) {
          if (s >= Math.pow(2, 30)) {return (s / Math.pow(2, 30)).toFixed(1) + " GB"};
          if (s >= Math.pow(2, 20)) {return (s / Math.pow(2, 20)).toFixed(1) + " MB"};
          if (s >= Math.pow(2, 10)) {return (s / Math.pow(2, 10)).toFixed(1) + " KB"};
          return s + " B";
        } else {
          return '';
        }
      }
    }
  }
};

config.download = {
  "has": {
    "permission": function (top, url) {
      if (config.url.check.ad.test(url)) return false;
      if ((url.indexOf(".googlevideo.") !== -1) || (top && top.indexOf("www.youtube.") !== -1)) return false;
      /*  */
      return true;
    }
  },
  "item": {
    "add": function (top, e) {
      let tmp = config.media.list;
      let domain = config.download.item.extract.domain(top);
      /*  */
      let list = tmp[domain] || [];
      list.push(e);
      tmp[domain] = list;
      /*  */
      config.media.list = tmp;
    },
    "is": {
      "valid": function (top, url) {
        let tmp = config.media.list;
        let domain = config.download.item.extract.domain(top);
        /*  */
        let list = tmp[domain] || [];
        for (let i = 0; i < list.length; i++) {
          if (list[i].url === url) return false;
        }
        /*  */
        return true;
      }
    },
    "extract": {
      "domain": function (url) {
        let s = url.indexOf("//") + 2;
        if (s > 1) {
          let o = url.indexOf('/', s);
          if (o > 0) {
            return url.substring(s, o);
          } else {
            o = url.indexOf('?', s);
            return o > 0 ? url.substring(s, o) : url.substring(s);
          }
        } else {
          return url;
        }
      }
    }
  }
};
