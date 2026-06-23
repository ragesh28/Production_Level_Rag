var background = {
  "port": null,
  "message": {},
  "receive": function (id, callback) {
    if (id) {
      background.message[id] = callback;
    }
  },
  "send": function (id, data) {
    if (id) {
      chrome.runtime.sendMessage({
        "method": id,
        "data": data,
        "path": "popup-to-background"
      }, function () {
        return chrome.runtime.lastError;
      });
    }
  },
  "connect": function (port) {
    chrome.runtime.onMessage.addListener(background.listener); 
    /*  */
    if (port) {
      background.port = port;
      background.port.onMessage.addListener(background.listener);
      background.port.onDisconnect.addListener(function () {
        background.port = null;
      });
    }
  },
  "post": function (id, data) {
    if (id) {
      if (background.port) {
        background.port.postMessage({
          "method": id,
          "data": data,
          "path": "popup-to-background",
          "port": background.port.name
        });
      }
    }
  },
  "listener": function (e) {
    if (e) {
      for (let id in background.message) {
        if (background.message[id]) {
          if ((typeof background.message[id]) === "function") {
            if (e.path === "background-to-popup") {
              if (e.method === id) {
                background.message[id](e.data);
              }
            }
          }
        }
      }
    }
  }
};

var config = {
  "listener": {
    "toggle": function () {
      config.download.index.video = config.download.index.audio = 0;
      background.send("toggle", config.download.stream);
    },
    "clear": function () {
      if (config.download.stream) {
        mscConfirm("Clear", "Are you sure you want to clear media list?", function () {
          background.send("clear", config.download.stream);
        });
      }
    }
  },
  "load": function () {
    const theme = document.getElementById("theme");
    const support = document.getElementById("support");
    const reload = document.getElementById("reload-ui");
    const donation = document.getElementById("donation");
    const joiner = document.getElementById("audio-joiner");
    const toggle = document.getElementById("toggle-addon");
    const clear = document.getElementById("clear-media-list");
    const convert = document.getElementById("convert-to-mp3");
    const muxer = document.getElementById("video-audio-muxer");
    /*  */
    config.download.button.audio = document.getElementById("audio-download");
    config.download.button.video = document.getElementById("video-download");
    config.download.button.audio.addEventListener("click", config.download.action);
    config.download.button.video.addEventListener("click", config.download.action);
    /*  */
    clear.addEventListener("click", config.listener.clear);
    toggle.addEventListener("click", config.listener.toggle);
    reload.addEventListener("click", function () {document.location.reload()});
    support.addEventListener("click", function () {background.send("support")});
    donation.addEventListener("click", function () {background.send("donation")});
    joiner.addEventListener("click", function () {background.send("audio-joiner")});
    convert.addEventListener("click", function () {background.send("convert-to-mp3")});
    muxer.addEventListener("click", function () {background.send("video-audio-muxer")});
    /*  */
    theme.addEventListener("click", function () {
      let attribute = document.documentElement.getAttribute("theme");
      attribute = attribute === "dark" ? "light" : "dark";
      /*  */
      document.documentElement.setAttribute("theme", attribute);
      background.send("theme", attribute);
    });
    /*  */
    background.send("load");
    window.removeEventListener("load", config.load, false);
  },
  "download": {
    "zip": {},
    "meta": {},
    "stream": {},    
    "button": {},
    "timeout": {},
    "index": {
      "audio": 0, 
      "video": 0
    },
    "extension": {
      "list": {
        "audio": [
          ".ra",
          ".au",
          ".wv",
          ".wav",
          ".bwf",
          ".raw",
          ".m4a",
          ".pac",
          ".tta",
          ".3gp",
          ".act",
          ".dct",
          ".dss",
          ".gsm",
          ".m4p",
          ".mmf",
          ".ast",
          ".aac",
          ".mp2",
          ".mp3",
          ".mp4",
          ".amr",
          ".s3m",
          ".mpc",
          ".ogg",
          ".oga",
          ".sln",
          ".vox",
          ".opus",
          ".aiff",
          ".flac",
          ".weba"
        ],
        "video": [
          ".qt",
          ".rm",
          ".mkv",
          ".flv",
          ".vob",
          ".ogv",
          ".ogg",
          ".rrc",
          ".mng",
          ".mov",
          ".avi",
          ".wmv",
          ".yuv",
          ".asf",
          ".amv",
          ".mp4",
          ".m4p",
          ".m4v",
          ".mpg",
          ".mp2",
          ".mpe",
          ".mpv",
          ".m4v",
          ".svi",
          ".3gp",
          ".3g2",
          ".mxf",
          ".roq",
          ".nsv",
          ".flv",
          ".f4v",
          ".f4p",
          ".f4a",
          ".f4b",
          ".mod",
          ".gifv",
          ".mpeg",
          ".webm"
        ]
      }
    },
    "action": function () {
      const button = this;
      const id = button.getAttribute("id");
      const type = id.replace("-download", '');
      if (type) {
        mscConfirm("Download All", "Are you sure you want to download all " + type + " items?", async function () {
          const items = [];
          const names = [];
          const list = config.download.stream.list;
          /*  */
          if (list && list.length) {
            const info = button.querySelector(".info");
            /*  */
            button.setAttribute("disabled", '');
            for (let i = 0; i < list.length; i++) {
              if (list[i].type.indexOf(type) !== -1) {
                names.push(config.download.filename.make(i));
                items.push(list[i].url);
              }
            }
            /*  */
            if (items.length) {
              const result = {
                "type": type,
                "items": items,
                "names": names,
                "filename": config.download.stream.top,
                "extension": list[0].extension ? list[0].extension : (type === "video" ? ".mp4" : ".mp3")
              };
              /*  */
              config.download.zip[result.filename] = new JSZip();
              /*  */
              let count = 0;
              for (let item of result.items) {
                const txt = config.download.filename.truncate(result.names[count], 30);
                /*  */
                try {
                  info.textContent = "Fetching " + txt + " header, please wait...";
                  const response = await fetch(item);
                  /*  */
                  if (response.ok) {
                    const name = ++count + '-' + result.type + '-' + Math.floor(Math.random() * 1e10) + result.extension;
                    /*  */
                    info.textContent = "Fetching " + txt + ", please wait...";
                    const blob = await response.blob();
                    /*  */
                    info.textContent = txt + " is fetched. Adding to the zip...";
                    await config.download.zip[result.filename].file(name, blob);
                  }
                } catch (e) {
                  info.textContent = "Fetching " + txt + " failed!";
                }
              }
              /*  */
              const suffix = '-' + result.type + ".zip";
              info.textContent = "Preparing " + result.type + " file as: " + result.filename + suffix;
              const blob = await config.download.zip[result.filename].generateAsync({"type": "blob"});
              /*  */
              result.filename += suffix;
              result.url = URL.createObjectURL(blob);
              background.send("download-all-items", result);
              /*  */
              info.textContent = "Downloading " + txt + ", please wait...";
              await new Promise((resolve) => {setTimeout(resolve, 3000)});
              info.textContent = "Click to download all " + type + " items";
            }
            /*  */
            button.removeAttribute("disabled");
          }
        });
      }
    },
    "filename": {
      "truncate": function (str, len) {
        if (str.length <= len) return str;
        const frontChars = Math.ceil((len - 3) / 2), backChars = Math.floor((len - 3) / 2);
        /*  */
        return str.slice(0, frontChars) + "..." + str.slice(str.length - backChars);
      },
      "make": function (i) {
        let url = config.download.stream.list[i].url;
        let title = config.download.stream.list[i].title || '';
        let ext = config.download.stream.list[i].extension || '';
        let page = config.download.stream.list[i].pagetitle || '';
        /*  */
        if (ext) ext = ext.replace('_', '.');
        /*  */
        let name_1 = '';
        let name_2 = '';
        let name_3 = '';
        let match = /\=\"(.+)\"/.exec(title || '');
        /*  */
        if (match && match.length) name_1 = match[1];
        match = url.match(/([^\/]+)(?=\.\w+$)(\.\w+)+/);
        if (match && match.length) name_2 = match[0];
        name_3 = "captured-media-" + i + (ext || ".mp3");
        /*  */
        if (page) {
          return page + (ext || ".mp3");
        } else {
          return name_1 ? name_1 : (name_2 ? name_2 : name_3);
        }
      }
    },
    "quality": function (parent, t, m, e, s) {
      try {
        const type = m === "video" ? "video" : "audio";
        const source = document.createElement("source");
        const media = document.createElement(type);
        /*  */
        //source.setAttribute("type", t); // dont need to add type
        media.setAttribute("type", type);
        media.setAttribute("preload", "metadata");
        /*  */
        media.onerror = function (e) {e.target.closest("td").textContent = "Error"};
        source.onerror = function (e) {e.target.closest("td").textContent = "Error"};
        media.onloadedmetadata = function (e) {
          const filename = e.target.closest("td").getAttribute("filename");
          const td = document.querySelector('td[filename="' + filename + '"]');
          /*  */
          if (td) {
            const date = new Date(null);
            const type = e.target.getAttribute("type");
            const info = {"kbps": '', "dimension": '', "duration": ''};
            /*  */
            date.setSeconds(e.target.duration || 0);
            info.duration = date.toISOString().slice(11, 19);
            /*  */
            if (e.target.videoHeight) {
              info.dimension = e.target.videoHeight + "p";
            } else {
              const kbit = parseInt(s || '0') / 128;
              info.kbps = Math.ceil(Math.round(kbit / (e.target.duration || 0)) / 16) * 16 + "Kbps";
            }
            /*  */
            const meta = {
              "kbps": info.kbps, 
              "duration": info.duration, 
              "dimension": info.dimension
            };
            /*  */
            td.closest("tr").querySelector('td[rule="duration"]').textContent = info.duration;
            config.download.meta[e.target.firstChild.src] = meta;
            background.send("metadata", config.download.meta);
            e.target.firstChild.src = "about:blank";
            td.setAttribute("title", "Quality");
            /*  */
            if (type === "video") {
              td.textContent = info.dimension ? info.dimension : "N/A";
            } else {
              td.textContent = info.kbps ? info.kbps : "N/A";
            }
          } else {
            e.target.closest("td").textContent = "Error";
          }
        };
        /*  */
        media.appendChild(source);
        parent.appendChild(media);
        source.src = e;
      } catch (e) {
        parent.textContent = "Error";
      }
    }
  },
  "interface": {
    "render": function (stream) {
      const toggle = document.getElementById("toggle-addon");
      /*  */
      config.download.stream = stream;
      config.download.meta = config.download.stream.metadata;
      config.download.index.video = config.download.index.audio = 0;
      document.documentElement.setAttribute("theme", stream.theme !== undefined ? stream.theme : "light");
      /*  */
      document.getElementById("audio-list-table").textContent = '';
      document.getElementById("video-list-table").textContent = '';
      toggle.setAttribute("title", config.download.stream.state.toUpperCase());
      toggle.setAttribute("state", config.download.stream.state);  
      /*  */
      if (config.download.stream.state === "inactive") {
        document.getElementById("video-count").textContent = "Video list - no video found!";
        document.getElementById("audio-count").textContent = "Audio list - no audio found!";
      } else if (config.download.stream.list && config.download.stream.list.length) {
        for (let i = config.download.stream.list.length - 1; i > -1; i--) {
          const _is = {"audio": false, "video": false};
          const _type = config.download.stream.list[i].type;
          const _extn = config.download.stream.list[i].extension;
          const _isvideo = config.download.extension.list.video.indexOf(_extn) !== -1;
          const _isaudio = config.download.extension.list.audio.indexOf(_extn) !== -1;
          /*  */
          let type = '';
          if (_type.indexOf("video") !== -1) {
            type = _type;
            _is.video = true;
          } else if (_type.indexOf("audio") !== -1) {
            type = _type;
            _is.audio = true;
          } else if (_isvideo) {
            _is.video = true;
            type = "video/" + _extn.replace('.', '');
          } else if (_isaudio) {
            _is.audio = true;
            type = "audio/" + _extn.replace('.', '');
          } else {
            type = "audio/mp3";
          }
          /*  */

          //console.error(type);
          //type = "audio/x-m4a";

          config.interface.add.row(i, type, _is.video ? "video": "audio");
        }
      }
    },
    "add": {
      "row": function (i, type, media) {
        const tr = document.createElement("tr");
        const audio = document.getElementById("audio-list-table");
        const video = document.getElementById("video-list-table");
        const table = media === "video" ? video : audio;
        const n = media === "video" ? config.download.index.video++ : config.download.index.audio++;
        /*  */
        let filename = config.download.filename.make(i);
        document.getElementById("video-count").textContent = "Video list" + " - " + config.download.index.video + " item(s) found";
        document.getElementById("audio-count").textContent = "Audio list" + " - " + config.download.index.audio + " item(s) found";
        /*  */
        config.interface.add.column(type, media, tr, (n + 1), "index", '');
        config.interface.add.column(type, media, tr, config.download.stream.list[i].url, "url", filename, filename);
        config.interface.add.column(type, media, tr, "00:00:00", "duration", "Duration", config.download.stream.list[i].url, config.download.stream.list[i].originalsize);
        config.interface.add.column(type, media, tr, "Quality", "resolution", '', config.download.stream.list[i].url, config.download.stream.list[i].originalsize);
        config.interface.add.column(type, media, tr, config.download.stream.list[i].size, "size", "File size");
        config.interface.add.column(type, media, tr, '⚼', "copy", "Copy the link to the clipboard");
        config.interface.add.column(type, media, tr, '⇩', "download", "Click to download media");
        config.interface.add.column(type, media, tr, '✕', "delete", "Click to delete the media");
        /*  */
        tr.setAttribute("index", i);
        table.appendChild(tr);
      },
      "column": function (type, media, tr, url, rule, title, filename, size) {
        const td = document.createElement("td");
        td.setAttribute("rule", rule);
        /*  */
        if (filename) {
          const a = document.createElement('a');
          /*  */
          a.setAttribute("href", url);
          a.setAttribute("download", filename);
          a.textContent = config.download.filename.truncate(filename, 45);
          a.addEventListener("click", function (e) {
            if (e.preventDefault) e.preventDefault();
            const index = parseInt(this.parentNode.parentNode.getAttribute("index"));
            const extension = config.download.stream.list[index].extension || ".mp3";
            /*  */
            filename = filename.replace(/[ `~!@#$%^&*()_|+\-=?;:'",.<>{}[\]\\/]/gi, '-');
            background.send("download-one-item", {
              "url": url,
              "filename": filename, 
              "extension": extension,
            });
          });
          /*  */
          td.setAttribute("title", title);
          td.appendChild(a);
        } else {
          td.textContent = url;
        }
        /*  */
        if (rule === "resolution") {
          td.setAttribute("type", type);
          td.setAttribute("size", size);
          td.setAttribute("media", media);
          td.setAttribute("filename", filename);
          td.setAttribute("title", config.download.meta[filename] ? "Quality" : "Click to see quality");        
          td.textContent = config.download.meta[filename] ? (config.download.meta[filename].dimension || config.download.meta[filename].kbps) : "Quality";
          /*  */
          td.addEventListener("click", function (e) {
            e.target.textContent = "Loading...";
            const type = e.target.getAttribute("type");
            const size = e.target.getAttribute("size");
            const media = e.target.getAttribute("media");
            const filename = e.target.getAttribute("filename");
            config.download.quality(e.target, type, media, filename, size);
          });
        }
        /*  */
        if (rule === "size") {
          td.setAttribute("title", title);
        }
        /*  */
        if (rule === "duration") {
          td.setAttribute("title", title);
          td.textContent = config.download.meta[filename] ? config.download.meta[filename].duration : "00:00:00";
        }
        /*  */
        if (rule === "copy") {
          td.textContent = '';
          td.setAttribute("title", title);
          td.addEventListener("click", function (e) {
            const a = e.target.parentNode.querySelector('a');
            const url = a.getAttribute("href");
            background.send("copy-to-clipboard", url);
            window.prompt("Copy to clipboard: Ctrl C + Enter", url);
          });
        }
        /*  */
        if (rule === "download") {
          td.textContent = '';
          td.setAttribute("title", title);
          td.addEventListener("click", function (e) {
            const a = e.target.parentNode.querySelector('a');
            if (a) a.click();
          });
        }
        /*  */
        if (rule === "delete") {
          td.setAttribute("title", title);
          td.addEventListener("click", function (e) {
            const t = e.target.parentNode.getAttribute("index");
            config.download.stream.list.splice(t, 1);
            background.send("store", config.download.stream);
          });
        }
        /*  */
        tr.appendChild(td);
      }
    }
  }
};

background.receive("load", config.interface.render);
background.connect(chrome.runtime.connect({"name": "popup"}));
background.receive("reload", function () {document.location.reload()});

window.addEventListener("load", config.load, false);
