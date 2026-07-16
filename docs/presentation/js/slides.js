/* ============================================================================
   Motor de render COMPARTIDO de la presentacion AgroSatCopilot.
   Produce el HTML de cada lamina desde content/{es,en}.json. Lo usan tanto la
   presentacion en vivo (js/render.js + Reveal) como la pagina de impresion
   (print.html), para que el PDF sea IDENTICO a la pantalla.
   Expone window.AgroDeck = { renderSlide, figUrl }.
   ============================================================================ */
(function () {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function rich(s) {
    return esc(s)
      .replace(/&lt;(\/?)(strong|em|b|i)&gt;/g, "<$1$2>")
      .replace(/&lt;br\s*\/?&gt;/g, "<br>");
  }
  function figUrl(img) {
    if (!img) return "";
    return img.indexOf("/") >= 0 ? img : `assets/figs/${img}`;
  }

  function kicker(s) {
    return s.kicker ? `<div class="kicker">${esc(s.kicker)}</div>` : "";
  }
  function title(s, tag) {
    tag = tag || "h2";
    return s.title ? `<${tag}>${esc(s.title)}</${tag}>` : "";
  }
  function paras(arr) {
    return (arr || []).map((p) => `<p>${rich(p)}</p>`).join("");
  }
  function bullets(arr) {
    if (!arr || !arr.length) return "";
    return `<ul class="ulist">${arr.map((b) => `<li>${rich(b)}</li>`).join("")}</ul>`;
  }
  function cards(items, cols) {
    const inner = (items || [])
      .map((it) => {
        const v = it.variant && it.variant !== "normal" ? ` ${it.variant.trim()}` : "";
        return `<div class="card${v}">` +
          (it.title ? `<h3>${rich(it.title)}</h3>` : "") +
          (it.body ? `<p>${rich(it.body)}</p>` : "") + `</div>`;
      })
      .join("");
    return `<div class="cards ${cols || "c3"}">${inner}</div>`;
  }
  function kpiRow(items) {
    const inner = (items || [])
      .map((k) =>
        `<div class="kpi"><div class="num${k.accent ? " accent" : ""}">${esc(k.num)}</div>` +
        `<div class="lbl">${esc(k.label)}</div></div>`
      )
      .join("");
    return `<div class="kpi-row">${inner}</div>`;
  }
  function table(t) {
    const head = `<tr>${(t.headers || []).map((h) => `<th>${esc(h)}</th>`).join("")}</tr>`;
    const body = (t.rows || [])
      .map((r) =>
        `<tr${r.highlight ? ' class="highlight"' : ""}>` +
        (r.cells || []).map((c) => `<td>${esc(c)}</td>`).join("") + `</tr>`
      )
      .join("");
    return `<table>${head}${body}</table>`;
  }
  function figBlock(s) {
    return `<div class="fig"><img src="${esc(figUrl(s.img))}" alt="">` +
      (s.caption ? `<div class="cap">${esc(s.caption)}</div>` : "") + `</div>`;
  }

  function renderSlide(s) {
    const sec = s.sec ? ` data-sec="${esc(s.sec)}"` : "";
    let cls = "";
    let body = "";

    switch (s.layout) {
      case "cover": {
        cls = ' class="cover-slide"';
        const brand = s.brand ? `<h1 class="cover-brand">${esc(s.brand)}</h1>` : title(s, "h1");
        const sub = s.subtitle ? `<p class="cover-sub">${esc(s.subtitle)}</p>` : paras(s.paras);
        const meta = (s.meta || []).length
          ? `<div class="cover-meta">${s.meta.map((m) => `<div>${esc(m)}</div>`).join("")}</div>`
          : "";
        const img = s.img ? figUrl(s.img) : "assets/figs/cover.png";
        const logo =
          `<img class="cover-logo" src="assets/tecnologico-de-monterrey-blue.png" alt="Tecnologico de Monterrey"` +
          ` onerror="this.style.display='none'">`;
        body =
          `<div class="cover"><div class="cover-text">${logo}${brand}${sub}${meta}</div>` +
          `<div class="cover-img" style="background-image:url('${img}')"></div></div>`;
        break;
      }
      case "divider":
        cls = ' class="divider"';
        body =
          `<div class="divider-grid"><div class="divider-text">` +
          (s.sec ? `<div class="sec-num">${esc(s.sec)}</div>` : "") +
          kicker(s) + title(s) + paras(s.paras) +
          `</div><div class="divider-art"><img src="${esc(figUrl(s.img))}" alt=""></div></div>`;
        break;
      case "closing":
        cls = ' class="closing center"';
        body = kicker(s) + title(s, "h1") + paras(s.paras);
        break;
      case "kpi":
        cls = ' class="center"';
        body = kicker(s) + title(s) + kpiRow(s.items) +
          (s.sub2 && s.cards && s.cards.length ? `<div class="cards-sub">${esc(s.sub2)}</div>` : "") +
          (s.cards && s.cards.length ? cards(s.cards, s.cards2 || "c2") : "");
        break;
      case "cards":
        body = kicker(s) + title(s) + cards(s.items, s.cols);
        if (s.items2 && s.items2.length) {
          body += (s.sub2 ? `<div class="cards-sub">${esc(s.sub2)}</div>` : "") +
            cards(s.items2, s.cols2 || s.cols);
        }
        break;
      case "table":
        body = kicker(s) + title(s) + table(s.table) + paras(s.paras);
        break;
      case "fig":
        body = kicker(s) + title(s) + figBlock(s) + bullets(s.bullets);
        break;
      case "twocol": {
        if (!s.img) {
          body = kicker(s) + title(s) + paras(s.paras) + bullets(s.bullets);
          break;
        }
        const wide = s.side === "right" ? " wide-right" : (s.side === "left" ? " wide-left" : " wide-right");
        const textCol = `<div class="col-text">${title(s)}${paras(s.paras)}${bullets(s.bullets)}</div>`;
        const artCol = `<div><img src="${esc(figUrl(s.img))}" alt="">` +
          (s.caption ? `<div class="cap">${esc(s.caption)}</div>` : "") + `</div>`;
        body = kicker(s) +
          `<div class="two-col${wide}">` +
          (s.side === "left" ? textCol + artCol : artCol + textCol) + `</div>`;
        break;
      }
      default:
        body = kicker(s) + title(s) + paras(s.paras) + bullets(s.bullets);
    }
    return `<section${cls}${sec}>${body}</section>`;
  }

  window.AgroDeck = { renderSlide, figUrl };
})();
