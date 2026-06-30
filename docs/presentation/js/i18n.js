/* Switch de idioma ES/EN para la presentacion.
   Cada bloque de texto se duplica con [data-lang-es] / [data-lang-en]; el CSS
   oculta el que no corresponde segun html[data-lang]. Persistimos la eleccion en
   localStorage y respetamos ?lang= en la URL (util para GitHub Pages / deep links). */
(function () {
  function getInitialLang() {
    var params = new URLSearchParams(window.location.search);
    var q = params.get("lang");
    if (q === "es" || q === "en") return q;
    var saved = window.localStorage.getItem("agrosat_lang");
    if (saved === "es" || saved === "en") return saved;
    // Por defecto: espanol (audiencia del curso), salvo navegador claramente EN.
    return (navigator.language || "es").toLowerCase().startsWith("en") ? "en" : "es";
  }

  function setLang(lang) {
    document.documentElement.setAttribute("data-lang", lang);
    window.localStorage.setItem("agrosat_lang", lang);
    document.querySelectorAll("#lang-toggle button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.lang === lang);
    });
    // Reveal recalcula el layout tras cambiar el contenido visible.
    if (window.Reveal && window.Reveal.layout) window.Reveal.layout();
  }

  window.addEventListener("DOMContentLoaded", function () {
    setLang(getInitialLang());
    var toggle = document.getElementById("lang-toggle");
    if (toggle) {
      toggle.addEventListener("click", function (e) {
        if (e.target.dataset && e.target.dataset.lang) setLang(e.target.dataset.lang);
      });
    }
  });
})();
