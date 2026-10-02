// Theme toggle: an explicit choice is stored; otherwise the OS preference applies.
(function () {
  var root = document.documentElement;
  var btn = document.querySelector(".theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", function () {
    var current = root.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });
})();

// Tag filter on /posts/. Reads and writes ?tag= so tag links on posts land filtered.
(function () {
  var bar = document.querySelector(".tag-filter");
  if (!bar) return;
  var buttons = document.querySelectorAll(".tag-filter button[data-tag]");
  var items = document.querySelectorAll(".archive-item");
  var years = document.querySelectorAll(".archive-year");
  var empty = document.querySelector(".archive-empty");

  function apply(tag) {
    var shown = 0;
    items.forEach(function (li) {
      var match = !tag || li.getAttribute("data-tags").split("|").indexOf(tag) !== -1;
      li.hidden = !match;
      if (match) shown++;
    });
    years.forEach(function (sec) {
      sec.hidden = !sec.querySelector(".archive-item:not([hidden])");
    });
    buttons.forEach(function (b) {
      var on = b.getAttribute("data-tag") === tag;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    if (empty) empty.hidden = shown !== 0;
  }

  document.addEventListener("click", function (e) {
    var b = e.target.closest(".tag-filter button[data-tag]");
    if (!b) return;
    var tag = b.getAttribute("data-tag");
    apply(tag);
    var url = new URL(location.href);
    if (tag) url.searchParams.set("tag", tag); else url.searchParams.delete("tag");
    history.replaceState(null, "", url);
  });

  var initial = new URLSearchParams(location.search).get("tag") || "";
  apply(initial);
  var more = document.querySelector(".tag-more");
  if (initial && more && more.querySelector('[data-tag="' + CSS.escape(initial) + '"]')) more.open = true;
})();
