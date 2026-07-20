/* USG chat overlay loader.
 *
 * One line in the site template:
 *   <script src="https://YOUR-HOST/embed.js"
 *           data-widget-url="https://YOUR-HOST/index.html" defer></script>
 *
 * Injects a transparent corner iframe running the widget in overlay mode
 * and resizes it when the widget reports open/closed via postMessage.
 */
(function () {
  var script = document.currentScript;
  var widgetUrl = script && script.getAttribute("data-widget-url");
  if (!widgetUrl) {
    console.error("usg-chat embed.js: missing data-widget-url");
    return;
  }
  var widgetOrigin = new URL(widgetUrl, location.href).origin;

  var CLOSED = { width: "96px", height: "96px" };
  var OPEN = { width: "min(470px, 100vw)", height: "min(760px, 100vh)" };

  var iframe = document.createElement("iframe");
  iframe.src = widgetUrl + (widgetUrl.indexOf("?") === -1 ? "?" : "&") + "embed=overlay";
  iframe.title = "USG Chat";
  iframe.setAttribute("allowtransparency", "true");
  iframe.style.cssText =
    "position:fixed;right:0;bottom:0;border:none;z-index:2147483647;" +
    "background:transparent;color-scheme:normal;";
  iframe.style.width = CLOSED.width;
  iframe.style.height = CLOSED.height;
  document.body.appendChild(iframe);

  window.addEventListener("message", function (e) {
    if (e.origin !== widgetOrigin) return;
    if (!e.data || e.data.type !== "usg-chat") return;
    var size = e.data.open ? OPEN : CLOSED;
    iframe.style.width = size.width;
    iframe.style.height = size.height;
  });
})();
