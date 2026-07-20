# Website Integration

The widget is one file (`docs/index.html`) with three modes, selected by the
`?embed=` query param:

| Mode | URL | Use |
|------|-----|-----|
| bubble | `index.html` | Top-level page: floating launcher + dimmed backdrop |
| panel | `index.html?embed=panel` | Inside an iframe on a dedicated "Ask USG" page — **use this now** |
| overlay | `index.html?embed=overlay` | Site-wide corner floater, driven by `embed.js` — **later, after ITS approval** |

Widget host: `https://arsalanghogari.com/usc-usg-ai-chatbot` (GitHub Pages, custom domain — also in the backend CORS allowlist).

## Now — dedicated "Ask USG" page (single iframe)

Paste into a WordPress page (Divi code module or Custom HTML block):

```html
<iframe
  src="https://arsalanghogari.com/usc-usg-ai-chatbot/index.html?embed=panel"
  title="Ask USG"
  style="width:100%; height:80vh; min-height:520px; border:1px solid #e5e7eb; border-radius:16px;"
></iframe>
```

WordPress only holds this pointer; updates to the widget deploy via GitHub
Pages with no WordPress changes.

## Later — site-wide overlay (one line in the global template)

**Do not ship until CORS is restricted to the USG domain (Phase 5).**

```html
<script src="https://arsalanghogari.com/usc-usg-ai-chatbot/embed.js"
        data-widget-url="https://arsalanghogari.com/usc-usg-ai-chatbot/index.html" defer></script>
```

`embed.js` injects a transparent corner iframe in overlay mode and resizes it
when the widget reports open/closed (origin-validated postMessage).

## Config checklist

- `OPENAI_API_KEY` — backend `.env` + Render env.
- `CHAT_MODEL` — set explicitly on Render to a model the account has.
- `CONTACT_FORM_URL` / `CONTACT_EMAIL` — real USG contact form + inbox (used
  by the staleness notice; defaults are in `server.js`).
- `API` constant in `docs/index.html` — deployed backend URL.
- Widget host in the snippets above changes? Update `CORS_ORIGINS` (or the default list in `server.js`) to match.
