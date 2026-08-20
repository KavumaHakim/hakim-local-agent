"""Presentation helpers for the Streamlit app.

Kept separate from app.py so the page logic stays readable: this module holds
the stylesheet and the small HTML fragments, app.py holds the flow.
"""

from __future__ import annotations

import html

# Streamlit ships its own layout, so this restyles rather than rebuilds:
# user turns become right-aligned bubbles, assistant turns sit flat on the
# page like a normal chat client, and the chrome we do not need is hidden.
CSS = """
<style>
:root {
  --surface: #161922;
  --surface-2: #1d2130;
  --border: #262b3a;
  --accent: #7c5cff;
  --accent-soft: rgba(124, 92, 255, 0.14);
  --muted: #8b90a3;
  --ok: #3ecf8e;
  --bad: #ff6b6b;
}

/* --- chrome we do not need --- */
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {
  display: none !important;
}
[data-testid="stAppViewContainer"] > .main {
  padding-top: 0;
}
[data-testid="stMainBlockContainer"] {
  padding-top: 3rem;
  padding-bottom: 7rem;
  max-width: 48rem;
}

/* --- header --- */
.chat-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding-bottom: 0.35rem;
}
.chat-header .mark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), #4dd0e1);
  display: grid;
  place-items: center;
  font-size: 1rem;
}
.chat-header h1 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 0;
  letter-spacing: -0.01em;
}
.chat-header .sub {
  color: var(--muted);
  font-size: 0.78rem;
  margin-top: 1px;
}

/* --- messages --- */
[data-testid="stChatMessage"] {
  background: transparent;
  padding: 0.35rem 0;
  gap: 0.75rem;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {
  line-height: 1.65;
}

/* Roles are told apart by Streamlit's own aria-label rather than by its
   emotion-cache class names, which change between builds. */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  flex-direction: row-reverse;
  margin-left: auto;
}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
  background: var(--accent-soft);
  border: 1px solid rgba(124, 92, 255, 0.28);
  border-radius: 14px 14px 4px 14px;
  padding: 0.6rem 0.9rem;
  max-width: 82%;
  /* Streamlit sets margin-right:auto as well; leaving it would let the two
     auto margins split the free space and centre the bubble. */
  margin-left: auto;
  margin-right: 0;
  flex-grow: 0;
}

/* assistant turn: flat, full width */
[data-testid="stChatMessageContent"]:not([aria-label="Chat message from user"]) {
  padding: 0.1rem 0;
}

/* --- tool activity --- */
.tool-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.15rem 0 0.55rem;
}
.tool-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.24rem 0.7rem;
  font-size: 0.75rem;
  color: var(--muted);
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}
.tool-pill .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ok);
  flex: none;
}
.tool-pill.bad .dot { background: var(--bad); }
.tool-pill .name { color: #c9cede; }
.tool-pill .val {
  color: var(--muted);
  max-width: 34ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* --- typing indicator --- */
.typing { display: inline-flex; gap: 5px; padding: 0.5rem 0; }
.typing span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted);
  animation: blink 1.3s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.18s; }
.typing span:nth-child(3) { animation-delay: 0.36s; }
@keyframes blink {
  0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
  40% { opacity: 1; transform: translateY(-2px); }
}

.turn-meta {
  color: var(--muted);
  font-size: 0.72rem;
  margin-top: 0.35rem;
}

/* --- empty state --- */
.empty-hero {
  text-align: center;
  padding: 3.2rem 0 1.4rem;
}
.empty-hero h2 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0 0 0.4rem;
  letter-spacing: -0.02em;
}
.empty-hero p {
  color: var(--muted);
  font-size: 0.88rem;
  margin: 0;
}

/* suggestion buttons */
[data-testid="stButton"] > button {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  color: #c9cede;
  font-size: 0.82rem;
  font-weight: 400;
  text-align: left;
  padding: 0.7rem 0.9rem;
  transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stButton"] > button:hover {
  border-color: var(--accent);
  background: var(--surface-2);
  color: #fff;
}

/* --- chat input --- */
[data-testid="stChatInput"] {
  border-radius: 14px;
  border: 1px solid var(--border);
  background: var(--surface);
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent);
}
[data-testid="stBottomBlockContainer"] {
  max-width: 48rem;
  padding-bottom: 1.4rem;
}

/* --- sidebar --- */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--border);
  background: #10131a;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
  padding-top: 1.1rem;
}
/* Streamlit's own collapse arrow: make it a visible control, not a hint. */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
  color: #c9cede !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
  border-color: var(--accent) !important;
  background: var(--surface-2) !important;
}

.side-label {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 0.95rem 0 0.35rem;
  padding-top: 0.7rem;
  border-top: 1px solid var(--border);
}
/* The first label sits right under the header, so it needs no rule above. */
.side-label.first { border-top: none; padding-top: 0; margin-top: 0.3rem; }

/* sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--surface);
  border-color: var(--border);
  border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
  font-size: 0.78rem;
  padding: 0.5rem 0.75rem;
  text-align: center;
}

/* collapsible sections: quiet until opened */
[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  margin-bottom: 0.35rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size: 0.78rem;
  color: #c9cede;
  padding: 0.5rem 0.7rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: #fff;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .side-label:first-child {
  margin-top: 0.2rem;
  border-top: none;
  padding-top: 0;
}

/* history entries */
.hist-empty {
  font-size: 0.75rem;
  color: var(--muted);
  padding: 0.3rem 0 0.5rem;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button.hist {
  text-align: left;
}
.status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
  padding: 0.5rem 0.7rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface);
}
.status .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--ok); flex: none;
  box-shadow: 0 0 0 3px rgba(62, 207, 142, 0.15);
}
.status.down .dot {
  background: var(--bad);
  box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.15);
}
.status .txt { color: #c9cede; }
.tool-item {
  font-size: 0.78rem;
  padding: 0.32rem 0.55rem;
  border-radius: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  margin-bottom: 0.3rem;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  color: #c9cede;
}
.tool-item .cat { color: var(--muted); }
.tool-off {
  font-size: 0.72rem;
  color: var(--muted);
  line-height: 1.45;
  padding: 0.35rem 0 0.5rem;
  border-left: 2px solid var(--border);
  padding-left: 0.6rem;
  margin-bottom: 0.4rem;
}
.tool-off b { color: #b9bed0; }
.path {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 0.7rem;
  color: var(--muted);
  word-break: break-all;
  line-height: 1.5;
}
</style>
"""

TYPING = '<div class="typing"><span></span><span></span><span></span></div>'


def tool_pills(calls: list[dict]) -> str:
    """Render the tool calls made during one turn as a row of pills."""
    if not calls:
        return ""
    pills = []
    for call in calls:
        state = "" if call["ok"] else " bad"
        name = html.escape(call["name"])
        summary = html.escape(call["summary"])
        pills.append(
            f'<span class="tool-pill{state}"><span class="dot"></span>'
            f'<span class="name">{name}</span>'
            f'<span class="val">{summary}</span></span>'
        )
    return f'<div class="tool-row">{"".join(pills)}</div>'


def header(subtitle: str) -> str:
    return (
        '<div class="chat-header"><div class="mark">🤖</div><div>'
        "<h1>Hakim AI System</h1>"
        f'<div class="sub">{html.escape(subtitle)}</div>'
        "</div></div>"
    )


def status(healthy: bool, url: str) -> str:
    cls = "status" if healthy else "status down"
    text = "Model online" if healthy else "Model offline"
    return (
        f'<div class="{cls}"><span class="dot"></span>'
        f'<span class="txt">{text}</span></div>'
        f'<div class="path">{html.escape(url)}</div>'
    )
