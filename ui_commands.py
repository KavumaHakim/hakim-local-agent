"""Slash commands for the chat box, and the dropdown that offers them.

Streamlit's chat input is a plain textarea with no autocomplete, so the
dropdown is built by a small script that attaches to it from a zero-height
component iframe. If that iframe cannot reach the parent document the script
does nothing at all and the commands still work when typed in full - the
palette is a convenience, never the only way in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    args: str
    help: str

    @property
    def takes_args(self) -> bool:
        return bool(self.args)


COMMANDS: tuple[Command, ...] = (
    Command("/help", "", "Show the available commands"),
    Command("/models", "", "List models and which one is loaded"),
    Command("/model", "<key>", "Switch to a model"),
    Command("/unload", "", "Unload the current model and free its RAM"),
    Command("/tools", "", "List the tools the agent can use"),
    Command("/auto", "", "Toggle automatic model routing"),
    Command("/clear", "", "Start a new conversation"),
)


def help_text() -> str:
    lines = ["**Commands**", ""]
    for command in COMMANDS:
        label = f"`{command.name} {command.args}`".replace(" `", "`") if command.args else f"`{command.name}`"
        lines.append(f"- {label} — {command.help}")
    return "\n".join(lines)


def palette_script() -> str:
    """HTML for a zero-height component that adds the dropdown."""
    payload = json.dumps(
        [
            {"name": c.name, "args": c.args, "help": c.help, "takesArgs": c.takes_args}
            for c in COMMANDS
        ]
    )
    return _TEMPLATE.replace("__COMMANDS__", payload) + _SIDEBAR_TOGGLE


_TEMPLATE = r"""
<script>
(function () {
  const COMMANDS = __COMMANDS__;

  // The component runs in an iframe; everything it touches lives in the page
  // that embeds it. If that is blocked, do nothing rather than half-work.
  let doc;
  try {
    doc = window.parent.document;
    if (!doc) return;
  } catch (err) {
    return;
  }

  const PANEL_ID = "agent-cmd-palette";
  let selected = 0;
  let matches = [];

  function panel() {
    let el = doc.getElementById(PANEL_ID);
    if (!el) {
      el = doc.createElement("div");
      el.id = PANEL_ID;
      el.style.cssText = [
        "position:fixed", "z-index:1000000", "display:none",
        "background:#161922", "border:1px solid #262b3a", "border-radius:12px",
        "padding:6px", "box-shadow:0 12px 34px rgba(0,0,0,.45)",
        "font-family:ui-monospace,'Cascadia Code',Consolas,monospace",
        "font-size:13px", "max-height:260px", "overflow-y:auto"
      ].join(";");
      doc.body.appendChild(el);
    }
    return el;
  }

  function box() {
    return doc.querySelector('[data-testid="stChatInputTextArea"]');
  }

  function hide() {
    panel().style.display = "none";
    matches = [];
  }

  function render(area) {
    const el = panel();
    if (!matches.length) { hide(); return; }

    el.innerHTML = "";
    matches.forEach(function (cmd, i) {
      const row = doc.createElement("div");
      row.style.cssText = [
        "display:flex", "gap:10px", "align-items:baseline",
        "padding:7px 10px", "border-radius:8px", "cursor:pointer",
        "background:" + (i === selected ? "#1d2130" : "transparent")
      ].join(";");
      row.innerHTML =
        '<span style="color:#c9cede">' + cmd.name +
        (cmd.args ? ' <span style="color:#8b90a3">' + cmd.args + "</span>" : "") +
        '</span><span style="color:#8b90a3;font-size:12px">' + cmd.help + "</span>";
      row.addEventListener("mousedown", function (ev) {
        ev.preventDefault();      // keep focus in the textarea
        accept(area, cmd);
      });
      el.appendChild(row);
    });

    const rect = area.getBoundingClientRect();
    el.style.display = "block";
    el.style.left = rect.left + "px";
    el.style.width = Math.max(rect.width, 260) + "px";
    // Measure after painting so the panel sits directly above the box.
    el.style.top = (rect.top - el.offsetHeight - 8) + "px";
  }

  function setValue(area, text) {
    // React owns this textarea, so set the value through the native setter
    // and fire the event React listens for.
    const setter = Object.getOwnPropertyDescriptor(
      window.parent.HTMLTextAreaElement.prototype, "value"
    ).set;
    setter.call(area, text);
    area.dispatchEvent(new window.parent.Event("input", { bubbles: true }));
  }

  function accept(area, cmd) {
    setValue(area, cmd.name + (cmd.takesArgs ? " " : ""));
    hide();
    area.focus();
  }

  function update(area) {
    const value = area.value || "";
    if (!value.startsWith("/") || value.includes("\n")) { hide(); return; }

    const typed = value.split(" ")[0].toLowerCase();
    // Once a complete command with arguments is typed, get out of the way.
    if (value.includes(" ") && COMMANDS.some(c => c.name === typed)) { hide(); return; }

    matches = COMMANDS.filter(c => c.name.startsWith(typed));
    if (selected >= matches.length) selected = 0;
    render(area);
  }

  function attach() {
    const area = box();
    if (!area || area.dataset.cmdPalette === "1") return;
    area.dataset.cmdPalette = "1";

    area.addEventListener("input", function () { selected = 0; update(area); });
    area.addEventListener("blur", function () { setTimeout(hide, 120); });

    area.addEventListener("keydown", function (ev) {
      if (panel().style.display !== "block" || !matches.length) return;

      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        selected = (selected + (ev.key === "ArrowDown" ? 1 : -1) + matches.length)
                   % matches.length;
        render(area);
      } else if (ev.key === "Enter" || ev.key === "Tab") {
        // Stop Streamlit from submitting a half-typed command.
        ev.preventDefault();
        ev.stopPropagation();
        accept(area, matches[selected]);
      } else if (ev.key === "Escape") {
        hide();
      }
    }, true);   // capture, so this runs before Streamlit's own handler
  }

  // Streamlit replaces DOM nodes on rerun, so keep re-attaching.
  attach();
  setInterval(attach, 700);
})();
</script>
"""


# A always-visible pill that drives Streamlit's own collapse control, so the
# sidebar can be hidden and brought back from one obvious place. It clicks the
# native button rather than setting CSS, which keeps Streamlit's own state in
# step with what is on screen.
_SIDEBAR_TOGGLE = r"""
<script>
(function () {
  let doc;
  try { doc = window.parent.document; if (!doc) return; } catch (e) { return; }

  const ID = "agent-sidebar-toggle";

  // Both native controls stay in the DOM, so which one exists says nothing
  // about the current state. The rendered width does.
  function isExpanded() {
    const bar = doc.querySelector('[data-testid="stSidebar"]');
    return !!bar && bar.getBoundingClientRect().width > 120;
  }

  function clickNative() {
    const expanded = isExpanded();
    const el =
      (expanded
        ? doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
        : doc.querySelector('[data-testid="stSidebarCollapsedControl"] button')) ||
      doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
      doc.querySelector('[data-testid="stSidebarCollapsedControl"] button');
    if (el) el.click();
  }

  function build() {
    if (doc.getElementById(ID)) return;
    const btn = doc.createElement("button");
    btn.id = ID;
    btn.type = "button";
    // Pinned top-right rather than beside the sidebar: measuring the sidebar
    // mid-transition returns a 1px width and throws the button on top of it.
    // The toolbar that normally lives here is hidden by our own CSS.
    // z-index must clear Streamlit's own header, which sits at 999990 with an
    // opaque background and would otherwise paint straight over this.
    btn.style.cssText = [
      "position:fixed", "top:12px", "right:12px", "z-index:1000001",
      "display:flex", "align-items:center", "gap:7px",
      "background:#161922", "color:#c9cede",
      "border:1px solid #262b3a", "border-radius:10px",
      "padding:7px 11px", "cursor:pointer", "font-size:12px",
      "font-family:inherit", "line-height:1",
      "box-shadow:0 6px 18px rgba(0,0,0,.35)"
    ].join(";");
    btn.addEventListener("mouseenter", function () {
      btn.style.borderColor = "#7c5cff";
    });
    btn.addEventListener("mouseleave", function () {
      btn.style.borderColor = "#262b3a";
    });
    btn.addEventListener("click", function () {
      clickNative();
      // Repaint after the slide finishes, not during it.
      setTimeout(paint, 450);
    });
    doc.body.appendChild(btn);
  }

  function paint() {
    const btn = doc.getElementById(ID);
    if (!btn) return;
    const expanded = isExpanded();
    btn.innerHTML =
      '<span style="font-size:13px">' + (expanded ? "&#10094;" : "&#9776;") +
      "</span><span>" + (expanded ? "Hide panel" : "Show panel") + "</span>";
    btn.style.display = "flex";
  }

  build();
  paint();
  setInterval(function () { build(); paint(); }, 700);
})();
</script>
"""
