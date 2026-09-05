# Skills

Written instructions the model can ask for by name: how something is done in
this project, kept out of the context until it is wanted.

A skill is not a document to search — that is what `search_documents` is for.
It is a procedure: *when you are asked to do X, do it like this*.

## Adding one

Either shape works. A folder is the better home once a skill grows:

```
skills/
  chemistry-notation/
    SKILL.md
  quick-note.md
```

```markdown
---
name: chemistry-notation
description: Writing formulae so this interface renders them properly
---

Instructions go here, in markdown.
```

Frontmatter is optional. Without it the filename becomes the name and the
first non-empty line becomes the description, so a plain `.md` file dropped in
here works — which is what most people will try first.

The name must be lowercase letters, digits and hyphens: the model has to
repeat it back, and an enum on the tool stops it inventing one.

## Skills that need tools

A third key, `tools:`, names the tool groups the instructions assume. Loading
the skill opens them.

```markdown
---
name: plotting
description: How charts are drawn and where they are saved
tools: python, filesystem
---
```

`tools: [python, filesystem]` means the same thing, and so does a space
instead of the comma. The group names are the ones in the `load_tools` index —
`filesystem`, `python`, `terminal`, `git`, `http`, `memory`, `documents`,
`ocr`, and `mcp:<server>` for an MCP server.

Why it exists: instructions that say *"plot it with matplotlib"* are no use to
a model that cannot see the python tool, and making it spend a second round
trip on `load_tools` to discover that is a round trip the skill already knew
about. Opening a group costs a prefix-cache miss whichever way it happens;
this only moves it earlier.

A name that is misspelled, or belongs to a tool that is switched off, simply
opens nothing. The model is told what **actually** opened rather than what was
asked for — being told about a tool whose schema is not coming is worse than
not being told at all.

## How it reaches the model

As a **tool result**, never as part of the prompt. That is a hardware
decision, not a stylistic one: llama.cpp keys its prefix cache on the prompt,
so injecting a skill would re-read the whole conversation — about 200 seconds
on this machine. A tool call appends to the end, which the cache does not
mind.

So the model sees an index of names and descriptions in the `load_skill`
tool, and asks for a body only when one matches. Bodies are capped at 6,000
characters; two skills should not fill a 4,096-token window between them.

The tool is not registered at all when this folder is empty — an index of
nothing is a schema paid for on every request.
