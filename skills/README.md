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
