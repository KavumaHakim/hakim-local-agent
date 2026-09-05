---
name: chemistry-notation
description: Writing formulae and equations so this interface renders them properly
---

# Chemical notation in this interface

The transcript renders a small, specific subset of LaTeX. Writing outside it
produces raw backslashes on the screen, so this is worth getting right.

## Equations

Put a reaction on its own line, in a display block:

```
\[
2\text{Mg}(s) + \text{O}_2(g) \rightarrow 2\text{MgO}(s)
\]
```

`$$ … $$` works identically. Both are followed and centred.

## What renders

| You write | The reader sees |
|---|---|
| `\text{Mg}` | Mg, upright |
| `_2` or `_{10}` | a real subscript |
| `^{2+}` | a real superscript |
| `\rightarrow` | → |
| `\rightleftharpoons` | ⇌ |
| `\frac{1}{2}` | a stacked fraction with a rule |
| `\sqrt{x}` | √ with an overline |
| `\approx` `\cdot` `\times` `\pm` `\leq` | ≈ · × ± ≤ |

Greek letters, `\int`, `\sum` and `\infty` all work. An unknown command is
shown as its own text rather than swallowed, so a mistake is visible rather
than silently dropping content.

## Rules worth following

- **Wrap element symbols in `\text{}`.** Without it, `Mg` is read as two
  variables and set in italics, which is wrong for a formula.
- **State symbols go in `\text{}` too**: `(s)`, `(l)`, `(g)`, `(aq)`.
- **Inline maths uses `\( … \)` or `$ … $`** — for a formula inside a
  sentence, not for a full equation.
- **Do not use `\begin{align}` or matrices.** They are not supported and will
  render as their own source.
- **Balance the equation before writing it.** A rendered equation looks
  authoritative, which makes an unbalanced one worse than an ugly one.

## Plain text is fine too

The models here already emit proper Unicode subscripts, and `H₂O` renders
perfectly as ordinary text. Reach for LaTeX when there is a reaction arrow, a
fraction or a charge — not for every mention of a compound.
