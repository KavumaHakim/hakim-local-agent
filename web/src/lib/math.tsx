/**
 * Maths, rendered in two layers: a small renderer that is always here, and
 * KaTeX fetched only when a formula actually appears.
 *
 * **Why not just KaTeX.** Measured across the stored replies: 365 messages,
 * of which **13 contain maths at all** — 68 math spans using ten distinct
 * commands, every one of which the small renderer below already handles. The
 * long tail KaTeX exists to cover (matrices, aligned environments, big
 * operators with limits) appears **zero** times. KaTeX is 273 kB of JS, 25 kB
 * of CSS and 296 kB of woff2 against a 322 kB bundle, so loading it eagerly
 * would roughly double what every conversation downloads to typeset something
 * 96% of messages do not contain.
 *
 * So it is imported dynamically, on first formula, and cached for the rest of
 * the session. A conversation with no maths pays nothing.
 *
 * **Why the small renderer stays.** It is the instant paint — maths is never
 * blank while the chunk is in flight — and, more importantly, it is the
 * fallback when the chunk cannot be fetched at all. This application is meant
 * to run with no network; a formula must still render on a machine that is
 * offline, and KaTeX failing to load is not an error worth showing a person.
 * It is also the fallback for a formula KaTeX refuses to parse.
 *
 * **The trade this makes.** The small renderer emits React elements, never
 * HTML strings, so there is nothing to sanitise. KaTeX emits a string, which
 * means `dangerouslySetInnerHTML`. That is safe *because* `trust` is left at
 * its default of false — with it off KaTeX will not emit `\href`, `\htmlClass`
 * or any other raw-markup escape, and text is escaped. Turning `trust` on
 * would hand a model the ability to write markup into the page, so it stays
 * off, and this comment is here so that stays a decision rather than an
 * accident.
 *
 * **Only ever called on text between math delimiters.** That is not a detail.
 * Two of the "commands" found in the corpus were `\Windows` and `\nLine` — a
 * file path and an escape that had nothing to do with maths — so anything
 * that treated a stray backslash as LaTeX would mangle ordinary prose. The
 * delimiters are the whole licence to transform.
 */

import { useEffect, useState, type ReactNode } from 'react'

/** Single-token commands: a name in, a character out. */
const SYMBOLS: Record<string, string> = {
  rightarrow: '→',
  to: '→',
  longrightarrow: '⟶',
  leftarrow: '←',
  Rightarrow: '⇒',
  leftrightarrow: '↔',
  rightleftharpoons: '⇌', // reversible reaction
  times: '×',
  cdot: '·',
  div: '÷',
  pm: '±',
  mp: '∓',
  approx: '≈',
  neq: '≠',
  ne: '≠',
  leq: '≤',
  le: '≤',
  geq: '≥',
  ge: '≥',
  ll: '≪',
  gg: '≫',
  equiv: '≡',
  propto: '∝',
  infty: '∞',
  partial: '∂',
  nabla: '∇',
  int: '∫',
  iint: '∬',
  oint: '∮',
  sum: '∑',
  prod: '∏',
  degree: '°',
  circ: '∘',
  ldots: '…',
  dots: '…',
  cdots: '⋯',
  in: '∈',
  notin: '∉',
  subset: '⊂',
  cup: '∪',
  cap: '∩',
  forall: '∀',
  exists: '∃',
  // Greek, lower then upper. Only the ones that turn up in school-level
  // chemistry and physics; the rest can be added when something needs them.
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  epsilon: 'ε',
  theta: 'θ',
  lambda: 'λ',
  mu: 'μ',
  pi: 'π',
  rho: 'ρ',
  sigma: 'σ',
  tau: 'τ',
  phi: 'φ',
  omega: 'ω',
  Delta: 'Δ',
  Sigma: 'Σ',
  Omega: 'Ω',
  Phi: 'Φ',
}

/** Functions that are set upright and keep their name. */
const FUNCTIONS = new Set([
  'ln', 'log', 'exp', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot',
  'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
  'lim', 'max', 'min', 'det', 'gcd', 'mod',
])

/** Commands that only produce space, and how much. */
const SPACING: Record<string, string> = {
  ',': ' ',
  ';': ' ',
  ':': ' ',
  ' ': ' ',
  quad: ' ',
  qquad: '  ',
  '!': '',
}

/** Commands taking one braced argument, rendered as their content. */
const TRANSPARENT = new Set([
  'text', 'mathrm', 'mathbf', 'mathit', 'textbf', 'textit', 'operatorname',
  'mathsf', 'mbox', 'textrm',
])

interface Reader {
  source: string
  at: number
}

/** Read a `{...}` group, honouring nesting. Assumes `{` is next. */
function readGroup(reader: Reader): string {
  let depth = 0
  const start = reader.at + 1
  while (reader.at < reader.source.length) {
    const character = reader.source[reader.at]
    if (character === '\\') {
      reader.at += 2 // an escaped brace is not a delimiter
      continue
    }
    if (character === '{') depth += 1
    else if (character === '}') {
      depth -= 1
      if (depth === 0) {
        const body = reader.source.slice(start, reader.at)
        reader.at += 1
        return body
      }
    }
    reader.at += 1
  }
  return reader.source.slice(start) // unclosed: take the rest
}

/**
 * The next argument to a command: a braced group, a command, or one character.
 *
 * `\sqrt{1+x}`, `\sqrt2` and `x^\alpha` all have to work, because all three
 * are things a model writes.
 */
function readArgument(reader: Reader): string {
  while (reader.source[reader.at] === ' ') reader.at += 1
  const character = reader.source[reader.at]
  if (character === undefined) return ''
  if (character === '{') return readGroup(reader)
  if (character === '\\') {
    const match = /^\\([a-zA-Z]+|.)/.exec(reader.source.slice(reader.at))
    if (match) {
      reader.at += match[0].length
      return match[0]
    }
  }
  reader.at += 1
  return character
}

function Fraction({ over, under }: { over: string; under: string }) {
  return (
    <span className="mx-[0.15em] inline-flex flex-col items-center align-middle text-[0.95em] leading-tight">
      <span className="px-[0.3em]">{render(over)}</span>
      <span className="w-full border-t border-current px-[0.3em]">
        {render(under)}
      </span>
    </span>
  )
}

function Root({ body }: { body: string }) {
  return (
    <span className="whitespace-nowrap">
      {'√'}
      {/* The overline is what makes the extent of the root readable. */}
      <span className="border-t border-current pt-[0.1em]">{render(body)}</span>
    </span>
  )
}

/** Turn a LaTeX fragment into React nodes. */
function render(source: string): ReactNode[] {
  const reader: Reader = { source, at: 0 }
  const nodes: ReactNode[] = []
  let text = ''
  let key = 0

  const flush = () => {
    if (text) {
      nodes.push(text)
      text = ''
    }
  }

  while (reader.at < source.length) {
    const character = source[reader.at]

    if (character === '\\') {
      const match = /^\\([a-zA-Z]+|.)/.exec(source.slice(reader.at))
      if (!match) {
        text += character
        reader.at += 1
        continue
      }
      const name = match[1]
      reader.at += match[0].length

      if (name === 'frac' || name === 'dfrac' || name === 'tfrac') {
        const over = readArgument(reader)
        const under = readArgument(reader)
        flush()
        nodes.push(<Fraction key={key++} over={over} under={under} />)
      } else if (name === 'sqrt') {
        // An index - \sqrt[3]{x} - is read and shown before the sign.
        let index = ''
        if (source[reader.at] === '[') {
          const close = source.indexOf(']', reader.at)
          if (close !== -1) {
            index = source.slice(reader.at + 1, close)
            reader.at = close + 1
          }
        }
        const body = readArgument(reader)
        flush()
        if (index) nodes.push(<sup key={key++}>{render(index)}</sup>)
        nodes.push(<Root key={key++} body={body} />)
      } else if (TRANSPARENT.has(name)) {
        // The content is ordinary text: recursed, so `\text{H}_2` still
        // subscripts, which is exactly how the chemistry is written.
        text += readArgument(reader)
      } else if (name === 'left' || name === 'right') {
        const bracket = readArgument(reader)
        text += bracket === '.' ? '' : bracket
      } else if (name in SPACING) {
        text += SPACING[name]
      } else if (name in SYMBOLS) {
        text += SYMBOLS[name]
      } else if (FUNCTIONS.has(name)) {
        text += name
      } else if (name === '\\') {
        flush()
        nodes.push(<br key={key++} />)
      } else if (/^[{}$%&_#]$/.test(name)) {
        text += name // an escaped literal
      } else {
        // Unknown: show it as written rather than swallowing it, so a gap in
        // this table looks like a gap and not like missing content.
        text += `\\${name}`
      }
      continue
    }

    if (character === '_' || character === '^') {
      reader.at += 1
      const body = readArgument(reader)
      flush()
      const Tag = character === '_' ? 'sub' : 'sup'
      nodes.push(<Tag key={key++}>{render(body)}</Tag>)
      continue
    }

    if (character === '{' || character === '}') {
      // Grouping braces carry no meaning of their own once parsed.
      reader.at += 1
      continue
    }

    text += character
    reader.at += 1
  }

  flush()
  return nodes
}

/**
 * One piece of maths, typeset by the renderer above.
 *
 * `display` gets its own centred line, which is what `$$` and `\[` mean;
 * inline maths sits in the sentence.
 */
/*
 * Named `Formula` rather than `Math` on purpose: `Math` shadows the global
 * inside any module that imports it, and the first thing that broke was an
 * innocent `Math.min` in the heading renderer.
 */
export function PlainFormula({
  source,
  display = false,
}: {
  source: string
  display?: boolean
}) {
  const body = <span className="font-serif">{render(source)}</span>
  if (!display) return body
  return (
    <div className="my-3 overflow-x-auto py-1 text-center text-[1.05em]">
      {body}
    </div>
  )
}

// --- KaTeX, fetched on first formula ----------------------------------------

/**
 * Only what is called, rather than `typeof import('katex')` — the namespace
 * type carries a self-reference the default export does not have, so naming
 * the whole module here does not typecheck against what actually arrives.
 */
type Katex = {
  renderToString: (
    tex: string,
    options?: import('katex').KatexOptions,
  ) => string
}

/**
 * The KaTeX module, or null until it arrives.
 *
 * Module-level rather than per-component state because there is exactly one
 * KaTeX and every formula on the page wants the same one: the import is
 * started once, and each mounted formula is told when it lands. Without this
 * a reply with eight formulae would start eight imports and re-render eight
 * times over.
 */
let katex: Katex | null = null
let loading: Promise<void> | null = null
let unavailable = false
const waiting = new Set<() => void>()

function loadKatex(): void {
  if (katex || unavailable || loading) return
  loading = Promise.all([
    import('katex'),
    // Imported here rather than at the top of the file so the stylesheet is
    // part of the lazy chunk too. At the top, Vite would fold it into the
    // main stylesheet and the "costs nothing until needed" claim would be
    // false for 25 kB of it.
    import('katex/dist/katex.min.css'),
  ])
    .then(([module]) => {
      katex = module.default ?? (module as unknown as Katex)
    })
    .catch(() => {
      // Offline, or the chunk is missing. The renderer above already drew
      // this formula, so there is nothing to report and nothing to retry.
      unavailable = true
    })
    .finally(() => {
      loading = null
      for (const wake of waiting) wake()
    })
}

/** Re-render this formula once KaTeX is available. */
function useKatex(): Katex | null {
  const [module, setModule] = useState<Katex | null>(katex)

  useEffect(() => {
    if (module || unavailable) return
    const wake = () => setModule(katex)
    waiting.add(wake)
    loadKatex()
    return () => {
      waiting.delete(wake)
    }
  }, [module])

  return module
}

/**
 * One piece of maths: the small renderer now, KaTeX once it has loaded.
 *
 * The swap is deliberate rather than a loading state. Something correct is on
 * screen from the first paint, and if KaTeX never arrives — no network, chunk
 * missing — what is already there simply stays.
 */
export function Formula({
  source,
  display = false,
}: {
  source: string
  display?: boolean
}) {
  const katexModule = useKatex()

  if (!katexModule) return <PlainFormula source={source} display={display} />

  let html: string
  try {
    html = katexModule.renderToString(source, {
      displayMode: display,
      // `trust` is left at its default of false on purpose - see the note at
      // the top of this file. With it off, KaTeX emits no raw markup.
      throwOnError: true,
      strict: false,
    })
  } catch {
    // A formula KaTeX will not parse. The small renderer is more forgiving -
    // it shows an unknown command as its own text rather than failing - so
    // fall back to it rather than showing an error where maths should be.
    return <PlainFormula source={source} display={display} />
  }

  if (!display) {
    return <span dangerouslySetInnerHTML={{ __html: html }} />
  }
  return (
    <div
      className="my-3 overflow-x-auto py-1"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
