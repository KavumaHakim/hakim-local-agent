/**
 * A small renderer for the LaTeX the models here actually write.
 *
 * Written rather than installed, for the same reason the markdown renderer
 * was: the subset that turns up is small and known. Counted across the stored
 * replies, twelve commands appear at all, and six of them account for almost
 * everything — `\text` (51 uses), `\sqrt` (24), `\frac` (19), `\rightarrow`
 * (10), `\approx` (6), `\cdot` (5). KaTeX would render the tail better and
 * costs about as much as the entire current bundle; that trade can be made
 * later, and this file is the thing it would replace.
 *
 * Output is React elements, never HTML strings, so there is nothing to
 * sanitise: an unknown command renders as its own text rather than as markup.
 *
 * **Only ever called on text between math delimiters.** That is not a detail.
 * Two of the twelve "commands" found in the corpus were `\Windows` and
 * `\nLine` — a file path and an escape that had nothing to do with maths — so
 * anything that treated a stray backslash as LaTeX would mangle ordinary
 * prose. The delimiters are the whole licence to transform.
 */

import type { ReactNode } from 'react'

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
 * One piece of maths.
 *
 * `display` gets its own centred line, which is what `$$` and `\[` mean;
 * inline maths sits in the sentence.
 */
/*
 * Named `Formula` rather than `Math` on purpose: `Math` shadows the global
 * inside any module that imports it, and the first thing that broke was an
 * innocent `Math.min` in the heading renderer.
 */
export function Formula({
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
