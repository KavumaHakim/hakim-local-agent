/**
 * Syntax highlighting for fenced code blocks.
 *
 * Written rather than installed, for the reason the markdown renderer was:
 * highlight.js is ~120 KB minified for its common languages and Shiki ships a
 * WASM regex engine, which is more than the whole current bundle. The same
 * argument that parked KaTeX applies here, and the subset that actually
 * appears in these replies is small.
 *
 * This produces **tokens, not HTML**. The caller turns them into React
 * elements, so there is still no `dangerouslySetInnerHTML` anywhere in the
 * app and a code block containing `<script>` is still five harmless
 * characters.
 *
 * Two properties matter more than breadth of language support:
 *
 * **An unknown language is not guessed at.** It returns one plain token, and
 * the block renders exactly as it did before. Guessing gets Python keywords
 * highlighted inside a log file, which looks like a bug in the model's output
 * rather than in the renderer.
 *
 * **A half-arrived block must not misbehave.** Code streams in a token at a
 * time, so at some point every string literal is unterminated and every block
 * comment is unclosed. Every rule here therefore accepts an unterminated form
 * that stops at the end of the line (or, where the construct really does span
 * lines, at the end of the text). Without that, one `"` arriving mid-stream
 * paints the rest of the file as a string and then unpaints it a second later.
 */

export type TokenKind = 'comment' | 'string' | 'keyword' | 'number' | 'name' | 'plain'

export interface Token {
  text: string
  kind: TokenKind
}

/**
 * Above this many characters, the block is left plain.
 *
 * Tokenising runs on every render, and a streaming reply re-renders on every
 * token. A 20,000-character block is already past the point where anyone is
 * reading it as code rather than scrolling through it, and this machine has
 * two cores to spare between the browser and a running model.
 */
const MAX_HIGHLIGHT_CHARS = 20_000

// --- the pieces languages are assembled from ---

// A quoted string, stopping at the end of the line if it is never closed.
// The optional closing quote is what makes a streaming block behave.
const dq = String.raw`"(?:\\.|[^"\\\n])*"?`
const sq = String.raw`'(?:\\.|[^'\\\n])*'?`

// Genuinely multi-line constructs: unterminated, they run to the end of the
// text, which is correct - that is what they mean.
const triple = String.raw`"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)`
const backtick = String.raw`\`(?:\\.|[^\`\\])*\`?`

const slashComment = String.raw`//[^\n]*|/\*[\s\S]*?(?:\*/|$)`
const hashComment = String.raw`#[^\n]*`
const dashComment = String.raw`--[^\n]*|/\*[\s\S]*?(?:\*/|$)`
const blockComment = String.raw`/\*[\s\S]*?(?:\*/|$)`

const number = String.raw`\b(?:0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)\b`
const word = String.raw`[A-Za-z_$][A-Za-z0-9_$]*`

interface Spec {
  /** Regex sources, in priority order. Whatever matches earliest wins. */
  parts: string[]
  keywords: Set<string>
  /** Words that are a `name` rather than a `keyword` - types, builtins. */
  names?: Set<string>
  /** Keywords are case-insensitive. SQL is written both ways and read as one. */
  foldCase?: boolean
  /** Bare words are names, not plain text. HTML attributes, and nothing else. */
  bareWordsAreNames?: boolean
}

const set = (words: string) => new Set(words.trim().split(/\s+/))

const PYTHON: Spec = {
  parts: [hashComment, triple, dq, sq, number, word],
  keywords: set(`
    False None True and as assert async await break class continue def del elif
    else except finally for from global if import in is lambda nonlocal not or
    pass raise return try while with yield match case self cls
  `),
  names: set(`
    print len range str int float bool list dict set tuple open enumerate zip
    isinstance sorted sum min max abs any all type super repr Exception
  `),
}

const JS: Spec = {
  parts: [slashComment, backtick, dq, sq, number, word],
  keywords: set(`
    as async await break case catch class const continue debugger default
    delete do else enum export extends false finally for from function get if
    implements import in instanceof interface let new null of package private
    protected public return satisfies set static super switch this throw true
    try type typeof var void while with yield keyof readonly declare namespace
    abstract override infer asserts is
  `),
  names: set(`
    any bigint boolean never number object string symbol undefined unknown
    Array Boolean Date Error JSON Map Math Number Object Promise RegExp Set
    String Symbol console document window
  `),
}

const JSON_SPEC: Spec = {
  parts: [dq, number, word],
  keywords: set('true false null'),
}

const SHELL: Spec = {
  parts: [
    hashComment,
    dq,
    sq,
    // A variable, with or without braces. Named before the word rule so `$PATH`
    // is one token rather than a stray `$` and a name.
    String.raw`\$\{[^}\n]*\}?|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9@*?#!$]`,
    number,
    word,
  ],
  keywords: set(`
    if then elif else fi for while until do done case esac in function return
    break continue local export readonly declare source exit trap set unset
    shift eval exec
  `),
  names: set(`
    echo cd ls cat grep sed awk find mkdir rm cp mv touch chmod curl wget git
    python python3 pip npm node docker make sudo test printf read pwd which
  `),
}

const CSS: Spec = {
  parts: [
    blockComment,
    dq,
    sq,
    // An at-rule, a hex colour, or a unit-carrying number. Ahead of the plain
    // number rule, which would otherwise split `1.5rem` in two.
    String.raw`@[A-Za-z-]+|#[0-9a-fA-F]{3,8}\b|\b\d[\d.]*(?:px|rem|em|%|vh|vw|s|ms|deg|fr|ch)\b`,
    number,
    String.raw`[A-Za-z-][A-Za-z0-9-]*`,
  ],
  keywords: set(`
    important inherit initial unset none auto flex grid block inline absolute
    relative fixed sticky hidden visible solid dashed dotted bold italic center
    left right transparent currentColor var calc
  `),
}

const SQL: Spec = {
  parts: [dashComment, dq, sq, number, word],
  keywords: set(`
    ADD ALL ALTER AND AS ASC BETWEEN BY CASE CHECK COLUMN CONSTRAINT CREATE
    DEFAULT DELETE DESC DISTINCT DROP ELSE END EXISTS FOREIGN FROM FULL GROUP
    HAVING IN INDEX INNER INSERT INTO IS JOIN KEY LEFT LIKE LIMIT NOT NULL
    OFFSET ON OR ORDER OUTER PRIMARY REFERENCES RIGHT SELECT SET TABLE THEN
    UNION UNIQUE UPDATE VALUES VIEW WHEN WHERE WITH
  `),
  foldCase: true,
}

/**
 * HTML, which is tags rather than words.
 *
 * A tag name is styled as a keyword and an attribute name as a name, which is
 * the distinction a reader is actually making. Text between tags stays plain,
 * because it is prose.
 */
const HTML: Spec = {
  parts: [
    String.raw`<!--[\s\S]*?(?:-->|$)`,
    String.raw`<[!/]?[A-Za-z][A-Za-z0-9:-]*`,
    dq,
    sq,
    String.raw`[A-Za-z_:][A-Za-z0-9_:.-]*(?=\s*=)`,
  ],
  keywords: new Set<string>(),
  bareWordsAreNames: true,
}

/** What a fence's language word means. Aliases are how people actually write. */
const LANGUAGES: Record<string, Spec> = {
  python: PYTHON,
  py: PYTHON,
  javascript: JS,
  js: JS,
  jsx: JS,
  typescript: JS,
  ts: JS,
  tsx: JS,
  json: JSON_SPEC,
  jsonc: JSON_SPEC,
  bash: SHELL,
  sh: SHELL,
  shell: SHELL,
  zsh: SHELL,
  console: SHELL,
  css: CSS,
  scss: CSS,
  sql: SQL,
  html: HTML,
  xml: HTML,
  svg: HTML,
}

/** Whether a fence label is one this can colour. Used to decide, not to guess. */
export function canHighlight(language: string): boolean {
  return normalise(language) in LANGUAGES
}

function normalise(language: string): string {
  // A fence is sometimes written ```python title=x, and the first word is the
  // part that names the language.
  return (language || '').trim().toLowerCase().split(/[\s:,]/)[0]
}

/**
 * Split `code` into coloured tokens.
 *
 * Everything is one pass of one alternation, rather than a rule at a time over
 * the whole string. That ordering is the only thing that keeps a keyword
 * inside a comment from being coloured as a keyword: whichever rule matches
 * *earliest* wins the text, and the ones that swallow the most - comments and
 * strings - are listed first so they win ties at the same position.
 */
export function highlight(code: string, language: string): Token[] {
  const spec = LANGUAGES[normalise(language)]
  if (!spec || code.length > MAX_HIGHLIGHT_CHARS) return [{ text: code, kind: 'plain' }]

  const pattern = new RegExp(spec.parts.map((part) => `(?:${part})`).join('|'), 'g')
  // Which comment syntaxes this language actually has. Asked of the spec
  // rather than guessed from the text, because the same two characters mean
  // opposite things elsewhere: `--main` is a CSS custom property and a SQL
  // comment, and `#fff` is a CSS colour and a shell comment.
  const hash = spec.parts.includes(hashComment)
  const dash = spec.parts.includes(dashComment)

  const tokens: Token[] = []
  let cursor = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(code)) !== null) {
    const text = match[0]
    // A zero-width match would not advance and would spin here. None of the
    // rules above can produce one, but a future one could.
    if (!text) {
      pattern.lastIndex += 1
      continue
    }
    if (match.index > cursor) push(tokens, code.slice(cursor, match.index), 'plain')
    push(tokens, text, classify(text, spec, hash, dash))
    cursor = match.index + text.length
  }

  if (cursor < code.length) push(tokens, code.slice(cursor), 'plain')
  return tokens
}

function classify(text: string, spec: Spec, hash: boolean, dash: boolean): TokenKind {
  const first = text[0]

  if (text.startsWith('<!--') || text.startsWith('/*') || text.startsWith('//')) {
    return 'comment'
  }
  if (hash && first === '#') return 'comment'
  if (dash && text.startsWith('--')) return 'comment'

  if (first === '"' || first === "'" || first === '`') return 'string'
  if (first === '<') return 'keyword' // an HTML tag name
  if (first === '@' || first === '#') return 'keyword' // at-rule, hex colour
  if (first === '$') return 'name' // a shell variable
  if (/^[\d.]/.test(text)) return 'number'

  if (spec.keywords.has(text)) return 'keyword'
  if (spec.foldCase && spec.keywords.has(text.toUpperCase())) return 'keyword'
  if (spec.names?.has(text)) return 'name'
  return spec.bareWordsAreNames ? 'name' : 'plain'
}

/** Append, merging into the previous token when the colour is the same. */
function push(tokens: Token[], text: string, kind: TokenKind): void {
  const last = tokens[tokens.length - 1]
  if (last && last.kind === kind) last.text += text
  else tokens.push({ text, kind })
}
