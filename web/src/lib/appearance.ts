/**
 * How the interface looks, and remembering it.
 *
 * Two things live here. The theme, which the rail already toggled but which
 * reset to dark on every reload because nothing was stored. And the reading
 * settings — the size and family of the conversation text — which are the
 * point of the exercise: the chrome is dense on purpose and should stay that
 * way, while a long reply on a small screen is a different problem with a
 * different answer.
 *
 * So this deliberately does not scale the whole interface. It changes the
 * text you actually read, and leaves the rail, the composer and the panels at
 * the density the design system chose.
 *
 * Stored in localStorage rather than on the server: it describes this browser,
 * not this agent, and the same weights opened from a different machine should
 * be free to look different. Every access is guarded — a browser with storage
 * disabled throws on read, and appearance is not worth failing to start over.
 */

export type Theme = 'dark' | 'light'
export type ReadingFont = 'sans' | 'serif' | 'mono'

export interface Appearance {
  theme: Theme
  /** Conversation text size, in px. */
  readingSize: number
  readingFont: ReadingFont
}

const KEY = 'hakim.appearance'

export const DEFAULTS: Appearance = {
  theme: 'dark',
  readingSize: 15,
  readingFont: 'sans',
}

/** The offered sizes. A slider would invite 14.5px, which helps nobody. */
export const READING_SIZES = [
  { label: 'Small', value: 13 },
  { label: 'Default', value: 15 },
  { label: 'Large', value: 17 },
  { label: 'Larger', value: 20 },
] as const

/**
 * The stacks each choice maps to.
 *
 * Serif is here because it is the one that earns its place: long prose reads
 * better in it, and this is an app that produces long prose slowly. Mono is
 * for reading output as output.
 */
export const READING_FONTS: Record<ReadingFont, { label: string; stack: string }> = {
  sans: {
    label: 'Sans',
    stack:
      "'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  },
  serif: {
    label: 'Serif',
    stack: "ui-serif, Georgia, Cambria, 'Times New Roman', Times, serif",
  },
  mono: {
    label: 'Mono',
    stack:
      "ui-monospace, 'Cascadia Code', 'JetBrains Mono', Consolas, monospace",
  },
}

/**
 * What the operating system asks for.
 *
 * The starting value only. Once a choice has been stored it wins, because
 * someone who picked dark on a machine set to light meant it.
 */
export function systemTheme(): Theme {
  try {
    return window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark'
  } catch {
    return 'dark'
  }
}

function clean(raw: unknown): Appearance {
  const value = (raw ?? {}) as Partial<Appearance>
  const sizes = READING_SIZES.map((size) => size.value) as readonly number[]
  return {
    theme: value.theme === 'light' ? 'light' : 'dark',
    // Only sizes this build offers, so a value left by an older one - or by
    // someone editing localStorage - cannot produce 400px body text.
    readingSize: sizes.includes(value.readingSize as number)
      ? (value.readingSize as number)
      : DEFAULTS.readingSize,
    readingFont:
      value.readingFont && value.readingFont in READING_FONTS
        ? value.readingFont
        : DEFAULTS.readingFont,
  }
}

export function load(): Appearance {
  const fresh = { ...DEFAULTS, theme: systemTheme() }
  try {
    const stored = window.localStorage.getItem(KEY)
    return stored ? clean(JSON.parse(stored)) : fresh
  } catch {
    // Storage disabled, or a value that is not JSON. Neither is a reason to
    // fail to start.
    return fresh
  }
}

export function save(appearance: Appearance): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(appearance))
  } catch {
    /* Not being able to remember is survivable; not starting is not. */
  }
}

/**
 * Put it on the document.
 *
 * Called once before React renders as well as on every change, so the first
 * paint is already correct rather than flashing the default and correcting
 * itself.
 */
export function apply(appearance: Appearance): void {
  const root = document.documentElement
  root.dataset.theme = appearance.theme
  root.style.setProperty('--reading-size', `${appearance.readingSize}px`)
  root.style.setProperty(
    '--reading-family',
    READING_FONTS[appearance.readingFont].stack,
  )
}
