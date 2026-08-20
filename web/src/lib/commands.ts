/**
 * The command list, declared once.
 *
 * Both the slash hints in the composer and the ⌘K palette read this, so the
 * two can never drift apart. The server knows nothing about commands: it has
 * REST endpoints, and these are the client's shorthand for them. That is the
 * difference from the Streamlit build, where `/model` was parsed server-side
 * out of the message text.
 */

export type CommandId =
  | 'help'
  | 'tools'
  | 'models'
  | 'model'
  | 'load'
  | 'unload'
  | 'auto'
  | 'thinking'
  | 'clear'
  | 'new'

export interface CommandSpec {
  id: CommandId
  slash: string
  /** Placeholder shown after the command when it takes an argument. */
  argument?: string
  title: string
  hint: string
}

export const COMMANDS: CommandSpec[] = [
  { id: 'new', slash: '/new', title: 'New conversation', hint: 'Start a fresh transcript' },
  {
    id: 'model',
    slash: '/model',
    argument: '<key>',
    title: 'Switch model',
    hint: 'Select a model; it loads on your next message',
  },
  { id: 'models', slash: '/models', title: 'Show models', hint: 'What exists and what is loaded' },
  { id: 'load', slash: '/load', argument: '<key>', title: 'Load model now', hint: 'Start the server before sending' },
  { id: 'unload', slash: '/unload', title: 'Unload model', hint: 'Give the RAM back' },
  {
    id: 'auto',
    slash: '/auto',
    title: 'Toggle auto-routing',
    hint: 'Simple prompts to the fast model, involved ones to the strong one',
  },
  {
    id: 'thinking',
    slash: '/thinking',
    title: 'Toggle extended thinking',
    hint: 'Qwen3 reasons before answering. Much slower on CPU',
  },
  { id: 'tools', slash: '/tools', title: 'Show tools', hint: 'Enabled tools, and why the rest are not' },
  { id: 'clear', slash: '/clear', title: 'Clear conversation', hint: 'Empty the transcript' },
  { id: 'help', slash: '/help', title: 'Help', hint: 'List every command' },
]

/** Commands whose slash form starts with `text`. */
export function matchCommands(text: string): CommandSpec[] {
  const query = text.trim().toLowerCase()
  if (!query.startsWith('/')) return []
  const word = query.split(/\s+/)[0]
  return COMMANDS.filter((command) => command.slash.startsWith(word))
}

/** Split "/model tiny" into its command spec and argument. */
export function parseCommand(
  text: string,
): { spec: CommandSpec; argument: string } | null {
  const trimmed = text.trim()
  if (!trimmed.startsWith('/')) return null
  const [word, ...rest] = trimmed.split(/\s+/)
  const spec = COMMANDS.find((command) => command.slash === word.toLowerCase())
  if (!spec) return null
  return { spec, argument: rest.join(' ').trim() }
}
