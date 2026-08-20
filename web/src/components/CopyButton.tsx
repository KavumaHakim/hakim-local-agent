/**
 * Copy text to the clipboard, and say whether it worked.
 *
 * `navigator.clipboard` needs a secure context. `http://127.0.0.1` counts as
 * one — loopback is treated as potentially trustworthy — so this works here
 * without HTTPS. It can still be refused (a permissions policy, a browser that
 * disagrees), so the failure is shown rather than swallowed: a copy button
 * that silently does nothing is worse than one that admits it.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckIcon, CopyIcon } from './Icons'

type State = 'idle' | 'copied' | 'failed'

interface Props {
  text: string
  label?: string
  className?: string
}

export function CopyButton({ text, label = 'Copy', className = '' }: Props) {
  const [state, setState] = useState<State>('idle')
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const copy = useCallback(async () => {
    window.clearTimeout(timer.current)
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    } catch {
      setState('failed')
    }
    timer.current = window.setTimeout(() => setState('idle'), 1600)
  }, [text])

  return (
    <button
      type="button"
      onClick={copy}
      title={state === 'failed' ? 'The browser refused the clipboard' : label}
      aria-label={label}
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] transition ${
        state === 'copied'
          ? 'text-ok'
          : state === 'failed'
            ? 'text-danger'
            : 'text-faint hover:text-fg'
      } ${className}`}
    >
      {state === 'copied' ? (
        <CheckIcon className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      <span>
        {state === 'copied' ? 'Copied' : state === 'failed' ? 'Failed' : label}
      </span>
    </button>
  )
}
