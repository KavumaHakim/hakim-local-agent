/**
 * What the API refused, said where it cannot be missed.
 *
 * These messages were a line of small red text at the foot of whichever pane
 * you happened to be in, and the useful ones are exactly the ones you do not
 * see there: "A turn is running. Tool changes apply from the next turn." You
 * flip a switch, the switch flips back, and the sentence explaining why is
 * eleven pixels tall and below the fold of a 262px column.
 *
 * The models pane was worse. `useModels` has tracked an error since it was
 * written and nothing ever rendered it, so a load that ran out of RAM failed
 * in total silence.
 *
 * So: one dialog, mounted once, fed by whichever hook last refused. It is
 * deliberately small and has one button — there is nothing to decide here,
 * only something to read. Escape and the backdrop close it too, because a
 * dialog you cannot dismiss without aiming is worse than the red text was.
 */

import { useEffect, useRef } from 'react'
import { AlertIcon } from './Icons'

interface Props {
  message: string
  onDismiss: () => void
}

export function Alert({ message, onDismiss }: Props) {
  const button = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onDismiss()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onDismiss])

  // So Enter and Space dismiss it without anyone reaching for the mouse, and
  // so a screen reader lands on the message rather than the page behind it.
  useEffect(() => {
    button.current?.focus()
  }, [])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onMouseDown={onDismiss}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-label="That did not work"
        className="w-full max-w-sm overflow-hidden rounded-2xl border border-line bg-raised shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-3 p-4">
          <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-danger/15 text-danger">
            <AlertIcon className="size-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold">That did not work</h2>
            {/* The server's own words, wrapped and scrollable rather than
                truncated: the sentence that says what to do instead is
                usually the second one.

                `break-words` because these messages quote paths and urls,
                and setting any overflow on one axis makes the other `auto` -
                so without it a path scrolls sideways out of the dialog
                instead of wrapping, which is the truncation this replaced. */}
            <p className="mt-1 max-h-56 overflow-y-auto text-xs leading-relaxed break-words whitespace-pre-wrap text-muted">
              {message}
            </p>
          </div>
        </div>

        <div className="flex justify-end border-t border-line px-4 py-3">
          <button
            ref={button}
            type="button"
            onClick={onDismiss}
            className="rounded-md border border-line px-3 py-1.5 text-xs transition hover:border-accent-line hover:text-accent"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
