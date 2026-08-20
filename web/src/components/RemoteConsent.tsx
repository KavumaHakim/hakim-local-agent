/**
 * Asks before a turn leaves this machine.
 *
 * Shown only when the auto-router chose a hosted model. Picking one yourself
 * is already a deliberate act and is never interrupted.
 *
 * It says what actually goes, not a vague "data may be shared": the prompt,
 * the conversation so far, and any tool results — which, with filesystem
 * reading on by default, means file contents. Someone agreeing to this should
 * know that before they click, not afterwards.
 */

import type { RemoteConfirmation } from '../lib/api'
import { AlertIcon, CloudIcon } from './Icons'

interface Props {
  request: RemoteConfirmation
  localLabel: string
  onApprove: () => void
  onDecline: () => void
  onDismiss: () => void
}

export function RemoteConsent({
  request,
  localLabel,
  onApprove,
  onDecline,
  onDismiss,
}: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onMouseDown={onDismiss}
    >
      <div
        className="w-full max-w-md overflow-hidden rounded-2xl border border-line bg-raised shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-line p-4">
          <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-warn/15 text-warn">
            <CloudIcon className="size-4" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold">Send this turn to {request.label}?</h2>
            <p className="mt-1 text-xs text-muted">
              Auto-routing picked it — {request.reason}
            </p>
          </div>
        </div>

        <div className="space-y-3 p-4">
          <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2">
            <AlertIcon className="mt-0.5 size-3.5 shrink-0 text-warn" />
            <div className="text-xs leading-relaxed text-muted">
              <p>
                This leaves your machine. Sent to{' '}
                <span className="text-fg">{request.provider}</span>:
              </p>
              <ul className="mt-1.5 list-disc space-y-0.5 pl-4">
                <li>your prompt</li>
                <li>this conversation's history</li>
                <li>
                  any tool results — including the contents of files the agent
                  reads
                </li>
              </ul>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={onApprove}
              className="w-full rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:opacity-90"
            >
              Send to {request.label}
            </button>
            <button
              type="button"
              onClick={onDecline}
              className="w-full rounded-lg border border-line px-3 py-2 text-sm transition hover:border-accent/50 hover:text-accent"
            >
              Keep it local — run on {localLabel}
            </button>
            <button
              type="button"
              onClick={onDismiss}
              className="w-full px-3 py-1 text-xs text-faint transition hover:text-fg"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
