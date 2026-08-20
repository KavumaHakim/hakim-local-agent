/**
 * What the current turn is doing, while it does it.
 *
 * Each stage is named rather than folded into one spinner. On this hardware a
 * turn can sit in the queue for minutes, spend two more loading a model, and
 * only then start generating - and a single "thinking..." for all three gives
 * no way to tell progress from a hang.
 */

import { Markdown } from '../lib/markdown'
import { useElapsed } from '../hooks/useResources'
import type { TurnState } from '../hooks/useChat'
import { formatDuration, ToolPills } from './Messages'
import { AlertIcon, ChipIcon, StopIcon } from './Icons'

interface Props {
  turn: TurnState
  onEscalate: () => void
  onDismiss: () => void
  onCancel: () => void
}

export function TurnStatus({ turn, onEscalate, onDismiss, onCancel }: Props) {
  const generating = useElapsed(turn.startedAt)

  if (turn.phase === 'idle') return null

  if (turn.phase === 'error') {
    const { error } = turn
    if (!error) return null
    return (
      <div className="animate-rise rounded-xl border border-danger/40 bg-danger/5 p-4">
        <div className="flex items-start gap-3">
          <AlertIcon className="mt-0.5 size-4 shrink-0 text-danger" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-danger">
              {error.kind === 'iteration_limit'
                ? 'Stopped before finishing'
                : 'That turn failed'}
            </p>
            <p className="mt-1 text-sm break-words text-muted">{error.message}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {error.canEscalate && (
                <button
                  type="button"
                  onClick={onEscalate}
                  className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition hover:opacity-90"
                >
                  Retry on the strong model
                </button>
              )}
              <button
                type="button"
                onClick={onDismiss}
                className="rounded-lg border border-line px-3 py-1.5 text-xs text-muted transition hover:text-fg"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-rise space-y-3">
      {turn.routeReason && (
        <div className="flex items-start gap-2 rounded-lg border border-accent/30 bg-accent-dim/40 px-3 py-2 text-xs text-muted">
          <ChipIcon className="mt-px size-3.5 shrink-0 text-accent" />
          <span>
            Switched to <span className="text-fg">{turn.modelLabel}</span> —{' '}
            {turn.routeReason}
          </span>
        </div>
      )}

      {turn.ramWarning && (
        <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-warn">
          <AlertIcon className="mt-px size-3.5 shrink-0" />
          <span>{turn.ramWarning}</span>
        </div>
      )}

      {turn.phase === 'queued' && (
        <StatusLine
          label={
            turn.position > 0
              ? `Queued — ${turn.position} turn${turn.position === 1 ? '' : 's'} ahead`
              : 'Queued'
          }
          detail="Turns run one at a time: this machine has two cores and room for one model."
          onCancel={onCancel}
        />
      )}

      {turn.phase === 'loading' && (
        <StatusLine
          label={`Loading ${turn.modelLabel ?? 'the model'}…`}
          detail="Cold starts take a few minutes here. The weights are read from disk."
          onCancel={onCancel}
        />
      )}

      {turn.phase === 'generating' && (
        <div>
          <ToolPills tools={turn.tools} />
          {turn.text ? (
            <div className="text-[15px] text-fg">
              <Markdown text={turn.text} />
              <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-accent align-middle" />
            </div>
          ) : (
            <StatusLine
              label={
                turn.tools.length
                  ? 'Working through the tool results…'
                  : 'Generating…'
              }
              detail={`${turn.modelLabel ?? ''} · under one token per second on this CPU`}
              onCancel={onCancel}
            />
          )}
          {generating > 0 && (
            <div className="mt-2 text-xs text-faint tabular-nums">
              {formatDuration(generating)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function StatusLine({
  label,
  detail,
  onCancel,
}: {
  label: string
  detail: string
  onCancel: () => void
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-line bg-surface px-4 py-3">
      <span className="mt-1.5 flex gap-1">
        {[0, 1, 2].map((dot) => (
          <span
            key={dot}
            className="size-1.5 animate-breathe rounded-full bg-accent"
            style={{ animationDelay: `${dot * 0.25}s` }}
          />
        ))}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm text-fg">{label}</p>
        <p className="mt-0.5 text-xs text-faint">{detail}</p>
      </div>
      <button
        type="button"
        onClick={onCancel}
        title="Stop watching. The turn finishes and its answer is saved."
        className="rounded-lg border border-line p-1.5 text-faint transition hover:text-fg"
      >
        <StopIcon className="size-3.5" />
      </button>
    </div>
  )
}
