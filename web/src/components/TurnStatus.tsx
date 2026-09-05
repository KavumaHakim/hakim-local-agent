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
import type { Turn } from '../hooks/useChat'
import { formatDuration, ReasoningPanel, ToolPills } from './Messages'
import { AlertIcon, ChipIcon, CloudIcon, HomeIcon, StopIcon } from './Icons'

interface Props {
  turn: Turn
  onEscalate: () => void
  onDismiss: () => void
  onStop: () => void
  /** Answer a command the agent is waiting on. */
  onApprove: (granted: boolean) => void
  /** Abandon this turn and ask the same thing without thinking. */
  onSkipReasoning: () => void
}

/**
 * A command the agent wants to run, and the two buttons that settle it.
 *
 * The turn is genuinely stopped here - a worker thread is sitting on an event
 * waiting for this - so it is drawn as the loudest thing on screen rather than
 * a notice to notice. The command is shown verbatim and in monospace, because
 * the whole value of the gate is that a person can read exactly what will run;
 * a summary would be asking someone to approve a description.
 *
 * Declining is the wider button and comes first. Nothing is lost by declining
 * - the model is told and carries on - and the cost of a mistaken yes is the
 * command actually running.
 */
function ApprovalPrompt({
  approval,
  onAnswer,
}: {
  approval: NonNullable<Turn['approval']>
  onAnswer: (granted: boolean) => void
}) {
  return (
    <div className="animate-rise rounded-xl border border-warn/50 bg-warn/5 p-3">
      <div className="flex items-start gap-2.5">
        <AlertIcon className="mt-px size-4 shrink-0 text-warn" />
        <div className="min-w-0 flex-1">
          <p className="text-[12.5px] text-fg">
            The agent wants to run a command that {approval.reason}.
          </p>
          <pre className="mt-2 overflow-x-auto rounded-md border border-line bg-sunken px-2.5 py-1.5 font-mono text-[12px] text-fg">
            {approval.command}
          </pre>
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => onAnswer(false)}
              className="rounded-md border border-line px-3 py-1.5 text-[11.5px] text-fg transition hover:border-accent-line"
            >
              Don't run it
            </button>
            <button
              type="button"
              onClick={() => onAnswer(true)}
              className="rounded-md border border-warn px-3 py-1.5 text-[11.5px] text-warn transition hover:bg-warn/10"
            >
              Run it
            </button>
            <span className="text-[10.5px] text-faint">
              Declined automatically after{' '}
              {Math.round(approval.timeout / 60)} minutes.
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export function TurnStatus({
  turn,
  onEscalate,
  onDismiss,
  onStop,
  onApprove,
  onSkipReasoning,
}: Props) {
  const generating = useElapsed(turn.startedAt)
  const stopping = turn.stopping

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
                  className="rounded-md border border-accent px-3 py-1.5 text-xs text-accent transition hover:bg-accent-tint"
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
        <div className="flex items-start gap-2 rounded-lg border border-accent/30 bg-accent-tint px-3 py-2 text-xs text-muted">
          <ChipIcon className="mt-px size-3.5 shrink-0 text-accent" />
          <span>
            Switched to <span className="text-fg">{turn.modelLabel}</span> —{' '}
            {turn.routeReason}
          </span>
        </div>
      )}

      {turn.fallback && (
        <div className="flex items-start gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-xs text-muted">
          <HomeIcon className="mt-px size-3.5 shrink-0 text-faint" />
          <span>{turn.fallback.reason}</span>
        </div>
      )}

      {turn.remote && (
        <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-warn">
          <CloudIcon className="mt-px size-3.5 shrink-0" />
          <span>
            This turn is running on {turn.provider ?? 'a hosted provider'} — the
            prompt, history and any tool results are leaving this machine.
          </span>
        </div>
      )}

      {turn.ramWarning && (
        <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-warn">
          <AlertIcon className="mt-px size-3.5 shrink-0" />
          <span>{turn.ramWarning}</span>
        </div>
      )}

      {turn.approval && (
        <ApprovalPrompt
          approval={turn.approval}
          onAnswer={(granted) => onApprove(granted)}
        />
      )}

      {turn.phase === 'queued' && (
        <StatusLine
          label={
            turn.position > 0
              ? `Queued — ${turn.position} turn${turn.position === 1 ? '' : 's'} ahead`
              : 'Queued'
          }
          detail={
            turn.prompt
              ? `“${turn.prompt}” · turns run one at a time: two cores, room for one model.`
              : 'Turns run one at a time: this machine has two cores and room for one model.'
          }
          onStop={onStop}
          stopping={stopping}
        />
      )}

      {turn.phase === 'loading' && (
        <StatusLine
          label={`Loading ${turn.modelLabel ?? 'the model'}…`}
          detail="Cold starts take a few minutes here. The weights are read from disk."
          onStop={onStop}
          stopping={stopping}
        />
      )}

      {turn.phase === 'generating' && (
        <div>
          {/* Open while it streams: on this hardware the thinking is often
              the only thing arriving for minutes, and hiding it would put the
              screen back to blank. */}
          <ReasoningPanel text={turn.reasoning} live />
          {/*
            Only while thinking is all that has arrived. A model cannot be
            told to stop reasoning - there is no such signal in the API - so
            this ends the turn and asks the same question again with thinking
            off. On hardware where deliberating costs minutes, that is the
            button that saves them, and it is worthless once the answer has
            started.
          */}
          {turn.reasoning && !turn.text && !turn.stopping && (
            <div className="mb-2 flex items-center gap-2">
              <button
                type="button"
                onClick={onSkipReasoning}
                className="rounded-md border border-line px-2.5 py-1 text-[11px] text-muted transition hover:border-accent-line hover:text-fg"
              >
                Skip thinking
              </button>
              <span className="text-[10.5px] text-faint">
                Asks again without it. What it has thought so far is discarded.
              </span>
            </div>
          )}
          <ToolPills tools={turn.tools} />
          {turn.text ? (
            <div className="text-[15px] text-fg">
              <Markdown text={turn.text} />
              <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-blink bg-accent align-middle" />
              {/* Once prose is arriving there is no status line to hang the
                  control off, and this is exactly when someone can see the
                  answer going the wrong way and want it stopped. */}
              <div className="mt-2">
                <StopButton onStop={onStop} stopping={stopping} withLabel />
              </div>
            </div>
          ) : (
            <StatusLine
              label={
                turn.tools.length
                  ? 'Working through the tool results…'
                  : 'Generating…'
              }
              detail={`${turn.modelLabel ?? ''} · under one token per second on this CPU`}
              onStop={onStop}
          stopping={stopping}
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
  onStop,
  stopping,
}: {
  label: string
  detail: string
  onStop: () => void
  stopping: boolean
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
      <StopButton onStop={onStop} stopping={stopping} />
    </div>
  )
}

/**
 * Ends the turn, rather than only stopping watching it.
 *
 * It stays enabled while stopping. The turn ends at its next checkpoint - the
 * next token, tool result or model round - and during the one silent stretch
 * where that can take a while, the model reading the prompt, a control that
 * greyed itself out would look like the click had been swallowed.
 */
function StopButton({
  onStop,
  stopping,
  withLabel = false,
}: {
  onStop: () => void
  stopping: boolean
  withLabel?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onStop}
      title="End this turn. Anything already written is kept."
      className={`flex shrink-0 items-center gap-1.5 rounded-lg border border-line text-faint transition hover:border-danger/50 hover:text-danger ${
        withLabel ? 'px-2 py-1 text-[11px]' : 'p-1.5'
      }`}
    >
      <StopIcon className={`size-3.5 ${stopping ? 'animate-breathe' : ''}`} />
      {withLabel && (stopping ? 'Stopping…' : 'Stop')}
    </button>
  )
}
