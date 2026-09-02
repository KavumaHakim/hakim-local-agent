/**
 * Read one reply aloud.
 *
 * Opt-in per message rather than automatic: a long answer full of tool output
 * reading itself at you is worse than silence, and pressing a button is a
 * cheaper way to say "this one" than a setting is.
 *
 * The first press after an idle period waits about seven seconds while the
 * voice loads, and the ones after it are under half a second — so the loading
 * state is not decoration, it is most of what you will see the first time.
 *
 * Markdown is stripped before sending. The voice pronounces `**` and backticks
 * as nothing useful, and a reply full of code fences read literally is
 * unlistenable.
 */

import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api'
import { SpeakerIcon, StopIcon } from './Icons'

type State = 'idle' | 'loading' | 'playing'

export function SpeakButton({ text }: { text: string }) {
  const [state, setState] = useState<State>('idle')
  const [error, setError] = useState<string | null>(null)
  const audio = useRef<HTMLAudioElement | null>(null)
  const url = useRef<string | null>(null)

  // Stop and release on unmount. An object URL kept past its audio element is
  // a leak the browser will not collect on its own, and audio that outlives
  // the message it belongs to is worse than a leak.
  useEffect(() => () => release(), [])

  function release() {
    audio.current?.pause()
    audio.current = null
    if (url.current) {
      URL.revokeObjectURL(url.current)
      url.current = null
    }
  }

  async function play() {
    if (state === 'playing') {
      release()
      setState('idle')
      return
    }

    setState('loading')
    setError(null)
    try {
      const wav = await api.speak(text)
      release()
      url.current = URL.createObjectURL(wav)
      const element = new Audio(url.current)
      element.onended = () => {
        setState('idle')
        release()
      }
      element.onerror = () => {
        setError('That audio would not play.')
        setState('idle')
        release()
      }
      audio.current = element
      await element.play()
      setState('playing')
    } catch (failure) {
      setError(
        failure instanceof ApiError
          ? failure.message
          : 'That could not be read aloud.',
      )
      setState('idle')
      release()
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => void play()}
        disabled={state === 'loading'}
        title={
          state === 'playing'
            ? 'Stop'
            : state === 'loading'
              ? 'Loading the voice…'
              : 'Read this aloud'
        }
        className={
          state === 'playing'
            ? 'grid size-[22px] place-items-center rounded-sm text-accent transition hover:bg-tint'
            : 'grid size-[22px] place-items-center rounded-sm text-faint transition hover:bg-tint hover:text-fg disabled:opacity-40'
        }
      >
        {state === 'playing' ? (
          <StopIcon className="size-3" />
        ) : (
          <SpeakerIcon
            className={state === 'loading' ? 'size-3.5 animate-pulse' : 'size-3.5'}
          />
        )}
      </button>
      {error && <span className="text-[11px] text-danger">{error}</span>}
    </>
  )
}

/**
 * Markdown, reduced to what a voice can say.
 *
 * Code blocks go entirely rather than being read: a fenced block is the one
 * part of an answer nobody wants spoken, and reading punctuation aloud for
 * thirty seconds is how you learn that.
 */
export function speakable(markdown: string): string {
  return (
    markdown
      // Fenced code, gone. The rest of the reply usually says what it does.
      .replace(/```[\s\S]*?```/g, ' (code) ')
      .replace(/`([^`]*)`/g, '$1')
      // Links read as their text, not their URL.
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
      // Emphasis and headings are punctuation to a voice.
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/(\*\*|__)(.*?)\1/g, '$2')
      .replace(/(\*|_)(.*?)\1/g, '$2')
      .replace(/^\s*[-*+]\s+/gm, '')
      .replace(/^\s*>\s?/gm, '')
      // Table pipes and rules become nothing, not "pipe pipe pipe".
      .replace(/^\s*\|.*\|\s*$/gm, ' ')
      .replace(/^\s*[-=_]{3,}\s*$/gm, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  )
}
