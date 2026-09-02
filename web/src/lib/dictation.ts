/**
 * Recording a clip in the browser and turning it into something whisper reads.
 *
 * `MediaRecorder` gives you WebM/Opus in Chrome and Ogg/Opus in Firefox, and
 * whisper.cpp reads neither reliably — its own help lists flac, mp3, ogg and
 * wav, and the browsers do not agree on which of those they will produce. So
 * the clip is decoded and re-encoded here as the 16 kHz mono 16-bit WAV that
 * whisper resamples to anyway.
 *
 * Doing it in the browser rather than on the server is what keeps ffmpeg out
 * of the install. The browser already has a full audio decoder — it just
 * decoded the thing it recorded — and `OfflineAudioContext` resamples for
 * free, so this costs a few dozen lines instead of a dependency.
 */

/** 16 kHz mono is what whisper works in; anything else it resamples itself. */
const SAMPLE_RATE = 16000

/** Above this a clip is a recording session, not a dictated message. */
export const MAX_SECONDS = 120

export interface Recording {
  blob: Blob
  seconds: number
}

export function recordingSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia)
  )
}

/**
 * One recording session, started and stopped by the caller.
 *
 * The microphone track is stopped explicitly on the way out. Without it the
 * browser's recording indicator stays lit after the clip is finished, which
 * looks exactly like an application still listening to you.
 */
export class Recorder {
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []
  private stream: MediaStream | null = null
  private startedAt = 0

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // The browser's own cleanup is better than anything done afterwards,
        // and whisper is markedly happier without a room in the background.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    this.chunks = []
    this.recorder = new MediaRecorder(this.stream)
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data)
    }
    this.startedAt = Date.now()
    this.recorder.start()
  }

  get recording(): boolean {
    return this.recorder?.state === 'recording'
  }

  /** Stop, and resolve with what was captured. */
  stop(): Promise<Recording> {
    return new Promise((resolve, reject) => {
      const recorder = this.recorder
      if (!recorder) {
        reject(new Error('Nothing was being recorded.'))
        return
      }
      recorder.onstop = () => {
        const seconds = (Date.now() - this.startedAt) / 1000
        const blob = new Blob(this.chunks, { type: recorder.mimeType })
        this.release()
        resolve({ blob, seconds })
      }
      recorder.onerror = () => {
        this.release()
        reject(new Error('The recording failed.'))
      }
      recorder.stop()
    })
  }

  /** Stop and throw the audio away — for a cancelled recording. */
  cancel(): void {
    try {
      if (this.recorder?.state === 'recording') {
        this.recorder.onstop = null
        this.recorder.stop()
      }
    } finally {
      this.release()
    }
  }

  private release(): void {
    this.stream?.getTracks().forEach((track) => track.stop())
    this.stream = null
    this.recorder = null
    this.chunks = []
  }
}

/**
 * Decode a recorded clip and re-encode it as 16 kHz mono 16-bit WAV.
 *
 * Channels are averaged rather than one being picked: a laptop with a stereo
 * array microphone puts real signal in both, and taking only the left of it
 * throws away half the voice.
 */
export async function toWav(blob: Blob): Promise<Blob> {
  const bytes = await blob.arrayBuffer()

  const AudioCtx =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext
  const context = new AudioCtx()
  let decoded: AudioBuffer
  try {
    decoded = await context.decodeAudioData(bytes)
  } finally {
    // Every open AudioContext holds an audio device open. Left behind, a few
    // recordings in a row exhaust the browser's limit and the next one throws.
    void context.close()
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE))
  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()
  const resampled = await offline.startRendering()

  return encodeWav(resampled.getChannelData(0), SAMPLE_RATE)
}

/**
 * A minimal RIFF/PCM container around 16-bit samples.
 *
 * Written out by hand because it is 20 lines and the alternative is a library
 * for a format that has not changed since 1991.
 */
function encodeWav(samples: Float32Array, rate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i))
    }
  }

  text(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM header length
  view.setUint16(20, 1, true) // format: uncompressed PCM
  view.setUint16(22, 1, true) // channels
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true) // bytes per second
  view.setUint16(32, 2, true) // bytes per frame
  view.setUint16(34, 16, true) // bits per sample
  text(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  for (let i = 0; i < samples.length; i += 1) {
    // Clamped before scaling: decoded audio can sit slightly outside ±1, and
    // letting that wrap turns a loud syllable into a burst of noise.
    const sample = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, sample * 0x7fff, true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}
