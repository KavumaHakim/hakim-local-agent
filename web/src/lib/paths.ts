/**
 * Small helpers for the filesystem paths the API deals in.
 *
 * They are strings from the server, not browser paths, and they arrive in
 * whichever style the machine uses — so anything here has to handle both
 * separators rather than assuming the one this file was written on.
 */

/** The last segment of a path, for a label. */
export function basename(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean)
  return parts[parts.length - 1] ?? path
}
