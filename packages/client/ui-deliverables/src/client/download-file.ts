/** Browser-side download of a delivered file, through the Host's authorized byte read. */
import type { RemoteResult } from '@deepseek-ai/dsh-api-remotes/client'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { WorkspaceFileBytes } from '@deepseek-ai/dsh-api-workspace-files/types'

/** The slice of the Client Remote the download performs. */
export interface WorkspaceFilesDownloadRemote {
  readonly workspaceFiles: {
    /**
     * Read a complete file without text decoding.
     * @param sessionId - the session whose workspace resolves `path`.
     * @param path - workspace path, absolute or relative to the workspace root.
     * @param options - empty options read the complete file under the Host's cap.
     * @param signal - cancels the call.
     * @returns the file's native bytes, or the failure the Host declares.
     */
    readBytes(
      sessionId: SessionId,
      path: string,
      options: Record<string, never>,
      signal?: AbortSignal,
    ): Promise<RemoteResult<WorkspaceFileBytes<Uint8Array<ArrayBuffer>>>>
  }
}

/** Download lifecycle published on the owning file card. */
export type PresentedDownloadPhase = 'downloading' | 'downloaded' | 'error'

/**
 * Read one workspace file through the Host and hand it to the browser as a download.
 * @param remote - the Client Remote carrying the `workspaceFiles` namespace.
 * @param sessionId - the session whose workspace resolves `path`.
 * @param path - workspace path, absolute or relative to the workspace root.
 * @param name - file name the browser saves as.
 * @param signal - caller cancellation.
 * @returns whether the download started.
 */
export async function downloadWorkspaceFile(
  remote: WorkspaceFilesDownloadRemote,
  sessionId: SessionId,
  path: string,
  name: string,
  signal: AbortSignal,
): Promise<boolean> {
  // A missing namespace (plugin composed without remote.workspaceFiles) fails here, not at the call site.
  const read = remote?.workspaceFiles?.readBytes
  if (typeof read !== 'function') return false
  const result = await read.call(remote.workspaceFiles, sessionId, path, {}, signal)
  if (!result.ok) return false
  const data = result.value.data
  if (!(data instanceof Uint8Array)) return false
  // A copy keeps the detached view the transport returned valid for the Blob read.
  const blob = new Blob([data.slice().buffer as ArrayBuffer], { type: 'application/octet-stream' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.rel = 'noopener'
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  // Revoke on the next macrotask so the navigation the click started has the URL.
  setTimeout(() => { URL.revokeObjectURL(url) }, 10_000)
  return true
}
