/** Shared download status for delivered files, mirroring the native-open controller. */
import { createSnapshotStore } from '@deepseek-ai/dsh-client-store'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import { downloadWorkspaceFile, type PresentedDownloadPhase, type WorkspaceFilesDownloadRemote } from './download-file.ts'
import { presentedFileUrl } from '../presented.ts'

/** Success feedback remains fully visible for five seconds before fading. */
const DOWNLOAD_SUCCESS_HOLD_MS = 5000

/** One browser plugin's file-download requests, cancelled when that plugin is disposed. */
export class PresentedDownloadController {
  /** File action URLs key the state across Sessions and turns. */
  readonly state = createSnapshotStore<Record<string, PresentedDownloadPhase | undefined>>({})
  private readonly expiry = new Map<string, ReturnType<typeof setTimeout>>()
  private readonly lifetime = new AbortController()
  private readonly pending = new Set<Promise<void>>()

  /**
   * Download one delivered file once while a request for the same coordinates is pending.
   * Failures remain visible on the card and a later gesture retries them.
   * @param sessionId - viewed Session, including a fork's own identity.
   * @param seq - durable delivery event sequence.
   * @param index - original file index within that event.
   * @param path - workspace path of the file.
   * @param name - file name the browser saves as.
   * @returns after card status is published.
   */
  download(
    remote: WorkspaceFilesDownloadRemote, sessionId: SessionId, seq: number, index: number, path: string, name: string,
  ): Promise<void> {
    return this.run(presentedFileUrl(sessionId, seq, index), remote, sessionId, path, name)
  }

  /** Invalidate state on connection replacement; mounted cards keep their last status. */
  reset(): void {
    for (const url of this.expiry.keys()) this.clearExpiry(url)
    this.state.set({})
  }

  /** Cancel outstanding requests and wait until no request can publish state. */
  async dispose(): Promise<void> {
    this.lifetime.abort()
    for (const url of this.expiry.keys()) this.clearExpiry(url)
    await Promise.all(this.pending)
  }

  private async run(
    url: string, remote: WorkspaceFilesDownloadRemote, sessionId: SessionId, path: string, name: string,
  ): Promise<void> {
    const phase = this.state.getSnapshot()[url]
    if (this.lifetime.signal.aborted || phase === 'downloading' || phase === 'downloaded') return
    this.clearExpiry(url)
    this.state.update((state) => { state[url] = 'downloading' })
    const task = downloadWorkspaceFile(remote, sessionId, path, name, this.lifetime.signal)
      .then((ok) => {
        this.publish(url, ok ? 'downloaded' : 'error')
      })
      .catch(() => {
        this.publish(url, 'error')
      })
    this.pending.add(task)
    try { await task }
    finally { this.pending.delete(task) }
  }

  private publish(url: string, phase: PresentedDownloadPhase): void {
    if (this.lifetime.signal.aborted) return
    if (phase === 'downloaded') {
      this.expiry.set(url, setTimeout(() => {
        this.expiry.delete(url)
        this.state.update((state) => { Reflect.deleteProperty(state, url) })
      }, DOWNLOAD_SUCCESS_HOLD_MS))
    }
    this.state.update((state) => { state[url] = phase })
  }

  private clearExpiry(url: string): void {
    clearTimeout(this.expiry.get(url))
    this.expiry.delete(url)
  }
}
