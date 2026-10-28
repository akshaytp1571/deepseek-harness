// @vitest-environment jsdom
/** Delivered-file downloads publish card status, coalesce, and cancel with the plugin. */
import { afterEach, expect, it, vi } from 'vitest'
import { SessionId } from '@deepseek-ai/dsh-session/types'
import { PresentedDownloadController } from '../src/client/download-controller.ts'
import { presentedFileUrl } from '../src/presented.ts'

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

const id = SessionId('fork')
const url = presentedFileUrl(id, 2, 1)

/** A remote whose byte read answers with one byte and counts its calls. */
function stubRemote() {
  const readBytes = vi.fn(async () => ({ ok: true as const, value: { data: new Uint8Array([1]) } }))
  return {
    readBytes,
    remote: { workspaceFiles: { readBytes } } as never,
  }
}

/** jsdom has no object-URL registry, so downloads stub the three DOM calls they make. */
function stubDownloadDom() {
  const clicks: string[] = []
  vi.stubGlobal('URL', { ...URL, createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} })
  const append = vi.spyOn(document.body, 'append').mockImplementation((...nodes) => {
    for (const node of nodes) if (node instanceof HTMLAnchorElement) clicks.push(node.download)
  })
  return {
    clicks,
    restore: () => { append.mockRestore() },
  }
}

it('publishes downloading then downloaded for one delivered file', async () => {
  const { remote, readBytes } = stubRemote()
  const dom = stubDownloadDom()
  const controller = new PresentedDownloadController()
  try {
    await controller.download(remote, id, 2, 1, 'reports/out.pdf', 'out.pdf')
    expect(readBytes).toHaveBeenCalledTimes(1)
    expect(controller.state.getSnapshot()[url]).toBe('downloaded')
    // The browser saves under the delivered basename the caller passed.
    expect(dom.clicks).toEqual(['out.pdf'])
  } finally {
    dom.restore()
    await controller.dispose()
  }
})

it('coalesces a second gesture while the first is pending', async () => {
  const pending = Promise.withResolvers<{ ok: true; value: { data: Uint8Array } }>()
  const readBytes = vi.fn(() => pending.promise)
  const dom = stubDownloadDom()
  const controller = new PresentedDownloadController()
  try {
    const first = controller.download({ workspaceFiles: { readBytes } } as never, id, 2, 1, 'a.txt', 'a.txt')
    // A repeated gesture is dropped, so the Host read happens once.
    await controller.download({ workspaceFiles: { readBytes } } as never, id, 2, 1, 'a.txt', 'a.txt')
    expect(readBytes).toHaveBeenCalledTimes(1)
    pending.resolve({ ok: true, value: { data: new Uint8Array([1]) } })
    await first
  } finally {
    dom.restore()
    await controller.dispose()
  }
})

it('publishes error when the Host read fails or the namespace is absent', async () => {
  const dom = stubDownloadDom()
  const controller = new PresentedDownloadController()
  try {
    await controller.download({ workspaceFiles: { readBytes: async () => ({ ok: false as const }) } } as never, id, 2, 1, 'a.txt', 'a.txt')
    expect(controller.state.getSnapshot()[url]).toBe('error')
    await controller.download({} as never, id, 3, 0, 'b.txt', 'b.txt')
    expect(controller.state.getSnapshot()[presentedFileUrl(id, 3, 0)]).toBe('error')
    expect(dom.clicks).toEqual([])
  } finally {
    dom.restore()
    await controller.dispose()
  }
})

it('clears the success phase after the hold window', async () => {
  vi.useFakeTimers()
  const { remote } = stubRemote()
  const dom = stubDownloadDom()
  const controller = new PresentedDownloadController()
  try {
    await controller.download(remote, id, 2, 1, 'a.txt', 'a.txt')
    expect(controller.state.getSnapshot()[url]).toBe('downloaded')
    vi.advanceTimersByTime(5_000)
    expect(controller.state.getSnapshot()[url]).toBeUndefined()
  } finally {
    dom.restore()
    await controller.dispose()
  }
})

it('drops a gesture after disposal publishes nothing', async () => {
  const { remote } = stubRemote()
  const dom = stubDownloadDom()
  const controller = new PresentedDownloadController()
  try {
    await controller.dispose()
    await controller.download(remote, id, 2, 1, 'a.txt', 'a.txt')
    expect(controller.state.getSnapshot()[url]).toBeUndefined()
  } finally {
    dom.restore()
  }
})
