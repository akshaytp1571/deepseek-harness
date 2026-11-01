import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import {
  classifySummarizationFailure,
  isGatewayBodyLimitError,
} from '@deepseek-ai/dsh-compaction-basic/src/failure-class.ts'
import {
  shrinkSpanToBudget,
  selectCompactableRange,
} from '@deepseek-ai/dsh-compaction-basic/src/region.ts'
import { CONTEXT_WINDOW_EXCEEDED_CODE, LlmError } from '@deepseek-ai/dsh-llm'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import SessionStore, { Session, SessionId } from '@deepseek-ai/dsh-session'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import TokenMeter from '@deepseek-ai/dsh-token-meter'

/**
 * Over-budget classification and span shrinking. The reproducer is a real
 * session log: 101 compaction attempts all failed, 94 of them with an nginx
 * `413 Request Entity Too Large` page the harness classified as a context
 * overflow, and 3 with `context_length_exceeded` from the replay exceeding the
 * summarizer route's 524,000-token window. Nothing shrank, so nothing ever
 * compacted.
 */

describe('isGatewayBodyLimitError', () => {
  it('recognizes a proxy body-limit page', () => {
    expect(isGatewayBodyLimitError('413 <html><body><h1>413 Request Entity Too Large</h1></body></html>')).toBe(true)
  })

  it('recognizes a bare status line without a structured body', () => {
    expect(isGatewayBodyLimitError('413')).toBe(true)
    expect(isGatewayBodyLimitError('413 Request Entity Too Large')).toBe(true)
  })

  it('rejects a provider context-overflow report', () => {
    expect(isGatewayBodyLimitError(
      '400: {"code":"context_length_exceeded","message":"Context far exceeds the limit"}',
    )).toBe(false)
    expect(isGatewayBodyLimitError(
      '413: {"code":"context_length_exceeded","message":"context_length_exceeded"}',
    )).toBe(false)
  })

  it('rejects unrelated statuses and text', () => {
    expect(isGatewayBodyLimitError('401 unauthorized client detected')).toBe(false)
    expect(isGatewayBodyLimitError('429 rate limited')).toBe(false)
    expect(isGatewayBodyLimitError('4130 characters is a lot')).toBe(false)
  })
})

describe('classifySummarizationFailure', () => {
  it('treats a provider context overflow as over budget', () => {
    expect(classifySummarizationFailure(
      new LlmError('context_length_exceeded', CONTEXT_WINDOW_EXCEEDED_CODE),
    )).toBe('over-budget')
  })

  it('treats a pi-ai detected overflow as over budget', () => {
    expect(classifySummarizationFailure(
      new LlmError('pi-ai detected context overflow for model "m"', CONTEXT_WINDOW_EXCEEDED_CODE),
    )).toBe('over-budget')
  })

  it('treats a gateway body-limit rejection as over budget', () => {
    expect(classifySummarizationFailure(
      new LlmError('413 <html><body><h1>413 Request Entity Too Large</h1></body></html>', CONTEXT_WINDOW_EXCEEDED_CODE),
    )).toBe('over-budget')
  })

  it('treats route and content faults as terminal', () => {
    expect(classifySummarizationFailure(new LlmError('unauthorized client detected', 'AUTH'))).toBe('other')
    expect(classifySummarizationFailure(new LlmError('temporary provider outage', 'SERVER'))).toBe('other')
    expect(classifySummarizationFailure(new LlmError('账户已欠费', 'QUOTA'))).toBe('other')
    expect(classifySummarizationFailure(
      new LlmError('summary is not smaller than the shadowed content (1 >= 1)', 'PI_AI_ERROR'),
    )).toBe('other')
  })
})

/** Context without an LLM service, for measurement only. */
function meterContext(): Context {
  const ctx = new Context()
  void new LlmRuntime(ctx)
  new SessionProjectionRegistry(ctx)
  void new SessionStore(ctx)
  void new TokenMeter(ctx)
  return ctx
}

/** Six closed exchanges with no system head. */
function exchanges(count: number): Session {
  const session = Session.create(SessionId(`budget-${count}`))
  for (let turn = 1; turn <= count; turn += 1) {
    session.append('turn/start', { turn })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `question ${turn} ${'x'.repeat(400)}` }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('step/start', { turn, step: 1 })
    session.append('assistant/message', {
      turn,
      step: 1,
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: `answer ${turn} ${'y'.repeat(400)}` }],
        source: { kind: 'test' },
      },
    }, { surfaceOp: 'append' })
    session.append('step/end', { turn, step: 1 })
    session.append('turn/end', { turn, reason: { kind: 'completed' } })
  }
  return session
}

describe('shrinkSpanToBudget', () => {
  it('returns null when a budget cannot shrink the span at all', () => {
    const ctx = meterContext()
    const session = exchanges(4)
    const measurement = ctx.tokenMeter.measure(session)
    const range = selectCompactableRange(session, measurement, 0)
    expect(range).not.toBeNull()
    const nodes = session.surface.nodes
    const startIdx = nodes.indexOf(range!.start)
    const endIdx = nodes.indexOf(range!.end)
    expect(startIdx).toBe(0)
    expect(endIdx).toBeGreaterThan(0)

    // A budget of zero must not select anything: every node costs tokens.
    expect(shrinkSpanToBudget(session, measurement, {
      start: range!.start,
      end: range!.end,
      startIdx,
      endIdx,
      shadowedSeqs: nodes.slice(startIdx, endIdx + 1),
    }, 0)).toBeNull()
  })

  it('shrinks to the oldest prefix and keeps every cut balanced', () => {
    const ctx = meterContext()
    const session = exchanges(4)
    const measurement = ctx.tokenMeter.measure(session)
    const nodes = session.surface.nodes
    const startIdx = 0
    const endIdx = nodes.length - 2
    const selection = {
      start: nodes[0]!,
      end: nodes[endIdx]!,
      startIdx,
      endIdx,
      shadowedSeqs: nodes.slice(startIdx, endIdx + 1),
    }
    const priced = measurement.nodes
    const full = priced.slice(startIdx, endIdx + 1)
      .reduce((total, node) => total + node.tokens, 0)
    expect(full).toBeGreaterThan(0)

    const half = shrinkSpanToBudget(session, measurement, selection, full / 2)
    expect(half).not.toBeNull()
    expect(half!.startIdx).toBe(0)
    expect(half!.endIdx).toBeLessThan(endIdx)
    const shrunken = priced.slice(0, half!.endIdx + 1)
      .reduce((total, node) => total + node.tokens, 0)
    expect(shrunken).toBeLessThanOrEqual(full / 2)
    // Only whole exchanges survive, and fewer than the full span.
    expect(half!.shadowedSeqs.length).toBeLessThan(selection.shadowedSeqs.length)
    expect(half!.shadowedSeqs.length).toBeGreaterThan(0)
  })
})
