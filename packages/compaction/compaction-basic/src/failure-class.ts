/**
 * Classify summarization failures into recoverable and terminal causes.
 *
 * @module @deepseek-ai/dsh-compaction-basic/failure-class
 */

import { CONTEXT_WINDOW_EXCEEDED_CODE, LlmError, isContextWindowExceededError } from '@deepseek-ai/dsh-llm'

/**
 * Why one summarization request failed, narrowed for compaction recovery.
 * `over-budget` means the replayed history itself does not fit the summarizer
 * route's request limit, so the failure is a payload-size problem rather than
 * a transient provider or gateway fault.
 */
export type SummarizationFailureClass = 'over-budget' | 'other'

/**
 * Whether one provider detail string is a gateway body-limit rejection rather
 * than a provider context-overflow report.
 *
 * The harness's own classifier maps any leading HTTP 413 to
 * `CONTEXT_WINDOW_EXCEEDED`, so an nginx "Request Entity Too Large" page
 * arriving through a proxy adapter is labelled a context overflow even though
 * nothing about the model's window was evaluated. Those failures are
 * request-size problems: compaction cannot shrink the payload further by
 * retrying the same span, so honouring the overflow label only burns a retry
 * before the same rejection.
 * @param detail - provider error code/type/message text joined into one string.
 * @returns true when the detail is a body-limit rejection carrying a non-provider body.
 */
export function isGatewayBodyLimitError(detail: string): boolean {
  if (!/(?:^|[^\w])413(?:[^\w]|$)/i.test(detail)) return false
  // A provider reports a structured context-overflow body; a proxy reports a
  // page or a bare status line.
  return /<(?:\/?)(?:html|head|title|body|center|hr)\b/i.test(detail)
    || !/context[\s_-]?(?:length|window)[\s_-]?exceed/i.test(detail)
}

/**
 * Classify one thrown summarization failure.
 *
 * Over-budget covers the two shapes that mean the replayed history itself
 * cannot fit the summarizer route: a provider-confirmed context overflow and a
 * gateway body-limit rejection of the multi-megabyte replay. Everything else —
 * auth, quota, transport, empty output, a summary not smaller than its source —
 * is a route or content fault that shrinking the span cannot repair.
 * @param error - the caught value.
 * @returns the narrowed failure class.
 */
export function classifySummarizationFailure(error: unknown): SummarizationFailureClass {
  if (error instanceof LlmError) {
    if (error.code === CONTEXT_WINDOW_EXCEEDED_CODE) return 'over-budget'
    const detail = `${error.code} ${error.failure.message}`
    if (isContextWindowExceededError(detail) || isGatewayBodyLimitError(detail)) return 'over-budget'
    return 'other'
  }
  const detail = error instanceof Error ? error.message : String(error)
  if (isContextWindowExceededError(detail) || isGatewayBodyLimitError(detail)) return 'over-budget'
  return 'other'
}
