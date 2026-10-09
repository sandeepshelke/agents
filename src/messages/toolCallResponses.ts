import { isAIMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes } from '@/common';

export const isResponseIndex = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0;

/**
 * Records, on each `tool_call` content part, which model response issued it, so
 * `formatAgentMessages` can rebuild one assistant message per response.
 *
 * The tag is `response_index`: the index of the issuing AI message in
 * `runMessages`, matched by tool call id. It is set on the part (never inside
 * `tool_call`) and no part is added, removed or reordered.
 *
 * `runMessages` indices restart when a run is rebuilt (resume) or its history
 * compacted, so parts whose call is absent from `runMessages` keep their
 * earlier tag, and calls found in `runMessages` are numbered above every such
 * tag. A new round therefore can never share a tag with an earlier segment.
 * Idempotent for a fixed `runMessages`.
 *
 * @returns the number of parts tagged
 */
export function tagToolCallResponses(
  contentParts: Array<t.MessageContentComplex | undefined> | undefined,
  runMessages: BaseMessage[] | undefined
): number {
  if (contentParts == null || runMessages == null) {
    return 0;
  }

  const responseByCallId = new Map<string, number>();
  for (let i = 0; i < runMessages.length; i++) {
    const message = runMessages[i] as BaseMessage | null | undefined;
    if (message == null || !isAIMessage(message)) {
      continue;
    }
    const toolCalls = message.tool_calls;
    if (toolCalls == null) {
      continue;
    }
    for (const toolCall of toolCalls) {
      const id = toolCall.id;
      if (id != null && id !== '' && !responseByCallId.has(id)) {
        responseByCallId.set(id, i);
      }
    }
  }
  if (responseByCallId.size === 0) {
    return 0;
  }

  const matched: Array<[t.ToolCallContent, number]> = [];
  let highestPreserved = -1;
  for (const part of contentParts) {
    if (part?.type !== ContentTypes.TOOL_CALL) {
      continue;
    }
    const toolCallPart = part as t.ToolCallContent;
    const id = toolCallPart.tool_call?.id;
    const responseIndex =
      id != null && id !== '' ? responseByCallId.get(id) : undefined;
    if (responseIndex != null) {
      matched.push([toolCallPart, responseIndex]);
      continue;
    }
    if (isResponseIndex(toolCallPart.response_index)) {
      highestPreserved = Math.max(
        highestPreserved,
        toolCallPart.response_index
      );
    }
  }

  const offset = highestPreserved + 1;
  for (const [part, responseIndex] of matched) {
    part.response_index = offset + responseIndex;
  }
  return matched.length;
}
