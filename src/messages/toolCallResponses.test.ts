import { AIMessage, ToolMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers, StepTypes, GraphEvents, ContentTypes } from '@/common';
import { formatAgentMessages } from './format';
import {
  isResponseIndex,
  MAX_RESPONSE_INDEX,
  tagToolCallResponses,
} from './toolCallResponses';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { createContentAggregator } from '@/stream';

type Call = { id: string; name: string; args: Record<string, string> };

const call = (id: string): Call => ({
  id,
  name: 'lookup',
  args: { q: id },
});

const aiWithCalls = (...ids: string[]): AIMessage =>
  new AIMessage({ content: '', tool_calls: ids.map(call) });

const toolResult = (id: string): ToolMessage =>
  new ToolMessage({ tool_call_id: id, name: 'lookup', content: `out-${id}` });

/** Run messages in the order a graph produces them. */
const buildRun = (rounds: string[][], finalText?: string): BaseMessage[] => {
  const messages: BaseMessage[] = [new HumanMessage('question')];
  for (const ids of rounds) {
    messages.push(aiWithCalls(...ids));
    messages.push(...ids.map(toolResult));
  }
  if (finalText != null) {
    messages.push(new AIMessage(finalText));
  }
  return messages;
};

/** Saves run messages the way the host does: through the content aggregator. */
const aggregate = (
  run: BaseMessage[],
  finalText?: string
): t.MessageContentComplex[] => {
  const { contentParts, aggregateContent } = createContentAggregator();
  let index = 0;
  run.forEach((message, i) => {
    if (
      !(message instanceof AIMessage) ||
      (message.tool_calls ?? []).length === 0
    ) {
      return;
    }
    const stepId = `step_${i}`;
    const toolCalls = message.tool_calls ?? [];
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: stepId,
        index,
        type: StepTypes.TOOL_CALLS,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: toolCalls.map((c) => ({ ...c, args: c.args })),
        },
      } as unknown as t.RunStep,
    });
    index += toolCalls.length;
    for (const c of toolCalls) {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: stepId,
            index,
            type: 'tool_call',
            tool_call: {
              id: c.id,
              name: c.name,
              args: JSON.stringify(c.args),
              output: `out-${c.id}`,
              progress: 1,
            },
          },
        } as unknown as t.ToolEndEvent as never,
      });
    }
  });
  const parts = contentParts.filter(
    (p): p is t.MessageContentComplex => p != null
  );
  if (finalText != null) {
    parts.push({ type: ContentTypes.TEXT, text: finalText });
  }
  return parts;
};

const persist = (parts: t.MessageContentComplex[]): t.MessageContentComplex[] =>
  JSON.parse(JSON.stringify(parts));

const rebuild = (
  parts: t.MessageContentComplex[],
  provider?: Providers
): BaseMessage[] =>
  formatAgentMessages(
    [
      { role: 'user', content: 'question' },
      { role: 'assistant', content: parts as never },
    ],
    undefined,
    undefined,
    undefined,
    provider != null ? { provider } : undefined
  ).messages;

const callIdsOf = (message: BaseMessage): string[] => {
  const ai = message as AIMessage;
  const fromCalls = (ai.tool_calls ?? []).map((c) => c.id ?? '');
  if (!Array.isArray(ai.content)) {
    return fromCalls;
  }
  const fromBlocks = (ai.content as Array<{ type: string; id?: string }>)
    .filter((b) => b.type === 'tool_use')
    .map((b) => b.id ?? '');
  return [...fromBlocks, ...fromCalls];
};

/** Role plus ordered call ids per message; the cache-relevant request shape. */
const shapeOf = (messages: BaseMessage[]): string[] =>
  messages.map((m) => {
    if (m instanceof ToolMessage) {
      return `tool:${m.tool_call_id}:${String(m.content)}`;
    }
    if (m instanceof AIMessage) {
      return `ai:${callIdsOf(m).join(',')}`;
    }
    return 'user';
  });

const toolPart = (
  id: string,
  response_index?: number
): t.MessageContentComplex =>
  ({
    type: ContentTypes.TOOL_CALL,
    tool_call: {
      id,
      name: 'lookup',
      args: JSON.stringify({ q: id }),
      output: `out-${id}`,
    },
    ...(response_index != null && { response_index }),
  }) as t.MessageContentComplex;

const responseIndexes = (
  parts: t.MessageContentComplex[]
): Array<number | undefined> =>
  parts.map((p) => (p as t.ToolCallContent).response_index);

describe('tagToolCallResponses', () => {
  it('tags each call with the index of its AI message, parallel calls together', () => {
    const run = buildRun([
      ['a', 'b'],
      ['c', 'd', 'e'],
    ]);
    const parts = [
      toolPart('a'),
      toolPart('b'),
      toolPart('c'),
      toolPart('d'),
      toolPart('e'),
    ];
    expect(tagToolCallResponses(parts, run)).toBe(5);
    expect(responseIndexes(parts)).toEqual([1, 1, 4, 4, 4]);
  });

  it('keeps the tag on the part, never inside tool_call, and does not add or reorder parts', () => {
    const run = buildRun([['a'], ['b']], 'done');
    const parts = [
      toolPart('a'),
      toolPart('b'),
      { type: ContentTypes.TEXT, text: 'done' } as t.MessageContentComplex,
    ];
    const before = parts.slice();
    tagToolCallResponses(parts, run);
    expect(parts).toHaveLength(3);
    expect(parts.every((p, i) => p === before[i])).toBe(true);
    for (const p of parts.slice(0, 2)) {
      expect(
        Object.keys((p as t.ToolCallContent).tool_call ?? {})
      ).not.toContain('response_index');
    }
    expect(responseIndexes(parts)).toEqual([1, 3, undefined]);
  });

  it('is idempotent', () => {
    const run = buildRun([['a', 'b'], ['c']]);
    const parts = [toolPart('a'), toolPart('b'), toolPart('c')];
    tagToolCallResponses(parts, run);
    const once = JSON.stringify(parts);
    tagToolCallResponses(parts, run);
    tagToolCallResponses(parts, run);
    expect(JSON.stringify(parts)).toBe(once);
  });

  it('skips tagging without run messages or content parts', () => {
    const parts = [toolPart('a')];
    expect(tagToolCallResponses(parts, undefined)).toBe(0);
    expect(tagToolCallResponses(undefined, buildRun([['a']]))).toBe(0);
    expect(tagToolCallResponses(parts, [])).toBe(0);
    expect(responseIndexes(parts)).toEqual([undefined]);
  });

  it('leaves calls missing from the run untagged and ignores other parts', () => {
    const run = buildRun([['a']]);
    const parts = [
      toolPart('x'),
      undefined,
      { type: ContentTypes.TEXT, text: 'hi' } as t.MessageContentComplex,
      { type: ContentTypes.TOOL_CALL } as t.MessageContentComplex,
      toolPart('a'),
    ];
    tagToolCallResponses(parts, run);
    expect(
      responseIndexes(
        parts.filter((p) => p != null) as t.MessageContentComplex[]
      )
    ).toEqual([undefined, undefined, undefined, 1]);
  });

  it('preserves earlier tags and numbers a restarted run above them', () => {
    const resumed = buildRun([['c'], ['d']]);
    const parts = [
      toolPart('a', 1),
      toolPart('b', 3),
      toolPart('c'),
      toolPart('d'),
    ];
    tagToolCallResponses(parts, resumed);
    expect(responseIndexes(parts)).toEqual([1, 3, 5, 7]);
    const snapshot = JSON.stringify(parts);
    tagToolCallResponses(parts, resumed);
    expect(JSON.stringify(parts)).toBe(snapshot);
  });

  it('never merges a restarted round into the earlier segment', () => {
    const resumed = buildRun([['c']]);
    const parts = [toolPart('a', 1), toolPart('c')];
    tagToolCallResponses(parts, resumed);
    const messages = rebuild(parts);
    expect(shapeOf(messages)).toEqual([
      'user',
      'ai:a',
      'tool:a:out-a',
      'ai:c',
      'tool:c:out-c',
    ]);
  });

  it('keeps compacted-away tags distinct from retained rounds', () => {
    const first = buildRun([['a'], ['b'], ['c']]);
    const parts = [toolPart('a'), toolPart('b'), toolPart('c')];
    tagToolCallResponses(parts, first);
    expect(responseIndexes(parts)).toEqual([1, 3, 5]);
    const compacted = [new HumanMessage('summary'), ...first.slice(5)];
    tagToolCallResponses(parts, compacted);
    const [a, b, c] = responseIndexes(parts);
    expect(a).toBe(1);
    expect(b).toBe(3);
    expect(c).toBeGreaterThan(3);
  });

  it('does not mutate run messages', () => {
    const run = buildRun([['a']]);
    const snapshot = JSON.stringify(run);
    tagToolCallResponses([toolPart('a')], run);
    expect(JSON.stringify(run)).toBe(snapshot);
  });
});

describe.each([[undefined], [Providers.ANTHROPIC]])(
  'tool rounds rebuild (provider: %s)',
  (provider) => {
    const text = (s: string, ids?: string[]): t.MessageContentComplex =>
      ({
        type: ContentTypes.TEXT,
        text: s,
        ...(ids != null && { tool_call_ids: ids }),
      }) as t.MessageContentComplex;

    it('splits responses and keeps parallel calls together', () => {
      const parts = [
        toolPart('a', 1),
        toolPart('b', 1),
        toolPart('c', 3),
        toolPart('d', 3),
        text('final'),
      ];
      const messages = rebuild(parts, provider);
      expect(shapeOf(messages)).toEqual([
        'user',
        'ai:a,b',
        'tool:a:out-a',
        'tool:b:out-b',
        'ai:c,d',
        'tool:c:out-c',
        'tool:d:out-d',
        'ai:',
      ]);
    });

    it('rebuilds untagged parts exactly as before', () => {
      const parts = [
        toolPart('a'),
        toolPart('b'),
        toolPart('c'),
        toolPart('d'),
      ];
      expect(shapeOf(rebuild(parts, provider))).toEqual([
        'user',
        'ai:a,b,c,d',
        'tool:a:out-a',
        'tool:b:out-b',
        'tool:c:out-c',
        'tool:d:out-d',
      ]);
    });

    it('keeps one assistant message for three parallel calls', () => {
      const parts = [toolPart('a', 2), toolPart('b', 2), toolPart('c', 2)];
      const shape = shapeOf(rebuild(parts, provider));
      expect(shape.filter((s) => s.startsWith('ai:'))).toEqual(['ai:a,b,c']);
    });

    it('mixes text-led responses, tagged and legacy calls', () => {
      const parts = [
        toolPart('a', 1),
        text('then', ['b']),
        toolPart('b', 3),
        toolPart('c', 3),
        toolPart('d', 5),
        toolPart('e'),
      ];
      const shape = shapeOf(rebuild(parts, provider));
      expect(shape.filter((s) => s.startsWith('ai:'))).toEqual([
        'ai:a',
        'ai:b,c',
        'ai:d,e',
      ]);
    });

    it('lets a text tool_call_ids boundary reset the remembered index', () => {
      const parts = [toolPart('a', 4), text('next', ['b']), toolPart('b', 4)];
      const shape = shapeOf(rebuild(parts, provider));
      expect(shape.filter((s) => s.startsWith('ai:'))).toEqual([
        'ai:a',
        'ai:b',
      ]);
    });

    it('does not leak response_index into the request', () => {
      const messages = rebuild([toolPart('a', 1), toolPart('b', 3)], provider);
      expect(JSON.stringify(messages)).not.toContain('response_index');
    });

    it('round trips aggregator -> tag -> persist -> format to the run request prefix', () => {
      const run = buildRun([['a', 'b'], ['c'], ['d', 'e', 'f']], 'final');
      const parts = aggregate(run, 'final');
      tagToolCallResponses(parts, run);
      const reloaded = persist(parts);
      expect(responseIndexes(reloaded).slice(0, 6)).toEqual([1, 1, 4, 6, 6, 6]);
      const rebuilt = rebuild(reloaded, provider);

      const expected = shapeOf(run.slice(0, -1));
      expect(shapeOf(rebuilt).slice(0, expected.length)).toEqual(expected);
      const runCallIds = run.flatMap((m) =>
        m instanceof AIMessage ? callIdsOf(m) : []
      );
      const rebuiltCallIds = rebuilt.flatMap((m) =>
        m instanceof AIMessage ? callIdsOf(m) : []
      );
      expect(rebuiltCallIds).toEqual(runCallIds);
    });

    it('without tagging, the aggregated answer merges rounds (regression baseline)', () => {
      const run = buildRun([['a'], ['b']]);
      const rebuilt = rebuild(persist(aggregate(run)), provider);
      expect(shapeOf(rebuilt).filter((s) => s.startsWith('ai:'))).toEqual([
        'ai:a,b',
      ]);
    });

    it('round trips a resume whose run indices restart', () => {
      const firstRun = buildRun([['a'], ['b']]);
      const parts = aggregate(firstRun);
      tagToolCallResponses(parts, firstRun);
      const resumedRun = buildRun([['c'], ['d', 'e']]);
      const resumedParts = persist(parts).concat(aggregate(resumedRun));
      tagToolCallResponses(resumedParts, resumedRun);
      const rebuilt = rebuild(persist(resumedParts), provider);
      expect(shapeOf(rebuilt).filter((s) => s.startsWith('ai:'))).toEqual([
        'ai:a',
        'ai:b',
        'ai:c',
        'ai:d,e',
      ]);
    });
  }
);

describe('content aggregator', () => {
  it('keeps response_index when an already tagged tool call is finalized again', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 's',
        index: 0,
        type: StepTypes.TOOL_CALLS,
        stepDetails: { type: StepTypes.TOOL_CALLS, tool_calls: [call('a')] },
      } as unknown as t.RunStep,
    });
    (contentParts[0] as t.ToolCallContent).response_index = 7;
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 's',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'a',
            name: 'lookup',
            args: '{}',
            output: 'o',
            progress: 1,
          },
        },
      } as unknown as t.ToolEndEvent as never,
    });
    const part = contentParts[0] as t.ToolCallContent;
    expect(part.response_index).toBe(7);
    expect(part.tool_call?.output).toBe('o');
  });
});

describe('tool rounds model-facing request prefix', () => {
  const tricky = {
    q: 'naïve 日本語 😀 "quoted" \\ back\nline\ttab </tag> {"k":[1,2]}',
    n: 1,
  };
  const trickyOutput = 'résultat: "ok" \\ \n 日本語 😀 {"a":"b"}';

  const richCall = (id: string) => ({
    id,
    name: 'lookup',
    args: { ...tricky, id },
  });
  const richResult = (id: string): ToolMessage =>
    new ToolMessage({
      tool_call_id: id,
      name: 'lookup',
      content: `${trickyOutput} ${id}`,
    });
  const richPart = (id: string): t.MessageContentComplex =>
    ({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id,
        name: 'lookup',
        args: JSON.stringify({ ...tricky, id }),
        output: `${trickyOutput} ${id}`,
      },
    }) as t.MessageContentComplex;
  const textPart = (s: string, ids?: string[]): t.MessageContentComplex =>
    ({
      type: ContentTypes.TEXT,
      text: s,
      ...(ids != null && { tool_call_ids: ids }),
    }) as t.MessageContentComplex;

  /** Rounds: parallel pair, text-led single call, then a lone call; then final text. */
  const run: BaseMessage[] = [
    new HumanMessage('question'),
    new AIMessage({ content: '', tool_calls: [richCall('a'), richCall('b')] }),
    richResult('a'),
    richResult('b'),
    new AIMessage({
      content: 'Let me check “this” 日本語',
      tool_calls: [richCall('c')],
    }),
    richResult('c'),
    new AIMessage({ content: '', tool_calls: [richCall('d')] }),
    richResult('d'),
  ];
  const savedParts = (): t.MessageContentComplex[] => [
    richPart('a'),
    richPart('b'),
    textPart('Let me check “this” 日本語', ['c']),
    richPart('c'),
    richPart('d'),
    textPart('final answer'),
  ];

  const serializers: Array<
    [string, (m: BaseMessage[]) => unknown, Providers?]
  > = [
    ['openai', (m) => _convertMessagesToOpenAIParams(m, 'gpt-4o'), undefined],
    [
      'anthropic',
      (m) => _convertMessagesToAnthropicPayload(m),
      Providers.ANTHROPIC,
    ],
  ];

  it.each(serializers)(
    'tagged rebuild serializes to the run prefix (%s)',
    (_name, serialize, provider) => {
      const parts = savedParts();
      expect(tagToolCallResponses(parts, run)).toBe(4);
      const rebuilt = rebuild(persist(parts), provider);
      expect(JSON.stringify(rebuilt)).not.toContain('response_index');
      expect(JSON.stringify(rebuilt[rebuilt.length - 1].content)).toContain(
        'final answer'
      );
      const prefix = rebuilt.slice(0, run.length);
      // The rebuilt user turn is shaped by the host, not by the tool rounds under test.
      expect(serialize(prefix)).toEqual(
        serialize([rebuilt[0], ...run.slice(1)])
      );
      expect(JSON.stringify(serialize(prefix))).not.toContain('response_index');
    }
  );

  it.each(serializers)(
    'untagged rebuild differs from the run prefix (%s)',
    (_name, serialize, provider) => {
      const rebuilt = rebuild(persist(savedParts()), provider);
      expect(serialize(rebuilt.slice(0, run.length))).not.toEqual(
        serialize([rebuilt[0], ...run.slice(1)])
      );
    }
  );
});

describe('response_index hardening', () => {
  const part = (
    id: string,
    response_index?: unknown
  ): t.MessageContentComplex =>
    ({
      type: ContentTypes.TOOL_CALL,
      tool_call: { id, name: 'lookup', args: '{}', output: id },
      ...(response_index !== undefined && { response_index }),
    }) as t.MessageContentComplex;

  it.each([
    ['unsafe integer', Number.MAX_SAFE_INTEGER + 2],
    ['above the cap', MAX_RESPONSE_INDEX + 1],
    ['negative', -1],
    ['fractional', 1.5],
    ['string', '3'],
    ['NaN', Number.NaN],
  ])('rejects a %s tag', (_label, value) => {
    expect(isResponseIndex(value)).toBe(false);
  });

  it('accepts the largest allowed tag', () => {
    expect(isResponseIndex(MAX_RESPONSE_INDEX)).toBe(true);
    expect(isResponseIndex(0)).toBe(true);
  });

  it('ignores a malformed retained tag instead of offsetting by it', () => {
    const run = buildRun([['b'], ['c']]);
    const parts = [
      part('x', Number.MAX_SAFE_INTEGER + 2),
      part('b'),
      part('c'),
    ];
    tagToolCallResponses(parts, run);
    expect(responseIndexes(parts)).toEqual([Number.MAX_SAFE_INTEGER + 2, 1, 3]);
  });

  it('leaves parts untouched when a retained tag leaves no room for the run', () => {
    const run = buildRun([['b'], ['c']]);
    const parts = [part('x', MAX_RESPONSE_INDEX), part('b'), part('c')];
    expect(tagToolCallResponses(parts, run)).toBe(0);
    expect(responseIndexes(parts)).toEqual([
      MAX_RESPONSE_INDEX,
      undefined,
      undefined,
    ]);
  });

  it('never produces colliding tags for distinct rounds', () => {
    const run = buildRun([['b'], ['c']]);
    const parts = [part('x', 2 ** 30), part('b'), part('c')];
    tagToolCallResponses(parts, run);
    const [, b, c] = responseIndexes(parts);
    expect(b).not.toBe(c);
    expect(b).toBeGreaterThan(2 ** 30);
  });

  it('does not read run messages when there are no tool_call parts', () => {
    const runMessages = new Proxy([] as BaseMessage[], {
      get: () => {
        throw new Error('run messages must not be read');
      },
    });
    const parts = [
      { type: ContentTypes.TEXT, text: 'hi' },
    ] as t.MessageContentComplex[];
    expect(tagToolCallResponses(parts, runMessages)).toBe(0);
  });
});
