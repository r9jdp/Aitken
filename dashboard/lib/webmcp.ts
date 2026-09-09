import { validDate } from './research';
import type { Catalog } from './research';

type Tool = {
  name: string;
  title: string;
  description: string;
  inputSchema: object;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (input: unknown) => unknown;
};
export type ModelContext = {
  registerTool: (
    tool: Tool,
    options?: { signal: AbortSignal },
  ) => void | Promise<void>;
};

export function validateMapRequest(
  input: unknown,
  catalog: Catalog,
): { model: string; date: string } {
  if (!input || typeof input !== 'object' || Array.isArray(input))
    throw new Error('Expected a model and a date.');
  const args = input as Record<string, unknown>;
  if (Object.keys(args).some((k) => !['model', 'date'].includes(k)))
    throw new Error('Unknown input field.');
  if (
    typeof args.model !== 'string' ||
    !catalog.models.some((m) => m.id === args.model && m.hasMaps)
  )
    throw new Error('Choose a model with saved full-grid maps.');
  if (
    typeof args.date !== 'string' ||
    !validDate(args.date, catalog.start, catalog.end)
  )
    throw new Error('Choose a valid date in 2024.');
  return { model: args.model, date: args.date };
}

export function registerDashboardTools(
  context: ModelContext | undefined,
  catalog: Catalog,
  generate: (model: string, date: string) => Promise<unknown>,
) {
  if (!context?.registerTool) return () => {};
  const lifecycle = new AbortController();
  const tools: Tool[] = [
    {
      name: 'read_model_benchmark',
      title: 'Read verified model scores',
      description:
        'Read all seven saved model scores and the shared retrospective 2024 evaluation scope. Does not change the dashboard.',
      inputSchema: {
        type: 'object',
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute(input) {
        if (
          !input ||
          typeof input !== 'object' ||
          Array.isArray(input) ||
          Object.keys(input).length
        )
          throw new Error('This tool takes an empty object.');
        return {
          training: '2021–2023',
          evaluation: '2024 retrospective',
          stationDays: catalog.stationDays,
          models: catalog.models.map((m) => ({
            id: m.id,
            label: m.label,
            mae: m.metrics.mae,
            rmse: m.metrics.rmse,
            r2: m.metrics.r2,
            hasMaps: m.hasMaps,
          })),
        };
      },
    },
    {
      name: 'generate_archived_heatmap',
      title: 'Display an archived heatmap',
      description:
        'Select a saved model and 2024 date, load that prediction grid, and update the visible heatmap explorer. This does not train or forecast new data.',
      inputSchema: {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            enum: catalog.models.filter((m) => m.hasMaps).map((m) => m.id),
          },
          date: {
            type: 'string',
            format: 'date',
            description: '2024-01-01 through 2024-12-31',
          },
        },
        required: ['model', 'date'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input) {
        const args = validateMapRequest(input, catalog);
        return generate(args.model, args.date);
      },
    },
  ];
  for (const tool of tools) {
    try {
      void Promise.resolve(
        context.registerTool(tool, { signal: lifecycle.signal }),
      ).catch(() => {
        /* Optional browser capability; the normal UI remains usable. */
      });
    } catch {
      /* Unsupported registry versions must not break the dashboard. */
    }
  }
  return () => lifecycle.abort();
}
