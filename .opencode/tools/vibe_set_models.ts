/**
 * vibe_set_models — update the default models/codec for FUTURE harness runs.
 *
 * Writes `.opencode/vibeharness.json` (the single human-editable defaults file).
 * Only the fields you pass are changed; the rest keep their current values.
 * Returns a tiny confirmation of the new effective defaults.
 */
import { tool } from "@opencode-ai/plugin";
import { loadConfig, saveConfig } from "../lib/vibe";

export default tool({
  description:
    "Set default models/codec for future VibeHarness runs by writing " +
    ".opencode/vibeharness.json. Only provided fields change.",
  args: {
    baseModel: tool.schema
      .string()
      .optional()
      .describe("Ollama model for the base web agent, e.g. 'qwen3:4b'."),
    validatorModel: tool.schema
      .string()
      .optional()
      .describe("Validator model, e.g. 'glm-5.2'."),
    validatorProvider: tool.schema
      .string()
      .optional()
      .describe("Validator provider key, e.g. 'zhipuai' (falls back to local)."),
    escalationModel: tool.schema
      .string()
      .optional()
      .describe("Escalation model used when the local model gets wedged."),
    codec: tool.schema
      .string()
      .optional()
      .describe("Tool-call wire codec: hermes|json|xml|tagged_json|codeact|gbnf."),
    headless: tool.schema
      .boolean()
      .optional()
      .describe("Whether the browser runs headless by default."),
    maxSteps: tool.schema
      .number()
      .optional()
      .describe("Default max turns per run."),
  },
  async execute(args, ctx) {
    const patch: Record<string, unknown> = {};
    for (const k of [
      "baseModel",
      "validatorModel",
      "validatorProvider",
      "escalationModel",
      "codec",
      "headless",
      "maxSteps",
    ] as const) {
      if (args[k] !== undefined) patch[k] = args[k];
    }
    if (Object.keys(patch).length === 0) {
      const cur = loadConfig(ctx.directory);
      return JSON.stringify({
        ok: true,
        note: "no changes; current defaults",
        defaults: {
          baseModel: cur.baseModel,
          validatorModel: cur.validatorModel,
          validatorProvider: cur.validatorProvider,
          escalationModel: cur.escalationModel,
          codec: cur.codec,
          headless: cur.headless,
          maxSteps: cur.maxSteps,
        },
      });
    }
    const next = saveConfig(ctx.directory, patch as any);
    return JSON.stringify({
      ok: true,
      updated: Object.keys(patch),
      defaults: {
        baseModel: next.baseModel,
        validatorModel: next.validatorModel,
        validatorProvider: next.validatorProvider,
        escalationModel: next.escalationModel,
        codec: next.codec,
        headless: next.headless,
        maxSteps: next.maxSteps,
      },
    });
  },
});
