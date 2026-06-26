/**
 * vibe_start — start a VibeHarness web run as a DETACHED background process.
 *
 * `task` (the natural-language instructions) is REQUIRED — it is the critical
 * input handed to the harness. Models/codec/headless/maxSteps come from
 * `.opencode/vibeharness.json`, with optional per-run overrides here.
 *
 * Returns a SMALL string `{ runId, status, logPath, pid }`. Live progress is
 * streamed into this session out-of-band by the vibe-stream plugin; use
 * vibe_status / vibe_info to inspect the run.
 */
import { tool } from "@opencode-ai/plugin";
import { spawnRun, type RunOverrides } from "../lib/vibe";

export default tool({
  description:
    "Start a VibeHarness web run (job application / browse task) detached in " +
    "the background. 'task' is required. Returns a small {runId,status,logPath}.",
  args: {
    task: tool.schema
      .string()
      .describe(
        "REQUIRED. The natural-language instructions for the web agent, e.g. " +
          "'Go to example.com, read the heading, and report it.'",
      ),
    baseModel: tool.schema
      .string()
      .optional()
      .describe("Override the base model for this run only."),
    headless: tool.schema
      .boolean()
      .optional()
      .describe("Override headless mode for this run only."),
    maxSteps: tool.schema
      .number()
      .optional()
      .describe("Override max turns for this run only."),
    workdir: tool.schema
      .string()
      .optional()
      .describe(
        "Override the run workspace dir (absolute). Defaults to an isolated " +
          "dir under .opencode/vibe-runs/<runId>/workspace.",
      ),
  },
  async execute(args, ctx) {
    const task = (args.task || "").trim();
    if (!task) {
      return JSON.stringify({
        ok: false,
        error: "task is required and must be non-empty",
      });
    }
    const overrides: RunOverrides = {};
    if (args.baseModel) overrides.baseModel = args.baseModel;
    if (args.headless !== undefined) overrides.headless = args.headless;
    if (args.maxSteps !== undefined) overrides.maxSteps = args.maxSteps;
    if (args.workdir) overrides.workdir = args.workdir;

    try {
      const res = spawnRun(ctx.directory, task, overrides, ctx.sessionID ?? null);
      return JSON.stringify({
        ok: true,
        runId: res.runId,
        status: res.status,
        pid: res.pid,
        logPath: res.logPath,
        note: "detached; live progress streams into this session — use vibe_status",
      });
    } catch (e: any) {
      return JSON.stringify({ ok: false, error: String(e?.message ?? e) });
    }
  },
});
