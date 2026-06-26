/**
 * vibe_restart — re-spawn a run from its saved task + per-run overrides as a
 * NEW run (new runId). Defaults to the latest run. Current models/codec from
 * vibeharness.json are applied fresh. Returns a tiny {runId,status,logPath}.
 */
import { tool } from "@opencode-ai/plugin";
import { readMeta, resolveRunId, spawnRun } from "../lib/vibe";

export default tool({
  description:
    "Restart a VibeHarness run (latest by default) from its saved task as a " +
    "new run. Small output.",
  args: {
    runId: tool.schema
      .string()
      .optional()
      .describe("Which run to clone; defaults to the most recent run."),
  },
  async execute(args, ctx) {
    const runId = resolveRunId(ctx.directory, args.runId);
    if (!runId) return JSON.stringify({ ok: false, error: "no runs found" });
    const meta = readMeta(ctx.directory, runId);
    if (!meta) {
      return JSON.stringify({ ok: false, error: `unknown runId: ${runId}` });
    }
    try {
      // Re-spawn with the original task + per-run overrides, EXCEPT the workdir
      // (a fresh isolated workspace avoids clobbering the old run's .vibe log).
      const overrides = { ...meta.overrides };
      delete (overrides as any).workdir;
      const res = spawnRun(
        ctx.directory,
        meta.task,
        overrides,
        ctx.sessionID ?? meta.sessionID ?? null,
      );
      return JSON.stringify({
        ok: true,
        runId: res.runId,
        clonedFrom: runId,
        status: res.status,
        pid: res.pid,
        logPath: res.logPath,
      });
    } catch (e: any) {
      return JSON.stringify({ ok: false, error: String(e?.message ?? e) });
    }
  },
});
