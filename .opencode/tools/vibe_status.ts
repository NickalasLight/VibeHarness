/**
 * vibe_status — compact status of a run (latest by default).
 *
 * Reads per-run state + the harness `.vibe/` runlog and process liveness, and
 * returns a SMALL summary: running/finished/failed/stopped, turn count, last
 * action, validator verdict (if done), elapsed seconds. Never a transcript.
 */
import { tool } from "@opencode-ai/plugin";
import { computeStatus, readMeta, resolveRunId } from "../lib/vibe";

export default tool({
  description:
    "Compact status of a VibeHarness run (latest by default): " +
    "status, turns, last action, validator verdict, elapsed. Small output.",
  args: {
    runId: tool.schema
      .string()
      .optional()
      .describe("Which run; defaults to the most recent run."),
  },
  async execute(args, ctx) {
    const runId = resolveRunId(ctx.directory, args.runId);
    if (!runId) {
      return JSON.stringify({ ok: false, error: "no runs found" });
    }
    const meta = readMeta(ctx.directory, runId);
    if (!meta) {
      return JSON.stringify({ ok: false, error: `unknown runId: ${runId}` });
    }
    const s = computeStatus(ctx.directory, meta);
    return JSON.stringify({
      ok: true,
      runId: s.runId,
      status: s.status,
      turns: s.turns,
      lastAction: s.lastAction,
      verdict: s.verdict,
      finished: s.finished,
      elapsedSec: s.elapsedSec,
      task: s.task.length > 120 ? s.task.slice(0, 120) + " …" : s.task,
    });
  },
});
