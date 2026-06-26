/**
 * vibe_stop — kill a run's process tree (latest by default).
 *
 * Windows: `taskkill /F /T /PID <pid>` tears down the whole tree (the Python
 * harness plus its Playwright/browser children). POSIX falls back to
 * process.kill. Marks the run 'stopped'. Returns a tiny result.
 */
import { tool } from "@opencode-ai/plugin";
import { isAlive, readMeta, resolveRunId, writeMeta } from "../lib/vibe";

export default tool({
  description:
    "Stop a VibeHarness run (latest by default) by killing its process tree. " +
    "Small output.",
  args: {
    runId: tool.schema
      .string()
      .optional()
      .describe("Which run; defaults to the most recent run."),
  },
  async execute(args, ctx) {
    const runId = resolveRunId(ctx.directory, args.runId);
    if (!runId) return JSON.stringify({ ok: false, error: "no runs found" });
    const meta = readMeta(ctx.directory, runId);
    if (!meta) {
      return JSON.stringify({ ok: false, error: `unknown runId: ${runId}` });
    }
    if (!meta.pid) {
      return JSON.stringify({ ok: false, runId, error: "no pid recorded" });
    }
    const wasAlive = isAlive(meta.pid);
    let killed = false;
    let detail = "";
    try {
      if (process.platform === "win32") {
        // @ts-ignore - Bun runtime
        const r = Bun.spawnSync({
          cmd: ["taskkill", "/F", "/T", "/PID", String(meta.pid)],
          stdout: "pipe",
          stderr: "pipe",
        });
        detail = new TextDecoder()
          .decode(r.stdout?.length ? r.stdout : r.stderr)
          .trim();
        killed = r.exitCode === 0 || !isAlive(meta.pid);
      } else {
        try {
          process.kill(-meta.pid, "SIGKILL");
        } catch {
          process.kill(meta.pid, "SIGKILL");
        }
        killed = !isAlive(meta.pid);
      }
    } catch (e: any) {
      detail = String(e?.message ?? e);
    }

    writeMeta(ctx.directory, {
      ...meta,
      status: "stopped",
      finishedAt: meta.finishedAt || new Date().toISOString(),
    });

    return JSON.stringify({
      ok: true,
      runId,
      wasAlive,
      killed,
      status: "stopped",
      detail: detail.slice(0, 200) || undefined,
    });
  },
});
