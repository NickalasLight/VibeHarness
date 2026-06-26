/**
 * vibe_info — paths + optional log tail for a run (latest by default).
 *
 * `which` selects what to surface WITHOUT dumping the whole transcript:
 *   - summary     : final summary + validator verdict + key paths (default)
 *   - runlog      : paths to the .vibe runlog json/md + a short tail of the md
 *   - screenshots : list of screenshot/snapshot files in the workspace/.vibe
 *   - workspace   : workspace path + top-level file listing
 *   - tail        : last lines of the harness stdout log
 *
 * The FULL transcript is reachable only through the paths returned here.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { tool } from "@opencode-ai/plugin";
import {
  computeStatus,
  findRunlogJson,
  preview,
  readMeta,
  resolveRunId,
} from "../lib/vibe";

function tailFile(path: string, lines: number): string {
  if (!existsSync(path)) return "";
  const text = readFileSync(path, "utf8");
  const arr = text.split(/\r?\n/);
  return arr.slice(Math.max(0, arr.length - lines)).join("\n");
}

function listFiles(dir: string, max = 40): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const n of readdirSync(dir)) {
    try {
      const p = join(dir, n);
      out.push(statSync(p).isDirectory() ? `${n}/` : n);
    } catch {
      out.push(n);
    }
    if (out.length >= max) break;
  }
  return out;
}

export default tool({
  description:
    "Paths + optional log tail for a VibeHarness run (latest by default). " +
    "which=summary|runlog|screenshots|workspace|tail. Small output; the full " +
    "transcript is only reachable via the returned paths.",
  args: {
    runId: tool.schema
      .string()
      .optional()
      .describe("Which run; defaults to the most recent run."),
    which: tool.schema
      .enum(["summary", "runlog", "screenshots", "workspace", "tail"])
      .optional()
      .describe("What to surface. Default 'summary'."),
    lines: tool.schema
      .number()
      .optional()
      .describe("For which=tail/runlog: how many trailing lines (default 25)."),
  },
  async execute(args, ctx) {
    const runId = resolveRunId(ctx.directory, args.runId);
    if (!runId) return JSON.stringify({ ok: false, error: "no runs found" });
    const meta = readMeta(ctx.directory, runId);
    if (!meta) {
      return JSON.stringify({ ok: false, error: `unknown runId: ${runId}` });
    }
    const which = args.which ?? "summary";
    const lines = args.lines ?? 25;
    const s = computeStatus(ctx.directory, meta);
    const runlogJson = findRunlogJson(meta.vibeDir);
    const runlogMd = runlogJson ? runlogJson.replace(/\.json$/, ".md") : null;
    const base = {
      ok: true as const,
      runId,
      which,
      status: s.status,
    };

    if (which === "tail") {
      return JSON.stringify({
        ...base,
        logPath: meta.logPath,
        tail: tailFile(meta.logPath, lines),
      });
    }
    if (which === "runlog") {
      return JSON.stringify({
        ...base,
        runlogJson,
        runlogMd,
        tail: runlogMd ? tailFile(runlogMd, lines) : "",
      });
    }
    if (which === "screenshots") {
      const ws = listFiles(meta.workdir).filter((f) =>
        /\.(png|jpg|jpeg)$/i.test(f),
      );
      const diag = join(meta.vibeDir, `${runlogPathStamp(runlogJson)}-diagnostics`);
      return JSON.stringify({
        ...base,
        workspace: meta.workdir,
        screenshots: ws,
        diagnosticsDir: existsSync(diag) ? diag : meta.vibeDir,
        snapshotCount: existsSync(diag)
          ? readdirSync(diag).filter((f) => /snapshot/.test(f)).length
          : 0,
      });
    }
    if (which === "workspace") {
      return JSON.stringify({
        ...base,
        workspace: meta.workdir,
        files: listFiles(meta.workdir),
      });
    }
    // summary (default)
    return JSON.stringify({
      ...base,
      task: preview(meta.task, 160),
      finished: s.finished,
      verdict: s.verdict,
      finalSummary: s.finalSummary ? preview(s.finalSummary, 240) : null,
      elapsedSec: s.elapsedSec,
      turns: s.turns,
      paths: {
        runlogJson,
        runlogMd,
        log: meta.logPath,
        workspace: meta.workdir,
        vibeDir: meta.vibeDir,
      },
    });
  },
});

function runlogPathStamp(runlogJson: string | null): string {
  if (!runlogJson) return "";
  const m = runlogJson.match(/(\d{8}_\d{6})\.json$/);
  return m ? m[1] : "";
}
