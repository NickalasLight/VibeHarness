/**
 * Shared control-plane logic for the VibeHarness OpenCode integration.
 *
 * This module is imported by the six `.opencode/tools/*.ts` tools and by the
 * companion `.opencode/plugins/vibe-stream.ts` plugin. It is intentionally the
 * SINGLE place that knows how to:
 *   - load/save the editable defaults file `.opencode/vibeharness.json`
 *   - read/write per-run state under `.opencode/vibe-runs/<runId>/meta.json`
 *   - resolve a `runId` (latest by default)
 *   - assemble the harness CLI arguments from config + per-run overrides
 *   - spawn the INTACT Python harness as a detached background process
 *   - locate + parse the harness `.vibe/<stamp>.json` runlog for compact status
 *
 * The Python harness itself (agent loop, snapshot system, codec, validator,
 * model routing) is never touched — we only drive it via its existing CLI:
 *   python -m vibeharness "<task>" --agent web --model <m> --codec hermes \
 *          --max-steps N [--headless] --workdir <ws> --set <key> <val> ...
 *
 * Runtime: this runs inside OpenCode's bundled Bun runtime, so the `Bun`
 * global and Node's `fs`/`path`/`os` modules are available.
 */
import {
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VibeConfig {
  baseModel: string;
  validatorModel: string;
  validatorProvider: string;
  escalationModel: string;
  escalationProvider?: string;
  codec: string;
  headless: boolean;
  maxSteps: number;
  /** Command used to launch Python (default "python"). */
  python?: string;
  /** Harness agent type (default "web"). */
  agent?: string;
}

export interface RunOverrides {
  baseModel?: string;
  headless?: boolean;
  maxSteps?: number;
  workdir?: string;
}

export type RunStatus =
  | "running"
  | "finished"
  | "failed"
  | "stopped";

export interface RunMeta {
  runId: string;
  pid: number | null;
  task: string;
  /** Fully-resolved config actually used for this run. */
  config: VibeConfig;
  overrides: RunOverrides;
  sessionID: string | null;
  status: RunStatus;
  startedAt: string;
  finishedAt?: string | null;
  exitCode?: number | null;
  /** Absolute path the harness ran in (`--workdir`); its `.vibe/` lives here. */
  workdir: string;
  /** Absolute path to the captured stdout/stderr of the harness process. */
  logPath: string;
  /** Absolute path to the harness `.vibe/` runlog directory. */
  vibeDir: string;
  /** The exact argv handed to the spawn (for diagnostics / restart). */
  argv: string[];
}

// ---------------------------------------------------------------------------
// Defaults + config file
// ---------------------------------------------------------------------------

export const DEFAULT_CONFIG: VibeConfig = {
  baseModel: "qwen3:4b",
  validatorModel: "glm-5.2",
  validatorProvider: "zhipuai",
  escalationModel: "glm-5.2",
  escalationProvider: "zhipuai",
  codec: "hermes",
  headless: false,
  maxSteps: 15,
  python: "python",
  agent: "web",
};

export function opencodeDir(directory: string): string {
  return join(directory, ".opencode");
}

export function configPath(directory: string): string {
  return join(opencodeDir(directory), "vibeharness.json");
}

export function runsDir(directory: string): string {
  return join(opencodeDir(directory), "vibe-runs");
}

/** Load `.opencode/vibeharness.json`, merged over built-in defaults. */
export function loadConfig(directory: string): VibeConfig {
  const p = configPath(directory);
  let onDisk: Partial<VibeConfig> = {};
  if (existsSync(p)) {
    try {
      onDisk = JSON.parse(readFileSync(p, "utf8")) as Partial<VibeConfig>;
    } catch {
      onDisk = {};
    }
  }
  return { ...DEFAULT_CONFIG, ...onDisk };
}

/** Persist a partial config update to `.opencode/vibeharness.json`. */
export function saveConfig(
  directory: string,
  patch: Partial<VibeConfig>,
): VibeConfig {
  const p = configPath(directory);
  // Only persist the human-editable, spec'd keys (keep the file clean).
  const current = loadConfig(directory);
  const next: VibeConfig = { ...current, ...patch };
  const persisted = {
    baseModel: next.baseModel,
    validatorModel: next.validatorModel,
    validatorProvider: next.validatorProvider,
    escalationModel: next.escalationModel,
    codec: next.codec,
    headless: next.headless,
    maxSteps: next.maxSteps,
  };
  mkdirSync(opencodeDir(directory), { recursive: true });
  writeFileSync(p, JSON.stringify(persisted, null, 2) + "\n", "utf8");
  return next;
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

function pad(n: number, w = 2): string {
  return String(n).padStart(w, "0");
}

export function newRunId(d = new Date()): string {
  const stamp =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}_` +
    `${pad(d.getMilliseconds(), 3)}`;
  const rand = Math.random().toString(36).slice(2, 6);
  return `run_${stamp}_${rand}`;
}

export function runDir(directory: string, runId: string): string {
  return join(runsDir(directory), runId);
}

export function metaPath(directory: string, runId: string): string {
  return join(runDir(directory, runId), "meta.json");
}

export function writeMeta(directory: string, meta: RunMeta): void {
  mkdirSync(runDir(directory, meta.runId), { recursive: true });
  writeFileSync(
    metaPath(directory, meta.runId),
    JSON.stringify(meta, null, 2) + "\n",
    "utf8",
  );
}

export function readMeta(directory: string, runId: string): RunMeta | null {
  const p = metaPath(directory, runId);
  if (!existsSync(p)) return null;
  try {
    return JSON.parse(readFileSync(p, "utf8")) as RunMeta;
  } catch {
    return null;
  }
}

/** All run metas, newest first (by startedAt). */
export function listRuns(directory: string): RunMeta[] {
  const dir = runsDir(directory);
  if (!existsSync(dir)) return [];
  const metas: RunMeta[] = [];
  for (const name of readdirSync(dir)) {
    const m = readMeta(directory, name);
    if (m) metas.push(m);
  }
  metas.sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
  return metas;
}

export function latestRunId(directory: string): string | null {
  const runs = listRuns(directory);
  return runs.length ? runs[0].runId : null;
}

/** Resolve an optional runId to a concrete one (latest when omitted). */
export function resolveRunId(
  directory: string,
  runId?: string | null,
): string | null {
  if (runId && runId.trim()) return runId.trim();
  return latestRunId(directory);
}

// ---------------------------------------------------------------------------
// CLI assembly
// ---------------------------------------------------------------------------

/**
 * Map config + per-run overrides to the harness CLI argv (excluding the
 * leading `-m vibeharness`). The task is the positional argument.
 *
 * NOTE on validator/escalation routing: the harness `--set` flag only accepts a
 * small whitelist (temp/model/codec/max-steps/...), NOT `validation_*` /
 * `escalation_*`. Those Config fields are instead supplied via a per-run
 * settings.json under a dedicated `VIBEHARNESS_HOME` (see `runSettings` +
 * `spawnRun`), which `Settings.apply` folds in beneath the CLI flags.
 */
export function buildArgv(
  cfg: VibeConfig,
  task: string,
  overrides: RunOverrides,
  workdir: string,
): string[] {
  const model = overrides.baseModel || cfg.baseModel;
  const headless =
    overrides.headless !== undefined ? overrides.headless : cfg.headless;
  const maxSteps =
    overrides.maxSteps !== undefined ? overrides.maxSteps : cfg.maxSteps;

  const argv: string[] = [
    "-m",
    "vibeharness",
    task,
    "--agent",
    cfg.agent || "web",
    "--model",
    model,
    "--codec",
    cfg.codec,
    "--max-steps",
    String(maxSteps),
    "--workdir",
    workdir,
    "--no-color",
  ];
  if (headless) argv.push("--headless");
  return argv;
}

/**
 * The per-run settings.json (placed under a private `VIBEHARNESS_HOME`) that
 * carries the validator/escalation routing the `--set` whitelist won't take.
 * `Settings.apply` matches these to `Config` field names beneath the CLI flags.
 */
export function runSettings(cfg: VibeConfig): Record<string, unknown> {
  return {
    validation_provider: cfg.validatorProvider,
    validation_model: cfg.validatorModel,
    escalation_provider: cfg.escalationProvider || cfg.validatorProvider,
    escalation_model: cfg.escalationModel,
  };
}

// ---------------------------------------------------------------------------
// Spawning the detached harness process
// ---------------------------------------------------------------------------

export interface SpawnResult {
  runId: string;
  pid: number | null;
  status: RunStatus;
  logPath: string;
}

/**
 * Spawn the harness detached, capturing stdout+stderr to the run's log file,
 * and persist a fresh meta.json. Returns a SMALL result for the tool to relay.
 */
export function spawnRun(
  directory: string,
  task: string,
  overrides: RunOverrides,
  sessionID: string | null,
  reuseRunId?: string,
): SpawnResult {
  const cfg = loadConfig(directory);
  const runId = reuseRunId || newRunId();
  const rdir = runDir(directory, runId);
  mkdirSync(rdir, { recursive: true });

  // Each run gets its own isolated workspace so its `.vibe/` runlog is easy to
  // find and runs never clobber each other. An explicit override wins.
  const workdir = overrides.workdir
    ? overrides.workdir
    : join(rdir, "workspace");
  mkdirSync(workdir, { recursive: true });

  const logPath = join(rdir, "stdout.log");
  const argv = buildArgv(cfg, task, overrides, workdir);
  const python = cfg.python || "python";

  // Validator/escalation routing goes through a per-run, isolated
  // VIBEHARNESS_HOME so we never clobber the user's global ~/.vibeharness
  // settings. `Settings.apply` reads `<VIBEHARNESS_HOME>/settings.json`.
  const vibeHome = join(rdir, "vibehome");
  mkdirSync(vibeHome, { recursive: true });
  writeFileSync(
    join(vibeHome, "settings.json"),
    JSON.stringify(runSettings(cfg), null, 2),
    "utf8",
  );

  // Redirect stdout+stderr to the log file via a real fd (Bun.spawn accepts a
  // numeric fd). The harness rewrites its `.vibe/` runlog every turn, so the
  // streaming plugin can read structured progress without parsing this log.
  const fd = openSync(logPath, "a");

  // The process MUST outlive this tool call (detached background run). On
  // Windows there is no POSIX `detached`; we spawn via Bun and `unref()` so the
  // child is not tied to the tool's event loop. `taskkill /F /T /PID` (see
  // vibe_stop) later tears down the whole process tree.
  // @ts-ignore - Bun is provided by the OpenCode runtime.
  const proc = Bun.spawn({
    cmd: [python, ...argv],
    cwd: directory,
    stdin: "ignore",
    stdout: fd,
    stderr: fd,
    env: { ...process.env, VIBEHARNESS_HOME: vibeHome },
  });
  try {
    proc.unref();
  } catch {
    /* ignore */
  }

  const meta: RunMeta = {
    runId,
    pid: proc.pid ?? null,
    task,
    config: cfg,
    overrides,
    sessionID,
    status: "running",
    startedAt: new Date().toISOString(),
    finishedAt: null,
    exitCode: null,
    workdir,
    logPath,
    vibeDir: join(workdir, ".vibe"),
    argv: [python, ...argv],
  };
  writeMeta(directory, meta);

  return { runId, pid: meta.pid, status: "running", logPath };
}

// ---------------------------------------------------------------------------
// Process liveness + runlog parsing
// ---------------------------------------------------------------------------

export function isAlive(pid: number | null | undefined): boolean {
  if (!pid) return false;
  try {
    // Signal 0 only checks for existence; works on Windows under Bun/Node.
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    // EPERM means the process exists but we can't signal it -> still alive.
    return e && e.code === "EPERM";
  }
}

const RUNLOG_RE = /^\d{8}_\d{6}\.json$/;

/** Newest harness runlog JSON in a `.vibe/` dir, or null. */
export function findRunlogJson(vibeDir: string): string | null {
  if (!existsSync(vibeDir)) return null;
  const cands = readdirSync(vibeDir).filter((n) => RUNLOG_RE.test(n));
  if (!cands.length) return null;
  cands.sort();
  return join(vibeDir, cands[cands.length - 1]);
}

export interface RunlogAction {
  tool: string | null;
  ok: boolean;
  observation: string;
}
export interface RunlogTurn {
  index: number;
  actions: RunlogAction[];
}
export interface Runlog {
  finished: boolean;
  final_summary: string;
  validations: Array<{ turn: number; passed: boolean; reason: string }>;
  turns: RunlogTurn[];
}

export function parseRunlog(vibeDir: string): Runlog | null {
  const p = findRunlogJson(vibeDir);
  if (!p) return null;
  try {
    const raw = JSON.parse(readFileSync(p, "utf8"));
    const turns: RunlogTurn[] = (raw.turns || []).map((t: any) => ({
      index: t.index,
      actions: (t.actions || []).map((a: any) => ({
        tool: a.tool ?? null,
        ok: !!a.ok,
        observation: String(a.observation ?? ""),
      })),
    }));
    return {
      finished: !!raw.finished,
      final_summary: String(raw.final_summary ?? ""),
      validations: (raw.validations || []).map((v: any) => ({
        turn: v.turn,
        passed: !!v.passed,
        reason: String(v.reason ?? ""),
      })),
      turns,
    };
  } catch {
    return null;
  }
}

/** One-line preview, whitespace-collapsed and length-capped. */
export function preview(s: string, max = 160): string {
  const flat = (s || "").replace(/\s+/g, " ").trim();
  return flat.length > max ? flat.slice(0, max) + " …" : flat;
}

export interface ComputedStatus {
  runId: string;
  status: RunStatus;
  alive: boolean;
  turns: number;
  lastAction: string | null;
  lastPage: string | null;
  finished: boolean;
  verdict: string | null;
  finalSummary: string | null;
  elapsedSec: number;
  task: string;
  logPath: string;
  workdir: string;
}

/**
 * Compute compact status from meta + runlog + process liveness, and persist any
 * terminal-state transition back into meta.json (running -> finished/failed).
 */
export function computeStatus(
  directory: string,
  meta: RunMeta,
): ComputedStatus {
  const alive = isAlive(meta.pid);
  const log = parseRunlog(meta.vibeDir);
  const turns = log?.turns.length ?? 0;

  let lastAction: string | null = null;
  let lastPage: string | null = null;
  if (log && log.turns.length) {
    const lastTurn = log.turns[log.turns.length - 1];
    const acts = lastTurn.actions;
    if (acts.length) {
      const a = acts[acts.length - 1];
      lastAction = `${a.tool ?? "?"} -> ${preview(a.observation, 100)}`;
    }
    // The web snapshot is not in the runlog turn shape; surface the last goto/url
    // tool arg when present is out of scope — keep page null unless observable.
  }

  let verdict: string | null = null;
  if (log && log.validations.length) {
    const v = log.validations[log.validations.length - 1];
    verdict = `${v.passed ? "PASS" : "FAIL"}: ${preview(v.reason, 120)}`;
  }

  let status: RunStatus = meta.status;
  if (meta.status === "stopped") {
    status = "stopped";
  } else if (log?.finished) {
    status = "finished";
  } else if (!alive) {
    status = "failed";
  } else {
    status = "running";
  }

  // Persist terminal transitions so later reads are stable + cheap.
  if (status !== meta.status && status !== "running") {
    const updated: RunMeta = {
      ...meta,
      status,
      finishedAt: meta.finishedAt || new Date().toISOString(),
    };
    try {
      writeMeta(directory, updated);
    } catch {
      /* ignore */
    }
  }

  const started = Date.parse(meta.startedAt);
  const end =
    status === "running" ? Date.now() : Date.parse(meta.finishedAt || "") || Date.now();
  const elapsedSec = Math.max(0, Math.round((end - started) / 1000));

  return {
    runId: meta.runId,
    status,
    alive,
    turns,
    lastAction,
    lastPage,
    finished: log?.finished ?? false,
    verdict,
    finalSummary: log?.final_summary ? log.final_summary : null,
    elapsedSec,
    task: meta.task,
    logPath: meta.logPath,
    workdir: meta.workdir,
  };
}
