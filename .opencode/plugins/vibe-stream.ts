/**
 * vibe-stream — companion plugin that streams live VibeHarness progress into the
 * OpenCode session OUT-OF-BAND.
 *
 * The six vibe_* tools return only a TINY summary. This plugin holds the SDK
 * `client` and, on a short poll, tails every active run's structured `.vibe/`
 * runlog (rewritten by the harness each turn) and forwards per-action progress
 * lines into the originating session via
 *   client.session.prompt({ path:{id:sessionID}, body:{ noReply:true, parts:[…] }})
 * plus milestone toasts via client.tui.showToast(...). This keeps the full
 * stream visible to the human while the agent's context stays small.
 *
 * Polling the structured runlog (vs. parsing partial stdout) gives clean,
 * de-duplicated per-turn / per-action lines.
 */
import type { Plugin } from "@opencode-ai/plugin";
import { listRuns, parseRunlog, isAlive, preview } from "../lib/vibe";

interface RunStream {
  reportedActions: number; // cumulative actions already forwarded
  lastTurnToasted: number;
  doneToasted: boolean;
  validationsReported: number;
}

const POLL_MS = 1500;
const KEY = "__vibeStreamState__";

export const VibeStream: Plugin = async ({ client, directory }) => {
  // Guard against duplicate intervals if the plugin module is re-evaluated.
  const g = globalThis as any;
  if (g[KEY]?.timer) {
    clearInterval(g[KEY].timer);
  }
  const streams = new Map<string, RunStream>();
  g[KEY] = { timer: null as any };

  async function say(sessionID: string | null, text: string) {
    if (!sessionID) return;
    try {
      await client.session.prompt({
        path: { id: sessionID },
        body: { noReply: true, parts: [{ type: "text", text }] },
      });
    } catch {
      /* session may be gone; ignore */
    }
  }

  async function toast(message: string, variant: string = "info") {
    try {
      await (client as any).tui?.showToast?.({
        body: { message, variant },
      });
    } catch {
      /* tui may be unavailable in headless/run mode; ignore */
    }
  }

  async function tick() {
    let runs;
    try {
      runs = listRuns(directory);
    } catch {
      return;
    }
    for (const meta of runs) {
      const st = streams.get(meta.runId) ?? {
        reportedActions: 0,
        lastTurnToasted: 0,
        doneToasted: false,
        validationsReported: 0,
      };
      streams.set(meta.runId, st);

      // Skip runs we've already wrapped up.
      if (st.doneToasted) continue;

      const log = parseRunlog(meta.vibeDir);
      const alive = isAlive(meta.pid);

      if (log) {
        // Forward any new actions, oldest-first, as compact lines.
        let seen = 0;
        for (const turn of log.turns) {
          if (turn.index > st.lastTurnToasted) {
            st.lastTurnToasted = turn.index;
            void toast(`vibe ${meta.runId}: turn ${turn.index}`, "info");
          }
          for (const a of turn.actions) {
            seen++;
            if (seen > st.reportedActions) {
              const mark = a.ok ? "✓" : "✗";
              await say(
                meta.sessionID,
                `[vibe ${meta.runId} · turn ${turn.index}] ${mark} ` +
                  `${a.tool ?? "?"} — ${preview(a.observation, 200)}`,
              );
            }
          }
        }
        st.reportedActions = Math.max(st.reportedActions, seen);

        // Forward new validator verdicts.
        for (let i = st.validationsReported; i < log.validations.length; i++) {
          const v = log.validations[i];
          await say(
            meta.sessionID,
            `[vibe ${meta.runId}] validator ${v.passed ? "PASS" : "FAIL"} — ` +
              `${preview(v.reason, 200)}`,
          );
        }
        st.validationsReported = log.validations.length;
      }

      // Terminal handling: finished per runlog, or the process is gone.
      const finished = log?.finished ?? false;
      if (finished || (!alive && meta.status !== "running")) {
        // status already terminal in meta, or runlog says finished
      }
      if (finished) {
        await say(
          meta.sessionID,
          `[vibe ${meta.runId}] ✅ finished — ${preview(
            log?.final_summary ?? "",
            240,
          )}`,
        );
        void toast(`vibe ${meta.runId}: finished`, "success");
        st.doneToasted = true;
      } else if (!alive && log && log.turns.length === 0) {
        // Process died before producing any runlog — likely a spawn/config error.
        await say(
          meta.sessionID,
          `[vibe ${meta.runId}] ⚠️ process exited before producing a runlog ` +
            `(see vibe_info which:tail).`,
        );
        void toast(`vibe ${meta.runId}: failed`, "error");
        st.doneToasted = true;
      } else if (!alive && !finished) {
        await say(
          meta.sessionID,
          `[vibe ${meta.runId}] ⚠️ process ended without a finish verdict ` +
            `(see vibe_status / vibe_info).`,
        );
        void toast(`vibe ${meta.runId}: ended`, "warning");
        st.doneToasted = true;
      }
    }
  }

  g[KEY].timer = setInterval(() => {
    void tick();
  }, POLL_MS);
  // Don't keep the host process alive solely for this poller.
  try {
    (g[KEY].timer as any).unref?.();
  } catch {
    /* ignore */
  }

  return {
    // The streaming happens on the interval above; we also expose an event hook
    // so the plugin participates in the lifecycle (and could react to session
    // events later). No-op for now.
    event: async () => {},
  };
};

export default VibeStream;
