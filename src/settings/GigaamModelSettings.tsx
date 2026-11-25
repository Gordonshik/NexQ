import { useCallback, useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { AlertTriangle, CheckCircle, FolderOpen, HardDrive, Loader2, RefreshCw, Trash2, XCircle } from "lucide-react";
import { detectGigaamModel, validateGigaamModel } from "../lib/ipc";
import { GIGAAM_LANGUAGE, GIGAAM_NAME } from "../lib/gigaam";
import type { GigaAmModelInfo } from "../lib/types";
import { useConfigStore } from "../stores/configStore";

export interface GigaamValidationState {
  status: "idle" | "loading" | "ready" | "error";
  path: string | null;
}

interface GigaamModelSettingsProps {
  onStatusChange?: (state: GigaamValidationState) => void;
}

export function GigaamModelSettings({ onStatusChange }: GigaamModelSettingsProps) {
  const modelPath = useConfigStore((s) => s.gigaamModelPath);
  const setModelPath = useConfigStore((s) => s.setGigaamModelPath);
  const [draftPath, setDraftPath] = useState(modelPath ?? "");
  const [info, setInfo] = useState<GigaAmModelInfo | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const requestRef = useRef(0);

  const validate = useCallback(async (path: string, persist: boolean) => {
    const selected = path.trim();
    if (!selected) {
      setError("Choose the GigaAM model folder first.");
      return;
    }
    const request = ++requestRef.current;
    setDraftPath(selected);
    setLoading(true);
    setError("");
    setInfo(null);
    try {
      const result = await validateGigaamModel(selected);
      if (request !== requestRef.current) return;
      setDraftPath(result.model_dir);
      setInfo(result);
      if (persist) setModelPath(result.model_dir);
    } catch (cause) {
      if (request !== requestRef.current) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  }, [setModelPath]);

  useEffect(() => {
    const path = modelPath?.trim() ?? "";
    setDraftPath(path);
    setInfo(null);
    setError("");
    if (path) void validate(path, false);
    else {
      requestRef.current += 1;
      setLoading(false);
    }
  }, [modelPath, validate]);

  useEffect(() => {
    onStatusChange?.({
      status: loading ? "loading" : error ? "error" : info ? "ready" : "idle",
      path: draftPath.trim() || null,
    });
  }, [draftPath, error, info, loading, onStatusChange]);

  const detectDefault = async () => {
    const request = ++requestRef.current;
    setLoading(true);
    setError("");
    setInfo(null);
    try {
      const result = await detectGigaamModel();
      if (request !== requestRef.current) return;
      await validate(result.model_dir, true);
    } catch (cause) {
      if (request !== requestRef.current) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  };

  const chooseFolder = async () => {
    let selected: string | string[] | null;
    try {
      selected = await open({ directory: true, title: "Choose GigaAM Model Folder", multiple: false });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    if (typeof selected === "string") await validate(selected, true);
  };

  const clear = () => {
    requestRef.current += 1;
    setModelPath(null);
    setDraftPath("");
    setInfo(null);
    setError("");
    setLoading(false);
  };

  return (
    <section className="rounded-xl border border-border/30 bg-card/50 p-5" aria-labelledby="gigaam-model-title">
      <div className="mb-2 flex items-center gap-1.5">
        <HardDrive className="h-4 w-4 text-primary/80" />
        <h3 id="gigaam-model-title" className="text-sm font-semibold text-primary/80">{GIGAAM_NAME} Model</h3>
      </div>
      <p className="mb-4 text-xs leading-relaxed text-muted-foreground">
        Enter or choose one exported v3 e2e RNNT folder for both You and Them. Both channels share the same loaded model and transcribe Russian ({GIGAAM_LANGUAGE}) only.
      </p>

      <div className="rounded-lg border border-border/40 bg-background/60 p-3.5">
        <label htmlFor="gigaam-model-path" className="mb-1.5 block text-meta font-medium uppercase tracking-wide text-muted-foreground/70">Model folder</label>
        <input
          id="gigaam-model-path"
          type="text"
          value={draftPath}
          onChange={(event) => {
            setDraftPath(event.target.value);
            setInfo(null);
            setError("");
          }}
          disabled={loading}
          placeholder="Paste the full model folder path"
          autoComplete="off"
          spellCheck={false}
          aria-invalid={error ? true : undefined}
          className="w-full rounded-md border border-border/50 bg-background px-3 py-2 font-mono text-xs text-foreground outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-primary focus:ring-1 focus:ring-primary/20 disabled:cursor-not-allowed disabled:opacity-70"
        />
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" onClick={detectDefault} disabled={loading} className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Detect Default Folder
        </button>
        <button type="button" onClick={chooseFolder} disabled={loading} className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
          <FolderOpen className="h-3.5 w-3.5" />
          Choose Folder
        </button>
        <button type="button" onClick={() => void validate(draftPath, true)} disabled={loading || !draftPath.trim()} className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
          Apply Path
        </button>
        <button type="button" onClick={() => void validate(draftPath, false)} disabled={loading || !draftPath.trim()} className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50">
          <RefreshCw className="h-3.5 w-3.5" />
          Check Model
        </button>
        {modelPath && (
          <button type="button" onClick={clear} disabled={loading} className="inline-flex items-center gap-1.5 rounded-lg border border-destructive/30 px-3 py-2 text-xs text-destructive transition-colors hover:bg-destructive/10 disabled:cursor-not-allowed disabled:opacity-50">
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
        )}
      </div>

      <div className="mt-3 min-h-[18px]" role="status" aria-live="polite">
        {loading && <p className="text-xs text-muted-foreground">Checking the exported model...</p>}
        {!loading && info && (
          <div className="text-xs text-success">
            <p className="flex items-center gap-1.5 font-medium">
              <CheckCircle className="h-3.5 w-3.5" />
              {info.model_name} is ready
            </p>
            <p className="mt-1 text-muted-foreground">
              {info.sample_rate / 1000} kHz · {info.features} mel bins · {info.vocabulary_size} vocabulary tokens
            </p>
          </div>
        )}
        {!loading && info && !!info.warnings?.length && (
          <ul className="mt-2 space-y-1.5 text-warning">
            {info.warnings.map((warning) => (
              <li key={warning} className="flex items-start gap-1.5">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {error && (
        <p role="alert" className="mt-2 flex items-start gap-1.5 text-xs text-destructive">
          <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </p>
      )}
    </section>
  );
}
