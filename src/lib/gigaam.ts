import type { MeetingAudioConfig, PartyAudioConfig } from "./types";

export const GIGAAM_NAME = "GigaAM v3 RNNT";
export const GIGAAM_LANGUAGE = "ru-RU";

function withModelPath(party: PartyAudioConfig, modelPath: string | null): PartyAudioConfig {
  if (party.stt_provider !== "gigaam") return party;
  const { local_model_id: _modelId, local_model_path: _modelPath, ...rest } = party;
  const next: PartyAudioConfig = {
    ...rest,
    language: GIGAAM_LANGUAGE,
    ...(modelPath && { local_model_path: modelPath }),
  };
  if (next.local_model_path === party.local_model_path && next.language === party.language && !("local_model_id" in party)) {
    return party;
  }
  return next;
}

export function selectGigaam(party: PartyAudioConfig, modelPath: string | null): PartyAudioConfig {
  return withModelPath({ ...party, stt_provider: "gigaam" }, modelPath?.trim() || null);
}

export function withGigaamModelPath(config: MeetingAudioConfig | null, modelPath: string | null): MeetingAudioConfig | null {
  if (!config) return config;
  const path = modelPath?.trim() || null;
  const you = withModelPath(config.you, path);
  const them = withModelPath(config.them, path);
  return you === config.you && them === config.them ? config : { ...config, you, them };
}
