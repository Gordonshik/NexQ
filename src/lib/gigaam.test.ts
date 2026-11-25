import { describe, expect, it } from "vitest";
import type { MeetingAudioConfig, PartyAudioConfig } from "./types";
import { GIGAAM_LANGUAGE, selectGigaam, withGigaamModelPath } from "./gigaam";

const you: PartyAudioConfig = {
  role: "You",
  device_id: "Mic",
  is_input_device: true,
  stt_provider: "deepgram",
  local_model_id: "nova-3",
  language: "en-US",
  local_model_path: "C:\\Old\\Model",
};
const them: PartyAudioConfig = {
  role: "Them",
  device_id: "Speakers",
  is_input_device: false,
  stt_provider: "gigaam",
  language: "auto",
};
const config: MeetingAudioConfig = {
  you,
  them,
  recording_enabled: true,
  preset_name: null,
};

describe("GigaAM party configuration", () => {
  it("selects Russian and clears incompatible model settings", () => {
    expect(selectGigaam(you, null)).toEqual({
      role: "You",
      device_id: "Mic",
      is_input_device: true,
      stt_provider: "gigaam",
      language: "ru-RU",
    });
  });

  it("projects one shared folder to both GigaAM sides without touching another provider", () => {
    const updated = withGigaamModelPath(config, "C:\\GigaPisar\\model");
    expect(updated?.you).toEqual(you);
    expect(updated?.them.local_model_path).toBe("C:\\GigaPisar\\model");
    expect(updated?.them.language).toBe(GIGAAM_LANGUAGE);
    expect(updated?.recording_enabled).toBe(true);
  });

  it("removes a cleared folder from GigaAM sides", () => {
    const updated = withGigaamModelPath(config, "  ");
    expect(updated?.them).toEqual({ role: "Them", device_id: "Speakers", is_input_device: false, stt_provider: "gigaam", language: "ru-RU" });
    expect(updated?.them).not.toHaveProperty("local_model_path");
  });
});
