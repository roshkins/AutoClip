//! CLI parsing, env-var <-> flag translation, and help text.
//!
//! AutoClip's convention: every supported env var (e.g. `CLIP_FACE_RATIO`)
//! is also accepted as a CLI flag by lowercasing and replacing `_` with `-`
//! (`--clip-face-ratio=0.45`). The `ENV_SPECS` table is the single source
//! of truth for which env vars are recognised and whether they expect a
//! value, are optional booleans, or are bare flags.

use anyhow::Result;

#[derive(Clone, Copy, Debug)]
pub enum EnvValueMode {
    /// `--foo VALUE` or `--foo=VALUE` — a value is required.
    Required,
    /// `--foo` (defaults to `"1"`) or `--foo=VALUE`.
    Optional,
    /// `--foo` only (no value form); presence sets `"1"`.
    Flag,
}

#[derive(Clone, Copy, Debug)]
pub struct EnvSpec {
    pub env: &'static str,
    pub mode: EnvValueMode,
}

pub const ENV_SPECS: &[EnvSpec] = &[
    EnvSpec { env: "CLIP_PAGE_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LAYOUT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_CROP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_CONTEXT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ZOOM", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_BOX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_REGION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ANCHOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAME_CENTER", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAME_REGION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_BACKEND", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_DYLIB", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_ORT_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_GPU_MEM_LIMIT_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_ORT_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_DUMP_DIR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_DUMP_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_PICK_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TRACK_STEP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TILE_MIN_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_TILE_MAX_DEPTH", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_HEAD_TOP_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_TOP_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_EYE_CHIN_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_FRAME_SHOULDER_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL_MIN_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_MODEL_MAX_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_LOAD_TIMEOUT_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_BACKEND", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_MESH_TRACT_OPT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_MAX", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_INPUT_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_REGION_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_HEADROOM", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_MESH_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_MODEL_MIN_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_MODEL_MAX_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_BACKEND", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_POSE_TRACT_OPT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_INPUT_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_HEAD_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_SHOULDER_MARGIN", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_POSE_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ACTIVE_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ACTIVE_AREA_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_FILE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_THRESHOLD", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_REQUIRE_MOTION", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_FACE_ID_BGR", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_ID_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_TRACK", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_AUDIO_NORM", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAPTIONS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAPTIONS_POSITION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_FONT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_COLOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_OUTLINE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_OUTLINE_COLOR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MIN_WORD_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MAX_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_CHEST_RATIO", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_MARGIN_OFFSET", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAPTIONS_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CLOSED_CAPTIONS", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENABLE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_ENDPOINT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TEMPERATURE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TIMEOUT_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_BACKOFF_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_MAX_TRANSCRIPT_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_TITLE_MAX_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_DESCRIPTION_MAX_CHARS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_GPU_LAYERS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LLM_FILE_RENAME", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LLM_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_FACE_BUDGET_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_DETECT_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_SAMPLES", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_START", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_STEP", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_DETECT_FULL", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_DETECT_BUDGET_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_TS_REALTIME", mode: EnvValueMode::Flag },
    EnvSpec { env: "CLIP_GAMEPLAY", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_REGION_DETECT", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LIVE_CONFIG", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LIVE_CONFIG_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_LIVE_FAST", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_LIVE_LAYOUT_TTL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_FILE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_POLL_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAMS_MAX_CONCURRENT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_M3U8_REFRESH_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_STREAM_OFFLINE_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_WAKE_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_PROFILE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_ENABLE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_AUDIO", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_FACE", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_EMOTION_THRESHOLD", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_WORDS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_RMS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_PEAK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_AUDIO_WEIGHT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_TEXT_WEIGHT", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_REFRACTORY_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_FACE_MOTION", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_EMOTION_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_GAMEPLAY_MODEL_DIR", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TEXT_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_VISION_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TOKENIZER", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_STRIDE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_TOPK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_SIZE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_NEG_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_GAMEPLAY_DEBUG", mode: EnvValueMode::Optional },
    EnvSpec { env: "CLIP_CAM_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_NEG_LABELS", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_SCORE", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_TOPK", mode: EnvValueMode::Required },
    EnvSpec { env: "CLIP_CAM_REGION_SCALE", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_URL_OVERRIDE", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_TEST_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "M3U8_PAGE_URL", mode: EnvValueMode::Required },
    EnvSpec { env: "COOKIE_HEADER", mode: EnvValueMode::Required },
    EnvSpec { env: "KICK_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "TIKTOK_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_COOKIE", mode: EnvValueMode::Required },
    EnvSpec { env: "HEADLESS_M3U8_SCRIPT", mode: EnvValueMode::Required },
    EnvSpec { env: "HEADLESS_M3U8_SCRIPT_TIKTOK", mode: EnvValueMode::Required },
    EnvSpec { env: "LOG_M3U8_HEADERS", mode: EnvValueMode::Flag },
    EnvSpec { env: "TWITCH_CLIENT_ID", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_OAUTH_TOKEN", mode: EnvValueMode::Required },
    EnvSpec { env: "TWITCH_AUTH_TOKEN", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_REFRACTORY_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_BUFFER_HEADROOM_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_BUFFER_RESTART_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_NO_WORDS_SECS", mode: EnvValueMode::Required },
    EnvSpec { env: "WAKE_FF_AF", mode: EnvValueMode::Required },
    EnvSpec { env: "SKIP_CLIP_SAVE", mode: EnvValueMode::Optional },
    EnvSpec { env: "MIC_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_RT_TARGET", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_MODEL_CANDIDATES", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_CLIP_GPU", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_CUBLAS", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_LOG_LEVEL", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_ISOLATE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_MODE", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_STATUS_PATH", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_WORKER_MEDIA_URL_PATH", mode: EnvValueMode::Required },
    EnvSpec { env: "WHISPER_WORKER_LOG_RAW", mode: EnvValueMode::Optional },
    EnvSpec { env: "WHISPER_WORKER_STATUS_MS", mode: EnvValueMode::Required },
    EnvSpec { env: "GGML_LOG_LEVEL", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_BIN", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_ENCODER", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL_DEVICE", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_HWACCEL_FALLBACK", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_MIN_FREE_VRAM_MB", mode: EnvValueMode::Required },
    EnvSpec { env: "FFMPEG_ENCODE_TIMEOUT_SECS", mode: EnvValueMode::Required },
];

pub struct ParsedCli {
    pub positionals: Vec<String>,
    pub override_phrase: Option<String>,
    pub log_raw_wake: bool,
    pub log_raw_wake_set: bool,
    pub mic_device: Option<String>,
    pub stream_urls: Vec<String>,
    pub streams_file: Option<String>,
    pub env_overrides: Vec<(String, String)>,
}

pub fn env_to_flag(env: &str) -> String {
    env.to_ascii_lowercase().replace('_', "-")
}

fn split_stream_list(raw: &str) -> Vec<String> {
    raw.split(|c| matches!(c, ',' | '|' | ';' | '\n' | '\r'))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub fn parse_cli_args(args: &[String]) -> Result<ParsedCli> {
    let mut positionals = Vec::new();
    let mut env_overrides = Vec::new();
    let mut override_phrase = None;
    let mut log_raw_wake = true;
    let mut log_raw_wake_set = false;
    let mut mic_device = None;
    let mut stream_urls: Vec<String> = Vec::new();
    let mut streams_file: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            positionals.extend(args[i + 1..].iter().cloned());
            break;
        }
        if let Some(rest) = arg.strip_prefix("--phrase=") {
            override_phrase = Some(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--stream=") {
            stream_urls.extend(split_stream_list(rest));
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--streams=") {
            streams_file = Some(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--phrases=") {
            override_phrase = Some(rest.to_string());
            i += 1;
            continue;
        }
        if arg == "--phrase" || arg == "--phrases" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--phrase expects a value");
            };
            override_phrase = Some(val.clone());
            i += 1;
            continue;
        }
        if arg == "--stream" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--stream expects a value");
            };
            stream_urls.extend(split_stream_list(val));
            i += 1;
            continue;
        }
        if arg == "--streams" {
            i += 1;
            let Some(val) = args.get(i) else {
                anyhow::bail!("--streams expects a value");
            };
            streams_file = Some(val.clone());
            i += 1;
            continue;
        }
        if arg == "--log-raw-wake" {
            log_raw_wake = true;
            log_raw_wake_set = true;
            i += 1;
            continue;
        }
        if arg == "--no-log-raw-wake" {
            log_raw_wake = false;
            log_raw_wake_set = true;
            i += 1;
            continue;
        }

        let mut handled_env = false;
        if let Some(raw_flag) = arg.strip_prefix("--") {
            let (flag, inline_value) = match raw_flag.split_once('=') {
                Some((f, v)) => (f.to_ascii_lowercase(), Some(v.to_string())),
                None => (raw_flag.to_ascii_lowercase(), None),
            };
            for spec in ENV_SPECS {
                let spec_flag = env_to_flag(spec.env);
                if spec_flag == flag {
                    let value = match spec.mode {
                        EnvValueMode::Required => {
                            if let Some(val) = inline_value {
                                val
                            } else {
                                let next = args.get(i + 1).ok_or_else(|| {
                                    anyhow::anyhow!("--{flag} expects a value")
                                })?;
                                if next.starts_with("--") {
                                    anyhow::bail!("--{flag} expects a value");
                                }
                                i += 1;
                                next.clone()
                            }
                        }
                        EnvValueMode::Optional => inline_value.unwrap_or_else(|| "1".to_string()),
                        EnvValueMode::Flag => inline_value.unwrap_or_else(|| "1".to_string()),
                    };
                    if spec.env == "MIC_DEVICE" {
                        mic_device = Some(value.clone());
                    }
                    env_overrides.push((spec.env.to_string(), value));
                    handled_env = true;
                    break;
                }
            }
        }

        if handled_env {
            i += 1;
            continue;
        }

        if arg.starts_with('-') {
            i += 1;
            continue;
        }

        positionals.push(arg.clone());
        i += 1;
    }

    Ok(ParsedCli {
        positionals,
        override_phrase,
        log_raw_wake,
        log_raw_wake_set,
        mic_device,
        stream_urls,
        streams_file,
        env_overrides,
    })
}

pub fn apply_env_overrides(overrides: &[(String, String)]) {
    for (key, value) in overrides {
        std::env::set_var(key, value);
    }
}

pub fn print_help(bin: &str) {
    println!("Usage:");
    println!("  {bin} <page_url> [options]  (scheme optional, e.g. kick.com/user)");
    println!("  {bin} demo-buffer");
    println!("  {bin} demo-hls-buffer <page_url>");
    println!("  {bin} demo-ts <path_to_ts> [--phrase WORDS] [--no-log-raw-wake]");
    println!("  {bin} reprocess-ts <path_to_ts>");
    println!("  {bin} check-gameplay-model [model_dir]");
    println!("  {bin} demo-detect <media_path>");
    println!("  {bin} face-sweep <positives_dir> <negatives_dir> [score_start score_end score_step] [out_csv]");
    println!("  {bin} demo-wakeword-mic <page_url> [--phrase WORDS] [--log-raw-wake] [--mic-device NAME]");
    println!("");
    println!("Options:");
    println!("  --phrase WORDS          Override wake phrase(s), comma/pipe separated (default: 'orange')");
    println!("  --stream URL            Add a stream URL to watch (repeat or comma/pipe separated)");
    println!("  --streams PATH          Streams file to sync/watch for multi-stream runs");
    println!("  --log-raw-wake         Log raw/normalized transcripts (default on)");
    println!("  --no-log-raw-wake      Disable transcript logging");
    println!("  --mic-device NAME      Microphone device for mic wake mode");
    println!("  -h, --help             Show this help");
    println!("");
    println!("CLI overrides:");
    println!("  Any environment variable below can be passed as a CLI flag by lowercasing");
    println!("  and replacing '_' with '-' (e.g., CLIP_LAYOUT -> --clip-layout=stacked).");
    println!("  For boolean env vars, use the '=true'/'=false' form to avoid ambiguity.");
    println!("");
    println!("Environment (selected):");
    println!("  CLIP_PAGE_URL            Default page when none is passed");
    println!("  CLIP_LAYOUT              Layout mode: stacked (default) or full");
    println!("  CLIP_FACE_RATIO          Height ratio reserved for face panel (default 0.40)");
    println!("  CLIP_FACE_CROP           Face crop expr w:h:x:y (optional, overrides detection/anchor)");
    println!("  CLIP_FACE_CONTEXT        Face crop expansion scale for detected face (default 6.0)");
    println!("  CLIP_FACE_ZOOM           Face crop zoom factor (>1 zooms out, <1 zooms in; default 1.0)");
    println!("  CLIP_FACE_BOX            Normalized face box x:y:w:h (0..1) for auto-crop");
    println!("  CLIP_FACE_REGION         Normalized face bounds x:y:w:h (0..1) for mid-shot framing");
    println!("  CLIP_FACE_ANCHOR         Anchor for default face crop (top-left default)");
    println!("  CLIP_GAME_CENTER         Normalized gameplay center x:y (0..1) for reticle centering");
    println!("  CLIP_GAME_REGION         Normalized gameplay bounds x:y:w:h (0..1) for layout checks");
    println!("  CLIP_FACE_MODEL          Face detector model path (default models/face_detection_yunet_2023mar.onnx)");
    println!("  CLIP_FACE_BACKEND        Face detector backend: auto (default), ort, or tract");
    println!("  CLIP_ORT_DYLIB           Path to onnxruntime.dll (optional)");
    println!("  CLIP_ORT_DEVICE          ORT CUDA device id, auto, or cpu (default auto)");
    println!("  CLIP_ORT_GPU_MEM_LIMIT_MB ORT CUDA arena limit in MB (default 0 = unlimited)");
    println!("  CLIP_ORT_MIN_FREE_VRAM_MB Min free VRAM before using ORT CUDA (default 512)");
    println!("  CLIP_FACE_DUMP_DIR       Write face debug images with rectangles to this folder");
    println!("  CLIP_FACE_DUMP_RAW       Dump raw face candidates (no score filtering) when enabled");
    println!("  CLIP_FACE_PICK_RAW       Pick faces using raw detector score (default true)");
    println!("  CLIP_FACE_SCORE          Face detection confidence threshold (default 0.6)");
    println!("  CLIP_FACE_TRACK_STEP     Seconds between face tracking samples (default 2.0)");
    println!("  CLIP_AUDIO_NORM          Normalize clip audio loudness (default true)");
    println!("  CLIP_CAPTIONS            Enable word-by-word open captions (default false)");
    println!("  CLIP_CAPTIONS_POSITION   Caption placement: margin (default) or chest");
    println!("  CLIP_CAPTIONS_FONT       Caption font name or TTF path (optional)");
    println!("  CLIP_CAPTIONS_SIZE       Caption font size px or ratio (<=2 treated as ratio)");
    println!("  CLIP_CAPTIONS_COLOR      Caption text color (default white)");
    println!("  CLIP_CAPTIONS_OUTLINE    Caption outline width (default 3)");
    println!("  CLIP_CAPTIONS_OUTLINE_COLOR Caption outline color (default black)");
    println!("  CLIP_CAPTIONS_MIN_WORD_SECS Minimum per-word on-screen time (default 0.12s)");
    println!("  CLIP_CAPTIONS_MAX_WORDS  Cap on rendered words (default 300)");
    println!("  CLIP_CAPTIONS_CHEST_RATIO Caption Y ratio when placed on chest (default 0.65)");
    println!("  CLIP_CAPTIONS_MARGIN_OFFSET Extra Y offset for margin captions (default 0)");
    println!("  CLIP_CAPTIONS_DEBUG      Log caption timings (default false)");
    println!("  CLIP_CLOSED_CAPTIONS     Embed closed captions track when available (default true)");
    println!("  CLIP_LLM_ENABLE          Enable LLM metadata (default false)");
    println!("  CLIP_LLM_ENDPOINT        LLM chat completions endpoint (default localhost:1234)");
    println!("  CLIP_LLM_MODEL           LLM model name or path");
    println!("  CLIP_LLM_TEMPERATURE     LLM temperature (default 0.2)");
    println!("  CLIP_LLM_TIMEOUT_SECS    LLM request timeout seconds (default 20)");
    println!("  CLIP_LLM_BACKOFF_SECS    LLM backoff seconds after failure (default 120)");
    println!("  CLIP_LLM_MAX_TRANSCRIPT_CHARS Transcript truncation limit (default 4000)");
    println!("  CLIP_LLM_TITLE_MAX_CHARS LLM title max length (default 80)");
    println!("  CLIP_LLM_DESCRIPTION_MAX_CHARS LLM description max length (default 280)");
    println!("  CLIP_LLM_GPU_LAYERS      LLM GPU layers (0 = CPU, empty = model default)");
    println!("  CLIP_LLM_FILE_RENAME     Rename clip files using LLM title (default true)");
    println!("  CLIP_LLM_DEBUG           Log LLM parse details (default false)");
    println!("  CLIP_FACE_BUDGET_SECS    Override face detection time budget in seconds");
    println!("  CLIP_FACE_TILE_MIN_SCORE Tile search min score (default 0.60; set <= 0 to disable)");
    println!("  CLIP_FACE_TILE_MAX_DEPTH Max bisection depth for tile search (default 3)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP      Head top offset vs face box (default -0.28)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MIN  Min head top offset (default -0.8)");
    println!("  CLIP_FACE_FRAME_HEAD_TOP_MAX  Max head top offset (default 0.2)");
    println!("  CLIP_FACE_FRAME_EYE_TOP_RATIO Eye->top ratio for head estimate (default 0.45)");
    println!("  CLIP_FACE_FRAME_EYE_CHIN_RATIO Eye->chin ratio for head estimate (default 0.55)");
    println!("  CLIP_FACE_FRAME_SHOULDER_SCALE Shoulder width scale vs face (default 3.2)");
    println!("  CLIP_FACE_MESH          Enable face mesh framing (default true)");
    println!("  CLIP_FACE_MESH_MODEL    Face mesh ONNX model path");
    println!("  CLIP_FACE_MESH_MODEL_MIN_MB Min face mesh model size in MB (default 1)");
    println!("  CLIP_FACE_MESH_MODEL_MAX_MB Max face mesh model size in MB (default 64)");
    println!("  CLIP_FACE_MESH_LOAD_TIMEOUT_SECS Max seconds to load face mesh model (default 60)");
    println!("  CLIP_FACE_MESH_BACKEND  Face mesh backend: auto (default), ort, or tract");
    println!("  CLIP_FACE_MESH_TRACT_OPT Enable tract optimizations for face mesh (default false)");
    println!("  CLIP_FACE_MESH_INPUT_SIZE Fallback face mesh input size (default 192)");
    println!("  CLIP_FACE_MESH_INPUT_MAX Max face mesh input side length before clamping (default 512)");
    println!("  CLIP_FACE_MESH_INPUT_SCALE Input scale for face mesh model (default 1/255)");
    println!("  CLIP_FACE_MESH_REGION_SCALE Face mesh crop expansion scale (default 1.35)");
    println!("  CLIP_FACE_MESH_HEADROOM Extra headroom ratio above mesh top (default 0.12)");
    println!("  CLIP_FACE_MESH_DEBUG    Log face mesh bounds (default false)");
    println!("  CLIP_POSE               Enable MoveNet pose framing (default true)");
    println!("  CLIP_POSE_MODEL         MoveNet Thunder ONNX model path");
    println!("  CLIP_POSE_MODEL_MIN_MB  Min pose model size in MB (default 1)");
    println!("  CLIP_POSE_MODEL_MAX_MB  Max pose model size in MB (default 64)");
    println!("  CLIP_POSE_LOAD_TIMEOUT_SECS Max seconds to load pose model (default 60)");
    println!("  CLIP_POSE_BACKEND       Pose backend: auto (default), ort, or tract");
    println!("  CLIP_POSE_TRACT_OPT     Enable tract optimizations for pose (default false)");
    println!("  CLIP_POSE_SCORE         Min keypoint score for pose framing (default 0.30)");
    println!("  CLIP_POSE_INPUT_SIZE    Fallback pose input size when model is dynamic (default 256)");
    println!("  CLIP_POSE_INPUT_MAX     Max pose input side length before clamping (default 512)");
    println!("  CLIP_POSE_INPUT_SCALE   Input scale for pose model (default 1/255)");
    println!("  CLIP_POSE_HEAD_RATIO    Head-top margin ratio from pose (default 0.60)");
    println!("  CLIP_POSE_SHOULDER_MARGIN Shoulder width margin multiplier (default 1.10)");
    println!("  CLIP_POSE_DEBUG         Log pose keypoints and frame spec (default false)");
    println!("  CLIP_FACE_ID            Enable face-ID matching (default false)");
    println!("  CLIP_FACE_ID_FILE       Face-ID embedding file (json)");
    println!("  CLIP_FACE_ID_MODEL      Face-ID ONNX model path");
    println!("  CLIP_FACE_ID_THRESHOLD  Face-ID cosine threshold (default 0.35)");
    println!("  CLIP_FACE_ID_REQUIRE_MOTION Require motion for face-ID (default true)");
    println!("  CLIP_FACE_ID_MOTION     Motion threshold for face-ID (default 0.015)");
    println!("  CLIP_FACE_ID_BGR        Use BGR input for face-ID model (default true)");
    println!("  CLIP_FACE_ID_DEBUG      Log face-ID scores (default false)");
    println!("  CLIP_FACE_DEBUG          Log face detector outputs and best score");
    println!("  CLIP_FACE_TRACK          Track face across the full clip (default true)");
    println!("  CLIP_DETECT              Enable auto-detection for stacked layout (default true)");
    println!("  CLIP_DETECT_SIZE         Reticle detection frame size WxH (default 960x540)");
    println!("  CLIP_DETECT_SAMPLES      Number of detection frames to sample (default 3)");
    println!("  CLIP_DETECT_START        Detection sample start time in seconds (default 1.0)");
    println!("  CLIP_DETECT_STEP         Seconds between detection samples (default 1.5)");
    println!("  CLIP_DETECT_FULL         Sample detection frames across the full clip (local files only)");
    println!("  CLIP_DETECT_BUDGET_SECS  Max seconds to spend analyzing detection samples");
    println!("  CLIP_GAMEPLAY            Enable CLIP model usage (default true)");
    println!("  CLIP_REGION_DETECT       Enable CLIP region detection (default true)");
    println!("  CLIP_LIVE_CONFIG         Path to live config file for hot-reload overrides");
    println!("  CLIP_LIVE_CONFIG_POLL_SECS   Live config poll interval in seconds (default 2)");
    println!("  CLIP_LIVE_FAST           Skip heavy detection/captions for faster live renders (default true)");
    println!("  CLIP_LIVE_LAYOUT_TTL_SECS Reuse detected layout hints for N seconds (default 120)");
    println!("  CLIP_STREAMS_FILE        Path to streams file for multi-stream runs");
    println!("  CLIP_STREAMS_POLL_SECS   Streams file poll interval in seconds (default 5)");
    println!("  CLIP_STREAMS_MAX_CONCURRENT Max number of concurrent streams (default 1)");
    println!("  CLIP_M3U8_REFRESH_SECS   Refresh signed m3u8 URL every N seconds (default 240)");
    println!("  CLIP_STREAM_OFFLINE_SECS Exit if no new segments for N seconds (default 120)");
    println!("  CLIP_WAKE_WORDS          Wake phrase list (comma/pipe separated) for clip trigger");
    println!("  CLIP_PROFILE             Enable timing logs for hotspots (default false)");
    println!("  CLIP_EMOTION_ENABLE      Enable emotion triggers from audio/face (default false)");
    println!("  CLIP_EMOTION_WORDS       Emotion keyword list (comma/pipe separated)");
    println!("  CLIP_EMOTION_THRESHOLD   Emotion score threshold (default 1.5)");
    println!("  CLIP_EMOTION_AUDIO_RMS   RMS loudness threshold (default 0.08)");
    println!("  CLIP_EMOTION_AUDIO_PEAK  Peak loudness threshold (default 0.35)");
    println!("  CLIP_EMOTION_FACE_MOTION Face motion threshold (default 0.035)");
    println!("  CLIP_GAMEPLAY_LABELS     CLIP gameplay positive labels");
    println!("  CLIP_GAMEPLAY_NEG_LABELS CLIP gameplay negative labels");
    println!("  CLIP_GAMEPLAY_SCORE      CLIP gameplay score threshold (default 0.12)");
    println!("  CLIP_GAMEPLAY_TOPK       CLIP gameplay top-k patches (default 6)");
    println!("  CLIP_CAM_LABELS          CLIP cam positive labels");
    println!("  CLIP_CAM_NEG_LABELS      CLIP cam negative labels");
    println!("  CLIP_CAM_SCORE           CLIP cam score threshold (default CLIP_GAMEPLAY_SCORE)");
    println!("  CLIP_CAM_TOPK            CLIP cam top-k patches (default CLIP_GAMEPLAY_TOPK)");
    println!("  CLIP_CAM_REGION_SCALE    Expand CLIP cam region bounds (default 1.6)");
    println!("  CLIP_TS_REALTIME         When set, read local TS files at realtime speed");
    println!("  M3U8_URL_OVERRIDE        Skip discovery; use this master URL directly");
    println!("  COOKIE_HEADER / KICK_COOKIE / TIKTOK_COOKIE / TWITCH_COOKIE   Cookies to send on discovery");
    println!("  HEADLESS_M3U8_SCRIPT / HEADLESS_M3U8_SCRIPT_TIKTOK   Override Playwright scripts");
    println!("  WAKE_REFRACTORY_SECS     Cooldown between wake detections (default 12)");
    println!("  WAKE_BUFFER_HEADROOM_SECS   Extra buffer headroom for wake timing (default 20)");
    println!("  WAKE_BUFFER_RESTART_SECS    Restart if buffer exceeds seconds (default 500, 0 disables)");
    println!("  WAKE_NO_WORDS_SECS       Restart if wake worker stalls for seconds (default 300, 0 disables)");
    println!("  SKIP_CLIP_SAVE           If set to 1/true, skip writing clips");
    println!("  WHISPER_MODEL            Path to whisper model (default auto)");
    println!("  WHISPER_GPU              Enable GPU for live wake (default true)");
    println!("  WHISPER_CLIP_GPU         Enable GPU for clip transcription (default false)");
    println!("  WHISPER_MIN_FREE_VRAM_MB Min free VRAM before using whisper GPU (default 2048)");
    println!("  WHISPER_ISOLATE          Run live wake in a helper process (default false)");
    println!("  WHISPER_WORKER_STATUS_MS Poll interval for wake worker status (default 500ms)");
    println!("  FFMPEG_ENCODER / FFMPEG_HWACCEL / FFMPEG_HWACCEL_DEVICE   Encoder/accel knobs");
    println!("  FFMPEG_HWACCEL_FALLBACK  Fallback hwaccel (e.g. d3d11va, cuda, none; default none)");
    println!("  FFMPEG_MIN_FREE_VRAM_MB  Min free VRAM before using GPU encode (default 512)");
    println!("  FFMPEG_ENCODE_TIMEOUT_SECS   Hard cap for ffmpeg encode wall time (seconds)");
    println!("  LOG_M3U8_HEADERS         Log request headers when fetching playlists");
    println!("  TWITCH_CLIENT_ID         Twitch Client-ID for playback token (default web client)");
    println!("  TWITCH_OAUTH_TOKEN / TWITCH_AUTH_TOKEN   Twitch OAuth token for gated streams (optional)");
    println!("");
    println!("Notes:");
    println!("  - Main path: page URL -> HLS discovery (TikTok HTTP first, headless fallback) -> wake detection -> clip.");
    println!("  - Demo modes: buffer-only, HLS buffer demo, or mic wake demo for quick sanity checks.");
    println!("  - Live config file supports KEY=VALUE, KEY: VALUE, or flag lines like clip-face-debug.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_to_flag_lowercases_and_dashes() {
        assert_eq!(env_to_flag("CLIP_FACE_RATIO"), "clip-face-ratio");
        assert_eq!(env_to_flag("WHISPER_GPU"), "whisper-gpu");
        assert_eq!(env_to_flag("FFMPEG_MIN_FREE_VRAM_MB"), "ffmpeg-min-free-vram-mb");
    }

    #[test]
    fn parse_cli_collects_positionals_and_flags() {
        let args: Vec<String> = ["https://kick.com/test", "--phrase", "hello", "--no-log-raw-wake"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let parsed = parse_cli_args(&args).unwrap();
        assert_eq!(parsed.positionals, vec!["https://kick.com/test"]);
        assert_eq!(parsed.override_phrase.as_deref(), Some("hello"));
        assert!(!parsed.log_raw_wake);
        assert!(parsed.log_raw_wake_set);
    }

    #[test]
    fn parse_cli_translates_env_flags() {
        let args: Vec<String> = ["--clip-face-ratio=0.42", "--clip-detect"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let parsed = parse_cli_args(&args).unwrap();
        assert!(parsed
            .env_overrides
            .iter()
            .any(|(k, v)| k == "CLIP_FACE_RATIO" && v == "0.42"));
        assert!(parsed
            .env_overrides
            .iter()
            .any(|(k, v)| k == "CLIP_DETECT" && v == "1"));
    }
}
