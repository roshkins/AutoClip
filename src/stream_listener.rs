use std::env;
use std::io::{BufRead, BufReader, IsTerminal, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};

fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn ansi_alert_enabled() -> bool {
    if env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if let Ok(value) = env::var("CLIP_COLOR") {
        if let Some(enabled) = parse_bool_env(&value) {
            return enabled;
        }
    }
    std::io::stderr().is_terminal()
}

fn format_wake_alert(message: &str) -> String {
    if ansi_alert_enabled() {
        format!("\x1b[1;31m{message}\x1b[0m")
    } else {
        message.to_string()
    }
}

/// Start a background listener that invokes whisper.cpp `stream` binary with VAD and watches stdout for the wake phrase.
pub fn start_stream_listener(
    stream_exe: &Path,
    model_path: &Path,
    phrases: Vec<String>,
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
) -> Result<()> {
    if phrases.is_empty() {
        anyhow::bail!("no phrases provided for stream listener");
    }
    let phrases: Vec<String> = phrases.into_iter().map(|p| normalize(&p)).collect();
    let stream_exe = stream_exe.to_path_buf();
    let model_path = model_path.to_path_buf();

    std::thread::spawn(move || {
        if let Err(err) = run_stream_listener(&stream_exe, &model_path, &phrases, log_raw, stop, fired) {
            eprintln!("stream listener error: {err:#}");
        }
    });

    Ok(())
}

fn run_stream_listener(
    stream_exe: &Path,
    model_path: &Path,
    phrases: &[String],
    log_raw: bool,
    stop: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
) -> Result<()> {
    if !stream_exe.exists() {
        anyhow::bail!("stream executable not found at {} (set WHISPER_STREAM_EXE)", stream_exe.display());
    }
    if !model_path.exists() {
        anyhow::bail!("whisper model not found at {}", model_path.display());
    }

    let mut child = spawn_stream(stream_exe, model_path)?;

    // whisper-stream emits transcripts on stderr; capture that for matching.
    let pipe: Box<dyn Read + Send> = if let Some(out) = child.stdout.take() {
        Box::new(out)
    } else if let Some(err) = child.stderr.take() {
        Box::new(err)
    } else {
        anyhow::bail!("failed to capture stream stdout/stderr");
    };
    let mut reader = BufReader::new(pipe);
    let mut line = String::new();

    while !fired.load(Ordering::Relaxed) {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break; // process ended
        }
        let norm = normalize(&line);
        if log_raw && !norm.is_empty() {
            println!("stream raw: {}", line.trim());
            println!("stream norm: {norm}");
        }
        if norm.is_empty() {
            continue;
        }
        if matches_phrase(&norm, phrases) {
            let already = fired.swap(true, Ordering::Relaxed);
            if !already {
                let msg = format!("wake phrase detected via stream: {norm}");
                eprintln!("{}", format_wake_alert(&msg));
            }
        }
    }

    let _ = child.kill();
    Ok(())
}

fn spawn_stream(stream_exe: &Path, model_path: &Path) -> Result<Child> {
    let mut cmd = Command::new(stream_exe);
    cmd.arg("-m")
        .arg(model_path)
        .arg("--step").arg("1000")
        .arg("--length").arg("8000")
        .arg("--keep").arg("500")
        .arg("--vad-thold").arg("0.6")
        .arg("--language").arg("en")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = cmd.spawn().context("spawning whisper stream")?;
    Ok(child)
}

fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_space = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        }
    }
    out.trim().to_string()
}

fn matches_phrase(transcript: &str, phrases: &[String]) -> bool {
    phrases.iter().any(|p| transcript.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize("  Clip, THAT! now"), "clip that now");
    }

    #[test]
    fn matches_contains() {
        let phrases = vec!["clip that".to_string()];
        assert!(matches_phrase("please clip that now", &phrases));
        assert!(!matches_phrase("nothing relevant", &phrases));
    }
}
