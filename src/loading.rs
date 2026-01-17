use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub struct LoadingTicker {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl LoadingTicker {
    pub fn start(label: &'static str, interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let label_owned = label.to_string();
        let handle = thread::spawn(move || {
            let start = Instant::now();
            loop {
                thread::sleep(interval);
                if stop_clone.load(Ordering::Relaxed) {
                    break;
                }
                eprintln!("{}... {:.1}s", label_owned, start.elapsed().as_secs_f32());
            }
        });
        eprintln!("{label}...");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for LoadingTicker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
