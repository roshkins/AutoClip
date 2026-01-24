#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(test)]
fn env_mutex() -> &'static Mutex<()> {
    static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_MUTEX.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
fn lock_env() -> MutexGuard<'static, ()> {
    match env_mutex().lock() {
        Ok(guard) => guard,
        Err(err) => err.into_inner(),
    }
}

#[cfg(test)]
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(String, Option<String>)>,
}

#[cfg(test)]
impl EnvGuard {
    pub fn new() -> Self {
        Self {
            _lock: lock_env(),
            saved: Vec::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        if !self.saved.iter().any(|(k, _)| k == key) {
            self.saved.push((key.to_string(), std::env::var(key).ok()));
        }
        std::env::set_var(key, value);
    }

    pub fn remove(&mut self, key: &str) {
        if !self.saved.iter().any(|(k, _)| k == key) {
            self.saved.push((key.to_string(), std::env::var(key).ok()));
        }
        std::env::remove_var(key);
    }
}

#[cfg(test)]
impl Drop for EnvGuard {
    fn drop(&mut self) {
        while let Some((key, val)) = self.saved.pop() {
            match val {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
