use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct ResizePolicy {
    pub min_width: u32,
    pub min_height: u32,
    pub aspect_ratio: f32,
}

impl ResizePolicy {
    pub fn constrain(&self, width: u32, height: u32) -> (u32, u32) {
        let mut constrained_w = width.max(self.min_width).max(1);
        let mut constrained_h = height.max(self.min_height).max(1);

        let current_ratio = (constrained_w as f32) / (constrained_h as f32);
        if current_ratio > self.aspect_ratio {
            constrained_w = ((constrained_h as f32) * self.aspect_ratio)
                .round()
                .max(1.0) as u32;
        } else {
            constrained_h = ((constrained_w as f32) / self.aspect_ratio)
                .round()
                .max(1.0) as u32;
        }

        if constrained_w < self.min_width {
            constrained_w = self.min_width;
            constrained_h = ((constrained_w as f32) / self.aspect_ratio)
                .round()
                .max(1.0) as u32;
        }
        if constrained_h < self.min_height {
            constrained_h = self.min_height;
            constrained_w = ((constrained_h as f32) * self.aspect_ratio)
                .round()
                .max(1.0) as u32;
        }

        (constrained_w, constrained_h)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeDebounce {
    pub min_interval: Duration,
    pub epsilon_width: u32,
    pub epsilon_height: u32,
}

impl Default for ResizeDebounce {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(8),
            epsilon_width: 1,
            epsilon_height: 1,
        }
    }
}

impl ResizeDebounce {
    pub fn should_skip(
        &self,
        last: Option<(u32, u32, Instant)>,
        current_size: (u32, u32),
        now: Instant,
    ) -> bool {
        if let Some((last_w, last_h, last_ts)) = last {
            let width_delta = last_w.abs_diff(current_size.0);
            let height_delta = last_h.abs_diff(current_size.1);
            if width_delta <= self.epsilon_width
                && height_delta <= self.epsilon_height
                && now.duration_since(last_ts) < self.min_interval
            {
                return true;
            }
        }

        false
    }
}

struct HostResizeState {
    policy: ResizePolicy,
    debounce: ResizeDebounce,
    last_host_size: Mutex<(u32, u32)>,
    pending_requested_size: Mutex<Option<(u32, u32)>>,
    last_resize_request: Mutex<Option<(u32, u32, Instant)>>,
}

#[derive(Clone)]
pub struct HostResizeCoordinator {
    state: Arc<HostResizeState>,
}

impl HostResizeCoordinator {
    pub fn new(
        initial_size: (u32, u32),
        policy: ResizePolicy,
        debounce: ResizeDebounce,
    ) -> Self {
        let initial_size = policy.constrain(initial_size.0, initial_size.1);
        Self {
            state: Arc::new(HostResizeState {
                policy,
                debounce,
                last_host_size: Mutex::new(initial_size),
                pending_requested_size: Mutex::new(None),
                last_resize_request: Mutex::new(None),
            }),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        if let Some(pending) = *self
            .state
            .pending_requested_size
            .lock()
            .expect("pending size lock poisoned")
        {
            return pending;
        }

        *self
            .state
            .last_host_size
            .lock()
            .expect("host size lock poisoned")
    }

    pub fn min_size(&self) -> (u32, u32) {
        (self.state.policy.min_width, self.state.policy.min_height)
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.state.policy.aspect_ratio
    }

    pub fn on_host_resized(&self, width: u32, height: u32) -> (u32, u32) {
        let constrained = self.state.policy.constrain(width, height);

        {
            let mut guard = self
                .state
                .last_host_size
                .lock()
                .expect("host size lock poisoned");
            *guard = constrained;
        }

        {
            let mut pending = self
                .state
                .pending_requested_size
                .lock()
                .expect("pending size lock poisoned");
            if pending.is_some() {
                *pending = None;
            }
        }

        constrained
    }

    pub fn begin_request_from_ui(
        &self,
        width: u32,
        height: u32,
        now: Instant,
    ) -> Option<(u32, u32)> {
        let constrained = self.state.policy.constrain(width, height);

        {
            let mut last = self
                .state
                .last_resize_request
                .lock()
                .expect("last resize request lock poisoned");
            if self
                .state
                .debounce
                .should_skip(*last, constrained, now)
            {
                return None;
            }

            *last = Some((constrained.0, constrained.1, now));
        }

        {
            let mut pending = self
                .state
                .pending_requested_size
                .lock()
                .expect("pending size lock poisoned");
            *pending = Some(constrained);
        }

        Some(constrained)
    }

    pub fn reject_pending_request(&self) {
        let mut pending = self
            .state
            .pending_requested_size
            .lock()
            .expect("pending size lock poisoned");
        *pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{HostResizeCoordinator, ResizeDebounce, ResizePolicy};
    use std::time::{Duration, Instant};

    #[test]
    fn policy_constrains_to_minimum_and_aspect() {
        let policy = ResizePolicy {
            min_width: 760,
            min_height: 460,
            aspect_ratio: 980.0 / 654.0,
        };

        let size = policy.constrain(10, 10);
        assert!(size.0 >= 760);
        assert!(size.1 >= 460);

        let ratio = size.0 as f32 / size.1 as f32;
        assert!((ratio - policy.aspect_ratio).abs() < 0.02);
    }

    #[test]
    fn debounce_skips_only_small_fast_changes() {
        let debounce = ResizeDebounce::default();
        let now = Instant::now();
        let last = Some((1000, 700, now));

        assert!(debounce.should_skip(last, (1001, 701), now + Duration::from_millis(2)));
        assert!(!debounce.should_skip(last, (1030, 730), now + Duration::from_millis(2)));
        assert!(!debounce.should_skip(last, (1001, 701), now + Duration::from_millis(20)));
    }

    #[test]
    fn coordinator_tracks_pending_and_host_commit() {
        let policy = ResizePolicy {
            min_width: 760,
            min_height: 460,
            aspect_ratio: 980.0 / 654.0,
        };
        let coordinator = HostResizeCoordinator::new((980, 654), policy, ResizeDebounce::default());
        let now = Instant::now();

        let requested = coordinator
            .begin_request_from_ui(1200, 700, now)
            .expect("first request should pass");
        assert_eq!(coordinator.size(), requested);

        let committed = coordinator.on_host_resized(requested.0, requested.1);
        assert_eq!(committed, requested);
        assert_eq!(coordinator.size(), committed);
    }
}
