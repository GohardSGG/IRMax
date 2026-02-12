use crossbeam_channel::{
    unbounded, Receiver, RecvTimeoutError, Sender, TryRecvError,
};
use irmax_core::MatrixWorker;
use log::info;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

enum WorkerControl {
    Clear,
    Stop,
}

pub struct WorkerThread {
    tx: Option<Sender<MatrixWorker>>,
    control_tx: Option<Sender<WorkerControl>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl WorkerThread {
    pub fn new(logger: Option<std::sync::Arc<crate::logger::InstanceLogger>>) -> Self {
        let (tx, rx) = unbounded::<MatrixWorker>();
        let (control_tx, control_rx) = unbounded::<WorkerControl>();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();

        let handle = thread::spawn(move || {
            info!("IRMax: Worker Thread Started.");
            worker_loop(rx, control_rx, stop_thread, logger);
            info!("IRMax: Worker Thread Stopped.");
        });

        Self {
            tx: Some(tx),
            control_tx: Some(control_tx),
            stop,
            handle: Some(handle),
        }
    }

    pub fn get_sender(&self) -> Sender<MatrixWorker> {
        self.tx
            .as_ref()
            .expect("worker sender missing")
            .clone()
    }

    pub fn clear_current_worker(&self) {
        if let Some(control_tx) = &self.control_tx {
            let _ = control_tx.send(WorkerControl::Clear);
        }
    }
}

impl Drop for WorkerThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(control_tx) = &self.control_tx {
            let _ = control_tx.send(WorkerControl::Stop);
        }
        self.control_tx.take();
        // Close command channel so a blocked recv_timeout loop can exit promptly.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    rx: Receiver<MatrixWorker>,
    control_rx: Receiver<WorkerControl>,
    stop: Arc<AtomicBool>,
    _logger: Option<std::sync::Arc<crate::logger::InstanceLogger>>,
) {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
        };
        // Keep plugin worker responsive without contending with DAW RT-critical scheduling.
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST as i32);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        // Restore high QoS so worker threads are scheduled on performance cores when available.
        let qos_class = libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE;
        libc::pthread_set_qos_class_self_np(qos_class, 0);
    }

    let mut current_worker: Option<MatrixWorker> = None;

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }

        loop {
            match control_rx.try_recv() {
                Ok(WorkerControl::Clear) => {
                    current_worker = None;
                }
                Ok(WorkerControl::Stop) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        if current_worker.is_none() {
            match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(w) => {
                    info!("IRMax: Background Worker Received New Engine Task.");
                    current_worker = Some(w);
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.try_recv() {
                Ok(w) => {
                    info!("IRMax: Background Worker Swapping Engine Task.");
                    current_worker = Some(w);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }

        if stop.load(Ordering::Acquire) {
            break;
        }

        if let Some(w) = &mut current_worker {
            if !w.wait_and_process() {
                let _ = w.wait_for_signal_timeout(Duration::from_millis(1));
            }
        }
    }
}
