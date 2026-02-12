use std::{mem, ptr::{null, null_mut}, sync::{atomic::{AtomicBool, Ordering}, Arc}};

use uuid::Uuid;
use windows::{Win32::{UI::{WindowsAndMessaging::{WNDCLASSW, CS_OWNDC, DefWindowProcW, HICON, HCURSOR, RegisterClassW, CreateWindowExW, GetMessageW, TranslateMessage, DispatchMessageW, WM_CHAR, WM_IME_CHAR, WM_IME_COMPOSITION, PostMessageW, SetWindowLongPtrW, GWLP_USERDATA, GetWindowLongPtrW, DestroyWindow, UnregisterClassW, WS_CHILD, WM_KEYDOWN, WM_KEYUP}, Input::{KeyboardAndMouse::SetFocus, Ime::{GCS_RESULTSTR, ImmGetCompositionStringW, ImmGetContext, ImmReleaseContext}}}, Graphics::Gdi::HBRUSH, Foundation::{HWND, WPARAM, LPARAM, LRESULT}}, core::PCWSTR};
use windows_core::BOOL;

use crate::error::Error;

use super::{to_wstr, PLUGIN_HINSTANCE, WM_USER_CHAR, WM_USER_KEY_DOWN, WM_USER_KEY_UP};

pub struct MessageWindow {
    hwnd: usize,
    main_window_hwnd: usize,
    window_class: u16,
}

impl MessageWindow {
    pub fn new(main_window_hwnd: HWND) -> Result<Self, Error> {
        let class_name = to_wstr("plugin-canvas-message-window-".to_string() + &Uuid::new_v4().simple().to_string());
        let window_name = to_wstr("Message window");

        let window_class_attributes = WNDCLASSW {
            style: CS_OWNDC,
            lpfnWndProc: Some(wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: PLUGIN_HINSTANCE.with(|hinstance| *hinstance),
            hIcon: HICON(null_mut()),
            hCursor: HCURSOR(null_mut()),
            hbrBackground: HBRUSH(null_mut()),
            lpszMenuName: PCWSTR(null()),
            lpszClassName: PCWSTR(class_name.as_ptr()),
        };

        let window_class = unsafe { RegisterClassW(&window_class_attributes) };
        if window_class == 0 {
            return Err(Error::PlatformError("Failed to register window class".into()));
        }

        let hwnd = unsafe { CreateWindowExW(
            Default::default(),
            PCWSTR(window_class as _),
            PCWSTR(window_name.as_ptr() as _),
            WS_CHILD,
            0,
            0,
            0,
            0,
            Some(main_window_hwnd),
            None,
            Some(PLUGIN_HINSTANCE.with(|hinstance| *hinstance)),
            None,
        ).unwrap() };

        unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, main_window_hwnd.0 as _) };

        Ok(Self {
            hwnd: hwnd.0 as _,
            main_window_hwnd: main_window_hwnd.0 as _,
            window_class,
        })
    }

    pub fn run(&self, running: Arc<AtomicBool>) {
        unsafe {
            let hwnd = HWND(self.hwnd as _);
            let mut msg = mem::zeroed();

            while running.load(Ordering::Acquire) {
                match GetMessageW(&mut msg, Some(hwnd), 0, 0) {
                    BOOL(-1) => {
                        panic!()
                    }

                    BOOL(0) => {
                        return;
                    }

                    _ => {}
                }

                // We can ignore the return value
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }    
        }
    }

    pub fn set_focus(&self, focus: bool) {
        let hwnd = HWND(if focus {
            self.hwnd
        } else {
            self.main_window_hwnd
        } as _);

        let _ = unsafe { SetFocus(Some(hwnd)) };
    }
}

impl Drop for MessageWindow {
    fn drop(&mut self) {
        unsafe {
            // It's ok if this fails; window might already be deleted if our parent window was deleted
            DestroyWindow(HWND(self.hwnd as _)).ok();
            UnregisterClassW(PCWSTR(self.window_class as _), Some(PLUGIN_HINSTANCE.with(|hinstance| *hinstance))).unwrap();
        }
    }
}

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let main_window_hwnd = unsafe { HWND(GetWindowLongPtrW(hwnd, GWLP_USERDATA) as _) };

    fn post_utf16_to_main_window(main_window_hwnd: HWND, utf16_units: &[u16]) {
        for unit in utf16_units {
            if *unit == 0 {
                continue;
            }

            unsafe {
                let _ = PostMessageW(
                    Some(main_window_hwnd),
                    WM_USER_CHAR,
                    WPARAM(*unit as usize),
                    LPARAM(0),
                );
            }
        }
    }

    fn post_ime_result_to_main_window(hwnd: HWND, main_window_hwnd: HWND) {
        let himc = unsafe { ImmGetContext(hwnd) };
        if himc.0.is_null() {
            return;
        }

        let bytes_required = unsafe { ImmGetCompositionStringW(himc, GCS_RESULTSTR, None, 0) };
        if bytes_required > 0 {
            let bytes_required = bytes_required as usize;
            let unit_count = bytes_required / std::mem::size_of::<u16>();
            if unit_count > 0 {
                let mut utf16 = vec![0u16; unit_count];
                let copied = unsafe {
                    ImmGetCompositionStringW(
                        himc,
                        GCS_RESULTSTR,
                        Some(utf16.as_mut_ptr().cast()),
                        bytes_required as u32,
                    )
                };

                if copied > 0 {
                    let copied_units = (copied as usize) / std::mem::size_of::<u16>();
                    let copied_units = copied_units.min(utf16.len());
                    post_utf16_to_main_window(main_window_hwnd, &utf16[..copied_units]);
                }
            }
        }

        let _ = unsafe { ImmReleaseContext(hwnd, himc) };
    }

    match msg {
        WM_CHAR | WM_IME_CHAR => {
            let _ = unsafe { PostMessageW(Some(main_window_hwnd), WM_USER_CHAR, wparam, lparam) };
            LRESULT(0)
        },

        WM_IME_COMPOSITION => {
            let composition_flags = lparam.0 as u32;
            if composition_flags & GCS_RESULTSTR.0 != 0 {
                post_ime_result_to_main_window(hwnd, main_window_hwnd);
                return LRESULT(0);
            }

            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }

        WM_KEYDOWN => {
            let _ = unsafe {
                PostMessageW(
                    Some(main_window_hwnd),
                    WM_USER_KEY_DOWN,
                    WPARAM(wparam.0),
                    LPARAM(0),
                )
            };
            
            LRESULT(0)
        }

        WM_KEYUP => {
            let _ = unsafe {
                PostMessageW(
                    Some(main_window_hwnd),
                    WM_USER_KEY_UP,
                    WPARAM(wparam.0),
                    LPARAM(0),
                )
            };
            
            LRESULT(0)
        }

        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }
}
