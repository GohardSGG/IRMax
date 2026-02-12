//! Resizable window wrapper for Egui editor.

use egui_baseview::egui::emath::GuiRounding;

use crate::egui::{pos2, Area, Context, Id, Order, Pos2, Response, Sense, Ui, Vec2};
use crate::EguiState;

#[derive(Debug, Clone, Copy)]
struct ResizeDragState {
    last_sent_size: (u32, u32),
    drag_start_pointer: Option<Pos2>,
    drag_start_size: Option<(u32, u32)>,
}

#[inline]
fn rounded_size(size: Vec2) -> (u32, u32) {
    let width = size.x.max(1.0).round() as u32;
    let height = size.y.max(1.0).round() as u32;
    (width, height)
}

/// Adds a corner to the plugin window that can be dragged in order to resize it.
/// Resizing happens through plugin API, hence a custom implementation is needed.
pub struct ResizableWindow {
    id: Id,
    min_size: Vec2,
    /// If set, the window will maintain this aspect ratio (width / height) when resizing.
    aspect_ratio: Option<f32>,
}

impl ResizableWindow {
    pub fn new(id_source: impl std::hash::Hash) -> Self {
        Self {
            id: Id::new(id_source),
            min_size: Vec2::splat(16.0),
            aspect_ratio: None,
        }
    }

    /// Won't shrink to smaller than this
    #[inline]
    pub fn min_size(mut self, min_size: impl Into<Vec2>) -> Self {
        self.min_size = min_size.into();
        self
    }

    /// Lock the window to a fixed aspect ratio (width / height).
    /// When the user drags the resize corner, the window will maintain this ratio.
    #[inline]
    pub fn with_aspect_ratio(mut self, ratio: f32) -> Self {
        self.aspect_ratio = Some(ratio);
        self
    }

    /// Apply aspect ratio constraint by projecting to a 1D scale axis.
    ///
    /// This guarantees that while dragging, width/height always move on the same
    /// proportional line instead of switching between width-driven and height-driven
    /// branches, eliminating ratio jitter.
    fn apply_aspect_ratio_from_drag(&self, start_size: (u32, u32), delta: Vec2) -> Vec2 {
        if let Some(ratio) = self.aspect_ratio {
            let axis = Vec2::new(ratio, 1.0);
            let axis_len_sq = axis.x * axis.x + axis.y * axis.y;

            let scale_delta = if axis_len_sq > 0.0 {
                (delta.x * axis.x + delta.y * axis.y) / axis_len_sq
            } else {
                0.0
            };

            let start_h = start_size.1 as f32;
            let mut next_h = start_h + scale_delta;

            let min_h_by_size = self.min_size.y;
            let min_h_by_width = self.min_size.x / ratio;
            let min_h = min_h_by_size.max(min_h_by_width).max(1.0);
            next_h = next_h.max(min_h);

            let next_w = (next_h * ratio).max(self.min_size.x).max(1.0);
            Vec2::new(next_w, next_h)
        } else {
            Vec2::new(start_size.0 as f32 + delta.x, start_size.1 as f32 + delta.y)
                .max(self.min_size)
        }
    }

    /// Show the resizable window. The closure receives the Context instead of a Ui,
    /// allowing the use of egui's panel system (TopBottomPanel, SidePanel, CentralPanel).
    pub fn show<R>(
        self,
        context: &Context,
        egui_state: &EguiState,
        add_contents: impl FnOnce(&Context) -> R,
    ) -> R {
        // 1. Let the user add their panels first
        let ret = add_contents(context);

        // Keep the wrapper-side constraint in sync with the UI-side minimum size.
        egui_state.set_min_size_hint(rounded_size(self.min_size));

        // 2. Draw the floating resize handle in the bottom-right corner using Area
        let fallback_screen_rect = context.viewport_rect();
        let screen_rect = context.input(|input| {
            input
                .viewport()
                .inner_rect
                .unwrap_or(fallback_screen_rect)
        });
        let corner_size = 16.0;
        let corner_pos = screen_rect.max - Vec2::splat(corner_size);
        let drag_state_id = self.id.with("resize_drag_state");

        Area::new(self.id.with("resize_corner"))
            .fixed_pos(corner_pos)
            .order(Order::Foreground) // Ensure it's on top
            .show(context, |ui| {
                let (_rect, response) =
                    ui.allocate_exact_size(Vec2::splat(corner_size), Sense::drag());

                let pointer_pos = response
                    .interact_pointer_pos()
                    .or_else(|| context.input(|input| input.pointer.latest_pos()));

                if let Some(pointer_pos) = pointer_pos {
                    let mut drag_state = context
                        .memory(|mem| mem.data.get_temp::<ResizeDragState>(drag_state_id))
                        .unwrap_or(ResizeDragState {
                            last_sent_size: egui_state.size.load(),
                            drag_start_pointer: None,
                            drag_start_size: None,
                        });

                    if response.drag_started() {
                        drag_state.last_sent_size = (0, 0);
                        drag_state.drag_start_pointer = Some(pointer_pos);
                        drag_state.drag_start_size = Some(egui_state.size.load());
                    }

                    let (start_pointer, start_size) = match (
                        drag_state.drag_start_pointer,
                        drag_state.drag_start_size,
                    ) {
                        (Some(p), Some(s)) => (p, s),
                        _ => {
                            let fallback_size = egui_state.size.load();
                            drag_state.drag_start_pointer = Some(pointer_pos);
                            drag_state.drag_start_size = Some(fallback_size);
                            (pointer_pos, fallback_size)
                        }
                    };

                    // Stable drag calculation: use delta from drag-start + start window size,
                    // instead of recomputing from current pointer-to-screen min every frame.
                    let delta = pointer_pos - start_pointer;
                    let desired_size = self.apply_aspect_ratio_from_drag(start_size, delta);
                    let desired_size_rounded = rounded_size(desired_size);

                    if response.dragged() {
                        // Only update the requested size; keep host handshake state intact.
                        if desired_size_rounded != egui_state.size.load()
                            && drag_state.last_sent_size != desired_size_rounded
                        {
                            egui_state.set_requested_size(desired_size_rounded);
                            drag_state.last_sent_size = desired_size_rounded;
                            context.request_repaint();
                        }
                    }

                    if response.drag_stopped() {
                        // Always send the final exact size once at the end of drag to guarantee convergence.
                        if desired_size_rounded != egui_state.size.load()
                            && drag_state.last_sent_size != desired_size_rounded
                        {
                            egui_state.set_requested_size(desired_size_rounded);
                            egui_state.mark_resize_commit_urgent();
                        }
                        drag_state.last_sent_size = desired_size_rounded;
                        drag_state.drag_start_pointer = None;
                        drag_state.drag_start_size = None;
                    }

                    context.memory_mut(|mem| {
                        mem.data.insert_temp(drag_state_id, drag_state);
                    });
                }

                // Draw the resize corner pattern
                paint_resize_corner(ui, &response);
            });

        ret
    }
}

pub fn paint_resize_corner(ui: &Ui, response: &Response) {
    let stroke = ui.style().interact(response).fg_stroke;

    let painter = ui.painter();
    let rect = response.rect.translate(-Vec2::splat(2.0)); // move away from the corner
    let cp = rect.max.round_to_pixels(painter.pixels_per_point());

    let mut w = 2.0;

    while w <= rect.width() && w <= rect.height() {
        painter.line_segment([pos2(cp.x - w, cp.y), pos2(cp.x, cp.y - w)], stroke);
        w += 4.0;
    }
}
