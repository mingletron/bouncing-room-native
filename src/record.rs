//! Recording a clip: frames in, an animated GIF out.
//!
//! The frames come from [`crate::gfx::Gfx::render_to_pixels`], so a clip is the
//! 3D scene alone — no panel, whatever the window is showing. Encoding a GIF
//! means quantising every frame to 256 colours, which is far too slow to do on
//! the thread that draws; frames are handed to a worker over a channel and the
//! window never waits for it.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;

/// Frames per second of a clip. GIF delays are whole hundredths of a second,
/// and 25 fps is the fastest rate that is one exactly (4).
pub const FPS: f32 = 25.0;
const FRAME_DELAY_CS: u16 = 4;
/// Clip width in pixels; the height follows the window's shape.
pub const CLIP_WIDTH: u32 = 640;
/// A clip stops itself after this long. A forgotten recording should not fill
/// a disk.
pub const MAX_SECONDS: f32 = 20.0;

pub struct Recorder {
    frames_out: Sender<Vec<u8>>,
    worker: JoinHandle<Result<(), String>>,
    path: PathBuf,
    width: u32,
    height: u32,
    /// Time owed to the clip: grows with every frame drawn, shrinks by one
    /// frame period for every frame captured.
    owed: f32,
    seconds: f32,
    frames: usize,
}

/// The clip size for a window of this shape: [`CLIP_WIDTH`] wide, with an even
/// height that keeps the window's aspect ratio.
pub fn clip_size(window_width: u32, window_height: u32) -> (u32, u32) {
    let aspect = window_height.max(1) as f32 / window_width.max(1) as f32;
    let height = ((CLIP_WIDTH as f32 * aspect).round() as u32).clamp(64, 1024);
    (CLIP_WIDTH, height & !1)
}

/// Somewhere sensible to put a clip: the user's video folder when there is one
/// (`~/Videos` on Linux, `~/Movies` on macOS), otherwise the working directory.
pub fn default_clip_path(stamp: u64) -> PathBuf {
    let name = format!("bouncing-room-{stamp}.gif");
    if let Some(home) = std::env::var_os("HOME") {
        for folder in ["Videos", "Movies"] {
            let dir = Path::new(&home).join(folder);
            if dir.is_dir() {
                return dir.join(name);
            }
        }
    }
    PathBuf::from(name)
}

impl Recorder {
    pub fn start(path: PathBuf, width: u32, height: u32) -> Result<Recorder, String> {
        let (w, h) = (
            u16::try_from(width).map_err(|_| "clip too wide for a GIF".to_owned())?,
            u16::try_from(height).map_err(|_| "clip too tall for a GIF".to_owned())?,
        );
        let file = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (frames_out, frames_in) = mpsc::channel::<Vec<u8>>();
        let worker = std::thread::spawn(move || -> Result<(), String> {
            let mut encoder = gif::Encoder::new(BufWriter::new(file), w, h, &[])
                .map_err(|e| e.to_string())?;
            encoder
                .set_repeat(gif::Repeat::Infinite)
                .map_err(|e| e.to_string())?;
            // Runs until the sender is dropped, which is how a clip is ended.
            for mut rgba in frames_in {
                // Speed 10 of 1..30: the quantiser's own middle setting.
                let mut frame = gif::Frame::from_rgba_speed(w, h, &mut rgba, 10);
                frame.delay = FRAME_DELAY_CS;
                encoder.write_frame(&frame).map_err(|e| e.to_string())?;
            }
            Ok(())
        });
        Ok(Recorder {
            frames_out,
            worker,
            path,
            width,
            height,
            // Owed one frame from the start, so the first frame drawn is captured.
            owed: 1.0 / FPS,
            seconds: 0.0,
            frames: 0,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn seconds(&self) -> f32 {
        self.seconds
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Account for `dt` seconds of drawing; true when a frame is now due.
    pub fn due(&mut self, dt: f32) -> bool {
        self.owed += dt;
        self.seconds += dt;
        self.owed >= 1.0 / FPS
    }

    /// True once the clip has run as long as one is allowed to.
    pub fn full(&self) -> bool {
        self.seconds >= MAX_SECONDS
    }

    /// Hand over one frame of tightly packed RGBA, `width * height * 4` bytes.
    pub fn push(&mut self, rgba: Vec<u8>) {
        // Never more than one frame behind: if drawing stalled for a second the
        // clip skips that second rather than replaying it as a burst.
        self.owed = (self.owed - 1.0 / FPS).min(1.0 / FPS);
        if rgba.len() == (self.width * self.height * 4) as usize && self.frames_out.send(rgba).is_ok() {
            self.frames += 1;
        }
    }

    /// Stop taking frames. The worker finishes what it was sent and closes the
    /// file; the handle says when, so nothing has to wait for it.
    pub fn finish(self) -> (JoinHandle<Result<(), String>>, PathBuf) {
        drop(self.frames_out);
        (self.worker, self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_keeps_the_windows_shape_and_an_even_height() {
        assert_eq!(clip_size(900, 600), (640, 426));
        assert_eq!(clip_size(1280, 800), (640, 400));
        let (_, h) = clip_size(1000, 333);
        assert_eq!(h % 2, 0);
    }

    #[test]
    fn frames_are_due_at_the_clip_rate_not_the_display_rate() {
        let dir = std::env::temp_dir().join(format!("bouncing-room-clip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.gif");
        let mut rec = Recorder::start(path.clone(), 64, 64).unwrap();
        let mut captured = 0;
        // One second of a 100 Hz display.
        for _ in 0..100 {
            if rec.due(0.01) {
                rec.push(vec![40u8; 64 * 64 * 4]);
                captured += 1;
            }
        }
        assert!((24..=26).contains(&captured), "captured {captured} frames in a second");
        let (worker, path) = rec.finish();
        worker.join().unwrap().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        let _ = std::fs::remove_dir_all(dir);
    }
}
