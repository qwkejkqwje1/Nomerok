//! Видео и IP-камеры через ffmpeg: он декодирует любой формат и поток
//! (rtsp/http) и отдаёт кадры картинками PPM в канал — размер кадра записан
//! в заголовке каждой картинки, поэтому ffprobe не нужен.

use crate::imgops::{self, Rgb};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[cfg(windows)]
const NO_WINDOW: u32 = 0x0800_0000; // CREATE_NO_WINDOW

fn cmd(exe: &Path) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(NO_WINDOW);
    }
    c
}

/// ffmpeg рядом с программой, в подпапке ffmpeg\bin или в PATH.
pub fn find_ffmpeg(exe_dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    let mut c = vec![
        exe_dir.join(name),
        exe_dir.join("ffmpeg").join("bin").join(name),
        exe_dir.join("ffmpeg").join(name),
    ];
    if let Some(path) = std::env::var_os("PATH") {
        c.extend(std::env::split_paths(&path).map(|d| d.join(name)));
    }
    c.into_iter().find(|p| p.is_file())
}

pub fn is_stream(s: &str) -> bool {
    s.contains("://")
}

pub struct Frames {
    child: Child,
    out: BufReader<ChildStdout>,
    pub index: u64,
    pub err_tail: std::sync::Arc<std::sync::Mutex<String>>,
}

impl Frames {
    /// Кадры с частотой fps (ffmpeg сам прореживает и не тратит время на лишние).
    pub fn open(ffmpeg: &Path, input: &str, fps: f32) -> Result<Frames, String> {
        let mut c = cmd(ffmpeg);
        c.args(["-hide_banner", "-nostdin", "-loglevel", "error"]);
        if input.starts_with("rtsp://") || input.starts_with("rtsps://") {
            c.args(["-rtsp_transport", "tcp", "-timeout", "10000000"]);
        }
        if is_stream(input) {
            c.args(["-fflags", "nobuffer", "-flags", "low_delay"]);
        }
        c.args(["-i", input, "-an", "-sn", "-vf"])
            .arg(format!("fps={fps}"))
            .args(["-f", "image2pipe", "-vcodec", "ppm", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = c
            .spawn()
            .map_err(|e| format!("не удалось запустить ffmpeg: {e}"))?;
        let out = BufReader::with_capacity(1 << 20, child.stdout.take().unwrap());
        let err_tail = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        if let Some(se) = child.stderr.take() {
            let t = err_tail.clone();
            std::thread::spawn(move || {
                for l in BufReader::new(se).lines().map_while(Result::ok) {
                    let mut s = t.lock().unwrap();
                    s.push_str(&l);
                    s.push('\n');
                    if s.len() > 4000 {
                        let cut = s.len() - 2000;
                        let cut = (cut..s.len()).find(|&i| s.is_char_boundary(i)).unwrap_or(0);
                        *s = s[cut..].to_string();
                    }
                }
            });
        }
        Ok(Frames {
            child,
            out,
            index: 0,
            err_tail,
        })
    }

    fn token(&mut self) -> Option<String> {
        let mut s = String::new();
        let mut b = [0u8; 1];
        loop {
            if self.out.read(&mut b).ok()? == 0 {
                return None;
            }
            let c = b[0];
            if c == b'#' {
                let mut junk = Vec::new();
                self.out.read_until(b'\n', &mut junk).ok()?;
                continue;
            }
            if c.is_ascii_whitespace() {
                if s.is_empty() {
                    continue;
                }
                return Some(s);
            }
            s.push(c as char);
        }
    }

    /// Следующий кадр или None, когда видео кончилось / поток оборвался.
    pub fn next_frame(&mut self) -> Option<Rgb> {
        if self.token()? != "P6" {
            return None;
        }
        let w: u32 = self.token()?.parse().ok()?;
        let h: u32 = self.token()?.parse().ok()?;
        let maxv: u32 = self.token()?.parse().ok()?;
        if maxv != 255 || w == 0 || h == 0 || w > 16384 || h > 16384 {
            return None;
        }
        let mut data = vec![0u8; (w * h * 3) as usize];
        self.out.read_exact(&mut data).ok()?;
        self.index += 1;
        imgops::from_raw(w, h, data)
    }

    pub fn errors(&self) -> String {
        self.err_tail.lock().unwrap().trim().to_string()
    }
}

impl Drop for Frames {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Запись видео с рамками (H.264, если есть в сборке ffmpeg, иначе MPEG-4).
pub struct Writer {
    child: Child,
    stdin: Option<ChildStdin>,
    w: u32,
    h: u32,
}

impl Writer {
    pub fn open(ffmpeg: &Path, out: &Path, w: u32, h: u32, fps: f32) -> Result<Writer, String> {
        // чётные размеры — требование yuv420p
        let (w, h) = (w & !1, h & !1);
        let mut child = cmd(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"])
            .args(["-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &format!("{fps}")])
            .args(["-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast"])
            .args(["-crf", "26"])
            .arg(out)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("не удалось запустить ffmpeg: {e}"))?;
        let stdin = child.stdin.take();
        Ok(Writer { child, stdin, w, h })
    }

    pub fn push(&mut self, img: &Rgb) -> bool {
        let Some(s) = self.stdin.as_mut() else {
            return false;
        };
        let frame = if img.width() != self.w || img.height() != self.h {
            image::imageops::crop_imm(img, 0, 0, self.w, self.h).to_image()
        } else {
            img.clone()
        };
        s.write_all(frame.as_raw()).is_ok()
    }

    pub fn finish(mut self) -> bool {
        drop(self.stdin.take());
        self.child.wait().map(|s| s.success()).unwrap_or(false)
    }
}
