//! Нейросети: поиск номеров на кадре (YOLOv9, open-image-models) и чтение
//! символов (fast-plate-ocr), плюс маленькая сеть углов номера (вариант 2
//! выпрямления, обучена нами: eval/train_corners.py). Все модели в формате ONNX,
//! запускаются через ONNX Runtime на процессоре.

use crate::imgops::{self, Rgb};
use crate::plate::{self, Reading};
use crate::rectify::{self, Pt};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

pub type Res<T> = Result<T, String>;

fn e<E: std::fmt::Display>(ctx: &str) -> impl Fn(E) -> String + '_ {
    move |err| format!("{ctx}: {err}")
}

const DET_ONNX: &[u8] = include_bytes!("../models/detector.onnx");
const OCR_ONNX: &[u8] = include_bytes!("../models/ocr.onnx");
const OCR_YAML: &str = include_str!("../models/ocr.yaml");
const CORNERS_ONNX: &[u8] = include_bytes!("../models/corners.onnx");

/// Ищет onnxruntime рядом с программой. Полный путь важен: в System32 у Windows 10
/// лежит своя старая onnxruntime.dll, и её подхватывать нельзя.
pub fn init_runtime(exe_dir: &Path) -> Res<()> {
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    let mut cands = vec![exe_dir.join(name)];
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        cands.insert(0, PathBuf::from(p));
    }
    let lib = cands.into_iter().find(|p| p.exists()).ok_or_else(|| {
        format!(
            "не найден {name}: он должен лежать рядом с программой ({})",
            exe_dir.display()
        )
    })?;
    ort::init_from(lib.display().to_string())
        .with_name("nomerok")
        .commit()
        .map_err(e("не удалось запустить ONNX Runtime"))?;
    Ok(())
}

fn session(bytes: &[u8], threads: usize) -> Res<Session> {
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(threads))
        .and_then(|b| b.commit_from_memory(bytes))
        .map_err(e("не удалось загрузить модель"))
}

/// Модель можно заменить файлом рядом с программой (models\detector.onnx,
/// models\ocr.onnx + models\ocr.yaml) — например, дообученной под свою камеру.
fn model_bytes(over: Option<&Path>, builtin: &'static [u8]) -> Res<std::borrow::Cow<'static, [u8]>> {
    match over {
        Some(p) => std::fs::read(p)
            .map(Into::into)
            .map_err(|er| format!("{}: {er}", p.display())),
        None => Ok(builtin.into()),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Det {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl Det {
    pub fn w(&self) -> f32 {
        self.x2 - self.x1
    }
    pub fn h(&self) -> f32 {
        self.y2 - self.y1
    }
    pub fn iou(&self, o: &Det) -> f32 {
        let ix = (self.x2.min(o.x2) - self.x1.max(o.x1)).max(0.0);
        let iy = (self.y2.min(o.y2) - self.y1.max(o.y1)).max(0.0);
        let i = ix * iy;
        let u = self.w() * self.h() + o.w() * o.h() - i;
        if u <= 0.0 {
            0.0
        } else {
            i / u
        }
    }
}

pub struct Detector {
    s: Session,
    size: u32,
    pub conf: f32,
}

impl Detector {
    pub fn new(over: Option<&Path>, threads: usize, conf: f32) -> Res<Detector> {
        let s = session(&model_bytes(over, DET_ONNX)?, threads)?;
        let size = s
            .inputs
            .first()
            .and_then(|i| i.input_type.tensor_shape().map(|d| d.to_vec()))
            .and_then(|d| d.get(2).copied())
            .filter(|&v| v > 0)
            .unwrap_or(640) as u32;
        Ok(Detector { s, size, conf })
    }

    /// ШАГ 2. Где на кадре номера: кадр вписывается в квадрат 640×640
    /// (letterbox), сеть выдаёт рамки с уверенностью, слабые отбрасываем.
    pub fn detect(&mut self, img: &Rgb) -> Res<Vec<Det>> {
        let (data, r, dw, dh) = imgops::letterbox_chw(img, self.size);
        let n = self.size as i64;
        let t = Tensor::from_array((vec![1i64, 3, n, n], data)).map_err(e("тензор"))?;
        let out = self.s.run(ort::inputs![t]).map_err(e("детектор"))?;
        let (shape, v) = out[0].try_extract_tensor::<f32>().map_err(e("детектор"))?;
        let cols = *shape.last().unwrap_or(&7) as usize;
        let mut dets = Vec::new();
        if cols < 7 {
            return Ok(dets);
        }
        let (w, h) = (img.width() as f32, img.height() as f32);
        for row in v.chunks_exact(cols) {
            let score = row[6];
            if score < self.conf {
                continue;
            }
            let d = Det {
                x1: ((row[1] - dw) / r).clamp(0.0, w),
                y1: ((row[2] - dh) / r).clamp(0.0, h),
                x2: ((row[3] - dw) / r).clamp(0.0, w),
                y2: ((row[4] - dh) / r).clamp(0.0, h),
                score,
            };
            if d.w() >= 4.0 && d.h() >= 2.0 {
                dets.push(d);
            }
        }
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(dets)
    }
}

pub struct Reader {
    s: Session,
    slots: usize,
    alphabet: Vec<char>,
    pad: char,
    w: u32,
    h: u32,
    gray: bool,
    /// Какие виды номеров разрешены (--formats); пусто — все.
    pub allowed: Vec<bool>,
}

fn yaml_val<'a>(y: &'a str, key: &str) -> Option<&'a str> {
    y.lines().find_map(|l| {
        let l = l.trim();
        let rest = l.strip_prefix(key)?.trim_start().strip_prefix(':')?;
        Some(rest.split(" #").next()?.trim().trim_matches(|c| c == '\'' || c == '"'))
    })
}

impl Reader {
    pub fn new(over: Option<(&Path, &Path)>, threads: usize) -> Res<Reader> {
        let (bytes, yaml) = match over {
            Some((m, c)) => (
                model_bytes(Some(m), OCR_ONNX)?,
                std::fs::read_to_string(c).map_err(|er| format!("{}: {er}", c.display()))?,
            ),
            None => (OCR_ONNX.into(), OCR_YAML.to_string()),
        };
        let num = |k: &str, d: usize| yaml_val(&yaml, k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let alphabet: Vec<char> = yaml_val(&yaml, "alphabet")
            .ok_or("в ocr.yaml нет alphabet")?
            .chars()
            .collect();
        let pad = yaml_val(&yaml, "pad_char")
            .and_then(|s| s.chars().next())
            .unwrap_or('_');
        Ok(Reader {
            s: session(&bytes, threads)?,
            slots: num("max_plate_slots", 10),
            alphabet,
            pad,
            w: num("img_width", 128) as u32,
            h: num("img_height", 64) as u32,
            gray: yaml_val(&yaml, "image_color_mode") == Some("grayscale"),
            allowed: Vec::new(),
        })
    }

    /// ШАГ 4. Читает пачку вырезанных номеров за один прогон: каждая вырезка
    /// сжимается до 128×64, сеть выдаёт для 10 позиций вероятности 37 знаков
    /// (0–9, A–Z, «_» — пусто), а plate::decode_shaped собирает из них номер РФ.
    pub fn read(&mut self, crops: &[Rgb]) -> Res<Vec<Reading>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let c = if self.gray { 1 } else { 3 };
        let mut data = Vec::with_capacity(crops.len() * (self.w * self.h) as usize * c);
        for cr in crops {
            let r = imgops::resize(cr, self.w, self.h);
            if self.gray {
                data.extend(r.pixels().map(|p| {
                    ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8
                }));
            } else {
                data.extend_from_slice(r.as_raw());
            }
        }
        let shape = vec![crops.len() as i64, self.h as i64, self.w as i64, c as i64];
        let t = Tensor::from_array((shape, data)).map_err(e("тензор"))?;
        let out = self.s.run(ort::inputs![t]).map_err(e("распознаватель"))?;
        // выход «plate»: [N, слоты, алфавит]
        let idx = out.keys().position(|k| k == "plate").unwrap_or(0);
        let (_, v) = out[idx].try_extract_tensor::<f32>().map_err(e("распознаватель"))?;
        let per = self.slots * self.alphabet.len();
        Ok(v
            .chunks_exact(per)
            .zip(crops)
            .map(|(p, cr)| {
                let aspect = cr.width() as f32 / cr.height().max(1) as f32;
                let one_line = if aspect > 2.6 {
                    Some(true)
                } else if aspect < 1.8 {
                    Some(false)
                } else {
                    None
                };
                plate::decode_shaped(p, self.slots, &self.alphabet, self.pad, one_line, &self.allowed)
            })
            .collect())
    }
}

/// ШАГ 3б, вариант 2. Маленькая сеть углов номера: по серой области вокруг рамки
/// детектора (192×64) сразу выдаёт 4 угла таблички — 8 чисел в долях области.
/// Обучена на фото AUTO.RIA с размеченными углами (eval/train_corners.py), ≈0,7 млн
/// параметров; все номера кадра — одной пачкой.
pub struct CornerNet {
    s: Session,
}

impl CornerNet {
    pub fn new(over: Option<&Path>, threads: usize) -> Res<CornerNet> {
        Ok(CornerNet { s: session(&model_bytes(over, CORNERS_ONNX)?, threads)? })
    }

    /// Углы табличек для всех рамок кадра одной пачкой. None — рамка не подходит
    /// (мелкая, двухстрочная) или ответ сети не прошёл проверку rectify::sane.
    pub fn find(&mut self, img: &Rgb, dets: &[Det]) -> Res<Vec<Option<[Pt; 4]>>> {
        let regs: Vec<Option<(f32, f32, f32, f32)>> = dets.iter().map(rectify::net_region).collect();
        let (w, h) = (rectify::NET_W, rectify::NET_H);
        let mut data = Vec::new();
        let mut n = 0i64;
        for &(x, y, rw, rh) in regs.iter().flatten() {
            let rect = [(x, y), (x + rw, y), (x + rw, y + rh), (x, y + rh)];
            let crop = rectify::warp(img, &rect, w, h).ok_or("сеть углов: не удалось вырезать область")?;
            data.extend(crop.pixels().map(|p| {
                (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0
            }));
            n += 1;
        }
        if n == 0 {
            return Ok(vec![None; dets.len()]);
        }
        let t = Tensor::from_array((vec![n, 1, h as i64, w as i64], data)).map_err(e("тензор"))?;
        let out = self.s.run(ort::inputs![t]).map_err(e("сеть углов"))?;
        let (_, v) = out[0].try_extract_tensor::<f32>().map_err(e("сеть углов"))?;
        let mut it = v.chunks_exact(8);
        Ok(regs
            .iter()
            .zip(dets)
            .map(|(r, d)| {
                let (x, y, rw, rh) = (*r)?;
                let c = it.next()?;
                let pts = [0, 1, 2, 3].map(|k| (x + c[2 * k] * rw, y + c[2 * k + 1] * rh));
                rectify::sane(&pts, d).then_some(pts)
            })
            .collect())
    }
}
