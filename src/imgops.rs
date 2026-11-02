//! Операции с картинками: подготовка кадра для детектора, вырезание номера,
//! резкость, рамки и подписи на снимках.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb as Px, RgbImage};
use std::path::Path;

pub type Rgb = RgbImage;

static FONT: &[u8] = include_bytes!("../assets/LiberationSans-Bold.ttf");

pub fn resize(img: &Rgb, w: u32, h: u32) -> Rgb {
    if img.width() == w && img.height() == h {
        return img.clone();
    }
    image::imageops::resize(img, w, h, FilterType::Triangle)
}

/// Вписать кадр в квадрат size×size с серыми полями (как при обучении YOLO).
/// Возвращает CHW float [0..1], масштаб и сдвиги полей.
pub fn letterbox_chw(img: &Rgb, size: u32) -> (Vec<f32>, f32, f32, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let r = (size as f32 / h).min(size as f32 / w);
    let nw = ((w * r).round() as u32).clamp(1, size);
    let nh = ((h * r).round() as u32).clamp(1, size);
    let dw = (size - nw) as f32 / 2.0;
    let dh = (size - nh) as f32 / 2.0;
    let left = (dw - 0.1).round() as u32;
    let top = (dh - 0.1).round() as u32;
    let small = resize(img, nw, nh);
    let plane = (size * size) as usize;
    let mut out = vec![114.0 / 255.0; plane * 3];
    for (x, y, p) in small.enumerate_pixels() {
        let i = ((y + top) * size + x + left) as usize;
        if i < plane {
            out[i] = p[0] as f32 / 255.0;
            out[plane + i] = p[1] as f32 / 255.0;
            out[2 * plane + i] = p[2] as f32 / 255.0;
        }
    }
    (out, r, dw, dh)
}

/// Вырезать прямоугольник с запасом по краям (доля ширины/высоты).
pub fn crop(img: &Rgb, x1: f32, y1: f32, x2: f32, y2: f32, pad_x: f32, pad_y: f32) -> Rgb {
    let (w, h) = (x2 - x1, y2 - y1);
    let cx1 = (x1 - w * pad_x).floor().max(0.0) as u32;
    let cy1 = (y1 - h * pad_y).floor().max(0.0) as u32;
    let cx2 = ((x2 + w * pad_x).ceil() as u32).min(img.width());
    let cy2 = ((y2 + h * pad_y).ceil() as u32).min(img.height());
    let cw = cx2.saturating_sub(cx1).max(1);
    let ch = cy2.saturating_sub(cy1).max(1);
    image::imageops::crop_imm(img, cx1.min(img.width() - 1), cy1.min(img.height() - 1), cw, ch)
        .to_image()
}

/// Резкость: дисперсия лапласиана на уменьшенной до 128×32 копии.
#[allow(dead_code)]
pub fn sharpness(img: &Rgb) -> f32 {
    let g = image::imageops::grayscale(&resize(img, 128, 32));
    let (w, h) = g.dimensions();
    let px = |x: u32, y: u32| g.get_pixel(x, y)[0] as f32;
    let mut vals = Vec::with_capacity((w * h) as usize);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            vals.push(4.0 * px(x, y) - px(x - 1, y) - px(x + 1, y) - px(x, y - 1) - px(x, y + 1));
        }
    }
    let n = vals.len().max(1) as f32;
    let m = vals.iter().sum::<f32>() / n;
    vals.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / n
}

pub fn rect(img: &mut Rgb, x1: f32, y1: f32, x2: f32, y2: f32, color: [u8; 3], th: u32) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x1, y1, x2, y2) = (x1 as i64, y1 as i64, x2 as i64, y2 as i64);
    for t in 0..th as i64 {
        for x in (x1 - t).max(0)..=(x2 + t).min(w - 1) {
            for y in [y1 - t, y2 + t] {
                if (0..h).contains(&y) {
                    img.put_pixel(x as u32, y as u32, Px(color));
                }
            }
        }
        for y in (y1 - t).max(0)..=(y2 + t).min(h - 1) {
            for x in [x1 - t, x2 + t] {
                if (0..w).contains(&x) {
                    img.put_pixel(x as u32, y as u32, Px(color));
                }
            }
        }
    }
}

fn fill(img: &mut Rgb, x: i64, y: i64, w: i64, h: i64, color: [u8; 3]) {
    for yy in y.max(0)..(y + h).min(img.height() as i64) {
        for xx in x.max(0)..(x + w).min(img.width() as i64) {
            img.put_pixel(xx as u32, yy as u32, Px(color));
        }
    }
}

/// Подпись на плашке: левый нижний угол плашки в (x, y).
pub fn label(img: &mut Rgb, x: f32, y: f32, text: &str, px: f32, bg: [u8; 3], fg: [u8; 3]) {
    let font = FontRef::try_from_slice(FONT).expect("шрифт");
    let sf = font.as_scaled(PxScale::from(px));
    let tw: f32 = text.chars().map(|c| sf.h_advance(font.glyph_id(c))).sum();
    let pad = (px * 0.25).max(2.0);
    let bh = sf.ascent() - sf.descent() + pad;
    let bw = tw + pad * 2.0;
    let bx = x.clamp(0.0, (img.width() as f32 - bw).max(0.0));
    let by = (y - bh).max(0.0);
    fill(img, bx as i64, by as i64, bw.ceil() as i64, bh.ceil() as i64, bg);
    let mut cx = bx + pad;
    let base = by + pad * 0.5 + sf.ascent();
    for c in text.chars() {
        let gid = font.glyph_id(c);
        let g = gid.with_scale_and_position(PxScale::from(px), ab_glyph::point(cx, base));
        cx += sf.h_advance(gid);
        if let Some(o) = font.outline_glyph(g) {
            let b = o.px_bounds();
            o.draw(|gx, gy, cov| {
                let (xx, yy) = (b.min.x as i64 + gx as i64, b.min.y as i64 + gy as i64);
                if xx >= 0 && yy >= 0 && (xx as u32) < img.width() && (yy as u32) < img.height() {
                    let p = img.get_pixel_mut(xx as u32, yy as u32);
                    for k in 0..3 {
                        p[k] = (p[k] as f32 * (1.0 - cov) + fg[k] as f32 * cov) as u8;
                    }
                }
            });
        }
    }
}

pub fn save_jpg(img: &Rgb, path: &Path, quality: u8) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, quality)
        .encode_image(img)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Уменьшить до max_w по ширине (для снимков в отчёт).
pub fn fit_width(img: &Rgb, max_w: u32) -> Rgb {
    if img.width() <= max_w {
        return img.clone();
    }
    let h = (img.height() as f32 * max_w as f32 / img.width() as f32).round().max(1.0) as u32;
    resize(img, max_w, h)
}

pub fn from_raw(w: u32, h: u32, data: Vec<u8>) -> Option<Rgb> {
    ImageBuffer::from_raw(w, h, data)
}
