//! ШАГ 3. Выпрямление номера (без нейросети).
//!
//! Камера часто смотрит на номер сбоку или сверху: табличка на снимке —
//! наклонённый и скошенный четырёхугольник. Распознаватель же учился на ровных
//! прямоугольных табличках (как их вырезали из фото AUTO.RIA). Поэтому находим
//! 4 угла таблички и преобразованием перспективы (гомографией) «разворачиваем»
//! её в ровный прямоугольник с пропорциями 520×112 мм по ГОСТ Р 50577.
//!
//! Как ищем углы (всё внутри рамки детектора с запасом):
//!  1. переводим область в серое и приводим к ширине 220 точек;
//!  2. символы — тёмные пятна на светлом фоне: адаптивный порог + связные
//!     области подходящего размера;
//!  3. через низы символов проводим прямую (RANSAC) — это наклон строки;
//!  4. подбираем наклон вертикалей: при верном наклоне символы самые «узкие»;
//!  5. верх и низ таблички ставим по высоте цифр (пропорции замерены на
//!     обучающих снимках), левый и правый край ищем по перепаду яркости
//!     «светлый фон таблички → тёмная рамка»;
//!  6. проверяем, что результат правдоподобен, иначе отказываемся (None) —
//!     тогда программа читает обычную вырезку, как в версии 0.1.
//!
//! Углы найдены — дальше homography() считает преобразование перспективы,
//! а warp() вырезает по нему ровную табличку 297×64 из исходного кадра.

use crate::engine::Det;
use crate::imgops::{self, Rgb};

/// Точка (x, y) в пикселях кадра.
pub type Pt = (f32, f32);

/// Размер выпрямленного номера, который отдаём распознавателю.
pub const OUT_W: u32 = 297;
pub const OUT_H: u32 = 64;

/// Найденная табличка: углы по часовой стрелке от левого верхнего
/// (левый верх, правый верх, правый низ, левый низ) и сколько символов нашли.
#[derive(Clone, Debug)]
pub struct Corners {
    pub pts: [Pt; 4],
    pub chars: usize,
}

/// Серое изображение в числах с плавающей точкой.
struct Gray {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

impl Gray {
    fn px(&self, x: usize, y: usize) -> f32 {
        self.v[y * self.w + x]
    }
    /// Яркость в дробной точке (билинейно); None — за краем.
    fn at(&self, x: f32, y: f32) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (self.w - 1) as f32 || y > (self.h - 1) as f32 {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = self.px(x0, y0) * (1.0 - fx) + self.px(x1, y0) * fx;
        let bot = self.px(x0, y1) * (1.0 - fx) + self.px(x1, y1) * fx;
        Some(top * (1.0 - fy) + bot * fy)
    }
}

/// Связная область (кандидат в символ).
struct Blob {
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    pixels: Vec<(u16, u16)>,
}

impl Blob {
    fn h(&self) -> f32 {
        (self.y2 - self.y1 + 1) as f32
    }
    fn w(&self) -> f32 {
        (self.x2 - self.x1 + 1) as f32
    }
    fn cx(&self) -> f32 {
        (self.x1 + self.x2) as f32 / 2.0 + 0.5
    }
    fn bottom(&self) -> f32 {
        self.y2 as f32 + 1.0
    }
}

/// Вход сети углов (вариант 2, engine.rs → CornerNet): серая область 192×64.
pub const NET_W: u32 = 192;
pub const NET_H: u32 = 64;

/// Область для сети углов: рамка детектора ±15 % по ширине и ±50 % по высоте
/// (x, y, ширина, высота). За краем кадра — повтор крайних точек, как при обучении.
/// None — рамка слишком мелкая или вытянута вверх. Порог по форме мягче, чем
/// у find_corners: сильно наклонённый номер даёт почти квадратную рамку,
/// а именно такие номера выпрямление спасает чаще всего.
pub fn net_region(d: &Det) -> Option<(f32, f32, f32, f32)> {
    let (bw, bh) = (d.w(), d.h());
    if bw < 24.0 || bh < 8.0 || bw / bh < 0.6 {
        return None;
    }
    Some((d.x1 - 0.15 * bw, d.y1 - 0.50 * bh, 1.30 * bw, 2.0 * bh))
}

/// Главная функция шага (вариант 1, без нейросети): углы таблички в координатах кадра или None.
pub fn find_corners(img: &Rgb, d: &Det) -> Option<Corners> {
    let (bw, bh) = (d.w(), d.h());
    // совсем мелкие и почти квадратные (двухстрочные) таблички не выпрямляем
    if bw < 24.0 || bh < 8.0 || bw / bh < 1.2 {
        return None;
    }
    // 1. область с запасом и масштаб: ширина рамки → 220 точек
    let rx1 = (d.x1 - 0.12 * bw).max(0.0).floor();
    let ry1 = (d.y1 - 0.40 * bh).max(0.0).floor();
    let rx2 = (d.x2 + 0.12 * bw).min(img.width() as f32).ceil();
    let ry2 = (d.y2 + 0.40 * bh).min(img.height() as f32).ceil();
    let sc = (220.0 / bw).clamp(0.2, 8.0);
    let gw = ((rx2 - rx1) * sc).round().max(8.0) as u32;
    let gh = ((ry2 - ry1) * sc).round().max(8.0) as u32;
    let part = image::imageops::crop_imm(img, rx1 as u32, ry1 as u32, (rx2 - rx1) as u32, (ry2 - ry1) as u32).to_image();
    let small = imgops::resize(&part, gw, gh);
    let g = Gray {
        w: gw as usize,
        h: gh as usize,
        v: small
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .collect(),
    };
    // рамка детектора в координатах уменьшенной области
    let bx = ((d.x1 - rx1) * sc, (d.y1 - ry1) * sc, (d.x2 - rx1) * sc, (d.y2 - ry1) * sc);

    // шаги 2–5 для тёмных символов на светлом фоне и для светлых на тёмном
    // (синие номера полиции); берём вариант, где нашлось больше символов
    let best = [false, true]
        .iter()
        .filter_map(|&inv| plate_in(&g, bx, inv))
        .max_by_key(|c| c.chars)?;
    // назад в координаты кадра
    let pts = best.pts.map(|(x, y)| (rx1 + x / sc, ry1 + y / sc));
    sane(&pts, d).then_some(Corners { pts, chars: best.chars })
}

/// Шаги 2–5 на уменьшенной серой области. Углы — в её координатах.
fn plate_in(g: &Gray, bx: (f32, f32, f32, f32), inverted: bool) -> Option<Corners> {
    // 2. «чернила»: точки темнее окружения, и связные области из них
    let ink = ink_mask(g, bx, inverted);
    let blobs = components(&ink, g.w, g.h);
    let cands: Vec<&Blob> = blobs.iter().filter(|b| looks_like_char(b, bx, g.w, g.h)).collect();
    // 3. строка символов: прямая через низы
    let (row, slope) = baseline(&cands)?;
    let theta = slope.atan();
    let e = (theta.cos(), theta.sin()); // направление строки
    // 4. направление вертикалей
    let alpha = best_shear(&row, e);
    let f = (alpha.sin(), alpha.cos()); // «вниз» по табличке
    // Дальше работаем в координатах таблички: точка = t·e + s·f
    // (t — вдоль строки, s — вниз по вертикалям символов).
    let det = e.0 * f.1 - e.1 * f.0;
    let to_ts = |(x, y): (f32, f32)| ((x * f.1 - y * f.0) / det, (e.0 * y - e.1 * x) / det);
    let to_xy = |(t, s): (f32, f32)| (t * e.0 + s * f.0, t * e.1 + s * f.1);
    // размах символа в координатах таблички: (t_min, t_max, s_min, s_max)
    let extent = |b: &Blob| {
        let mut r = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &(x, y) in &b.pixels {
            let (t, s) = to_ts((x as f32 + 0.5, y as f32 + 0.5));
            r = (r.0.min(t - 0.5), r.1.max(t + 0.5), r.2.min(s - 0.5), r.3.max(s + 0.5));
        }
        r
    };
    let ext: Vec<_> = row.iter().map(|b| extent(b)).collect();
    let mut heights: Vec<f32> = ext.iter().map(|x| x.3 - x.2).collect();
    let mut bottoms: Vec<f32> = ext.iter().map(|x| x.3).collect();
    heights.sort_by(f32::total_cmp);
    bottoms.sort_by(f32::total_cmp);
    let hd = heights[heights.len() * 3 / 4]; // высота цифр
    let base = bottoms[bottoms.len() / 2]; // низ строки
    if hd < 0.06 * (bx.2 - bx.0) {
        return None;
    }
    let mut t_min = ext.iter().map(|x| x.0).fold(f32::MAX, f32::min);
    let mut t_max = ext.iter().map(|x| x.1).fold(f32::MIN, f32::max);
    // 4б. Тонкие буквы (Е, С) иногда рассыпаются на куски и не проходят
    // отбор. Добираем любые куски, стоящие на той же строке вплотную к краям.
    let mut spans: Vec<(f32, f32)> = ext.iter().map(|x| (x.0, x.1)).collect();
    let mut extras: Vec<(f32, f32)> = blobs
        .iter()
        .filter(|b| b.pixels.len() >= 4)
        .map(|b| extent(b))
        .filter(|x| (x.3 - base).abs() < 0.15 * hd && x.2 > base - 1.15 * hd && x.3 - x.2 > 0.2 * hd && x.1 - x.0 < 1.2 * hd)
        .map(|x| (x.0, x.1))
        .collect();
    extras.sort_by(|a, b| a.0.total_cmp(&b.0));
    loop {
        let before = (t_min, t_max);
        for &(a, b) in &extras {
            if a < t_min && b > t_min - 0.6 * hd {
                t_min = a;
                spans.push((a, b));
            }
            if b > t_max && a < t_max + 0.6 * hd {
                t_max = b;
                spans.push((a, b));
            }
        }
        if before == (t_min, t_max) {
            break;
        }
    }
    // сколько символов в строке: перекрывающиеся куски считаем одним
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut count = 0;
    let mut end = f32::MIN;
    for (a, b) in spans {
        if a > end - 0.1 * hd {
            count += 1;
        }
        end = end.max(b);
    }

    // 5а. Верх и низ — по пропорциям: на снимках, где учился распознаватель
    // (AUTO.RIA), поле над цифрами ≈ 0,40 их высоты, под ними ≈ 0,34.
    // Так выпрямленная вырезка выглядит ровно как обучающие.
    let top = base - hd - 0.40 * hd;
    let bot = base + 0.34 * hd;
    let hp = bot - top;

    // 5б. Левый и правый края ищем на грубо выпрямленной картинке R(t, s)
    let (t0, t1) = (t_min - 2.2 * hd, t_max + 5.2 * hd);
    let rw = (t1 - t0).ceil() as usize + 1;
    let rh = (bot - top).ceil() as usize + 1;
    if rw * rh > 200_000 {
        return None;
    }
    let sign = if inverted { -1.0 } else { 1.0 }; // фон таблички светлее символов
    let mut r = vec![f32::NAN; rw * rh];
    for j in 0..rh {
        for i in 0..rw {
            let (x, y) = to_xy((t0 + i as f32, top + j as f32));
            if let Some(v) = g.at(x - 0.5, y - 0.5) {
                r[j * rw + i] = sign * v;
            }
        }
    }
    // яркость фона таблички: светлые точки между символами
    let mut band: Vec<f32> = Vec::new();
    for j in 0..rh {
        for i in ((t_min - t0) as usize)..((t_max - t0) as usize).min(rw) {
            let v = r[j * rw + i];
            if v.is_finite() {
                band.push(v);
            }
        }
    }
    if band.len() < 20 {
        return None;
    }
    band.sort_by(f32::total_cmp);
    let bg = band[band.len() * 8 / 10];
    let dark = band[band.len() / 10];
    let thr = bg - 0.4 * (bg - dark).max(10.0);
    // по столбцам: перепад яркости (+ — правее светлее) и доля «фоновых» точек
    let mut edge = vec![0f32; rw];
    let mut light = vec![0f32; rw];
    for i in 0..rw {
        let (mut sum, mut n, mut l, mut m) = (0.0, 0, 0, 0);
        for j in 0..rh {
            let v = r[j * rw + i];
            if v.is_finite() {
                m += 1;
                if v > thr {
                    l += 1;
                }
            }
            if i > 0 && i + 1 < rw {
                let dv = r[j * rw + i + 1] - r[j * rw + i - 1];
                if dv.is_finite() {
                    sum += dv;
                    n += 1;
                }
            }
        }
        edge[i] = if n > 0 { sum / n as f32 } else { 0.0 };
        light[i] = if m > 0 { l as f32 / m as f32 } else { 0.0 };
    }
    let avg = |a: f32, b: f32| -> f32 {
        let (i0, i1) = (((a - t0).max(0.0)) as usize, ((b - t0).max(0.0) as usize).min(rw - 1));
        if i1 < i0 {
            return 0.0;
        }
        light[i0..=i1].iter().sum::<f32>() / (i1 - i0 + 1) as f32
    };
    let at = |t: f32| ((t - t0).round().max(0.0) as usize).min(rw - 1);
    let q = 0.3 * hd;
    // край: сильный перепад + внутри светлый фон таблички, снаружи — нет
    let left_score = |l: f32| edge[at(l)].clamp(0.0, 40.0) + 30.0 * (avg(l + 1.0, l + q) - avg(l - q, l - 1.0));
    let right_score = |r: f32| (-edge[at(r)]).clamp(0.0, 40.0) + 30.0 * (avg(r - q, r - 1.0) - avg(r + 1.0, r + q));
    // ожидания: ширина ≈ 4,9 высоты (520×112 мм плюс небольшой взгляд сверху);
    // если нашлись все 6 знаков «А123ВС» — ещё и по раскладке номера РФ
    // (замерено на AUTO.RIA): до первой буквы 0,17 длины строки, после последней 0,48
    let span = t_max - t_min;
    let full = (6..=7).contains(&count);
    let mut best = (f32::MIN, 0.0, 0.0);
    let ls: Vec<(f32, f32)> = (0..)
        .map(|k| t_min - 0.05 * hd - k as f32 * 0.5)
        .take_while(|&l| l > t_min - 2.0 * hd)
        .map(|l| (l, left_score(l)))
        .collect();
    let rs: Vec<(f32, f32)> = (0..)
        .map(|k| t_max + 0.05 * hd + k as f32 * 0.5)
        .take_while(|&r| r < t_max + 5.0 * hd)
        .map(|r| (r, right_score(r)))
        .collect();
    for &(l, sl) in &ls {
        for &(r, sr) in &rs {
            let ratio = (r - l) / hp;
            let mut v = sl + sr - 0.5 * ((ratio / 4.9).ln() / 0.15).powi(2);
            if full {
                v -= 0.5 * ((l - (t_min - 0.167 * span)) / (0.06 * span)).powi(2);
                v -= 0.5 * ((r - (t_max + 0.478 * span)) / (0.08 * span)).powi(2);
            }
            if v > best.0 {
                best = (v, l, r);
            }
        }
    }
    // Обучающие вырезки AUTO.RIA чуть шире самой таблички (замерено):
    // слева ещё 3,6 % ширины, справа 1,9 %.
    let (_, l, r) = best;
    let (left, right) = (l - 0.036 * (r - l), r + 0.019 * (r - l));
    if best.0 == f32::MIN {
        return None;
    }
    let pts = [(left, top), (right, top), (right, bot), (left, bot)].map(to_xy);
    Some(Corners { pts, chars: count })
}

/// 2а. Маска «чернил»: точка темнее среднего по окну вокруг неё (окно ≈
/// половина высоты символа). Для светлых символов на тёмном — наоборот.
fn ink_mask(g: &Gray, bx: (f32, f32, f32, f32), inverted: bool) -> Vec<bool> {
    let (w, h) = (g.w, g.h);
    // интегральное изображение для быстрого среднего по окну
    let mut ii = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += g.px(x, y) as f64;
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + row;
        }
    }
    // контраст внутри рамки — от него порог
    let mut inside: Vec<f32> = Vec::new();
    for y in (bx.1.max(0.0) as usize)..(bx.3 as usize).min(h) {
        for x in (bx.0.max(0.0) as usize)..(bx.2 as usize).min(w) {
            inside.push(g.px(x, y));
        }
    }
    if inside.len() < 50 {
        return vec![false; w * h];
    }
    inside.sort_by(f32::total_cmp);
    let contrast = inside[inside.len() * 95 / 100] - inside[inside.len() * 5 / 100];
    let delta = (0.08 * contrast).max(6.0);
    // высота таблички ≈ ширина / 4,6; символ ≈ 0,6 высоты; окно — половина символа
    let rad = ((0.07 * (bx.2 - bx.0)) as usize).max(3);
    let mut mask = vec![false; w * h];
    for y in 0..h {
        let (ya, yb) = (y.saturating_sub(rad), (y + rad + 1).min(h));
        for x in 0..w {
            let (xa, xb) = (x.saturating_sub(rad), (x + rad + 1).min(w));
            let sum = ii[yb * (w + 1) + xb] - ii[ya * (w + 1) + xb] - ii[yb * (w + 1) + xa] + ii[ya * (w + 1) + xa];
            let mean = (sum / ((yb - ya) * (xb - xa)) as f64) as f32;
            let v = g.px(x, y);
            mask[y * w + x] = if inverted { v > mean + delta } else { v < mean - delta };
        }
    }
    mask
}

/// 2б. Связные области маски (соседи по 8 направлениям).
fn components(mask: &[bool], w: usize, h: usize) -> Vec<Blob> {
    let mut seen = vec![false; w * h];
    let mut blobs = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut b = Blob { x1: w, y1: h, x2: 0, y2: 0, pixels: Vec::new() };
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            b.x1 = b.x1.min(x);
            b.x2 = b.x2.max(x);
            b.y1 = b.y1.min(y);
            b.y2 = b.y2.max(y);
            b.pixels.push((x as u16, y as u16));
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let q = ny as usize * w + nx as usize;
                        if mask[q] && !seen[q] {
                            seen[q] = true;
                            stack.push(q);
                        }
                    }
                }
            }
        }
        blobs.push(b);
    }
    blobs
}

/// 2в. Похожа ли область на символ: размер, пропорции, заполненность.
fn looks_like_char(b: &Blob, bx: (f32, f32, f32, f32), w: usize, h: usize) -> bool {
    let (bw, bh) = (bx.2 - bx.0, bx.3 - bx.1);
    let fill = b.pixels.len() as f32 / (b.w() * b.h());
    b.h() >= 7.0
        && b.h() >= 0.06 * bw
        && b.h() <= 0.95 * bh
        && b.w() <= 1.3 * b.h()
        && b.w() >= 0.08 * b.h()
        && (0.12..=0.92).contains(&fill)
        && b.cx() > bx.0 - 0.05 * bw
        && b.cx() < bx.2 + 0.05 * bw
        && (b.y1 + b.y2) as f32 / 2.0 > bx.1
        && ((b.y1 + b.y2) as f32 / 2.0) < bx.3
        && b.x1 > 0
        && b.y1 > 0
        && b.x2 + 1 < w
        && b.y2 + 1 < h
}

/// 3. RANSAC: пара символов задаёт прямую через низы; считаем, сколько ещё
/// символов похожей высоты лежит на ней. Возвращает символы строки и прямую y = a + m·x.
fn baseline<'a>(blobs: &[&'a Blob]) -> Option<(Vec<&'a Blob>, f32)> {
    let n = blobs.len();
    if n < 4 || n > 120 {
        return None;
    }
    let mut best: Option<(usize, f32, Vec<usize>)> = None;
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (blobs[i], blobs[j]);
            let hm = (a.h() + b.h()) / 2.0;
            let ratio = a.h() / b.h();
            if !(0.6..=1.67).contains(&ratio) || (a.cx() - b.cx()).abs() < 0.5 * hm {
                continue;
            }
            let m = (b.bottom() - a.bottom()) / (b.cx() - a.cx());
            if m.abs() > 0.7 {
                continue;
            }
            let mut idx: Vec<usize> = (0..n)
                .filter(|&k| {
                    let c = blobs[k];
                    let y = a.bottom() + m * (c.cx() - a.cx());
                    (c.bottom() - y).abs() <= 0.15 * hm && (0.55..=1.8).contains(&(c.h() / hm))
                })
                .collect();
            // символы идут подряд: рвём цепочку на больших промежутках
            idx.sort_by(|&p, &q| blobs[p].cx().total_cmp(&blobs[q].cx()));
            let mut runs: Vec<Vec<usize>> = vec![Vec::new()];
            for &k in &idx {
                if let Some(&last) = runs.last().unwrap().last() {
                    if blobs[k].cx() - blobs[last].cx() > 1.9 * hm {
                        runs.push(Vec::new());
                    }
                }
                runs.last_mut().unwrap().push(k);
            }
            let run = runs.into_iter().max_by_key(|r| r.len()).unwrap_or_default();
            let score: f32 = run.iter().map(|&k| blobs[k].h()).sum();
            if best.as_ref().is_none_or(|b| (run.len(), score) > (b.0, b.1)) {
                best = Some((run.len(), score, run));
            }
        }
    }
    let (count, _, run) = best?;
    if count < 4 {
        return None;
    }
    // уточняем прямую методом наименьших квадратов
    let pts: Vec<(f32, f32)> = run.iter().map(|&k| (blobs[k].cx(), blobs[k].bottom())).collect();
    let nn = pts.len() as f32;
    let mx = pts.iter().map(|p| p.0).sum::<f32>() / nn;
    let my = pts.iter().map(|p| p.1).sum::<f32>() / nn;
    let sxx: f32 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f32 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let m = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let _ = my - m * mx;
    Some((run.iter().map(|&k| blobs[k]).collect(), m))
}

/// 4. Наклон вертикалей (угол от вертикали кадра, радианы). Скошенный символ
/// «шире», чем прямой: перебираем наклон и берём тот, при котором символы в
/// сумме самые узкие (наименьший разброс точек поперёк вертикали).
fn best_shear(chars: &[&Blob], e: (f32, f32)) -> f32 {
    let score = |alpha: f32| -> f32 {
        let f = (alpha.sin(), alpha.cos());
        let det = e.0 * f.1 - e.1 * f.0;
        chars
            .iter()
            .map(|b| {
                let ts: Vec<f32> = b
                    .pixels
                    .iter()
                    .map(|&(x, y)| ((x as f32 + 0.5) * f.1 - (y as f32 + 0.5) * f.0) / det)
                    .collect();
                let m = ts.iter().sum::<f32>() / ts.len() as f32;
                ts.iter().map(|t| (t - m).powi(2)).sum::<f32>() / ts.len() as f32
            })
            .sum()
    };
    let deg = std::f32::consts::PI / 180.0;
    let mut best = (0.0f32, score(0.0));
    for k in -40..=40 {
        let a = k as f32 * deg;
        let v = score(a);
        if v < best.1 {
            best = (a, v);
        }
    }
    let c = best.0;
    for k in -8..=8 {
        let a = c + k as f32 * 0.125 * deg;
        let v = score(a);
        if v < best.1 {
            best = (a, v);
        }
    }
    best.0
}

/// 6. Проверка правдоподобия: выпуклый четырёхугольник разумного размера
/// рядом с рамкой детектора. Ею же проверяются ответы сети углов (вариант 2).
pub fn sane(p: &[Pt; 4], d: &Det) -> bool {
    let cross = |a: Pt, b: Pt, c: Pt| (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
    let convex = (0..4).all(|i| cross(p[i], p[(i + 1) % 4], p[(i + 2) % 4]) > 0.0);
    let area = 0.5 * (0..4).map(|i| p[i].0 * p[(i + 1) % 4].1 - p[(i + 1) % 4].0 * p[i].1).sum::<f32>();
    let box_area = d.w() * d.h();
    let (mx, my) = (0.25 * d.w(), 0.6 * d.h());
    let inside = p.iter().all(|&(x, y)| x > d.x1 - mx && x < d.x2 + mx && y > d.y1 - my && y < d.y2 + my);
    convex && inside && area > 0.25 * box_area && area < 1.6 * box_area
}

/// Гомография: матрица 3×3, переводящая 4 точки `src` в 4 точки `dst`
/// (решаем систему 8×8 методом Гаусса).
pub fn homography(src: &[Pt; 4], dst: &[Pt; 4]) -> Option<[f64; 9]> {
    let mut a = [[0f64; 9]; 8];
    for k in 0..4 {
        let (x, y) = (src[k].0 as f64, src[k].1 as f64);
        let (u, v) = (dst[k].0 as f64, dst[k].1 as f64);
        a[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for c in 0..8 {
        let piv = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[piv][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, piv);
        for r in 0..8 {
            if r != c {
                let k = a[r][c] / a[c][c];
                for q in c..9 {
                    a[r][q] -= k * a[c][q];
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

/// Вырезает табличку по 4 углам в ровный прямоугольник w×h (каждая точка —
/// среднее нескольких выборок, чтобы мелкий текст не «рассыпался»).
pub fn warp(img: &Rgb, pts: &[Pt; 4], w: u32, h: u32) -> Option<Rgb> {
    let (wf, hf) = (w as f32, h as f32);
    let rect = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];
    let m = homography(&rect, pts)?;
    let plate_w = ((pts[1].0 - pts[0].0).hypot(pts[1].1 - pts[0].1)).max(1.0);
    let plate_h = ((pts[3].0 - pts[0].0).hypot(pts[3].1 - pts[0].1)).max(1.0);
    let ss = ((plate_w / wf).max(plate_h / hf).ceil() as u32).clamp(1, 4);
    let (iw, ih) = (img.width() as f64, img.height() as f64);
    let sample = |x: f64, y: f64| -> [f64; 3] {
        let x = x.clamp(0.0, iw - 1.0);
        let y = y.clamp(0.0, ih - 1.0);
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(img.width() - 1), (y0 + 1).min(img.height() - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let mut out = [0f64; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let p = |xx, yy| img.get_pixel(xx, yy)[c] as f64;
            let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
            let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
            *o = top * (1.0 - fy) + bot * fy;
        }
        out
    };
    let mut out = Rgb::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f64; 3];
            for sy in 0..ss {
                for sx in 0..ss {
                    let u = x as f64 + (sx as f64 + 0.5) / ss as f64;
                    let v = y as f64 + (sy as f64 + 0.5) / ss as f64;
                    let z = m[6] * u + m[7] * v + m[8];
                    let px = (m[0] * u + m[1] * v + m[2]) / z - 0.5;
                    let py = (m[3] * u + m[4] * v + m[5]) / z - 0.5;
                    let s = sample(px, py);
                    for c in 0..3 {
                        acc[c] += s[c];
                    }
                }
            }
            let n = (ss * ss) as f64;
            out.put_pixel(x, y, image::Rgb(acc.map(|v| (v / n).round().clamp(0.0, 255.0) as u8)));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_maps_corners() {
        let a = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)];
        let b = [(3.0, 2.0), (40.0, 8.0), (38.0, 20.0), (1.0, 15.0)];
        let h = homography(&a, &b).unwrap();
        for k in 0..4 {
            let (x, y) = (a[k].0 as f64, a[k].1 as f64);
            let z = h[6] * x + h[7] * y + 1.0;
            assert!(((h[0] * x + h[1] * y + h[2]) / z - b[k].0 as f64).abs() < 1e-3);
            assert!(((h[3] * x + h[4] * y + h[5]) / z - b[k].1 as f64).abs() < 1e-3);
        }
    }
}
