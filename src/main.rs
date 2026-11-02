//! Номерок — распознавание автомобильных номеров РФ на фото, видео и с IP-камер.
//! Работает офлайн: детектор номеров YOLOv9 + распознаватель символов (ONNX),
//! проверка формата по ГОСТ Р 50577, трекинг и голосование по кадрам на видео.
//!
//! Порядок работы (номера шагов есть в комментариях по всему коду):
//!   ШАГ 1. кадр: фото с диска (load_image) или кадр видео/камеры (video.rs)
//!   ШАГ 2. поиск номеров на кадре — детектор (engine.rs, Detector::detect)
//!   ШАГ 3. вырезка номера: обычная (imgops::crop) и выпрямленная по 4 углам
//!          преобразованием перспективы (rectify.rs) — Ctx::read_all.
//!          Углы ищем двумя способами: по яркости (вариант 1, rectify::find_corners)
//!          и маленькой нейросетью (вариант 2, engine.rs, CornerNet)
//!   ШАГ 4. чтение символов (engine.rs, Reader::read) и проверка формата РФ
//!          (plate.rs, decode_shaped)
//!   ШАГ 5. страховка: из всех прочтений берём лучшее (better)
//!   ШАГ 6. только видео: один номер — одна машина, голосование по кадрам (track.rs)
//!   ШАГ 7. запись результата (report.rs)

mod engine;
mod imgops;
mod plate;
mod rectify;
mod report;
mod track;
mod video;

use engine::{CornerNet, Det, Detector, Reader, Res};
use imgops::Rgb;
use plate::Reading;
use report::{Record, Report};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];
const VIDEO_EXT: &[&str] = &[
    "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "ts", "mts", "m2ts", "3gp", "mpg",
    "mpeg", "asf", "dav", "h264", "h265", "264", "265",
];

const HELP: &str = "\
Номерок — распознавание автомобильных номеров РФ (офлайн)

Запуск:
  nomerok <фото | папка | видео | rtsp://камера> [параметры]
  или перетащите файл/папку на nomerok.exe

Параметры:
  -o, --out ПАПКА     куда сохранить результат (по умолчанию рядом с исходником)
  --fps N             видео: сколько кадров в секунду анализировать (6)
  --sure X            порог «уверен», 0–1 (0.7); ниже — «не уверен»
  --find СПИСОК       искать номера: «А123ВС77,К777ОО» или файл .txt (по строке)
  --video             сохранить видео с рамками (annotated.mp4)
  --det-conf X        порог детектора номеров, 0–1 (0.3)
  --rectify РЕЖИМ     выпрямление наклонённых номеров: auto — читать обычную
                      и выпрямленные вырезки и брать лучшую (по умолчанию),
                      off — выключить, on — всегда выпрямленную, если углы
                      найдены; classic / net — только один способ поиска углов
                      (по яркости / нейросетью) плюс обычная вырезка
  --formats СПИСОК    какие номера ждать, через запятую: легковые, такси,
                      прицепы, полиция, мото, транзит (по умолчанию все)
  --gap С             видео: через сколько секунд без номера машина «уехала» (1.5)
  --repeat С          не повторять тот же номер чаще, чем раз в С секунд (20)
  --threads N         потоков процессора (все)
  --det ФАЙЛ.onnx     своя модель поиска номеров
  --ocr ФАЙЛ.onnx     своя модель чтения + --ocr-config ФАЙЛ.yaml
  --corners ФАЙЛ.onnx своя модель углов номера
  --no-open           не открывать отчёт в конце

Результат: plates.csv (Excel), plates.jsonl, report.html, папки crops и frames.
Для видео и камер нужен ffmpeg.exe: рядом с программой, в папке ffmpeg\\bin или в PATH.";

struct Opts {
    input: Option<String>,
    out: Option<PathBuf>,
    fps: f32,
    sure: f32,
    det_conf: f32,
    video: bool,
    find: Vec<String>,
    threads: usize,
    gap: f64,
    repeat: f64,
    det: Option<PathBuf>,
    ocr: Option<PathBuf>,
    ocr_cfg: Option<PathBuf>,
    corners: Option<PathBuf>,
    open: bool,
    rectify: Rectify,
    formats: Option<Vec<bool>>,
}

/// Режим выпрямления номера (шаг 3).
#[derive(Clone, Copy, PartialEq)]
enum Rectify {
    /// обычная + оба способа поиска углов, берём лучшее прочтение
    Auto,
    Off,
    /// всегда выпрямленная (углы сети, если нет — по яркости)
    On,
    /// обычная + углы по яркости (как в 0.2)
    Classic,
    /// обычная + углы нейросетью
    Net,
}

impl Rectify {
    fn classic(self) -> bool {
        matches!(self, Rectify::Auto | Rectify::On | Rectify::Classic)
    }
    fn net(self) -> bool {
        matches!(self, Rectify::Auto | Rectify::On | Rectify::Net)
    }
}

fn parse_args(args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        input: None,
        out: None,
        fps: 6.0,
        sure: 0.7,
        det_conf: 0.3,
        video: false,
        find: Vec::new(),
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        gap: 1.5,
        repeat: 20.0,
        det: None,
        ocr: None,
        ocr_cfg: None,
        corners: None,
        open: true,
        rectify: Rectify::Auto,
        formats: None,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("после {name} нужно значение"));
        let num = |s: String, name: &str| s.replace(',', ".").parse::<f64>().map_err(|_| format!("{name}: не число «{s}»"));
        match a.as_str() {
            "-h" | "--help" | "/?" => return Err(String::new()),
            "-o" | "--out" => o.out = Some(val(a)?.into()),
            "--fps" => o.fps = num(val(a)?, a)?.clamp(0.2, 60.0) as f32,
            "--sure" => o.sure = num(val(a)?, a)?.clamp(0.0, 1.0) as f32,
            "--det-conf" => o.det_conf = num(val(a)?, a)?.clamp(0.01, 0.99) as f32,
            "--gap" => o.gap = num(val(a)?, a)?.clamp(0.2, 60.0),
            "--repeat" => o.repeat = num(val(a)?, a)?.max(0.0),
            "--threads" => o.threads = num(val(a)?, a)?.clamp(1.0, 64.0) as usize,
            "--video" => o.video = true,
            "--no-open" => o.open = false,
            "--rectify" => {
                o.rectify = match val(a)?.to_lowercase().as_str() {
                    "auto" | "авто" => Rectify::Auto,
                    "off" | "no" | "0" | "выкл" | "нет" => Rectify::Off,
                    "on" | "yes" | "1" | "вкл" | "да" | "only" => Rectify::On,
                    "classic" | "яркость" => Rectify::Classic,
                    "net" | "сеть" => Rectify::Net,
                    v => return Err(format!("--rectify: auto, off, on, classic или net, а не «{v}»")),
                }
            }
            "--formats" => o.formats = Some(plate::parse_kinds(&val(a)?)?),
            "--det" => o.det = Some(val(a)?.into()),
            "--ocr" => o.ocr = Some(val(a)?.into()),
            "--ocr-config" => o.ocr_cfg = Some(val(a)?.into()),
            "--corners" => o.corners = Some(val(a)?.into()),
            "--find" => {
                let v = val(a)?;
                let text = if Path::new(&v).is_file() {
                    std::fs::read_to_string(&v).map_err(|e| format!("{v}: {e}"))?
                } else {
                    v
                };
                o.find.extend(
                    text.split([',', ';', '\n', '\r'])
                        .map(plate::normalize)
                        .filter(|s| s.len() >= 4),
                );
            }
            s if s.starts_with('-') && s.len() > 1 && !Path::new(s).exists() => {
                return Err(format!("неизвестный параметр {s}\n\n{HELP}"))
            }
            s => o.input = Some(s.trim_matches('"').to_string()),
        }
    }
    Ok(o)
}

/// Какая вырезка дала прочтение (шаг 3).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Crop {
    /// прямоугольник детектора (как в 0.1)
    Plain,
    /// выпрямленная по углам, найденным по яркости (вариант 1)
    Classic,
    /// выпрямленная по углам, найденным нейросетью (вариант 2)
    Net,
}

impl Crop {
    pub fn name(self) -> &'static str {
        match self {
            Crop::Plain => "обычная",
            Crop::Classic => "выпрямленная",
            Crop::Net => "выпрямленная сетью",
        }
    }
}

/// Одно прочтение одной вырезки.
#[derive(Clone)]
pub struct Variant {
    pub kind: Crop,
    pub read: Reading,
    pub corners: Option<[rectify::Pt; 4]>,
}

/// Один номер на кадре после шагов 2–5.
#[derive(Clone)]
pub struct Seen {
    /// Рамка детектора.
    pub det: Det,
    /// Выбранное прочтение.
    pub read: Reading,
    /// Вырезка, с которой оно сделано (её же сохраняем в crops).
    pub crop: Rgb,
    /// С какой вырезки выбранное прочтение.
    pub kind: Crop,
    /// 4 угла таблички: выбранной выпрямленной вырезки, а если выбрана
    /// обычная — найденные сетью или по яркости (для рамки на картинке).
    pub corners: Option<[rectify::Pt; 4]>,
    /// Все прочтения этого номера (для сравнения в plates.jsonl).
    pub variants: Vec<Variant>,
}

/// ШАГ 5. Лучше ли прочтение `new`, чем текущее `best`.
/// Номер допустимого формата РФ лучше «не похожего на номер»; при равенстве
/// побеждает большая уверенность (минимальная вероятность символа).
fn better(best: &Reading, new: &Reading) -> bool {
    match (best.format.is_some(), new.format.is_some()) {
        (false, true) => true,
        (true, false) => false,
        _ => new.conf > best.conf,
    }
}

struct Ctx {
    det: Detector,
    ocr: Reader,
    corner_net: Option<CornerNet>,
    o: Opts,
    rep: Report,
    exe_dir: PathBuf,
    emitted: HashMap<String, f64>,
}

fn hms(t: f64) -> String {
    let ms = (t * 10.0).round() as u64;
    format!("{:02}:{:02}:{:02}.{}", ms / 36000, ms / 600 % 60, ms / 10 % 60, ms % 10)
}

fn now_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

fn load_image(p: &Path) -> Res<Rgb> {
    use image::ImageDecoder;
    let r = image::ImageReader::open(p)
        .and_then(|r| r.with_guessed_format())
        .map_err(|e| format!("{}: {e}", p.display()))?;
    let mut dec = r.into_decoder().map_err(|e| format!("{}: {e}", p.display()))?;
    let orient = dec.orientation().ok();
    let mut img = image::DynamicImage::from_decoder(dec).map_err(|e| format!("{}: {e}", p.display()))?;
    if let Some(o) = orient {
        img.apply_orientation(o);
    }
    Ok(img.to_rgb8())
}

impl Ctx {
    fn matches(&self, latin: &str) -> bool {
        self.o.find.iter().any(|q| {
            latin == q || (q.len() >= 6 && latin.starts_with(q.as_str()))
        })
    }

    /// ШАГИ 3–5 для всех номеров кадра: вырезаем каждый номер до трёх раз
    /// (обычная + выпрямленные по двум способам поиска углов), читаем все
    /// вырезки одной пачкой и для каждого номера оставляем лучшее прочтение.
    fn read_all(&mut self, img: &Rgb, dets: &[Det]) -> Res<Vec<Seen>> {
        let mode = self.o.rectify;
        // ШАГ 3а. Обычная вырезка: прямоугольник детектора с небольшим запасом
        // (так работала версия 0.1).
        let mut cands: Vec<Vec<(Crop, Option<[rectify::Pt; 4]>, Rgb)>> = dets
            .iter()
            .map(|d| vec![(Crop::Plain, None, imgops::crop(img, d.x1, d.y1, d.x2, d.y2, 0.02, 0.06))])
            .collect();
        // ШАГ 3б. Выпрямленные вырезки: ищем 4 угла таблички и приводим её
        // к ровному прямоугольнику преобразованием перспективы (rectify::warp).
        let mut add = |k: usize, kind: Crop, pts: [rectify::Pt; 4]| {
            if let Some(crop) = rectify::warp(img, &pts, rectify::OUT_W, rectify::OUT_H) {
                cands[k].push((kind, Some(pts), crop));
            }
        };
        //   вариант 1 — углы по яркости (rectify.rs, find_corners)
        if mode.classic() {
            for (k, d) in dets.iter().enumerate() {
                if let Some(c) = rectify::find_corners(img, d) {
                    add(k, Crop::Classic, c.pts);
                }
            }
        }
        //   вариант 2 — углы нейросетью (engine.rs, CornerNet::find)
        if mode.net() {
            if let Some(net) = self.corner_net.as_mut() {
                for (k, pts) in net.find(img, dets)?.into_iter().enumerate() {
                    if let Some(p) = pts {
                        add(k, Crop::Net, p);
                    }
                }
            }
        }
        // ШАГ 4. Распознаватель читает все вырезки за один прогон (engine.rs, Reader::read).
        let batch: Vec<Rgb> = cands.iter().flatten().map(|(_, _, c)| c.clone()).collect();
        let mut reads = self.ocr.read(&batch)?.into_iter();
        // ШАГ 5. Страховка: из всех прочтений номера берём лучшее (better).
        let mut out = Vec::new();
        for (d, cs) in dets.iter().zip(cands) {
            let mut variants = Vec::new();
            let mut crops = Vec::new();
            for (kind, corners, crop) in cs {
                let read = reads.next().ok_or("распознаватель вернул меньше ответов")?;
                variants.push(Variant { kind, read, corners });
                crops.push(crop);
            }
            let mut best = 0;
            for (i, v) in variants.iter().enumerate().skip(1) {
                let pick = if mode == Rectify::On {
                    // всегда выпрямленная: сеть важнее яркости
                    best == 0 || v.kind == Crop::Net
                } else {
                    better(&variants[best].read, &v.read)
                };
                if pick {
                    best = i;
                }
            }
            let corners = variants[best].corners.or_else(|| {
                let of = |k: Crop| variants.iter().find(|v| v.kind == k).and_then(|v| v.corners);
                of(Crop::Net).or_else(|| of(Crop::Classic))
            });
            out.push(Seen {
                det: *d,
                read: variants[best].read.clone(),
                crop: crops.swap_remove(best),
                kind: variants[best].kind,
                corners,
                variants,
            });
        }
        Ok(out)
    }

    fn save_crop(&self, n: usize, plate: &str, crop: &Rgb) -> String {
        let name = format!("crops/{n:05}_{}.jpg", safe_name(plate));
        let h = crop.height().max(1);
        let big = if h < 80 {
            imgops::resize(crop, (crop.width() * 80 / h).max(1), 80)
        } else {
            crop.clone()
        };
        let _ = imgops::save_jpg(&big, &self.rep.dir.join(&name), 90);
        name
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, time: String, latin: &str, display: &str, conf: f32, format: &str,
            frames: usize, source: &str, seen: &Seen, frame: Option<&Rgb>) {
        let n = self.rep.records.len() + 1;
        let crop_name = self.save_crop(n, display, &seen.crop);
        let frame_name = match frame {
            Some(f) => {
                let name = format!("frames/{n:05}.jpg");
                let _ = imgops::save_jpg(&imgops::fit_width(f, 1280), &self.rep.dir.join(&name), 85);
                name
            }
            None => String::new(),
        };
        let found = self.matches(latin);
        let sure = conf >= self.o.sure;
        let mark = if found {
            "  <<< СОВПАДЕНИЕ"
        } else if sure {
            ""
        } else {
            "  (не уверен)"
        };
        eprint!("\r{:70}\r", "");
        println!("{:>10}  {:<12} {:>3.0}%  кадров: {frames:<3} {source}{mark}",
            if time.len() > 12 { &time[11..] } else { &time }, display, conf * 100.0);
        let d = &seen.det;
        self.rep.add(Record {
            n,
            time,
            plate: display.to_string(),
            plate_latin: latin.to_string(),
            conf,
            sure,
            format: format.to_string(),
            frames,
            source: source.to_string(),
            crop: crop_name,
            frame: frame_name,
            found,
            raw: seen.read.raw.clone(),
            bbox: [d.x1, d.y1, d.x2, d.y2].map(|v| v.round()),
            corners: seen.corners.map(|c| c.map(|(x, y)| [(x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0]).to_vec()).unwrap_or_default(),
            crop_kind: seen.kind.name().to_string(),
            variants: seen
                .variants
                .iter()
                .map(|v| report::VariantRec {
                    kind: v.kind.name().to_string(),
                    plate: v.read.display(),
                    plate_latin: v.read.latin.clone(),
                    raw: v.read.raw.clone(),
                    conf: (v.read.conf * 1000.0).round() / 1000.0,
                    format_ok: v.read.format.is_some(),
                    corners: v.corners.map(|c| c.map(|(x, y)| [(x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0]).to_vec()).unwrap_or_default(),
                })
                .collect(),
        });
    }

    fn annotate(&self, img: &mut Rgb, d: &Det, corners: Option<&[rectify::Pt; 4]>, text: &str, conf: f32) {
        let ok = conf >= self.o.sure;
        let color = if self.matches(&plate::normalize(text)) {
            [40, 200, 90]
        } else if ok {
            [255, 200, 0]
        } else {
            [255, 90, 60]
        };
        let th = (img.width() / 640).clamp(2, 6);
        imgops::rect(img, d.x1, d.y1, d.x2, d.y2, color, th);
        // найденные углы таблички — голубым
        if let Some(p) = corners {
            for i in 0..4 {
                imgops::line(img, p[i], p[(i + 1) % 4], [0, 200, 255], (th / 2).max(1));
            }
        }
        let px = (img.width() as f32 / 45.0).clamp(14.0, 48.0);
        imgops::label(img, d.x1, d.y1 - th as f32, &format!("{text}  {:.0}%", conf * 100.0), px, color, [0, 0, 0]);
    }

    // ───────────── фото ─────────────

    fn photo(&mut self, p: &Path) -> Res<()> {
        // ШАГ 1 — кадр, ШАГ 2 — где на нём номера
        let img = load_image(p)?;
        let dets = self.det.detect(&img)?;
        let src = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if dets.is_empty() {
            println!("{:>10}  номеров не найдено  {src}", "—");
            return Ok(());
        }
        let seen = self.read_all(&img, &dets)?;
        let mut ann = img.clone();
        for s in &seen {
            self.annotate(&mut ann, &s.det, s.corners.as_ref(), &s.read.display(), s.read.conf);
        }
        for s in &seen {
            let r = &s.read;
            // мелкий номер (уже 40 точек) читается ненадёжно
            let conf = if s.det.w() < 40.0 { r.conf * 0.5 } else { r.conf };
            self.emit(String::new(), &r.latin, &r.display(), conf, r.format_name(), 1, &src, s, Some(&ann));
        }
        Ok(())
    }

    // ───────────── видео и камеры ─────────────

    fn video(&mut self, input: &str) -> Res<()> {
        let ff = video::find_ffmpeg(&self.exe_dir).ok_or(
            "для видео и камер нужен ffmpeg.exe: скачайте «ffmpeg-release-essentials.zip» с \
             https://www.gyan.dev/ffmpeg/builds/ и положите bin\\ffmpeg.exe рядом с nomerok.exe",
        )?;
        let stream = video::is_stream(input);
        let src_name = if stream {
            input.to_string()
        } else {
            Path::new(input).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
        };
        let mut attempt = 0u32;
        loop {
            let started = Instant::now();
            let r = self.video_once(&ff, input, stream, &src_name);
            if !stream {
                return r;
            }
            // камера: переподключаемся, пока программу не закроют
            if let Err(e) = &r {
                eprintln!("камера: {e}");
            }
            attempt = if started.elapsed().as_secs() > 60 { 1 } else { attempt + 1 };
            let wait = (5 * attempt).min(60);
            eprintln!("поток прервался — переподключение через {wait} с (Ctrl+C — выход)");
            self.rep.write_html();
            std::thread::sleep(std::time::Duration::from_secs(wait as u64));
        }
    }

    fn video_once(&mut self, ff: &Path, input: &str, stream: bool, src: &str) -> Res<()> {
        let fps = self.o.fps;
        let mut fr = video::Frames::open(ff, input, fps)?;
        let mut tracker = track::Tracker::new(self.o.gap);
        let mut writer: Option<video::Writer> = None;
        let mut early: Vec<u64> = Vec::new();
        let t0 = Instant::now();
        let mut last_print = Instant::now();
        let mut t = 0.0f64;
        let mut work = 0.0f64;
        while let Some(img) = fr.next_frame() {
            let tw = Instant::now();
            t = if stream {
                t0.elapsed().as_secs_f64()
            } else {
                (fr.index - 1) as f64 / fps as f64
            };
            // ШАГ 2 — номера на кадре, ШАГИ 3–5 — вырезать и прочитать
            let dets: Vec<Det> = self.det.detect(&img)?.into_iter().take(8).collect();
            let seen = self.read_all(&img, &dets)?;
            // ШАГ 6 — связать с номерами прошлых кадров (одна машина — один трек)
            let reads: Vec<Reading> = seen.iter().map(|s| s.read.clone()).collect();
            let ids = tracker.update(&dets, &reads, t);
            let mut ann = img;
            for (k, d) in dets.iter().enumerate() {
                let tr = &tracker.tracks[ids[k]];
                let v = tr.verdict();
                let (text, conf) = v.map(|v| (v.display, v.conf)).unwrap_or_default();
                self.annotate(&mut ann, d, seen[k].corners.as_ref(), &text, conf);
            }
            // лучший снимок трека: самый уверенный и крупный номер
            for (k, &i) in ids.iter().enumerate() {
                let w = tracker.tracks[i].obs.last().map(|o| o.weight).unwrap_or(0.0);
                if tracker.tracks[i].best.as_ref().is_none_or(|b| w > b.0) {
                    tracker.tracks[i].best = Some((w, seen[k].clone(), imgops::fit_width(&ann, 1280)));
                }
            }
            if self.o.video {
                if writer.is_none() {
                    writer = Some(video::Writer::open(ff, &self.rep.dir.join("annotated.mp4"), ann.width(), ann.height(), fps)?);
                }
                if let Some(w) = writer.as_mut() {
                    w.push(&ann);
                }
            }
            // уверенный номер у стоящей машины (шлагбаум) выдаём сразу, не дожидаясь отъезда
            let mut ready = Vec::new();
            for tr in &tracker.tracks {
                if tr.obs.len() >= 4 && !early.contains(&tr.id) {
                    if let Some(v) = tr.verdict().filter(|v| v.conf >= self.o.sure) {
                        ready.push((tr.id, v));
                    }
                }
            }
            for (id, v) in ready {
                early.push(id);
                let tr = tracker.tracks.iter().find(|x| x.id == id).unwrap();
                let (crop, frame) = tr.best.as_ref().map(|b| (b.1.clone(), b.2.clone())).unwrap();
                let first = tr.first_t;
                self.emit_verdict(stream, first, &v, src, &crop, &frame);
            }
            for tr in tracker.finished(t, false) {
                self.finish_track(stream, &tr, &mut early, src);
            }
            work += tw.elapsed().as_secs_f64();
            if last_print.elapsed().as_secs() >= 3 {
                last_print = Instant::now();
                eprint!("\r  кадр {} ({}), обработка {:.1} кадр/с   ", fr.index,
                    if stream { now_str() } else { hms(t) }, fr.index as f64 / work.max(1e-3));
            }
        }
        for tr in tracker.finished(t, true) {
            self.finish_track(stream, &tr, &mut early, src);
        }
        if let Some(w) = writer {
            w.finish();
        }
        let errs = fr.errors();
        if fr.index == 0 {
            return Err(if errs.is_empty() {
                "ffmpeg не выдал ни одного кадра".into()
            } else {
                format!("ffmpeg: {errs}")
            });
        }
        if stream {
            return Err(if errs.is_empty() { "поток закончился".into() } else { errs });
        }
        eprintln!("\r  обработано кадров: {}, скорость {:.1} кадр/с{:30}", fr.index, fr.index as f64 / work.max(1e-3), "");
        Ok(())
    }

    fn finish_track(&mut self, stream: bool, tr: &track::Track, early: &mut Vec<u64>, src: &str) {
        if let Some(pos) = early.iter().position(|&i| i == tr.id) {
            early.remove(pos);
            return;
        }
        let Some(v) = tr.verdict() else { return };
        // одиночный неуверенный кадр — скорее всего ложная находка
        if v.reads == 1 && v.conf < 0.5 {
            return;
        }
        if let Some((_, crop, frame)) = &tr.best {
            let (crop, frame) = (crop.clone(), frame.clone());
            self.emit_verdict(stream, tr.first_t, &v, src, &crop, &frame);
        }
    }

    fn emit_verdict(&mut self, stream: bool, t: f64, v: &track::Verdict, src: &str, crop: &Seen, frame: &Rgb) {
        if let Some(&last) = self.emitted.get(&v.latin) {
            if t - last < self.o.repeat && v.format.is_some() {
                return;
            }
        }
        self.emitted.insert(v.latin.clone(), t);
        let time = if stream { now_str() } else { hms(t) };
        let fmt = v.format.map(|f| plate::FORMATS[f].name).unwrap_or("не РФ / не распознан формат");
        self.emit(time, &v.latin, &v.display, v.conf, fmt, v.reads, src, crop, Some(frame));
    }
}

fn out_dir(o: &Opts, input: &str) -> PathBuf {
    if let Some(d) = &o.out {
        return d.clone();
    }
    let base = if video::is_stream(input) {
        std::env::current_dir()
            .unwrap_or_default()
            .join(format!("nomerok_{}", chrono::Local::now().format("%Y%m%d_%H%M%S")))
    } else {
        let p = Path::new(input);
        if p.is_dir() {
            p.join("_nomerok")
        } else {
            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            p.with_file_name(format!("{stem}_nomerok"))
        }
    };
    if !base.exists() {
        return base;
    }
    (2..1000)
        .map(|i| PathBuf::from(format!("{}_{i}", base.display())))
        .find(|p| !p.exists())
        .unwrap_or(base)
}

/// Запущен двойным щелчком / перетаскиванием (окно консоли только наше) —
/// тогда в конце ждём Enter, чтобы окно не закрылось сразу.
fn own_console() -> bool {
    #[cfg(windows)]
    unsafe {
        let mut ids = [0u32; 4];
        windows_sys::Win32::System::Console::GetConsoleProcessList(ids.as_mut_ptr(), 4) <= 1
    }
    #[cfg(not(windows))]
    false
}

fn open_in_shell(p: &Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer.exe").arg(p).spawn();
    }
    #[cfg(not(windows))]
    let _ = p;
}

fn read_line(prompt: &str) -> String {
    use std::io::Write;
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
    s.trim().trim_matches('"').to_string()
}

fn run(o: Opts, exe_dir: PathBuf) -> Res<PathBuf> {
    let input = match o.input.clone() {
        Some(i) => i,
        None => {
            println!("{HELP}\n");
            let s = read_line("Перетащите сюда файл или папку, или введите адрес камеры, и нажмите Enter:\n> ");
            if s.is_empty() {
                return Err("ничего не выбрано".into());
            }
            s
        }
    };
    if !video::is_stream(&input) && !Path::new(&input).exists() {
        return Err(format!("не найдено: {input}"));
    }
    engine::init_runtime(&exe_dir)?;
    let models = exe_dir.join("models");
    let det_path = o.det.clone().or_else(|| Some(models.join("detector.onnx")).filter(|p| p.is_file()));
    let ocr_path = o.ocr.clone().or_else(|| Some(models.join("ocr.onnx")).filter(|p| p.is_file()));
    let ocr_cfg = o.ocr_cfg.clone().or_else(|| ocr_path.as_ref().map(|p| p.with_extension("yaml")));
    let det = Detector::new(det_path.as_deref(), o.threads, o.det_conf)?;
    let mut ocr = match (&ocr_path, &ocr_cfg) {
        (Some(m), Some(c)) => Reader::new(Some((m, c)), o.threads)?,
        _ => Reader::new(None, o.threads)?,
    };
    ocr.allowed = o.formats.clone().unwrap_or_default();
    let dir = out_dir(&o, &input);
    let rep = Report::create(&dir, &input)?;
    println!("Номерок {} · результат: {}\n", env!("CARGO_PKG_VERSION"), dir.display());
    // вариант 2: сеть углов (своя модель — --corners или models\\corners.onnx)
    let corner_net = if o.rectify.net() {
        let p = o.corners.clone().or_else(|| Some(models.join("corners.onnx")).filter(|p| p.is_file()));
        Some(CornerNet::new(p.as_deref(), o.threads)?)
    } else {
        None
    };
    let mut cx = Ctx { det, ocr, corner_net, o, rep, exe_dir, emitted: HashMap::new() };
    let t0 = Instant::now();
    let p = PathBuf::from(&input);
    if video::is_stream(&input) {
        cx.video(&input)?;
    } else if p.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&p)
            .map_err(|e| format!("{input}: {e}"))?
            .flatten()
            .map(|e| e.path())
            .filter(|f| f.is_file())
            .collect();
        files.sort();
        for f in files {
            let ext = ext_of(&f);
            let r = if IMAGE_EXT.contains(&ext.as_str()) {
                cx.photo(&f)
            } else if VIDEO_EXT.contains(&ext.as_str()) {
                cx.emitted.clear();
                cx.video(&f.display().to_string())
            } else {
                continue;
            };
            if let Err(e) = r {
                eprintln!("{}: {e}", f.display());
            }
        }
    } else if IMAGE_EXT.contains(&ext_of(&p).as_str()) {
        cx.photo(&p)?;
    } else {
        cx.video(&input)?;
    }
    cx.rep.write_html();
    let n = cx.rep.records.len();
    let sure = cx.rep.records.iter().filter(|r| r.sure).count();
    println!("\nГотово за {:.1} с: номеров {n} (уверенно {sure}, не уверен {}).", t0.elapsed().as_secs_f64(), n - sure);
    println!("Отчёт: {}", dir.join("report.html").display());
    if cx.o.open && n > 0 {
        open_in_shell(&dir.join("report.html"));
    }
    Ok(dir)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    let pause = own_console();
    let code = match parse_args(&args) {
        Err(e) if e.is_empty() => {
            println!("{HELP}");
            0
        }
        Err(e) => {
            eprintln!("Ошибка: {e}");
            2
        }
        Ok(o) => match run(o, exe_dir) {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("\nОшибка: {e}");
                1
            }
        },
    };
    if pause {
        read_line("\nНажмите Enter, чтобы закрыть окно…");
    }
    std::process::exit(code);
}
