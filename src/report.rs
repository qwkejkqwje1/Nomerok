//! Результаты: plates.csv (для Excel), plates.jsonl (для программ) и report.html
//! со снимками номеров. CSV и JSONL дописываются сразу после каждой записи —
//! при работе с камерой ничего не теряется, даже если закрыть окно.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Serialize)]
pub struct Record {
    pub n: usize,
    /// Время: метка в видео «00:01:23.4», дата-время для камеры, пусто для фото.
    pub time: String,
    pub plate: String,
    pub plate_latin: String,
    pub conf: f32,
    pub sure: bool,
    pub format: String,
    pub frames: usize,
    pub source: String,
    pub crop: String,
    pub frame: String,
    /// Совпал со списком --find.
    pub found: bool,
    pub raw: String,
}

pub struct Report {
    pub dir: PathBuf,
    csv: File,
    jsonl: File,
    pub records: Vec<Record>,
    title: String,
    html_at: Instant,
}

fn csv_field(s: &str) -> String {
    if s.contains([';', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Report {
    pub fn create(dir: &Path, title: &str) -> Result<Report, String> {
        for sub in ["", "crops", "frames"] {
            std::fs::create_dir_all(dir.join(sub))
                .map_err(|e| format!("не удалось создать {}: {e}", dir.display()))?;
        }
        let open = |n: &str| {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(dir.join(n))
                .map_err(|e| format!("{n}: {e}"))
        };
        let mut csv = open("plates.csv")?;
        // BOM и «;» — чтобы русский Excel открыл файл двойным щелчком
        let _ = csv.write_all(
            "\u{feff}№;Время;Номер;Номер (латиница);Уверенность, %;Статус;Формат;Кадров;Источник;Снимок номера;Кадр;Совпадение\r\n"
                .as_bytes(),
        );
        Ok(Report {
            dir: dir.to_path_buf(),
            csv,
            jsonl: open("plates.jsonl")?,
            records: Vec::new(),
            title: title.to_string(),
            html_at: Instant::now(),
        })
    }

    pub fn add(&mut self, r: Record) {
        let line = [
            r.n.to_string(),
            r.time.clone(),
            r.plate.clone(),
            r.plate_latin.clone(),
            format!("{:.0}", r.conf * 100.0),
            if r.sure { "уверен" } else { "не уверен" }.to_string(),
            r.format.clone(),
            r.frames.to_string(),
            r.source.clone(),
            r.crop.clone(),
            r.frame.clone(),
            if r.found { "да" } else { "" }.to_string(),
        ]
        .iter()
        .map(|s| csv_field(s))
        .collect::<Vec<_>>()
        .join(";");
        let _ = write!(self.csv, "{line}\r\n");
        let _ = self.csv.flush();
        if let Ok(j) = serde_json::to_string(&r) {
            let _ = writeln!(self.jsonl, "{j}");
            let _ = self.jsonl.flush();
        }
        self.records.push(r);
        if self.html_at.elapsed().as_secs() >= 5 {
            self.write_html();
        }
    }

    pub fn write_html(&mut self) {
        self.html_at = Instant::now();
        let sure = self.records.iter().filter(|r| r.sure).count();
        let mut rows = String::new();
        for r in self.records.iter().rev() {
            let cls = if r.found {
                "found"
            } else if r.sure {
                ""
            } else {
                "unsure"
            };
            let frame = if r.frame.is_empty() {
                String::new()
            } else {
                format!("<a href=\"{0}\" target=\"_blank\"><img class=\"fr\" src=\"{0}\" loading=\"lazy\"></a>", esc(&r.frame))
            };
            rows.push_str(&format!(
                "<tr class=\"{cls}\"><td>{n}</td><td>{time}</td><td><img class=\"crop\" src=\"{crop}\" loading=\"lazy\"></td>\
                 <td class=\"plate\">{plate}<div class=\"raw\">{fmt}</div></td><td>{conf:.0}%<div class=\"raw\">{st}</div></td>\
                 <td>{frames}</td><td class=\"src\">{src}</td><td>{frame}</td></tr>\n",
                n = r.n,
                time = esc(&r.time),
                crop = esc(&r.crop),
                plate = esc(&r.plate),
                fmt = esc(&r.format),
                conf = r.conf * 100.0,
                st = if r.found {
                    "СОВПАДЕНИЕ"
                } else if r.sure {
                    "уверен"
                } else {
                    "не уверен"
                },
                frames = r.frames,
                src = esc(&r.source),
            ));
        }
        let html = format!(
            r#"<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Номерок — {title}</title>
<style>
body{{font:14px/1.4 "Segoe UI",system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1d2228}}
header{{background:#1d2228;color:#fff;padding:16px 24px}} header h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;color:#aab2bd}}
main{{padding:16px 24px}} table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0002}}
th,td{{padding:8px 10px;border-bottom:1px solid #e6e8eb;text-align:left;vertical-align:middle}} th{{background:#eef0f3;font-weight:600}}
.plate{{font:600 22px/1.1 Consolas,monospace;letter-spacing:1px}} .raw{{font:12px "Segoe UI",sans-serif;color:#7b8794;letter-spacing:0}}
.crop{{height:44px;border:1px solid #ccc;border-radius:4px;image-rendering:auto}} .fr{{height:64px;border-radius:4px}}
tr.unsure td{{background:#fff8e6}} tr.found td{{background:#e3f7e9}} .src{{color:#7b8794;font-size:12px;max-width:220px;word-break:break-all}}
</style></head><body><header><h1>Номерок — найденные номера</h1><p>{title} · записей: {total}, уверенно: {sure}, не уверен: {unsure}</p></header>
<main><table><tr><th>№</th><th>Время</th><th>Снимок</th><th>Номер</th><th>Уверенность</th><th>Кадров</th><th>Источник</th><th>Кадр</th></tr>
{rows}</table></main></body></html>"#,
            title = esc(&self.title),
            total = self.records.len(),
            unsure = self.records.len() - sure,
        );
        let _ = std::fs::write(self.dir.join("report.html"), html);
    }
}
