//! ШАГ 6 (только видео): связываем номера между кадрами (одна машина — один
//! трек), читаем номер на каждом кадре и голосуем по символам. На выходе —
//! одна запись на машину.

use crate::engine::Det;
use crate::imgops::Rgb;
use crate::Seen;
use crate::plate::{self, Reading, FORMATS};

#[allow(dead_code)]
pub struct Obs {
    pub det: Det,
    pub read: Reading,
    pub t: f64,
    pub weight: f32,
}

pub struct Track {
    pub id: u64,
    pub last: Det,
    vel: (f32, f32),
    pub first_t: f64,
    pub last_t: f64,
    pub obs: Vec<Obs>,
    /// Лучший снимок: (вес, вырезанный номер, кадр с рамкой).
    pub best: Option<(f32, Seen, Rgb)>,
}

/// Итог голосования.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub latin: String,
    pub display: String,
    pub format: Option<usize>,
    pub conf: f32,
    pub reads: usize,
}

impl Track {
    fn predict(&self, t: f64) -> Det {
        let dt = (t - self.last_t) as f32;
        let (dx, dy) = (self.vel.0 * dt, self.vel.1 * dt);
        Det {
            x1: self.last.x1 + dx,
            y1: self.last.y1 + dy,
            x2: self.last.x2 + dx,
            y2: self.last.y2 + dy,
            score: self.last.score,
        }
    }

    pub fn verdict(&self) -> Option<Verdict> {
        vote(&self.obs)
    }
}

/// Вес прочтения: уверенность × размер номера (мелкие номера читаются хуже).
pub fn weight(read: &Reading, det: &Det) -> f32 {
    read.conf.max(0.02) * (det.w() / 120.0).clamp(0.25, 1.0)
}

pub fn vote(obs: &[Obs]) -> Option<Verdict> {
    if obs.is_empty() {
        return None;
    }
    // 1) какой формат (длина и расстановка букв/цифр) набрал больше веса
    let key = |o: &Obs| (o.read.format, o.read.latin.chars().count());
    let mut groups: Vec<((Option<usize>, usize), f32)> = Vec::new();
    for o in obs {
        let k = key(o);
        match groups.iter_mut().find(|g| g.0 == k) {
            Some(g) => g.1 += o.weight,
            None => groups.push((k, o.weight)),
        }
    }
    groups.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (gk, _) = groups[0];
    let members: Vec<&Obs> = obs.iter().filter(|o| key(o) == gk).collect();
    let n = gk.1;
    // 2) посимвольно: вес символа = вес кадра × вероятность символа
    let mut latin = String::new();
    let mut min_share = 1.0f32;
    for i in 0..n {
        let mut votes: Vec<(char, f32)> = Vec::new();
        for o in &members {
            let c = o.read.latin.chars().nth(i).unwrap_or('?');
            let w = o.weight * o.read.probs.get(i).copied().unwrap_or(0.5);
            match votes.iter_mut().find(|v| v.0 == c) {
                Some(v) => v.1 += w,
                None => votes.push((c, w)),
            }
        }
        let total: f32 = votes.iter().map(|v| v.1).sum();
        let (c, w) = votes
            .iter()
            .cloned()
            .fold(('?', -1.0), |a, b| if b.1 > a.1 { b } else { a });
        latin.push(c);
        if total > 0.0 {
            min_share = min_share.min(w / total);
        }
    }
    // 3) уверенность: согласие кадров × «хотя бы одно из лучших прочтений верно»
    let mut agree: Vec<f32> = members
        .iter()
        .filter(|o| o.read.latin == latin)
        .map(|o| o.read.conf)
        .collect();
    agree.sort_by(|a, b| b.total_cmp(a));
    let p_any = if agree.is_empty() {
        members.iter().map(|o| o.read.conf).fold(0.0, f32::max) * 0.5
    } else {
        1.0 - agree.iter().take(3).map(|c| 1.0 - c).product::<f32>()
    };
    let total_w: f32 = obs.iter().map(|o| o.weight).sum();
    let group_share = groups[0].1 / total_w.max(1e-6);
    let conf = (min_share * p_any * group_share.sqrt()).clamp(0.0, 1.0);
    let region = gk.0.map(|f| FORMATS[f].region).unwrap_or(0);
    Some(Verdict {
        display: plate::display(&latin, region),
        latin,
        format: gk.0,
        conf,
        reads: obs.len(),
    })
}

pub struct Tracker {
    pub tracks: Vec<Track>,
    next_id: u64,
    /// Сколько секунд трек может не обновляться, прежде чем машина «уехала».
    pub gap: f64,
}

impl Tracker {
    pub fn new(gap: f64) -> Tracker {
        Tracker {
            tracks: Vec::new(),
            next_id: 1,
            gap,
        }
    }

    /// Привязывает номера кадра к трекам. Возвращает индекс трека для каждого номера.
    pub fn update(&mut self, dets: &[Det], reads: &[Reading], t: f64) -> Vec<usize> {
        let mut used = vec![false; self.tracks.len()];
        let mut out = Vec::with_capacity(dets.len());
        for (d, r) in dets.iter().zip(reads) {
            let mut best: Option<(usize, f32)> = None;
            for (i, tr) in self.tracks.iter().enumerate() {
                if used[i] || t - tr.last_t > self.gap {
                    continue;
                }
                let p = tr.predict(t);
                let iou = d.iou(&p);
                let (cx, cy) = ((d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0);
                let (px, py) = ((p.x1 + p.x2) / 2.0, (p.y1 + p.y2) / 2.0);
                let dist = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt() / d.w().max(p.w()).max(1.0);
                let ratio = d.w() / p.w().max(1.0);
                let same_text = tr.obs.last().is_some_and(|o| o.read.latin == r.latin);
                let s = if iou > 0.1 {
                    iou
                } else if dist < 1.5 && (0.5..2.0).contains(&ratio) {
                    0.1 * (1.5 - dist) / 1.5
                } else if same_text && dist < 4.0 {
                    0.01
                } else {
                    continue;
                };
                if best.is_none_or(|b| s > b.1) {
                    best = Some((i, s));
                }
            }
            let w = weight(r, d);
            let obs = Obs {
                det: *d,
                read: r.clone(),
                t,
                weight: w,
            };
            match best {
                Some((i, _)) => {
                    used[i] = true;
                    let tr = &mut self.tracks[i];
                    let dt = (t - tr.last_t).max(1e-3) as f32;
                    let v = (
                        ((d.x1 + d.x2) - (tr.last.x1 + tr.last.x2)) / 2.0 / dt,
                        ((d.y1 + d.y2) - (tr.last.y1 + tr.last.y2)) / 2.0 / dt,
                    );
                    tr.vel = if tr.obs.len() == 1 {
                        v
                    } else {
                        (tr.vel.0 * 0.5 + v.0 * 0.5, tr.vel.1 * 0.5 + v.1 * 0.5)
                    };
                    tr.last = *d;
                    tr.last_t = t;
                    tr.obs.push(obs);
                    out.push(i);
                }
                None => {
                    self.tracks.push(Track {
                        id: self.next_id,
                        last: *d,
                        vel: (0.0, 0.0),
                        first_t: t,
                        last_t: t,
                        obs: vec![obs],
                        best: None,
                    });
                    self.next_id += 1;
                    used.push(true);
                    out.push(self.tracks.len() - 1);
                }
            }
        }
        out
    }

    /// Забирает треки, которые давно не обновлялись (или все — в конце видео).
    pub fn finished(&mut self, t: f64, all: bool) -> Vec<Track> {
        let mut done = Vec::new();
        let mut i = 0;
        while i < self.tracks.len() {
            if all || t - self.tracks[i].last_t > self.gap {
                done.push(self.tracks.remove(i));
            } else {
                i += 1;
            }
        }
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(latin: &str, conf: f32) -> Reading {
        let a: Vec<char> = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_".chars().collect();
        let k = a.len();
        let mut p = vec![0.0f32; 10 * k];
        for s in 0..10 {
            let c = latin.chars().nth(s).unwrap_or('_');
            let ci = a.iter().position(|&x| x == c).unwrap();
            for i in 0..k {
                p[s * k + i] = (1.0 - conf) / (k as f32 - 1.0);
            }
            p[s * k + ci] = conf;
        }
        plate::decode(&p, 10, &a, '_')
    }

    fn det(x: f32) -> Det {
        Det { x1: x, y1: 100.0, x2: x + 120.0, y2: 128.0, score: 0.9 }
    }

    #[test]
    fn votes_fix_one_bad_frame() {
        let obs: Vec<Obs> = [("A123BC77", 0.9), ("A123BC77", 0.8), ("A128BC77", 0.85), ("A123BC77", 0.7)]
            .iter()
            .enumerate()
            .map(|(i, (s, c))| {
                let r = rd(s, *c);
                Obs { det: det(i as f32 * 10.0), weight: weight(&r, &det(0.0)), read: r, t: i as f64 }
            })
            .collect();
        let v = vote(&obs).unwrap();
        assert_eq!(v.display, "А123ВС 77");
        assert!(v.conf > 0.5, "{}", v.conf);
        assert_eq!(v.reads, 4);
    }

    #[test]
    fn tracks_moving_plate_and_splits_two_cars() {
        let mut t = Tracker::new(1.0);
        for i in 0..5 {
            let ds = [det(100.0 + i as f32 * 40.0), det(900.0 - i as f32 * 30.0)];
            let rs = [rd("A123BC77", 0.9), rd("K777OO197", 0.9)];
            t.update(&ds, &rs, i as f64 * 0.2);
        }
        assert_eq!(t.tracks.len(), 2);
        assert!(t.finished(1.0, false).is_empty());
        let done = t.finished(3.0, false);
        assert_eq!(done.len(), 2);
        let names: Vec<String> = done.iter().map(|d| d.verdict().unwrap().display).collect();
        assert!(names.contains(&"А123ВС 77".to_string()) && names.contains(&"К777ОО 197".to_string()));
    }
}
