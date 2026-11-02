//! ШАГ 4б. Номера РФ: форматы по ГОСТ Р 50577, коды регионов, декодирование
//! выхода распознавателя с учётом формата и вывод кириллицей.
//!
//! Распознаватель выдаёт для каждой из N позиций вероятности всех символов
//! алфавита (+ символ-заполнитель). Вместо «самый вероятный символ на каждой
//! позиции» перебираем допустимые шаблоны (буква/цифра на каждом месте) и
//! выбираем шаблон с наибольшей суммарной вероятностью. Так «0» на месте
//! буквы превращается в «О», «8» — в «В» и т. д.

/// 12 букв, допустимых в номерах РФ (латинские двойники кириллицы).
pub const LETTERS: &str = "ABEKMHOPCTYX";

/// Виды российских номеров (ГОСТ Р 50577-2018 «Знаки государственные
/// регистрационные транспортных средств»). У разных видов транспорта разный
/// порядок букв и цифр — это не прихоть программы, а так выдаёт ГИБДД:
///
/// | вид                                | пример       | шаблон       |
/// |------------------------------------|--------------|--------------|
/// | легковые, грузовые                 | А123ВС 77    | Б ЦЦЦ ББ рег |
/// | такси (жёлтые), автобусы           | АВ123 77     | ББ ЦЦЦ рег   |
/// | прицепы                            | АВ1234 77    | ББ ЦЦЦЦ рег  |
/// | полиция (синие)                    | А1234 77     | Б ЦЦЦЦ рег   |
/// | мотоциклы, военные, тракторы       | 1234 АВ 77   | ЦЦЦЦ ББ рег  |
/// | транзитные                         | АВ123С 77    | ББ ЦЦЦ Б рег |
///
/// Б — буква, Ц — цифра, «рег» — код региона из 2 или 3 цифр в правой части
/// таблички. Букв всего 12 — те, что пишутся одинаково кириллицей и латиницей:
/// А В Е К М Н О Р С Т У Х.
/// В коде шаблон записан строкой: L — место буквы, D — место цифры,
/// например «LDDDLLDD» = А123ВС77.
///
/// Зачем это нужно: распознаватель иногда путает похожие знаки (О и 0, В и 8).
/// Зная, где по шаблону должна стоять буква, а где цифра, такие ошибки
/// исправляем (см. decode_shaped). Если нужны не все виды, лишние можно
/// отключить параметром --formats — тогда ошибок вида «прочитал легковой номер
/// как полицейский» станет ещё меньше.
pub struct Format {
    /// L — буква, D — цифра.
    pub pattern: &'static str,
    /// Название для отчёта.
    pub name: &'static str,
    /// Вид для параметра --formats.
    pub kind: Kind,
    /// Сколько последних символов — код региона.
    pub region: usize,
    /// Насколько вид редок (логарифм априорной вероятности): легковых номеров
    /// на дорогах подавляющее большинство, поэтому при сомнении выигрывают они.
    pub prior: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Car,
    Taxi,
    Trailer,
    Police,
    Moto,
    Transit,
}

/// Названия видов для --formats (можно по-русски или по-английски).
pub const KINDS: &[(Kind, &[&str])] = &[
    (Kind::Car, &["легковые", "легковой", "обычные", "car"]),
    (Kind::Taxi, &["такси", "автобусы", "taxi", "bus"]),
    (Kind::Trailer, &["прицепы", "прицеп", "trailer"]),
    (Kind::Police, &["полиция", "police"]),
    (Kind::Moto, &["мото", "мотоциклы", "военные", "moto"]),
    (Kind::Transit, &["транзит", "транзитные", "transit"]),
];

pub const FORMATS: &[Format] = &[
    Format { pattern: "LDDDLLDD", name: "легковой", kind: Kind::Car, region: 2, prior: 0.0 },
    Format { pattern: "LDDDLLDDD", name: "легковой", kind: Kind::Car, region: 3, prior: 0.0 },
    Format { pattern: "LLDDDDD", name: "такси/автобус", kind: Kind::Taxi, region: 2, prior: -2.0 },
    Format { pattern: "LLDDDDDD", name: "такси/автобус", kind: Kind::Taxi, region: 3, prior: -2.0 },
    Format { pattern: "LLDDDDDD", name: "прицеп", kind: Kind::Trailer, region: 2, prior: -2.5 },
    Format { pattern: "LLDDDDDDD", name: "прицеп", kind: Kind::Trailer, region: 3, prior: -2.5 },
    Format { pattern: "DDDDLLDD", name: "мото/военный", kind: Kind::Moto, region: 2, prior: -2.5 },
    Format { pattern: "DDDDLLDDD", name: "мото/военный", kind: Kind::Moto, region: 3, prior: -2.5 },
    Format { pattern: "LDDDDDD", name: "полиция", kind: Kind::Police, region: 2, prior: -3.0 },
    Format { pattern: "LDDDDDDD", name: "полиция", kind: Kind::Police, region: 3, prior: -3.0 },
    Format { pattern: "LLDDDLDD", name: "транзит", kind: Kind::Transit, region: 2, prior: -3.0 },
    Format { pattern: "LLDDDLDDD", name: "транзит", kind: Kind::Transit, region: 3, prior: -3.0 },
];

/// «легковые,такси» → какие строки FORMATS разрешены.
pub fn parse_kinds(list: &str) -> Result<Vec<bool>, String> {
    let mut on = vec![false; FORMATS.len()];
    for w in list.split([',', ';', ' ']).map(|w| w.trim().to_lowercase()).filter(|w| !w.is_empty()) {
        let kind = KINDS
            .iter()
            .find(|(_, names)| names.contains(&w.as_str()))
            .map(|(k, _)| *k)
            .ok_or(format!("--formats: неизвестный вид «{w}» (легковые, такси, прицепы, полиция, мото, транзит)"))?;
        for (i, f) in FORMATS.iter().enumerate() {
            on[i] |= f.kind == kind;
        }
    }
    Ok(on)
}

/// Трёхзначные коды регионов (двузначные 01–99 считаются допустимыми все).
const REGIONS3: &[u16] = &[
    102, 113, 116, 121, 122, 123, 124, 125, 126, 134, 136, 138, 142, 147, 150, 152, 154, 155,
    156, 159, 161, 163, 164, 173, 174, 177, 178, 180, 181, 184, 185, 186, 190, 193, 196, 197,
    198, 199, 250, 252, 277, 299, 702, 716, 725, 750, 761, 763, 774, 777, 790, 797, 799, 977,
];

pub fn region_ok(code: &str) -> bool {
    match code.len() {
        2 => code != "00" && code.bytes().all(|b| b.is_ascii_digit()),
        3 => code.parse::<u16>().is_ok_and(|n| REGIONS3.contains(&n)),
        _ => false,
    }
}

/// Латиница → кириллица (только 12 букв номеров, остальное как есть).
pub fn to_cyr(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A' => 'А',
            'B' => 'В',
            'E' => 'Е',
            'K' => 'К',
            'M' => 'М',
            'H' => 'Н',
            'O' => 'О',
            'P' => 'Р',
            'C' => 'С',
            'T' => 'Т',
            'Y' => 'У',
            'X' => 'Х',
            c => c,
        })
        .collect()
}

/// Кириллица/любой регистр → латинская форма для сравнения со списками.
pub fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| match c.to_uppercase().next().unwrap_or(c) {
            'А' => 'A',
            'В' => 'B',
            'Е' => 'E',
            'К' => 'K',
            'М' => 'M',
            'Н' => 'H',
            'О' => 'O',
            'Р' => 'P',
            'С' => 'C',
            'Т' => 'T',
            'У' => 'Y',
            'Х' => 'X',
            c => c,
        })
        .collect()
}

/// «A123BC77» → «А123ВС 77» (регион отделён пробелом).
pub fn display(latin: &str, region: usize) -> String {
    let cyr = to_cyr(latin);
    let n = cyr.chars().count();
    if region == 0 || region >= n {
        return cyr;
    }
    let main: String = cyr.chars().take(n - region).collect();
    let reg: String = cyr.chars().skip(n - region).collect();
    format!("{main} {reg}")
}

/// Результат чтения одного снимка номера.
#[derive(Clone, Debug)]
pub struct Reading {
    /// Латиницей, без пробелов: «A123BC77».
    pub latin: String,
    /// Вероятность каждого символа.
    pub probs: Vec<f32>,
    /// Уверенность: минимальная вероятность символа (и «пустоты» после номера).
    pub conf: f32,
    /// Индекс в FORMATS или None, если номер не похож на российский.
    pub format: Option<usize>,
    /// Как прочитала модель без учёта формата.
    pub raw: String,
}

impl Reading {
    pub fn region_len(&self) -> usize {
        self.format.map(|f| FORMATS[f].region).unwrap_or(0)
    }
    pub fn display(&self) -> String {
        display(&self.latin, self.region_len())
    }
    pub fn format_name(&self) -> &'static str {
        self.format.map(|f| FORMATS[f].name).unwrap_or("не РФ / не распознан формат")
    }
}

/// Декодирует выход распознавателя: `probs[slot * k + class]`.
#[allow(dead_code)]
pub fn decode(probs: &[f32], slots: usize, alphabet: &[char], pad: char) -> Reading {
    decode_shaped(probs, slots, alphabet, pad, None, &[])
}

/// То же с учётом формы таблички: `one_line` — вытянутая однострочная
/// (ширина/высота больше ~2,6), иначе почти квадратная двухстрочная
/// (мотоциклы, некоторые прицепы). Форма меняет априорную частоту форматов.
/// `allowed` — какие строки FORMATS разрешены (пусто — все).
pub fn decode_shaped(
    probs: &[f32],
    slots: usize,
    alphabet: &[char],
    pad: char,
    one_line: Option<bool>,
    allowed: &[bool],
) -> Reading {
    let k = alphabet.len();
    let idx = |c: char| alphabet.iter().position(|&a| a == c);
    let pad_i = idx(pad);
    let letters: Vec<usize> = LETTERS.chars().filter_map(idx).collect();
    let digits: Vec<usize> = ('0'..='9').filter_map(idx).collect();
    // Визуально одинаковые пары: если модель уверена в «0» там, где по формату
    // буква, считаем, что это может быть «О» (с понижающим коэффициентом).
    let twins: Vec<(usize, usize, f32)> = [('O', '0', 0.25), ('B', '8', 0.15)]
        .iter()
        .filter_map(|&(l, d, f)| Some((idx(l)?, idx(d)?, f)))
        .flat_map(|(l, d, f)| [(l, d, f), (d, l, f)])
        .collect();
    let eff = |slot: usize, i: usize| -> f32 {
        let mut p = probs[slot * k + i];
        for &(a, b, f) in &twins {
            if a == i {
                p = p.max(probs[slot * k + b] * f);
            }
        }
        p
    };
    let best_of = |slot: usize, set: &[usize]| -> (usize, f32) {
        set.iter()
            .map(|&i| (i, eff(slot, i)))
            .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a })
    };
    let pad_p = |slot: usize| pad_i.map(|i| probs[slot * k + i]).unwrap_or(0.0);

    // как есть
    let mut raw = String::new();
    let mut raw_probs = Vec::new();
    let mut raw_conf = 1.0f32;
    let mut raw_score = 0.0f32;
    for s in 0..slots {
        let (i, p) = (0..k)
            .map(|i| (i, probs[s * k + i]))
            .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a });
        raw_conf = raw_conf.min(p);
        raw_score += p.max(1e-6).ln();
        if Some(i) != pad_i {
            raw.push(alphabet[i]);
            raw_probs.push(p);
        }
    }

    // лучший допустимый шаблон
    let mut best: Option<(f32, usize, String, Vec<f32>, f32)> = None;
    for (fi, f) in FORMATS.iter().enumerate() {
        let n = f.pattern.len();
        if n > slots || allowed.get(fi) == Some(&false) {
            continue;
        }
        let two_line_fmt = f.pattern.starts_with("DDDD");
        let mut score = match one_line {
            Some(true) if two_line_fmt => -4.5,
            Some(false) if two_line_fmt => 0.0,
            Some(false) => f.prior - 1.5,
            _ => f.prior,
        };
        let mut text = String::new();
        let mut ps = Vec::new();
        let mut conf = 1.0f32;
        for (s, t) in f.pattern.chars().enumerate() {
            let (i, p) = best_of(s, if t == 'L' { &letters } else { &digits });
            score += p.max(1e-6).ln();
            conf = conf.min(p);
            text.push(alphabet[i]);
            ps.push(p);
        }
        for s in n..slots {
            let p = pad_p(s);
            score += p.max(1e-6).ln();
            conf = conf.min(p);
        }
        // модель сама прочитала ровно этот шаблон — доверяем ей больше, чем «двойникам»
        if text == raw && n == raw.len() {
            score += 2.0;
        }
        if !region_ok(&text[n - f.region..]) {
            score += 0.3f32.ln();
            conf *= 0.5;
        }
        if best.as_ref().is_none_or(|b| score > b.0) {
            best = Some((score, fi, text, ps, conf));
        }
    }
    match best {
        // формат РФ не натягиваем на явно другой номер: тогда отдаём как есть
        Some((score, fi, text, ps, conf)) if score - raw_score > -7.0 => Reading {
            latin: text,
            probs: ps,
            conf,
            format: Some(fi),
            raw,
        },
        _ => Reading {
            latin: raw.clone(),
            probs: raw_probs,
            conf: raw_conf * 0.5,
            format: None,
            raw,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

    /// Вероятности: на каждой позиции заданный символ с p, остальное поровну.
    fn fake(slots: &[(char, f32, Option<(char, f32)>)], total: usize) -> Vec<f32> {
        let a: Vec<char> = ALPHA.chars().collect();
        let k = a.len();
        let mut v = vec![0.0; total * k];
        for s in 0..total {
            let (c, p, alt) = slots.get(s).cloned().unwrap_or(('_', 0.99, None));
            let rest = 1.0 - p - alt.map(|x| x.1).unwrap_or(0.0);
            for i in 0..k {
                v[s * k + i] = rest / (k as f32 - 1.0);
            }
            v[s * k + a.iter().position(|&x| x == c).unwrap()] = p;
            if let Some((c2, p2)) = alt {
                v[s * k + a.iter().position(|&x| x == c2).unwrap()] = p2;
            }
        }
        v
    }

    fn dec(s: &[(char, f32, Option<(char, f32)>)]) -> Reading {
        let a: Vec<char> = ALPHA.chars().collect();
        decode(&fake(s, 10), 10, &a, '_')
    }

    fn sure(t: &str) -> Vec<(char, f32, Option<(char, f32)>)> {
        t.chars().map(|c| (c, 0.95, None)).collect()
    }

    #[test]
    fn plain_plate() {
        let r = dec(&sure("A123BC77"));
        assert_eq!(r.latin, "A123BC77");
        assert_eq!(r.display(), "А123ВС 77");
        assert_eq!(r.format_name(), "легковой");
        assert!(r.conf > 0.9);
        let r = dec(&sure("K777OO197"));
        assert_eq!(r.display(), "К777ОО 197");
    }

    #[test]
    fn fixes_o_zero_by_position() {
        // модель считает, что на 1-м месте «0», а на 5-м «8», но буквы вторые по вероятности
        let mut s = sure("O123BC77");
        s[0] = ('0', 0.6, Some(('O', 0.35)));
        s[4] = ('8', 0.55, Some(('B', 0.4)));
        s[2] = ('O', 0.5, Some(('0', 0.45)));
        let r = dec(&s);
        assert_eq!(r.latin, "O103BC77");
        assert_eq!(r.raw, "01O38C77");
        assert!(r.conf < 0.5);
    }

    #[test]
    fn three_digit_region_and_taxi() {
        assert_eq!(dec(&sure("E001KX750")).display(), "Е001КХ 750");
        let r = dec(&sure("AB12377"));
        assert_eq!(r.format_name(), "такси/автобус");
        assert_eq!(r.display(), "АВ123 77");
        // неизвестный трёхзначный регион — уверенность падает
        assert!(dec(&sure("A123BC555")).conf < 0.6);
    }

    #[test]
    fn foreign_plate_kept_raw() {
        let r = dec(&sure("WOB1234"));
        assert_eq!(r.format, None);
        assert_eq!(r.latin, "WOB1234");
    }

    #[test]
    fn formats_filter() {
        let a: Vec<char> = ALPHA.chars().collect();
        // «АВ123 77» без такси в списке превращается в ближайший легковой или остаётся «не РФ»
        let only_cars = parse_kinds("легковые").unwrap();
        let r = decode_shaped(&fake(&sure("AB12377"), 10), 10, &a, '_', None, &only_cars);
        assert_ne!(r.format_name(), "такси/автобус");
        assert!(parse_kinds("такси, мото").unwrap().iter().filter(|&&x| x).count() == 4);
        assert!(parse_kinds("самолёты").is_err());
    }

    #[test]
    fn helpers() {
        assert_eq!(normalize("а 123 вс 77"), "A123BC77");
        assert_eq!(to_cyr("Y999XX99"), "У999ХХ99");
        assert!(region_ok("77") && region_ok("777") && !region_ok("00") && !region_ok("555"));
    }
}
