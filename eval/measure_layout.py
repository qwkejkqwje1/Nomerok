"""Замер раскладки номера РФ на обучающих вырезках AUTO.RIA (ocrtest/img__*.png).
Отсюда числа в rectify.rs: высота цифр ≈ 0,6 высоты вырезки, поле над ними ≈ 0,4
высоты цифр, под ними ≈ 0,27 (+ поправка bias.py), до первой буквы ≈ 0,17 длины
«А123ВС», после последней ≈ 0,48."""
import cv2, glob, numpy as np, json
fs = sorted(glob.glob('ocrtest/img__*.png'))[::7][:400]
hd_h, top_h, bot_h, left_s, right_s = [], [], [], [], []
for f in fs:
    txt = json.load(open(f.replace('img__', 'ann__').replace('.png', '.json')))['description']
    if len(txt) not in (8, 9) or not txt[0].isalpha() or not txt[1:4].isdigit():
        continue
    g = cv2.imread(f, 0); g = cv2.resize(g, None, fx=3, fy=3); H, W = g.shape
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(H * 0.9) | 1, 10)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b)
    cs = sorted([s for s in st[1:] if 0.35 * H < s[3] < 0.85 * H and 0.1 * s[3] < s[2] < 0.9 * s[3]
                 and s[0] > 0 and s[0] + s[2] < W], key=lambda s: s[0])
    if len(cs) < 6:
        continue
    bots = np.array([s[1] + s[3] for s in cs]); mb = np.median(bots[:4])
    main = [s for s in cs if abs(s[1] + s[3] - mb) < 0.08 * H]
    if len(main) != 6:
        continue
    hd = sorted(s[3] for s in main)[4]
    S = main[-1][0] + main[-1][2] - main[0][0]
    hd_h.append(hd / H); top_h.append(min(s[1] for s in main) / H); bot_h.append(mb / H)
    left_s.append(main[0][0] / S); right_s.append((W - main[-1][0] - main[-1][2]) / S)
for name, a in [('высота цифр / высота', hd_h), ('верх цифр / высота', top_h), ('низ строки / высота', bot_h),
                ('слева / длина строки', left_s), ('справа / длина строки', right_s)]:
    print(f'{name:24} n={len(a)}  10/50/90%: {np.percentile(a, [10, 50, 90]).round(3)}')
