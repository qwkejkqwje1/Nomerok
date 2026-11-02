import sys, csv, numpy as np, onnxruntime as ort
from PIL import Image
m = sys.argv[1]
s = ort.InferenceSession(m, providers=['CPUExecutionProvider'])
A = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
rows = list(csv.DictReader(open('ds/ru_test.csv')))
ok = 0; n = 0; bad = []
B = 64
for i in range(0, len(rows), B):
    ch = rows[i:i+B]
    x = np.stack([np.array(Image.open('ds/'+r['image_path'] if not r['image_path'].startswith('ds/') else r['image_path']).convert('RGB').resize((128,64))) for r in ch]).astype(np.uint8)
    out = s.run(None, {s.get_inputs()[0].name: x})[0]
    for r, o in zip(ch, out):
        t = ''.join(A[k] for k in o.argmax(-1)).rstrip('_')
        g = r['plate_text'].rstrip('_')
        n += 1; ok += t == g
        if t != g: bad.append((g, t))
print(m, 'exact', ok, '/', n, '=%.4f' % (ok/n))
print(bad[:25])
