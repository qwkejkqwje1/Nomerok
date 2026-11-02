# Проверка на НАСТОЯЩИХ табличках (без вклейки): сцены val_h как есть.
# orig — без наклона, крупно (по нему берём «эталонный» текст, если прочитан уверенно);
# roll / side — та же сцена с наклоном, номер 70–200 точек, JPEG.
import json, os, random, numpy as np, cv2, sys
from synth_lib import *
rng=random.Random(23); np.random.seed(23)
Q=json.load(open('val_h/quads.json')); out='realh'; os.makedirs(out,exist_ok=True); labels={}
for fn in sorted(Q):
    if Q[fn]['cls']=='emptyPlate': continue
    img=cv2.imread('val_h/'+fn)
    if img is None: continue
    q0=order(Q[fn]['quad']); base=os.path.splitext(fn)[0]
    enc,q=camera(img,q0.copy(),rng,190,220); name=f'orig_{base}.jpg'; open(f'{out}/{name}','wb').write(enc.tobytes())
    labels[name]={'quad':q.tolist(),'mode':'orig','scene':base}
    for mode in ['roll','side']:
        im,q=geom(img,q0.copy(),mode,rng); enc,q=camera(im,q,rng)
        name=f'{mode}_{base}.jpg'; open(f'{out}/{name}','wb').write(enc.tobytes())
        labels[name]={'quad':q.tolist(),'mode':mode,'scene':base}
json.dump(labels,open(f'{out}/labels.json','w')); print(len(labels))
