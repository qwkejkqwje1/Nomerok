# Проверочный набор (held-out): сцены val_h (не участвовали ни в настройке, ни в обучении) + вырезки ocrtest
import json, glob, os, random, numpy as np, cv2, sys
from synth_lib import *
rng=random.Random(11); np.random.seed(11)
Q=json.load(open('val_h/quads.json'))
crops=sorted(glob.glob('ocrtest/img__*.png')); rng.shuffle(crops)
out=sys.argv[1] if len(sys.argv)>1 else 'evalh'; os.makedirs(out,exist_ok=True)
labels={}; ci=0
for fn in sorted(Q):
    img=cv2.imread('val_h/'+fn)
    if img is None: continue
    q0=order(Q[fn]['quad'])
    for mode in ['plain','roll','side']:
        cp=crops[ci%len(crops)]; ci+=1
        text=json.load(open(cp.replace('img__','ann__').replace('.png','.json')))['description']
        pl=cv2.imread(cp)
        comp=paste(img,q0,pl); comp,q=geom(comp,q0.copy(),mode,rng); enc,q=camera(comp,q,rng)
        name=f'{mode}_{len(labels):04d}.jpg'; open(f'{out}/{name}','wb').write(enc.tobytes())
        labels[name]={'text':text,'quad':q.tolist(),'mode':mode,'gt_aspect':pl.shape[1]/pl.shape[0]}
json.dump(labels,open(f'{out}/labels.json','w')); print(len(labels))
