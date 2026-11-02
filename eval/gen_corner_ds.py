# Обучающие примеры для сети углов: сцены train_sc + вырезки ocrtrain; рамку даёт настоящий детектор.
# Сохраняем «контекст» вокруг рамки (серый, ширина рамки -> 144 точки) + углы и рамку в его координатах.
import json, glob, os, random, sys, zlib, numpy as np, cv2
from synth_lib import *
from detpy import Det
shard,nsh,K=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
m=json.load(open('train_via_region_data.json')); m=m.get('_via_img_metadata',m)
Q={}
for v in m.values():
    rs=v['regions']
    if len(rs)==1:
        s=rs[0]['shape_attributes']; xs=s.get('all_points_x',[])
        if len(xs)==4:
            q=np.array(list(zip(xs,s['all_points_y'])),np.float32); c=q.mean(0); w=q[:,0].max()-q[:,0].min(); R=1.6*w
            Q[os.path.splitext(v['filename'])[0]+'.jpg']=q-[max(0,int(c[0]-R)),max(0,int(c[1]-R*0.75))]
crops=[f for f in sorted(glob.glob('ocrtrain/img__*.png'))]
asp={}
det=Det(); os.makedirs('cds',exist_ok=True)
done=set(l.split('\t')[0] for l in open(f'cds/meta{shard}.tsv')) if os.path.exists(f'cds/meta{shard}.tsv') else set()
out=open(f'cds/meta{shard}.tsv','a')
scenes=[f for f in sorted(os.listdir('train_sc')) if f.endswith('.jpg') and zlib.crc32(f.encode())%nsh==shard and f in Q]
n=0
for fn in scenes:
    if fn+'#0' in done: continue
    img=cv2.imread('train_sc/'+fn)
    if img is None: continue
    rng=random.Random(zlib.crc32(fn.encode())); q0=order(Q[fn])
    for k in range(K):
        key=f'{fn}#{k}'
        while True:
            cp=rng.choice(crops); pl=cv2.imread(cp)
            if pl is not None and pl.shape[1]/pl.shape[0]>=2.6: break
        mode=rng.choices(['plain','roll','side','both'],[0.25,0.3,0.3,0.15])[0]
        comp=paste(img,q0,pl); comp,q=geom(comp,q0.copy(),mode,rng,strong=True)
        enc,q=camera(comp,q,rng,50,230,strong=True); fr=cv2.imdecode(enc,cv2.IMREAD_COLOR)
        ds=det(fr); c=q.mean(0); qa=[q[:,0].min(),q[:,1].min(),q[:,0].max(),q[:,1].max()]
        def iou(b):
            ix=max(0,min(b[2],qa[2])-max(b[0],qa[0])); iy=max(0,min(b[3],qa[3])-max(b[1],qa[1])); I=ix*iy
            return I/((b[2]-b[0])*(b[3]-b[1])+(qa[2]-qa[0])*(qa[3]-qa[1])-I)
        ds=[b for b in ds if b[0]<=c[0]<=b[2] and b[1]<=c[1]<=b[3]]
        if not ds: out.write(f'{key}\tnodet\n'); continue
        b=max(ds,key=iou); x1,y1,x2,y2=b[:4]; w,h=x2-x1,y2-y1
        if w<20 or h<6: out.write(f'{key}\tsmall\n'); continue
        cx1=int(np.floor(x1-0.35*w)); cy1=int(np.floor(y1-1.1*h)); cx2=int(np.ceil(x2+0.35*w)); cy2=int(np.ceil(y2+1.1*h))
        P=max(0,-cx1,-cy1,cx2-fr.shape[1],cy2-fr.shape[0])
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        if P: g=cv2.copyMakeBorder(g,P,P,P,P,cv2.BORDER_REPLICATE)
        part=g[cy1+P:cy2+P,cx1+P:cx2+P]; s=144.0/w
        tw,th=max(8,round(part.shape[1]*s)),max(8,round(part.shape[0]*s))
        ctx=cv2.resize(part,(tw,th),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
        sx,sy=tw/part.shape[1],th/part.shape[0]
        qc=(q-[cx1,cy1])*[sx,sy]; bc=[(x1-cx1)*sx,(y1-cy1)*sy,(x2-cx1)*sx,(y2-cy1)*sy]
        name=f"{os.path.splitext(fn)[0]}_{k}.png"; cv2.imwrite('cds/'+name,ctx)
        out.write(f"{key}\t{name}\t{json.dumps({'q':np.round(qc,2).tolist(),'b':np.round(bc,2).tolist(),'mode':mode,'s':round(b[4],3),'pw':round(float(np.linalg.norm(q[1]-q[0])),1)})}\n")
    out.flush(); n+=1
    if n%100==0: print(shard,n,flush=True)
print('DONE',shard,n,flush=True)
