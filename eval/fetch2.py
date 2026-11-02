# fetch2.py scenes train|val START END OUTDIR  — фото с 1 номером (4 угла), вырезаем окно вокруг номера
# fetch2.py ocr train|test N OUTDIR            — вырезки номеров из OCR-датасета
import json, sys, os, random, threading, io
import numpy as np, cv2
from rzip import open_zip
from concurrent.futures import ThreadPoolExecutor
U1='https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2023-03-06.zip'
U2='https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip'
kind,split=sys.argv[1],sys.argv[2]
loc=threading.local()
if kind=='scenes':
    a,b,out=int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]; os.makedirs(out,exist_ok=True)
    m=json.load(open(f'{split}_via_region_data.json')); m=m.get('_via_img_metadata',m)
    sel=[]
    for v in m.values():
        rs=v['regions']
        if len(rs)==1:
            s=rs[0]['shape_attributes']; xs=s.get('all_points_x',[])
            if len(xs)==4 and max(xs)-min(xs)>=60: sel.append((v['filename'],xs,s['all_points_y'],rs[0]['region_attributes'].get('class','')))
    random.seed(1); random.shuffle(sel); sel=sel[a:b]
    print('selected',len(sel),flush=True)
    prefix=f'autoriaNumberplateDataset-2023-03-06/{split}/'
    meta={}
    def get(t):
        fn,xs,ys,cl=t
        if not hasattr(loc,'z'): loc.z=open_zip(U1)
        try: d=loc.z.read(prefix+fn)
        except KeyError: return None
        img=cv2.imdecode(np.frombuffer(d,np.uint8),cv2.IMREAD_COLOR)
        if img is None: return None
        q=np.array(list(zip(xs,ys)),np.float32); c=q.mean(0); w=q[:,0].max()-q[:,0].min()
        R=1.6*w; x0=int(max(0,c[0]-R)); y0=int(max(0,c[1]-R*0.75)); x1=int(min(img.shape[1],c[0]+R)); y1=int(min(img.shape[0],c[1]+R*0.75))
        crop=img[y0:y1,x0:x1]; name=os.path.splitext(fn)[0]+'.jpg'
        cv2.imwrite(f'{out}/{name}',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
        return name,(q-[x0,y0]).tolist(),cl
    with ThreadPoolExecutor(8) as ex:
        for i,r in enumerate(ex.map(get,sel)):
            if r: meta[r[0]]={'quad':r[1],'cls':r[2]}
            if i%250==0: print(split,i,flush=True)
    json.dump(meta,open(f'{out}/quads.json','w'))
else:
    n,out=int(sys.argv[3]),sys.argv[4]; os.makedirs(out,exist_ok=True)
    z=open_zip(U2); names=[x for x in z.namelist() if f'/{split}/' in x and not x.endswith('/')]
    imgs=[x for x in names if '/img/' in x]; random.seed(2); random.shuffle(imgs); imgs=imgs[:n]
    print(len(names),len(imgs),imgs[:2],flush=True)
    def get(x):
        if not hasattr(loc,'z'): loc.z=open_zip(U2)
        p=out+'/'+x.split(f'/{split}/')[1].replace('/','__')
        if not os.path.exists(p): open(p,'wb').write(loc.z.read(x))
        if split=='test':
            a=x.replace('/img/','/ann/').rsplit('.',1)[0]+'.json'; pa=out+'/'+a.split(f'/{split}/')[1].replace('/','__')
            if not os.path.exists(pa): open(pa,'wb').write(loc.z.read(a))
    with ThreadPoolExecutor(8) as ex:
        for i,_ in enumerate(ex.map(get,imgs)):
            if i%1000==0: print('ocr',i,flush=True)
print('DONE',flush=True)
