import json, sys, os, random, threading
from rzip import open_zip
from concurrent.futures import ThreadPoolExecutor
U1='https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2023-03-06.zip'
U2='https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip'
which=sys.argv[1]
if which=='det':
    z=open_zip(U1); names=z.namelist()
    val=[n for n in names if '/val/' in n and not n.endswith('/')]
    print(len(names), len(val), val[:3], flush=True)
    m=json.load(open('via_val.json'))['_via_img_metadata']
    sel=[]
    for v in m.values():
        rs=v['regions']
        if len(rs)==1:
            s=rs[0]['shape_attributes']; xs=s.get('all_points_x',[])
            if len(xs)==4 and max(xs)-min(xs)>=60: sel.append(v['filename'])
    random.seed(1); random.shuffle(sel); sel=sel[:700]
    byname={os.path.basename(n):n for n in val}
    sel=[s for s in sel if s in byname]
    print('selected',len(sel),flush=True)
    os.makedirs('val',exist_ok=True)
    loc=threading.local()
    def get(fn):
        if not hasattr(loc,'z'): loc.z=open_zip(U1)
        p='val/'+fn
        if not os.path.exists(p):
            d=loc.z.read(byname[fn]); open(p,'wb').write(d)
    with ThreadPoolExecutor(6) as ex:
        for i,_ in enumerate(ex.map(get,sel)):
            if i%50==0: print('det',i,flush=True)
else:
    z=open_zip(U2); names=z.namelist()
    test=[n for n in names if '/test/' in n and not n.endswith('/')]
    print(len(names), len(test), test[:4], flush=True)
    os.makedirs('ocrtest',exist_ok=True)
    loc=threading.local()
    def get(n):
        if not hasattr(loc,'z'): loc.z=open_zip(U2)
        p='ocrtest/'+n.split('/test/')[1].replace('/','__')
        if not os.path.exists(p): open(p,'wb').write(loc.z.read(n))
    with ThreadPoolExecutor(6) as ex:
        for i,_ in enumerate(ex.map(get,test)):
            if i%500==0: print('ocr',i,flush=True)
print('DONE',flush=True)
