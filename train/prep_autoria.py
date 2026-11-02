import zipfile, json, os, io, csv, sys
from PIL import Image
z=zipfile.ZipFile('ru_full.zip')
P='autoriaNumberplateOcrRu-2021-09-01/'
names=set(z.namelist())
ALLOWED=set('0123456789ABEKMHOPCTYX')
for split in ['train','val','test']:
    os.makedirs(f'ru/{split}',exist_ok=True)
    rows=[]; skipped=0
    anns=[n for n in names if n.startswith(P+split+'/ann/') and n.endswith('.json')]
    imgs={os.path.splitext(os.path.basename(n))[0]:n for n in names if n.startswith(P+split+'/img/') and not n.endswith('/')}
    for i,a in enumerate(sorted(anns)):
        try: j=json.loads(z.read(a))
        except Exception: skipped+=1; continue
        t=j.get('description','').upper()
        nm=j.get('name') or os.path.splitext(os.path.basename(a))[0]
        if not t or not set(t)<=ALLOWED or len(t)>10 or nm not in imgs: skipped+=1; continue
        out=f'ru/{split}/{i:06d}.png'
        try:
            im=Image.open(io.BytesIO(z.read(imgs[nm]))).convert('RGB')
        except Exception: skipped+=1; continue
        im.resize((128,64),Image.BILINEAR).save(out)
        rows.append((os.path.abspath(out),t,j.get('count_lines','0')))
        if i%5000==0: print(split,i,flush=True)
    with open(f'ru_{split}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['image_path','plate_text']); [w.writerow(r[:2]) for r in rows]
    with open(f'ru_{split}_lines.csv','w',newline='') as f:
        w=csv.writer(f); [w.writerow(r) for r in rows]
    print(split,len(rows),'skipped',skipped,flush=True)
