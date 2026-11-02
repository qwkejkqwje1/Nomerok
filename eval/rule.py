import json, sys
CY=dict(zip('АВЕКМНОРСТУХ','ABEKMHOPCTYX'))
def lat(s): return ''.join(CY.get(c,c) for c in s if c not in ' -')
L=json.load(open(sys.argv[2])); L={k:v for k,v in L.items() if v.get('gt_aspect',9)>=3.0}
recs=[json.loads(l) for l in open(sys.argv[1]+'/plates.jsonl')]
rows=[]
for r in recs:
    v=L.get(r['source'])
    if not v: continue
    import numpy as np
    c=np.array(v['quad']).mean(0); x1,y1,x2,y2=r['bbox']
    if not (x1<=c[0]<=x2 and y1<=c[1]<=y2): continue
    if r['other']:
        ot,oc=r['other'].rsplit(' ',1); ot=lat(ot); oc=float(oc[:-1])/100
        isfmt_o = ' ' in r['other'].rsplit(' ',1)[0]
    me=(r['plate_latin'],r['conf'],r['format']!='не РФ / не распознан формат')
    if not r['other']: rows.append((v['text'],me,None)); continue
    o=(ot,oc,isfmt_o)
    plain,rect=(o,me) if r['crop_kind']=='выпрямленная' else (me,o)
    rows.append((v['text'],plain,rect))
def run(rule):
    ok=0
    for t,p,rc in rows:
        pick=p if rc is None else (rc if rule(p,rc) else p)
        ok+=pick[0]==t
    return ok/len(rows)
print('n',len(rows))
print('plain', run(lambda p,r:False))
print('rect', run(lambda p,r:True))
print('A fmt+conf', run(lambda p,r: (r[2] and not p[2]) or (r[2]==p[2] and r[1]>p[1])))
for m in [0.05,0.1,0.2,0.3,0.5]:
    print('B margin',m, run(lambda p,r: (r[2] and not p[2]) or (r[2]==p[2] and r[1]>p[1]-m)))
reg=sum(1 for t,p,rc in rows if rc is not None and p[0]==t and ((rc if ((rc[2] and not p[2]) or (rc[2]==p[2] and rc[1]>p[1])) else p)[0])!=t)
fix=sum(1 for t,p,rc in rows if rc is not None and p[0]!=t and ((rc if ((rc[2] and not p[2]) or (rc[2]==p[2] and rc[1]>p[1])) else p)[0])==t)
print('стало хуже',reg,'стало лучше',fix)
