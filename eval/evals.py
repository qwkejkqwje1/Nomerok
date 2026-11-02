import json, sys, numpy as np
CY=dict(zip('АВЕКМНОРСТУХ','ABEKMHOPCTYX'))
def lat(s): return ''.join(CY.get(c,c) for c in s if c not in ' -')
d=sys.argv[1]; L=json.load(open(sys.argv[2]))
recs=[json.loads(l) for l in open(d+'/plates.jsonl')]
by={}
for r in recs: by.setdefault(r['source'],[]).append(r)
stats={}
cerr={}
L={k:v for k,v in L.items() if v["gt_aspect"]>=3.0}
for k,v in L.items():
    mode=v['mode']; st=stats.setdefault(mode,dict(n=0,det=0,found=0,plain=0,rect=0,auto=0,oracle=0)); st['n']+=1
    q=np.array(v['quad']); c=q.mean(0)
    rr=[r for r in by.get(k,[]) if r['bbox'][0]<=c[0]<=r['bbox'][2] and r['bbox'][1]<=c[1]<=r['bbox'][3]]
    if not rr: continue
    r=rr[0]; st['det']+=1
    t=v['text']
    other=lat(r['other'].rsplit(' ',1)[0]) if r['other'] else None
    if r['crop_kind']=='выпрямленная': rect,plain=r['plate_latin'],other
    else: plain,rect=r['plate_latin'],other
    if r['corners']:
        st['found']+=1
        H=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
        e=np.linalg.norm(np.array(r['corners'])-q,axis=1).mean()/H
        x1,y1,x2,y2=r['bbox']; bq=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        cerr.setdefault(mode,[]).append((e,np.linalg.norm(bq-q,axis=1).mean()/H))
    st['plain']+=plain==t; st['rect']+=(rect if rect else plain)==t; st['auto']+=r['plate_latin']==t
    st['oracle']+=(plain==t) or (rect==t)
tot={}
for mode,st in stats.items():
    for kk,vv in st.items(): tot[kk]=tot.get(kk,0)+vv
stats['ВСЕГО']=tot
print(f"{'mode':8} {'n':>4} {'det':>4} {'corn':>5} {'plain':>6} {'rect':>6} {'auto':>6} {'oracle':>6}")
for mode,st in stats.items():
    dd=max(st['det'],1)
    print(f"{mode:8} {st['n']:4} {st['det']:4} {st['found']/dd:5.0%} {st['plain']/dd:6.1%} {st['rect']/dd:6.1%} {st['auto']/dd:6.1%} {st['oracle']/dd:6.1%}")
for mode,a in cerr.items():
    a=np.array(a); print(mode,'corner err median rect %.3f bbox %.3f; rect<0.15: %.0f%%'%(np.median(a[:,0]),np.median(a[:,1]),100*(a[:,0]<0.15).mean()))
