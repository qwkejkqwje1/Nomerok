# evals2.py OUT labels.json — точность по режимам: обычная / +яркость (0.2) / +сеть / все три (auto) / оракул,
# сколько стало хуже и лучше относительно обычной вырезки, ошибка углов (в долях высоты таблички).
import json, sys, numpy as np
d=sys.argv[1]; L=json.load(open(sys.argv[2])); L={k:v for k,v in L.items() if v.get('gt_aspect',9)>=3.0}
recs=[json.loads(l) for l in open(d+'/plates.jsonl')]
by={}
for r in recs: by.setdefault(r['source'],[]).append(r)
def pick(vs):  # как better() в main.rs
    best=vs[0]
    for v in vs[1:]:
        if (v['format_ok'] and not best['format_ok']) or (v['format_ok']==best['format_ok'] and v['conf']>best['conf']): best=v
    return best
K={'обычная':'P','выпрямленная':'C','выпрямленная сетью':'N'}
modes=['plain','roll','side']; S={m:dict(n=0,det=0,P=0,PC=0,PN=0,PCN=0,orc=0,fC=0,fN=0,worseC=0,betterC=0,worseA=0,betterA=0,worseN=0,betterN=0) for m in modes}
E={m:{'C':[],'N':[],'B':[]} for m in modes}
for k,v in L.items():
    m=v['mode']; st=S[m]; st['n']+=1; q=np.array(v['quad']); c=q.mean(0)
    rr=[r for r in by.get(k,[]) if r['bbox'][0]<=c[0]<=r['bbox'][2] and r['bbox'][1]<=c[1]<=r['bbox'][3]]
    if not rr: continue
    r=rr[0]; st['det']+=1; t=v['text']
    vs={K[x['kind']]:x for x in r['variants']}
    P=vs['P']; ok=lambda x: x['plate_latin']==t
    res={'P':ok(P),'PC':ok(pick([P]+[vs[x] for x in 'C' if x in vs])),'PN':ok(pick([P]+[vs[x] for x in 'N' if x in vs])),'PCN':ok(pick([P]+[vs[x] for x in 'CN' if x in vs]))}
    for a,b in res.items(): st[a]+=b
    st['orc']+=any(ok(x) for x in vs.values())
    for tag,key in [('C','PC'),('N','PN'),('A','PCN')]:
        st['worse'+tag]+=res['P'] and not res[key]; st['better'+tag]+=(not res['P']) and res[key]
    H=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    x1,y1,x2,y2=r['bbox']; E[m]['B'].append(np.linalg.norm(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])-q,axis=1).mean()/H)
    for x in 'CN':
        if x in vs: st['f'+x]+=1; E[m][x].append(np.linalg.norm(np.array(vs[x]['corners'])-q,axis=1).mean()/H)
tot={kk:sum(S[m][kk] for m in modes) for kk in S['plain']}; S['ВСЕГО']=tot
for x in 'CNB': E['ВСЕГО']={x:sum((E[m][x] for m in modes),[]) for x in 'CNB'}
print(f"{'снимки':8} {'n':>4} {'дет':>4} {'углы1':>6} {'углы2':>6} | {'обычн':>6} {'+ярк':>6} {'+сеть':>6} {'все':>6} {'оракул':>6} | хуже/лучше: ярк сеть все")
for m,st in S.items():
    dd=max(st['det'],1); f=lambda a: f"{100*st[a]/dd:6.1f}"
    print(f"{m:8} {st['n']:4} {st['det']:4} {100*st['fC']/dd:5.0f}% {100*st['fN']/dd:5.0f}% | {f('P')} {f('PC')} {f('PN')} {f('PCN')} {f('orc')} |  {st['worseC']}/{st['betterC']}  {st['worseN']}/{st['betterN']}  {st['worseA']}/{st['betterA']}")
print('ошибка углов, медиана (доля высоты таблички): яркость / сеть / рамка детектора;  доля <0.15')
for m,e in E.items():
    md=lambda a: np.median(a) if a else float('nan'); lt=lambda a: 100*np.mean(np.array(a)<0.15) if a else float('nan')
    print(f"  {m:8} {md(e['C']):.3f} / {md(e['N']):.3f} / {md(e['B']):.3f}    <0.15: {lt(e['C']):.0f}% / {lt(e['N']):.0f}%")
