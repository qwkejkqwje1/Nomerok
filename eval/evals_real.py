# evals_real.py OUT realh/labels.json — настоящие таблички: эталон = уверенное (>=0.9, формат РФ) прочтение
# обычной вырезки на крупном снимке без наклона; сравниваем прочтения той же сцены с наклоном.
import json, sys, numpy as np
def pick(vs):  # как better() в main.rs
    best=vs[0]
    for v in vs[1:]:
        if (v['format_ok'] and not best['format_ok']) or (v['format_ok']==best['format_ok'] and v['conf']>best['conf']): best=v
    return best
K={'обычная':'P','выпрямленная':'C','выпрямленная сетью':'N'}
d=sys.argv[1]; L=json.load(open(sys.argv[2]))
recs=[json.loads(l) for l in open(d+'/plates.jsonl')]; by={}
for r in recs: by.setdefault(r['source'],[]).append(r)
def rec(k):
    v=L[k]; q=np.array(v['quad']); c=q.mean(0)
    rr=[r for r in by.get(k,[]) if r['bbox'][0]<=c[0]<=r['bbox'][2] and r['bbox'][1]<=c[1]<=r['bbox'][3]]
    return (rr[0],q) if rr else (None,q)
gt={}
for k,v in L.items():
    if v['mode']!='orig': continue
    r,_=rec(k)
    if r is None: continue
    P=[x for x in r['variants'] if x['kind']=='обычная'][0]
    if P['conf']>=0.9 and P['format_ok']: gt[v['scene']]=P['plate_latin']
print('сцен с эталоном',len(gt))
S={}; E={}
for k,v in L.items():
    m=v['mode']
    if m=='orig' or v['scene'] not in gt: continue
    st=S.setdefault(m,dict(n=0,det=0,P=0,PC=0,PN=0,PCN=0,wC=0,bC=0,wN=0,bN=0,wA=0,bA=0)); st['n']+=1
    r,q=rec(k)
    if r is None: continue
    st['det']+=1; t=gt[v['scene']]; vs={K[x['kind']]:x for x in r['variants']}; ok=lambda x: x['plate_latin']==t
    res={'P':ok(vs['P']),'PC':ok(pick([vs[x] for x in 'PC' if x in vs])),'PN':ok(pick([vs[x] for x in 'PN' if x in vs])),'PCN':ok(pick([vs[x] for x in 'PCN' if x in vs]))}
    for a,b in res.items(): st[a]+=b
    for tag,key in [('C','PC'),('N','PN'),('A','PCN')]: st['w'+tag]+=res['P'] and not res[key]; st['b'+tag]+=(not res['P']) and res[key]
    H=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
    for x in 'CN':
        if x in vs: E.setdefault(m,{}).setdefault(x,[]).append(np.linalg.norm(np.array(vs[x]['corners'])-q,axis=1).mean()/H)
for m,st in S.items():
    dd=max(st['det'],1); f=lambda a: f"{100*st[a]/dd:5.1f}"
    print(f"{m:5} n {st['n']} дет {st['det']} | обычн {f('P')}  +ярк {f('PC')}  +сеть {f('PN')}  все {f('PCN')} | хуже/лучше ярк {st['wC']}/{st['bC']} сеть {st['wN']}/{st['bN']} все {st['wA']}/{st['bA']}")
    e=E.get(m,{}); print('      ошибка углов до края таблички (медиана): яркость %.3f (найдено %d), сеть %.3f (найдено %d)'%(np.median(e.get('C',[np.nan])),len(e.get('C',[])),np.median(e.get('N',[np.nan])),len(e.get('N',[]))))

# ── Часть 2: все настоящие таблички (в основном украинские, формат РФ не подходит),
# сравниваем «сырое» прочтение (raw, без правил формата). Эталон — снимок без наклона,
# на котором все найденные вырезки (не меньше двух) прочитались одинаково.
print('\nсырое прочтение (raw), все таблички:')
gt={}
for k,v in L.items():
    if v['mode']!='orig': continue
    r,_=rec(k)
    if r is None: continue
    raws={x['raw'] for x in r['variants']}
    if len(r['variants'])>=2 and len(raws)==1 and len(next(iter(raws)))>=6: gt[v['scene']]=next(iter(raws))
print('сцен с эталоном',len(gt))
S={}
for k,v in L.items():
    m=v['mode']
    if m=='orig' or v['scene'] not in gt: continue
    st=S.setdefault(m,dict(n=0,det=0,P=0,C=0,N=0,PC=0,PN=0,PCN=0,nC=0,nN=0)); st['n']+=1
    r,q=rec(k)
    if r is None: continue
    st['det']+=1; t=gt[v['scene']]; vs={K[x['kind']]:x for x in r['variants']}; ok=lambda x: x['raw']==t
    st['P']+=ok(vs['P'])
    for x in 'CN': st[x]+=ok(vs[x]) if x in vs else ok(vs['P']); st['n'+x]+=x in vs
    for key in ['PC','PN','PCN']: st[key]+=ok(pick([vs[x] for x in key if x in vs]))
for m,st in S.items():
    dd=max(st['det'],1); f=lambda a: f"{100*st[a]/dd:5.1f}"
    print(f"{m:5} n {st['n']} дет {st['det']} | по отдельности: обычная {f('P')}  по яркости {f('C')}  сетью {f('N')} | с выбором: +ярк {f('PC')} +сеть {f('PN')} все {f('PCN')}")
