# Маленькая сеть углов номера: вход — серая область вокруг рамки детектора
# (рамка ±15 % по ширине, ±50 % по высоте), сжатая до 192×64; выход — 8 чисел
# (x, y четырёх углов: левый верх, правый верх, правый низ, левый низ) в долях области.
import json, glob, zlib, math, random, time, sys, numpy as np, cv2, torch, torch.nn as nn
W,H=192,64; PX,PY=0.15,0.50
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv)>2 else 2)
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 20
rows=[]
for f in sorted(glob.glob('cds/meta*.tsv')):
    for l in open(f):
        p=l.rstrip('\n').split('\t')
        if len(p)==3: rows.append((p[0].split('#')[0],p[1],json.loads(p[2])))
val=[r for r in rows if zlib.crc32(r[0].encode())%20==0]; tr=[r for r in rows if zlib.crc32(r[0].encode())%20!=0]
print('train',len(tr),'val',len(val),flush=True)
def load(rs): return [(cv2.imread('cds/'+n,0),np.array(j['q'],np.float32),np.array(j['b'],np.float32),j['mode']) for _,n,j in rs]
TR=load(tr); VA=load(val)
def region(b): w,h=b[2]-b[0],b[3]-b[1]; return b[0]-PX*w,b[1]-PY*h,(1+2*PX)*w,(1+2*PY)*h
def make(s,rng,aug):
    g,q,b,_=s; b=b.copy()
    if aug:
        w,h=b[2]-b[0],b[3]-b[1]; cx,cy=(b[0]+b[2])/2+rng.normal(0,0.03)*w,(b[1]+b[3])/2+rng.normal(0,0.06)*h
        w*=math.exp(rng.normal(0,0.04)); h*=math.exp(rng.normal(0,0.07)); b=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],np.float32)
    # центры пикселей как в rectify::warp: выход i+0.5 <-> вход rx+(i+0.5)*rw/W
    rx,ry,rw,rh=region(b); A=np.float32([[W/rw,0,(0.5-rx)*W/rw-0.5],[0,H/rh,(0.5-ry)*H/rh-0.5]])
    x=cv2.warpAffine(g,A,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    t=((q-[rx,ry])/[rw,rh]).astype(np.float32).reshape(-1)
    if aug:
        if rng.random()<0.25: x=cv2.GaussianBlur(x,(0,0),rng.uniform(0.3,1.0))
        x=x*rng.uniform(0.6,1.4)+rng.uniform(-40,40)
        if rng.random()<0.3: x=x+rng.normal(0,rng.uniform(2,10),x.shape)
        x=np.clip(x,0,255)
        if rng.random()<0.3: x=255*(x/255)**rng.uniform(0.6,1.6)
        if rng.random()<0.08: x=255-x
        if rng.random()<0.2:
            ex,ey=rng.integers(0,W-20),rng.integers(0,H-10); x[ey:ey+rng.integers(4,14),ex:ex+rng.integers(6,30)]=rng.uniform(0,255)
    return x/255.0,t
def bn(i,o,s): return nn.Sequential(nn.Conv2d(i,o,3,s,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(bn(1,16,2),bn(16,16,1),bn(16,32,2),bn(32,32,1),bn(32,64,2),bn(64,64,1),bn(64,96,2),bn(96,96,1),bn(96,128,2))
        s.h=nn.Sequential(nn.Flatten(),nn.Linear(128*2*6,256),nn.ReLU(inplace=True),nn.Dropout(0.1),nn.Linear(256,8))
    def forward(s,x): return s.h(s.f(x))
net=Net(); print('params',sum(p.numel() for p in net.parameters()),flush=True)
mean_t=np.mean([make(s,None,False)[1] for s in TR[:2000]],0)
with torch.no_grad(): net.h[-1].bias.copy_(torch.tensor(mean_t)); net.h[-1].weight.mul_(0.1)
SC=torch.tensor([W,H]*4,dtype=torch.float32)
def crit(p,t): return nn.functional.smooth_l1_loss(p*SC,t*SC,beta=1.0)
def errs(P,T):  # средняя ошибка угла в долях высоты таблички (в пикселях 192×64)
    P=P.reshape(-1,4,2)*np.array([W,H]); T=T.reshape(-1,4,2)*np.array([W,H])
    hp=(np.linalg.norm(T[:,3]-T[:,0],axis=1)+np.linalg.norm(T[:,2]-T[:,1],axis=1))/2
    return np.linalg.norm(P-T,axis=2).mean(1)/hp
VX=np.stack([make(s,None,False)[0] for s in VA])[:,None].astype(np.float32); VT=np.stack([make(s,None,False)[1] for s in VA])
BQ=[]
for s in VA:
    rx,ry,rw,rh=region(s[2]); b=s[2]; BQ.append(((np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])-[rx,ry])/[rw,rh]).reshape(-1))
print('val bbox-as-corners err median %.3f'%np.median(errs(np.array(BQ),VT)),flush=True)
opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4); BS=64
steps=EPOCHS*math.ceil(len(TR)/BS); sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=steps,pct_start=0.15)
rng=np.random.default_rng(0); best=9
for ep in range(EPOCHS):
    net.train(); idx=rng.permutation(len(TR)); t0=time.time(); tl=0
    for i in range(0,len(idx),BS):
        bt=[make(TR[j],rng,True) for j in idx[i:i+BS]]
        x=torch.tensor(np.stack([a for a,_ in bt])[:,None],dtype=torch.float32); t=torch.tensor(np.stack([b for _,b in bt]))
        loss=crit(net(x),t); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl+=loss.item()*len(bt)
    net.eval()
    with torch.no_grad(): P=np.concatenate([net(torch.tensor(VX[i:i+256])).numpy() for i in range(0,len(VX),256)])
    e=errs(P,VT); md=np.median(e)
    print(f'ep {ep} loss {tl/len(TR):.3f} val err median {md:.3f} mean {e.mean():.3f} <0.15 {100*(e<0.15).mean():.0f}% ({time.time()-t0:.0f}s)',flush=True)
    if md<best: best=md; torch.save(net.state_dict(),'corners.pt')
net.load_state_dict(torch.load('corners.pt')); net.eval()
modes=np.array([s[3] for s in VA])
with torch.no_grad(): P=net(torch.tensor(VX)).numpy()
e=errs(P,VT)
for m in ['plain','roll','side','both']: print(m,'median %.3f'%np.median(e[modes==m]),'n',(modes==m).sum())
torch.onnx.export(net,torch.zeros(1,1,H,W),'corners.onnx',input_names=['image'],output_names=['corners'],dynamic_axes={'image':{0:'n'},'corners':{0:'n'}},opset_version=17,dynamo=False)
print('exported',flush=True)
