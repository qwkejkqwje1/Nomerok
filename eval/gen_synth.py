import json, glob, os, random, numpy as np, cv2
random.seed(7); np.random.seed(7)
m=json.load(open('via_val.json'))['_via_img_metadata']
gt={}
for v in m.values():
    rs=v['regions']
    if len(rs)==1:
        s=rs[0]['shape_attributes']
        if len(s.get('all_points_x',[]))==4: gt[v['filename']]=np.array(list(zip(s['all_points_x'],s['all_points_y'])),np.float32)
def order(q):
    c=q.mean(0); ang=np.arctan2(q[:,1]-c[1],q[:,0]-c[0]); q=q[np.argsort(ang)]
    k=np.argmin(q.sum(1)); return np.roll(q,-k,0)
scenes=[f for f in sorted(os.listdir('val')) if f in gt]
crops=sorted(glob.glob('ocrtest/img__*.png')); random.shuffle(crops)
out='synth'; os.makedirs(out,exist_ok=True)
labels={}
ci=0
for rep in range(1):
  for fn in scenes:
    img=cv2.imread('val/'+fn)
    if img is None: continue
    q=order(gt[fn])
    for mode in ['plain','roll','side']:
        cp=crops[ci%len(crops)]; ci+=1
        text=json.load(open(cp.replace('img__','ann__').replace('.png','.json')))['description']
        pl=cv2.imread(cp); ph,pw=pl.shape[:2]
        src=np.float32([[0,0],[pw,0],[pw,ph],[0,ph]])
        Hm=cv2.getPerspectiveTransform(src,q)
        warped=cv2.warpPerspective(pl,Hm,(img.shape[1],img.shape[0]),flags=cv2.INTER_AREA if pw>(q[1,0]-q[0,0]) else cv2.INTER_LINEAR)
        mask=cv2.warpPerspective(np.full((ph,pw),255,np.uint8),Hm,(img.shape[1],img.shape[0]))
        mask=cv2.GaussianBlur(mask,(3,3),0).astype(np.float32)[...,None]/255
        comp=(img*(1-mask)+warped*mask).astype(np.uint8)
        c=q.mean(0); qq=q.copy()
        if mode=='roll':
            a=random.choice([-1,1])*random.uniform(5,15)
            M=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),a,1.0); M=np.vstack([M,[0,0,1]])
        elif mode=='side':
            k=random.choice([-1,1])*random.uniform(0.15,0.45); sx=random.uniform(0.6,0.9)
            T=np.array([[1,0,-c[0]],[0,1,-c[1]],[0,0,1]]); S=np.array([[sx,0,0],[k*sx,1,0],[0,0,1]])
            # small extra perspective
            P=np.eye(3); P[2,0]=random.uniform(-0.3,0.3)/max(img.shape)
            M=np.linalg.inv(T)@P@S@T
        else: M=np.eye(3)
        if mode!='plain':
            comp=cv2.warpPerspective(comp,M,(img.shape[1],img.shape[0]),borderMode=cv2.BORDER_REFLECT)
            qq=cv2.perspectiveTransform(q[None],M)[0]
        # target plate width 70..200 px
        w=np.linalg.norm(qq[1]-qq[0]); tw=random.uniform(70,200); s=min(1.0,tw/w)
        if s<1: comp=cv2.resize(comp,None,fx=s,fy=s,interpolation=cv2.INTER_AREA); qq=qq*s
        name=f'{mode}_{len(labels):04d}.jpg'
        cv2.imwrite(f'{out}/{name}',comp,[cv2.IMWRITE_JPEG_QUALITY,random.randint(45,85)])
        labels[name]={'text':text,'quad':qq.tolist(),'mode':mode}
json.dump(labels,open(f'{out}/labels.json','w'))
print(len(labels))
