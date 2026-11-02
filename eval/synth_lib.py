# Общая «фабрика» снимков: в настоящую сцену (4 размеченных угла) вклеиваем вырезку номера,
# наклоняем кадр (roll / side / both), уменьшаем, кладём в кадр «как с камеры», жмём JPEG.
import numpy as np, cv2, random
def order(q):
    q=np.asarray(q,np.float32); c=q.mean(0); ang=np.arctan2(q[:,1]-c[1],q[:,0]-c[0]); q=q[np.argsort(ang)]
    k=np.argmin(q.sum(1)); return np.roll(q,-k,0)
def paste(img,q,pl):
    ph,pw=pl.shape[:2]; src=np.float32([[0,0],[pw,0],[pw,ph],[0,ph]])
    Hm=cv2.getPerspectiveTransform(src,q); W,H=img.shape[1],img.shape[0]
    warped=cv2.warpPerspective(pl,Hm,(W,H),flags=cv2.INTER_AREA if pw>(q[1,0]-q[0,0]) else cv2.INTER_LINEAR)
    mask=cv2.warpPerspective(np.full((ph,pw),255,np.uint8),Hm,(W,H))
    mask=cv2.GaussianBlur(mask,(3,3),0).astype(np.float32)[...,None]/255
    return (img*(1-mask)+warped*mask).astype(np.uint8)
def geom(img,q,mode,rng,strong=False):
    c=q.mean(0); H,W=img.shape[:2]
    roll=15 if not strong else 20; sk=(0.15,0.45) if not strong else (0.1,0.55)
    M=np.eye(3)
    if mode in('roll','both'):
        a=rng.choice([-1,1])*rng.uniform(5 if mode=='roll' else 2,roll)
        R=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),a,1.0); M=np.vstack([R,[0,0,1]])@M
    if mode in('side','both'):
        k=rng.choice([-1,1])*rng.uniform(*sk); sx=rng.uniform(0.55 if strong else 0.6,0.95 if strong else 0.9)
        T=np.array([[1,0,-c[0]],[0,1,-c[1]],[0,0,1]]); S=np.array([[sx,0,0],[k*sx,1,0],[0,0,1]])
        P=np.eye(3); P[2,0]=rng.uniform(-0.3,0.3)/max(H,W)
        M=np.linalg.inv(T)@P@S@T@M
    if strong and mode=='plain':
        a=rng.uniform(-3,3); R=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),a,1.0); M=np.vstack([R,[0,0,1]])
    if not np.allclose(M,np.eye(3)):
        img=cv2.warpPerspective(img,M,(W,H),borderMode=cv2.BORDER_REFLECT); q=cv2.perspectiveTransform(q[None],M)[0]
    return img,q
def camera(img,q,rng,wmin=70,wmax=200,strong=False):
    w=np.linalg.norm(q[1]-q[0]); tw=rng.uniform(wmin,wmax); s=min(1.0,tw/w)
    if s<1: img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA); q=q*s
    pw=np.linalg.norm(q[1]-q[0])
    # кадр камеры: ширина = номер × 6..14, 16:9; фон — сильно размытая сцена
    FW=int(max(img.shape[1]+8,pw*rng.uniform(6,14))); FH=int(max(img.shape[0]+8,FW*9/16))
    bg=cv2.GaussianBlur(cv2.resize(img,(FW//8+1,FH//8+1),interpolation=cv2.INTER_AREA),(0,0),3)
    bg=cv2.resize(bg,(FW,FH),interpolation=cv2.INTER_LINEAR)
    ox=rng.randint(0,FW-img.shape[1]); oy=rng.randint(0,FH-img.shape[0])
    bg[oy:oy+img.shape[0],ox:ox+img.shape[1]]=img; q=q+[ox,oy]
    if strong:
        f=rng.uniform(0.6,1.3); b=rng.uniform(-40,40); bg=np.clip(bg.astype(np.float32)*f+b,0,255).astype(np.uint8)
        if rng.random()<0.3: bg=cv2.GaussianBlur(bg,(0,0),rng.uniform(0.4,1.2))
        if rng.random()<0.3: bg=np.clip(bg+np.random.normal(0,rng.uniform(2,9),bg.shape),0,255).astype(np.uint8)
    qj=rng.randint(35 if strong else 45,90 if strong else 85)
    ok,enc=cv2.imencode('.jpg',bg,[cv2.IMWRITE_JPEG_QUALITY,qj]); return enc,q
