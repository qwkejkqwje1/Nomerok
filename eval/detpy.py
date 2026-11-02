import onnxruntime as ort, numpy as np, cv2
class Det:
    def __init__(s, path='/data/nomerok/nomerok/models/detector.onnx', threads=1):
        o=ort.SessionOptions(); o.intra_op_num_threads=threads
        s.s=ort.InferenceSession(path,o,providers=['CPUExecutionProvider']); s.inp=s.s.get_inputs()[0].name
    def __call__(s, bgr, conf=0.3, size=640):
        h,w=bgr.shape[:2]; r=min(size/h,size/w); nw,nh=round(w*r),round(h*r)
        dw,dh=(size-nw)/2,(size-nh)/2; l,t=round(dw-0.1),round(dh-0.1)
        im=np.full((size,size,3),114,np.uint8); im[t:t+nh,l:l+nw]=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA if r<1 else cv2.INTER_LINEAR)[...,::-1]
        x=(im.transpose(2,0,1)[None].astype(np.float32))/255
        out=s.s.run(None,{s.inp:x})[0].reshape(-1,7)
        res=[]
        for row in out:
            if row[6]<conf: continue
            res.append([float(np.clip((row[1]-dw)/r,0,w)),float(np.clip((row[2]-dh)/r,0,h)),float(np.clip((row[3]-dw)/r,0,w)),float(np.clip((row[4]-dh)/r,0,h)),float(row[6])])
        return res
