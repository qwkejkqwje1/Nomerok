import json, sys, numpy as np, cv2
L=json.load(open(sys.argv[2])); recs=[json.loads(l) for l in open(sys.argv[1]+'/plates.jsonl')]
U=[]
for r in recs:
    if not r['corners'] or r['source'] not in L: continue
    q=np.float32(L[r['source']]['quad']); c=q.mean(0)
    x1,y1,x2,y2=r['bbox']
    if not (x1<=c[0]<=x2 and y1<=c[1]<=y2): continue
    H=cv2.getPerspectiveTransform(q,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    p=cv2.perspectiveTransform(np.float32(r['corners'])[None],H)[0]
    U.append([(p[0,0]+p[3,0])/2,(p[1,0]+p[2,0])/2,(p[0,1]+p[1,1])/2,(p[2,1]+p[3,1])/2, L[r['source']]['mode']])
for mode in ['plain','roll','side',None]:
    a=np.array([u[:4] for u in U if mode is None or u[4]==mode],float)
    print(mode, len(a), 'left %.3f right %.3f top %.3f bottom %.3f'%tuple(np.median(a,0)), ' IQR', np.round(np.percentile(a,75,0)-np.percentile(a,25,0),3))
