from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
mapping=ROOT/'experiments'/'verified_mapping.csv'
dfmap=pd.read_csv(mapping, dtype=str)
vals=[]
for _,r in dfmap.iterrows():
    fname=r['filename']
    pfile=ROOT/'thetis_output'/fname
    if not pfile.exists():
        vals.append({'filename':fname,'group':r['group'],'racket_max_velocity':None})
        continue
    df=pd.read_csv(pfile)
    # compute proxy same as script
    import numpy as np
    out=float('nan')
    found=None
    for key in ['racket','handright','wristright','righthand','hand_right']:
        xs=[c for c in df.columns if key in c.lower() and c.lower().endswith('_x')]
        if not xs: continue
        for xx in xs:
            prefix=xx.rsplit('_',1)[0]
            yy=prefix+'_Y'
            zz=prefix+'_Z'
            if yy in df.columns and zz in df.columns:
                found=(xx,yy,zz); break
        if found: break
    if found:
        pos=np.vstack([df[found[0]].values,df[found[1]].values,df[found[2]].values]).T.astype(float)
        vel=np.gradient(pos,axis=0)
        speed=np.linalg.norm(vel,axis=1)
        out=float(np.nanmax(speed))
    vals.append({'filename':fname,'group':r['group'],'racket_max_velocity':out})
res=pd.DataFrame(vals)
for g in ['expert','beginner']:
    sub=res[res['group']==g]
    a=sub['racket_max_velocity']
    cnt=a.notna().sum()
    mean=a.dropna().astype(float).mean() if cnt>0 else float('nan')
    mn=a.dropna().astype(float).min() if cnt>0 else float('nan')
    mx=a.dropna().astype(float).max() if cnt>0 else float('nan')
    print(f"Group={g}: count_non_na={cnt}, mean={mean:.4f}, min={mn:.4f}, max={mx:.4f}")
print('\nPer-file values:')
print(res.to_string(index=False))
