"""Synthetic V3 demonstrations; only generates NEW files, never V1/V2 data."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_quality_flip_demo():
    rng=np.random.default_rng(20260929)
    n=180
    x=rng.uniform(-1.8,1.8,n)
    noise=rng.normal(size=n)
    # Independent background, with no sample linear component along centered X.
    # This construction is explicit synthetic design, not manipulation of user data.
    centered=x-x.mean()
    y=noise-noise.mean()-centered*(centered@noise)/(centered@centered)
    x=np.r_[x, [9,10,11,12,13,14]]
    y=np.r_[y, [8,10,10,13,12,15]]
    return pd.DataFrame({'sample_id':np.arange(1,len(x)+1), 'exposure':x, 'outcome':y})


def generate_method_audit_demo():
    rng=np.random.default_rng(32026)
    n=90
    return pd.DataFrame({'sample_id':np.arange(1,2*n+1),
                         'teaching_mode':['A互动教学']*n+['B常规教学']*n,
                         'exam_score':np.r_[rng.normal(80,4,n),rng.normal(72,12,n)]})


if __name__=='__main__':
    root=Path(__file__).resolve().parent/'data'
    root.mkdir(exist_ok=True)
    for filename,generate in [('quality_flip_demo.csv',generate_quality_flip_demo),('method_audit_demo.csv',generate_method_audit_demo)]:
        generate().to_csv(root/filename,index=False,encoding='utf-8-sig')
        print(root/filename)
