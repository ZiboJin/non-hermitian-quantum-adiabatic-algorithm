"""Recompute all 11 theoretical threshold points in manuscript Figure 6(f)."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'recomputed/figure6-theory')
    args=parser.parse_args()
    from figure6_theory import compute_thresholds
    result=compute_thresholds()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output_dir/'thresholds_vs_size.csv',index=False)
