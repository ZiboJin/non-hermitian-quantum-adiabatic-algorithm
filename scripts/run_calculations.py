"""Run the manuscript settings. Full runs can be expensive; --dry-run prints commands."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def command(experiment,output,workers):
    common=['--gamma','10','--p','2','--q','4','--Omega','1']
    out=['--out-dir',str(output),'--max-workers',str(workers)]
    if experiment=='figure5-ck':
        name='run_mixed_scaling_extended.py'
        args=common+['--m-start','3','--m-fk-max','12','--m-nhhd-max','12','--hmis-same-as-nhhd',
                     '--fk-method','expm','--fk-step-factor','16','--fk-min-steps','200']+out
    elif experiment=='figure5-ck-like':
        name='run_ck_like_random_deletion_exp1.py'
        args=common+['--graph-pool',str(ROOT/'data/figure5/graph_pool.jsonl'),
                     '--m-min','3','--m-fk-max','12','--m-hd-max','12','--num-samples','200',
                     '--fk-method','expm','--fk-step-factor','8','--fk-min-steps','200']+out
    elif experiment=='figure6-grids':
        name='run_experiment_two_refined_cache.py'
        args=['--phase','all','--r-list','1','2','3','4','5','--center-list','E0','E1',
              '--grid-N-pair','81','--grid-N-pair-r5','241','--grid-N-local','161',
              '--local-contour-method','grid','--local-radius-factor','2','--pair-window-expand','2',
              '--mp-dps','70','--hp-n-iter','40','--hp-target-log10=-34',
              '--epsilon-vis','1e-32','--p','2','--q','4','--omega','1']+out
    elif experiment=='figure7':
        name='run_experiment_three.py'
        args=common+['--instance','paper_m2','--n-real','100','--n-noise','70',
                     '--steps-per-segment','8','--seed','1234','--seed-mode','paired_noise',
                     '--integrator','combined','--norm-method','power','--norm-iters','64',
                     '--noise-model','ginibre','--baseline-mode','midpoint','--fk-no-include-hx',
                     '--eps-list']+[f'1e{k}' for k in range(-15,0)]+out
    else:
        return [sys.executable,str(ROOT/'scripts/recompute_figure6_theory.py'),'--output-dir',str(output)]
    return [sys.executable,str(ROOT/'code'/name)]+args

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=['figure5-ck','figure5-ck-like','figure6-grids','figure6-theory','figure7'])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--max-workers',type=int,default=1,help='Increase only if sufficient RAM is available')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.max_workers<1:parser.error('--max-workers must be positive')
    output=(args.output_dir or ROOT/'recomputed'/args.experiment).resolve()
    if output==ROOT/'data' or ROOT/'data' in output.parents:parser.error('Choose an output outside archived data/')
    cmd=command(args.experiment,output,args.max_workers)
    print(subprocess.list2cmdline(cmd),flush=True)
    if not args.dry_run:
        env=os.environ.copy()
        env.setdefault('MPLBACKEND','Agg');env.setdefault('MPLCONFIGDIR',str(ROOT/'reproduced/.matplotlib'))
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env.setdefault(key,'1')
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
