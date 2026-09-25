"""Check file integrity, sample statistics, seeds, and Figure 6 grid results."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'reproduced/.matplotlib'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd

def read(path):
    return pd.read_csv(ROOT/path,float_precision='round_trip')

def check(condition,message):
    if not condition:raise ValueError(message)

def close(actual,expected,description,rtol=5e-12,atol=1e-15):
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,err_msg=description)

def verify(smoke=False):
    report={}
    entries=(ROOT/'checksums.sha256').read_text(encoding='utf-8').splitlines()
    for entry in entries:
        digest,relative_path=entry.split('  ',1)
        path=ROOT/relative_path
        check(path.is_file(),'Missing file: '+relative_path)
        check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,
              'Modified file: '+relative_path)
    report['files_checked']=len(entries)
    ck=read('data/figure5/ck_results.csv')
    check(ck.n.tolist()==list(range(9,46,4)),'Figure 5 CK sizes')
    with (ROOT/'data/figure5/graph_pool.jsonl').open(encoding='utf-8') as f:
        records=[json.loads(line) for line in f if line.strip()]
    pool={r['graph_id']:r for r in records}
    check(len(pool)==len(records)==2000,'Graph count / duplicate graph IDs')
    check(collections.Counter(r['m'] for r in records)=={m:200 for m in range(3,13)},'Graph sizes')
    from generate_ck_like_random_deletion_graphs import graph_hash
    for record in records:
        check(record['graph_hash']==graph_hash(record['m'],record['deleted_cross_edges']),'Graph hash')
    raw=read('data/figure5/ck_like_results.csv')
    summary=read('data/figure5/ck_like_summary.csv')
    check(len(raw)==6000 and len(summary)==30,'Figure 5 record count')
    check(raw.status.eq('success').all(),'Figure 5 failed realization')
    check(not raw.duplicated(['algorithm','graph_id']).any(),'Figure 5 duplicate realization')
    for row in raw.itertuples():
        check(row.graph_id in pool and row.graph_hash==pool[row.graph_id]['graph_hash'],'Result/graph mismatch')
    for row in summary.itertuples():
        values=raw[(raw.algorithm==row.algorithm)&(raw.m==row.m)].success_probability.to_numpy()
        check(len(values)==row.n_samples==200,'Figure 5 sample count')
        close([np.min(values),np.median(values),np.max(values)],
              [row.min_success,row.median_success,row.max_success],'Figure 5 aggregation')
    report['figure5']={'ck_rows':10,'graph_instances':2000,'realizations':6000,'summary_groups':30}
    print('PASS Figure 5: graphs, hashes and min/median/max statistics',flush=True)

    from experiment_two_resolvent_pseudospectrum import (
        clock_eigenvalues, component_contours_from_log_grid,
        single_component_contours_from_log_grid,pseudospectral_closing_threshold_grid_log,
    )
    meta=read('data/figure6/contours_metadata.csv')
    pair=read('data/figure6/pair_grid_manifest.csv')
    thresholds=read('data/figure6/thresholds_numerical.csv')
    grid_paths=set()
    for name in ['local_grid_manifest.csv','pair_grid_manifest.csv']:
        table=read('data/figure6/'+name)
        for row in table.itertuples():
            grid_paths.add(row.cache_path)
            with np.load(ROOT/row.cache_path,allow_pickle=False) as grid:
                check(grid['logF'].shape==(row.grid_N,row.grid_N),'Grid shape')
                check(not np.isnan(grid['logF']).any(),'Grid contains NaN')
                check(np.all(np.diff(grid['xs'])>0) and np.all(np.diff(grid['ys'])>0),'Grid coordinate ordering')
    check(len(grid_paths)==13,'Grid cache count')
    exact_contours=0
    with np.load(ROOT/'data/figure6/contours.npz',allow_pickle=False) as contours:
        check(len(contours.files)==len(meta)==13,'Contour count')
        for row in meta.itertuples():
            coords=contours[row.contour_key]
            check(coords.ndim==2 and coords.shape[1]==2 and np.isfinite(coords).all(),'Contour shape/data')
            if pd.isna(row.cache_path):continue
            with np.load(ROOT/row.cache_path,allow_pickle=False) as grid:
                if row.center_type=='connected_E0_E1':
                    result=component_contours_from_log_grid(grid['logF'],grid['xs'],grid['ys'],(0j,complex(row.E1,0)),-32.0)
                else:
                    result=single_component_contours_from_log_grid(grid['logF'],grid['xs'],grid['ys'],0j,-32.0,force_seed=False)
                check(any(np.array_equal(candidate,coords) for candidate in result['contours']),
                      'Grid contour differs from archived contour: '+row.contour_key)
                exact_contours+=1
    threshold_errors=[]
    for row in pair.itertuples():
        with np.load(ROOT/row.cache_path,allow_pickle=False) as grid:
            gap=float(clock_eigenvalues(int(row.L))[1])
            result=pseudospectral_closing_threshold_grid_log(grid['logF'],grid['xs'],grid['ys'],
                     eigenvalues=[0j,complex(gap,0)],ground_indices=[0],excited_indices=[1])
            expected=float(thresholds.loc[thresholds.r==row.r,'log10_epsilon_c_grid_hp'].iloc[0])
            error=abs(result['log10_epsilon_c']-expected)
            check(error<1e-11,'Grid threshold differs from archived threshold')
            threshold_errors.append(error)
    report['figure6']={'grid_files':13,'contours':13,'fk_contours_exactly_reextracted':exact_contours,
        'thresholds_recomputed':5,'max_log10_threshold_error':max(threshold_errors)}
    print('PASS Figure 6: all 13 grids, 9 FK contours and 5 closing thresholds',flush=True)

    raw=read('data/figure7/realizations.csv')
    summary=read('data/figure7/summary.csv')
    check(len(raw)==4500 and len(summary)==45,'Figure 7 record count')
    check(raw.status.eq('success').all(),'Figure 7 failed realization')
    check(not raw.duplicated(['algorithm','epsilon','realization']).any(),'Figure 7 duplicate realization')
    from experiment_three_noisy_dynamics import derive_seed
    for row in raw.itertuples():
        check(derive_seed(1234,row.algorithm,row.epsilon,row.realization,'paired_noise')==row.seed,'Noise seed mismatch')
    for row in summary.itertuples():
        group=raw[(raw.algorithm==row.algorithm)&(raw.epsilon==row.epsilon)]
        check(len(group)==row.n_success==100 and set(group.realization)==set(range(100)),'Figure 7 sample count/IDs')
        for field in ['p_raw','p_normalized']:
            values=group[field].to_numpy()
            actual=[np.min(values),np.quantile(values,.25),np.median(values),np.quantile(values,.75),np.max(values)]
            expected=[getattr(row,prefix+'_'+field) for prefix in ['min','q1','median','q3','max']]
            close(actual,expected,'Figure 7 five-number summary')
    for field,value in [('seed_mode','paired_noise'),('n_noise',70),('steps_per_segment',8),
                        ('integrator','combined'),('norm_iters',64)]:
        check(raw[field].eq(value).all(),'Figure 7 parameter: '+field)
    check(raw.loc[raw.algorithm=='NHMIS-FKQAA','fk_input_term'].eq('none').all(),'FK input term')
    close(raw.p_normalized,raw.p_raw/raw.p0,'Normalized probability')
    report['figure7']={'realizations':4500,'summary_groups':45,'samples_per_group':100,
        'seed_base':1234,'all_task_seeds_verified':True,'stored_n_real_values':{str(k):int(v) for k,v in raw.n_real.value_counts().items()}}
    print('PASS Figure 7: all seeds, sample IDs and min/quartiles/median/max statistics',flush=True)

    if smoke:
        from experiment_one_ck import prepare_ck_instance,run_prepared_instance
        result=run_prepared_instance(prepare_ck_instance(3),gamma=10.,fk_step_factor=16)
        expected=ck.loc[ck.m==3].iloc[0]
        errors={key:abs(result[key]-float(expected[key])) for key in ['p_FK','p_NHHD','p_HHD']}
        for key in errors:close(result[key],expected[key],'CK m=3 recomputation',rtol=1e-8,atol=1e-10)
        from experiment_two_resolvent_pseudospectrum import log10_sigma_min_fk_sector_hp,log10_sigma_min_fk_sector_mpmath_svd
        v=[2.,4.,.5];z=.17+.09j
        iterative=log10_sigma_min_fk_sector_hp(v,z,mp_dps=70,n_iter=80)
        dense=log10_sigma_min_fk_sector_mpmath_svd(v,z,mp_dps=70)
        close(iterative,dense,'High-precision inverse iteration vs dense SVD',rtol=0,atol=1e-7)
        report['smoke']={'ck_m3_absolute_errors':errors,'log10_sigma_error':abs(iterative-dense)}
        print('PASS smoke: CK m=3 for all three algorithms; high-precision solver vs dense SVD',flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='Also recompute a small CK instance and compare high-precision solvers')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();start=time.perf_counter()
    report=verify(args.smoke);report['elapsed_seconds']=time.perf_counter()-start
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"All requested checks passed in {report['elapsed_seconds']:.1f} s")
