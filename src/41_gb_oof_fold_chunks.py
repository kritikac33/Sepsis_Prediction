"""Resume one GradientBoosting OOF fold in ten-tree chunks, then record scores."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/features_v2.npz'; MODELS=ROOT/'outputs/models'; OUT=ROOT/'outputs/metrics'
def main(fold_no: int):
    with np.load(DATA,allow_pickle=True) as d:
        names=d['feature_names'].astype(str).tolist(); keep=np.array([not any(t in n for t in ('Unit1','Unit2','HospAdmTime','ICULOS')) for n in names])
        x,y,g=d['X_train'][:,keep],d['y_train'].astype(int),d['patient_ids_train'].astype(str)
    folds=list(GroupKFold(n_splits=3).split(x,y,g)); fit,valid=folds[fold_no-1]
    OUT.mkdir(exist_ok=True); checkpoint=OUT/f'gb_oof_fold_{fold_no}_checkpoint.pkl'; scores_path=OUT/f'gb_oof_fold_{fold_no}_scores.npz'
    if scores_path.exists(): print(f'Fold {fold_no} already complete.'); return
    template=joblib.load(MODELS/'gb_noiculos.pkl'); target=template.get_params()['n_estimators']
    if checkpoint.exists(): model=joblib.load(checkpoint); start=model.n_estimators
    else: model=clone(template); model.set_params(warm_start=True,n_estimators=10); start=0
    current=min(start+10,target); model.set_params(warm_start=True,n_estimators=current)
    print(f'GB OOF fold {fold_no}: trees {start+1}-{current}/{target}',flush=True); model.fit(x[fit],y[fit])
    if current < target: joblib.dump(model,checkpoint); print(f'Checkpoint {current}/{target}')
    else:
        model.set_params(warm_start=False); np.savez_compressed(scores_path, valid=valid, scores=model.predict_proba(x[valid])[:,1]); print(f'Fold {fold_no} scores saved')
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('fold',type=int,choices=[1,2,3]);main(p.parse_args().fold)
