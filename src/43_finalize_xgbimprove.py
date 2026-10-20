"""Memory-bounded finalization of the already selected XGBoost improvement.

Selection is not repeated: it consumes the train-only winner recorded by
42_xgbimprove.py (12h plus extra features, random-search configuration 6).
It constructs one matrix at a time, then evaluates test patients once.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("improve",ROOT/"src"/"42_xgbimprove.py"); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
SPLIT=json.loads((ROOT/"data/processed/split_xgbimprove.json").read_text())
OUT=ROOT/"outputs"; MODELS=OUT/"models"; SEED=m.SEED

def medians(paths,base):
    chunks=[]
    for n,path in enumerate(paths,1):
        chunks.append(pd.read_csv(path,sep="|")[base].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32))
        if n%1000==0: print(f"statistics {n}/{len(paths)}",flush=True)
    value=np.nanmedian(np.vstack(chunks),axis=0).astype(np.float32);value[~np.isfinite(value)]=0;return value
def config6(y):
    rng=np.random.default_rng(SEED); ratio=float(np.sqrt((y==0).sum()/max((y==1).sum(),1))); result=[]
    for _ in range(15): result.append({"max_depth":int(rng.integers(2,6)),"learning_rate":float(rng.uniform(.02,.05)),"min_child_weight":float(rng.uniform(5,30)),"gamma":float(rng.uniform(0,2)),"subsample":float(rng.uniform(.6,.9)),"colsample_bytree":float(rng.uniform(.5,.9)),"reg_lambda":float(rng.uniform(1,20)),"scale_pos_weight":float(rng.choice([1,ratio])),"objective":"binary:logistic","eval_metric":"aucpr","tree_method":"hist","n_jobs":-1})
    return result[5]
def main():
    all_paths,_=m.patients(); train_ids=set(SPLIT["train_patient_ids"]); train=[p for p in all_paths if p.stem in train_ids]; test=[p for p in all_paths if p.stem not in train_ids]
    base=[n for n in pd.read_csv(all_paths[0],sep="|").columns if n not in m.BASE_FEATURES]; med=medians(train,base)
    # Same 12-hour endpoints as the selected model: this preserves valid
    # paired patient resampling while retaining the current baseline params.
    print("building baseline 12h-basic representations",flush=True)
    xtr,ytr,gtr=m.construct(train,base,med,m.Representation(12,False)); xte,yte,gte=m.construct(test,base,med,m.Representation(12,False))
    baseline=xgb.XGBClassifier(**m.BASE_PARAMS); baseline.fit(xtr,ytr); bscore=baseline.predict_proba(xte)[:,1]; del xtr
    print("building selected 12h-extra representations",flush=True)
    xtr,ytr,gtr=m.construct(train,base,med,m.Representation(12,True)); xte_i,yte_i,gte_i=m.construct(test,base,med,m.Representation(12,True))
    if not np.array_equal(yte,yte_i) or not np.array_equal(gte,gte_i): raise RuntimeError("Test window ordering mismatch")
    best=config6(ytr); predictions=[]; ensemble=[]
    unique=np.unique(gtr)
    for seed in (11,23,37,53,71):
        labels=np.array([int(ytr[gtr==p].any()) for p in unique]); trp,vap=train_test_split(unique,test_size=.15,random_state=seed,stratify=labels)
        tr=np.isin(gtr,trp);va=np.isin(gtr,vap); model=xgb.XGBClassifier(**{**best,"n_estimators":1000,"early_stopping_rounds":40,"random_state":seed});model.fit(xtr[tr],ytr[tr],eval_set=[(xtr[va],ytr[va])],verbose=False); predictions.append(model.predict_proba(xte_i)[:,1]); ensemble.append(model);print(f"five-seed fit {seed} complete",flush=True)
    iscore=np.mean(predictions,axis=0); intervals=m.boot(yte,gte,bscore,iscore)
    result={"cohort":SPLIT,"selection":"12h+extras selected by prior train-only 3-fold GroupKFold; config 6 selected by prior train-only 15-configuration search","chosen_settings":best,"baseline":{"test_auroc":float(roc_auc_score(yte,bscore)),"test_pr_auc":float(average_precision_score(yte,bscore))},"improved":{"test_auroc":float(roc_auc_score(yte,iscore)),"test_pr_auc":float(average_precision_score(yte,iscore))},"patient_bootstrap_ci95":intervals}
    joblib.dump({"models":ensemble,"feature_names":m.names_for(base,m.Representation(12,True)),"imputation_medians":med,"representation":{"hours":12,"extras":True},"split_path":"data/processed/split_xgbimprove.json","settings":best},MODELS/"xgb_improved.pkl")
    (OUT/"metrics_xgbimprove.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    text="# XGBoost physiology-only improvement\n\n| Model | Test AUROC (95% patient bootstrap CI) | Test PR-AUC (95% patient bootstrap CI) |\n|---|---:|---:|\n| Baseline | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |\n| Improved (five-seed average) | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |\n\nChosen settings (train-only): `%s`\n\nPaired improved-minus-baseline AUROC CI: [%.4f, %.4f].\n"%(result['baseline']['test_auroc'],*intervals['b_auc'],result['baseline']['test_pr_auc'],*intervals['b_pr'],result['improved']['test_auroc'],*intervals['i_auc'],result['improved']['test_pr_auc'],*intervals['i_pr'],json.dumps(best),*intervals['delta'])
    (OUT/"xgb_improvement.md").write_text(text)
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
