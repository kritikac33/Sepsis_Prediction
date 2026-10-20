"""Train a strictly train-selected physiology-only XGBoost improvement cohort.

All outputs are new ``*_xgbimprove`` artifacts.  The pre-existing project
split, models, metrics, API and frontend are never read for selection or
modified.  Run from the project root with .\\venv\\Scripts\\python.exe.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DATA = ROOT / "data" / "processed"
OUT_MODELS = ROOT / "outputs" / "models"
OUT = ROOT / "outputs"
SEED = 20261015
BASE_FEATURES = ("Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel")
VITALS = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
ABNORMAL_RANGES = {"HR": (50, 100), "Temp": (36, 38), "Resp": (12, 20),
                   "MAP": (65, 110), "O2Sat": (92, 100), "Lactate": (0, 2)}

BASE_PARAMS = dict(n_estimators=30, max_depth=3, learning_rate=0.1,
                   objective="binary:logistic", eval_metric="aucpr",
                   tree_method="hist", n_jobs=-1, random_state=SEED)

@dataclass
class Representation:
    hours: int
    extras: bool

def patients() -> tuple[list[Path], np.ndarray]:
    paths = sorted(RAW.glob("*.psv"))
    if not paths: raise FileNotFoundError(f"No PSV files in {RAW}")
    labels = []
    for number, path in enumerate(paths, 1):
        labels.append(int(pd.read_csv(path, sep="|", usecols=["SepsisLabel"])["SepsisLabel"].fillna(0).any()))
        if number % 500 == 0:
            print(f"  read labels for {number}/{len(paths)} patients", flush=True)
    return paths, np.asarray(labels, dtype=np.int8)

def build_patient(path: Path, feature_names: list[str], medians: np.ndarray | None, rep: Representation) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep="|")
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    measured = numeric.notna().to_numpy(dtype=np.float32)
    # No backward fill: future values never fill an earlier hour.
    values = numeric.ffill().to_numpy(dtype=np.float32)
    if medians is not None:
        values = np.where(np.isnan(values), medians, values)
    labels = pd.to_numeric(frame["SepsisLabel"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    rows, out_labels = [], []
    for end in range(rep.hours - 1, len(values)):
        w, m = values[end-rep.hours+1:end+1], measured[end-rep.hours+1:end+1]
        row = np.concatenate([w[-1], w.mean(0), w.min(0), w.max(0)])
        if rep.extras:
            first = w[0]; std = w.std(0); delta3 = w[-1] - w[max(0, len(w)-3)]
            since = np.full(w.shape[1], rep.hours, dtype=np.float32)
            for col in range(w.shape[1]):
                observed = np.flatnonzero(m[:, col] > 0)
                if observed.size: since[col] = len(w) - 1 - observed[-1]
            idx = {name:i for i,name in enumerate(feature_names)}
            abnormal = np.zeros(w.shape[1], dtype=np.float32)
            for feature, (low, high) in ABNORMAL_RANGES.items():
                if feature in idx:
                    abnormal[idx[feature]] = float(((w[:,idx[feature]] < low) | (w[:,idx[feature]] > high)).sum())
            derived=[]
            if "HR" in idx and "SBP" in idx: derived.append(w[-1,idx["HR"]] / max(w[-1,idx["SBP"]], 1.0))
            if "O2Sat" in idx and "FiO2" in idx: derived.append(w[-1,idx["O2Sat"]] / max(w[-1,idx["FiO2"]], .01))
            row=np.concatenate([row, first, std, delta3, since, abnormal.astype(np.float32), np.asarray(derived,dtype=np.float32)])
        rows.append(row); out_labels.append(labels[end])
    return np.asarray(rows,dtype=np.float32), np.asarray(out_labels,dtype=np.int8)

def names_for(base: list[str], rep: Representation) -> list[str]:
    output=[f"{name}_{stat}" for stat in ("last","mean","min","max") for name in base]
    if rep.extras:
        output += [f"{name}_{stat}" for stat in ("first","std","delta3","hours_since_measured","abnormal_count") for name in base]
        output += ["shock_index", "spo2_fio2"]
    return output

def construct(paths: list[Path], base: list[str], medians: np.ndarray, rep: Representation) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    xs=[]; ys=[]; groups=[]
    for count,path in enumerate(paths,1):
        x,y=build_patient(path,base,medians,rep)
        if len(x): xs.append(x);ys.append(y);groups.append(np.full(len(y),path.stem))
        if count % 500 == 0: print(f"  represented {count}/{len(paths)} patients",flush=True)
    return np.vstack(xs),np.concatenate(ys),np.concatenate(groups)

def fit_predict(params: dict, xtr, ytr, xva, yva, seed: int) -> np.ndarray:
    params={**params,"random_state":seed,"n_estimators":1000,"early_stopping_rounds":40}
    model=xgb.XGBClassifier(**params)
    model.fit(xtr,ytr,eval_set=[(xva,yva)],verbose=False)
    return model.predict_proba(xva)[:,1]

def cv_pr(params: dict, x: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int=SEED) -> float:
    scores=[]
    for train_rows,val_rows in GroupKFold(n_splits=3).split(x,y,g):
        scores.append(average_precision_score(y[val_rows],fit_predict(params,x[train_rows],y[train_rows],x[val_rows],y[val_rows],seed)))
    return float(np.mean(scores))

def ci(values): return [float(item) for item in np.percentile(values,[2.5,97.5])]
def boot(y,g,baseline,improved):
    groups=np.unique(g); rng=np.random.default_rng(SEED); records={key:[] for key in ("b_auc","b_pr","i_auc","i_pr","delta")}
    for _ in range(200):
        chosen=rng.choice(groups,len(groups),replace=True); rows=np.concatenate([np.flatnonzero(g==p) for p in chosen])
        if np.unique(y[rows]).size<2: continue
        records["b_auc"].append(roc_auc_score(y[rows],baseline[rows]));records["b_pr"].append(average_precision_score(y[rows],baseline[rows]));records["i_auc"].append(roc_auc_score(y[rows],improved[rows]));records["i_pr"].append(average_precision_score(y[rows],improved[rows]));records["delta"].append(roc_auc_score(y[rows],improved[rows])-roc_auc_score(y[rows],baseline[rows]))
    return {key:ci(value) for key,value in records.items()}

def main() -> None:
    paths,patient_labels=patients(); ids=np.asarray([path.stem for path in paths])
    train_ids,test_ids=train_test_split(ids,test_size=.20,random_state=SEED,stratify=patient_labels)
    train_set=set(train_ids); train_paths=[p for p in paths if p.stem in train_set]; test_paths=[p for p in paths if p.stem not in train_set]
    first=pd.read_csv(paths[0],sep="|"); base=[name for name in first.columns if name not in BASE_FEATURES]
    # Scaling/imputation statistics are fitted from raw train patients only.
    train_values=[]
    for count,path in enumerate(train_paths,1):
        train_values.append(pd.read_csv(path,sep="|")[base].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float32))
        if count % 500 == 0: print(f"  fitted train statistics from {count}/{len(train_paths)} patients",flush=True)
    medians=np.nanmedian(np.vstack(train_values),axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    del train_values
    split={"seed":SEED,"train_patient_ids":train_ids.tolist(),"test_patient_ids":test_ids.tolist(),"train_patient_count":len(train_ids),"test_patient_count":len(test_ids),"train_septic_patient_count":int(patient_labels[np.isin(ids,train_ids)].sum()),"test_septic_patient_count":int(patient_labels[np.isin(ids,test_ids)].sum()),"feature_exclusions":["ICULOS","Unit1","Unit2","HospAdmTime"]}
    (OUT_DATA/"split_xgbimprove.json").write_text(json.dumps(split,indent=2),encoding="utf-8")
    print(f"Patients: train={len(train_ids)} ({split['train_septic_patient_count']} septic), test={len(test_ids)} ({split['test_septic_patient_count']} septic)")
    # Representation selection is train-only.
    candidates=[Representation(6,False),Representation(6,True),Representation(12,False),Representation(12,True)]
    prepared={}
    for rep in candidates:
        print(f"Building train representation: {rep}",flush=True); prepared[(rep.hours,rep.extras)]=construct(train_paths,base,medians,rep)
    rep_scores={}
    for rep in candidates:
        x,y,g=prepared[(rep.hours,rep.extras)]; rep_scores[f"{rep.hours}h_extras={rep.extras}"]=cv_pr(BASE_PARAMS,x,y,g)
        print(f"TRAIN-only 3-fold PR-AUC {rep}: {rep_scores[f'{rep.hours}h_extras={rep.extras}']:.5f}",flush=True)
    chosen=candidates[int(np.argmax(list(rep_scores.values())))]; xtrain,ytrain,gtrain=prepared[(chosen.hours,chosen.extras)]
    rng=np.random.default_rng(SEED); ratio=float(np.sqrt((ytrain==0).sum()/max((ytrain==1).sum(),1)))
    configs=[]
    for _ in range(15): configs.append({"max_depth":int(rng.integers(2,6)),"learning_rate":float(rng.uniform(.02,.05)),"min_child_weight":float(rng.uniform(5,30)),"gamma":float(rng.uniform(0,2)),"subsample":float(rng.uniform(.6,.9)),"colsample_bytree":float(rng.uniform(.5,.9)),"reg_lambda":float(rng.uniform(1,20)),"scale_pos_weight":float(rng.choice([1,ratio])),"objective":"binary:logistic","eval_metric":"aucpr","tree_method":"hist","n_jobs":-1})
    cv_scores=[]
    for number,params in enumerate(configs,1):
        value=cv_pr(params,xtrain,ytrain,gtrain);cv_scores.append(value);print(f"random search {number}/15 train-only PR-AUC={value:.5f}",flush=True)
    best=configs[int(np.argmax(cv_scores))]
    # Create the matching held-out test representation only after all choices above.
    xtest,ytest,gtest=construct(test_paths,base,medians,chosen)
    # Baseline is evaluated once on untouched test patients.
    base_model=xgb.XGBClassifier(**BASE_PARAMS);base_model.fit(xtrain,ytrain);base_scores=base_model.predict_proba(xtest)[:,1]
    ensemble=[];score_list=[]
    for seed in (11,23,37,53,71):
        # Patient-level validation slice is from train only for early stopping.
        tr_groups,va_groups=train_test_split(np.unique(gtrain),test_size=.15,random_state=seed,stratify=np.array([int(ytrain[gtrain==group].any()) for group in np.unique(gtrain)]))
        tr=np.isin(gtrain,tr_groups); va=np.isin(gtrain,va_groups); model=xgb.XGBClassifier(**{**best,"n_estimators":1000,"early_stopping_rounds":40,"random_state":seed});model.fit(xtrain[tr],ytrain[tr],eval_set=[(xtrain[va],ytrain[va])],verbose=False);ensemble.append(model);score_list.append(model.predict_proba(xtest)[:,1])
    improved_scores=np.mean(score_list,axis=0); intervals=boot(ytest,gtest,base_scores,improved_scores)
    result={"cohort":split,"chosen_representation":asdict(chosen),"representation_train_only_cv_pr_auc":rep_scores,"random_search_train_only_cv_pr_auc":cv_scores,"chosen_settings":best,"baseline":{"test_auroc":float(roc_auc_score(ytest,base_scores)),"test_pr_auc":float(average_precision_score(ytest,base_scores))},"improved":{"test_auroc":float(roc_auc_score(ytest,improved_scores)),"test_pr_auc":float(average_precision_score(ytest,improved_scores))},"patient_bootstrap_ci95":intervals}
    joblib.dump({"models":ensemble,"feature_names":names_for(base,chosen),"imputation_medians":medians,"representation":asdict(chosen),"split_path":"data/processed/split_xgbimprove.json"},OUT_MODELS/"xgb_improved.pkl")
    (OUT/"metrics_xgbimprove.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    text="# XGBoost physiology-only improvement\n\n| Model | Test AUROC (95% patient bootstrap CI) | Test PR-AUC (95% patient bootstrap CI) |\n|---|---:|---:|\n| Baseline | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |\n| Improved (5-seed average) | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |\n\nChosen only with train-patient 3-fold GroupKFold PR-AUC: `%s`.\n\nPaired improved-minus-baseline AUROC CI: [%.4f, %.4f].\n" % (result['baseline']['test_auroc'],*intervals['b_auc'],result['baseline']['test_pr_auc'],*intervals['b_pr'],result['improved']['test_auroc'],*intervals['i_auc'],result['improved']['test_pr_auc'],*intervals['i_pr'],json.dumps(best),*intervals['delta'])
    (OUT/"xgb_improvement.md").write_text(text,encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
