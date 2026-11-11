"""GRF baseline and 108-descriptor experiments; command line entry point."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from to_juefei import compute_td_all_A

CHANNELS = ['leftVForce','leftAPForce','leftMLForce','rightVForce','rightAPForce','rightMLForce']

def scalar(v):
    if isinstance(v, dict): v = list(v.values())
    a = np.asarray(v, dtype=float).ravel()
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan

def extract(a):
    meta = pd.read_csv(a.metadata, dtype=str).fillna('')
    mapping = {'f':0,'female':0,'0':0,'m':1,'male':1,'1':1}
    genders = {str(i).removesuffix('.0').zfill(3):mapping.get(g.strip().lower(), -1)
               for i,g in zip(meta['index'],meta['gender'])}
    root = Path(a.root)
    files = sorted(root.rglob('*.csv'))
    if a.limit: files = files[:a.limit]
    rows, errors = [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not set(CHANNELS) <= set(df): raise ValueError('Missing force channels')
            candidate = f.relative_to(root).parts[0].zfill(3)
            if genders.get(candidate,-1) < 0: raise ValueError('Missing gender metadata: '+candidate)
            if a.raw:
                frame = df[CHANNELS].apply(pd.to_numeric,errors='raise')
                frame['gender'] = genders[candidate]
                frame['index'] = int(candidate)
                frame['source'] = str(f.relative_to(root))
                rows.extend(frame.to_dict('records'))
            else:
                row = {}
                for c in CHANNELS:
                    x = pd.to_numeric(df[c], errors='raise').to_numpy(float)
                    if not len(x) or not np.isfinite(x).all(): raise ValueError('Empty/nonfinite signal')
                    row.update({c+'_'+k:scalar(v) for k,v in compute_td_all_A(x).items()})
                row.update(gender=genders[candidate],candidate=candidate,source=str(f.relative_to(root)))
                rows.append(row)
        except Exception as e:
            errors.append({'file':str(f),'error':str(e)})
    target = Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.with_suffix('.errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    if errors: raise ValueError(f'{len(errors)} invalid files; see error report. Output not replaced.')
    if not rows: raise ValueError('No trials found; verify --root')
    pd.DataFrame(rows).to_csv(target,index=False)
    print(f'Saved {len(rows)} rows to {target}')

def train(a):
    df = pd.read_csv(a.csv, nrows=a.limit)
    if a.target not in df: raise ValueError('Target column missing')
    if df[a.target].isna().any(): raise ValueError('Missing labels')
    if a.target == 'gender' and not set(df[a.target].unique()) <= {0,1}: raise ValueError('gender must be 0 or 1')
    cols = CHANNELS if a.raw else [c for c in df if c not in ['gender','candidate','source','index']]
    if not a.raw and len(cols)!=108: raise ValueError(f'Expected 108 features, got {len(cols)}')
    X = df[cols].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan).to_numpy()
    enc = LabelEncoder(); y = enc.fit_transform(df[a.target])
    if len(enc.classes_) < 2: raise ValueError('At least two classes required')
    tr,tmp = train_test_split(np.arange(len(y)),test_size=.5,random_state=42,stratify=y)
    va,te = train_test_split(tmp,test_size=.5,random_state=42,stratify=y[tmp])
    prep = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
    xt = prep.fit_transform(X[tr]).astype('float32')
    xv = prep.transform(X[va]).astype('float32'); xe = prep.transform(X[te]).astype('float32')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'preprocessing':prep,'labels':enc,'features':cols},out/'preprocessing.joblib')
    if a.model in ['rf','svm','linear']:
        model = {'rf':lambda:RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2),
                 'svm':lambda:SVC(C=10), 'linear':lambda:LinearRegression()}[a.model]()
        model.fit(xt,y[tr]); pred=model.predict(xe)
        joblib.dump(model,out/'model.joblib')
    else:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader,TensorDataset
        torch.manual_seed(42); torch.set_num_threads(2)
        k=xt.shape[1]; n=len(enc.classes_)
        if a.model=='cnn':
            model=nn.Sequential(nn.Unflatten(1,(1,k)),nn.Conv1d(1,64,3,padding=1),nn.ReLU(),nn.BatchNorm1d(64),nn.Dropout(.3),nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(.3),nn.Conv1d(128,64,3,padding=1),nn.ReLU(),nn.BatchNorm1d(64),nn.Dropout(.2),nn.Flatten(),nn.Linear(64*k,n))
        else:
            widths=[256,128,64,32] if a.model=='nn' else [128,128]
            layers=[]; previous=k
            for i,w in enumerate(widths):
                layers.extend([nn.Linear(previous,w),nn.ReLU()])
                if i<2: layers.append(nn.Dropout(.2))
                previous=w
            model=nn.Sequential(*layers,nn.Linear(previous,n))
        opt=torch.optim.Adam(model.parameters(),lr=.0005 if a.model=='cnn' else .001)
        lossfn=nn.CrossEntropyLoss(); best=float('inf'); logs=[]
        loader=DataLoader(TensorDataset(torch.tensor(xt),torch.tensor(y[tr])),batch_size=a.batch_size,shuffle=True)
        for epoch in range(a.epochs):
            model.train(); total=0
            for xb,yb in loader:
                opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
            model.eval()
            with torch.no_grad(): vl=lossfn(model(torch.tensor(xv)),torch.tensor(y[va])).item()
            logs.append({'epoch':epoch+1,'train_loss':total/len(tr),'validation_loss':vl})
            if vl<best:
                best=vl; torch.save(model.state_dict(),out/'model.pt')
            print(logs[-1])
        model.load_state_dict(torch.load(out/'model.pt',weights_only=True)); model.eval()
        with torch.no_grad(): pred=model(torch.tensor(xe)).argmax(1).numpy()
        pd.DataFrame(logs).to_csv(out/'history.csv',index=False)
    result={'model':a.model,'samples':len(y),'features':len(cols),'split':[len(tr),len(va),len(te)],'seed':42,'classes':enc.classes_.tolist()}
    if a.model=='linear': result['mse']=mean_squared_error(y[te],pred)
    else:
        result['accuracy']=accuracy_score(y[te],pred)
        result['classification_report']=classification_report(y[te],pred,output_dict=True,zero_division=0)
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame({'row':te,'actual':y[te],'prediction':pred}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract'); e.add_argument('--root',required=True); e.add_argument('--metadata',required=True); e.add_argument('--output',required=True); e.add_argument('--limit',type=int); e.add_argument('--raw',action='store_true'); e.set_defaults(func=extract)
    t=sub.add_parser('train'); t.add_argument('--csv',required=True); t.add_argument('--model',choices=['rf','nn','cnn','svm','mlp','linear'],default='nn'); t.add_argument('--output',default='results/run'); t.add_argument('--epochs',type=int,default=200); t.add_argument('--batch-size',type=int,default=128); t.add_argument('--limit',type=int); t.add_argument('--raw',action='store_true'); t.add_argument('--target',choices=['gender','index'],default='gender'); t.set_defaults(func=train)
    a=p.parse_args()
    if getattr(a,'epochs',1)<1 or getattr(a,'batch_size',1)<1 or (a.limit is not None and a.limit<1): p.error('Numeric arguments must be positive')
    a.func(a)

if __name__=='__main__': main()
