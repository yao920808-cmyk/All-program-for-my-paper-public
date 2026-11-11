"""Self-contained integration verification; synthetic data do not measure research accuracy."""
from pathlib import Path
import tempfile
import subprocess
import sys
import numpy as np
import pandas as pd
from main import CHANNELS

def main():
    rng=np.random.default_rng(42)
    here=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); root=d/'trials'; records=[]
        for i in range(12):
            subject=root/f'{i:03d}'; subject.mkdir(parents=True)
            records.append({'index':f'{i:03d}','gender':'f' if i%2==0 else 'm'})
            for j in range(4):
                pd.DataFrame({c:rng.normal(i%2,1,32) for c in CHANNELS}).to_csv(subject/f'{j}.csv',index=False)
        pd.DataFrame(records).to_csv(d/'metadata.csv',index=False)
        def run(*args):
            subprocess.run([sys.executable,*map(str,args)],cwd=here,check=True)
        for raw in [False,True]:
            csv=d/('raw.csv' if raw else 'features.csv')
            run(here/'main.py','extract','--root',root,'--metadata',d/'metadata.csv','--output',csv,*(['--raw'] if raw else []))
            for model in ['rf','nn','cnn','svm','mlp','linear']:
                run(here/'main.py','train','--csv',csv,'--model',model,'--epochs',1,'--output',d/model,*(['--raw'] if raw else []))
        run(here/'main.py','train','--csv',d/'raw.csv','--raw','--target','index','--epochs',1,'--output',d/'identity')
        run(here/'diffusion.py','--csv',d/'features.csv','--epochs',1,'--steps',4,'--output',d/'diffusion')
    print('PASS: extraction, six models on both representations, identity, diffusion')

if __name__=='__main__': main()
