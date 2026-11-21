"""Rock Physics ML core: synthetic-well training and held-out-well evaluation."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ['VSH','VSAND','VCARB','POR','SW','PD']
TARGETS = ['VP','VS']

@dataclass
class WellData:
    frame: pd.DataFrame
    @classmethod
    def load(cls, path):
        return cls(pd.read_csv(path))
    def validate(self):
        needed = FEATURES + TARGETS + ['WELL']
        missing = [c for c in needed if c not in self.frame]
        if missing: raise ValueError(f'Missing columns: {missing}')
        if self.frame.empty: raise ValueError('Dataset is empty')
        if self.frame[needed].isna().any().any(): raise ValueError('Missing values in required columns')
        if not np.isfinite(self.frame[FEATURES + TARGETS].to_numpy(dtype=float)).all():
            raise ValueError('Non-finite values found')
        if self.frame['WELL'].nunique() < 3: raise ValueError('At least three distinct wells are needed')
        return True

@dataclass
class PreparedDataSet:
    train: pd.DataFrame
    test: pd.DataFrame
    blind: pd.DataFrame
    @classmethod
    def create(cls, data, train_fraction=0.7, test_fraction=0.2, seed=42):
        data.validate()
        blind_fraction = 1 - train_fraction - test_fraction
        if min(train_fraction, test_fraction, blind_fraction) <= 0: raise ValueError('All splits must be positive')
        wells = sorted(data.frame.WELL.unique())
        if len(wells) * min(train_fraction, test_fraction, blind_fraction) < 1:
            raise ValueError('Split fractions too small for number of wells')
        tr, other = train_test_split(wells, train_size=train_fraction, random_state=seed)
        te, bl = train_test_split(other, train_size=test_fraction/(1-train_fraction), random_state=seed)
        return cls(*(data.frame[data.frame.WELL.isin(group)].copy() for group in (tr,te,bl)))

class ModelTrainingController:
    def __init__(self): self.models = {}
    def train(self, prepared):
        X = prepared.train[FEATURES]; y = prepared.train[TARGETS]
        self.models = {
            'SVR': make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=10, epsilon=0.05))),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'MLP': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), max_iter=600, early_stopping=True, random_state=42))
        }
        for model in self.models.values(): model.fit(X,y)
        return self.models

class PredictionController:
    @staticmethod
    def predict(models, frame):
        if not models: raise ValueError('Train models first')
        out = frame[['WELL'] + (['DEPTH'] if 'DEPTH' in frame else []) + TARGETS].copy()
        for name, model in models.items():
            predictions = model.predict(frame[FEATURES])
            for idx, target in enumerate(TARGETS): out[f'{name}_{target}'] = predictions[:,idx]
        return out

class EvaluationController:
    @staticmethod
    def evaluate(predictions):
        results=[]
        for column in predictions:
            for target in TARGETS:
                if column.endswith('_'+target):
                    y=predictions[target]; pred=predictions[column]
                    results.append({'model':column[:-(len(target)+1)], 'target':target,
                                    'R2':r2_score(y,pred), 'RMSE':np.sqrt(mean_squared_error(y,pred)),
                                    'MAE':mean_absolute_error(y,pred),
                                    'MAPE (%)':float(np.mean(np.abs((np.asarray(y)[np.abs(np.asarray(y)) > 1e-12] - np.asarray(pred)[np.abs(np.asarray(y)) > 1e-12]) / np.asarray(y)[np.abs(np.asarray(y)) > 1e-12])) * 100) if np.any(np.abs(np.asarray(y)) > 1e-12) else float('nan')})
        return pd.DataFrame(results)

class VisualizationController:
    @staticmethod
    def plot(predictions, output, label="blind validation"):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        for i, target in enumerate(TARGETS):
            ax[i].plot(predictions[target].to_numpy(),label='Known',linewidth=1.5)
            for col in predictions:
                if col.endswith('_'+target): ax[i].plot(predictions[col].to_numpy(),label=col.rsplit('_',1)[0],alpha=.7)
            ax[i].set_title(target+' — '+label); ax[i].set_xlabel('Sample'); ax[i].legend()
        fig.tight_layout(); fig.savefig(output,dpi=140); plt.close(fig)

class DataPreparationController:
    @staticmethod
    def load(path): return WellData.load(path)
    @staticmethod
    def prepare(data, train_fraction=.7, test_fraction=.2):
        return PreparedDataSet.create(data,train_fraction,test_fraction)

class UserInterface:
    """Command-line entry point; GUI will be added in the next milestone."""
    @staticmethod
    def run(path, output):
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        data=DataPreparationController.load(path)
        prepared=DataPreparationController.prepare(data)
        print('Rows:',len(data.frame),'Wells:',data.frame.WELL.nunique())
        print('Split wells:',*[f'{name}={frame.WELL.nunique()}' for name,frame in [('train',prepared.train),('test',prepared.test),('blind',prepared.blind)]])
        models=ModelTrainingController().train(prepared)
        for label,frame in [('test',prepared.test),('blind',prepared.blind)]:
            predictions=PredictionController.predict(models,frame)
            metrics=EvaluationController.evaluate(predictions)
            predictions.to_csv(output/f'{label}_predictions.csv',index=False)
            metrics.to_csv(output/f'{label}_metrics.csv',index=False)
            print('\n'+label.upper()+'\n'+metrics.to_string(index=False))
            if label=='blind': VisualizationController.plot(predictions,output/'blind_plot.png')
        print('\nResults saved to',output)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/synthetic_wells_combined.csv')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    UserInterface.run(args.data,args.output)
