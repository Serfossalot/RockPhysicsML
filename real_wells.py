"""Import and evaluate external real wells without fitting on their measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from rockphysics import FEATURES, TARGETS, PredictionController, EvaluationController

# Column order documented in RockPhysics_Theoretical_Model.ipynb, cell 25.
REAL_COLUMNS = ['DEPTH','VSH','VSAND','VCARB','POR','SW','PD','VP','VS','RHO']
REAL_NAMES = ('BT2','SR','F1B','19A')


def load_real_well(path):
    path = Path(path)
    if path.suffix.lower() != '.npy':
        raise ValueError('Select a .npy real-well file.')
    raw = np.load(path, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != len(REAL_COLUMNS) or len(raw) == 0:
        raise ValueError(f'Expected a nonempty numeric array with 10 columns; got {raw.shape}.')
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError('Well array must be numeric.')
    frame = pd.DataFrame(raw, columns=REAL_COLUMNS)
    required = FEATURES + TARGETS
    valid = np.isfinite(frame[required].to_numpy()).all(axis=1)
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError('No rows have finite required input and target values.')
    # No clipping or normalization: preserve observed measurements and surface anomalies.
    frame['WELL'] = path.stem
    return frame, invalid_count


def evaluate_real_well(models, frame):
    if not models:
        raise ValueError('Train on synthetic wells before evaluating real wells.')
    predictions = PredictionController.predict(models, frame)
    metrics = EvaluationController.evaluate(predictions)
    return predictions, metrics


def real_well_diagnostics(frame):
    issues = []
    fracs = frame[['VSH','VSAND','VCARB']].sum(axis=1)
    bad = int((~fracs.between(0.95,1.05)).sum())
    if bad: issues.append(f'{bad} rows have mineral fractions summing outside 0.95–1.05.')
    for col in ['VSH','VSAND','VCARB','POR','SW']:
        bad = int((~frame[col].between(0,1)).sum())
        if bad: issues.append(f'{bad} rows have {col} outside [0, 1].')
    if not frame['DEPTH'].is_monotonic_increasing:
        issues.append('Depth is not monotonically increasing; check plot ordering.')
    return issues
