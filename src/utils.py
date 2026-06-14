import pandas as pd
from src.datasets import SurvivalDataset

def load_case_study_data(
    features_path: str = 'data/case_study/nCounter_PostOp.csv',
    outcome_path: str = 'data/case_study/Outcome_PostOp.csv',
    dataset_name: str = 'mmc6_nCounter_PostOp'
) -> SurvivalDataset:
    """
    Loads the case study dataset from CSV files into a list of SurvivalDataset instances.
    """
    features = pd.read_csv(features_path, skiprows=2).drop(['ID'], axis=1)
    outcome = pd.read_csv(outcome_path).drop(['ID'], axis=1)

    mask = features.notna().all(axis=1)
    
    return SurvivalDataset(
        name=dataset_name, 
        features=features[mask].values, 
        time=outcome['Time to loco-regional failure (months)'][mask].values, 
        event=outcome['Loco-regional failure'][mask].astype(bool).values, 
        feature_names=features.columns.tolist()
    )