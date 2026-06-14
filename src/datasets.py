import numpy as np


class SurvivalDataset:
    """
    A container for survival analysis datasets.
    """
    def __init__(
        self,
        name: str,
        features: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: list[str]
    ):
        self.name = name
        self.features = features
        self.time = time
        self.event = event
        self.feature_names = feature_names
        self.censorship_ratio: float = (1 - self.event.sum() / self.event.shape[0])
        self.time_domain = (self.time.min(), self.time.max())
        self.n_samples = features.shape[0]
        self.n_features = features.shape[1]

    def __call__(self):
        return {'name': self.name,
                'n_samples': self.n_samples,
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'time_domain': self.time_domain,
                'censorship_ratio': self.censorship_ratio}