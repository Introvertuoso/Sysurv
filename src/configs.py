from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 10


@dataclass
class SoftRuleConfig(Config):
    alpha: float = 0.0
    temperature: float = 0.2
    bin_deviation: float = 0.2
    use_weights: bool = True
    classifier_epochs: int = 1000  # 2000
    classifier_lr: float = 1e-2  # 2e-3
    classifier_batch_size: int = -1
    optimize_temperature: bool = True
    e_steps: int = 0
    m_steps: int = 1


@dataclass
class SySurvConfig(SoftRuleConfig):
    contrastive: bool = False
    lambd: float = 2.0
