"""
SQL Injection Detection Research Module
Comprehensive experimental framework for thesis validation
"""

__version__ = "1.0.0"
__author__ = "Areesha Ahmed"
__email__ = "k237836@nu.edu.pk"

from .experiments.adversarial_robustness import AdversarialRobustnessTester
from .experiments.multi_dataset_testing import MultiDatasetEvaluator
from .experiments.explainability_analysis import ModelExplainability
from .experiments.concept_drift import ConceptDriftSimulator
from .experiments.federated_simulation import FederatedLearningSimulator

__all__ = [
    'AdversarialRobustnessTester',
    'MultiDatasetEvaluator',
    'ModelExplainability',
    'ConceptDriftSimulator',
    'FederatedLearningSimulator'
]