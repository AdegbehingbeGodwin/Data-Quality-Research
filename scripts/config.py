"""
Configuration file for Data Quality Research experiments
"""

import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 512
    device: str = "cuda"  # Will auto-fallback to cpu if cuda unavailable


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "masakhane/AfriDocMT"
    
    # Available configurations
    SENTENCE_CONFIGS: List[str] = None
    DOC_CONFIGS: List[str] = None
    
    def __post_init__(self):
        self.SENTENCE_CONFIGS = ["tech", "health"]
        self.DOC_CONFIGS = [
            "doc_tech", "doc_health",
            "doc_tech_5", "doc_health_5",
            "doc_tech_10", "doc_health_10",
            "doc_tech_25", "doc_health_25"
        ]
    
    # Language mappings: dataset code -> NLLB code
    LANG_MAP: Dict[str, str] = None
    
    def __post_init__(self):
        if self.SENTENCE_CONFIGS is None:
            self.SENTENCE_CONFIGS = ["tech", "health"]
        if self.DOC_CONFIGS is None:
            self.DOC_CONFIGS = [
                "doc_tech", "doc_health",
                "doc_tech_5", "doc_health_5",
                "doc_tech_10", "doc_health_10",
                "doc_tech_25", "doc_health_25"
            ]
        if self.LANG_MAP is None:
            self.LANG_MAP = {
                "am": "amh_Ethi",  # Amharic
                "ha": "hau_Latn",  # Hausa
                "sw": "swh_Latn",  # Swahili
                "yo": "yor_Latn",  # Yoruba
                "zu": "zul_Latn"   # Zulu
            }
    
    source_lang_dataset: str = "en"
    source_lang_nllb: str = "eng_Latn"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    max_samples: int = 500
    output_root: str = "results_complete"
    save_intermediate: bool = True
    verbose: bool = True
    
    # Error detection thresholds
    omission_threshold: float = 0.7  # hypothesis < 70% of reference length
    low_confidence_threshold: float = -6.0  # avg log probability
    high_conf_error_similarity: float = 0.5  # similarity threshold
    terminology_drift_threshold: float = 0.2  # 20% inconsistency
    min_term_length: int = 4  # minimum term length for terminology tracking
    
    def __post_init__(self):
        os.makedirs(self.output_root, exist_ok=True)


# Error type taxonomy
class ErrorTypes:
    """MT-specific error taxonomy"""
    OMISSION = "OMISSION"
    REPETITION = "REPETITION"
    TERMINOLOGY_DRIFT = "TERMINOLOGY_DRIFT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_CONFIDENCE_ERROR = "HIGH_CONFIDENCE_ERROR"
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [
            cls.OMISSION,
            cls.REPETITION,
            cls.TERMINOLOGY_DRIFT,
            cls.LOW_CONFIDENCE,
            cls.HIGH_CONFIDENCE_ERROR
        ]


# Global configurations
model_config = ModelConfig()
dataset_config = DatasetConfig()
experiment_config = ExperimentConfig()
