"""
Main script to run data quality experiments on AfriDocMT dataset
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from collections import defaultdict
import logging
from typing import List, Dict

from config import (
    ModelConfig, 
    DatasetConfig, 
    ExperimentConfig,
    model_config,
    dataset_config,
    experiment_config
)
from error_detector import MTErrorDetector, analyze_translation
from utils import setup_logging

logger = logging.getLogger(__name__)


class AfriDocMTExperiment:
    """
    Main experiment runner for AfriDocMT data quality analysis
    """
    
    def __init__(
        self,
        model_cfg: ModelConfig = None,
        dataset_cfg: DatasetConfig = None,
        experiment_cfg: ExperimentConfig = None
    ):
        """
        Initialize experiment runner
        
        Args:
            model_cfg: Model configuration
            dataset_cfg: Dataset configuration
            experiment_cfg: Experiment configuration
        """
        self.model_cfg = model_cfg or model_config
        self.dataset_cfg = dataset_cfg or dataset_config
        self.experiment_cfg = experiment_cfg or experiment_config
        
        # Setup logging
        log_file = os.path.join(
            self.experiment_cfg.output_root, 
            "experiment.log"
        )
        setup_logging(
            verbose=self.experiment_cfg.verbose, 
            log_file=log_file
        )
        
        # Setup device
        self.device = self.model_cfg.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_cfg.name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_cfg.name,
            output_attentions=True,
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def run_single_config(
        self,
        config_name: str,
        target_lang: str,
        split: str = "test"
    ) -> Dict:
        """
        Run experiment on a single configuration and language
        
        Args:
            config_name: Dataset configuration name (e.g., "health", "doc_health_5")
            target_lang: Target language code (dataset format, e.g., "sw")
            split: Dataset split to use (train/dev/test)
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info(f"Running experiment: config={config_name}, lang={target_lang}")
        
        # Get NLLB target language code
        if target_lang not in self.dataset_cfg.LANG_MAP:
            raise ValueError(f"Unsupported language: {target_lang}")
        
        nllb_target_lang = self.dataset_cfg.LANG_MAP[target_lang]
        
        # Create output directory
        output_dir = os.path.join(
            self.experiment_cfg.output_root,
            config_name,
            target_lang
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        try:
            dataset = load_dataset(
                self.dataset_cfg.name, 
                config_name
            )[split]
        except Exception as e:
            logger.error(f"Failed to load dataset config '{config_name}': {e}")
            logger.info("Make sure you have downloaded the dataset from HuggingFace")
            raise
        
        # Initialize error detector
        error_detector = MTErrorDetector()
        
        # Process samples
        case_studies = []
        summary = defaultdict(int)
        
        max_samples = min(len(dataset), self.experiment_cfg.max_samples)
        
        for idx in tqdm(range(max_samples), desc=f"{config_name}/{target_lang}"):
            example = dataset[idx]
            
            source = example[self.dataset_cfg.source_lang_dataset]
            reference = example[target_lang]
            
            # Analyze translation
            error_case, err_summary = analyze_translation(
                model=self.model,
                tokenizer=self.tokenizer,
                source=source,
                reference=reference,
                source_lang_code=self.dataset_cfg.source_lang_nllb,
                target_lang_code=nllb_target_lang,
                max_length=self.model_cfg.max_length,
                device=self.device,
                error_detector=error_detector,
                doc_id=idx
            )
            
            if error_case:
                case_studies.append(error_case.to_dict())
                for error_type, count in err_summary.items():
                    summary[error_type] += count
        
        # Save results
        case_studies_path = os.path.join(output_dir, "case_studies.json")
        with open(case_studies_path, "w", encoding="utf-8") as f:
            json.dump(case_studies, f, indent=2, ensure_ascii=False)
        
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dict(summary), f, indent=2)
        
        logger.info(f"Saved {len(case_studies)} error cases to {output_dir}")
        logger.info(f"Summary: {dict(summary)}")
        
        return dict(summary)
    
    def run_all_configs(
        self,
        configs: List[str] = None,
        languages: List[str] = None
    ) -> Dict:
        """
        Run experiments on all specified configurations and languages
        
        Args:
            configs: List of config names (defaults to all available)
            languages: List of language codes (defaults to all available)
            
        Returns:
            Nested dictionary with all summaries
        """
        configs = configs or (
            self.dataset_cfg.SENTENCE_CONFIGS + 
            self.dataset_cfg.DOC_CONFIGS
        )
        languages = languages or list(self.dataset_cfg.LANG_MAP.keys())
        
        logger.info(f"Running full experiment suite:")
        logger.info(f"  Configs: {configs}")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  Total combinations: {len(configs) * len(languages)}")
        
        all_summaries = {}
        
        for config in configs:
            all_summaries[config] = {}
            for lang in languages:
                try:
                    summary = self.run_single_config(config, lang)
                    all_summaries[config][lang] = summary
                except Exception as e:
                    logger.error(f"Failed on {config}/{lang}: {e}")
                    all_summaries[config][lang] = {"error": str(e)}
        
        # Save master summary
        master_summary_path = os.path.join(
            self.experiment_cfg.output_root,
            "master_summary.json"
        )
        with open(master_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)
        
        logger.info(f"Saved master summary to {master_summary_path}")
        logger.info("✅ EXPERIMENT SUITE COMPLETE")
        
        return all_summaries


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run AfriDocMT data quality experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Single config to run (e.g., 'health', 'doc_health_5')"
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Single language to run (e.g., 'sw', 'yo')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configs and languages"
    )
    parser.add_argument(
        "--sentence-only",
        action="store_true",
        help="Run only sentence-level configs (tech, health)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples per config/lang"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_complete",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Update experiment config
    experiment_config.max_samples = args.max_samples
    experiment_config.output_root = args.output_dir
    
    # Create experiment runner
    experiment = AfriDocMTExperiment()
    
    # Run experiments
    if args.config and args.lang:
        # Single config/lang pair
        experiment.run_single_config(args.config, args.lang)
    
    elif args.all:
        # All configs and languages
        experiment.run_all_configs()
    
    elif args.sentence_only:
        # Sentence-level only
        experiment.run_all_configs(
            configs=dataset_config.SENTENCE_CONFIGS
        )
    
    else:
        logger.error("Please specify either --config and --lang, --all, or --sentence-only")
        parser.print_help()


if __name__ == "__main__":
    main()
