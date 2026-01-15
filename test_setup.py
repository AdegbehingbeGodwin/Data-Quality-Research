"""
Quick start script to verify installation and test the pipeline
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_installation():
    """Verify all required packages are installed"""
    logger.info("Checking installation...")
    
    try:
        import numpy
        import pandas
        import datasets
        import captum
        import scipy
        import sklearn
        import tqdm
        logger.info("✓ All packages installed successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing package: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def check_gpu():
    """Check GPU availability"""
    logger.info("Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ GPU available: {gpu_name}")
        logger.info(f"  Memory: {gpu_memory:.1f} GB")
        return True
    else:
        logger.warning("✗ No GPU available, will use CPU (slower)")
        return False

def test_model_loading():
    """Test loading the NLLB model"""
    logger.info("Testing model loading (this may take a few minutes)...")
    
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        
        logger.info(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Loading model from {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        
        logger.info(f"✓ Model loaded successfully on {device}")
        
        # Test translation
        logger.info("Testing a simple translation...")
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("__swh_Latn__"),
                max_length=20
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ Translation test successful!")
        logger.info(f"  Input: Hello, how are you?")
        logger.info(f"  Output (Swahili): {translation}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def test_dataset_access():
    """Test dataset loading"""
    logger.info("Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        logger.info("Attempting to load AfriDocMT dataset (sentence-level)...")
        dataset = load_dataset("masakhane/AfriDocMT", "health", split="test")
        
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  Samples: {len(dataset)}")
        logger.info(f"  Columns: {dataset.column_names}")
        
        # Show sample
        sample = dataset[0]
        logger.info(f"\nSample translation:")
        logger.info(f"  EN: {sample['en'][:100]}...")
        logger.info(f"  SW: {sample['sw'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        logger.info("  This is normal if you haven't downloaded the dataset yet")
        logger.info("  The dataset will be downloaded automatically when you run experiments")
        return False

def run_mini_test():
    """Run a minimal test of the error detection pipeline"""
    logger.info("\nRunning mini error detection test...")
    
    try:
        from error_detector import MTErrorDetector
        from utils import detect_repetition, similarity, compute_avg_logprob
        
        # Test error detector
        detector = MTErrorDetector()
        
        # Create test cases
        test_cases = [
            {
                "source": "The patient has a fever and headache.",
                "reference": "Mgonjwa ana homa na maumivu ya kichwa.",
                "hypothesis": "Mgonjwa ana homa.",  # Omission
                "logprob": -3.0
            },
            {
                "source": "Take this medicine twice a day.",
                "reference": "Chukua dawa hii mara mbili kwa siku.",
                "hypothesis": "Chukua chukua dawa hii.",  # Repetition
                "logprob": -2.5
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            errors = detector.detect_errors(
                test["source"],
                test["reference"],
                test["hypothesis"],
                test["logprob"]
            )
            logger.info(f"\nTest case {i}:")
            logger.info(f"  Errors detected: {errors}")
        
        logger.info("\n✓ Error detection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error detection test failed: {e}")
        return False

def main():
    """Main verification routine"""
    logger.info("=" * 60)
    logger.info("DATA QUALITY RESEARCH - INSTALLATION VERIFICATION")
    logger.info("=" * 60)
    
    results = {
        "Installation": check_installation(),
        "GPU": check_gpu(),
        "Model Loading": test_model_loading(),
        "Dataset Access": test_dataset_access(),
        "Error Detection": run_mini_test()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test:.<40} {status}")
    
    all_critical_passed = results["Installation"] and results["Model Loading"] and results["Error Detection"]
    
    if all_critical_passed:
        logger.info("\n✓ All critical tests passed! You're ready to run experiments.")
        logger.info("\nQuick start commands:")
        logger.info("  python run_experiments.py --sentence-only  # Use local data")
        logger.info("  python run_experiments.py --config health --lang sw  # Single experiment")
    else:
        logger.warning("\n✗ Some critical tests failed. Please fix the issues above.")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Install dependencies: pip install -r requirements.txt")
        logger.info("  2. Check GPU drivers if GPU test failed")
        logger.info("  3. Ensure internet connection for model/dataset downloads")

if __name__ == "__main__":
    main()
