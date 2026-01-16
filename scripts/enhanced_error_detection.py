#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Error Detection Methods for Machine Translation Quality

Implements detailed sentence-level data quality issue categorization:

1. GENERATION ERRORS:
   - Under-generation: Target significantly shorter than source (< 0.6 ratio)
   - Over-generation: Target significantly longer than source (> 1.5 ratio)
   - Repetition: Repeated words/phrases within output
   - Off-target translation: Output in wrong language

2. LINGUISTIC AND SCRIPT ISSUES:
   - Orthographic inconsistencies: Spelling/script variations (e.g., diacritics)
   - Script-specific challenges: Ge'ez (Amharic), Latin script issues
   - Translation artifacts: Literal translation vs semantic meaning

3. CONTENT AND COHESION ERRORS:
   - Content errors: Direct mistakes/inaccuracies
   - Grammatical errors: Syntax/structure problems
   - Lexical cohesion errors: Inconsistent terminology

4. DATA PREPARATION NOISE:
   - Segmentation errors: Incorrect sentence boundaries
   - Annotation inconsistencies: Contradictory patterns

Reference:
   AfriDocMT: A Large-Scale Machine Translation Corpus for African Languages
   https://github.com/masakhane-io/AfriDocMT
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re


class DetailedQualityAnalyzer:
    """Comprehensive data quality analysis with granular error categorization"""
    
    def __init__(self):
        """Initialize analyzer with language-specific rules"""
        self.yoruba_diacritics = set('àáãèéẹìíĩòóộụùúũỹ')
        self.script_patterns = {
            'amharic_geez': r'[\u1200-\u137F]',  # Ge'ez script
            'latin': r'[a-zA-Z]',
            'arabic': r'[\u0600-\u06FF]',
        }
    
    # ========================================================================
    # GENERATION ERRORS
    # ========================================================================
    
    def detect_under_generation(self, dataset: List[Dict]) -> List[Dict]:
        """Detect under-generation (omission) errors"""
        errors = []
        
        for idx, example in enumerate(dataset):
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            source_len = len(source.split())
            target_len = len(target.split())
            
            if source_len > 0:
                ratio = target_len / source_len
                # Under-generation threshold: less than 60% of source length
                if ratio < 0.6:
                    errors.append({
                        "error_type": "under_generation",
                        "example_id": example.get("example_id", idx),
                        "split": example.get("split", "unknown"),
                        "source_words": source_len,
                        "target_words": target_len,
                        "ratio": ratio,
                        "source": source[:80],
                        "target": target[:80],
                        "severity": "high" if ratio < 0.4 else "medium"
                    })
        
        return errors
    
    def detect_over_generation(self, dataset: List[Dict]) -> List[Dict]:
        """Detect over-generation (excessive output) errors"""
        errors = []
        
        for idx, example in enumerate(dataset):
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            source_len = len(source.split())
            target_len = len(target.split())
            
            if source_len > 0:
                ratio = target_len / source_len
                # Over-generation threshold: more than 150% of source length
                if ratio > 1.5:
                    errors.append({
                        "error_type": "over_generation",
                        "example_id": example.get("example_id", idx),
                        "split": example.get("split", "unknown"),
                        "source_words": source_len,
                        "target_words": target_len,
                        "ratio": ratio,
                        "source": source[:80],
                        "target": target[:80],
                        "severity": "high" if ratio > 2.5 else "medium"
                    })
        
        return errors
    
    def detect_repetition(self, dataset: List[Dict]) -> List[Dict]:
        """Detect repetition errors (repeated words/phrases)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            target = example.get("target", "")
            
            if not target:
                continue
            
            words = target.split()
            
            # Check for word repetitions
            word_counts = Counter(words)
            repeated_words = {word: count for word, count in word_counts.items() 
                            if count > 2 and len(word) > 3}  # Multi-word and >3 chars
            
            if repeated_words:
                errors.append({
                    "error_type": "repetition",
                    "example_id": example.get("example_id", idx),
                    "split": example.get("split", "unknown"),
                    "target": target[:80],
                    "repeated_words": repeated_words,
                    "total_words": len(words),
                    "repetition_ratio": sum(repeated_words.values()) / len(words) if words else 0
                })
        
        return errors
    
    def detect_off_target_language(self, dataset: List[Dict], target_lang: str = "swh") -> List[Dict]:
        """Detect off-target translations (wrong language output)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            target = example.get("target", "")
            source = example.get("source", "")
            
            if not target:
                continue
            
            # Count script distributions
            english_words = sum(1 for w in target.split() 
                              if all(ord(c) < 128 for c in w) and len(w) > 1)
            total_words = len(target.split())
            
            english_ratio = english_words / total_words if total_words > 0 else 0
            
            # Flag if predominantly English (>60% of content)
            if english_ratio > 0.6:
                errors.append({
                    "error_type": "off_target_language",
                    "example_id": example.get("example_id", idx),
                    "split": example.get("split", "unknown"),
                    "english_ratio": english_ratio,
                    "english_words": english_words,
                    "total_words": total_words,
                    "target": target[:80],
                    "severity": "critical"
                })
        
        return errors
    
    # ========================================================================
    # LINGUISTIC AND SCRIPT ISSUES
    # ========================================================================
    
    def detect_orthographic_inconsistencies(self, dataset: List[Dict]) -> List[Dict]:
        """Detect orthographic inconsistencies (spelling, diacritics)"""
        errors = []
        term_spellings = defaultdict(set)
        
        # Collect variations
        for example in dataset:
            target = example.get("target", "")
            if not target:
                continue
            
            words = target.split()
            for word in words:
                # Normalize and track variations
                base_word = word.lower()
                term_spellings[base_word.replace('à', 'a').replace('á', 'a')
                             .replace('è', 'e').replace('é', 'e')
                             .replace('ì', 'i').replace('í', 'i')
                             .replace('ò', 'o').replace('ó', 'o')
                             .replace('ù', 'u').replace('ú', 'u')].add(base_word)
        
        # Find inconsistencies (multiple spellings of same word)
        for base, variations in term_spellings.items():
            if len(variations) > 1:
                errors.append({
                    "error_type": "orthographic_inconsistency",
                    "base_term": base,
                    "variations": list(variations),
                    "variation_count": len(variations),
                    "issue": "Inconsistent spelling or diacritics"
                })
        
        return errors
    
    def detect_script_issues(self, dataset: List[Dict]) -> List[Dict]:
        """Detect script-specific challenges (Ge'ez, Latin mix, etc.)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            target = example.get("target", "")
            
            if not target:
                continue
            
            # Check for script mixing
            has_geez = bool(re.search(self.script_patterns['amharic_geez'], target))
            has_latin = bool(re.search(self.script_patterns['latin'], target))
            has_arabic = bool(re.search(self.script_patterns['arabic'], target))
            
            scripts_used = sum([has_geez, has_latin, has_arabic])
            
            # Flag if mixing scripts (shouldn't have multiple scripts)
            if scripts_used > 1:
                errors.append({
                    "error_type": "script_mixing",
                    "example_id": example.get("example_id", idx),
                    "split": example.get("split", "unknown"),
                    "scripts": {
                        "geez": has_geez,
                        "latin": has_latin,
                        "arabic": has_arabic
                    },
                    "target": target[:80],
                    "severity": "high"
                })
        
        return errors
    
    def detect_translation_artifacts(self, dataset: List[Dict]) -> List[Dict]:
        """Detect translation artifacts (literal vs semantic meaning)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            source_words = set(source.lower().split())
            target_words = set(target.lower().split())
            
            # Check for direct word copying (literal translation indicator)
            common_words = source_words & target_words
            
            # High overlap indicates word-for-word/literal translation
            if len(source_words) > 5 and len(common_words) / len(source_words) > 0.4:
                errors.append({
                    "error_type": "translation_artifact",
                    "example_id": example.get("example_id", idx),
                    "split": example.get("split", "unknown"),
                    "word_overlap_ratio": len(common_words) / len(source_words),
                    "common_words": len(common_words),
                    "severity": "medium"
                })
        
        return errors
    
    # ========================================================================
    # CONTENT AND COHESION ERRORS
    # ========================================================================
    
    def detect_content_errors(self, dataset: List[Dict]) -> List[Dict]:
        """Detect content errors (missing/incorrect information)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            source_len = len(source.split())
            target_len = len(target.split())
            
            # Content errors indicated by extreme length mismatch
            if source_len > 0:
                ratio = target_len / source_len
                
                # Very unbalanced translations likely have content errors
                if ratio < 0.4 or ratio > 2.0:
                    errors.append({
                        "error_type": "content_error",
                        "example_id": example.get("example_id", idx),
                        "split": example.get("split", "unknown"),
                        "length_ratio": ratio,
                        "issue": "Potential missing or extra information",
                        "source_words": source_len,
                        "target_words": target_len
                    })
        
        return errors
    
    def detect_grammatical_errors(self, dataset: List[Dict]) -> List[Dict]:
        """Detect grammatical and syntax errors"""
        errors = []
        
        for idx, example in enumerate(dataset):
            target = example.get("target", "")
            
            if not target:
                continue
            
            words = target.split()
            
            # Check for sentence structure issues
            # Flag if no clear sentence structure (all caps, special chars)
            upper_count = sum(1 for w in words if w.isupper())
            special_char_count = sum(1 for w in words if not any(c.isalnum() for c in w))
            
            if len(words) > 3:
                upper_ratio = upper_count / len(words)
                special_ratio = special_char_count / len(words)
                
                if upper_ratio > 0.5 or special_ratio > 0.3:
                    errors.append({
                        "error_type": "grammatical_error",
                        "example_id": example.get("example_id", idx),
                        "split": example.get("split", "unknown"),
                        "upper_ratio": upper_ratio,
                        "special_char_ratio": special_ratio,
                        "target": target[:80],
                        "issue": "Suspicious structure (caps/special chars)"
                    })
        
        return errors
    
    def detect_lexical_cohesion_errors(self, dataset: List[Dict]) -> List[Dict]:
        """Detect inconsistent terminology (lexical cohesion issues)"""
        errors = []
        term_translations = defaultdict(set)
        term_frequency = defaultdict(int)
        
        # Collect term translations across dataset
        for example in dataset:
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            source_words = source.split()
            target_words = target.split()
            
            for s_word in source_words:
                if len(s_word) > 4:  # Content words
                    term_frequency[s_word] += 1
                    if target_words:
                        # Simple heuristic: first content word in target
                        term_translations[s_word].add(target_words[0])
        
        # Find inconsistencies
        for term, translations in term_translations.items():
            if term_frequency[term] >= 3 and len(translations) >= 2:
                errors.append({
                    "error_type": "lexical_cohesion_error",
                    "term": term,
                    "translation_variants": list(translations),
                    "variant_count": len(translations),
                    "frequency": term_frequency[term],
                    "severity": "high" if term_frequency[term] > 5 else "medium"
                })
        
        return errors
    
    # ========================================================================
    # DATA PREPARATION NOISE
    # ========================================================================
    
    def detect_segmentation_errors(self, dataset: List[Dict]) -> List[Dict]:
        """Detect segmentation errors (incorrect sentence boundaries)"""
        errors = []
        
        for idx, example in enumerate(dataset):
            source = example.get("source", "")
            target = example.get("target", "")
            
            if not source or not target:
                continue
            
            # Check for incomplete sentences
            source_ends_with_period = source.rstrip().endswith('.')
            target_ends_with_period = target.rstrip().endswith('.')
            
            # Mismatched sentence boundaries
            if source_ends_with_period != target_ends_with_period:
                errors.append({
                    "error_type": "segmentation_error",
                    "example_id": example.get("example_id", idx),
                    "split": example.get("split", "unknown"),
                    "source_complete": source_ends_with_period,
                    "target_complete": target_ends_with_period,
                    "issue": "Sentence boundary mismatch",
                    "source": source[:80],
                    "target": target[:80]
                })
        
        return errors
    
    def detect_annotation_inconsistencies(self, dataset: List[Dict]) -> List[Dict]:
        """Detect annotation inconsistencies in data"""
        errors = []
        pair_patterns = defaultdict(list)
        
        # Look for contradictory patterns
        for idx, example in enumerate(dataset):
            source = example.get("source", "").lower()
            target = example.get("target", "").lower()
            
            if not source or not target:
                continue
            
            # Create pattern signature
            pattern = (source[:20], len(source), len(target))  # First 20 chars + lengths
            pair_patterns[pattern].append((target, idx))
        
        # Find inconsistencies
        for pattern, translations in pair_patterns.items():
            if len(translations) > 2:
                unique_targets = set(t[0] for t in translations)
                if len(unique_targets) > 1:
                    errors.append({
                        "error_type": "annotation_inconsistency",
                        "pattern": pattern[0],
                        "variants": len(unique_targets),
                        "examples": len(translations),
                        "issue": f"Same/similar source translated differently {len(unique_targets)} ways"
                    })
        
        return errors
    
    def analyze_all_issues(self, dataset: List[Dict]) -> Dict:
        """Run comprehensive analysis of all error categories"""
        
        print("\n" + "="*80)
        print("DETAILED SENTENCE-LEVEL DATA QUALITY ANALYSIS")
        print("="*80)
        
        results = {}
        
        # Generation Errors
        print("\n[1/4] Analyzing Generation Errors...")
        results["generation_errors"] = {
            "under_generation": self.detect_under_generation(dataset),
            "over_generation": self.detect_over_generation(dataset),
            "repetition": self.detect_repetition(dataset),
            "off_target_language": self.detect_off_target_language(dataset)
        }
        
        # Linguistic Issues
        print("[2/4] Analyzing Linguistic and Script Issues...")
        results["linguistic_issues"] = {
            "orthographic_inconsistencies": self.detect_orthographic_inconsistencies(dataset),
            "script_mixing": self.detect_script_issues(dataset),
            "translation_artifacts": self.detect_translation_artifacts(dataset)
        }
        
        # Content and Cohesion
        print("[3/4] Analyzing Content and Cohesion Errors...")
        results["cohesion_errors"] = {
            "content_errors": self.detect_content_errors(dataset),
            "grammatical_errors": self.detect_grammatical_errors(dataset),
            "lexical_cohesion_errors": self.detect_lexical_cohesion_errors(dataset)
        }
        
        # Data Preparation Noise
        print("[4/4] Analyzing Data Preparation Noise...")
        results["data_noise"] = {
            "segmentation_errors": self.detect_segmentation_errors(dataset),
            "annotation_inconsistencies": self.detect_annotation_inconsistencies(dataset)
        }
        
        # Generate summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        summary = self._generate_summary(results)
        
        return {
            "detailed_errors": results,
            "summary": summary
        }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics"""
        summary = {}
        
        for category, subcategories in results.items():
            summary[category] = {}
            for error_type, errors in subcategories.items():
                count = len(errors)
                summary[category][error_type] = count
                print(f"  {error_type}: {count} issues")
        
        return summary


def categorize_errors_comprehensively(dataset: List[Dict]) -> Dict:
    """Main entry point for comprehensive error analysis"""
    analyzer = DetailedQualityAnalyzer()
    return analyzer.analyze_all_issues(dataset)


if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        {"source": "Hello world this is a test", "target": "Habari dunia hii", "split": "test"},
        {"source": "This is a longer source text", "target": "Hii", "split": "test"},  # Under-generation
        {"source": "Short", "target": "Hii ndiyo majibu ya kila kitu kwa sababu", "split": "test"},  # Over-generation
    ]
    
    results = categorize_errors_comprehensively(sample_data)
    print("\nDetailed analysis complete!")

