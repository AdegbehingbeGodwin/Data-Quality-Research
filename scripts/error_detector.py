"""
Error detection module for MT quality analysis
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import logging

from utils import (
    similarity, 
    detect_repetition, 
    compute_avg_logprob, 
    extract_terms,
    gradient_x_input
)
from config import ErrorTypes, experiment_config

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Data class for storing error detection results"""
    doc_id: int
    source: str
    reference: str
    hypothesis: str
    avg_logprob: float
    error_types: List[str]
    tokens: List[str]
    gradient_attribution: List[float]
    cross_attention: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class MTErrorDetector:
    """
    Error detector for Machine Translation quality issues
    """
    
    def __init__(
        self,
        omission_threshold: float = None,
        low_conf_threshold: float = None,
        high_conf_error_sim: float = None,
        terminology_drift_threshold: float = None,
        min_term_length: int = None
    ):
        """
        Initialize error detector with thresholds
        
        Args:
            omission_threshold: Length ratio threshold for omission detection
            low_conf_threshold: Log probability threshold for low confidence
            high_conf_error_sim: Similarity threshold for high confidence errors
            terminology_drift_threshold: Inconsistency threshold for terminology
            min_term_length: Minimum term length for terminology tracking
        """
        # Use config defaults if not provided
        config = experiment_config
        self.omission_threshold = omission_threshold or config.omission_threshold
        self.low_conf_threshold = low_conf_threshold or config.low_confidence_threshold
        self.high_conf_error_sim = high_conf_error_sim or config.high_conf_error_similarity
        self.terminology_drift_threshold = terminology_drift_threshold or config.terminology_drift_threshold
        self.min_term_length = min_term_length or config.min_term_length
        
        # Document-level memory for terminology tracking
        self.doc_term_memory = defaultdict(set)
        
        logger.info(f"Initialized MTErrorDetector with thresholds: "
                   f"omission={self.omission_threshold}, "
                   f"low_conf={self.low_conf_threshold}, "
                   f"high_conf_sim={self.high_conf_error_sim}")
    
    def detect_errors(
        self,
        source: str,
        reference: str,
        hypothesis: str,
        avg_logprob: float
    ) -> List[str]:
        """
        Detect all error types in a translation
        
        Args:
            source: Source text
            reference: Reference translation
            hypothesis: Hypothesis (generated) translation
            avg_logprob: Average log probability of generation
            
        Returns:
            List of detected error types
        """
        errors = []
        
        # 1. OMISSION: hypothesis significantly shorter than reference
        if self._detect_omission(hypothesis, reference):
            errors.append(ErrorTypes.OMISSION)
        
        # 2. REPETITION: duplicate tokens in hypothesis
        if self._detect_repetition(hypothesis):
            errors.append(ErrorTypes.REPETITION)
        
        # 3. LOW_CONFIDENCE: model uncertain about generation
        if self._detect_low_confidence(avg_logprob):
            errors.append(ErrorTypes.LOW_CONFIDENCE)
        
        # 4. HIGH_CONFIDENCE_ERROR: confident but wrong
        elif self._detect_high_confidence_error(hypothesis, reference, avg_logprob):
            errors.append(ErrorTypes.HIGH_CONFIDENCE_ERROR)
        
        # 5. TERMINOLOGY_DRIFT: inconsistent term translation
        if self._detect_terminology_drift(source, hypothesis):
            errors.append(ErrorTypes.TERMINOLOGY_DRIFT)
        
        return errors
    
    def _detect_omission(self, hypothesis: str, reference: str) -> bool:
        """Detect if hypothesis is significantly shorter than reference"""
        hyp_len = len(hypothesis.split())
        ref_len = len(reference.split())
        return hyp_len < self.omission_threshold * ref_len if ref_len > 0 else False
    
    def _detect_repetition(self, hypothesis: str) -> bool:
        """Detect repetition in hypothesis"""
        return detect_repetition(hypothesis)
    
    def _detect_low_confidence(self, avg_logprob: float) -> bool:
        """Detect low model confidence"""
        return avg_logprob < self.low_conf_threshold
    
    def _detect_high_confidence_error(
        self, 
        hypothesis: str, 
        reference: str, 
        avg_logprob: float
    ) -> bool:
        """Detect high confidence but poor quality translation"""
        is_high_confidence = avg_logprob >= self.low_conf_threshold
        is_low_similarity = similarity(hypothesis, reference) < self.high_conf_error_sim
        return is_high_confidence and is_low_similarity
    
    def _detect_terminology_drift(self, source: str, hypothesis: str) -> bool:
        """
        Detect inconsistent terminology translation within document
        
        Tracks whether same source terms are translated consistently
        """
        src_terms = extract_terms(source, self.min_term_length)
        tgt_terms = extract_terms(hypothesis, self.min_term_length)
        
        drift_detected = False
        for term in src_terms:
            if term in self.doc_term_memory and tgt_terms:
                # Check if translation is inconsistent with previous occurrences
                if self.doc_term_memory[term] != tgt_terms:
                    drift_detected = True
            # Update memory
            self.doc_term_memory[term] |= tgt_terms
        
        return drift_detected
    
    def reset_document_memory(self) -> None:
        """Reset document-level terminology memory (call between documents)"""
        self.doc_term_memory.clear()
        logger.debug("Reset document-level terminology memory")


def analyze_translation(
    model,
    tokenizer,
    source: str,
    reference: str,
    source_lang_code: str,
    target_lang_code: str,
    max_length: int,
    device: str,
    error_detector: MTErrorDetector,
    doc_id: int
) -> Tuple[ErrorCase, Dict[str, int]]:
    """
    Analyze a single translation and detect errors
    
    Args:
        model: Seq2Seq translation model
        tokenizer: Model tokenizer
        source: Source text
        reference: Reference translation
        source_lang_code: Source language code for NLLB
        target_lang_code: Target language code for NLLB
        max_length: Maximum sequence length
        device: Device (cuda/cpu)
        error_detector: Error detector instance
        doc_id: Document ID
        
    Returns:
        Tuple of (ErrorCase if errors found else None, error summary dict)
    """
    # Tokenize source
    tokenizer.src_lang = source_lang_code
    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(f"__{target_lang_code}__"),
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True
        )
    
    # Decode hypothesis
    hypothesis = tokenizer.decode(
        generation.sequences[0],
        skip_special_tokens=True
    )
    
    # Compute average log probability
    avg_logprob = compute_avg_logprob(generation.scores, generation.sequences)
    
    # Detect errors
    errors = error_detector.detect_errors(source, reference, hypothesis, avg_logprob)
    
    if not errors:
        return None, {}
    
    # Extract explainability features
    # Cross-attention (decoder -> encoder)
    cross_attn_per_head = torch.stack(generation.cross_attentions[-1], dim=1)
    cross_attn = cross_attn_per_head.mean(dim=1).mean(dim=1)
    cross_attn = cross_attn.detach().cpu().numpy()[0]
    
    # Gradient attribution
    grad_attr = gradient_x_input(model, inputs, generation.sequences)
    
    # Source tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Create error case
    error_case = ErrorCase(
        doc_id=doc_id,
        source=source,
        reference=reference,
        hypothesis=hypothesis,
        avg_logprob=avg_logprob,
        error_types=errors,
        tokens=tokens,
        gradient_attribution=grad_attr.tolist(),
        cross_attention=cross_attn.tolist()
    )
    
    # Update summary
    summary = defaultdict(int)
    for error in errors:
        summary[error] += 1
    summary["TOTAL_ERRORS"] += 1
    
    return error_case, dict(summary)
