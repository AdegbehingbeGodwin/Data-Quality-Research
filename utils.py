"""
Utility functions for data quality analysis
"""

import torch
import numpy as np
from difflib import SequenceMatcher
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


def similarity(text_a: str, text_b: str) -> float:
    """
    Compute similarity ratio between two strings
    
    Args:
        text_a: First text string
        text_b: Second text string
        
    Returns:
        Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, text_a, text_b).ratio()


def detect_repetition(text: str) -> bool:
    """
    Detect if text has repeated tokens
    
    Args:
        text: Text to check for repetition
        
    Returns:
        True if repetitions found, False otherwise
    """
    tokens = text.split()
    return len(tokens) > len(set(tokens))


def compute_avg_logprob(output_scores: List[torch.Tensor], sequences: torch.Tensor) -> float:
    """
    Compute average log probability for generated sequence
    
    Args:
        output_scores: List of score tensors from model.generate()
        sequences: Generated token sequences
        
    Returns:
        Average log probability across all tokens
    """
    logprobs = []
    for step_scores, token_id in zip(output_scores, sequences[0][1:]):
        logp = torch.log_softmax(step_scores, dim=-1)[0, token_id]
        logprobs.append(logp.item())
    
    return float(np.mean(logprobs)) if logprobs else -float('inf')


def extract_terms(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract terms (words) from text for terminology tracking
    
    Args:
        text: Input text
        min_length: Minimum word length to consider
        
    Returns:
        Set of lowercase terms meeting length requirement
    """
    return {word.lower() for word in text.split() if len(word) >= min_length}


def gradient_x_input(model, inputs: Dict[str, torch.Tensor], target_ids: torch.Tensor) -> np.ndarray:
    """
    Compute gradient × input attribution for source tokens
    
    Args:
        model: Seq2Seq model
        inputs: Tokenized inputs dict with input_ids and attention_mask
        target_ids: Target token IDs
        
    Returns:
        Attribution scores for each source token
    """
    # Get input embeddings
    embeddings = model.get_input_embeddings()(inputs["input_ids"])
    embeddings.requires_grad_(True)
    embeddings.retain_grad()
    
    # Forward pass
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=target_ids[:, :-1]
    )
    
    # Compute loss and backpropagate
    logits = outputs.logits
    loss = logits.max(dim=-1).values.mean()
    loss.backward()
    
    # Compute attributions
    grads = embeddings.grad
    attributions = (grads * embeddings).sum(dim=-1)
    attributions = attributions.detach().cpu().numpy()[0]
    
    return attributions


def setup_logging(verbose: bool = True, log_file: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        verbose: If True, set log level to INFO, else WARNING
        log_file: Optional log file path
    """
    level = logging.INFO if verbose else logging.WARNING
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
