"""
Generation algorithms for Φ-Mamba
Implements phase-locked generation with natural termination
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from math import pi, exp, log
from .encoding import TokenState, pentagon_reflection
from .utils import compute_berry_phase, is_phase_locked, PHI


def generate_with_phase_lock(
    model,
    context: List[TokenState],
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 10,
    repetition_penalty: float = 1.2
) -> List[TokenState]:
    """
    Generate tokens using phase-locked selection
    
    Key features:
    - Natural termination via energy decay
    - Pentagon reflection for non-locked paths
    - Retrocausal coherence checking
    
    Args:
        model: PhiLanguageModel instance
        context: Initial context (TokenStates)
        max_length: Maximum generation length
        temperature: Sampling temperature (0 = greedy)
        top_k: Consider top-k candidates
        repetition_penalty: Penalty for repeated tokens
        
    Returns:
        List of generated TokenStates
    """
    if not context:
        return []
        
    generated = []
    reflection_count = 0
    max_reflections = 5  # Natural termination after 5 bounces
    
    for step in range(max_length):
        # Get last state
        last_state = context[-1] if context else generated[-1]
        
        # Check energy threshold
        if last_state.energy < 0.01:
            print(f"Natural termination: energy = {last_state.energy:.6f}")
            break
            
        # Generate candidates
        candidates = _generate_candidates(
            model, 
            context + generated,
            repetition_penalty
        )
        
        if not candidates:
            break
            
        # Filter for phase-locked candidates
        locked_candidates = [
            c for c in candidates 
            if c['phase_locked']
        ]
        
        if locked_candidates:
            # We have phase-locked options
            selected = _sample_from_candidates(
                locked_candidates[:top_k],
                temperature
            )
            reflection_count = 0  # Reset reflection count
        else:
            # No phase-locked candidates - pentagon reflection
            print(f"No phase lock - pentagon reflection #{reflection_count + 1}")
            
            # Take best candidate and reflect
            best = candidates[0]
            selected = pentagon_reflection(best['state'])
            reflection_count += 1
            
            if reflection_count >= max_reflections:
                print("Max reflections reached - terminating")
                break
                
        # Add to generated sequence
        generated.append(selected)
        
        # Check for natural endpoints
        if selected.token in ['.', '!', '?']:
            print("Punctuation endpoint reached")
            break
            
    return generated


def _generate_candidates(
    model,
    context: List[TokenState],
    repetition_penalty: float
) -> List[Dict]:
    """
    Generate candidate next tokens with scores
    
    Returns list of dicts with:
    - state: TokenState
    - score: float
    - phase_locked: bool
    - berry_phase: float
    """
    if not context:
        return []
        
    last_state = context[-1]
    candidates = []
    
    # Track recent tokens for repetition penalty
    recent_tokens = [s.token for s in context[-10:]]
    
    # Consider each token in vocabulary
    for token_id in range(min(100, model.vocab_size)):  # Demo: limited vocab
        if token_id not in model.tokenizer.id_to_token:
            continue
            
        token = model.tokenizer.id_to_token[token_id]
        
        # Create candidate state
        candidate = TokenState(
            token=token,
            index=token_id,
            position=last_state.position + 1,
            vocab_size=model.vocab_size
        )
        
        # Calculate Berry phase
        berry_phase = compute_berry_phase(last_state, candidate)
        phase_locked = is_phase_locked(berry_phase)
        
        # Get coupling strength from model
        coupling = model.coupling_matrix[last_state.index, token_id]
        
        # Base score
        score = (
            (2.0 if phase_locked else 0.5) *  # Phase lock bonus
            (1.0 + coupling) *                 # Coupling strength
            candidate.energy                   # Energy level
        )
        
        # Apply repetition penalty
        if token in recent_tokens:
            count = recent_tokens.count(token)
            score /= (repetition_penalty ** count)
            
        candidates.append({
            'state': candidate,
            'score': score,
            'phase_locked': phase_locked,
            'berry_phase': berry_phase
        })
        
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates


def _sample_from_candidates(
    candidates: List[Dict],
    temperature: float
) -> TokenState:
    """
    Sample from candidates using temperature
    
    temperature = 0: greedy (always pick best)
    temperature > 0: sample from distribution
    """
    if not candidates:
        return None
        
    if temperature == 0:
        # Greedy selection
        return candidates[0]['state']
        
    # Extract scores and normalize with temperature
    scores = np.array([c['score'] for c in candidates])
    
    # Apply temperature and convert to probabilities
    if temperature != 1.0:
        scores = scores ** (1.0 / temperature)
        
    # Softmax
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    probs = exp_scores / exp_scores.sum()
    
    # Sample
    idx = np.random.choice(len(candidates), p=probs)
    
    return candidates[idx]['state']


def beam_search(
    model,
    context: List[TokenState],
    beam_width: int = 5,
    max_length: int = 50
) -> List[List[TokenState]]:
    """
    Beam search for finding multiple high-quality continuations
    
    Maintains beam_width parallel sequences and returns
    the top sequences by total phase coherence
    
    Args:
        model: PhiLanguageModel instance
        context: Initial context
        beam_width: Number of beams to maintain
        max_length: Maximum generation length
        
    Returns:
        List of beam_width sequences
    """
    # Initialize beams
    beams = [(context.copy(), 0.0)]  # (sequence, score)
    completed = []
    
    for step in range(max_length):
        new_beams = []
        
        for sequence, score in beams:
            if not sequence:
                continue
                
            # Check if this beam should terminate
            last_state = sequence[-1]
            if last_state.energy < 0.01 or last_state.token in ['.', '!', '?']:
                completed.append((sequence, score))
                continue
                
            # Generate candidates for this beam
            candidates = _generate_candidates(model, sequence, 1.0)
            
            # Consider top candidates
            for candidate in candidates[:beam_width]:
                new_sequence = sequence + [candidate['state']]
                new_score = score + log(candidate['score'] + 1e-10)
                new_beams.append((new_sequence, new_score))
                
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
        
        # Early stopping if all beams completed
        if not beams:
            break
            
    # Combine completed and active beams
    all_sequences = completed + beams
    all_sequences.sort(key=lambda x: x[1], reverse=True)
    
    # Return top sequences
    return [seq for seq, _ in all_sequences[:beam_width]]


def retrocausal_reranking(
    model,
    sequences: List[List[TokenState]]
) -> List[Tuple[List[TokenState], float]]:
    """
    Rerank sequences using retrocausal coherence
    
    This is unique to Φ-Mamba: we can score sequences
    by how well they satisfy retrocausal constraints
    
    Args:
        model: PhiLanguageModel instance
        sequences: List of token sequences to rerank
        
    Returns:
        List of (sequence, score) tuples sorted by score
    """
    scored_sequences = []
    
    for sequence in sequences:
        if len(sequence) < 2:
            scored_sequences.append((sequence, 0.0))
            continue
            
        # Calculate retrocausal coherence
        total_coherence = 0.0
        
        # Work backwards through sequence
        for i in range(len(sequence) - 1, 0, -1):
            future = sequence[i]
            past = sequence[i-1]
            
            # How well does future constrain past?
            berry_phase = compute_berry_phase(past, future)
            
            if is_phase_locked(berry_phase):
                coherence = 1.0
            else:
                # Penalty for non-locked transitions
                coherence = 0.1
                
            # Weight by position (later = more important)
            weight = PHI ** (-(len(sequence) - i))
            total_coherence += coherence * weight
            
        scored_sequences.append((sequence, total_coherence))
        
    # Sort by coherence score
    scored_sequences.sort(key=lambda x: x[1], reverse=True)
    
    return scored_sequences


def generate_with_constraints(
    model,
    context: List[TokenState],
    constraints: Dict[str, any],
    max_length: int = 50
) -> List[TokenState]:
    """
    Generate with specific constraints
    
    Constraints can include:
    - target_length: Aim for specific length
    - must_include: Tokens that must appear
    - avoid: Tokens to avoid
    - style: 'formal', 'casual', etc.
    
    Args:
        model: PhiLanguageModel instance
        context: Initial context
        constraints: Dictionary of constraints
        max_length: Maximum length
        
    Returns:
        Generated sequence satisfying constraints
    """
    target_length = constraints.get('target_length', None)
    must_include = set(constraints.get('must_include', []))
    avoid = set(constraints.get('avoid', []))
    
    generated = []
    included_tokens = set()
    
    for step in range(max_length):
        # Get candidates
        candidates = _generate_candidates(
            model,
            context + generated,
            repetition_penalty=1.0
        )
        
        # Filter by constraints
        valid_candidates = []
        
        for candidate in candidates:
            token = candidate['state'].token
            
            # Check avoid list
            if token in avoid:
                continue
                
            # Boost must_include tokens if not yet included
            if token in must_include and token not in included_tokens:
                candidate['score'] *= 5.0  # Strong boost
                
            # Adjust for target length
            if target_length:
                current_length = len(context) + len(generated) + 1
                if current_length < target_length:
                    # Penalize ending punctuation if too short
                    if token in ['.', '!', '?']:
                        candidate['score'] *= 0.1
                elif current_length > target_length:
                    # Boost ending punctuation if too long
                    if token in ['.', '!', '?']:
                        candidate['score'] *= 10.0
                        
            valid_candidates.append(candidate)
            
        if not valid_candidates:
            break
            
        # Select from valid candidates
        selected = _sample_from_candidates(
            valid_candidates[:10],
            temperature=0.8
        )
        
        generated.append(selected)
        included_tokens.add(selected.token)
        
        # Check termination
        if selected.energy < 0.01 or selected.token in ['.', '!', '?']:
            # Verify constraints satisfied
            if must_include.issubset(included_tokens):
                break
                
    return generated