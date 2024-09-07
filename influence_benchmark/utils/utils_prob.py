from typing import Dict, Tuple


def is_in_simplex(probabilities: list[float]) -> bool:
    """
    Check if the probabilities are in the simplex.
    """
    return all(p >= 0 for p in probabilities) and abs(sum(probabilities) - 1.0) < 1e-9


def check_simplex_and_transform(prob_dict: Dict[str, float], log_name: str) -> Tuple[bool, Dict[str, float]]:
    """
    Check and transform probabilities to ensure they live in the simplex.

    Args:
    prob_dict (Dict[str, float]): Dictionary mapping preferences to probabilities
    log_name (str): Name of class for logging purposes

    Returns:
    bool: This is a flag for whether the probs are unfixable.
    Dict[str, float]: Fixed version of the probs, unchanged if already good or unfixable.
    """
    probs = list(prob_dict.values())

    # Check if probabilities live in the simplex
    if is_in_simplex(probs):
        return False, prob_dict

    # Check if all elements are zero
    elif all(p == 0 for p in probs):
        print(f"Warning: All elements of {log_name} probabilities are zero. Returning default transition.")
        return True, prob_dict

    # Check for negative elements
    elif any(p < 0 for p in probs):
        print(f"Warning: Negative elements found in {log_name} probabilities. Returning default transition.")
        return True, prob_dict

    # Otherwise, normalize probabilities and log a warning
    else:
        print(f"Warning: {log_name} probabilities do not sum to 1. Normalizing.")
        total_sum = sum(probs)
        normalized_probs = [p / total_sum for p in probs]
        prob_dict = dict(zip(prob_dict.keys(), normalized_probs))
        return False, prob_dict
