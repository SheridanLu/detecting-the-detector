"""Seeds. The original runs used seeds 0-4 (LR, RoBERTa) and 0-1 (Qwen); random_state of LogisticRegression = seed."""
import os
import random

import numpy as np


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
