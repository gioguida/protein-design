"""
Test ESM2 model output
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from esme import ESM2
from esme.alphabet import Alphabet
from esme.alphabet import tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"


def add_context(cdr: str):
    left_context = "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
    right_context = "WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"

    return left_context + cdr + right_context

def load_esm():
    # Load ESM2 model
    esm_model_path: str = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"
    esm = ESM2.from_pretrained(esm_model_path, device=device)
    return esm

def main():
    esm = load_esm()

    # Example CDR sequences (without context)
    cdr_sequences = [
        "HMSMQQVVSAGWERADLVGDAFDV",
        "AASMQQVRSAGWERADLVGDAFEV",
        "ACSMQQVVSAGWSRADLVGDDFDV",
    ]

    # Mask sequences
    masked_sequences = [cdr[:7] + "<mask>" + cdr[8:] for cdr in cdr_sequences]
    # Add context to sequences
    sequences_with_context = [add_context(cdr) for cdr in masked_sequences]

    # Tokenize sequences
    alphabet = Alphabet()
    tokens = tokenize(sequences_with_context, alphabet=alphabet).to(device)

    # Get ESM2 representations
    with torch.no_grad():
        representations = esm(tokens)

    print("Output Object type :", type(representations))

    if isinstance(representations, torch.Tensor):
        print("Output shape :", representations.shape)
        
    if isinstance(representations, dict):
        for key, value in representations.items():
            print(f"Key: {key}, Value type: {type(value)}, Value shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")

if __name__ == "__main__":
    main()

