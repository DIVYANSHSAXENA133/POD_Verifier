#!/usr/bin/env python3
"""Write aws/lambda_scorer/model/best.pt (untrained MultiHeadEfficientNet weights). Replace for production."""

import os

import torch

_AWS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    os.chdir(os.path.join(_AWS_ROOT, "lambda_scorer"))
    import sys

    sys.path.insert(0, ".")
    from src.model import MultiHeadEfficientNet

    dest = os.path.join(_AWS_ROOT, "lambda_scorer", "model", "best.pt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    model = MultiHeadEfficientNet(num_attributes=4, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, dest)
    print(f"Wrote {dest} ({os.path.getsize(dest) // (1024 * 1024)} MiB)")
