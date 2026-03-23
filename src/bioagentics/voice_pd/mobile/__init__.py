"""Mobile deployment utilities for PD voice model.

Phase 6 modules:
- converter: PyTorch → ONNX → TFLite conversion and inference
- protocol: 30-second voice task protocol (sustained vowel + sentence reading)
- benchmark: End-to-end inference benchmarking
"""
