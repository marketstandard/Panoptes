"""Neural document detector infrastructure for Panoptes v2.1.

This package implements the preregistered train/development-only architecture
pilot (protocol v2.1, Phase 4) and the frozen final detector (Phase 5). The
data firewall is absolute here: pilot code reads only the training and
development partitions of a cohort; the final test labels are never loaded by
any windowing, training, selection, calibration, or aggregation path.

Modules:
  - windowing:  tokenizer-offset document windowing with overlap bookkeeping.
  - data:       windowed corpora with leakage-group and cohort metadata.
  - model:      window encoder + classification head + hierarchical summary head.
  - objectives: ERM, group-balanced, and GroupDRO objectives over audit groups.
  - aggregate:  overlap-corrected logit mean and summary-head document scoring.
  - train:      training loops with development-only early stopping/selection.
"""
