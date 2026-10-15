# Exp15 checkpoint epoch correction

Original config.json is preserved verbatim. Its best_epoch60 is inaccurate: the
trainer writes the final in-memory history epoch, not the selected checkpoint
epoch. Archived logs/exp15_chain.log lines67-68 mark the final improvement at42;
saved history.json ends at42, while training continues through60. The original
best.pt hash and deployed ONNX parity are independently verified; this metadata
correction changes neither weights, export, thresholds nor historical metrics.
