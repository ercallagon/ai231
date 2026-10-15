# Models — final deployable pretrained models.
#
#   models/vcm/<tag>.onnx          exported VCM (CPU deploy)
#   models/vcm/<tag>.labels.json   intent order + slot sets + OOS τ
#   models/wakeword/wakeword_exp15_mined_lp.onnx   wake-word detector
#
# ONNX + .pt artifacts are gitignored (regenerable via scripts/03_export_onnx.py).
# The deploy bundle (deploy/rpi/) is staged from here by scripts/05_stage_rpi.py.
