TASK: AI231 ME1 assignment in me1-einops-einsum/. 3-layer CNN for MNIST, tensor-ops only (NO nn.Conv2d/Linear/MaxPool2d/AvgPool2d). Use einops/einsum + torch.Tensor/Parameter. 5 epochs. Jupyter notebook, self-contained, seeds, 16 test imgs in 4x4 grid w/ GT+pred labels, report test acc.

ENV: /mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox (pwd). Python 3.13 /opt/miniconda3/bin/python3. 256 CPUs. Installing torch(cpu)+einops+numpy+matplotlib+nbformat+nbclient via serve job "pip-install".

GITHUB: login=ercallagon (id 291041090, type User). Token in $GITHUB_TOKEN (len 93). Repo ai231 does NOT exist (404) -> CREATE it (single repo, this assignment lives in me1-einops-einsum/ subdir). No gh CLI; use github_repo tool + git with token in remote URL (never commit token).

STEPS: 1) create repo ai231 2) write notebook 3) execute end-to-end, verify acc + no forbidden modules 4) git init/add/commit/push 5) summarize arch/shapes/setup/acc/repo status.

ARCH (planned): conv1 1->16 k3 s1 p1 (28->28) +maxpool2 (->14); conv2 16->32 k3 p1 (->14) +maxpool2 (->7); conv3 32->64 k3 p1 (->7) +global avg pool (->1); flatten 64 -> linear 64->10. Manual conv via einsum im2col. Adam lr 1e-3, batch 128, CE loss.