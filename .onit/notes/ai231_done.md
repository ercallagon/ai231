COMPLETE. AI231 ME1 done end-to-end.
- Repo: ercallagon/ai231 (public, main). Commit bdaf023 "ME1: 3-layer CNN..." on top of 0b7667b (initial). Pushed.
- Files committed: .gitignore, README.md, me1-einops-einsum/mnist_cnn_einops.ipynb (84KB, executed w/ outputs). data/ (64M MNIST) gitignored.
- Secret scan on committed files: CLEAN.
- Result: TEST ACC 95.95% (9595/10000). 5 epochs. 23,946 params.
- Arch: Conv1(1->16,3x3,p1)+ReLU+MP2(->16x14x14); Conv2(16->32)+ReLU+MP2(->32x7x7); Conv3(32->64)+ReLU+GAP(->64x1x1); flatten 64; FC(64->10) via einsum.
- Manual conv = F.pad + x.unfold(2,k,1).unfold(3,k,1) + rearrange 'b c h w i j -> b h w c i j' + torch.einsum 'b h w c i j, o c i j -> b o h w'. Validated vs nn.Conv2d (diff ~1e-5 float noise).
- Key gotchas solved: (1) einops.einsum pattern-first vs torch.einsum pattern-last -> used torch.einsum; (2) einops can't group h*ph or use ellipsis-in-parens -> used unfold; (3) torchvision 0.28 dataset[i][1] is int not tensor -> torch.tensor(list, dtype=long); (4) no jupyter kernel -> registered onit312 kernel spec + installed ipykernel; (5) packages live in /home/edgardo.callagon.jr/.conda/envs/onit/bin/python (3.12), NOT /opt/miniconda3 (3.13).