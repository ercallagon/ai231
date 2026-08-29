import torch
from einops import rearrange

def conv2d_same(x, w, b, k=3):
    import torch.nn.functional as F
    B, Cin, H, W = x.shape
    Cout = w.shape[0]
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad))
    # unfold: (B, Cin, H, W) -> (B, Cin, H, W, k*k)
    patches = x.unfold(2, k, 1).unfold(3, k, 1)   # (B, Cin, H, W, k, k)
    patches = patches.contiguous()
    # reorder to (B, H, W, Cin, k, k)
    patches = rearrange(patches, 'b c h w i j -> b h w c i j')
    # contract with filters
    out = torch.einsum('b h w c i j, o c i j -> b o h w', patches, w)
    out = out + b.view(1, Cout, 1, 1)
    return out

torch.manual_seed(0)
x = torch.randn(4, 1, 28, 28)
w1 = torch.randn(16, 1, 3, 3); b1 = torch.zeros(16)
w2 = torch.randn(32, 16, 3, 3); b2 = torch.zeros(32)
w3 = torch.randn(64, 32, 3, 3); b3 = torch.zeros(64)
fw = torch.randn(10, 64); fb = torch.zeros(10)

def maxpool2(x, k=2):
    x = rearrange(x, 'b c (h q) (w r) -> b c h w q r', q=k, r=k)
    return x.max(dim=-1).values.max(dim=-1).values

def gap(x):
    from einops import reduce
    return reduce(x, 'b c h w -> b c 1 1', 'mean')

x = conv2d_same(x, w1, b1, 3); print("after conv1:", tuple(x.shape)); assert x.shape==(4,16,28,28)
x = torch.relu(x); x = maxpool2(x); print("after pool1:", tuple(x.shape)); assert x.shape==(4,16,14,14)
x = conv2d_same(x, w2, b2, 3); print("after conv2:", tuple(x.shape)); assert x.shape==(4,32,14,14)
x = torch.relu(x); x = maxpool2(x); print("after pool2:", tuple(x.shape)); assert x.shape==(4,32,7,7)
x = conv2d_same(x, w3, b3, 3); print("after conv3:", tuple(x.shape)); assert x.shape==(4,64,7,7)
x = torch.relu(x); x = gap(x); print("after gap:", tuple(x.shape)); assert x.shape==(4,64,1,1)
x = rearrange(x, 'b c h w -> b (c h w)'); print("after flatten:", tuple(x.shape)); assert x.shape==(4,64)
logits = torch.einsum('b i, o i -> b o', x, fw) + fb; print("logits:", tuple(logits.shape)); assert logits.shape==(4,10)

# --- Correctness check vs nn.Conv2d (reference), on a FRESH input ---
xin = torch.randn(4, 1, 28, 28)
ref = torch.nn.Conv2d(1, 16, 3, padding=1, bias=False)
ref.weight.data = w1
xref = torch.nn.functional.relu(ref(xin))
xman = torch.nn.functional.relu(conv2d_same(xin, w1, b1, 3))
diff = (xref - xman).abs().max().item()
print("MAX DIFF vs nn.Conv2d (conv1):", diff)
assert diff < 1e-5, "CONV MISMATCH"

# also check conv2 (multi-channel) correctness
xin2 = torch.randn(4, 16, 14, 14)
ref2 = torch.nn.Conv2d(16, 32, 3, padding=1, bias=False)
ref2.weight.data = w2
d2 = (torch.nn.functional.relu(ref2(xin2)) - torch.nn.functional.relu(conv2d_same(xin2, w2, b2, 3))).abs().max().item()
print("MAX DIFF vs nn.Conv2d (conv2):", d2)
assert d2 < 1e-5, "CONV2 MISMATCH"
print("ALL SHAPE + CORRECTNESS CHECKS PASSED")
