import json, re

NB = "/mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox/me1-einops-einsum/mnist_cnn_einops.ipynb"
nb = json.load(open(NB))

# 1) Forbidden high-level modules (as actual calls/constructors)
forbidden = ["nn.Conv2d", "nn.Linear", "nn.MaxPool2d", "nn.AvgPool2d",
             "F.conv2d", "F.linear", "F.max_pool2d", "F.avg_pool2d",
             "nn.Conv1d", "nn.ConvTranspose2d", "nn.Upsample", "nn.Flatten"]
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
print("=== FORBIDDEN MODULE SCAN (code cells) ===")
found_any = False
for f in forbidden:
    # match the token not followed by '(' we still flag; but allow mention in comments? 
    # We flag any occurrence of the identifier.
    hits = [ln for ln in src.splitlines() if f in ln]
    if hits:
        found_any = True
        print(f"  FOUND '{f}':")
        for h in hits:
            print("     ", h.strip())
if not found_any:
    print("  NONE FOUND — clean.")

# 2) Confirm einops/einsum actually used
print("\n=== einops/einsum USAGE ===")
for tok in ["rearrange(", "reduce(", "torch.einsum(", ".unfold("]:
    n = src.count(tok)
    print(f"  {tok}: {n} occurrence(s)")

# 3) Confirm no external .py model file referenced
print("\n=== EXTERNAL MODEL FILE REFERENCE ===")
ext = [ln for ln in src.splitlines() if re.search(r"import\s+(model|net|cnn)|from\s+(model|net|cnn)\s+import", ln)]
print("  ", ext if ext else "none (model is defined inline) — good")

# 4) Check the image-grid cell produced a figure (image/png output)
print("\n=== IMAGE GRID OUTPUT ===")
grid_found = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for out in c.get("outputs", []):
        if out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                grid_found = True
                print(f"  cell {i}: has image/png output (len={len(data['image/png'])} b64 chars)")
print("  figure rendered:", grid_found)

# 5) Confirm all code cells executed (have execution_count)
print("\n=== EXECUTION COUNTS ===")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
        print(f"  cell {i}: exec_count={c.get('execution_count')} errors={len(errs)}")
