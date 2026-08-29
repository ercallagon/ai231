import sys, json
from nbformat import read, write
from nbclient import NotebookClient

NB = "/mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox/me1-einops-einsum/mnist_cnn_einops.ipynb"

nb = read(NB, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="onit312",
                        resources={"metadata": {"path": "/mnt/jfs_hpc/home/edgardo.callagon.jr/sandbox/me1-einops-einsum"}})
client.execute()
write(nb, NB)   # save executed notebook with outputs

# Collect all stream/text outputs for inspection
print("\n\n========== EXECUTION OUTPUT CAPTURE ==========")
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        if ot == "stream":
            sys.stdout.write(out.get("text", ""))
        elif ot == "error":
            print("!!! CELL", i, "ERROR !!!")
            print("\n".join(out.get("traceback", [])))
        elif ot == "execute_result":
            txt = out.get("data", {}).get("text/plain", "")
            if txt:
                sys.stdout.write(txt + "\n")
