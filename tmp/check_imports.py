import sys
print("python:", sys.executable)
import torch, einops, numpy, matplotlib, nbformat, nbclient, torchvision
print("torch", torch.__version__)
print("einops", einops.__version__)
print("numpy", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("torchvision", torchvision.__version__)
print("nbformat", nbformat.__version__)
print("nbclient", nbclient.__version__)
print("ALL_IMPORTS_OK")
