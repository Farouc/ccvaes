import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
print(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test rapide d'un tenseur sur le GPU
x = torch.rand(5, 3).cuda()
print("Test tenseur sur GPU: OK")