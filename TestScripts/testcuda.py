import torch

print("Torch version:", torch.__version__)
print("CUDA disponibile:", torch.cuda.is_available())
print("Versione CUDA supportata da PyTorch:", torch.version.cuda)
print("Numero di GPU:", torch.cuda.device_count())
print("Nome GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nessuna GPU trovata")
print("Dispositivo corrente:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
