import modal
import os

# Definiujemy obraz Modal z wszystkimi zależnościami YoNoSplat
# Wykorzystujemy CUDA 11.8 i PyTorch 2.1.2 zgodnie z README
yonosplat_image = (
    # Oficjalny obraz PyTorch z CUDA 11.8 - ma już torch, nvcc i g++ poprawnie skonfigurowane
    modal.Image.from_registry("pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Europe/Warsaw"})
    .run_commands(
        "python -m pip install --upgrade pip",
        # torch jest już zainstalowany, instalujemy tylko zależności YoNoSplat
        "python -m pip install \"numpy<2.0\" wheel tqdm hydra-core jaxtyping beartype wandb einops colorama scikit-image colorspacious matplotlib moviepy==1.0.3 imageio timm dacite lpips plyfile tabulate svg.py scikit-video opencv-python",
        "python -m pip install torchmetrics==1.2.1 pytorch-lightning==2.1.2 lightning-utilities==0.10.0",
        "python -m pip install e3nn==0.5.1"
    )
)

app = modal.App("yonosplat-demo", image=yonosplat_image)

# Miejsce na wagi modelu - możemy użyć Modal Volume, żeby nie ściągać ich za każdym razem
weights_volume = modal.Volume.from_name("yonosplat-weights", create_if_missing=True)

@app.function(
    gpu="A10G", # A10G wystarczy do skompilowania rasteryzatora
    volumes={"/pretrained_weights": weights_volume},
    timeout=3600
)
def run_inference(images_bytes):
    """
    Funkcja przyjmuje obrazy jako bajty i zwraca wygenerowany splat.
    """
    import torch
    import os
    import subprocess
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 1. Kompilacja i instalacja rasteryzatora wewnątrz działającego kontenera (z dostępem do GPU)
    if not os.path.exists("/usr/local/lib/python3.10/site-packages/diff_gaussian_rasterization"):
        print("Instalacja rasteryzatora (to potrwa kilka minut)...")
        # Pobieramy ninja, bo jest potrzebny do kompilacji
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ninja-build"], check=True)
        
        # Klonujemy i instalujemy
        subprocess.run(["git", "clone", "https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git", "/tmp/rasterizer"], check=True)
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = "8.6" # Architektura A10G
        subprocess.run(["python", "-m", "pip", "install", "/tmp/rasterizer", "--no-build-isolation"], env=env, check=True)
        print("Rasteryzator zainstalowany pomyślnie.")

    # 2. Zapisujemy obrazy lokalnie w kontenerze
    input_dir = "/tmp/input_images"
    os.makedirs(input_dir, exist_ok=True)
    for i, b in enumerate(images_bytes):
        with open(f"{input_dir}/{i:03d}.jpg", "wb") as f:
            f.write(b)
            
    # TODO: Logika YoNoSplat
    
    return "Obrazy odebrane. Środowisko skonfigurowane."

@app.local_entrypoint()
def main():
    import os
    print("Odpalam demo YoNoSplat na Modal.com...")
    
    image_dir = "test_images"
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
    
    # Przesyłamy obrazy jako bajty, bo funkcja Modal odpala się w chmurze
    images_bytes = []
    for path in image_paths:
        with open(path, "rb") as f:
            images_bytes.append(f.read())
            
    print(f"Wysyłam {len(images_bytes)} obrazów do Modala...")
    run_inference.remote(images_bytes)
