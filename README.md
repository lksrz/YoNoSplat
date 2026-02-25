<p align="center">
  <h2 align="center">YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting</h2>
 <p align="center">
    <a href="https://botaoye.github.io/">Botao Ye</a>
    ·
    <a href="https://scholar.google.com/citations?user=eAR23MAAAAAJ&hl=en">Boqi Chen</a>
    ·
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath/">Daniel Barath</a>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2511.07321"><img src="https://img.shields.io/badge/arXiv-2511.07321-b31b1b.svg" alt="arXiv"></a>
    &nbsp;
    <a href="https://botaoye.github.io/yonosplat/"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
    &nbsp;
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  </p>

<p align="center">
  <a href="">
    <img src="https://botaoye.github.io/yonosplat/assets/yonosplat_teaser.png" alt="Teaser" width="100%">
  </a>
</p>

<p align="center">
<strong>YoNoSplat</strong> reconstructs 3D Gaussian splats directly from unposed and uncalibrated images, while flexibly leveraging ground-truth camera poses or intrinsics when available.
</p>

## TODO

- [x] Release code and pretrained models
- [ ] Release high-resolution models
- [ ] Dynamic dataloaders for training on more datasets
- [ ] Release models trained on a mixture of more datasets

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation
Our code requires Python 3.10+ and is developed with PyTorch 2.1.2 and CUDA 11.8, but it should work with higher PyTorch/CUDA versions as well.

1. Clone YoNoSplat.
```bash
git clone https://github.com/cvg/YoNoSplat
cd YoNoSplat
```

2. Create the environment (example using conda).
```bash
conda create -y -n yonosplat python=3.10
conda activate yonosplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Pre-trained Checkpoints
## Pre-trained Checkpoints
Our models are hosted on [Hugging Face](https://huggingface.co/botaoye/YoNoSplat) 🤗

|                                            Model name                                             | Training resolutions | Training data |   
|:-------------------------------------------------------------------------------------------------:|:--------------------:|:-------------:|
| [re10k.ckpt]( https://huggingface.co/botaoye/YoNoSplat/resolve/main/re10k_224x224_ctx2to32.ckpt)  |       224x224        |     re10k     |
| [dl3dv.ckpt]( https://huggingface.co/botaoye/YoNoSplat/resolve/main/dl3dv_224x224_ctx2to32.ckpt ) |       224x224        |     dl3dv     |

Download the checkpoints and place them in the `pretrained_weights/` directory.

## Camera Conventions
The camera system follows the [pixelSplat](https://github.com/dcharatan/pixelsplat) convention:
- **Intrinsics**: Normalized camera intrinsic matrices (first row divided by image width, second row divided by image height)
- **Extrinsics**: OpenCV-style camera-to-world matrices (+X right, +Y down, +Z forward into the scene)

## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation instructions.

## Running the Code

### Evaluation

Please refer to [EVALUATION.md](EVALUATION.md) for detailed evaluation commands, including novel view synthesis, pose estimation, and metrics calculation.

### Training

First download the [Pi3](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors) pretrained model and save it as `./pretrained_weights/pi3.safetensors` directory.

#### Train on RealEstate10K (multi-view, 2-32 input views)
```bash
python -m src.main \
  +experiment=yono_re10k \
  trainer.num_nodes=8 \
  wandb.mode=online \
  wandb.name=re10k_ctx2to32 \
  optimizer.lr=1e-4 \
  data_loader.train.batch_size=1 \
  checkpointing.save_weights_only=false \
  dataset.re10k.view_sampler.num_context_views=[2,32]
```

#### Train on DL3DV (multi-view, 2-32 input views)
```bash
python -m src.main \
  +experiment=yono_dl3dv \
  trainer.num_nodes=8 \
  wandb.mode=online \
  wandb.name=dl3dv_ctx2to32 \
  optimizer.lr=1e-4 \
  data_loader.train.batch_size=1 \
  checkpointing.save_weights_only=false \
  dataset.dl3dv.view_sampler.num_context_views=[2,32]
```

You can adjust the batch size and number of GPUs/nodes to fit your hardware. Note that changing the total batch size may require adjusting the learning rate to maintain performance.

## Acknowledgements
This project builds upon several excellent repositories: [NoPoSplat](https://github.com/cvg/NoPoSplat), [Pi3](https://github.com/yyfz/Pi3), [pixelSplat](https://github.com/dcharatan/pixelsplat).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{ye2026yonosplat,
  title     = {YoNoSplat: You Only Need One Model for Feedforward 3D Gaussian Splatting},
  author    = {Ye, Botao and Chen, Boqi and Xu, Haofei and Barath, Daniel and Pollefeys, Marc},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
