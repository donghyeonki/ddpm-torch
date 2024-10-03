# How to accelerate diffusion model training?

## 242R [서울-대학원]신경망응용및실습(영강)(APPLICATIONS AND PRACTICE IN NEURAL NETWORKS(English))-00분반

## Email for QA
peop1e1n@korea.ac.kr

## Introduction

The complexity of the data distributions we aim to generate increases, training a diffusion model to convergence is becoming increasingly computationally demanding. 

- Training images with Stable-Diffusion-2.0(Rombach et al. 2022) requires 24,000 A100 GPU hours

- Open-Sora (Zheng, Peng, and You 2024) requires 48,000 H800 GPU hours for video training. 

Therefore, accelerating the training of diffusion models is a crucial challenge.

## baseline models

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- The above code is DDPM which is baseline of this project.
- You can develop an algorithm that can accelerate the training of diffusion models on this baseline code.

## Previous methods 

Changing the Weighting Scheme

- [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)

- [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227)

Changing the Sampling Timestep Scheme

- [A Closer Look at Time Steps is Worthy of Triple Speed-Up for Diffusion Model Training](https://arxiv.org/abs/2405.17403)

However, these methods are all heuristic and are fixed throughout the entire diffusion model training process.

## Goal of this project

The goal of this project is to develop an algorithm that can accelerate the training of diffusion models more effectively than existing methods by being adaptive to the training process.

It doesn't matter whether propose a new weighting method or a timestep sampling method.

The key is to achieve better performance than the baseline when comparing training times in terms of wall clock time, within the same training duration.

## Experiments Settings

Task : image generation

Datasets : [CIFAR-10 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

- The dataset is automatically downloaded when the code is executed.

Baseline : [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)

Network architecture : U-Net are famous architectures in the diffusion model area. 

Evaluation protocols : In inference, we default to generating 10K images. Frechet Inception Distance (FID) is used to evaluate both the fidelity and coverage of generated images. You have to use DDPM sampling. (not DDIM)



## Implementations

- [x] Original DDPM[^1] training & sampling
- [x] DDIM[^2] sampler
- [x] Standard evaluation metrics
	- [x] Fréchet Inception Distance[^3] (FID)
	- [x] Precision & Recall[^4]
- [x] Distributed Data Parallel[^5] (DDP) multi-GPU training


## Requirements

- torch>=1.12.0
- torchvision>=1.13.0
- scipy>=1.7.3

## Code usages
**Examples**

- Train CIFAR-10 model with single GPU (device id: 0) for a total of 50 epochs
    ```shell
    python train.py --dataset cifar10 --train-device cuda:0 --epochs 50
    ```
    
    ```shell
    python train.py --dataset cifar10 --num-accum 2 --num-gpus 4 --distributed --rigid-launch
    ```
    - `num-accum 2`: accumulate gradients for 2 mini-batches
    - `num-gpus`: number of GPU(s) to use for training, i.e. `WORLD_SIZE` of the process group
    - `distributed`: enable multi-gpu DDP training
    - `rigid-run`: use shared-file system initialization and `torch.multiprocessing`
    - 

- Generate 50,000 samples (128 per mini-batch) of the checkpoint located at `./chkpts/cifar10/cifar10_2040.pt` in parallel using 4 GPUs and DDIM sampler. The results are stored in `./images/eval/cifar10/cifar10_2040_ddim`
	```shell
	python generate.py --dataset cifar10 --chkpt-path ./chkpts/cifar10/cifar10_2040.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --suffix _ddim --num-gpus 4
	```
    - `use-ddim`: use DDIM
    - `skip-schedule quadratic`: use the quadratic schedule
    - `subseq-size`: length of sub-sequence, i.e. DDIM timesteps
    - `suffix`: suffix string to the dataset name in the folder name
    - `num-gpus`: number of GPU(s) to use for generation

- Evaluate FID, Precision/Recall of generated samples in `./images/eval/cifar10_2040`
	```shell
	python eval.py --dataset cifar10 --sample-folder ./images/eval/cifar10/cifar10_2040
	```
## References

[^1]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
[^2]: Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." International Conference on Learning Representations. 2020.
[^3]: Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium." Advances in neural information processing systems 30 (2017).
[^4]: Kynkäänniemi, Tuomas, et al. "Improved precision and recall metric for assessing generative models." Advances in Neural Information Processing Systems 32 (2019).
[^5]: DistributedDataParallel - PyTorch 1.12 Documentation, https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html.
[^6]: Torchrun (Elastic Launch) - PyTorch 1.12 Documentation*, https://pytorch.org/docs/stable/elastic/run.html. 

