# PyTorch Implementation of Denoising Diffusion Probabilistic Models [[paper]](https://arxiv.org/abs/2006.11239) [[official repo]](https://github.com/hojonathanho/diffusion)


## Features

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

