# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import numpy as np

import torch
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2


def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def main(cfg_scale=1.25):
    init()

    if dist.get_rank() != 0:
        dist.barrier()

    # Setup PyTorch:
    torch.manual_seed(dist.get_rank())
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sampling_steps = 25
    # cfg_scale = 1.5
    batch_size = 8
    num_samples = 500
    all_samples = []
    all_labels = []

    # Load model:
    image_size = 256
    assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
    latent_size = image_size // 8
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    if dist.get_rank() == 0:
        dist.barrier()

    for _ in range(math.ceil(num_samples / (batch_size * dist.get_world_size()))):
        # Labels to condition the model with:
        # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        class_labels = torch.randint(0, 1000, size=(batch_size,)).to(device)

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, 
            clip_denoised=False, model_kwargs=model_kwargs, 
            progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()

        samples = concat_all_gather(samples).cpu().numpy()
        labels = concat_all_gather(class_labels).cpu().numpy()
        all_samples.append(samples)
        all_labels.append(labels)
        if dist.get_rank() == 0:
            print(f"Sampled {(len(all_samples)*len(all_samples[0])):06d} images")

    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]

    if dist.get_rank() == 0:
        print(all_samples.shape, all_labels.shape)
        np.savez("samples.npz", all_samples, all_labels)

    dist.barrier()

    # Save and display images:
    # save_image(samples, f"sample_rank{dist.get_rank()}.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main(float(sys.argv[1]))