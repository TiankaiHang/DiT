set -ex

pip install diffusers timm accelerate

torchrun --standalone --nproc_per_node=8 sample.py 1.25

git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion
pip install -e .
pip install -r evaluations/requirements.txt

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

python evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz ../samples.npz