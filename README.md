# switch_generation

Repository for [Don't Throw Away Your Pretrained Model](https://arxiv.org/abs/2510.09913).

## Quick Start

#### Initialization

Create a conda environment for Switch Generation
```
conda env create -f switch.yml
conda activate switch_generation
```

Log into huggingface (for model access).
```
huggingface-cli login
```

#### Execute your first Switch Generation inference

```
bash main.sh
```

`main.sh` by default contains:

```
python main_generate.py \
        --input data/input_sample.jsonl \
        --gpu_ids 0,1,2,3 \
        --overide_selector_path bunsenfeng/PFA_switcher_1 \
        --total_max_length 256
```

`--input`: a JSONL file of inputs, look at `data/input_sample.jsonl` for an example of how to prepare your custom inputs. Output will come out at the same directory `data/input_sample_switch_generation.jsonl`.

`--gpu_ids`: a string of numbers separated by comma, 4 GPUs needed (one for P, F, A, and switcher each).

`--overide_selector_path`: path to the switcher LM on Huggingface. We provide `bunsenfeng/PFA_switcher_1`, `bunsenfeng/PFA_switcher_2` with different task and training exposure, you can also just try the aligned model itself `allenai/Llama-3.1-Tulu-3-8B` or any model that could follow instructions.

`--total_max_length`: essentially `max_new_tokens`.

#### Other Settings

Your own data: format it like `data/input_sample.jsonl`.

Your own candidate models: change in lines 46-48 in `main_generate.py`. Make sure `--gpu_ids` provides (n+1) GPU ids where n is the amount of candidate models. Can be other than 3 models. Another recommended set: `["Qwen/Qwen2.5-7B", "bunsenfeng/yuru_qw_oasst1", "Qwen/Qwen2.5-7B-Instruct"]`, where the middle is an SFT model we made in [here](https://arxiv.org/abs/2506.04721).

What's pending: code for switcher training, code for evals in the paper, compatibility such as fewer GPUs than n+1, etc.

#### Citation

If Switch Generation is helpful to you:

```
@article{feng2025don,
  title={Don't Throw Away Your Pretrained Model},
  author={Feng, Shangbin and Yu, Wenhao and Wang, Yike and Zhang, Hongming and Tsvetkov, Yulia and Yu, Dong},
  journal={arXiv preprint arXiv:2510.09913},
  year={2025}
}
```
