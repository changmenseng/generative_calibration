This is the official implementation of our code "Generative Calibration for In-context Learning" in EMNLP2023 Findings. In this paper, we propose generative calibration (GC), a lightweight calibration method for in-context learning (ICL) that significantly improves the ICL performance and achieves new state-of-art, as shown in the following figure:

![img](./img/4-shot.pdf)

To use the code, first, please install the requirements:

```
pip install -r requirements.txt
```

Then, please use the following command to run the experiment of the language model and dataset you want:

```
python run.py --dataset <dataset> --model <model>
```

Don't worry, this script will download the language model and the dataset at first. Experimental results (including generated sequences) are stored in `saved` folder by default, which you can specify it via argument `--save_path`. Please check the script for other arguments.

IMPORTANT: You should check the `llm_metadata.json` file. This file specify how to split the LLM into different GPUs. This file stores our default setting, which applyies to the devices in the following table:

| Model             | Devices  |
| ----------------- | -------- |
| GPT2-Large (774M) | 1*2080Ti |
| GPT2-XL (1.5B)    | 1*2080Ti |
| GPT-NEO (2.7B)    | 2*2080Ti |
| GPT-J (6B)        | 2*2080Ti |
| RWKV (3B)         | 2*2080Ti |
| RWKV (7B)         | 2*2080Ti |
| RWKV (14B)        | 3*2080Ti |
| GPT-NEOX (20B)    | 3*3090   |
| OPT (13B)         | 2*3090   |
| OPT (30B)         | 3*3090   |
| LLaMA (13B)       | 2*3090   |
| LLaMA (33B)       | 3*3090   |

The GPU spliting setting should be comfortable to arguments `--batch_size` and `--n_samples_once` of the script `run.py` so that OOM wouldn't happen.
