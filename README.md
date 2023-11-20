# C-VQA: Counterfactual Reasoning VQA Dataset

This is the code and data for the paper [What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models](https://arxiv.org/abs/2310.06627).

## Dataset

The dataset directory is `C-VQA`. You can find the questions in `.csv` files. 


### Download Images

After cloning:

```bash
pip install gdown
bash download_images.sh
```

## Scripts

The `scripts` directory contains all required scripts for running models in the paper. 
  
- `run_eval_cogvlm.py`: [CogVLM](https://github.com/THUDM/CogVLM).
  
- `run_eval_lavis.py`:  InstructBLIP and BLIP (in [LAVIS](https://github.com/salesforce/LAVIS)).
  
- `run_eval_minigpt4.py`: [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4).

- `run_eval_llava.py`: [LLaVA](https://github.com/haotian-liu/LLaVA).

- `run_eval_qwen.py`: [Qwen-VL](https://github.com/QwenLM/Qwen-VL).
  
- `run_eval_codellama.py`: [ViperGPT](https://github.com/cvlab-columbia/viper) with [CodeLlama](https://github.com/facebookresearch/codellama).
  
- `run_eval_visprog.py`: [VisProg](https://github.com/allenai/visprog).
  
- `run_eval_wizard.py`: [ViperGPT](https://github.com/cvlab-columbia/viper) with [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder).
  
Before you run a script, install the corresponding model and get the weights. Then put the script in the root directory of the model.

Please change `PATH_TO_IMAGES` in the scripts to the actual directory of images.

Please change `PATH_TO_MODEL` in the scripts for ViperGPT with different code generators to the actual directory of models.

For example, to run BLIP on C-VQA, run this command in the root directory of LLaVa:

```python
python run_eval_lavis.py --model-name blip2_t5 --model-type pretrain_flant5xxl --query PATH_TO_CSV_FILE
```

You can find more commands in [scripts/README](scripts/READMD.md).

After you get the results, run `format_response.py` to convert raw responses to formatted responses (a single number or a single `yes` or `no`). Then run `calc_acc.py` to get quantitative results of the formatted responses. Remenber to fill in file names in these two scripts.


### Download Code Generator Models

Change YOUR_HUGGINGFACE_TOKEN in `download_model.py` to your huggingface token. Then run:

```
pip install huggingface_hub
python download_model.py
```

You can add more code generators in `download_model.py` by adding models in repo_ids and local_dirs.


### Citation

If this code is useful for your research, please consider citing our work.

```
@InProceedings{zhang2023cvqa,
    author    = {Zhang, Letian and Zhai, Xiaotong and Zhao, Zhongkai and Wen, Xin and Zhao, Bingchen},
    title     = {What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-Modal Language Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year      = {2023}
}
```
