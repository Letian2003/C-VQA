# C-VQA: Counterfactual Reasoning VQA Dataset

This is the code and data for C-VQA.

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
- `run_eval_codellama.py`: [ViperGPT](https://github.com/cvlab-columbia/viper) with [CodeLlama](https://github.com/facebookresearch/codellama).
- `run_eval_lavis.py`:  InstructBLIP and BLIP (in [LLaVa](https://github.com/haotian-liu/LLaVA)).
- `run_eval_minigpt4.py`: [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
- `run_eval_visprog.py`: [VisProg](https://github.com/allenai/visprog).
- `run_eval_wizard.py`: ViperGPT with [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder).

Before you run the a script, install the corresponding model and get the weights. Then put the script in the root directory of the model.

You should change `PATH_TO_IMAGES` in the scripts to the actual directory of images.

You should change `PATH_TO_MODEL` in the scripts for ViperGPT with different code generators to the actual directory of models.

For example, to run BLIP on C-VQA, you should run this command in the root directory of LLaVa:

```python
python run_eval_lavis.py --model-name blip2_t5 --model-type pretrain_flant5xxl --query PATH_TO_CSV_FILE
```

### Download Code Generator Models

Change YOUR_HUGGINGFACE_TOKEN in `download_model.py` to your huggingface token. Then run:

```
pip install huggingface_hub
python download_model.py
```

You can add more code generators in `download_model.py` by adding models in repo_ids and local_dirs.
