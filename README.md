# C-VQA: Counterfactual Reasoning VQA Dataset

This is the code and data for C-VQA.

## Dataset

The dataset directory is `C-VQA`. You can find the questions in `.csv` files and corresponding images in `*_images` directory.

## Scripts

The `scripts` directory contains all required scripts for running models in the paper. 
- `run_eval_codellama.py`: ViperGPT with CodeLlama.
- `run_eval_lavis.py`:  InstructBLIP and BLIP (in LLaVa).
- `run_eval_minigpt4.py`: MiniGPT-4.
- `run_eval_visprog.py`: VisProg.
- `run_eval_wizard.py`: ViperGPT with WizardCoder.

Before you run the a script, install the corresponding model and get the weights. Then put the script in the root directory of the model.

You should change `PATH_TO_IMAGES` in the scripts to the actual directory of images.

You should change `PATH_TO_MODEL` in the scripts for ViperGPT with different code generators to the actual directory of models.

