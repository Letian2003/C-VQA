from huggingface_hub import snapshot_download
import os

repo_ids = [
    "huggyllama/llama-13b",
    "WizardLM/WizardCoder-15B-V1.0",
    "codellama/CodeLlama-13b-Instruct-hf",
    ] 
local_dirs = [
    "hf_model/llama-13b",
    "hf_model/WizardCoder-15B-V1.0",
    "hf_model/CodeLlama-13b-Instruct-hf",
    ] 

for repo_id, local_dir in zip(repo_ids,local_dirs):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    local_dir_use_symlinks = False  
    token = YOUR_HUGGINGFACE_TOKEN

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        token=token,
    )