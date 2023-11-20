## Command

For end-to-end models, run the following commands for corresponding models:

- CogVLM
```python
python run_eval_cogvlm.py --from_pretrained cogvlm-chat  --version chat --english --bf16 \
    --query PATH/TO/C-VQA-Real_questions.csv \
    --type cogvlm-chat
```

- LAVIS
```python
python run_eval_lavis.py \
    --model-name blip2_t5 \
    --model-type pretrain_flant5xxl \
    --query PATH/TO/C-VQA-Real_questions.csv \
    --type blip2
```
You can also use `-model-name blip2_vicuna_instruct --model-type vicuna7b`, `-model-name blip2_vicuna_instruct --model-type vicuna13b`, `-model-name blip2_t5_instruct --model-type flant5xxl` for other models.

- LLaVA

```python
python run_eval_llava.py \
    --query PATH/TO/C-VQA-Real_questions.csv \
    --model-path liuhaotian/llava-v1.5-7b \
    --type llava_v15_7b
```
You can also use `--model-path liuhaotian/llava-v1.5-13b`, `--model-path PATH/TO/LLaVA-7B-v0`, `--model-path PATH/TO/LLaVA-7B-v1-1`  for other models.

- MiniGPT-v2
```python
python run_eval_minigpt4.py \
    --query PATH/TO/C-VQA-Real_questions.csv \
    --cfg_path eval_configs/minigptv2_eval.yaml \
    --type minigptv2
```

- Qwen-VL
```python
python run_eval_qwen.py \
    --query PATH/TO/C-VQA-Real_questions.csv \
    --type qwen
```