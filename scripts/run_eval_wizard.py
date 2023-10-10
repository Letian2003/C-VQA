from main_simple_lib import *
import argparse
import os
import csv


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd

from PIL import Image

import sys

import torch
import transformers
from tqdm.contrib import tzip
import pathlib
from functools import partial
import warnings
import traceback


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def run_program(parameters, queues_in_, input_type_, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from video_segment import VideoSegment

    global queue_results

    code, sample_id, image, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match):\n' \
                  f'    # Answer is:'
    code = code_header + code

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code
    
def evaluate(
        batch_data,
        tokenizer,
        model,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        **kwargs,
):
    prompts = generate_prompt(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    # return  generation_output
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

Only answer with a function starting def execute_command.
### Response:"""


print("loading wizard...")

device = "cuda"

load_8bit: bool = False

base_model = PATH_TO_MODEL

tokenizer = AutoTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
elif device == "mps":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )

model.config.pad_token_id = tokenizer.pad_token_id

if not load_8bit:
    model.half()

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("loading wizard successfully!")

prompt_file = config.codex.prompt
with open(prompt_file) as f:
    base_prompt = f.read().strip()


def get_ex_prompt(query):
    extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", query).replace('INSERT_TYPE_HERE', query)
    return extended_prompt

def get_wizard_code(extended_prompt):
    _output = evaluate(extended_prompt, tokenizer, model)

    
    code = _output[0].split("### Response:")[1].strip()

    start = code.find("```")
    if start != -1:
        code = code[start + 3:]
    end = code.find("```")
    if end != -1:
        code = code[:end]
        
    start = code.find("def execute_command")
    code = code[start:]
    
    start = code.find("\n")
    code = code[start + 1:]

    end = code.find("def execute_command")
    if end != -1:
        code = code[:end]
    
    end = code.find("#")
    if end != -1:
        code = code[:end]
    
    end = code.find("Query:")
    if end != -1:
        code = code[:end]

    return code

def load_query_file(query_file):
    df = pd.read_csv(query_file)
    img_paths = df['img_path'].tolist()
    queries = df['query'].tolist()
    new_queries = df['new query'].tolist()
    answers = df['answer'].tolist()
    new_answers = df['new answer'].tolist()
    typies = df['type'].tolist()
    assert len(img_paths) == len(queries) == len(new_queries)
    return img_paths, queries, new_queries,answers,new_answers,typies

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    return args

def get_code2(extended_prompt):
    code = get_wizard_code(extended_prompt)
    code = f'def execute_command(image, my_fig, time_wait_between_lines, syntax):\n' + code
    
    code_for_syntax = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=False, start_line=0)
    console.print(syntax_1)
    code = ast.unparse(ast.parse(code))
    code_for_syntax_2 = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=False, start_line=0)
    return code, syntax_2

def main():
    args = parse_args()

    img_paths, queries, new_queries,answers,new_answers,typies = load_query_file(args.query)

    with open('{}_responses_vipergpt.csv'.format(args.type),'w',encoding = 'utf-8') as res_f, open('{}_codes_vipergpt.csv'.format(args.type),'w',encoding = 'utf-8') as code_f:

        fieldnames_res = ['img_path', 'query', 'answer', 'new query', 'new answer', 'type', 'response', 'new_response']
        fieldnames_code = ['code', 'new code']

        writer_res = csv.DictWriter(res_f, fieldnames=fieldnames_res)
        writer_res.writeheader()
        writer_code = csv.DictWriter(code_f, fieldnames=fieldnames_code)
        writer_code.writeheader()

        for (img_path, query, new_query,answer,new_answer,query_type) in tzip(img_paths, queries, new_queries,answers,new_answers,typies):

            img = load_image(PATH_TO_IMAGES + img_path)

            print(img_path)
            
            result, new_result = "failed", "failed"
            code, new_code = 'failed', 'failed'
            

            prompt = get_ex_prompt(query)
            try:
                code = get_code2(prompt)
                result = execute_code(code, img, show_intermediate_steps=False)
            except:
                result = 'fail'

            new_prompt = get_ex_prompt(new_query)
            try:
                new_code = get_code2(new_prompt)
                new_result = execute_code(new_code, img, show_intermediate_steps=False)
            except:
                new_result = 'fail'


            writer_res.writerow({
                'img_path': img_path, 
                'query': query, 
                'answer': answer, 
                'new query': new_query, 
                'new answer': new_answer, 
                'type': query_type, 
                'response': result, 
                'new_response': new_result})
            
            writer_code.writerow({
                'code': code[0], 
                'new code': new_code[0]})

            
            res_f.flush()
            code_f.flush()




if __name__ == "__main__":
    main()