from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

import argparse
import os
import csv

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

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    return args

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

def make_prompt(prompt):
    return prompt + " Answer the question using a single word or phrase."
# Specify hyperparameters for generation

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    img_paths, queries, new_queries,answers,new_answers,typies = load_query_file(args.query)


    with open('{}_responses.csv'.format(args.type),'w',encoding = 'utf-8') as res_f:

        fieldnames_res = ['img_path', 'query', 'answer', 'new query', 'new answer', 'type', 'response', 'new_response']

        writer_res = csv.DictWriter(res_f, fieldnames=fieldnames_res)
        writer_res.writeheader()

        for (img_path, query, new_query,answer,new_answer,query_type) in tzip(img_paths, queries, new_queries,answers,new_answers,typies):

            image = os.path.join(PATH_TO_IMAGES, img_path)

            q1 = tokenizer.from_list_format([
                {'image': image}, # Either a local path or an url
                {'text': make_prompt(query)},
            ])

            result, _ = model.chat(tokenizer, query=q1, history=None)

            q2 = tokenizer.from_list_format([
                {'image': image}, # Either a local path or an url
                {'text': make_prompt(new_query)},
            ])

            new_result, _ = model.chat(tokenizer, query=q2, history=None)
            
            writer_res.writerow({
                'img_path': img_path, 
                'query': query, 
                'answer': answer, 
                'new query': new_query, 
                'new answer': new_answer, 
                'type': query_type, 
                'response': result, 
                'new_response': new_result})
            
            res_f.flush()


if __name__ == "__main__":
    args = parse_args()
    main(args)