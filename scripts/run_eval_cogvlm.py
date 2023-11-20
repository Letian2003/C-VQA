# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if torch.cuda.is_available() else False,
        device='cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)
    
    with torch.no_grad():

        img_paths, queries, new_queries,answers,new_answers,typies = load_query_file(args.query)

        with open('{}_responses.csv'.format(args.type),'w',encoding = 'utf-8') as res_f:

            fieldnames_res = ['img_path', 'query', 'answer', 'new query', 'new answer', 'type', 'response', 'new_response']

            writer_res = csv.DictWriter(res_f, fieldnames=fieldnames_res)
            writer_res.writeheader()

            for (img_path, query, new_query,answer,new_answer,query_type) in tzip(img_paths, queries, new_queries,answers,new_answers,typies):

                image_path = os.path.join(PATH_TO_IMAGES, img_path)
                history = None
                cache_image = None
                q1 = make_prompt(query)

                if world_size > 1:
                    torch.distributed.broadcast_object_list(image_path, 0)

                result, _, _ = chat(
                    image_path, 
                    model, 
                    text_processor_infer,
                    image_processor,
                    q1, 
                    history=history, 
                    image=cache_image, 
                    max_length=args.max_length, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    no_prompt=args.no_prompt
                    )
                
                q2 = make_prompt(new_query)
                history = None
                cache_image = None

                new_result, _, _ = chat(
                    image_path, 
                    model, 
                    text_processor_infer,
                    image_processor,
                    q2, 
                    history=history, 
                    image=cache_image, 
                    max_length=args.max_length, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    no_prompt=args.no_prompt
                    )
                # print(q1,":",result)
                # print(q2,":",new_result)
            
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
        
    if world_size > 1:
        torch.distributed.broadcast_object_list(image_path, 0)


if __name__ == "__main__":
    main()
