from lavis.models import load_model_and_preprocess
import torch
import argparse

import os
import requests
import pandas as pd

from PIL import Image
from io import BytesIO
from tqdm.contrib import tzip

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_query_file(query_file):
    df = pd.read_csv(query_file)
    img_paths = df['img_path'].tolist()
    queries = df['query'].tolist()
    new_queries = df['new query'].tolist()
    assert len(img_paths) == len(queries) == len(new_queries)
    return img_paths, queries, new_queries

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()
    return args

def make_prompt(prompt):
    prompt = "Question: " + prompt + " Short answer:"
    return prompt

# def make_prompt_syn(prompt):
#     prompt =  "Question: "+ prompt
#     prompt = prompt.replace('Select the correct answer', 'Options')
#     prompt = prompt.replace('A:', '(a)')
#     prompt = prompt.replace('B:', '(b)')
#     prompt = prompt.replace('C:', '(c)')
#     prompt = prompt.replace('D:', '(d)')
#     prompt = prompt + ". Answer:"
#     return prompt

def main():
    args = parse_args()
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device="cuda",
    )

    min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty = \
        1, 250, "Beam search", 0.9, 5, 1, 1

    img_paths, queries, new_queries = load_query_file(args.query)
    responses, new_responses = [], []
    for (img_path, query, new_query) in tzip(img_paths, queries, new_queries):
        img_path = os.path.join(PATH_TO_IMAGES, img_path)
        image = vis_processors["eval"](load_image(img_path)).unsqueeze(0).cuda()

        q1 = make_prompt(query)

        samples = {
            "image": image,
            "prompt": q1,
        }

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=False,
        )
        responses.append(output[0])

        # print(q1,":",output[0])

        q2 = make_prompt(new_query)

        samples = {
            "image": image,
            "prompt": q2,
        }

        output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=False,
        )
        new_responses.append(output[0])

        # print(q2,":",output[0])

    # add responses and new_responses as new columns to the dataframe
    df = pd.read_csv(args.query)
    df['response'] = responses
    df['new_response'] = new_responses
    df.to_csv('{}_responses.csv'.format(args.type), index=False)


if __name__ == "__main__":
    main()