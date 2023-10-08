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
    args = parser.parse_args()
    return args


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
        img_path = PATH_TO_IMAGES + img_path
        image = vis_processors["eval"](load_image(img_path)).unsqueeze(0).cuda()

        samples = {
            "image": image,
            "prompt": query,
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
        # print(output[0])
        responses.append(output[0])

        samples = {
            "image": image,
            "prompt": new_query,
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
        # print(output[0])
        new_responses.append(output[0])

    # add responses and new_responses as new columns to the dataframe
    df = pd.read_csv(args.query)
    df['response'] = responses
    df['new_response'] = new_responses
    df.to_csv('{}_responses_{}_{}.csv'.format(args.query.split('/')[-1].split('.')[0], args.model_name, args.model_type), index=False)


if __name__ == "__main__":
    main()