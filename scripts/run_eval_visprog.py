import argparse

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import htmlmin

from PIL import Image
from IPython.core.display import HTML
from functools import partial

from engine.utils import ProgramGenerator, ProgramInterpreter
from prompts.gqa import create_prompt
from tqdm.contrib import tzip

def load_query_file(query_file):
    df = pd.read_csv(query_file)
    img_paths = df['img_path'].tolist()
    queries = df['query'].tolist()
    new_queries = df['new query'].tolist()
    assert len(img_paths) == len(queries) == len(new_queries)
    return img_paths, queries, new_queries

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    interpreter = ProgramInterpreter(dataset='gqa')
    prompter = partial(create_prompt,method='all')
    generator = ProgramGenerator(prompter=prompter)

    img_paths, queries, new_queries = load_query_file(args.query)
    responses, new_responses = [], []
    progs, new_progs = [], []
    html_strs, new_html_strs = [], []
    for (img_path, query, new_query) in tzip(img_paths, queries, new_queries):
        img = Image.open(PATH_TO_IMAGES + img_path)
        init_state = dict(
            IMAGE=img.convert('RGB')
        )
        
        result, new_result = "failed", "failed"
        prog, new_prog = "n/a", "n/a"
        html_str, new_html_str = "n/a", "n/a"

        try:
            prog = generator.generate(dict(question=query))
            result, prog_state, html_str = interpreter.execute(prog,init_state,inspect=True)
        except:
            pass

        responses.append(result)
        progs.append(prog.replace('\n',' '))
        html_strs.append(htmlmin.minify(html_str))
        
        try:
            new_prog = generator.generate(dict(question=new_query))
            new_result, new_prog_state, new_html_str = interpreter.execute(new_prog,init_state,inspect=True)
        except:
            pass
        
        new_responses.append(new_result)
        new_progs.append(new_prog.replace('\n',' '))
        new_html_strs.append(htmlmin.minify(new_html_str))

    # add responses and new_responses as new columns to the dataframe
    df = pd.read_csv(args.query)
    df['response'] = responses
    df['new_response'] = new_responses
    df['prog'] = progs
    df['new_prog'] = new_progs
    df['html_str'] = html_strs
    df['new_html_str'] = new_html_strs
    df.to_csv('{}_responses_visprog.csv'.format(args.query.split('/')[-1].split('.')[0]), index=False)


if __name__ == "__main__":
    main()