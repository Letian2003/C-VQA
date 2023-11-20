import argparse
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

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
    parser.add_argument("--cfg_path", required=True, help="path to configuration file.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def make_prompt(prompt):
    return "Based on the image, respond to this question with a single word or phrase: "+ prompt

def main():

    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing model')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model Initialization Finished')

    img_paths, queries, new_queries = load_query_file(args.query)
    responses, new_responses = [], []
    for (img_path, query, new_query) in tzip(img_paths, queries, new_queries):
        
        img_path = os.path.join(PATH_TO_IMAGES, img_path)
        # upload image
        chat_state = CONV_VISION_minigptv2.copy()
        img_list = []
        llm_message = chat.upload_img(img_path, chat_state, img_list)
        # print(llm_message)

        q1 = make_prompt(query)

        # ask a question
        chat.ask(q1, chat_state)
        chat.encode_img(img_list)
        # get answer
        response = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]

        responses.append(response)

        # upload image
        chat_state = CONV_VISION_minigptv2.copy()
        img_list = []
        llm_message = chat.upload_img(img_path, chat_state, img_list)
        # print(llm_message)

        q2 = make_prompt(new_query)
        # ask a question
        chat.ask(q2, chat_state)
        chat.encode_img(img_list)
        # get answer
        new_response = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]

        new_responses.append(new_response)

        # print(q1, ":" , response)
        # print(q2, ":" ,new_response)

    # add responses and new_responses as new columns to the dataframe
    df = pd.read_csv(args.query)
    df['response'] = responses
    df['new_response'] = new_responses
    df.to_csv('{}_responses.csv'.format(args.type), index=False)


if __name__ == "__main__":
    main()
