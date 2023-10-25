import runpod
from llava.serve.kwave import main as kwave_main

import torch
import json
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


class Args:
    def __init__(
        self,
        model_path,
        image_file,
        device="cuda",
        conv_mode=None,
        temperature=0.2,
        max_new_tokens=512,
        load_8bit=False,
        load_4bit=False,
        debug=False,
        model_base=None,
        image_aspect_ratio="pad",
        prompt="",
    ):
        self.model_path = model_path
        self.image_file = image_file
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.debug = debug
        self.model_base = model_base
        self.image_aspect_ratio = image_aspect_ratio
        self.prompt = prompt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
args = Args(
    model_path="/var/task/llava-v1.5-7b",
    image_file="",
    prompt="",
    load_4bit=True,
    device=device,
)
model_name = get_model_name_from_path(args.model_path)


disable_torch_init()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path,
    args.model_base,
    model_name,
    args.load_8bit,
    args.load_4bit,
    device=args.device,
)


# runpod handler
def handler(event):
    input_data = event["input"]
    questions = input_data["questions"]
    image = input_data["image"]
    # apply to args
    # args.prompt = prompt
    args.image_file = image
    answers = []

    for question in questions:
        args.prompt = question
        outputs = kwave_main(args, tokenizer, model, image_processor, context_len)
        answers.append(outputs)

    response = {
        "answers": answers,
    }

    return json.dumps(response)


# runpod config.
runpod.serverless.start({"handler": handler})
