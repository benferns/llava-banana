import runpod
import time
import json
from os import environ
from

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests


# runpod handler
def handler(event):
    prompt = event["prompt"]
    image = event["image"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args(
        model_path="/var/task/llava-v1.5-7b",
        external_prompt=prompt,
        image_file=image,
        load_4bit=True,
        device=device,
    )

    output = kwave_main(args, prompt)

    response = {
        "output": output,
    }
    return json.dumps(response)


# runpod config.
runpod.serverless.start({"handler": handler})
