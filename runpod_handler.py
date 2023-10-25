import runpod
from llava.serve.kwave import main as kwave_main

import torch


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
