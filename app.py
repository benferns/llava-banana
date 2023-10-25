from potassium import Potassium, Request, Response
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


app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    # create empty dict
    return {}


# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    image = request.json.get("image")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args(
        model_path="/var/task/llava-v1.5-7b",
        image_file=image,
        load_4bit=True,
        device=device,
    )

    output = kwave_main(args, prompt)

    return Response(json={"outputs": output}, status=200)


if __name__ == "__main__":
    app.serve()
