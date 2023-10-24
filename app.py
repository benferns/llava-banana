from potassium import Potassium, Request, Response

from transformers import LlavaProcessor, LlavaForCausalLM
from PIL import Image

import requests
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)
    PATH_TO_CONVERTED_WEIGHTS = "shauray/Llava-Llama-2-7B-hf"

    model = LlavaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    processor = LlavaProcessor.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

   
    context = {
        "model": model,
        "processor": processor
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    processor = context.get("processor")
    url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "How can you best describe this image?"

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generate_ids = model.generate(**inputs,
        do_sample=True,
        max_length=1024,
        temperature=0.1,
        top_p=0.9,
    )

    out = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    return Response(
        json = {"outputs": out}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()


"""


PATH_TO_CONVERTED_WEIGHTS = "shauray/Llava-Llama-2-7B-hf"

model = LlavaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
processor = LlavaProcessor.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "How can you best describe this image?"

inputs = processor(text=prompt, images=image, return_tensors="pt")

generate_ids = model.generate(**inputs,
     do_sample=True,
     max_length=1024,
     temperature=0.1,
     top_p=0.9,
 )

out = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    pipeline('fill-mask', model='bert-base-uncased')

if __name__ == "__main__":
    download_model()
    """