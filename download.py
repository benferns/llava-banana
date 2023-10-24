from transformers import LlavaForCausalLM, LlavaProcessor
import torch

PATH_TO_CONVERTED_WEIGHTS = "shauray/Llava-Llama-2-7B-hf"


def download_model():
    kwargs = {"device_map": "auto"}
    kwargs["torch_dtype"] = torch.float16
    # set text model type to llava
    

    # do a dry run of loading the huggingface model, which will download weights    
    model = LlavaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS,  low_cpu_mem_usage=True, **kwargs)
    
    processor = LlavaProcessor.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    #model.save_pretrained("Llava-Llama-2-7B-hf")

if __name__ == "__main__":
    download_model()