docker run --platform=linux/amd64 -it --rm -p 8000:8000 -v $(pwd)/kwave.py:/var/task/LLaVA/llava/serve/kwave.py -v $(pwd)/app.py:/var/task/LLaVA/app.py bferns/kwave-llava-runpod
