docker run --platform=linux/amd64 -it --rm -v $(pwd)/kwave.py:/var/task/LLaVA/llava/serve/kwave.py -v $(pwd)/app.py:/var/task/app.py bferns/kwave-llava-banana
