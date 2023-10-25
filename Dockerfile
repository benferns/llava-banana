FROM bferns/kwave-llava-runpod-base:latest

COPY ./requirements.txt /var/task/requirements.txt

RUN cd /var/task/ && pip install -r --ignore-installed requirements.txt

COPY ./runpod_handler.py /var/task/LLaVA

COPY ./kwave.py /var/task/LLaVA/llava/serve/kwave.py

EXPOSE 8000

CMD [ "python3", "-u", "/var/task/LLaVA//runpod_handler.py" ]
