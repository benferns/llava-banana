# This is a potassium-standard dockerfile, compatible with Banana
# Currently we only support python3.8 and the base image defined below at the moment. If you need a different base image or python version please contact https://banana.dev/support

FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

WORKDIR /var/task

# Install git
RUN apt update && \
    apt-get install -y python3 python3-pip git git-lfs && git lfs install


RUN git clone https://github.com/haotian-liu/LLaVA.git 

RUN cd LLaVA && pip install --upgrade pip  && \
    pip install -e .

#RUN cd /var/task && git clone https://huggingface.co/liuhaotian/llava-v1.5-7b 

COPY /task/llava-v1.5-7b /var/task/llava-v1.5-7b

COPY ./requirements.txt /var/task/requirements.txt

RUN cd /var/task/ && pip install -r --ignore-installed requirements.txt

COPY ./runpod_handler.py /var/task/LLaVA

COPY ./kwave.py /var/task/LLaVA/llava/serve/kwave.py

EXPOSE 8000

CMD [ "python3", "-u", "/var/task/LLaVA//runpod_handler.py" ]
