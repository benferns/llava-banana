# This is a potassium-standard dockerfile, compatible with Banana
# Currently we only support python3.8 and the base image defined below at the moment. If you need a different base image or python version please contact https://banana.dev/support

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /var/task

# Install git
RUN apt update && \
    apt-get install -y python3 python3-pip git git-lfs && git lfs install


RUN git clone https://github.com/haotian-liu/LLaVA.git && \
    cd LLaVA

RUN cd LLaVA && pip install --upgrade pip  && \
    pip install -e .

RUN cd /var/task && git clone https://huggingface.co/liuhaotian/llava-v1.5-7b    

ADD ./app.py /var/task/LLaVA

ADD ./kwave.py /var/task/LLaVA/llava/serve/kwave.py

EXPOSE 8000

CMD python3 -u /var/task/LLaVA/app.py
