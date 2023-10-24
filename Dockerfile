# This is a potassium-standard dockerfile, compatible with Banana
# Currently we only support python3.8 and the base image defined below at the moment. If you need a different base image or python version please contact https://banana.dev/support

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt update && \
    apt-get install -y python3 python3-pip git git-lfs && git lfs install

# Install python packages.
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install git+https://github.com/shauray8/transformers.git@llava#egg=transformers


# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py
# RUN git clone https://huggingface.co/liuhaotian/llava-v1.5-13b

ADD . .

EXPOSE 8000

CMD python3 -u app.py
