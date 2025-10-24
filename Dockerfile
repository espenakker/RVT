FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y git

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /workspace
CMD ["bash"]
