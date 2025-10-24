FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y git

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libice6 libxrender1

WORKDIR /workspace
CMD ["bash"]
