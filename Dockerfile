FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
  vim \
  git \
  make

WORKDIR /dvbf

COPY . .

RUN pip install --no-cache-dir -U -r requirements.txt -e .
