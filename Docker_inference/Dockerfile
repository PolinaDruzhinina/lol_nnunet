FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

WORKDIR /app

RUN apt-get update && apt-get install python3-pip -y

COPY requirements.txt /app/requirements.txt

RUN pip3 install --quiet --upgrade pip && \
    pip3 install -r /app/requirements.txt && \
    sync

COPY test.py /app/test.py
COPY best_model_allmod_5enc16_b4.pth /app/best_model_allmod_5enc16_b4.pth

#RUN export PATH=”$PATH:/usr/local/bin/python”

RUN chmod a+x /app/test.py

ENTRYPOINT ["python3", "/app/test.py", "--num_encoding_blocks", "5"]
