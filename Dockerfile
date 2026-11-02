# Floor / obstacle segmentation learning block.
# Studio runs it without network access: everything, including the
# pretrained weights, is in the image.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SEG_WEIGHTS_DIR=/app/weights

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY seg ./seg
COPY scripts/fetch_weights.py ./scripts/
RUN python3 scripts/fetch_weights.py /app/weights

COPY train.py parameters.json ./

ENTRYPOINT ["python3", "-u", "train.py"]
