FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (~800MB instead of 4GB+)
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# NeMo ASR + server deps
RUN pip install --no-cache-dir \
    nemo_toolkit[asr]==2.0.0 \
    fastapi uvicorn python-multipart soundfile "numpy<2" matplotlib

WORKDIR /app
COPY server.py .

# Download model at build time so it's baked into the image
RUN python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')"

ENV DEVICE=cpu
ENV PORT=9200

EXPOSE 9200
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9200"]
