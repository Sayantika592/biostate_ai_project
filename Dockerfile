# CPU image: song-to-dance with the AIST++ move-stitching engine. No GPU needed.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY . .
RUN bash scripts/setup_data.sh && rm -rf data/motions data/motions.zip
ENV DATA_DIR=/tmp/jobs
WORKDIR /app/server
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
