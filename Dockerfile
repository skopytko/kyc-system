FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps for OpenCV and basic runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (keep torch/torchvision from base image)
COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# Copy repo
COPY . /app

ENV PYTHONPATH=/app
ENV KYC_HOST=0.0.0.0
ENV KYC_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "webapp.main"]

