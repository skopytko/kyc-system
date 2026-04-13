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
COPY environment.yml /app/environment.yml
RUN python - <<'PY'
import re, sys
from pathlib import Path

txt = Path("environment.yml").read_text(encoding="utf-8")
# crude parse: take pip list under "- pip:" indentation
lines = txt.splitlines()
pip = []
in_pip = False
for ln in lines:
    if re.match(r"^\\s*-\\s*pip:\\s*$", ln):
        in_pip = True
        continue
    if in_pip:
        m = re.match(r"^\\s*-\\s*(.+?)\\s*$", ln)
        if m:
            pip.append(m.group(1).strip())
        else:
            # end of pip block
            if ln.strip() and not ln.lstrip().startswith("#"):
                in_pip = False

# Remove torch packages (already in base image)
skip = {"torch", "torchvision"}
out = []
for dep in pip:
    name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()
    if name in skip:
        continue
    out.append(dep)

Path("/app/requirements.docker.txt").write_text("\\n".join(out) + "\\n", encoding="utf-8")
print("Wrote requirements.docker.txt with", len(out), "deps")
PY

RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# Copy repo
COPY . /app

ENV PYTHONPATH=/app
ENV KYC_HOST=0.0.0.0
ENV KYC_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "webapp.main"]

