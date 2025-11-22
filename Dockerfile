# 1. Base Image: Official Python 3.10 "Slim" version (Smaller & Faster)
FROM python:3.10-slim

# 2. Set Environment Variables
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies for OpenCV
# OpenCV needs these low-level graphics libraries to run
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Work Directory
WORKDIR /app

# 5. Install Python Dependencies
# We copy just the requirements first to leverage Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the Project Files
# We copy everything else (src, models, etc.)
COPY . .

# 7. Default Command
# When the container starts, it runs your inference script
CMD ["python", "src/inference.py"]