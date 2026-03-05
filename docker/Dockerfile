# use a smaller footprint
FROM python:3.11-slim

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# set work directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy only the dependency files first to leverage Docker cache
COPY pyproject.toml .
# create a dummy src/newsclf 
RUN mkdir -p src/newsclf && touch src/newsclf/__init__.py

# install dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# download the specific spaCy model
RUN python -m spacy download en_core_web_sm

# copy the rest of the application code
COPY . .

# expose the port FastAPI runs on
EXPOSE 7860

# command to run the API
CMD ["uvicorn", "newsclf.api.main:app", "--host", "0.0.0.0", "--port", "7860"]