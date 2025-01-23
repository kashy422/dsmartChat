FROM python:3.11.9-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y espeak-ng ffmpeg alsa-utils

COPY src ./src
COPY requirements.txt /app
COPY history_log.json /app 
COPY tests ./tests
COPY .env .

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Load environment variables from .env file
RUN cat /app/.env | grep -v '#' | xargs -I {} echo export {} >> /root/.bashrc
RUN export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run uvicorn src.api:app --host 0.0.0.0 (localhost) --port 8000 when the container launches
EXPOSE 8507
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8507"]

