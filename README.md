# Medical Chatbot with CA

## Cloning the Repository and Switching to the dev Branch
>The `dev` branch has the code now. 

```commandline
>> git clone https://github.com/Hassibayub/medical-chatbot-with-ca--Kashif.git
>> cd medical-chatbot-with-ca--Kashif
>> git switch dev
```

## Install Docker on your machine
Follow the instructions on the [official Docker website](https://docs.docker.com/get-docker/).

## Run the Docker Container
```commandline
>> docker-compose up --build
>> docker-compose up --build -d # To run in detached mode
```

## Run the uvicorn
```commandline
>> activate venv
>> uvicorn src.api:app --host 0.0.0.0 --port 8507 --reload
```