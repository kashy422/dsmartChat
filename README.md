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

## New Features: GPT-powered Doctor Search and Symptom Analysis

The application now includes intelligent doctor search and symptom analysis capabilities powered by OpenAI's GPT models.

### Doctor Search API

Use natural language queries to search for doctors based on multiple criteria:

```
POST /search-doctors
{
    "query": "Show me dentists in Riyadh with 3+ ratings and at least 5 years experience"
}
```

The search supports criteria like:
- Doctor specialty (e.g., dentist, cardiologist)
- Location/city
- Rating requirements
- Price/fee limits
- Experience requirements
- Hospital/clinic name

### Symptom Analysis API

Get specialty recommendations based on symptom descriptions:

```
POST /analyze-symptoms
{
    "symptoms": "I have a persistent toothache and my gums are swollen"
}
```

This endpoint analyzes symptoms and recommends appropriate medical specialties to consult.

### Example User Queries

Users can search for doctors using queries like:
- "Show me dentists in Riyadh"
- "Find cardiologists with 5+ years experience in Jeddah"
- "Dermatologists with good ratings in Riyadh"
- "I need a doctor who charges less than 400 SAR for consultation"

## Environment Variables

Make sure to set the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DB_HOST`: Database hostname
- `DB_DATABASE`: Database name
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password

## Testing

Run the test script to verify functionality:

```commandline
>> python -m src.test_query_builder
```