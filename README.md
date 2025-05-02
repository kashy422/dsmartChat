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

## Improved Logging

The application now includes an improved logging system that makes the debug logs more insightful and readable while reducing duplicate content.

### Key Features

- Clear section headers for major operations (SYMPTOM ANALYSIS, DOCTOR SEARCH)
- Tool transitions that clearly indicate which component is active (QUERY BUILDER, DATABASE QUERY)
- Performance metrics showing elapsed time for operations
- Highlighted critical information (symptoms, specialties, warnings)
- Reduced duplicate content (SQL queries shown only once)
- Improved visual organization with component names and timing information

### Example Log Output

```
15:56:09.944 INFO     [agent       ] SYMPTOM DETECTION: Checking message for symptoms: 'i am feeling toothace...'

========== STARTING SYMPTOM ANALYSIS [15:56:10.144] ==========
15:56:10.144 INFO     [specialty_matcher] SYMPTOM ANALYSIS: Starting for user description: 'i am feeling toothace...'
15:56:10.195 INFO     [specialty_matcher] SYMPTOM ANALYSIS: Loaded 12 specialty records in 1.20s
15:56:10.196 INFO     [specialty_matcher] SYMPTOM ANALYSIS: Calling OpenAI API to analyze symptoms
15:56:11.967 INFO     [specialty_matcher] SYMPTOM ANALYSIS: Detected 1 symptoms and recommended 1 specialties
15:56:11.968 SUCCESS  [specialty_matcher] SYMPTOM ANALYSIS: Top recommendation - Specialty: Dentistry, Subspecialty: Endodontics, Confidence: 1.0
15:56:11.969 SUCCESS  [agent       ] SYMPTOM PROCESSING: Detected symptoms: toothache
15:56:11.970 SUCCESS  [agent       ] SYMPTOM PROCESSING: Found confident recommendation - Dentistry (subspecialty: Endodontics) with confidence 1.00

Symptom analysis completed in 1.83s

========== STARTING DOCTOR SEARCH [15:56:26.877] ==========
15:56:26.877 INFO     [agent_tools ] Starting dynamic doctor search with message: Find Dentistry specialist in Endodontics in Riyadh
15:56:26.878 INFO     [query_builder] Starting doctor search with query: 'Find Dentistry specialist in Endodontics in Riyadh'

--- QUERY BUILDER [15:56:27.275] ---
15:56:27.276 INFO     [query_builder] Extracted criteria: {'speciality': 'Dentistry', 'subspeciality': 'Endodontics', 'location': 'Riyadh'}
15:56:27.375 SUCCESS  [query_builder] Found subspeciality ID: 2 for Endodontics
15:56:27.376 INFO     [query_builder] Building query with 3 conditions

Query built in 0.69s

--- DATABASE QUERY [15:56:28.065] ---
15:56:28.066 INFO     [db          ] Executing query with parameters: {'p1': '%Dentistry%', 'p2e': 2, 'p3': '%Riyadh%'}
15:56:28.196 INFO     [db          ] Query executed successfully in 0.77 seconds
15:56:28.197 WARNING  [db          ] Found 0 matching doctors
15:56:28.198 INFO     [agent_tools ] Search completed with status: success
Doctor search success in 1.32s
```

The improved logging is automatically initialized when the API server starts.