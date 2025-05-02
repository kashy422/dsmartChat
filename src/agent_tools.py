from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging

from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps

# Import the patient utilities from the new package structure
from .utils import store_patient_details as store_patient_details_util

# Import the specialty matcher functionality
from .specialty_matcher import get_canonical_subspecialty

# Import the query builder functionality
from .query_builder_agent import search_doctors, detect_symptoms_and_specialties

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the database
db = DB()

class StorePatientDetails(BaseModel):
    """Schema for storing patient details"""
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")

def store_patient_details(
    Name: Optional[str] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None
) -> dict:
    """
    Store patient details in the session state for later use
    
    Args:
        Name: Patient's name
        Gender: Patient's gender
        Location: Patient's location
        Issue: Patient's health issue or concern
        session_id: Optional session identifier
        
    Returns:
        Dictionary with stored patient details
    """
    try:
        # Call the utility function with all parameters
        return store_patient_details_util(Name=Name, Gender=Gender, Location=Location, Issue=Issue, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error storing patient details: {str(e)}")
        # Return whatever was passed in, as a fallback
        return {
            "Name": Name,
            "Gender": Gender,
            "Location": Location,
            "Issue": Issue,
            "session_id": session_id,
            "error": str(e)
        }

# Create a StructuredTool for store_patient_details
store_patient_details_tool = StructuredTool.from_function(
    func=store_patient_details,
    name="store_patient_details",
    description="Store basic details of a patient",
    args_schema=StorePatientDetails,
    return_direct=False,
    handle_tool_error="Patient Details Incomplete",
)

def profile(func):
    """Simple profiling decorator to track function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run")
        return result
    return wrapper

@tool(return_direct=False)
@profile
def dynamic_doctor_search(user_query: str) -> Dict[str, Any]:
    """
    Search for doctors based on a natural language query.
    This analyzes the user's input and performs a dynamic search for doctors.
    
    Args:
        user_query: Natural language query like "find a dentist in Riyadh"
        
    Returns:
        Dictionary with doctors array and search metadata
    """
    try:
        result = search_doctors(user_query)
        return result
    except Exception as e:
        logger.error(f"Error in dynamic_doctor_search: {str(e)}")
        # Return a standardized error response
        return {
            "error": str(e),
            "doctors": [],
            "query": user_query
        }

@tool(return_direct=False)
@profile
def analyze_symptoms(symptom_description: str) -> Dict[str, Any]:
    """
    Analyze patient symptoms to determine potential medical specialties.
    This helps guide patients to the right specialist based on their described symptoms.
    
    Args:
        symptom_description: Description of patient symptoms and concerns
        
    Returns:
        Dictionary with specialty recommendations and analysis
    """
    try:
        result = detect_symptoms_and_specialties(symptom_description)
        return result
    except Exception as e:
        logger.error(f"Error in analyze_symptoms: {str(e)}")
        # Return a standardized error response
        return {
            "error": str(e),
            "query": symptom_description,
            "specialties": []
        }
