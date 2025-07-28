from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import logging
import uuid
import json
import threading

from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps

# Import the patient utilities from the new package structure
from .utils import store_patient_details as store_patient_details_util, thread_local, clear_symptom_analysis_data

# Import the specialty matcher functionality
from .specialty_matcher import detect_symptoms_and_specialties

# Import the query builder functionality
from .query_builder_agent import (
    unified_doctor_search_tool, 
    extract_search_criteria_tool, 
    normalize_specialty,
    extract_search_criteria_from_message,
    unified_doctor_search,
    extract_processed_criteria
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the database
db = DB()

def execute_offers_search(search_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the offers stored procedure in parallel with doctor search
    
    Args:
        search_criteria: Dictionary containing search criteria
        
    Returns:
        Dictionary containing offers search results
    """
    try:
        logger.info("🎁 Starting offers search with criteria: %s", search_criteria)
        
        # Debug: Print the structure of search_criteria to identify any nested dictionaries
        logger.info("🎁 DEBUG: search_criteria structure:")
        for key, value in search_criteria.items():
            logger.info(f"🎁   {key}: {value} (type: {type(value)})")
            if isinstance(value, dict):
                logger.warning(f"🎁   WARNING: {key} is a dictionary: {value}")
        
        # Extract parameters for the offers stored procedure
        # Handle both direct criteria and criteria from natural language processing
        speciality = search_criteria.get('speciality') or search_criteria.get('specialty')
        subspeciality = search_criteria.get('subspeciality') or search_criteria.get('subspecialty')
        latitude = search_criteria.get('latitude')
        longitude = search_criteria.get('longitude')
        price = search_criteria.get('max_price') or search_criteria.get('min_price') or search_criteria.get('price')
        branch_name = search_criteria.get('branch_name') or search_criteria.get('hospital_name')
        
        # Log the extracted branch name for debugging
        if branch_name:
            logger.info(f"🎁 Extracted branch name: '{branch_name}'")
        else:
            logger.info(f"🎁 No branch name found in search criteria")
        
        # Convert "dentist" to "Dentistry" for the offers stored procedure
        if speciality and speciality.lower() == "dentist":
            speciality = "Dentistry"
            logger.info(f"🎁 Converted specialty from 'dentist' to 'Dentistry' for offers stored procedure")
        
        # Ensure all values are properly converted to strings or None
        # Handle cases where values might be dictionaries or other complex objects
        def safe_convert(value):
            if value is None:
                return None
            elif isinstance(value, (str, int, float)):
                return str(value)
            elif isinstance(value, dict):
                # If it's a dictionary, try to extract a meaningful string representation
                logger.warning(f"🎁 Converting dictionary to string: {value}")
                return str(value)
            else:
                # For any other type, convert to string
                logger.warning(f"🎁 Converting {type(value)} to string: {value}")
                return str(value)
        
        # Build parameters for the stored procedure
        params = {}
        
        if speciality:
            params['@speciality'] = safe_convert(speciality)
        else:
            params['@speciality'] = None
            
        if subspeciality:
            params['@subspeciality'] = safe_convert(subspeciality)
        else:
            params['@subspeciality'] = None
            
        if latitude and longitude:
            params['@lat'] = safe_convert(latitude)
            params['@long'] = safe_convert(longitude)
        else:
            params['@lat'] = None
            params['@long'] = None
            
        if price:
            params['@price'] = safe_convert(price)
        else:
            params['@price'] = None
            
        if branch_name:
            params['@branchName'] = safe_convert(branch_name)
        else:
            params['@branchName'] = None
        
        logger.info("🎁 Executing offers stored procedure with params: %s", params)
        
        # Log each parameter individually for debugging
        logger.info("🎁 DEBUG: Individual parameters being passed to offers stored procedure:")
        for param_name, param_value in params.items():
            logger.info(f"🎁   {param_name}: {param_value} (type: {type(param_value)})")
        
        # Execute the stored procedure
        offers_result = db.execute_stored_procedure(
            "[dbo].[Sp_GetOffersBySpecialityAndLocation]", 
            params
        )
        
        logger.info("🎁 Offers search completed successfully")
        
        # Debug: Print the raw database result structure
        print(f"\n🎁 RAW DATABASE RESULT STRUCTURE:")
        print(f"Type: {type(offers_result)}")
        if isinstance(offers_result, dict):
            print(f"Keys: {list(offers_result.keys())}")
            for key, value in offers_result.items():
                print(f"  {key}: {type(value)} - {len(value) if isinstance(value, list) else 'not a list'}")
                if isinstance(value, list) and value:
                    print(f"    First item: {value[0]}")
        else:
            print(f"Value: {offers_result}")
        print("🎁 END RAW DATABASE RESULT STRUCTURE\n")
        logger.info("🎁 Offers result structure: %s", list(offers_result.keys()) if isinstance(offers_result, dict) else "Not a dict")
        
        # Debug: Print the exact offers_result for debugging
        logger.info(f"🎁 DEBUG: Exact offers_result: {offers_result}")
        if isinstance(offers_result, dict):
            for key, value in offers_result.items():
                logger.info(f"🎁 DEBUG: Key '{key}': {type(value)} - {value}")
                if isinstance(value, list) and value:
                    logger.info(f"🎁 DEBUG: List '{key}' has {len(value)} items")
                    if isinstance(value[0], dict):
                        logger.info(f"🎁 DEBUG: First item in '{key}': {value[0]}")
                        logger.info(f"🎁 DEBUG: Keys in first item: {list(value[0].keys())}")
        else:
            logger.info(f"🎁 DEBUG: offers_result is not a dict, type: {type(offers_result)}")
            logger.info(f"🎁 DEBUG: offers_result value: {offers_result}")
        
        # Store the raw offers_result for potential fallback extraction
        raw_offers_result = offers_result
        
        # Print offers results to terminal as requested
        print("\n" + "="*80)
        print("🎁 OFFERS SEARCH RESULTS")
        print("="*80)
        print(f"Speciality: {speciality}")
        print(f"Subspeciality: {subspeciality}")
        print(f"Location: lat={latitude}, long={longitude}")
        print(f"Price: {price}")
        print(f"Branch Name: {branch_name}")
        print("-"*80)
        
        # Debug: Print the full offers_result structure
        logger.info(f"🎁 DEBUG: Full offers_result structure: {list(offers_result.keys()) if isinstance(offers_result, dict) else 'Not a dict'}")
        
        offers_data = []
        if isinstance(offers_result, dict) and 'data' in offers_result:
            offers_data_dict = offers_result.get('data', {})
            logger.info(f"🎁 DEBUG: offers_data_dict type: {type(offers_data_dict)}")
            logger.info(f"🎁 DEBUG: offers_data_dict keys: {list(offers_data_dict.keys()) if isinstance(offers_data_dict, dict) else 'Not a dict'}")
            
            if isinstance(offers_data_dict, dict) and 'doctors' in offers_data_dict:
                # Check if the doctors list contains offer data (has OfferId, OfferName, etc.)
                doctors_list = offers_data_dict['doctors']
                if doctors_list and isinstance(doctors_list, list) and len(doctors_list) > 0:
                    first_item = doctors_list[0]
                    if isinstance(first_item, dict) and 'OfferId' in first_item:
                        # This is offers data, not doctors data
                        offers_data = doctors_list
                        logger.info(f"🎁 DEBUG: Found {len(offers_data)} offers in data.doctors (these are actually offers)")
                        logger.info(f"🎁 DEBUG: First offer: {first_item.get('OfferName_en', 'Unknown')} (ID: {first_item.get('OfferId', 'Unknown')})")
                    else:
                        logger.info(f"🎁 DEBUG: Found {len(doctors_list)} doctors in data.doctors (not offers)")
                else:
                    logger.info(f"🎁 DEBUG: No doctors found in data.doctors")
            elif isinstance(offers_data_dict, list):
                # If data is directly a list, use it as offers
                offers_data = offers_data_dict
                logger.info(f"🎁 DEBUG: Found {len(offers_data)} offers in data (direct list)")
            elif isinstance(offers_data_dict, dict):
                # If data is a dict but doesn't have 'doctors', check if it has any other keys
                logger.info(f"🎁 DEBUG: Data is dict but no 'doctors' key. Available keys: {list(offers_data_dict.keys())}")
                # Try to find any list in the data that might be offers
                for key, value in offers_data_dict.items():
                    if isinstance(value, list):
                        offers_data = value
                        logger.info(f"🎁 DEBUG: Using {key} as offers data: {len(offers_data)} items")
                        break
            else:
                logger.warning("🎁 DEBUG: Offers data structure not as expected")
                logger.warning(f"🎁 DEBUG: offers_data_dict: {offers_data_dict}")
        elif isinstance(offers_result, dict):
            # If no 'data' key, check if the result itself contains offers
            logger.info(f"🎁 DEBUG: No 'data' key in offers_result. Available keys: {list(offers_result.keys())}")
            # Look for any list in the result that might be offers
            for key, value in offers_result.items():
                if isinstance(value, list):
                    offers_data = value
                    logger.info(f"🎁 DEBUG: Using {key} as offers data: {len(offers_data)} items")
                    break
        else:
            logger.warning("🎁 DEBUG: Offers result is not a dict or doesn't contain 'data'")
            logger.warning(f"🎁 DEBUG: offers_result type: {type(offers_result)}")
        
        # Since we're seeing offers printed in the terminal, let's check if the offers_result
        # itself contains the offers data directly
        if not offers_data and isinstance(offers_result, dict):
            logger.info(f"🎁 DEBUG: Checking if offers_result contains offers directly")
            logger.info(f"🎁 DEBUG: offers_result keys: {list(offers_result.keys())}")
            # Look for any key that might contain offers data
            for key, value in offers_result.items():
                if isinstance(value, list) and value:
                    # Check if this looks like offers data (has OfferId, OfferName, etc.)
                    if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                        offers_data = value
                        logger.info(f"🎁 DEBUG: Found offers data in key '{key}': {len(offers_data)} items")
                        break
        
        # If still no offers_data, try to extract from the raw offers_result
        if not offers_data and isinstance(offers_result, dict):
            logger.info(f"🎁 DEBUG: Final attempt to extract offers from raw offers_result")
            # Look for any list in the entire offers_result structure
            def find_offers_in_dict(data, path=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_path = f"{path}.{key}" if path else key
                        if isinstance(value, list) and value:
                            # Check if this looks like offers data
                            if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                                logger.info(f"🎁 DEBUG: Found offers in {current_path}: {len(value)} items")
                                return value
                        elif isinstance(value, dict):
                            result = find_offers_in_dict(value, current_path)
                            if result:
                                return result
                return None
            
            offers_data = find_offers_in_dict(offers_result)
            if offers_data:
                logger.info(f"🎁 DEBUG: Successfully extracted {len(offers_data)} offers from raw offers_result")
        
        if offers_data:
            print(f"Found {len(offers_data)} offers:")
            for i, offer in enumerate(offers_data, 1):
                print(f"  {i}. {offer}")
            print(f"✅ These offers will be included in the API response")
            logger.info(f"🎁 DEBUG: offers_data length: {len(offers_data)}")
            logger.info(f"🎁 DEBUG: offers_data type: {type(offers_data)}")
            if offers_data:
                logger.info(f"🎁 DEBUG: First offer sample: {offers_data[0]}")
        else:
            print("No offers data found in expected format")
            logger.warning("🎁 DEBUG: offers_data is empty or None")
            logger.warning(f"🎁 DEBUG: offers_data value: {offers_data}")
            
            # Final fallback: if we have raw_offers_result but no offers_data,
            # try to extract offers from the raw result
            if raw_offers_result and isinstance(raw_offers_result, dict):
                logger.info(f"🎁 DEBUG: Attempting final fallback extraction from raw_offers_result")
                # Look for any list that might contain offers
                for key, value in raw_offers_result.items():
                    if isinstance(value, list) and value:
                        # Check if this looks like offers data
                        if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                            offers_data = value
                            logger.info(f"🎁 DEBUG: Fallback found {len(offers_data)} offers in key '{key}'")
                            print(f"🎁 Fallback: Found {len(offers_data)} offers in raw result")
                            break
        
        print("="*80 + "\n")
        
        # Return both doctors and offers in a unified structure
        # This ensures both doctors and offers are at the same level
        if offers_data:
            logger.info(f"🎁 [execute_offers_search] RETURNING: {len(offers_data)} offers in unified structure")
            logger.info(f"🎁 [execute_offers_search] RETURN DATA: {{'data': {{'doctors': [], 'offers': {len(offers_data)} items}}}}")
            result_data = {
                "data": {
                    "doctors": [],
                    "offers": offers_data
                }
            }
            logger.info(f"🎁 [execute_offers_search] FINAL RETURN: {result_data}")
            return result_data
        else:
            # Final fallback: if we still don't have offers_data but we have raw_offers_result,
            # and it contains offers (as evidenced by the terminal print), let's try to extract them
            logger.info("🎁 [execute_offers_search] No offers_data found, but offers were printed to terminal")
            logger.info("🎁 [execute_offers_search] Attempting to extract offers from raw_offers_result for return")
            
            # If raw_offers_result is a dict, look for any list that might be offers
            if isinstance(raw_offers_result, dict):
                for key, value in raw_offers_result.items():
                    if isinstance(value, list) and value:
                        # Check if this looks like offers data
                        if isinstance(value[0], dict) and any(field in value[0] for field in ['OfferId', 'OfferName', 'BeforePrice', 'AfterPrice']):
                            logger.info(f"🎁 [execute_offers_search] Fallback: Found {len(value)} offers in key '{key}' for return")
                            logger.info(f"🎁 [execute_offers_search] RETURNING: {len(value)} offers from fallback")
                            result_data = {
                                "data": {
                                    "doctors": [],
                                    "offers": value
                                }
                            }
                            logger.info(f"🎁 [execute_offers_search] FALLBACK RETURN: {result_data}")
                            return result_data
            
            logger.info("🎁 [execute_offers_search] No offers data found, returning empty structure")
            logger.info("🎁 [execute_offers_search] RETURN DATA: {{'data': {{'doctors': [], 'offers': []}}}}")
            result_data = {
                "data": {
                    "doctors": [],
                    "offers": []
                }
            }
            logger.info(f"🎁 [execute_offers_search] EMPTY RETURN: {result_data}")
            return result_data
        
    except Exception as e:
        logger.error("🎁 Error in offers search: %s", str(e), exc_info=True)
        return {"error": str(e), "offers": []}

class StorePatientDetails(BaseModel):
    """Schema for storing patient details"""
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Age: Optional[int] = Field(default=None, description="Age of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")

class AnalyzeSymptomInput(BaseModel):
    """Schema for symptom analysis input"""
    symptom_description: str = Field(description="Description of symptoms or health concerns")

def store_patient_details(
    Name: Optional[str] = None,
    Age: Optional[int] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None
) -> dict:
    """
    Store patient details in the session state for later use
    
    Args:
        Name: Patient's name
        Age: Patient's age
        Gender: Patient's gender
        Location: Patient's location
        Issue: Patient's health issue or concern
        session_id: Optional session identifier
        
    Returns:
        Dictionary with stored patient details
    """
    try:
        # Call the utility function with all parameters
        return store_patient_details_util(Name=Name, Age=Age, Gender=Gender, Location=Location, Issue=Issue, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error storing patient details: {str(e)}")
        # Return whatever was passed in, as a fallback
        return {
            "Name": Name,
            "Age": Age,
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
def dynamic_doctor_search(search_query: Union[str, dict]) -> dict:
    """
    Search for doctors based on various criteria.
    
    Args:
        search_query: Either a JSON string with search criteria or a natural language query
            
    Returns:
        Dictionary with search results
    """
    logger.info(f"DOCTOR SEARCH: Using session ID: {getattr(thread_local, 'session_id', 'unknown')}")
    
    # Clear previous symptom analysis data to avoid contamination
    logger.info(f"DOCTOR SEARCH: Clearing previous symptom analysis data")
    clear_symptom_analysis_data("starting new direct search")
    
    try:
        # Extract search criteria for both doctor search and offers search
        search_criteria = None
        processed_criteria = None
        
        # If the search_query is a string, try to parse as JSON first
        if isinstance(search_query, str):
            try:
                criteria_dict = json.loads(search_query)
                logger.info(f"Detected structured criteria in JSON format: {criteria_dict}")
                search_criteria = criteria_dict
                
                # Extract specialty information if available
                if "speciality" in criteria_dict:
                    logger.info(f"Using specialty: {criteria_dict['speciality']}")
                
                # Extract coordinates
                if "latitude" in criteria_dict and "longitude" in criteria_dict:
                    thread_local.latitude = criteria_dict["latitude"]
                    thread_local.longitude = criteria_dict["longitude"]
                    logger.info(f"Using coordinates from search params: lat={criteria_dict['latitude']}, long={criteria_dict['longitude']}")
                
                logger.info(f"Calling unified_doctor_search with criteria: {criteria_dict}")
                result = unified_doctor_search(criteria_dict)
                
                # Extract processed criteria from the result or use original criteria
                processed_criteria = criteria_dict
                
            except json.JSONDecodeError:
                # If not valid JSON, treat as natural language query
                logger.info(f"Processing as natural language query: {search_query}")
                
                # Extract coordinates if available in thread_local
                coordinates = {}
                if hasattr(thread_local, 'latitude') and hasattr(thread_local, 'longitude'):
                    coordinates = {
                        "latitude": thread_local.latitude,
                        "longitude": thread_local.longitude
                    }
                    logger.info(f"Using coordinates from thread_local: lat={thread_local.latitude}, long={thread_local.longitude}")
                
                # Create parameters for unified search
                search_params = {
                    "user_message": search_query,
                    **coordinates
                }
                search_criteria = search_params
                
                logger.info(f"Calling unified_doctor_search with text and coordinates: {search_params}")
                result = unified_doctor_search(search_params)
                
                # For natural language queries, we need to extract the processed criteria
                # The unified_doctor_search processes the criteria but doesn't return it
                # So we'll extract it from the thread_local or use the original
                processed_criteria = search_params
        else:
            # Handle direct dictionary input
            logger.info(f"Using provided dictionary: {search_query}")
            search_criteria = search_query
            result = unified_doctor_search(search_query)
            processed_criteria = search_query
        
        # Execute offers search in parallel if we have search criteria
        offers_result = None
        if processed_criteria:
            logger.info("🎁 Starting parallel offers search with processed criteria")
            logger.info(f"🎁 Processed criteria: {processed_criteria}")
            
            # Extract the properly processed criteria using the helper function
            # This ensures we get the same processed criteria that unified_doctor_search uses
            final_processed_criteria = extract_processed_criteria(processed_criteria)
            logger.info(f"🎁 Final processed criteria for offers: {final_processed_criteria}")
            
            # Create a thread for offers search with better error handling
            # Use a shared variable instead of thread_local for better communication
            shared_offers_result = [None]  # Use list to make it mutable
            
            def execute_offers_in_thread():
                try:
                    logger.info(f"🎁 [Thread] Starting execute_offers_search with criteria: {final_processed_criteria}")
                    result = execute_offers_search(final_processed_criteria)
                    logger.info(f"🎁 [Thread] execute_offers_search returned: {type(result)}")
                    if isinstance(result, dict):
                        logger.info(f"🎁 [Thread] Result keys: {list(result.keys())}")
                        for key, value in result.items():
                            logger.info(f"🎁 [Thread] {key}: {type(value)} - {value}")
                            if isinstance(value, list) and value:
                                logger.info(f"🎁 [Thread] {key} has {len(value)} items")
                                if isinstance(value[0], dict):
                                    logger.info(f"🎁 [Thread] First item keys: {list(value[0].keys())}")
                    
                    shared_offers_result[0] = result  # Store in shared variable
                    logger.info(f"🎁 [Thread] Successfully stored result in shared_offers_result")
                    logger.info(f"🎁 [Thread] shared_offers_result[0] type: {type(shared_offers_result[0])}")
                except Exception as e:
                    logger.error(f"🎁 [Thread] Error in offers search: {str(e)}")
                    shared_offers_result[0] = {"data": {"doctors": [], "offers": []}}
            
            offers_thread = threading.Thread(target=execute_offers_in_thread)
            offers_thread.start()
            offers_thread.join()  # Wait for completion
            
            # Get the offers result from shared variable
            offers_result = shared_offers_result[0]
            logger.info(f"🎁 [dynamic_doctor_search] Retrieved offers_result from shared variable: {type(offers_result)}")
            
            # Debug: Print the exact offers_result content
            if offers_result is None:
                logger.error("🎁 [dynamic_doctor_search] ERROR: offers_result is None!")
            elif isinstance(offers_result, dict):
                logger.info(f"🎁 [dynamic_doctor_search] offers_result keys: {list(offers_result.keys())}")
                for key, value in offers_result.items():
                    logger.info(f"🎁 [dynamic_doctor_search] {key}: {type(value)} - {value}")
                    if isinstance(value, list) and value:
                        logger.info(f"🎁 [dynamic_doctor_search] {key} has {len(value)} items")
                        if isinstance(value[0], dict):
                            logger.info(f"🎁 [dynamic_doctor_search] First item keys: {list(value[0].keys())}")
            else:
                logger.warning(f"🎁 [dynamic_doctor_search] offers_result is not a dict: {offers_result}")
            
            logger.info("🎁 [dynamic_doctor_search] Parallel offers search completed")
            logger.info(f"🎁 [dynamic_doctor_search] RECEIVED offers_result: {type(offers_result)}")
            if isinstance(offers_result, dict):
                logger.info(f"🎁 [dynamic_doctor_search] offers_result keys: {list(offers_result.keys())}")
                if 'offers' in offers_result:
                    logger.info(f"🎁 [dynamic_doctor_search] offers_result['offers']: {len(offers_result['offers'])} items")
                if 'data' in offers_result:
                    logger.info(f"🎁 [dynamic_doctor_search] offers_result['data']: {type(offers_result['data'])}")
            else:
                logger.info(f"🎁 [dynamic_doctor_search] offers_result value: {offers_result}")
            
            # Add offers results to the result BEFORE formatting
            if offers_result:
                logger.info("🎁 [dynamic_doctor_search] Adding offers results to result before formatting")
                logger.info(f"🎁 [dynamic_doctor_search] offers_result type: {type(offers_result)}")
                logger.info(f"🎁 [dynamic_doctor_search] offers_result keys: {list(offers_result.keys()) if isinstance(offers_result, dict) else 'Not a dict'}")
                
                # Extract offers data from the offers_result
                offers_data = []
                logger.info(f"🎁 [dynamic_doctor_search] Starting offers extraction from offers_result")
                logger.info(f"🎁 [dynamic_doctor_search] offers_result type: {type(offers_result)}")
                logger.info(f"🎁 [dynamic_doctor_search] offers_result keys: {list(offers_result.keys()) if isinstance(offers_result, dict) else 'Not a dict'}")
                
                # Extract offers from the new unified structure
                if isinstance(offers_result, dict) and 'offers' in offers_result:
                    offers_data = offers_result['offers']
                    logger.info(f"🎁 [dynamic_doctor_search] Found {len(offers_data)} offers directly in offers_result['offers']")
                    logger.info(f"🎁 [dynamic_doctor_search] First offer sample: {offers_data[0] if offers_data else 'No offers'}")
                elif isinstance(offers_result, dict) and 'data' in offers_result:
                    offers_data_dict = offers_result.get('data', {})
                    logger.info(f"🎁 [dynamic_doctor_search] offers_data_dict type: {type(offers_data_dict)}")
                    logger.info(f"🎁 [dynamic_doctor_search] offers_data_dict keys: {list(offers_data_dict.keys()) if isinstance(offers_data_dict, dict) else 'Not a dict'}")
                    
                    if isinstance(offers_data_dict, dict) and 'offers' in offers_data_dict:
                        # Extract offers from data.offers
                        offers_data = offers_data_dict['offers']
                        logger.info(f"🎁 [dynamic_doctor_search] Found {len(offers_data)} offers in data.offers")
                        logger.info(f"🎁 [dynamic_doctor_search] First offer: {offers_data[0].get('OfferName_en', 'Unknown') if offers_data else 'No offers'} (ID: {offers_data[0].get('OfferId', 'Unknown') if offers_data else 'Unknown'})")
                    elif isinstance(offers_data_dict, dict) and 'doctors' in offers_data_dict:
                        # Check if the doctors list contains offer data (has OfferId, OfferName, etc.)
                        doctors_list = offers_data_dict['doctors']
                        if doctors_list and isinstance(doctors_list, list) and len(doctors_list) > 0:
                            first_item = doctors_list[0]
                            if isinstance(first_item, dict) and 'OfferId' in first_item:
                                # This is offers data, not doctors data - extract it as offers
                                offers_data = doctors_list
                                logger.info(f"🎁 [dynamic_doctor_search] Found {len(offers_data)} offers in data.doctors (these are actually offers)")
                                logger.info(f"🎁 [dynamic_doctor_search] First offer: {first_item.get('OfferName_en', 'Unknown')} (ID: {first_item.get('OfferId', 'Unknown')})")
                            else:
                                logger.info(f"🎁 [dynamic_doctor_search] Found {len(doctors_list)} doctors in data.doctors (not offers)")
                        else:
                            logger.info(f"🎁 [dynamic_doctor_search] No doctors found in data.doctors")
                    else:
                        logger.warning("🎁 [dynamic_doctor_search] Offers data structure not as expected")
                        logger.warning(f"🎁 [dynamic_doctor_search] offers_data_dict: {offers_data_dict}")
                        # Try alternative extraction methods
                        if isinstance(offers_data_dict, list):
                            offers_data = offers_data_dict
                            logger.info(f"🎁 [dynamic_doctor_search] Using offers_data_dict directly as list: {len(offers_data)} offers")
                        elif isinstance(offers_data_dict, dict):
                            # Look for any list in the data that might be offers
                            for key, value in offers_data_dict.items():
                                if isinstance(value, list):
                                    offers_data = value
                                    logger.info(f"🎁 [dynamic_doctor_search] Found offers in key '{key}': {len(offers_data)} offers")
                                    break
                else:
                    logger.warning("🎁 [dynamic_doctor_search] Offers result is not a dict or doesn't contain 'data'")
                    logger.warning(f"🎁 [dynamic_doctor_search] offers_result: {offers_result}")
                    # Try to extract offers directly from offers_result
                    if isinstance(offers_result, dict):
                        for key, value in offers_result.items():
                            if isinstance(value, list):
                                offers_data = value
                                logger.info(f"🎁 [dynamic_doctor_search] Found offers directly in offers_result key '{key}': {len(offers_data)} offers")
                                break
                
                # Add offers to the result
                if isinstance(result, dict):
                    result['offers'] = offers_data
                    logger.info(f"🎁 [dynamic_doctor_search] Added {len(offers_data)} offers to result before formatting")
                    logger.info(f"🎁 [dynamic_doctor_search] Result keys after adding offers: {list(result.keys())}")
                    logger.info(f"🎁 [dynamic_doctor_search] Result offers type: {type(result['offers'])}")
                    logger.info(f"🎁 [dynamic_doctor_search] Result offers length: {len(result['offers']) if isinstance(result['offers'], list) else 'Not a list'}")
                    
                    # Ensure offers are at the top level, not inside any nested structure
                    if 'offers' in result:
                        logger.info(f"🎁 [dynamic_doctor_search] Offers confirmed at top level of result")
                    else:
                        logger.error("🎁 [dynamic_doctor_search] Offers not found at top level after adding")
                else:
                    logger.error("🎁 [dynamic_doctor_search] Result is not a dict, cannot add offers")
                    logger.error(f"🎁 [dynamic_doctor_search] Result type: {type(result)}")
            else:
                logger.warning("🎁 [dynamic_doctor_search] No offers_result to add")
        
        # Ensure proper formatting of result (now with offers included)
        result = ensure_proper_doctor_search_format(result, str(search_query))
        
        # Debug: Check if offers are properly placed in the response object after formatting
        if isinstance(result, dict) and 'response' in result:
            if 'offers' in result['response']:
                logger.info(f"🎁 [dynamic_doctor_search] FINAL: Offers properly placed in response object: {len(result['response']['offers'])} offers")
            else:
                logger.info(f"🎁 [dynamic_doctor_search] FINAL: No offers found in response object after formatting")
            logger.info(f"🎁 [dynamic_doctor_search] FINAL: Final result keys: {list(result.keys())}")
        else:
            logger.info(f"🎁 [dynamic_doctor_search] FINAL: Result is not a dict or missing response after formatting")
        
        logger.info(f"🎁 [dynamic_doctor_search] RETURNING: Final result with {len(result.get('response', {}).get('offers', []))} offers")
        
        # Ensure the function has a name attribute that matches its actual name
        dynamic_doctor_search.__name__ = "dynamic_doctor_search"
        
        return result
    except Exception as e:
        logger.error(f"Error in doctor search: {str(e)}", exc_info=True)
        return {"error": str(e), "doctors": []}
    finally:
        # Add a timestamp to the last search
        thread_local.last_search_time = time.time()

def ensure_proper_doctor_search_format(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """
    Ensures doctor search results are returned in the correct format with separate
    message and data sections.
    
    Args:
        result: The search result to format
        query: The original query for context
        
    Returns:
        A properly formatted response dictionary
    """
    logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: result type: {type(result)}")
    logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    if isinstance(result, dict) and 'offers' in result:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: {len(result['offers'])} offers in result")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RECEIVED: No offers in result")
    
    # Check for nested response objects and unwrap them
    if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
        if "response" in result["response"]:
            # Unwrap doubly-nested response
            logger.info(f"DEBUG FORMAT: Unwrapping doubly-nested response")
            result = {"response": result["response"]["response"]}
    
    # Extract data for standardized format
    doctors = []
    message = None
    patient_info = {"session_id": getattr(thread_local, 'session_id', '')}
    
    # Handle the specific format returned by unified_doctor_search_tool
    if isinstance(result, dict) and "response" in result:
        logger.info(f"DEBUG FORMAT: Found 'response' key in result")
        response_data = result.get("response", {})
        
        # Check if response_data itself contains a response key (nested responses)
        if "response" in response_data:
            logger.info(f"DEBUG FORMAT: Found nested response, unwrapping")
            response_data = response_data["response"]
            
        # Get doctors list from response
        if isinstance(response_data, dict):
            if "data" in response_data and isinstance(response_data["data"], list):
                doctors = response_data["data"]
                logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in response.data")
            
            # Get message
            if "message" in response_data:
                message = response_data["message"]
            
            # Get patient info if available
            if "patient" in response_data:
                patient_info = response_data["patient"]
                
            # Preserve offers data if it exists in the original result
            offers_data = []
            if isinstance(result, dict) and 'offers' in result:
                offers_data = result['offers']
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Preserved {len(offers_data)} offers from original result")
            else:
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: No offers found in result to preserve")
            
            # Return properly formatted result with offers
            formatted_result = {
                "response": {
                    "message": message,
                    "patient": patient_info,
                    "data": doctors,
                    "doctor_count": len(doctors),
                    "is_doctor_search": True
                },
                "display_results": True,
                "doctor_count": len(doctors)
            }
            
            # Add offers to the response object
            if offers_data:
                formatted_result['response']['offers'] = offers_data
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Added {len(offers_data)} offers inside response object")
            else:
                formatted_result['response']['offers'] = []
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Initialized empty offers array inside response object")
            
            logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Final result keys: {list(formatted_result.keys())}")
            if 'response' in formatted_result and 'offers' in formatted_result['response']:
                logger.info(f"🎁 [ensure_proper_doctor_search_format] Early return: Final offers count: {len(formatted_result['response']['offers'])}")
            
            return formatted_result
    
    # Extract doctor data from various possible locations
    if isinstance(result, dict):
        # Get doctors list
        if "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data.doctors")
        elif "doctors" in result:
            doctors = result["doctors"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in top level")
        elif "data" in result and isinstance(result["data"], list):
            doctors = result["data"]
            logger.info(f"DEBUG FORMAT: Found {len(doctors)} doctors in data array")
        
        # Get message
        if "message" in result:
            message = result["message"]
        elif "status" in result and result["status"] == "not_found":
            message = None
        else:
            message = None
        
        # Get patient info if available
        if "patient" in result:
            patient_info = result["patient"]
        elif "data" in result and "patient" in result["data"]:
            patient_info = result["data"]["patient"]
    
    # Create the standardized response structure
    standardized_result = {
        "response": {
            "message": message,
            "patient": patient_info,
            "data": doctors,
            "doctor_count": len(doctors),
            "is_doctor_search": True
        },
        "display_results": True,
        "doctor_count": len(doctors)
    }
    
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Created base standardized_result with {len(doctors)} doctors")
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Base standardized_result keys: {list(standardized_result.keys())}")
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Base response keys: {list(standardized_result['response'].keys())}")
    
    # Preserve offers data if it exists in the original result
    offers_data = []
    if isinstance(result, dict) and 'offers' in result:
        offers_data = result['offers']
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Preserved {len(offers_data)} offers from original result")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] No offers found in result to preserve")
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    # Add offers to the response object at the same level as patient, message, and data
    if offers_data:
        standardized_result['response']['offers'] = offers_data
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Added {len(offers_data)} offers inside response object")
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Offers sample: {offers_data[0] if offers_data else 'No offers'}")
    else:
        standardized_result['response']['offers'] = []
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Initialized empty offers array inside response object")
    
    # Debug: Verify offers are in the final structure
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Final standardized_result keys: {list(standardized_result.keys())}")
    if 'response' in standardized_result:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] Response keys: {list(standardized_result['response'].keys())}")
        if 'offers' in standardized_result['response']:
            logger.info(f"🎁 [ensure_proper_doctor_search_format] Response offers count: {len(standardized_result['response']['offers'])}")
        else:
            logger.info(f"🎁 [ensure_proper_doctor_search_format] No offers key in response")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] No response key in standardized_result")
    
    # Ensure the structure matches the expected format with both data and offers
    if 'data' not in standardized_result:
        standardized_result['data'] = {"doctors": doctors}
    elif isinstance(standardized_result['data'], dict) and 'doctors' not in standardized_result['data']:
        standardized_result['data']['doctors'] = doctors
    
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Created standardized result with {len(doctors)} doctors")
    logger.info(f"🎁 [ensure_proper_doctor_search_format] Final standardized result keys: {list(standardized_result.keys())}")
    if 'response' in standardized_result and 'offers' in standardized_result['response']:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RETURNING: {len(standardized_result['response']['offers'])} offers in response object")
    else:
        logger.info(f"🎁 [ensure_proper_doctor_search_format] RETURNING: No offers in response object")
    return standardized_result

@tool(return_direct=False)
@profile
def analyze_symptoms(symptom_description: str) -> Dict[str, Any]:
    """
    Analyze the patient's symptom description to determine likely medical specialties required.
    Uses the MedicalSpecialty model to determine the right specialty based on symptoms mentioned.
    
    Args:
        symptom_description: Description of symptoms from the patient
        
    Returns:
        Dictionary with list of detected symptoms and matched specialties
    """
    function_start = time.time()
    logger.info(f"DETAILED DEBUG: Starting symptom analysis for: '{symptom_description[:50]}...'")
    
    try:
        # Detect symptoms and specialties
        result = detect_symptoms_and_specialties(symptom_description)
        
        # Store in thread_local for unified search access
        thread_local.symptom_analysis = result
        logger.info(f"DETAILED DEBUG: Stored symptom analysis in thread_local for session {getattr(thread_local, 'session_id', 'unknown')}")
        
        # Create a simpler output format
        simplified_result = {}
        
        # Check if specialty is not available in database
        speciality_not_available = False
        if "symptom_analysis" in result and "speciality_not_available" in result["symptom_analysis"]:
            speciality_not_available = result["symptom_analysis"]["speciality_not_available"]
        elif "speciality_not_available" in result:
            speciality_not_available = result["speciality_not_available"]
        elif "status" in result and result["status"] == "no_matching_specialty":
            speciality_not_available = True
            
        # Add the flag to the simplified result
        simplified_result["speciality_not_available"] = speciality_not_available
        
        if speciality_not_available:
            logger.info(f"DETAILED DEBUG: Specialty not available in database for the described symptoms")
            
            # Return early with only detected symptoms but no specialties
            simplified_result["symptoms_detected"] = result.get("symptom_analysis", {}).get("detected_symptoms", [])
            if not simplified_result["symptoms_detected"] and "detected_symptoms" in result:
                simplified_result["symptoms_detected"] = result["detected_symptoms"]
                
            simplified_result["top_specialties"] = []
            simplified_result["detailed_analysis"] = result
            
            logger.info(f"DETAILED DEBUG: Final simplified result for unmatched symptoms: {simplified_result}")
            return simplified_result
        
        # Extract detected symptoms
        symptoms_detected = []
        if "symptom_analysis" in result and "detected_symptoms" in result["symptom_analysis"]:
            symptoms_detected = result["symptom_analysis"]["detected_symptoms"]
            logger.info(f"DETAILED DEBUG: Extracted symptoms from symptom_analysis.detected_symptoms: {symptoms_detected}")
        simplified_result["symptoms_detected"] = symptoms_detected
        
        # Extract specialties
        top_specialties = []
        specialties_data = []
        
        if "specialties" in result:
            specialties_data = result["specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from result.specialties")
        elif "symptom_analysis" in result and "recommended_specialties" in result["symptom_analysis"]:
            specialties_data = result["symptom_analysis"]["recommended_specialties"]
            logger.info(f"DETAILED DEBUG: Found specialties from symptom_analysis.recommended_specialties")
        
        # Convert subspecialty to subspeciality format for compatibility
        if specialties_data:
            for specialty in specialties_data:
                if "subspecialty" in specialty and "subspeciality" not in specialty:
                    specialty["subspeciality"] = specialty["subspecialty"]
            
            logger.info(f"DETAILED DEBUG: Extracted {len(specialties_data)} specialties: {specialties_data}")
            
            # Extract top specialty names
            for specialty in specialties_data:
                if "name" in specialty:
                    top_specialties.append(specialty["name"])
                elif "specialty" in specialty:
                    top_specialties.append(specialty["specialty"])
        
        simplified_result["top_specialties"] = top_specialties
        logger.info(f"DETAILED DEBUG: Top specialties: {top_specialties}")
        
        # Include the full detailed analysis
        simplified_result["detailed_analysis"] = result
        
        logger.info(f"DETAILED DEBUG: Final simplified result: {simplified_result}")
        
        # Ensure the function has a name attribute that matches its actual name
        analyze_symptoms.__name__ = "analyze_symptoms"
        
        return simplified_result
    except Exception as e:
        logger.error(f"Error in symptom analysis: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "symptoms_detected": [],
            "top_specialties": [],
            "detailed_analysis": {"error": str(e)}
        }

# Create a StructuredTool for symptom analysis
analyze_symptoms_tool = StructuredTool.from_function(
    func=analyze_symptoms,
    name="analyze_symptoms",
    description="Analyze patient symptoms to determine appropriate medical specialties",
    args_schema=AnalyzeSymptomInput,
    return_direct=False,
    handle_tool_error="Could not analyze symptoms. Please try again with more details."
)
