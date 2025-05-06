import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel
from sqlalchemy import text
from decimal import Decimal
from .specialty_matcher import SpecialtyDataCache
from .db import DB
import threading
import time
from openai import OpenAI
import uuid

# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

class SearchCriteria(BaseModel):
    """Model representing search criteria extracted from user query"""
    speciality: Optional[str] = None
    subspeciality: Optional[str] = None
    location: Optional[str] = None
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_experience: Optional[float] = None
    gender: Optional[str] = None
    hospital_name: Optional[str] = None
    branch_name: Optional[str] = None
    doctor_name: Optional[str] = None
    original_message: Optional[str] = None
    
    def dict(self, exclude_none: bool = False, **kwargs):
        """Return dictionary representation with option to exclude None values"""
        result = super().dict(**kwargs)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

def build_query(criteria: SearchCriteria) -> Tuple[str, Dict[str, Any]]:
    """
    Build WHERE clause for SpDyamicQueryBuilder stored procedure
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Tuple of (stored procedure name, parameters dictionary with where clause)
    """
    try:
        # Log the criteria we're using
        logger.info(f"Building query with criteria: {criteria.dict(exclude_none=True)}")
        
        # Debug log for specialty/subspecialty tracking
        if criteria.speciality:
            logger.info(f"DEBUG: Using specialty '{criteria.speciality}' in WHERE clause")
            if criteria.subspeciality:
                logger.info(f"DEBUG: Using subspecialty '{criteria.subspeciality}' in WHERE clause")
        else:
            logger.warning("DEBUG: No specialty found in search criteria!")
        
        # Start building dynamic WHERE clause
        where_conditions = []
        
        # Doctor name search (high priority)
        if criteria.doctor_name:
            doctor_name = criteria.doctor_name.replace("'", "''")
            where_conditions.append(f"AND (le.DocName_en LIKE N'%{doctor_name}%' OR le.DocName_ar LIKE N'%{doctor_name}%')")
        
        # Hospital/branch name search
        if criteria.branch_name:
            branch_name = criteria.branch_name.replace("'", "''")
            where_conditions.append(f"AND (b.BranchName_en LIKE N'%{branch_name}%' OR b.BranchName_ar LIKE N'%{branch_name}%')")
            
        # Hospital name (if different from branch name)
        if criteria.hospital_name and criteria.hospital_name != criteria.branch_name:
            hospital_name = criteria.hospital_name.replace("'", "''")
            where_conditions.append(f"AND (b.BranchName_en LIKE N'%{hospital_name}%' OR b.BranchName_ar LIKE N'%{hospital_name}%')")
        
        # Specialty search
        if criteria.speciality:
            specialty_value = criteria.speciality.replace("'", "''")
            where_conditions.append(f"AND le.Specialty = N'{specialty_value}'")
            
            # Include subspecialty if provided
            if criteria.subspeciality:
                subspecialties = [s.strip() for s in criteria.subspeciality.split(',')] if ',' in criteria.subspeciality else [criteria.subspeciality]
                if len(subspecialties) == 1:
                    sub_value = subspecialties[0].replace("'", "''")
                    where_conditions.append(f"AND s.SubSpeciality = N'{sub_value}'")
                else:
                    # Fix the string replacement for single quotes in the IN clause
                    subspecialties_list = ", ".join([f"N'{sub.replace('\'', '\'\'')}'" for sub in subspecialties])
                    where_conditions.append(f"AND s.SubSpeciality IN ({subspecialties_list})")
        
        # Location search
        if criteria.location:
            location_pattern = criteria.location.replace("'", "''")
            where_conditions.append(f"AND (b.Address_en LIKE N'%{location_pattern}%' OR b.Address_ar LIKE N'%{location_pattern}%' OR b.BranchName_en LIKE N'%{location_pattern}%' OR b.BranchName_ar LIKE N'%{location_pattern}%')")
        
        # Rating filter
        if criteria.min_rating is not None:
            where_conditions.append(f"AND le.Rating >= {float(criteria.min_rating)}")
        
        # Price/fee filters
        if criteria.min_price is not None and criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee BETWEEN {float(criteria.min_price)} AND {float(criteria.max_price)}")
        elif criteria.min_price is not None:
            where_conditions.append(f"AND le.Fee >= {float(criteria.min_price)}")
        elif criteria.max_price is not None:
            where_conditions.append(f"AND le.Fee <= {float(criteria.max_price)}")
        
        # Experience filter
        if criteria.min_experience is not None:
            where_conditions.append(f"AND le.Experience >= {float(criteria.min_experience)}")
        
        # Gender filter
        if criteria.gender:
            gender_value = criteria.gender.replace("'", "''")
            where_conditions.append(f"AND le.Gender = N'{gender_value}'")
        
        # Combine conditions into a single where clause
        # The stored procedure expects the WHERE clause with AND at the beginning of each condition
        where_clause = " ".join(where_conditions)
            
        logger.info(f"Built WHERE clause: {where_clause}")
        
        # Return stored procedure name and parameters
        return "[dbo].[SpDyamicQueryBuilder]", {"@DynamicWhereClause": where_clause}
        
    except Exception as e:
        logger.error(f"Error building query: {str(e)}")
        raise

def unified_doctor_search(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Unified doctor search function that handles both direct criteria and natural language queries.
    This function is called by the agent tools.
    
    Args:
        input_data: Either a string (natural language query) or dict containing:
            - speciality: Specialty from symptom analysis
            - subspeciality: Subspecialty from symptom analysis
            - location: Optional location
            - other search criteria parameters
        
    Returns:
        Search results in the standardized format
    """
    try:
        start_time = time.time()
        logger.info(f"DOCTOR SEARCH: Starting unified search with input: {input_data}")
        final_criteria = {}
        
        # Convert input to SearchCriteria
        if isinstance(input_data, str):
            logger.info("DOCTOR SEARCH: Processing string input")
            
            # Extract criteria from the message
            extracted = extract_search_criteria_from_message(input_data)
            logger.info(f"DOCTOR SEARCH: Extracted criteria from message: {extracted}")
            final_criteria.update(extracted)
            
            # Get symptom analysis from thread_local if it exists
            symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
            if symptom_analysis:
                logger.info(f"DOCTOR SEARCH: Found symptom analysis in thread_local")
                try:
                    # Safely access specialty information, checking multiple possible locations
                    specialties = []
                    
                    # Check multiple possible locations where specialties could be stored
                    if 'symptom_analysis' in symptom_analysis:
                        sa = symptom_analysis.get('symptom_analysis', {})
                        if 'matched_specialties' in sa and sa['matched_specialties']:
                            specialties = sa['matched_specialties']
                        elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                            specialties = sa['recommended_specialties']
                        elif 'specialties' in sa and sa['specialties']:
                            specialties = sa['specialties']
                    
                    # Only try to get specialty info if we found specialties
                    if specialties and len(specialties) > 0:
                        specialty_info = specialties[0]
                        specialty_name = specialty_info.get('specialty') or specialty_info.get('name')
                        subspecialty_name = specialty_info.get('subspecialty')
                        
                        specialty_criteria = {}
                        if specialty_name:
                            specialty_criteria['speciality'] = specialty_name
                        if subspecialty_name:
                            specialty_criteria['subspeciality'] = subspecialty_name
                            
                        if specialty_criteria:
                            logger.info(f"DOCTOR SEARCH: Adding specialty from symptom analysis: {specialty_criteria}")
                            final_criteria.update(specialty_criteria)
                except Exception as e:
                    logger.error(f"DOCTOR SEARCH: Error getting specialty info: {str(e)}")
            
            # Save original message
            final_criteria['original_message'] = input_data
        else:
            logger.info("DOCTOR SEARCH: Using provided dictionary criteria")
            # Use the directly provided criteria
            final_criteria.update(input_data)
        
        # Create criteria object
        criteria = SearchCriteria(**final_criteria)
        logger.info(f"DOCTOR SEARCH: Final search criteria: {criteria.dict(exclude_none=True)}")
        
        # Log warning if no significant criteria
        if not criteria.speciality and not criteria.location and not criteria.doctor_name and not criteria.branch_name:
            logger.warning("DOCTOR SEARCH: No significant search criteria found!")
            
            # Try to get specialty from symptom analysis as a last resort
            symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
            current_session_id = getattr(thread_local, 'session_id', None)
            
            # Only use symptom analysis if it exists and belongs to the current session
            if symptom_analysis:
                # Check if this symptom analysis has a session_id and matches current session
                analysis_session = getattr(symptom_analysis, 'session_id', None)
                
                if analysis_session and analysis_session != current_session_id:
                    logger.warning(f"DOCTOR SEARCH: Ignoring symptom analysis from different session (analysis: {analysis_session}, current: {current_session_id})")
                    symptom_analysis = None
                else:
                    # Log that we're using symptom analysis from the current session
                    logger.info(f"DOCTOR SEARCH: Using symptom analysis from current session")
            
            if symptom_analysis:
                try:
                    # Look for specialties in multiple possible locations
                    specialties = []
                    
                    # Check various possible locations for specialty data
                    if 'symptom_analysis' in symptom_analysis:
                        sa = symptom_analysis.get('symptom_analysis', {})
                        if 'matched_specialties' in sa and sa['matched_specialties']:
                            specialties = sa['matched_specialties']
                        elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                            specialties = sa['recommended_specialties']
                        elif 'specialties' in sa and sa['specialties']:
                            specialties = sa['specialties']
                    
                    # Process the first specialty if available
                    if specialties and len(specialties) > 0:
                        top_specialty = specialties[0]
                        specialty = top_specialty.get('specialty') or top_specialty.get('name')
                        subspecialty = top_specialty.get('subspecialty')
                        
                        if specialty:
                            logger.info(f"DOCTOR SEARCH: Using specialty '{specialty}' from symptom analysis")
                            criteria.speciality = specialty
                            
                            if subspecialty:
                                logger.info(f"DOCTOR SEARCH: Using subspecialty '{subspecialty}' from symptom analysis")
                                criteria.subspeciality = subspecialty
                except Exception as e:
                    logger.error(f"DOCTOR SEARCH: Error getting specialty from symptom analysis: {str(e)}")
        
        # Build and execute query
        query, params = build_query(criteria)
        
        # Add detailed debug logging for the SQL query
        debug_sql = f"""
        -- Debug SQL for SpDyamicQueryBuilder 
        EXEC [dbo].[SpDyamicQueryBuilder] @DynamicWhereClause = N'{params["@DynamicWhereClause"]}'
        """
        logger.info(f"DOCTOR SEARCH: SQL Query to execute:\n{debug_sql}")
        
        # Execute the stored procedure
        logger.info(f"DOCTOR SEARCH: Executing query: {query} with params: {params}")
        results = db.execute_stored_procedure(query, params)
        logger.info(f"DOCTOR SEARCH: Got {len(results) if results else 0} results")
        
        # Format results
        doctors = []
        for row in results:
            # Get values with proper type conversion and fallbacks
            doctor = {
                "DoctorId": row.get('DoctorId'),
                "DoctorName_ar": row.get('DoctorName_ar'),
                "DoctorName_en": row.get('DoctorName_en'),
                "Rating": float(row.get('Rating')) if row.get('Rating') is not None else 0,
                "Fee": float(row.get('Fee')) if row.get('Fee') is not None else 0,
                "Speciality": row.get('Speciality'),  # Note: Stored proc uses 'Speciality' not 'Specialty'
                "Subspecialities": row.get('Subspecialities'),  # Match the stored proc column name
                "Branch_ar": row.get('Branch_ar'),
                "Branch_en": row.get('Branch_en'),
                "Address_ar": row.get('Address_ar'),
                "Address_en": row.get('Address_en'),
                "DiscountValue": float(row.get('DiscountValue')) if row.get('DiscountValue') is not None else 0,
                "DiscountType": row.get('DiscountType', 'None'),
                "HasDiscount": bool(row.get('HasDiscount', False)),
                "Gender": row.get('Gender', ''),
                "Experience": float(row.get('Experience')) if row.get('Experience') is not None else 0
            }
            doctors.append(doctor)
        
        # Get session info from thread_local
        session_id = getattr(thread_local, 'session_id', '')
        patient_info = getattr(thread_local, 'patient_info', {}) or {}
        symptom_info = getattr(thread_local, 'symptom_analysis', {}) or {}
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
            
        response = {
            "response": {
                "message": f"I've found {len(doctors)} doctors that may be able to help with your issue.",
                "patient": {
                    "Name": patient_info.get('name', ''),
                    "Gender": patient_info.get('gender', ''),
                    "Location": criteria.location or patient_info.get('location', ''),
                    "Issue": symptom_info.get('symptom_description', ''),
                    "session_id": session_id
                },
                "data": doctors
            },
            "processing_time": processing_time,
            "performance": {
                "total_time": round(processing_time, 2),
                "processing_time": round(processing_time, 2)
            }
        }
        
        # Create a simple message without doctor details
        specialty_text = criteria.speciality or "medical"
        location_text = f" in {criteria.location}" if criteria.location else ""
        
        # Generate an appropriate message without doctor details
        if len(doctors) > 0:
            message = f"I found {len(doctors)} {specialty_text} specialists{location_text} based on your search."
        else:
            message = f"We are currently certifying doctors in our network. Please check back soon for {specialty_text} specialists{location_text}."
        
        # Replace the detailed message with a simple one
        response["response"]["message"] = message
        
        logger.info(f"DOCTOR SEARCH: Returning response with {len(doctors)} doctors")
        return response
        
    except Exception as e:
        logger.error(f"DOCTOR SEARCH ERROR: {str(e)}", exc_info=True)
        try:
            end_time = time.time()
            processing_time = end_time - start_time
        except:
            processing_time = 0
            return {
            "response": {
                "message": f"We are currently certifying doctors in our network. Please check back soon.",
                "patient": {
                    "Name": "",
                    "Gender": "",
                    "Location": "",
                    "Issue": "",
                    "session_id": getattr(thread_local, 'session_id', '')
                },
                "data": []
            },
            "processing_time": processing_time,
            "performance": {
                "total_time": round(processing_time, 2),
                "processing_time": round(processing_time, 2)
            }
        }

def extract_search_criteria_from_message(message: str) -> Dict[str, Any]:
    """
    Uses GPT to extract search criteria from a message.
    This ensures consistent understanding between symptom analysis and search criteria.
    
    Args:
        message: User's message to extract criteria from
        
    Returns:
        Dictionary containing extracted search criteria
    """
    try:
        # Prepare prompt for criteria extraction
        system_prompt = """You are an intelligent assistant. Your task is to extract structured search criteria from the user's message. The user message will be in natural lanaguage. Also keep in mind that users can also use arabic language.

        For the prices, also encorporate natural language to specify the range of prices. For example: User can ask show me cheapest doctors, or show me doctors with prices between 100 and 500 SAR. You need to extract this information and convert it to the correct format.

        For rating, also encorporate natural language to specify the range of ratings. For example: User can ask show me doctors with ratings above 4.5, or show me high or low ratings, or show me doctors with ratings between 3.5 and 4.5. You need to extract this information and convert it to the correct format.

        For experience, also encorporate natural language to specify the range of experience. For example: User can ask show me doctors with experience above 10 years, or show me doctors with experience between 5 and 10 years. You need to extract this information and convert it to the correct format.

        Focus only on the following fields, and extract them **only if they are clearly mentioned**:

        - **location**: A City
        - **min_price**: Minimum price in SAR
        - **max_price**: Maximum price in SAR
        - **min_rating**: Minimum acceptable rating (0–5 scale)
        - **min_experience**: Minimum required years of experience
        - **doctor_name**: Name of the doctor (remove title like “Dr.” or “Doctor”)
        - **branch_name**: Clinic or branch name
        - **gender**: Gender preference (either "Male" or "Female")

        **Output a JSON object** using the format below. Include only the fields that are explicitly or implicitly mentioned:

        ```json
        {
        "location": "city_name",
        "min_price": number,
        "max_price": number,
        "min_rating": number,
        "min_experience": number,
        "doctor_name": "name",
        "branch_name": "name",
        "gender": "Male" or "Female"
        }

        """
        
        # Add message for extraction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Get GPT response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # Parse the response to extract criteria
        try:
            import json
            # Extract the JSON content from the response
            content = response.choices[0].message.content
            extracted = json.loads(content)
            
            # Validate and clean the extracted data
            criteria = {}
            
            # Location validation
            if "location" in extracted:
                location = extracted["location"].capitalize()
                if location in ["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam"]:
                    criteria["location"] = location
            
            # Numeric fields validation
            if "min_price" in extracted and isinstance(extracted["min_price"], (int, float)):
                criteria["min_price"] = float(extracted["min_price"])
            if "max_price" in extracted and isinstance(extracted["max_price"], (int, float)):
                criteria["max_price"] = float(extracted["max_price"])
            if "min_rating" in extracted and isinstance(extracted["min_rating"], (int, float)):
                criteria["min_rating"] = float(extracted["min_rating"])
            if "min_experience" in extracted and isinstance(extracted["min_experience"], (int, float)):
                criteria["min_experience"] = float(extracted["min_experience"])
                
            # Name fields validation
            if "doctor_name" in extracted and extracted["doctor_name"]:
                criteria["doctor_name"] = extracted["doctor_name"].strip()
            if "branch_name" in extracted and extracted["branch_name"]:
                criteria["branch_name"] = extracted["branch_name"].strip()
            
            # Gender validation
            if "gender" in extracted and extracted["gender"]:
                gender = extracted["gender"].strip()
                if gender.lower() in ["male", "female"]:
                    criteria["gender"] = gender.capitalize()
            
            logger.info(f"Extracted search criteria: {criteria}")
            return criteria
            
        except json.JSONDecodeError as json_error:
            logger.error(f"Error parsing GPT JSON response: {str(json_error)}")
            # Fallback to simple extraction
            return _fallback_extraction(message)
        except Exception as parse_error:
            logger.error(f"Error processing GPT response: {str(parse_error)}")
            return _fallback_extraction(message)
            
    except Exception as e:
        logger.error(f"Error in criteria extraction: {str(e)}")
        return _fallback_extraction(message)

def _fallback_extraction(message: str) -> Dict[str, Any]:
    """Fallback extraction for when GPT fails"""
    # Extract location from message
    location = None
    words = message.lower().split()
    cities = ["riyadh", "jeddah", "mecca", "medina", "dammam"]
    for city in cities:
        if city in words:
            location = city.capitalize()
            break
            
    # Extract gender if mentioned
    gender = None
    if "male doctor" in message.lower() or "man doctor" in message.lower():
        gender = "Male"
    elif "female doctor" in message.lower() or "woman doctor" in message.lower() or "lady doctor" in message.lower():
        gender = "Female"
            
    # Build criteria dictionary
    criteria = {}
    if location:
        criteria["location"] = location
    if gender:
        criteria["gender"] = gender
            
    logger.info(f"Fallback extraction: {criteria}")
    return criteria

def unified_doctor_search_tool(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tool wrapper for unified_doctor_search to be used by the agent.
    Searches for doctors based on criteria.
    """
    try:
        logger.info(f"TOOL CALL: unified_doctor_search with input: {input_data}")
        
        # Make sure the session_id is available in thread_local
        if hasattr(thread_local, 'session_id'):
            logger.info(f"Using existing session_id in thread_local: {thread_local.session_id}")
        else:
            # Generate a session ID if not present
            thread_local.session_id = str(uuid.uuid4())
            logger.info(f"Created new session_id in thread_local: {thread_local.session_id}")
            
        # If we're passed symptom analysis in the input, save it to thread_local
        if isinstance(input_data, dict) and 'symptom_analysis' in input_data:
            thread_local.symptom_analysis = input_data.pop('symptom_analysis')
            logger.info(f"Stored symptom analysis in thread_local")
            
        # If we're passed patient info in the input, save it to thread_local
        if isinstance(input_data, dict) and 'patient_info' in input_data:
            thread_local.patient_info = input_data.pop('patient_info')
            logger.info(f"Stored patient info in thread_local")
        
        # Call the unified search function once
        search_results = unified_doctor_search(input_data)
        logger.info("TOOL CALL: Got search results")
        
        # Extract the doctor count from the results for logging
        doctor_count = 0
        doctor_list = []
        
        # Check response structure
        logger.info(f"TOOL DEBUG: Search results keys: {list(search_results.keys()) if isinstance(search_results, dict) else 'Not a dict'}")
        
        if isinstance(search_results, dict):
            # Extract doctor data from the typical structure
            if "response" in search_results and "data" in search_results["response"]:
                doctor_list = search_results["response"]["data"]
                doctor_count = len(doctor_list)
                logger.info(f"TOOL DEBUG: Found {doctor_count} doctors in response.data")
        
        logger.info(f"TOOL RESULT: Found {doctor_count} doctors")
        
        # Store the search results in thread_local for future reference
        thread_local.last_search_results = search_results
        
        # Return results with explicit doctor count
        if isinstance(search_results, dict) and "response" in search_results:
            response = search_results["response"]
            # Add count for easier access
            if not "count" in response and "data" in response:
                response["count"] = len(response["data"])
                logger.info(f"TOOL DEBUG: Added count field with value {response['count']}")
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error in unified_doctor_search_tool: {str(e)}", exc_info=True)
        return {
            "response": {
                "message": f"We are currently certifying doctors in our network. Please check back soon.",
                "patient": {
                    "Name": "",
                    "Gender": "",
                    "Location": "",
                    "Issue": "",
                    "session_id": getattr(thread_local, 'session_id', '')
                },
                "data": []
            },
            "processing_time": 0,
            "performance": {
                "total_time": 0,
                "processing_time": 0
            }
        }

def extract_search_criteria_tool(user_query: str) -> Dict[str, Any]:
    """
    Tool wrapper for criteria extraction to be used by the agent.
    Uses GPT to extract search criteria and combines with symptom analysis.
    
    Args:
        user_query: Natural language query from the user
        
    Returns:
        Dictionary with extracted search criteria
    """
    try:
        logger.info(f"TOOL CALL: extract_search_criteria with query: '{user_query[:50]}...'")
        
        # Extract criteria using GPT
        extracted = extract_search_criteria_from_message(user_query)
        
        # Get symptom analysis from thread_local if it exists
        symptom_analysis = getattr(thread_local, 'symptom_analysis', None)
        if symptom_analysis:
            logger.info(f"Found symptom analysis in thread_local for criteria extraction")
            try:
                # Extract specialty info from symptom analysis - with safety checks
                specialties = []
                
                # Check multiple possible locations for specialty data
                if 'symptom_analysis' in symptom_analysis:
                    sa = symptom_analysis.get('symptom_analysis', {})
                    if 'matched_specialties' in sa and sa['matched_specialties']:
                        specialties = sa['matched_specialties']
                    elif 'recommended_specialties' in sa and sa['recommended_specialties']:
                        specialties = sa['recommended_specialties']
                    elif 'specialties' in sa and sa['specialties']:
                        specialties = sa['specialties']
                
                # Process the first specialty if available
                if specialties and len(specialties) > 0:
                    top_specialty = specialties[0]
                    
                    # Only add specialty if not already present in extracted criteria
                    if 'speciality' not in extracted:
                        specialty = top_specialty.get('specialty') or top_specialty.get('name')
                        if specialty:
                            extracted['speciality'] = specialty
                            logger.info(f"Added specialty '{specialty}' from symptom analysis")
                    
                    # Only add subspecialty if not already present in extracted criteria
                    if 'subspeciality' not in extracted:
                        subspecialty = top_specialty.get('subspecialty')
                        if subspecialty:
                            extracted['subspeciality'] = subspecialty
                            logger.info(f"Added subspecialty '{subspecialty}' from symptom analysis")
            except Exception as e:
                logger.error(f"Error extracting specialty info from symptom analysis: {str(e)}")
        
        # Determine what information is missing
        missing_info = []
        if not extracted.get('speciality') and not extracted.get('doctor_name') and not extracted.get('branch_name'):
            missing_info.append('specialty')
        if not extracted.get('location'):
            missing_info.append('location')
            
        # Store the extracted criteria in thread_local for later use
        thread_local.extracted_criteria = extracted
            
        return {
            "criteria": extracted,
            "needs_more_info": len(missing_info) > 0,
            "missing_info": missing_info,
            "extracted_items": list(extracted.keys()),
            "symptom_analysis_used": symptom_analysis is not None
        }
        
    except Exception as e:
        logger.error(f"Error in extract_search_criteria_tool: {str(e)}")
        return {
            "error": str(e),
            "criteria": {},
            "needs_more_info": True,
            "missing_info": ["error occurred"],
            "extracted_items": []
        }

# Export all necessary functions
__all__ = ['SearchCriteria', 'unified_doctor_search', 'unified_doctor_search_tool', 'extract_search_criteria_tool'] 