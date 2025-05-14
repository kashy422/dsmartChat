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
import json

# Initialize thread_local storage
thread_local = threading.local()

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Initialize the database connection
db = DB()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))

class SearchCriteria(BaseModel):
    """Model representing search criteria extracted from user query"""
    speciality: Optional[str] = None
    subspeciality: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
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
    Build parameters for SpDyamicQueryBuilderLatLng stored procedure
    
    Args:
        criteria: SearchCriteria object containing search parameters
        
    Returns:
        Tuple of (stored procedure name, parameters dictionary)
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
            if criteria.subspeciality:

                # Special case for Dentistry
                specialty_value = criteria.speciality.replace("'", "''")
                # Restore the N prefix for Unicode strings
                where_conditions.append(f"AND le.Specialty LIKE N'%{specialty_value}%'")
                logger.info(f"Added specialty filter: le.Specialty LIKE N'%{specialty_value}%'")
            else:
                # If no subspecialty, we can use the specialty directly
                specialty_value = criteria.speciality.replace("'", "''")
                # Restore the N prefix for Unicode strings
                where_conditions.append(f"AND (le.Specialty LIKE N'%{specialty_value}%' OR ds.Subspecialities LIKE N'%{specialty_value}%')")
                logger.info(f"AND (le.Specialty LIKE N'%{specialty_value}%' OR ds.Subspecialities LIKE N'%{specialty_value}%')")
            
        # Include subspecialty if provided
        if criteria.subspeciality:
            subspecialties = [s.strip() for s in criteria.subspeciality.split(',')] if ',' in criteria.subspeciality else [criteria.subspeciality]
            
            # Create properly formatted IN clause for subspecialties using ds.Subspecialities
            # Add N prefix for Unicode strings
            subspecialties_list = []
            for sub in subspecialties:
                # Properly escape single quotes for SQL
                escaped_sub = sub.replace("'", "''")
                subspecialties_list.append(f"N'{escaped_sub}'")
            
            # Join with commas for the IN clause
            subspecialties_str = ", ".join(subspecialties_list)
            where_conditions.append(f"AND ds.Subspecialities IN ({subspecialties_str})")
            logger.info(f"Added subspecialty filter using IN clause: ds.Subspecialities IN ({subspecialties_str})")
        
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
        
        # Create parameters dictionary with latitude, longitude, and WHERE clause
        params = {
            "@Latitude": criteria.latitude if criteria.latitude is not None else 0.0,
            "@Longitude": criteria.longitude if criteria.longitude is not None else 0.0,
            "@DynamicWhereClause": where_clause
        }
        
        logger.info(f"Using coordinates: Lat={params['@Latitude']}, Long={params['@Longitude']}")
        
        # Execute the stored procedure and log the results
        sp_name = "[dbo].[SpDyamicQueryBuilderLatLng]"
        logger.info(f"Executing stored procedure: {sp_name}")
        logger.info(f"With parameters: {params}")
        
        result = db.execute_stored_procedure(sp_name, params)
        logger.info(f"Database returned result: {json.dumps(result, indent=2)}")
        
        # Return stored procedure name and parameters
        return result
        
    except Exception as e:
        logger.error(f"Error building query: {str(e)}")
        raise

def normalize_specialty(specialty_name: str) -> dict:
    """
    Normalize specialty names to match the database terminology.
    This is needed because user queries and LLM outputs may use different terms
    than what's used in the database.
    
    Args:
        specialty_name: The original specialty name
        
    Returns:
        Dictionary with normalized specialty and subspecialty if applicable
    """
    if not specialty_name:
        return {"specialty": specialty_name}
        
    # Convert to lowercase for comparison
    specialty_lower = specialty_name.lower()
    
    # Special case: GP is a subspecialty of DENTISTRY
    gp_terms = ["gp", "general practice", "general practitioner", "family medicine", 
                "general physician", "primary care"]
    
    for term in gp_terms:
        if term in specialty_lower:
            logger.info(f"SPECIALTY: Detected GP term in '{specialty_name}', setting as subspecialty of Dentistry")
            return {
                "specialty": "Dentistry",
                "subspecialty": "GP"
            }
    
    # Common specialty name mappings - use proper case as expected by the stored procedure
    mappings = {
        "dentist": "Dentistry",
        "dental": "Dentistry",
        "dentistry": "Dentistry",
        "cardiology": "Cardiology",
        "cardiologist": "Cardiology",
        "heart": "Cardiology",
        "dermatology": "Dermatology",
        "dermatologist": "Dermatology",
        "skin": "Dermatology",
        "pediatric": "Pediatrics",
        "pediatrics": "Pediatrics",
        "pediatrician": "Pediatrics",
        "children": "Pediatrics",
        "orthopedic": "Orthopedics",
        "orthopedics": "Orthopedics",
        "bones": "Orthopedics",
        "gynecology": "Obstetrics & Gynecology",
        "obgyn": "Obstetrics & Gynecology",
        "obstetrics": "Obstetrics & Gynecology"
    }
    
    # Check for exact matches first
    if specialty_lower in mappings:
        normalized = mappings[specialty_lower]
        logger.info(f"SPECIALTY: Normalized '{specialty_name}' to '{normalized}'")
        return {"specialty": normalized}
    
    # Check for partial matches
    for key, value in mappings.items():
        if key in specialty_lower:
            logger.info(f"SPECIALTY: Partial match normalized '{specialty_name}' to '{value}'")
            return {"specialty": value}
    
    # If no mapping found, return the original with first letter of each word capitalized
    logger.info(f"SPECIALTY: No mapping found for '{specialty_name}', keeping as is")
    return {"specialty": specialty_name}

def unified_doctor_search(search_criteria: Union[dict, str]) -> dict:
    """Unified doctor search function that handles both structured and natural language queries"""
    logger.info(f"TOOL CALL: unified_doctor_search with input: {search_criteria}")
    
    # Create a new session ID for this search
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session_id in thread_local: {session_id}")
    
    # Initialize search parameters
    search_params = {}
    
    # Handle different input types
    if isinstance(search_criteria, str):
        try:
            # Try to parse as JSON first
            json_data = json.loads(search_criteria)
            if isinstance(json_data, dict):
                search_params = json_data
                logger.info(f"Parsed JSON search criteria: {search_params}")
        except json.JSONDecodeError:
            # If not JSON, treat as natural language query
            logger.info(f"Processing natural language query: '{search_criteria}'")
            # Extract search criteria from natural language
            print("\n" + "="*50)
            print("DEBUG: About to call extract_search_criteria_from_message")
            print("DEBUG: Input message:", search_criteria)
            print("="*50 + "\n")
            
            search_params = extract_search_criteria_from_message(search_criteria)
            logger.info(f"Extracted search criteria: {search_params}")
            
    elif isinstance(search_criteria, dict):
        # If it's already a dict, check if it contains a user_message
        if "user_message" in search_criteria:
            user_message = search_criteria["user_message"]
            logger.info(f"Found user_message in criteria: '{user_message}'")
            
            # Extract search criteria from the user message
            print("\n" + "="*50)
            print("DEBUG: About to call extract_search_criteria_from_message")
            print("DEBUG: Input message:", user_message)
            print("="*50 + "\n")
            
            extracted_criteria = extract_search_criteria_from_message(user_message)
            logger.info(f"Extracted search criteria: {extracted_criteria}")
            
            # Update search params with extracted criteria
            search_params.update(extracted_criteria)
            
            # Preserve any additional parameters from original criteria
            for key, value in search_criteria.items():
                if key != "user_message":
                    search_params[key] = value
        else:
            search_params = search_criteria
            logger.info(f"Using provided dictionary criteria: {search_params}")
    
    # Get coordinates from search params
    lat = search_params.get("latitude")
    long = search_params.get("longitude")
    
    if lat is not None and long is not None:
        logger.info(f"UNIFIED_SEARCH: Coordinates - lat: {lat}, long: {long}")
        logger.info(f"UNIFIED_SEARCH: Added coordinates to search criteria: lat={lat}, long={long}")
    else:
        logger.warning("UNIFIED_SEARCH: No coordinates provided in search criteria")
    
    # Convert to SearchCriteria object
    logger.info(f"UNIFIED_SEARCH: Converting search criteria to SearchCriteria object: {search_params}")
    try:
        criteria = SearchCriteria(**search_params)
        logger.info(f"UNIFIED_SEARCH: Converted to SearchCriteria: {criteria.dict()}")
    except Exception as e:
        logger.error(f"UNIFIED_SEARCH: Error converting to SearchCriteria: {str(e)}")
        # If conversion fails, try to extract from user message
        if isinstance(search_criteria, dict) and "user_message" in search_criteria:
            user_message = search_criteria["user_message"]
            logger.info(f"UNIFIED_SEARCH: Attempting to extract criteria from user message: {user_message}")
            extracted = extract_search_criteria_from_message(user_message)
            logger.info(f"UNIFIED_SEARCH: Extracted criteria: {extracted}")
            try:
                criteria = SearchCriteria(**extracted)
                logger.info(f"UNIFIED_SEARCH: Successfully converted extracted criteria to SearchCriteria: {criteria.dict()}")
            except Exception as e2:
                logger.error(f"UNIFIED_SEARCH: Error converting extracted criteria: {str(e2)}")
                # If all else fails, create empty criteria
                criteria = SearchCriteria()
                logger.warning("UNIFIED_SEARCH: Using empty SearchCriteria as fallback")
    
    # Build and execute query
    logger.info(f"Building query with criteria: {criteria.dict()}")
    try:
        result = build_query(criteria)
        
        # Process results
        if result and "data" in result and "doctors" in result["data"]:
            doctors = result["data"]["doctors"]
            logger.info(f"UNIFIED_SEARCH: Found {len(doctors)} doctors")
            logger.info(f"UNIFIED_SEARCH: Returning result with {len(doctors)} doctors")
        else:
            logger.warning("UNIFIED_SEARCH: No doctors found in result")
            result = {"data": {"doctors": []}}
        
        return result
    except Exception as e:
        logger.error(f"UNIFIED_SEARCH: Error building or executing query: {str(e)}")
        return {"data": {"doctors": []}}

def extract_search_criteria_from_message(message: str) -> Dict[str, Any]:
    """
    Uses GPT to extract search criteria from a message.
    This ensures consistent understanding between symptom analysis and search criteria.
    
    Args:
        message: User's message to extract criteria from
        
    Returns:
        Dictionary containing extracted search criteria
    """
    # Add very obvious debug statement
    print("\n" + "!"*50)
    print("DEBUG: EXTRACTION FUNCTION CALLED!")
    print("DEBUG: Input message:", message)
    print("!"*50 + "\n")
    
    try:
        # Log to both root logger and module logger for visibility
        root_logger.info("\n" + "="*50)
        root_logger.info("🔍 Starting search criteria extraction for message: '%s'", message)
        root_logger.info("="*50)
        
        # Prepare prompt for criteria extraction
        system_prompt = """Extract search criteria from the user's message into a structured format. Focus on:
        - Specialty/type of doctor (e.g., dentist, cardiologist, pediatrician) - ALWAYS in English
        - Price range (min and max in SAR) in western numbers
        - Rating requirements (minimum rating out of 5) in western numbers
        - Experience requirements (minimum years) in western numbers
        - Doctor name if mentioned (with title Dr/Doctor removed) - KEEP in original language (Arabic/English)
        - Clinic/branch name if mentioned - KEEP in original language (Arabic/English)
        - Gender preference ('male' or 'female' doctor) - ALWAYS in English
        
        IMPORTANT RULES:
        1. Doctor names and branch names:
           - Keep in original language (Arabic or English)
           - Remove titles like "Dr.", "Doctor", "الدكتور", "دكتور", "د.", "دكتر", "دكتوره", "دكتورة"
           - For Arabic names, pay special attention to these patterns:
             * "الدكتور [name]" -> extract "[name]"
             * "دكتور [name]" -> extract "[name]"
             * "ابحث عن الدكتور [name]" -> extract "[name]"
             * "اريد الدكتور [name]" -> extract "[name]"
             * "عايز الدكتور [name]" -> extract "[name]"
             * "عند الدكتور [name]" -> extract "[name]"
             * "مع الدكتور [name]" -> extract "[name]"
             * "عند دكتور [name]" -> extract "[name]"
             * "مع دكتور [name]" -> extract "[name]"
             * "عايز دكتور [name]" -> extract "[name]"
             * "ابحث عن دكتور [name]" -> extract "[name]"
             * "اريد دكتور [name]" -> extract "[name]"
             * "ابحث عن [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "اريد [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "عايز [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "عند [name]" -> extract "[name]" (if context suggests it's a doctor)
             * "مع [name]" -> extract "[name]" (if context suggests it's a doctor)
           - Examples:
             * "Dr. Smith" -> "Smith"
             * "الدكتور أحمد" -> "أحمد"
             * "دكتور محمد" -> "محمد"
             * "ابحث عن الدكتور الغريب" -> "الغريب"
             * "اريد الدكتور يوسف" -> "يوسف"
             * "عايز الدكتور علي" -> "علي"
             * "عند الدكتور خالد" -> "خالد"
             * "مع الدكتور سعيد" -> "سعيد"
             * "ابحث عن الغريب" -> "الغريب"
             * "اريد يوسف" -> "يوسف"
             * "عايز علي" -> "علي"
             * "عند خالد" -> "خالد"
             * "مع سعيد" -> "سعيد"
             * "مستشفى الملك فهد" -> "مستشفى الملك فهد"
             * "King Fahd Hospital" -> "King Fahd Hospital"
        
        2. All other fields must be in English:
           - Specialty: "طبيب أسنان" -> "dentist"
           - Gender: "طبيبة" -> "female", "طبيب" -> "male"
           - Numbers: Use western numerals (1, 2, 3) not Arabic numerals (١، ٢، ٣)
        
        3. For gender, look for any indication that the user wants only male or female doctors:
           - "only females" or "only males"
           - "female doctors" or "male doctors"
           - "women doctors" or "men doctors"
           - "show me females" or "show me males"
           - "طبيب" -> "male"
           - "طبيبة" -> "female"
           - "دكتور" -> "male"
           - "دكتورة" -> "female"
        
        4. For specialty, look for:
           - Direct mentions (e.g., "dentist", "cardiologist")
           - Common variations (e.g., "dental", "heart doctor")
           - Context clues (e.g., "teeth" -> dentist, "heart" -> cardiologist)
        
        Return ONLY a JSON object with these fields (include only if mentioned):
        {
            "speciality": "string (in English)",
            "min_price": number,
            "max_price": number,
            "min_rating": number,
            "min_experience": number,
            "doctor_name": "name (in original language)",
            "branch_name": "name (in original language)",
            "gender": "male" or "female"
        }
        """
        
        # Add message for extraction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        root_logger.info("🤖 Calling GPT for criteria extraction...")
        
        # Call GPT to extract criteria with improved configuration
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better Arabic understanding
            messages=messages,
            temperature=0.3,  # Slightly higher temperature for better handling of variations
            max_tokens=500,  # Increased token limit for better extraction
            presence_penalty=0.1,  # Encourage more diverse token usage
            frequency_penalty=0.1  # Discourage repetition
        )
        
        # Log the raw response
        raw_response = response.choices[0].message.content
        root_logger.info("\n📝 Raw GPT Response:")
        root_logger.info("-"*30)
        root_logger.info(raw_response)
        root_logger.info("-"*30)
        
        # Parse the response
        try:
            extracted = json.loads(raw_response)
            root_logger.info("\n✅ Successfully parsed JSON response")
            root_logger.info("\n🔍 Extracted search criteria:")
            for key, value in extracted.items():
                root_logger.info(f"   - {key}: {value}")
            
            # Additional validation for doctor name extraction
            if "doctor_name" in extracted:
                original_name = extracted["doctor_name"]
                # Remove any remaining titles that might have been missed
                doctor_name = original_name
                titles_to_remove = ["الدكتور", "دكتور", "د.", "دكتر", "دكتوره", "دكتورة", "Dr.", "Doctor"]
                for title in titles_to_remove:
                    if doctor_name.startswith(title + " "):
                        doctor_name = doctor_name[len(title) + 1:]
                extracted["doctor_name"] = doctor_name.strip()
                
                if original_name != extracted["doctor_name"]:
                    root_logger.info(f"\n🔄 Cleaned doctor name: '{original_name}' -> '{extracted['doctor_name']}'")
            
            root_logger.info("\n📊 Final extracted criteria:")
            root_logger.info(json.dumps(extracted, indent=2, ensure_ascii=False))
            root_logger.info("\n" + "="*50 + "\n")  # Visual separator
            
            # Also log to module logger
            logger.info(f"Extracted criteria: {json.dumps(extracted, indent=2, ensure_ascii=False)}")
            
            return extracted
            
        except json.JSONDecodeError as e:
            root_logger.error(f"\n❌ Failed to parse GPT response as JSON: {str(e)}")
            root_logger.error(f"\n❌ Raw response that failed to parse:")
            root_logger.error("-"*30)
            root_logger.error(raw_response)
            root_logger.error("-"*30)
            return {}
            
    except Exception as e:
        root_logger.error(f"\n❌ Error extracting search criteria: {str(e)}")
        return {}

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
            if "data" in search_results and "doctors" in search_results["data"]:
                doctor_list = search_results["data"]["doctors"]
                doctor_count = len(doctor_list)
                logger.info(f"TOOL DEBUG: Found {doctor_count} doctors in response.data")
        
        logger.info(f"TOOL RESULT: Found {doctor_count} doctors")
        
        # Store the search results in thread_local for future reference
        thread_local.last_search_results = search_results
        
        # Return results with explicit doctor count
        if isinstance(search_results, dict) and "data" in search_results:
            result = search_results["data"]
            # Add count for easier access
            if not "count" in result and "doctors" in result:
                result["count"] = len(result["doctors"])
                logger.info(f"TOOL DEBUG: Added count field with value {result['count']}")
        
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