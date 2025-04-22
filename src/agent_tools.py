from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Union

from sqlalchemy import text
from .db import DB
import re
from decimal import Decimal
import time
from functools import wraps


db = DB()

# Define Speciality Enum
class SpecialityEnum(str, Enum):
    DERMATOLOGY = "Dermatology"
    DENTISTRY = "Dentistry"
    CARDIOLOGY = "Cardiology"
    ORTHOPEDICS = "Orthopedics"
    
SPECIALITY_MAP = {
#    "Orthodontist": "DENTISTRY",
#     "Periodontist": "DENTISTRY",
#     "Prosthodontist": "DENTISTRY",
#     "General Dentistry": "DENTISTRY",
#     "Implantology": "DENTISTRY",
#     "Cosmetic Dentist": "DENTISTRY",

    "Endodontics": "DENTISTRY",
    "Periodontics": "DENTISTRY",
    "Oral Surgery" : "DENTISTRY",
    "Oral and Maxillofacial Surgery" : "DENTISTRY",
    "Surgical Orthodontics" : "DENTISTRY",
    "Orthodontics" : "DENTISTRY",
    "Dental Implants" : "DENTISTRY",
    "Pediatric Dentistry" : "DENTISTRY",
    "Restorative Dentistry" : "DENTISTRY",
    "Forensic Dentistry" : "DENTISTRY"
}
# class GetDoctorsBySpecialityInput(BaseModel):
#     speciality: SpecialityEnum = Field(description="Speciality of the doctor")
#     location: str = Field(description="Location of the doctor")

class GetDoctorsBySpecialityInput(BaseModel):
    speciality: str = Field(description="Speciality of the doctor")
    location: str = Field(description="Location of the doctor")
    sub_speciality: Optional[Union[str, List[str]]] = Field(default=None, description="Sub-speciality of the doctor (if applicable). Can be a single string or a list of sub-specialities.")


class StorePatientDetails(BaseModel):
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")
    # Contact: Optional[str] = Field(default=None, description="Mobile Number or Contact details or Email")
    # Email:str = Field(description="Contact Email of the Patient")
    

# def get_doctor_name_by_speciality(speciality: SpecialityEnum, location: str) -> list[dict[str, str | int | float | bool]]:
#     """Take input of speciality and location and return available doctors for that speciality"""
#     return db.get_doctor_name_by_speciality(speciality, location)

def get_doctor_name_by_speciality(speciality: str, location: str, sub_speciality: Optional[str] = None) -> list[dict[str, str | int | float | bool]]:
    """
    Fetch doctors by calling the stored procedure `GetRandomEntitiesByCriteria`.
    Ensures speciality, sub-speciality (if detected), and location are passed correctly.
    """
    try:
        print("INITIAL INPUTS:", 
              f"speciality='{speciality}'", 
              f"location='{location}'", 
              f"sub_speciality='{sub_speciality}'")
        
        # Normalize inputs before database query
        # Normalize speciality
        speciality = speciality.strip() if speciality else ""
        
        # Fast path for common cases
        if speciality.lower() in ["dentist", "dentistry"]:
            speciality = "DENTISTRY"
            
        # Make specialty uppercase for consistency
        if speciality:
            speciality = speciality.upper()

        # Handle common specialty synonyms
        specialty_mapping = {
            "DENTAL": "DENTISTRY",
            "TEETH": "DENTISTRY", 
            "TOOTH": "DENTISTRY",
            "HEART": "CARDIOLOGY",
            "CARDIAC": "CARDIOLOGY",
            "BONES": "ORTHOPEDICS", 
            "JOINT": "ORTHOPEDICS",
            "SKIN": "DERMATOLOGY"
        }
        
        if speciality in specialty_mapping:
            speciality = specialty_mapping[speciality]

        # Process subspecialty information
        if sub_speciality:
            # Clean and normalize subspecialty
            sub_speciality = sub_speciality.strip()
            
            # Special case: Check if it's "General Dentist"
            if sub_speciality.lower() == "general dentist":
                speciality = "DENTISTRY"
                # Leave subspecialty as None for General Dentist
                sub_speciality = None
                print("Setting subspecialty to None for General Dentist")
            # Check if it's one of our mapped subspecialties
            elif sub_speciality in SPECIALITY_MAP:
                # Get the main specialty for this subspecialty
                mapped_specialty = SPECIALITY_MAP[sub_speciality]
                
                # If user-specified specialty doesn't match the mapped one, 
                # use the mapped specialty for consistency
                if speciality and speciality != mapped_specialty:
                    print(f"Overriding user specialty '{speciality}' with mapped specialty '{mapped_specialty}' based on subspecialty")
                    
                speciality = mapped_specialty
            else:
                # Try to find closest match among subspecialties
                sub_speciality_lower = sub_speciality.lower()
                match_found = False
                for known_sub in SPECIALITY_MAP.keys():
                    if known_sub.lower() in sub_speciality_lower or sub_speciality_lower in known_sub.lower():
                        print(f"Replacing subspecialty '{sub_speciality}' with known subspecialty '{known_sub}'")
                        sub_speciality = known_sub
                        speciality = SPECIALITY_MAP[known_sub]
                        match_found = True
                        break
                
                # If no match found and we have a speciality, check if we should clear the subspecialty
                if not match_found and speciality:
                    print(f"Unrecognized subspecialty '{sub_speciality}' with specialty '{speciality}', using general specialty")
                    sub_speciality = None
                        
        # Default sub-speciality if empty for dentistry - but let's keep it None for API call
        default_subspecialty = None
        if speciality == "DENTISTRY" and not sub_speciality:
            default_subspecialty = "General Dentist"
            print(f"Using default subspecialty for DENTISTRY: {default_subspecialty} (but keeping None for API)")

        # Get database connection - reusing connection from db module
        cursor = db.engine.connect()

        # Call stored procedure with final parameters
        print(f"Executing query with: speciality={speciality}, sub_speciality={sub_speciality}, location={location}")
        stored_proc_query = text("EXEC draide_prod.dbo.GetRandomEntitiesByCriteria :speciality, :sub_speciality, :location")
        
        # Execute query with params
        result = cursor.execute(stored_proc_query, {
            'speciality': speciality,
            'sub_speciality': sub_speciality,
            'location': location
        })
        
        # Process results efficiently
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: float(v) if isinstance(v, Decimal) else v for k, v in obj.items()}
            return obj
            
        records = [convert_decimals(dict(row)) for row in result.mappings()]
        cursor.close()
        
        print(f"Query returned {len(records)} doctor records")
        
        # Don't modify the response structure for frontend compatibility
        # Just log what was detected for debugging purposes
        if speciality == "DENTISTRY" and not sub_speciality:
            print(f"Debug: Using DENTISTRY with no subspecialty (general dentist implied)")
        
        return records

    except Exception as e:
        print(f"Error retrieving doctors for specialty: {e}")
        return []

@tool(return_direct=False)
def get_available_doctors_specialities() -> list[dict[str, str]]:
    """
    Return all available medical specialities and sub-specialities along with example symptoms.
    This helps the AI assistant detect them in patient conversations.
    """
    specialities = []
    for sub_speciality, speciality in SPECIALITY_MAP.items():
        # Add sample symptoms for each sub-speciality
        symptoms = SYMPTOM_TO_SPECIALITY.get(sub_speciality, "General Symptoms")
        specialities.append({
            "speciality": speciality,
            "sub_speciality": sub_speciality,
            "example_symptoms": symptoms
        })
    
    return specialities  # ✅ Now OpenAI sees symptoms → speciality → sub-speciality

SYMPTOM_TO_SPECIALITY = {

    "Deep dental caries" : "Endodontics",
    "Pulpitis" : "Endodontics",
    "Persistent throbbing pain" : "Endodontics",
    "Pain worsens at night" : "Endodontics",
    "Severe heat sensitivity" : "Endodontics",
    "Severe cold sensitivity" : "Endodontics",
    "Abscess or gum swelling" : "Endodontics",
    "Tooth discoloration" : "Endodontics",
    "Bad breath" : "Endodontics",
    "Inability to chew" : "Endodontics",

    "Bleeding during brushing": "Periodontics",
    "Gum redness": "Periodontics",
    "Gum swelling": "Periodontics",
    "Gum recession": "Periodontics",
    "Root sensitivity": "Periodontics",
    "Pain while chewing": "Periodontics",
    "Pus discharge from gums": "Periodontics",
    "Tooth mobility": "Periodontics",
    "Chewing difficulty due to gums": "Periodontics",


    "Oral swelling": "Oral Surgery",
    "Presence of abscess": "Oral Surgery",
    "Difficulty opening mouth": "Oral Surgery",
    "Impacted wisdom tooth": "Oral Surgery",
    "Post-extraction pain": "Oral Surgery",
    "Post-surgical bleeding": "Oral Surgery",
    "Delayed healing": "Oral Surgery",
    "Foul odor from wound": "Oral Surgery",
    "Lip numbness": "Oral Surgery",
    "Pain while swallowing": "Oral Surgery",

    "Jaw fracture": "Oral and Maxillofacial Surgery",
    "Misaligned upper/lower jaw": "Oral and Maxillofacial Surgery",
    "Facial swelling": "Oral and Maxillofacial Surgery",
    "Persistent lip/face numbness": "Oral and Maxillofacial Surgery",
    "Chronic TMJ pain": "Oral and Maxillofacial Surgery",
    "Difficulty opening mouth": "Oral and Maxillofacial Surgery",
    "Speech difficulties": "Oral and Maxillofacial Surgery",
    "Jaw tumors or cysts": "Oral and Maxillofacial Surgery",
    "Congenital facial deformities": "Oral and Maxillofacial Surgery",
    "Difficulty eating or swallowing": "Oral and Maxillofacial Surgery",

    "Upper/lower jaw protrusion": "Surgical Orthodontics",
    "Mandibular prognathism or retrognathism": "Surgical Orthodontics",
    "Dental misalignment": "Surgical Orthodontics",
    "Bite problems": "Surgical Orthodontics",
    "TMJ pain": "Surgical Orthodontics",
    "Anterior teeth wear": "Surgical Orthodontics",
    "Jaw muscle fatigue": "Surgical Orthodontics",
    "Chronic headaches": "Surgical Orthodontics",
    "Facial asymmetry": "Surgical Orthodontics",
    "Speech issues": "Surgical Orthodontics",

    "Crowded teeth": "Orthodontics",
    "Protruding front teeth": "Orthodontics",
    "Midline deviation": "Orthodontics",
    "Gaps between teeth": "Orthodontics",
    "Tooth overlap": "Orthodontics",
    "Deep bite": "Orthodontics",
    "Open bite": "Orthodontics",
    "Crossbite": "Orthodontics",
    "Difficulty cleaning teeth": "Orthodontics",
    "Irregular teeth wear": "Orthodontics",

     "Missing teeth": "Dental Implants",
    "Oral gaps": "Dental Implants",
    "Jaw bone resorption": "Dental Implants",
    "Difficulty eating": "Dental Implants",
    "Unstable dentures": "Dental Implants",
    "Speech changes": "Dental Implants",
    "Facial profile changes": "Dental Implants",
    "Food impaction": "Dental Implants",
    "Gum pain during chewing": "Dental Implants",
    "Low self-esteem": "Dental Implants",

    "Cavities in baby teeth": "Pediatric Dentistry",
    "Tooth discoloration": "Pediatric Dentistry",
    "Pain during eating": "Pediatric Dentistry",
    "Crying due to pain": "Pediatric Dentistry",
    "Gum swelling in kids": "Pediatric Dentistry",
    "Delayed tooth eruption": "Pediatric Dentistry",
    "Pediatric abscess": "Pediatric Dentistry",
    "Dental anxiety": "Pediatric Dentistry",
    "Tooth fracture from play": "Pediatric Dentistry",
    "Bad breath in child": "Pediatric Dentistry",

    "Visible decay": "Restorative Dentistry",
    "Hole in tooth": "Restorative Dentistry",
    "Sensitivity while eating": "Restorative Dentistry",
    "Pain with hot/cold drinks": "Restorative Dentistry",
    "Tooth fractures": "Restorative Dentistry",
    "Odor from tooth": "Restorative Dentistry",
    "Tooth discoloration": "Restorative Dentistry",
    "Lost old filling": "Restorative Dentistry",
    "Dark lines on teeth": "Restorative Dentistry",
    "Food sticking to tooth": "Restorative Dentistry",

    "Identifying unknown persons": "Forensic Dentistry",
    "Examining oral injuries": "Forensic Dentistry",
    "Documenting bite marks": "Forensic Dentistry",
    "Detecting neglect": "Forensic Dentistry",
    "Identifying remains via teeth": "Forensic Dentistry",
    "DNA from teeth": "Forensic Dentistry",
    "Age estimation via dentition": "Forensic Dentistry",
    "Dental record comparison": "Forensic Dentistry",
    "Injury type identification": "Forensic Dentistry",
    "Violence analysis": "Forensic Dentistry",




    # ✅ DENTISTRY
    # "crooked teeth": "Orthodontist",
    # "misaligned teeth": "Orthodontist",
    # "braces": "Orthodontist",

    # "gum disease": "Periodontist",
    # "bleeding gums": "Periodontist",
    # "receding gums": "Periodontist",

    # "missing teeth": "Prosthodontist",
    # "dental bridge": "Prosthodontist",
    # "dentures": "Prosthodontist",

    # "tooth pain": "General Dentist",
    # "cavities": "General Dentist",
    # "tooth sensitivity": "General Dentist",
    # "bad breath": "General Dentist",

    # "dental implants": "Implantology",
    # "missing tooth": "Implantology",
    # "tooth replacement": "Implantology",

    # "teeth whitening": "Cosmetic Dentist",
    # "veneers": "Cosmetic Dentist",
    # "smile makeover": "Cosmetic Dentist",
}

# # Registering tool with structured response
# get_doc_by_speciality_tool = StructuredTool.from_function(
#     func=get_doctor_name_by_speciality,
#     name="get_doctor_name_by_speciality",
#     description="Get the list of available doctors of given speciality and location",
#     args_schema=GetDoctorsBySpecialityInput,
#     return_direct=False,
#     handle_tool_error="No doctors found for the given speciality and location",
# )

get_doc_by_speciality_tool = StructuredTool.from_function(
    func=get_doctor_name_by_speciality,
    name="get_doctor_by_speciality",
    description="Get the list of available doctors for a given speciality and sub-speciality, based on symptoms.",
    args_schema=GetDoctorsBySpecialityInput,
    return_direct=False,
    handle_tool_error="No doctors found for the given speciality, sub-speciality, and location.",
)

def detect_speciality_subspeciality(user_input: str) -> tuple[Optional[str], Optional[str]]:
    """
    Improved detection of speciality and sub-speciality from user symptoms.
    Uses a combination of:
    1. Direct symptom matching against the SYMPTOM_TO_SPECIALITY dictionary
    2. Sub-speciality name detection in the input text
    3. Fallback to general speciality detection
    
    Returns a tuple of (speciality, sub_speciality)
    """
    if not user_input:
        return None, None
        
    # Normalize input text: remove punctuation, convert to lowercase
    normalized_input = re.sub(r'[^\w\s]', ' ', user_input.lower())
    words = set(normalized_input.split())
    
    print(f"Processing symptoms: {normalized_input}")
    
    # Check for general dentist first - this has special handling
    if "general dentist" in normalized_input:
        print("Detected 'general dentist' explicitly in input - using DENTISTRY with no subspecialty")
        return "DENTISTRY", None
    
    # 1. Symptom-based matching (most accurate)
    detected_symptoms = []
    potential_subspecialties = {}
    
    # Search for known symptoms in the input
    for symptom, subspecialty in SYMPTOM_TO_SPECIALITY.items():
        # Check if all words in the symptom appear in the input
        symptom_lower = symptom.lower()
        symptom_words = set(re.sub(r'[^\w\s]', ' ', symptom_lower).split())
        
        # Check for partial matches - at least 60% of the symptom words should match
        matching_words = words.intersection(symptom_words)
        if len(matching_words) >= max(1, len(symptom_words) * 0.6):
            detected_symptoms.append(symptom)
            # Count occurrences to find most relevant subspecialty
            potential_subspecialties[subspecialty] = potential_subspecialties.get(subspecialty, 0) + 1
    
    print(f"Detected symptoms: {detected_symptoms}")
    print(f"Potential subspecialties by symptom: {potential_subspecialties}")
    
    # If we found symptoms, find the most likely subspecialty
    if potential_subspecialties:
        # Find subspecialty with most symptom matches
        best_subspecialty = max(potential_subspecialties.items(), key=lambda x: x[1])[0]
        
        # Special case: if it's "General Dentist", return DENTISTRY with no subspecialty
        if best_subspecialty.lower() == "general dentist":
            return "DENTISTRY", None
            
        # Map to main specialty
        for sub, main in SPECIALITY_MAP.items():
            if sub == best_subspecialty:
                return main, best_subspecialty
    
    # 2. Direct subspecialty name detection (medium accuracy)
    for subspecialty in SPECIALITY_MAP.keys():
        subspecialty_lower = subspecialty.lower()
        # Check if the subspecialty name appears in the input
        if subspecialty_lower in normalized_input:
            if subspecialty_lower == "general dentist":
                return "DENTISTRY", None
            main_specialty = SPECIALITY_MAP[subspecialty]
            return main_specialty, subspecialty
    
    # 3. Fallback to main specialty detection (lowest accuracy)
    unique_specialties = set(SPECIALITY_MAP.values())
    for specialty in unique_specialties:
        if specialty.lower() in normalized_input:
            return specialty, None
    
    # 4. Secondary fallback to partial symptom matching
    if normalized_input:
        # Try to match based on common terms
        dental_terms = {'tooth', 'teeth', 'gum', 'dental', 'mouth', 'bite', 'jaw', 'chew'}
        if any(term in words for term in dental_terms):
            return "DENTISTRY", None
    
    return None, None
    

# def store_patient_details(Name:str,Gender:str,Address:str,Issue:str,Contact:str,Email:str) -> list[dict[str, str | int | float | bool]]:
#     """Store the information of a paitent"""
#     return {Name,Gender,Address,Issue,Contact,Email}

def store_patient_details(
    Name: Optional[str] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None  # Add session_id as an optional argument
) -> dict:
    """Store the information of a patient with default values for missing fields."""
    # Validate location - ensure it's not empty if provided
    if Location is not None and Location.strip() == "":
        Location = None
        print("Warning: Empty location provided, setting to None")
    
    # Create patient info with validated fields - preserve existing fields
    patient_info = {}
    
    # If this is an update, get existing data first
    if session_id:
        from .agent import get_session_history
        history = get_session_history(session_id)
        existing_data = history.get_patient_data()
        if existing_data:
            # Start with existing data
            patient_info = dict(existing_data)
            print(f"Found existing patient data for session {session_id}")
    
    # Update with new values, only if provided
    if Name is not None:
        patient_info["Name"] = Name
    if Gender is not None:
        patient_info["Gender"] = Gender
    if Location is not None:
        patient_info["Location"] = Location
    if Issue is not None:
        patient_info["Issue"] = Issue
    
    # Always include session_id
    patient_info["session_id"] = session_id
    
    # Log for debugging
    print(f"Storing patient info: Name={'Name' in patient_info}, Gender={'Gender' in patient_info}, " +
          f"Location={'Location' in patient_info}, Issue_length={len(patient_info.get('Issue', '')) if 'Issue' in patient_info else 0}")
    
    return patient_info



store_patient_details_tool = StructuredTool.from_function(
    func=store_patient_details,
    name="store_patient_details",
    description="Store basic details of a patient",
    args_schema=StorePatientDetails,
    return_direct=False,
    handle_tool_error="Patient Details Incomplete",
)

# Simple profiling decorator
def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function {func.__name__} took {elapsed_time:.2f} seconds to run")
        return result
    return wrapper

@tool(return_direct=False)
def find_doctors_by_speciality(args=None) -> list[dict[str, str | int | float | bool]]:
    """
    Fetch doctors by speciality, location, and sub-speciality from the database.
    The function detects speciality from patient's symptoms if speciality is not provided.
    """
    start_time = time.time()
    
    if args is None:
        print("Warning: find_doctors_by_speciality called with no arguments")
        return []
    
    # Handle both dict input and object input
    if isinstance(args, dict):
        speciality = args.get("speciality")
        location = args.get("location")
        sub_speciality = args.get("sub_speciality")
    else:
        # Object input (from LangChain)
        speciality = args.speciality
        location = args.location
        sub_speciality = args.sub_speciality
    
    print(f"Processing doctor search: speciality={speciality}, location={location}, sub_speciality={sub_speciality}")
    
    processing_time = time.time() - start_time
    print(f"Args processing took {processing_time:.4f} seconds")
    
    # Call the database function with profiling to time the operation
    query_start = time.time()
    doctors = get_doctor_name_by_speciality(speciality, location, sub_speciality)
    query_time = time.time() - query_start
    print(f"Database query took {query_time:.4f} seconds")
    
    # Format results for efficient serialization to JSON
    total_time = time.time() - start_time
    print(f"Total find_doctors_by_speciality execution time: {total_time:.4f} seconds")
    
    return doctors
