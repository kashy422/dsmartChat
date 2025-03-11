from langchain_core.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import List, Optional, Union

from sqlalchemy import text
from .db import DB
import re
from decimal import Decimal


db = DB()

# Define Speciality Enum
class SpecialityEnum(str, Enum):
    DERMATOLOGY = "Dermatology"
    DENTISTRY = "Dentistry"
    CARDIOLOGY = "Cardiology"
    ORTHOPEDICS = "Orthopedics"
    
SPECIALITY_MAP = {
   "Orthodontist": "DENTISTRY",
    "Periodontist": "DENTISTRY",
    "Prosthodontist": "DENTISTRY",
    "General Dentist": "DENTISTRY",
    "Implantology": "DENTISTRY",
    "Cosmetic Dentist": "DENTISTRY"
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
        cursor = db.engine.connect()


        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("BEFORE")
        print("SPECIALTY: ", speciality)
        print("SUB SPECIALTY: ",sub_speciality)
        print("+++++++++++++++++++++++++++++++++++++++++++++++")


        # Normalize speciality
        speciality = speciality.strip() if speciality else ""

        # Normalize sub_speciality (ensure it's a list and convert to string)
        if isinstance(sub_speciality, list):
            sub_speciality = [s.strip() for s in sub_speciality if s]  # Remove empty values
            sub_speciality = ', '.join(set(sub_speciality))  # Remove duplicates and convert to string
        elif isinstance(sub_speciality, str):
            sub_speciality = sub_speciality.strip()
        else:
            sub_speciality = ""

        # Check if speciality needs mapping
        if speciality in SPECIALITY_MAP:
            mapped_speciality = SPECIALITY_MAP[speciality]
            if speciality not in SPECIALITY_MAP.values():
                speciality, sub_speciality = mapped_speciality, speciality  # Move original speciality to sub-speciality

        # Preserve valid sub_specialities (like "Orthodontist")
        if sub_speciality and sub_speciality not in SPECIALITY_MAP:
            pass  # Keep sub_speciality unchanged

        # Default sub_speciality to "General Dentist" only if speciality is DENTISTRY and sub_speciality is empty
        if not sub_speciality and speciality == "DENTISTRY":
            sub_speciality = "General Dentist"

        # Ensure `sub_speciality` is a string
        sub_speciality = sub_speciality if sub_speciality else ""






        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("AFTER")
        print("SPECIALTY: ", speciality)
        print("SUB SPECIALTY: ",sub_speciality)
        print("++++++++++++++++++++++++++++++++++++++++++++")

        # Call stored procedure
        stored_proc_query = text("EXEC GetRandomEntitiesByCriteria :speciality, :sub_speciality, :location")

        result = cursor.execute(stored_proc_query, {
            'speciality': speciality,
            'sub_speciality': sub_speciality,
            'location': location
        })

        def convert_decimals(obj):
            if isinstance(obj, list):
                return [convert_decimals(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj

        records = [convert_decimals(dict(row)) for row in result.mappings()]

        # Fetch and return results
        # records = [dict(row) for row in result.mappings()]


        cursor.close()

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
    # ✅ DENTISTRY
    "crooked teeth": "Orthodontist",
    "misaligned teeth": "Orthodontist",
    "braces": "Orthodontist",

    "gum disease": "Periodontist",
    "bleeding gums": "Periodontist",
    "receding gums": "Periodontist",

    "missing teeth": "Prosthodontist",
    "dental bridge": "Prosthodontist",
    "dentures": "Prosthodontist",

    "tooth pain": "General Dentist",
    "cavities": "General Dentist",
    "tooth sensitivity": "General Dentist",
    "bad breath": "General Dentist",

    "dental implants": "Implantology",
    "missing tooth": "Implantology",
    "tooth replacement": "Implantology",

    "teeth whitening": "Cosmetic Dentist",
    "veneers": "Cosmetic Dentist",
    "smile makeover": "Cosmetic Dentist",
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
    func=lambda speciality, location, sub_speciality=None: get_doctor_name_by_speciality(
        speciality, 
        location, 
        sub_speciality
    ),
    name="get_doctor_by_speciality",
    description="Get the list of available doctors for a given speciality and sub-speciality, based on symptoms.",
    args_schema=GetDoctorsBySpecialityInput,
    return_direct=False,
    handle_tool_error="No doctors found for the given speciality, sub-speciality, and location.",
)

def detect_speciality_subspeciality(user_input: str) -> tuple[Optional[str], Optional[str]]:
    """
    Detects speciality and sub-speciality from the user input.
    - Assumes input is in English (no translation or language detection).
    - Matches sub-speciality first, then maps to main speciality using `SPECIALITY_MAP`.
    - Returns (speciality, sub_speciality).
    """
    print("||-----------------------------------------------------")
    
    print("user_input",user_input)
    # Normalize input: remove punctuation and convert to lowercase
    normalized_input = re.sub(r'[^\w\s]', '', user_input)
    
    print(normalized_input)
    print("||-----------------------------------------------------")
    detected_sub_speciality = None
    detected_speciality = None

    # Check for sub-speciality first
    for sub_speciality, speciality in SPECIALITY_MAP.items():
        print(1)
        if sub_speciality.lower() in normalized_input:
            print(2)
            detected_sub_speciality = sub_speciality
            print(3)
            detected_speciality = speciality
            break  # Stop once we find a match

    # If no sub-speciality detected, check for main speciality
    if not detected_speciality:
        unique_specialities = set(SPECIALITY_MAP.values())  # Get unique specialities
        print(20)
        for speciality in unique_specialities:
            print(21)
            if speciality.lower() in normalized_input:
                print(22)
                detected_speciality = speciality
                break  # Stop once we find a match

    
    print("-----------------------------------------------------")
    print(detected_speciality)
    print(detected_sub_speciality)
    print("-----------------------------------------------------")
    
    #detected_sub_speciality = None
    return detected_speciality, detected_sub_speciality
    

# def store_patient_details(Name:str,Gender:str,Address:str,Issue:str,Contact:str,Email:str) -> list[dict[str, str | int | float | bool]]:
#     """Store the information of a paitent"""
#     return {Name,Gender,Address,Issue,Contact,Email}

def store_patient_details(
    Name: Optional[str] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    # Contact: Optional[str] = None,
) -> dict:
    """Store the information of a patient with default values for missing fields."""
    return {
        "Name": Name or None,
        "Gender": Gender or None,
        "Address": Location or None,
        "Issue": Issue or None,
        # "Contact": Contact or "Not Provided",
    }



store_patient_details_tool = StructuredTool.from_function(
    func=store_patient_details,
    name="store_patient_details",
    description="Store basic details of a paitent",
    args_schema=StorePatientDetails,
    return_direct=True,
    handle_tool_error="Paitent Details Incomplete",
)
