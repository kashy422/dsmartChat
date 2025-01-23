from langchain_core.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from enum import Enum
from typing import Optional
from .db import DB

db = DB()

# Define Speciality Enum
class SpecialityEnum(str, Enum):
    DERMATOLOGY = "Dermatology"
    DENTISTRY = "Dentistry"
    CARDIOLOGY = "Cardiology"
    ORTHOPEDICS = "Orthopedics"

class GetDoctorsBySpecialityInput(BaseModel):
    speciality: SpecialityEnum = Field(description="Speciality of the doctor")
    location: str = Field(description="Location of the doctor")

class StorePatientDetails(BaseModel):
    Name: Optional[str] = Field(default=None, description="Name of the patient")
    Gender: Optional[str] = Field(default=None, description="Gender of the patient")
    Location: Optional[str] = Field(default=None, description="Location of the patient")
    Issue: Optional[str] = Field(default=None, description="The Health Concerns or Symptoms of a patient")
    # Contact: Optional[str] = Field(default=None, description="Mobile Number or Contact details or Email")
    # Email:str = Field(description="Contact Email of the Patient")
    

def get_doctor_name_by_speciality(speciality: SpecialityEnum, location: str) -> list[dict[str, str | int | float | bool]]:
    """Take input of speciality and location and return available doctors for that speciality"""
    return db.get_doctor_name_by_speciality(speciality, location)

@tool(return_direct=False)
def get_available_doctors_specialities() -> list[str]:
    """Return all available doctor specialities"""
    return db.get_available_doctors_specialities()

# Registering tool with structured response
get_doc_by_speciality_tool = StructuredTool.from_function(
    func=get_doctor_name_by_speciality,
    name="get_doctor_name_by_speciality",
    description="Get the list of available doctors of given speciality and location",
    args_schema=GetDoctorsBySpecialityInput,
    return_direct=False,
    handle_tool_error="No doctors found for the given speciality and location",
)

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
    return_direct=False,
    handle_tool_error="Paitent Details Incomplete",
)
