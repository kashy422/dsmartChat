from typing import Optional

def store_patient_details(
    Name: Optional[str] = None,
    Gender: Optional[str] = None,
    Location: Optional[str] = None,
    Issue: Optional[str] = None,
    session_id: Optional[str] = None  # Add session_id as an argument
) -> dict:
    """Store the information of a patient with default values for missing fields."""
    patient_info = {
        "Name": Name or None,
        "Gender": Gender or None,
        "Location": Location or None,
        "Issue": Issue or None,
        "session_id": session_id  # Include session_id in the output
    }
    print("Storing patient info:", patient_info)  # Debugging statement
    return patient_info  # Return a dictionary 