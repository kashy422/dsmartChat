from typing import Any, Optional
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
import json 
import urllib.parse
import threading



thread_local = threading.local() # for session id sharing


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



class CustomCallBackHandler(BaseCallbackHandler):
    def __init__(self):
        self.docs_data = {}  # Stores doctor-related data
        self.patient_data = {}  # Stores patient data, keyed by session_id
        self.patient_data_stored = False  # Flag to track if patient data has been stored

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when the tool ends running."""
        tool_name = kwargs.get('name')
        session_id = getattr(thread_local, 'session_id', None)  # Extract session_id from thread-local storage
    
        print(f"Debug: session_id = {session_id}")  # Debugging statement

        if not session_id:
            raise ValueError("session_id is required to store patient data.")

        print("\n\n")
        print(f"Tool ended: {tool_name}, Output: {output}")  # Debugging statement

        if tool_name == 'store_patient_details':
            # Store patient data for the specific session_id
            self.patient_data[session_id] = output  # Store patient data under the session_id key
            self.patient_data_stored = True  # Set flag to true
            print(f"Patient data stored for session {session_id}: {self.patient_data[session_id]}")  # Debugging statement

        elif tool_name in ['get_doctor_name_by_speciality', 'get_doctor_by_speciality']:
            print(1)
            if self.patient_data_stored and session_id in self.patient_data:
                print(2)
                combined_data = {
                    "patient": self.patient_data[session_id],  # Include stored patient data for the session
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output  # The output from the doctor tool
                }
                print(3)
                self.docs_data = combined_data
            else:
                print(4)
                # If patient data is not available, just store doctor data
                self.docs_data = {
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output
                }
                print(5)

            # Debugging statement to check combined response
            print(f"Combined response: {self.docs_data}")
            

