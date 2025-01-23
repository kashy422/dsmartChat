from typing import Any
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler

# class CustomCallBackHandler(BaseCallbackHandler):

#     def __init__(self):
#         self.docs_data = []
#         self.paitent_data = []
        

#     def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
#         """Run when tool ends running."""
#         if kwargs['name'] == 'get_doctor_name_by_speciality':
#             print("My custom tool ended: ", output)
#             # self.docs_data = output
#             self.docs_data = {
#                 "message": "Here are some available doctors according to your requirments:",
#                 "data": output  # Assuming output is already a list of doctors
#             }


def store_patient_details(
    Name: str = None,
    Gender: str = None,
    Location: str = None,
    Issue: str = None,
) -> dict:
    """Store the information of a patient with only provided fields."""
    patient_info = {
        "Name": Name if Name else None,
        "Gender": Gender if Gender else None,
        "Location": Location if Location else None,
        "Issue": Issue if Issue else None,
        # "Contact": Contact if Contact else "Not Provided",
    }
    print("Storing patient info:", patient_info)  # Debugging statement
    return patient_info

# class CustomCallBackHandler(BaseCallbackHandler):

#     def __init__(self):
#         self.docs_data = {}
#         self.patient_data = {}

#     def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
#         """Run when the tool ends running."""
#         tool_name = kwargs.get('name')
#         print(f"Tool ended: {tool_name}, Output: {output}")  # Debugging statement

#         # if tool_name == 'store_patient_details':
#         #     # Store the patient data
#         #     self.patient_data = output  # Store patient data correctly
#         #     print(f"Patient data stored: {self.patient_data}")  # Debugging statement

#         if tool_name == 'get_doctor_name_by_speciality':
#             if tool_name == 'store_patient_details':
#                 # Store the patient data
#                 self.patient_data = output  # Store patient data correctly
#                 print(f"Patient data stored: {self.patient_data}")  # Debugging statement
#             combined_data = {
#                 "patient": self.patient_data,  # Patient details first
#                 "message": "Here are some available doctors according to your requirements:",
#                 "doctor_data": output  # The output from the get_doctor_name_by_speciality
#             }
#             self.docs_data = combined_data
            
#             # Debugging statement to check combined response
#             print(f"Combined response: {self.docs_data}")


class CustomCallBackHandler(BaseCallbackHandler):

    def __init__(self):
        self.docs_data = {}
        self.patient_data = {}
        self.patient_data_stored = False  # Flag to track if patient data has been stored

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when the tool ends running."""
        tool_name = kwargs.get('name')
        print(f"Tool ended: {tool_name}, Output: {output}")  # Debugging statement

        if tool_name == 'store_patient_details':
            # Store the patient data
            self.patient_data = output  # Store patient data correctly
            self.patient_data_stored = True  # Set flag to true
            print(f"Patient data stored: {self.patient_data}")  # Debugging statement
            
        elif tool_name == 'get_doctor_name_by_speciality':
            if self.patient_data_stored:
                combined_data = {
                    "patient": self.patient_data,  # Include stored patient data
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output  # The output from the get_doctor_name_by_speciality
                }
                self.docs_data = combined_data
            else:
                # If patient data is not available, just store doctor data
                self.docs_data = {
                    "message": "Here are some available doctors according to your requirements:",
                    "data": output
                }

            # Debugging statement to check combined response
            print(f"Combined response: {self.docs_data}")






        # if tool_name == 'get_doctor_name_by_speciality':
        #     # Store the patient data
        #     self.patient_data = output  # Store patient data correctly
        #     print(f"Patient data stored: {self.patient_data}")  # Debugging statement

