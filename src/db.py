import os
from pprint import pprint
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import urllib
from enum import Enum
import uuid
import datetime 
from decimal import Decimal


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Speciality Enum
class SpecialityEnum(str, Enum):
    DERMATOLOGIST ="Dermatologist"
    DERMATOLOGY = "Dermatology"
    
    DENTISTRY = "Dentistry"
    CARDIOLOGY = "Cardiology"
    ORTHOPEDICS = "Orthopedics"
    GENERALSURGERY = "General Surgery"
    GENERALDENTIST = "General Dentist"
    ORTHODONTIST = "Orthodontist"

SPECIALITY_MAP = {
#    "Orthodontist": "DENTISTRY",
#     "Periodontist": "DENTISTRY",
#     "Prosthodontist": "DENTISTRY",
#     "General Dentist": "DENTISTRY",
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

class DB:
    def __init__(self):
        self.__connect()

    def __connect(self):
        try:
            # Establish the connection
            # self.connection = pymysql.connect(
            #     host=os.environ.get('DB_HOST'),
            #     user=os.environ.get('DB_USER'),
            #     password=os.environ.get('DB_PASSWORD'),
            #     database=os.environ.get('DB_DATABASE'),
            #     cursorclass=pymysql.cursors.DictCursor
            # )
            params = urllib.parse.quote_plus(
                  f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                  f"SERVER={os.environ.get('DB_HOST')};"
                  f"DATABASE={os.environ.get('DB_DATABASE')};"
                  f"UID={os.environ.get('DB_USER')};"
                  f"PWD={os.environ.get('DB_PASSWORD')};"
                  f"TrustServerCertificate=yes"
                )

            # SQLAlchemy connection string
            db_url = f"mssql+pyodbc:///?odbc_connect={params}"
            test_db_url = "mssql+pyodbc://@(localdb)\\MSSQLLocalDB/DrAide_Dev?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

            self.engine = create_engine(test_db_url,pool_pre_ping=True)

            print("Successfully connected to the MSSQL database.")

        except SQLAlchemyError as e:
            print(e)
            print("Error connecting to the MSSQL database.")

    def __del__(self):
        print("Connection to the MySQL database closed.")


    # def get_available_doctors_specialities(self) -> list[SpecialityEnum]:
    #     try:
    #         cursor = self.engine.connect()
    #         result = cursor.execute(text("select DISTINCT MainSpecialtyName from DocData;"))
    #         specialities = [doc['MainSpecialtyName'] for doc in result.mappings()]
    #         cursor.close()

    #         # Filter the specialties that match the Enum values
    #         available_specialities = [
    #             speciality for speciality in SpecialityEnum if speciality.value in specialities
    #         ]

    #         return available_specialities

    #     except Exception as e:
    #         logger.error(f"Error retrieving specialties: {e}")
    #         return []

    # Function to detect speciality and sub-speciality

    def get_available_doctors_specialities(self) -> list[SpecialityEnum]:
        try:
            # Return all available specialties from the enum
            return list(SpecialityEnum)

        except Exception as e:
            logger.error(f"Error retrieving specialties: {e}")
            return []

            
    def get_doctor_name_by_speciality(self, speciality: SpecialityEnum, location: str) -> list[dict[str, str | int | float | bool]]:
        try:
            cursor = self.engine.connect()
            # query = text(
            #     "select TOP 5 EntityName as DoctorName, OverallPercentage as Rating, BasicContactCity as City, MainSpecialtyName as Speciality, "
            #     "BranchAddress as Address, ImageUrl from DocData "
            #     "where MainSpecialtyName LIKE :speciality AND BasicContactCity LIKE :location;"
            # )
            # query = text(
            # "SELECT TOP 5 le.Id AS DoctorId, le.DocName AS DoctorName, le.Specialty AS Speciality, le.Fee AS Fee, le.Rating AS Rating, b.BranchName AS Branch, b.Address AS Address "
            # "FROM [DrAide_Dev].[dbo].[LowerEntity] le "
            # "JOIN [DrAide_Dev].[dbo].[Branch] b ON le.BranchId = b.Id "
            # "WHERE le.Specialty LIKE :speciality AND b.BranchName LIKE :location;"
            # )
            print("spciality")
            print(speciality)
            print("location")
            print(location)

            query = text(
                "SELECT TOP 5 le.Id AS DoctorId, le.DocName AS DoctorName, le.Specialty AS Speciality, le.Fee AS Fee, le.Rating AS Rating, le.HasDiscount AS HasDiscount, "
                "b.BranchName AS Branch, b.Address AS Address,"
                "d.DiscountType AS DiscountType, d.DiscountValue AS DiscountValue "
                "FROM [ [dbo].[LowerEntity] le "
                "JOIN  [dbo].[Branch] b ON le.BranchId = b.Id "
                "LEFT JOIN  [dbo].[Discount] d ON le.discount_id = d.discount_id "
                "WHERE le.Specialty LIKE :speciality AND b.Address LIKE :location AND le.isActive = 1;"
            )
            result = cursor.execute(query, {'speciality': f"%{speciality.value}%", 'location': f"%{location}%"})
            records = [dict(row) for row in result.mappings()]
            cursor.close()

            return records
        except Exception as e:
            logger.error(f"Error retrieving doctors for specialty: {e}")
            return []


if __name__ == "__main__":
    db = DB()
    docs = db.get_available_doctors_specialities()
    print("Available Specialities:")
    pprint(docs)

    print("Doctors for speciality: Dentistry in Jeddah")
    docs = db.get_doctor_name_by_speciality(SpecialityEnum.DENTISTRY, 'Jeddah')
    pprint(docs)
