"""
Test module for query_builder_agent.py to validate WHERE clause generation
"""
import logging
import unittest
from pprint import pprint
from .query_builder_agent import SearchCriteria, build_query

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestQueryBuilder(unittest.TestCase):
    """Test the query builder functions"""
    
    def test_specialty_where_clause(self):
        """Test WHERE clause generation for specialty criteria"""
        criteria = SearchCriteria(
            speciality="Cardiology",
            location="Riyadh"
        )
        query, params = build_query(criteria)
        
        logger.info(f"Specialty WHERE Clause: {params['@DynamicWhereClause']}")
        self.assertIn("le.Specialty = N'Cardiology'", params["@DynamicWhereClause"])
        self.assertIn("b.Address_en LIKE N'%Riyadh%'", params["@DynamicWhereClause"])
    
    def test_doctor_name_where_clause(self):
        """Test WHERE clause generation for doctor name criteria"""
        criteria = SearchCriteria(doctor_name="Ahmed")
        query, params = build_query(criteria)
        
        logger.info(f"Doctor Name WHERE Clause: {params['@DynamicWhereClause']}")
        self.assertIn("le.DocName_en LIKE N'%Ahmed%'", params["@DynamicWhereClause"])
    
    def test_complex_where_clause(self):
        """Test WHERE clause generation for complex criteria"""
        criteria = SearchCriteria(
            speciality="Pediatrics",
            subspeciality="Neonatology,Pediatric Cardiology",
            location="Jeddah",
            min_rating=4.0,
            min_price=100,
            max_price=500
        )
        query, params = build_query(criteria)
        
        logger.info(f"Complex WHERE Clause: {params['@DynamicWhereClause']}")
        where_clause = params["@DynamicWhereClause"]
        self.assertIn("le.Specialty = N'Pediatrics'", where_clause)
        self.assertIn("N'Neonatology'", where_clause)
        self.assertIn("N'Pediatric Cardiology'", where_clause)
        self.assertIn("b.Address_en LIKE N'%Jeddah%'", where_clause)
        self.assertIn("le.Rating >= 4.0", where_clause)
        self.assertIn("le.Fee BETWEEN 100.0 AND 500.0", where_clause)
    
    def test_branch_name_where_clause(self):
        """Test WHERE clause generation for branch name criteria"""
        criteria = SearchCriteria(branch_name="Deep Care Clinic")
        query, params = build_query(criteria)
        
        logger.info(f"Branch Name WHERE Clause: {params['@DynamicWhereClause']}")
        self.assertIn("b.BranchName_en LIKE N'%Deep Care Clinic%'", params["@DynamicWhereClause"])
    
    def test_escaping_single_quotes(self):
        """Test handling of single quotes in criteria values"""
        criteria = SearchCriteria(
            speciality="Children's Health",
            branch_name="St. Mary's Hospital"
        )
        query, params = build_query(criteria)
        
        logger.info(f"Escaped Quotes WHERE Clause: {params['@DynamicWhereClause']}")
        where_clause = params["@DynamicWhereClause"]
        self.assertIn("le.Specialty = N'Children''s Health'", where_clause)
        self.assertIn("b.BranchName_en LIKE N'%St. Mary''s Hospital%'", where_clause)

def run_tests():
    """Run all the query builder tests"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
if __name__ == "__main__":
    run_tests() 