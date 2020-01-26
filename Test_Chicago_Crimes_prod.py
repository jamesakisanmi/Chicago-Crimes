# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:34:21 2019

@author: jakisanm
"""

import Chicago_Crimes_prod
from pandas.util.testing import assert_frame_equal, assert_index_equal
from coverage import Coverage
import unittest
import os
import pandas as pd
import cx_Oracle


class TestGround(unittest.TestCase):
    
    def setUp(self):
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'My First Project-b2836bd00f72.json'         
        
        self.cov = Coverage()
        self.cov.start()
        
        self.homicides = Chicago_Crimes_prod.Chicago_Homicides_DC()
        self.homicides_ETL = Chicago_Crimes_prod.Chicago_Homicides_ETL()
        self.homicides_Pred = Chicago_Crimes_prod.Chicago_Homicides_Prediction()
        self.homicides_Integration = Chicago_Crimes_prod.Chicago_Homicides_Integration()
        
        self.crimesCount = self.homicides.run_query("""
        SELECT COUNT(CHI.UNIQUE_KEY) CRIMES,
        FORMAT_DATE("%B", DATE (DATETIME(CHI.DATE, "America/Chicago"))) AVAILABLE_MONTHS,
        CHI.YEAR FROM `bigquery-public-data.chicago_crime.crime` CHI
        WHERE CHI.PRIMARY_TYPE = 'HOMICIDE'
        GROUP BY FORMAT_DATE("%B", DATE (DATETIME(CHI.DATE, "America/Chicago"))), CHI.YEAR
                                          """)
    
    def tearDown(self):

        self.cov.stop()
        self.cov.html_report(directory=r'C:\Users\jakisanm\Desktop\Analytics\Chicago Crimes Prod\htmlcov')
        self.cov.erase()
        

class TestChicago_Homicides_DC(TestGround):
        
    def test_dataCollectionCols(self):
                
        expectedCols = 3
        resultCols = len(self.crimesCount.columns)
        self.assertEqual(expectedCols, resultCols)
    
    def test_dataCollectionRows(self):
        
        #Number of months from Jan 2001 to Jan 2020 = 229
        expectedRows = 229
        resultRows = len(self.crimesCount)
        
        self.assertGreaterEqual(resultRows, expectedRows)
        

class TestChicago_Homicides_Prediction(TestGround):
   
    def test_Prediction(self):
                    
        crimesML = self.crimesCount[['CRIMES', 'AVAILABLE_MONTHS']]
        crimesML = self.homicides_ETL.oneHotEncoder(crimesML)
        nextYear = self.homicides_ETL.followingYear()
        nextYearEncoded = self.homicides_ETL.oneHotEncoder(nextYear[['CRIMES', 'AVAILABLE_MONTHS']])    
        Prediction = self.homicides_Pred.machineLearningPredictionCrimes(crimesML, nextYearEncoded, nextYear)
        
        expected = 12
        result = len(Prediction)
        
        self.assertEqual(expected, result)
        self.assertIsNotNone(Prediction)
            
class TestChicago_Homicides_Integration(TestGround):
           
    def test_OracleConnection(self):
        
        RDSconnection = cx_Oracle.connect('xxxx', 'xxxx', '''(DESCRIPTION =
                            (ADDRESS = (PROTOCOL = TCP)(HOST = portfolio.cy9ogmkxtqlc.us-east-2.rds.amazonaws.com)(PORT = 1521))
                            (CONNECT_DATA = (SERVICE_NAME = portDB)))''')
    
        testStatement = """SELECT YEARS FROM CHICAGO_MONTHLY_HOMICIDES ORDER BY YEARS ASC"""
            
        testYear = pd.read_sql(testStatement, RDSconnection)
        
        expected = '2001'
        result = testYear['YEARS'][0]
        self.assertEqual(expected, result)
        
def main():
    TestGround()
## %%    
if __name__ == '__main__':
    unittest.main()    

    
    
