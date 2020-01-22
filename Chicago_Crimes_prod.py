# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:52:04 2019

@author: jakisanm
"""


#Predicting a crime could save a life – Advantage back to Chicago PD
#Author - James Akisanmi

import cx_Oracle
import smtplib
import inspect
import dateutil.parser as parser
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import bigquery, storage
from google.auth import compute_engine
import collections
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")
working_directory = r'C:\Users\jakisanm\Documents'
os.chdir(working_directory)


class Chicago_Homicides_DC():
    """ 
    Data collection using Google's big query and stored in AWS's RDS platform
    
    Google’s Big Query will be the primary source for data acquisition. From which, 
    three tables will be queried to obtain the summaries of crimes, geospatial information, 
    and Global Historical Climate Network (GHCN)
    
    Methods
    -------
    sendErrorEmail
        Send email to desired recipients in case any class/method fails
    writeError
        Write specific error to error file which is located at
        share.ad.qservco.com\metronet\Analytics\Solarwinds Servers\ErrorFile.txt on the server
    sendEmailError
        Information needed in the header and body of error mail
    handleException
        Collect specific error from error prone function and pass to writerError and sendEmailError functions
    run_query
        Collect data from Google's big query based on query passed into the function
    """
    
    def __init__(self):
        """Constructing Chicago_Homicides' variables/attributes"""
        try:
            self.project = "t-rider-240616"
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)

    @staticmethod
    def sendErrorEmail(sender, receiver, body):
        # Connect to the server    #### Does not work I need to fix this
        server = smtplib.SMTP("smtp.gmail.com:587")
        server.ehlo()
        server.starttls()
        #Need to create python account for error handling
        server.login("xxxxxx", "xxxxx")
        for recipient in receiver:
            server.sendmail(sender, recipient, body)  # Send the email
        server.quit()  # Disconnect from the server

    @staticmethod
    def writeError(error_str, msg):
        inFile = open(r'\\share.ad.qservco.com\metronet\Analytics\Solarwinds Servers\ErrorFile.txt', 'a') ### Need to fix
        inFile.write('--- ' + msg + repr(error_str) + '\n\n')
        inFile.close()

    @staticmethod
    def sendEmailError(ex, curTime, definition):
        bodyStr = "--- " + curTime + "({}) ".format(definition) + ex
        email_from = "jamesakisanmi.python@gmail.com"
        email_to = ["jamesakisanmi@gmail.com"]
        header = "From: %s\n" % "James Akisanmi" \
                 + "To: %s\n" % ",".join(email_to) \
                 + "Subject: %s\n\n" % "Chicago Crimes Errors Occurred"
        email_body = header + bodyStr
        Chicago_Homicides_DC.sendErrorEmail(email_from, email_to, email_body)

    @staticmethod
    def handleException(exception):
        # Get the current datetime
        curTime = datetime.now().strftime('%m-%d-%Y / %H:%M:%S ')
        fromDef = inspect.stack()[1][3]
        # Setup message for logging. Tells what def had the error
        errorHeader = "Error ({})  {}:".format(inspect.stack()[1][3], curTime)
        # Write the error to the logfile
        Chicago_Homicides_DC.writeError(exception, errorHeader)
        Chicago_Homicides_DC.sendEmailError(repr(exception), curTime, fromDef)

        #Need to modify, personalize and make better####
    def run_query(self, query):

        try:
            client = bigquery.Client(project = self.project)
            job = client.query(query)

            query_output = collections.defaultdict(list)
            for row in job.result(timeout=600):
                for key in row.keys():
                    query_output[key].append(row.get(key))

            return pd.DataFrame(query_output)

        except Exception as e:
            Chicago_Homicides_DC.handleException(e)

class Chicago_Homicides_ETL():
    """
    ETL class with the purpose of cleaning and transforming pecularities within the dataset
        
    Methods
    -------
    merge_dfs
        Function utilized for merging ACS block group data with
        crime(homicide) data and replacing null values 
        with Zeros
    oneHotEncoder
        Encoding categorical variables in the newly merged dataframe
    followingYear
        Creates a placeholder dataframe which will be filled by
        Chicago_Homicides_Prediction.machineLearningPredictionCrimes function
    """
    
    
    
    def merge_dfs(self, crimesCount, crimesBG):
        try:
            crimes = pd.merge(crimesCount, crimesBG, how = 'left', on = 'GEO_ID')
            blankColumns = {'median_age': 0, 'median_rent' : 0, 'income_per_capita': 0} ### Need to do EDA on rent and income per
            crimes = crimes.fillna(value=blankColumns)

            return crimes

        except Exception as e:
            Chicago_Homicides_DC.handleException(e)

    def oneHotEncoder(self, crimesML):

        try:
                        
            categoricalFeatures = crimesML.columns[crimesML.dtypes == 'object']
            #one_hot encoding
            encodingCategories = pd.get_dummies(crimesML[categoricalFeatures])
            # Drop column B as it is now encoded
            crimesML = crimesML.drop(crimesML[categoricalFeatures],axis = 1)
            # Join the encoded df
            crimesML = crimesML.join(encodingCategories)
            
            return crimesML
        
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)
                
    def followingYear(self):
        
        nextYear = pd.DataFrame(np.random.randint(0,10, size=(12,3)))
        nextYear.columns =  ['CRIMES', 'AVAILABLE_MONTHS', 'YEAR']
        
        nextYear['CRIMES'] = 0
        nextYear['AVAILABLE_MONTHS'] = ''
        nextYear['YEAR'] = ''
                    
        for months in range(0, 12):
            future_months = datetime.today() + relativedelta(month = (months+1))

            cleaning_months = parser.parse(str(future_months))
            nextYear['AVAILABLE_MONTHS'][months] = cleaning_months.strftime('%B')
            nextYear['YEAR'][months] = cleaning_months.strftime('%Y')
        
        return nextYear

class Chicago_Homicides_Prediction():
    """ 
    Chicago_Homicides_Prediction class designed and created with the purpose of training the
    loaded dataframes from Chicago_Homicides_ETL class and providing predictions using the functions below.
    
    Methods
    -------
    machineLearningETLCrimes
        Used for finding the most important independent variables which are critical for the dashboard
    machineLearningPredictionCrimes
        Used for predicting the values for the next 12 months within a reasonable prediction accuracy.
    """
    
    def machineLearningETLCrimes(self, crimesML):
        
        try:
            
            IV = crimesML.loc[:, crimesML.columns != 'CRIMES']   
            DV = crimesML.loc[:, crimesML.columns == 'CRIMES']   
            
            X_train, X_test, y_train, y_test = train_test_split(IV, DV, test_size = 0.2, random_state = 0)
            
            regressor = RandomForestRegressor(random_state = 0)
            regressor.fit(X_train, y_train)
 
            y_pred = regressor.predict(X_test)

            MAE = mean_absolute_error(y_pred, y_test)
            importances = regressor.feature_importances_

            feature_importances = pd.DataFrame({'feature':crimesML.iloc[:, crimesML.columns != 'CRIMES'], 'importance':importances})
            feature_importances.sort_values(by='importance', ascending=False, inplace=True)
            feature_importances.set_index('feature', inplace=True, drop=True)

            
            return (MAE, feature_importances)
        
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)
        
    def machineLearningPredictionCrimes(self, crimesML, nextYearEncoded, nextYear):
    
        try:
            
            IV = crimesML.loc[:, crimesML.columns != 'CRIMES']   
            DV = crimesML.loc[:, crimesML.columns == 'CRIMES']   
                 
            regressor = RandomForestRegressor(random_state = 0, n_estimators = 20, max_depth = 10, max_features = 8, min_samples_split = 40)
            regressor.fit(IV, DV)
            
            nextYearTrain = nextYearEncoded.loc[:, nextYearEncoded.columns != 'CRIMES']
            nextYearPred = regressor.predict(nextYearTrain)
            
            Prediction = pd.DataFrame(nextYearPred)
            Prediction = Prediction.join(nextYear[['AVAILABLE_MONTHS', 'YEAR']])
            
            return Prediction
        
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)

class Chicago_Homicides_Integration():
    """
    Chicago_Homicides_Integration class designed and created for inserting up-to-date data into AWS Oracle Database.
    
    
    Methods
    -------
    monthlyUpdates
        Updating Oracle db table for homicides on a monthly basis from 2001; columns will only include number of homicides,
        month and year
    monthlyPredictionUpdates
        Predicting number of homicides for the coming year (i.e the next months)
    DescriptiveStatsinsertRecords
        Inserting records in Oracle db table and summarizing descriptive stats on the dashboard with the goal of 
        providing high level insights
    """    
    
    def __init__(self):
        
        try:
            self.RDSconnection = cx_Oracle.connect('xxxxxx', 'xxxxx', '''(DESCRIPTION =
                            (ADDRESS = (PROTOCOL = TCP)(HOST = portfolio.cy9ogmkxtqlc.us-east-2.rds.amazonaws.com)(PORT = 1521))
                            (CONNECT_DATA = (SERVICE_NAME = portDB)))''')
            
            self.RDScur = self.RDSconnection.cursor()  
            
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)
    
    def monthlyUpdates(self, crimesCount):
        
        try:            
            crimesList = crimesCount.values.tolist()
            truncatesStatement = """delete from CHICAGO_MONTHLY_HOMICIDES"""
            monthlyUpdateStatement = """ insert into CHICAGO_MONTHLY_HOMICIDES(CRIMES, AVAILABLE_MONTHS, YEARS)
                                    VALUES (:1, :2, :3) """
            
            self.RDScur.execute(truncatesStatement)
            self.RDScur.executemany(monthlyUpdateStatement, crimesList)    ###Need to change to self.crimesCount or ...
            
            self.RDSconnection.commit()
        
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)
    
    def monthlyPredictionUpdates(self, Prediction):
        
        try:            
            ####Need to automate to include month number***
            crimesPrediction = Prediction.values.tolist()
            truncatesStatement = """delete from CHICAGO_PREDICTION"""
            monthlyUpdateStatement = """ insert into CHICAGO_PREDICTION(CRIMES, AVAILABLE_MONTHS, YEAR)
                                    VALUES (:1, :2, :3) """
             
            self.RDScur.execute(truncatesStatement)    
            self.RDScur.executemany(monthlyUpdateStatement, crimesPrediction)    ###Need to change to self.crimesCount or ...
            
            self.RDSconnection.commit()
            
        except Exception as e:
            Chicago_Homicides_DC.handleException(e)
            
    ###Descriptive Stats
    def DescriptiveStatsinsertRecords(self, crimesDF):
        
        try:
                                      
            columnNames = ""
            columnNoList = ""
            
            for columnNumbers, columns in enumerate(crimesDF):
                
                columnNames += '"' + str(columns) + '",'
                columnNoList += ':' + str(columnNumbers+1) + ","

            columnNames = columnNames.rstrip(',')
            columnNoList = columnNoList.rstrip(',')
            
            
            crimesDF = crimesDF.applymap(str)
            
            crimesDFInsert = crimesDF.values.tolist()
            
            truncatesStatement = """delete from CHICAGO_CRIMES"""
            Chicago_Statement = """insert into jakisanm.CHICAGO_CRIMES({})values({})""".format(columnNames, columnNoList)

            self.RDScur.execute(truncatesStatement)
            self.RDScur.executemany(Chicago_Statement, crimesDFInsert) #####Need to fix datatype conversion
            self.RDSconnection.commit()
        
        except Exception as e:
                Chicago_Homicides_DC.handleException(e) 
## %%
    

if __name__ == "__main__":
    
    # Set project explicitly in the environment to suppress some warnings.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'My First Project-b2836bd00f72.json' 
    storage_client = storage.Client.from_service_account_json('My First Project-b2836bd00f72.json')
    credentials = compute_engine.Credentials()

    #####Initializing objects#######
    homicides = Chicago_Homicides_DC()
    homicides_ETL = Chicago_Homicides_ETL()
    homicides_Pred = Chicago_Homicides_Prediction()
    homicides_Integration = Chicago_Homicides_Integration()
    
    crimesDF = homicides.run_query("""
    SELECT GEO.GEO_ID, CHI.UNIQUE_KEY, CHI.CASE_NUMBER, DATETIME(DATE, "America/Chicago") DATE, CHI.BLOCK, CHI.IUCR, CHI.PRIMARY_TYPE, CHI.DESCRIPTION,
    CHI.LOCATION_DESCRIPTION, CHI.ARREST, CHI.DOMESTIC, CHI.BEAT, CHI.DISTRICT, CHI.WARD, CHI.COMMUNITY_AREA, CHI.FBI_CODE,
    CHI.X_COORDINATE, CHI.Y_COORDINATE, CHI.YEAR, CHI.UPDATED_ON, CHI.LATITUDE,CHI.LONGITUDE,CHI.LOCATION,
    EXTRACT(HOUR FROM DATETIME(DATE, "America/Chicago")) HOUR,
    FORMAT_DATE("%A", DATE (DATETIME(CHI.DATE, "America/Chicago")))AS DAY,
    FORMAT_DATE("%B", DATE (DATETIME(CHI.DATE, "America/Chicago"))) MONTH,
    ACS.TOTAL_POP, 
    ACS.MALE_POP,
    ACS.FEMALE_POP, 
    ACS.MEDIAN_AGE,
    ACS.WHITE_POP,
    ACS.BLACK_POP, 
    ACS.ASIAN_POP, 
    ACS.HISPANIC_POP,
    ACS.AMERINDIAN_POP,
    ACS.COMMUTERS_BY_PUBLIC_TRANSPORTATION COMMUTERS_BY_PUBLIC_TRANSP, 
    ACS.HOUSEHOLDS,
    ACS.INCOME_PER_CAPITA,
    ACS.HOUSING_UNITS,
    ACS.MEDIAN_RENT,
    ACS.MILLION_DOLLAR_HOUSING_UNITS, 
    ACS.MORTGAGED_HOUSING_UNITS,
    ACS.INCOME_LESS_10000,
    ACS.INCOME_10000_14999, 
    ACS.INCOME_15000_19999,
    ACS.INCOME_20000_24999, 
    ACS.INCOME_25000_29999,
    ACS.INCOME_30000_34999,
    ACS.INCOME_35000_39999,
    ACS.INCOME_40000_44999,
    ACS.INCOME_45000_49999,
    ACS.INCOME_50000_59999, 
    ACS.INCOME_60000_74999,
    ACS.INCOME_75000_99999,
    ACS.INCOME_100000_124999,
    ACS.INCOME_125000_149999,
    ACS.INCOME_150000_199999,
    ACS.INCOME_200000_OR_MORE, 
    ACS.CIVILIAN_LABOR_FORCE,
    ACS.EMPLOYED_POP,
    ACS.UNEMPLOYED_POP,
    ACS.ASSOCIATES_DEGREE,
    ACS.BACHELORS_DEGREE,
    ACS.HIGH_SCHOOL_DIPLOMA,
    ACS.MASTERS_DEGREE,
    ACS.LESS_ONE_YEAR_COLLEGE, 
    ACS.ONE_YEAR_MORE_COLLEGE
    FROM `bigquery-public-data.chicago_crime.crime` CHI   -- NEED TO UPDATE TO DIRECTLY QUERY RESULTS
    LEFT JOIN
    (SELECT GEO_ID, BLOCKGROUP_GEOM FROM `bigquery-public-data.geo_census_blockgroups.blockgroups_17`)GEO ON GEO.GEO_ID IS NOT NULL
    LEFT JOIN
    (SELECT DISTINCT GEO_ID,
    TOTAL_POP, 
    MALE_POP,
    FEMALE_POP, 
    MEDIAN_AGE,
    WHITE_POP,
    BLACK_POP, 
    ASIAN_POP, 
    HISPANIC_POP,
    AMERINDIAN_POP,
    COMMUTERS_BY_PUBLIC_TRANSPORTATION, 
    HOUSEHOLDS,
    INCOME_PER_CAPITA,
    HOUSING_UNITS,
    MEDIAN_RENT,
    MILLION_DOLLAR_HOUSING_UNITS, 
    MORTGAGED_HOUSING_UNITS,
    INCOME_LESS_10000,
    INCOME_10000_14999, 
    INCOME_15000_19999,
    INCOME_20000_24999, 
    INCOME_25000_29999,
    INCOME_30000_34999,
    INCOME_35000_39999,
    INCOME_40000_44999,
    INCOME_45000_49999,
    INCOME_50000_59999, 
    INCOME_60000_74999,
    INCOME_75000_99999,
    INCOME_100000_124999,
    INCOME_125000_149999,
    INCOME_150000_199999,
    INCOME_200000_OR_MORE, 
    CIVILIAN_LABOR_FORCE,
    EMPLOYED_POP,
    UNEMPLOYED_POP,
    ASSOCIATES_DEGREE,
    BACHELORS_DEGREE,
    HIGH_SCHOOL_DIPLOMA,
    MASTERS_DEGREE,
    LESS_ONE_YEAR_COLLEGE, 
    ONE_YEAR_MORE_COLLEGE FROM `bigquery-public-data.census_bureau_acs.blockgroup_2017_5yr`) ACS ON GEO.GEO_ID = ACS.GEO_ID
    WHERE ST_WITHIN(ST_GEOGPOINT(CHI.LONGITUDE,CHI.LATITUDE), GEO.BLOCKGROUP_GEOM)
    AND DATETIME(CHI.DATE, "America/Chicago") >= '2015-01-01'
    AND CHI.PRIMARY_TYPE = 'HOMICIDE'  
    """)
 
# For ETL purposes    
    crimesHomicidesCount = homicides.run_query("""WITH crimesCount AS (SELECT COUNT(FINAL.UNIQUE_KEY) CRIMES,  FINAL.GEO_ID, DATES, FINAL.HOURS FROM (
    SELECT UNIQUE_KEY, GEO.GEO_ID, GEO.BLOCKGROUP_GEOM,
    EXTRACT(HOUR FROM DATETIME(DATE, "America/Chicago")) AS HOURS,
    FORMAT_DATE("%Y-%m-%d", DATE (DATETIME(DATE, "America/Chicago"))) DATES,
    LONGITUDE,
    LATITUDE
    FROM `bigquery-public-data.chicago_crime.crime` CHI
    LEFT JOIN
    (SELECT GEO_ID, BLOCKGROUP_GEOM FROM `bigquery-public-data.geo_census_blockgroups.blockgroups_17`)GEO ON GEO.GEO_ID IS NOT NULL
    WHERE ST_WITHIN(ST_GeogPoint(CHI.LONGITUDE,CHI.LATITUDE), GEO.BLOCKGROUP_GEOM)
    AND  DATETIME(DATE, "America/Chicago")>= '2010-01-01'
    AND CHI.PRIMARY_TYPE = 'HOMICIDE')FINAL
    GROUP BY FINAL.GEO_ID, FINAL.DATES, FINAL.HOURS
    ORDER BY CRIMES DESC)
    
    SELECT CRIMES, GEO_ID, FORMAT_DATE("%A", DATE(TIMESTAMP(DATES))) DOW, 
    FORMAT_DATE("%B", DATE(TIMESTAMP(DATES))) MONTH, DATES, CAST(HOURS AS STRING) HOURS FROM crimesCount
    """)
        
    crimesBG = homicides.run_query("""
    SELECT ACS.GEO_ID,
    GEO.BLOCKGROUP_GEOM,
    ACS.total_pop, 
    ACS.male_pop,
    ACS.female_pop, 
    ACS.median_age,
    ACS.white_pop,
    ACS.black_pop, 
    ACS.asian_pop, 
    ACS.hispanic_pop,
    ACS.amerindian_pop,
    ACS.commuters_by_public_transportation, 
    ACS.households,
    ACS.income_per_capita,
    ACS.housing_units,
    ACS.median_rent,
    ACS.million_dollar_housing_units, 
    ACS.mortgaged_housing_units,
    ACS.income_less_10000,
    ACS.income_10000_14999, 
    ACS.income_15000_19999,
    ACS.income_20000_24999, 
    ACS.income_25000_29999,
    ACS.income_30000_34999,
    ACS.income_35000_39999,
    ACS.income_40000_44999,
    ACS.income_45000_49999,
    ACS.income_50000_59999, 
    ACS.income_60000_74999,
    ACS.income_75000_99999,
    ACS.income_100000_124999,
    ACS.income_125000_149999,
    ACS.income_150000_199999,
    ACS.income_200000_or_more, 
    ACS.civilian_labor_force,
    ACS.employed_pop,
    ACS.unemployed_pop,
    ACS.associates_degree,
    ACS.bachelors_degree,
    ACS.high_school_diploma,
    ACS.masters_degree,
    ACS.less_one_year_college, 
    ACS.one_year_more_college
    FROM `bigquery-public-data.census_bureau_acs.blockgroup_2017_5yr` ACS
    RIGHT JOIN
    (SELECT GEO_ID, BLOCKGROUP_GEOM FROM `bigquery-public-data.geo_census_blockgroups.blockgroups_17`)GEO ON ACS.GEO_ID = GEO.GEO_ID
    """)    
    
    #####Descriptive Stat########
    homicides_Integration.DescriptiveStatsinsertRecords(crimesDF)
    #################################
   
    ########ETL Cycle##############
    crimesML = homicides_ETL.merge_dfs(crimesHomicidesCount, crimesBG)
    crimesML = crimesML[[
                       'CRIMES', 'GEO_ID', 'DOW', 'MONTH', 'HOURS',
                       'total_pop', 'male_pop', 'female_pop', 'median_age', 'white_pop',
                       'black_pop', 'asian_pop', 'hispanic_pop', 'amerindian_pop',
                       'commuters_by_public_transportation', 'households', 'income_per_capita',
                       'housing_units', 'median_rent', 'million_dollar_housing_units',
                       'mortgaged_housing_units', 'income_less_10000', 'income_10000_14999',
                       'income_15000_19999', 'income_20000_24999', 'income_25000_29999',
                       'income_30000_34999', 'income_35000_39999', 'income_40000_44999',
                       'income_45000_49999', 'income_50000_59999', 'income_60000_74999',
                       'income_75000_99999', 'income_100000_124999', 'income_125000_149999',
                       'income_150000_199999', 'income_200000_or_more', 'civilian_labor_force',
                       'employed_pop', 'unemployed_pop', 'associates_degree',
                       'bachelors_degree', 'high_school_diploma', 'masters_degree',
                       'less_one_year_college', 'one_year_more_college'
                       ]]

    crimesML = homicides_ETL.oneHotEncoder(crimesML)
    MSE, feature_importances = homicides_Pred.machineLearningETLCrimes(crimesML)

    #Prediction cycle##############
    #Months from year 2001
    crimesCount = homicides.run_query("""
    SELECT COUNT(CHI.UNIQUE_KEY) CRIMES,
    FORMAT_DATE("%B", DATE (DATETIME(CHI.DATE, "America/Chicago"))) AVAILABLE_MONTHS,
    CHI.YEAR FROM `bigquery-public-data.chicago_crime.crime` CHI
    WHERE CHI.PRIMARY_TYPE = 'HOMICIDE'
    GROUP BY FORMAT_DATE("%B", DATE (DATETIME(CHI.DATE, "America/Chicago"))), CHI.YEAR
                                      """)

    homicides_Integration.monthlyUpdates(crimesCount) ####Database Table monthly Update
    
    crimesML = crimesCount[['CRIMES', 'AVAILABLE_MONTHS']]
    crimesML = homicides_ETL.oneHotEncoder(crimesML)
    nextYear = homicides_ETL.followingYear() ####Get prediction months for coming year
    nextYearEncoded = homicides_ETL.oneHotEncoder(nextYear[['CRIMES', 'AVAILABLE_MONTHS']])    
    Prediction = homicides_Pred.machineLearningPredictionCrimes(crimesML, nextYearEncoded, nextYear)
    
    homicides_Integration.monthlyPredictionUpdates(Prediction) ####push to production table
    
    ######End of prediction cyle#########