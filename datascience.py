import pandas as pd 
import numpy as np 
import streamlit as st 
import sklearn 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer



st.header("Loan Prediction App: Cleaning")

st.subheader("This is a app designed to predict the likelyhood of someone defaulting on paying a loan.")

st.write("First, we load the data.")

data = pd.read_csv("C:\\Users\\Joel\\Documents\\athlete_events.csv")

st.write(data)

st.write("Now lets clean the dataset, first look for missing values:")

def check_data_quality(data):
    quality_report = {
        'missing_values': data.isnull().sum().to_dict(),
        'duplicates': data.duplicated().sum(),
        'total_rows': len(data)
        }
    return quality_report



def standardize_datatypes(data):
	for column in data.columns:
		#Try converting string dates to datetimes
		if data[column].dtype == 'object':
			try:
				data[column] = pd.to_datetime(data[column])
				st.write(f"Converted {column} to datetime")
			except ValueError:
				try:
					data[column] = pd.to_numeric(data[column].str.replace('$', "").str.replace(',', ''))
					st.write(f"Converted {column} to numeric")
				except:
					pass 
	return data 




#This is a function to impute missing values:
def handle_missing_values(data):
	#Handle numerc columns
	numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns 
	if len(numeric_columns) > 0:
		num_imputer = SimpleImputer(strategy='median')
		data[numeric_columns] = num_imputer.fit_transform(data[numeric_columns])

	#Handle catergorical columns
	categorical_columns = data.select_dtypes(include=['object']).columns
	if len(categorical_columns) > 0:
		cat_imputer = SimpleImputer(strategy='most_frequent')
		data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])
	return data 


#Step 4: Detect and handle Outliers:
def remove_outliers(data):
	numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
	outliers_removed = {}

	for column in numeric_columns:
		Q1 = data[column].quantile(0.25)
		Q3 = data[column].quantile(0.25)
		IQR = Q3 - Q1
		lower_bound = Q1 - 1.5 * IQR
		upper_bound = Q3 + 1.5 * IQR

		#Count outliers before removing
		outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]

		#Count the values instead of removing them
		data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

		if outliers > 0:
			outliers_removed[column] = outliers

	return data, outliers_removed


#Now validate the results:
def validate_cleaning(data, original_shape, cleaning_report):
	validation_results = {
	  'rows_remaining': len(data),
	  'missing_values_remaining': data.isnull().sum().sum(),
	  'duplicates_remaining': data.duplicated().sum(),
	  'data_loss_percentage': (1 - len(data)/original_shape[0]) * 100
	}
	#Add validation results to the cleaning report
	cleaning_report['validation'] = validation_results
	return cleaning_report


#Final report:
def automated_cleaning_pipeline(data):
	#Store original shape for reporting
	original_shape = data.shape 

	#Initialize cleaning report
	cleaning_report = {}

	#Executed each step and collect metrics
	cleaning_report['initial_quality'] = check_data_quality(data)

	data = standardize_datatypes(data)
	data = handle_missing_values(data)
	data, outliers = remove_outliers(data)
	cleaning_report['outliers_removed'] = outliers 

	#Validate and finalize report
	cleaning_report = validate_cleaning(data, original_shape, cleaning_report)

	return data, cleaning_report


st.write("Check Data Quality")
st.write(check_data_quality(data))


st.write("Handle Missing Values")
st.write(handle_missing_values(data))


st.button("Remove Outliers")
st.write(remove_outliers(data))


st.write("Cleaning and Finalized Report")
st.write(automated_cleaning_pipeline(data))

global numeric_columns
clean_data = data

st.write("Cleaned Data")
st.write(clean_data)
numeric_columns = list(clean_data.select_dtypes(['float', 'int']).columns)

filtered_data1 = st.multiselect("Filter column", options=list(clean_data.columns), default=list(clean_data.columns))

st.write("selected column:")
st.write(data[filtered_data1])

st.sidebar.header("Data visualization")

chart_select = st.sidebar.selectbox(
	label="Select the chart type",
	options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
	)

if chart_select == 'Scatterplots':
	st.sidebar.subheader("Scatterplot Settings")
	x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
	y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
	st.write(st.bar_chart(data=clean_data, x=x_values, y=y_values))





