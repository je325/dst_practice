import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

pd.set_option('display.max_rows', 500)

#Reading in the dataframe
df = pd.read_csv('data/InputData.csv')

####making a new data frame with the US Tech Companies (This can be easily adjusted to account for different key words we would be looking for)
df["Industry Groups"] = df["Industry Groups"].str.lower()
tech_keywords = ["technology", "software", "internet"]
industry_keywords = ["consumer internet", "software"]

filtered_df = df[df["Industry Groups"].str.contains("|".join(tech_keywords)) &
                 df["Industry Groups"].str.contains("|".join(industry_keywords))]

new_df = filtered_df
#new_df.to_csv("COMPANIES_JUST_SOFTWARE.csv")

# Checking for columns where every value is the same
same_value_columns = new_df.columns[new_df.nunique() == 1]

# Dropping columns where every value is the same
data = new_df.drop(columns=same_value_columns)
data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")

# Count non-null values for each column
non_null_counts = data.count()

# Create a DataFrame to store the counts
#non_null_counts_df = pd.DataFrame(non_null_counts, columns=['Non-Null Count'])

# Display the DataFrame
#print(non_null_counts_df.sort_values(by="Non-Null Count"))

# Define a threshold for the number of non-null values required to keep the column
threshold = 500  # Adjust this threshold according to your requirement

# Filter out columns where the number of non-null values is below the threshold
columns_to_keep = non_null_counts[non_null_counts >= threshold].index

# Keep only the selected columns
filtered_df = data[columns_to_keep]
other_cols = 'Number of Acquisitions'
#filtered_df.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")
filtered_df_2 = pd.concat([filtered_df, data[other_cols]], axis = 1)

#columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Industry Groups", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Top 5 Investors", "LinkedIn", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Amount Currency (in USD)", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
data = filtered_df_2[columns_to_keep_2]

#data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

pd.set_option('display.max_rows', 500)

#Reading in the dataframe
df = pd.read_csv('data/InputData.csv')

####making a new data frame with the US Tech Companies (This can be easily adjusted to account for different key words we would be looking for)
df["Industry Groups"] = df["Industry Groups"].str.lower()
tech_keywords = ["technology", "software", "internet"]
industry_keywords = ["consumer internet", "software"]

filtered_df = df[df["Industry Groups"].str.contains("|".join(tech_keywords)) &
                 df["Industry Groups"].str.contains("|".join(industry_keywords))]

new_df = filtered_df
#new_df.to_csv("COMPANIES_JUST_SOFTWARE.csv")

# Checking for columns where every value is the same
same_value_columns = new_df.columns[new_df.nunique() == 1]

# Dropping columns where every value is the same
data = new_df.drop(columns=same_value_columns)
data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")

# Count non-null values for each column
non_null_counts = data.count()

# Create a DataFrame to store the counts
#non_null_counts_df = pd.DataFrame(non_null_counts, columns=['Non-Null Count'])

# Display the DataFrame
#print(non_null_counts_df.sort_values(by="Non-Null Count"))

# Define a threshold for the number of non-null values required to keep the column
threshold = 500  # Adjust this threshold according to your requirement

# Filter out columns where the number of non-null values is below the threshold
columns_to_keep = non_null_counts[non_null_counts >= threshold].index

# Keep only the selected columns
filtered_df = data[columns_to_keep]
other_cols = 'Number of Acquisitions'
#filtered_df.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")
filtered_df_2 = pd.concat([filtered_df, data[other_cols]], axis = 1)

#columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Industry Groups", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Top 5 Investors", "LinkedIn", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Amount Currency (in USD)", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
data = filtered_df_2[columns_to_keep_2]

#data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

pd.set_option('display.max_rows', 500)

#Reading in the dataframe
df = pd.read_csv('data/InputData.csv')

####making a new data frame with the US Tech Companies (This can be easily adjusted to account for different key words we would be looking for)
df["Industry Groups"] = df["Industry Groups"].str.lower()
tech_keywords = ["technology", "software", "internet"]
industry_keywords = ["consumer internet", "software"]

filtered_df = df[df["Industry Groups"].str.contains("|".join(tech_keywords)) &
                 df["Industry Groups"].str.contains("|".join(industry_keywords))]

new_df = filtered_df
#new_df.to_csv("COMPANIES_JUST_SOFTWARE.csv")

# Checking for columns where every value is the same
same_value_columns = new_df.columns[new_df.nunique() == 1]

# Dropping columns where every value is the same
data = new_df.drop(columns=same_value_columns)
data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")

# Count non-null values for each column
non_null_counts = data.count()

# Create a DataFrame to store the counts
#non_null_counts_df = pd.DataFrame(non_null_counts, columns=['Non-Null Count'])

# Display the DataFrame
#print(non_null_counts_df.sort_values(by="Non-Null Count"))

# Define a threshold for the number of non-null values required to keep the column
threshold = 500  # Adjust this threshold according to your requirement

# Filter out columns where the number of non-null values is below the threshold
columns_to_keep = non_null_counts[non_null_counts >= threshold].index

# Keep only the selected columns
filtered_df = data[columns_to_keep]
other_cols = 'Number of Acquisitions'
#filtered_df.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")
filtered_df_2 = pd.concat([filtered_df, data[other_cols]], axis = 1)

#columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Industry Groups", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Top 5 Investors", "LinkedIn", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Amount Currency (in USD)", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
columns_to_keep_2 = ["Organization Name", "Total Funding Amount Currency (in USD)", "Last Funding Type", "Founded Date", "Last Funding Date", "Last Funding Amount Currency (in USD)", "Total Equity Funding Amount Currency (in USD)", "Number of Articles", "Number of Founders", "Number of Employees", "Number of Funding Rounds", "Funding Status", "Last Equity Funding Type", "Number of Lead Investors", "Number of Investors", "Number of Acquisitions", "Similar Companies", "SEMrush - Monthly Visits", "SEMrush - Global Traffic Rank"]
data = filtered_df_2[columns_to_keep_2]

#data.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")


# Convert the ranges to numerical values (e.g., the midpoints of the ranges)
data_copy['Number of Employees'] = data_copy['Number of Employees'].str.split('-').apply(lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) else x)
# Fill missing values with the median of the available ranges
median_employees = data_copy['Number of Employees'].median()
data_copy['Number of Employees'].fillna(median_employees, inplace=True)

# Fill missing values with median for other numerical columns
median_cols = ['Number of Lead Investors', 'Number of Investors', 'Similar Companies', 'SEMrush - Monthly Visits',
               'Number of Articles', 'SEMrush - Global Traffic Rank']
for col in median_cols:
    data_copy[col] = data_copy[col].apply(lambda x: str(x).replace(',', '') if pd.notnull(x) else x)
    data_copy[col] = pd.to_numeric(data_copy[col],
                                   errors='coerce')  # Convert to numeric, converting non-numeric values to NaN
    median_val = data_copy[col].median()
    data_copy[col].fillna(median_val, inplace=True)

# Changing Funding Dates to Years
data_copy['founded_date'] = pd.to_datetime(data_copy['Founded Date'])
data_copy['founded_year'] = data_copy['founded_date'].dt.year
data_copy.drop('founded_date', axis=1, inplace=True)
data_copy.drop('Founded Date', axis=1, inplace=True)


data_copy['last_funding_date'] = pd.to_datetime(data_copy['Last Funding Date'])
data_copy['last_funding_year'] = data_copy['last_funding_date'].dt.year
data_copy.drop('last_funding_date', axis=1, inplace=True)
data_copy.drop('Last Funding Date', axis=1, inplace=True)



#data_copy.to_csv("COMPANIES_JUST_SOFTWARE_UPDATED.csv")
data_copy_original = data_copy

# Convert categorical variables into dummy/indicator variables
data_copy = pd.get_dummies(data_copy, columns=['Last Funding Type', 'Funding Status', 'Last Equity Funding Type'])

# Splitting the data into features and target variable
X = data_copy.drop(columns=['Organization Name', 'Total Funding Amount Currency (in USD)'])
y = data_copy['Total Funding Amount Currency (in USD)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Make predictions
predictions = model.predict(X_test)

# Evaluating the model
print("R^2 score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))


# Identify top startups based on predictions and defined criteria
#top_startups_indices = np.where((predictions < 1e8) & (predictions > 2e8))[0]  # Ensure to access the first element of the tuple returned by np.where
# Filter top startups whose original Total Funding Amount Currency (in USD) is less than 100 million
top_startups_filtered = data_copy[data_copy['Total Funding Amount Currency (in USD)'] < 100000000]

# Sort filtered top startups by predicted funding amount
top_startups_filtered_sorted = top_startups_filtered.sort_values(by='Total Funding Amount Currency (in USD)', ascending=False)

print("Top 10 Startups (Sorted by Predicted Likelihood and Total Funding < $100M):")
print(top_startups_filtered_sorted.head(10))

feature_importances = model.feature_importances_
# print("Feature Importances:")
# for feature, importance in zip(X.columns, feature_importances):
print(f"{feature}: {importance}")

# Print the top three most important features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_three_features = X.columns[indices][:3]
top_three_importances = importances[indices][:3]

print("Top Three Most Important Features:")
for feature, importance in zip(top_three_features, top_three_importances):
    print(f"{feature}: {importance}")

data_copy = data_copy_original
# Convert categorical variables into dummy/indicator variables
data_copy = pd.get_dummies(data_copy, columns=['Last Funding Type', 'Funding Status', 'Last Equity Funding Type'])

# Splitting the data into features and target variable
X = data_copy.drop(columns=['Organization Name', 'Total Funding Amount Currency (in USD)', 'Total Equity Funding Amount Currency (in USD)'])
y = data_copy['Total Funding Amount Currency (in USD)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Make predictions
predictions = model.predict(X_test)

# Evaluating the model
print("R^2 score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))


# Identify top startups based on predictions and defined criteria
#top_startups_indices = np.where((predictions < 1e8) & (predictions > 2e8))[0]  # Ensure to access the first element of the tuple returned by np.where
top_startups_filtered = data_copy[data_copy['Total Funding Amount Currency (in USD)'] < 100000000]

# Sort filtered top startups by predicted funding amount
top_startups_filtered_sorted = top_startups_filtered.sort_values(by='Total Funding Amount Currency (in USD)', ascending=False)

print("Top 10 Startups (Sorted by Predicted Likelihood and Total Funding < $100M):")
print(top_startups_filtered_sorted.head(10))

feature_importances = model.feature_importances_
print("Feature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# Print the top three most important features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_three_features = X.columns[indices][:3]
top_three_importances = importances[indices][:3]

print("Top Three Most Important Features:")
for feature, importance in zip(top_three_features, top_three_importances):
    print(f"{feature}: {importance}")

    