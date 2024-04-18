import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFECV

from IPython.display import display

# Load the data
data = pd.read_excel('../data/TestModel_Data.xlsx')
# Define the features and target variable
features = ['(Crunchbase) Monthly Visits to Company Home Page (m)',
       '(Tracxn) Number of New Articles since Last Year',
       '(Crunchbase) Number of Acquisitions',
       '(Crunchbase) Number of Patents Granted',
       '(Crunchbase) Competitor Max Total Funding Amount ($m)',
       '(App Annie) Average Change in Downloads (k)',
       '(Crunchbase) Growth in Page Views:Visit Ratio',
       '(Crunchbase) Competitor Average Funding Amount ($m)',
       '(Tracxn) Change in Twitter Followers over 6 Months (k)',
       '(Glassdoor) CEO Rating',
       '(Crunchbase) Website Monthly Rank Change (#)',
       '(Tracxn) Number of Twitter Followers (k)',
       '(Crunchbase) 90 Days Trend Score', '(Crunchbase) Number of Founders',
       '(Tracxn) Score',
       '(Twitter) Maximum Number of Followers of Competitor (k)',
       '(LinkedIn) 2 Years Headcount Growth (%)',
       '(Crunchbase) Total Products Active',
       '(Crunchbase) Minimum Number of Articles about Competitors']
target = 'Valuation ($m)'
#print(data.shape) (1297,21)
# Filter out rows with 'n.a.' in Valuation column
data = data[data[target] != 'n.a.']
#print(data.shape) (728,21)

##Converting data to numeric
numeric_columns = data.columns[1:]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data.to_csv("CHECKIN.csv")

####
# Check for NaN values in the entire DataFrame
nan_values = data.isna().sum()

# Filter columns with NaN values
columns_with_nan = nan_values[nan_values > 0]

# Print columns with NaN values and their corresponding counts
#print("Columns with NaN values:")
#print(columns_with_nan)

#column_name = "(Crunchbase) Number of Patents Granted"
#rows_with_na = data[data[column_name].isna()]
#print(rows_with_na[["Company ID", "Valuation ($m)", "(Crunchbase) Number of Patents Granted"]])

data["(Crunchbase) Number of Patents Granted"].fillna(0, inplace=True)
data["(App Annie) Average Change in Downloads (k)"].fillna(0, inplace=True)
data["(Glassdoor) CEO Rating"].fillna(0, inplace=True)



#
#
#
# # Define binary target variables for $1 billion and $10 billion valuations
data['$1Billion'] = (data[target] >= 1000).astype(int)
data['$10Billion'] = (data[target] >= 10000).astype(int)

# Select features and target variable
X = data[features]
y_1billion = data['$1Billion']
y_10billion = data['$10Billion']
# #
# # Split the data into training and testing sets
X_train, X_test, y_train_1billion, y_test_1billion = train_test_split(X, y_1billion, test_size=0.2, random_state=42)
X_train, X_test, y_train_10billion, y_test_10billion = train_test_split(X, y_10billion, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression models
model_1billion = LogisticRegression()
model_10billion = LogisticRegression()
#
model_1billion.fit(X_train_scaled, y_train_1billion)
model_10billion.fit(X_train_scaled, y_train_10billion)

# Predictions
y_pred_1billion = model_1billion.predict(X_test_scaled)
y_pred_10billion = model_10billion.predict(X_test_scaled)
#
# # Model evaluation
print("Model evaluation for $1 Billion Valuation:")
print("Accuracy:", accuracy_score(y_test_1billion, y_pred_1billion))
print("Classification Report:")
print(classification_report(y_test_1billion, y_pred_1billion))

print("\nModel evaluation for $10 Billion Valuation:")
print("Accuracy:", accuracy_score(y_test_10billion, y_pred_10billion))
print("Classification Report:")
print(classification_report(y_test_10billion, y_pred_10billion))
# Coefficients and feature importance
coefficients_1billion = model_1billion.coef_[0]
coefficients_10billion = model_10billion.coef_[0]

feature_names = X.columns

coefficients_df_1billion = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients_1billion})
coefficients_df_1billion['Absolute Coefficient'] = coefficients_df_1billion['Coefficient'].abs()
coefficients_df_1billion = coefficients_df_1billion.sort_values(by='Absolute Coefficient', ascending=False)

coefficients_df_10billion = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients_10billion})
coefficients_df_10billion['Absolute Coefficient'] = coefficients_df_10billion['Coefficient'].abs()
coefficients_df_10billion = coefficients_df_10billion.sort_values(by='Absolute Coefficient', ascending=False)

print("\nFeature Importances for $1 Billion Valuation:")
print(coefficients_df_1billion.head(10))

print("\nFeature Importances for $10 Billion Valuation:")
print(coefficients_df_10billion.head(10))

# Predict probabilities of success for each company
probabilities_1billion = model_1billion.predict_proba(X_test_scaled)[:, 1]
probabilities_10billion = model_10billion.predict_proba(X_test_scaled)[:, 1]

# Combine probabilities and company IDs
results_1billion = pd.DataFrame({'Company ID': X_test.index, 'Probability of $1 Billion Valuation': probabilities_1billion})
results_10billion = pd.DataFrame({'Company ID': X_test.index, 'Probability of $10 Billion Valuation': probabilities_10billion})

# Sort by probability in descending order
top_companies_1billion = results_1billion.sort_values(by='Probability of $1 Billion Valuation', ascending=False).head(10)
top_companies_10billion = results_10billion.sort_values(by='Probability of $10 Billion Valuation', ascending=False).head(10)

# Print the top companies
print("\nTop companies likely to reach $1 Billion valuation:")
print(top_companies_1billion)

print("\nTop companies likely to reach $10 Billion valuation:")
print(top_companies_10billion)

# Feature importance analysis using Recursive Feature Elimination with Cross-Validation (RFECV)
def perform_feature_selection(model, X_train, y_train):
    selector = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
    selector.fit(X_train, y_train)
    selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
    return selector, selected_features

selector_1billion, selected_features_1billion = perform_feature_selection(model_1billion, X_train_scaled, y_train_1billion)
selector_10billion, selected_features_10billion = perform_feature_selection(model_10billion, X_train_scaled, y_train_10billion)

print("\nSelected features for $1 Billion Valuation:")
print(selected_features_1billion)

print("\nSelected features for $10 Billion Valuation:")
print(selected_features_10billion)

# Train models with selected features
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

def train_evaluate_selected_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)
    return acc, report

X_train_selected_1billion = selector_1billion.transform(X_train_scaled)
X_test_selected_1billion = selector_1billion.transform(X_test_scaled)

X_train_selected_10billion = selector_10billion.transform(X_train_scaled)
X_test_selected_10billion = selector_10billion.transform(X_test_scaled)

acc_1billion_selected, report_1billion_selected = train_evaluate_selected_model(model_1billion, X_train_selected_1billion, y_train_1billion, X_test_selected_1billion, y_test_1billion)
acc_10billion_selected, report_10billion_selected = train_evaluate_selected_model(model_10billion, X_train_selected_10billion, y_train_10billion, X_test_selected_10billion, y_test_10billion)

print("\nModel evaluation for $1 Billion Valuation after feature selection:")
print("Accuracy:", acc_1billion_selected)
print("Classification Report:")
print(report_1billion_selected)

print("\nModel evaluation for $10 Billion Valuation after feature selection:")
print("Accuracy:", acc_10billion_selected)
print("Classification Report:")
print(report_10billion_selected)