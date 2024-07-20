import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('eda_data.csv')

# Display the columns in the dataset
df.columns

# Select relevant columns for the model
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
               'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

# Create dummy variables for categorical features
df_dum = pd.get_dummies(df_model, dtype=int)

# Train-test split
from sklearn.model_selection import train_test_split

# Separate independent variables (X) and dependent variable (y)
X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple linear regression
import statsmodels.api as sm

# Add a constant to the model (intercept)
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

# Linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Cross-validation for linear regression
np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# Lasso regression
lm_l = Lasso(alpha=0.13)
lm_l.fit(X_train, y_train)

# Cross-validation for Lasso regression
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# Tuning alpha for Lasso regression
alpha = []
error = []

for i in range(1, 100):
    alpha.append(i / 100)
    lml = Lasso(alpha=(i / 100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# Plot alpha vs. error
plt.plot(alpha, error)

# Create a DataFrame for alpha and error
err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])

# Find the alpha with the maximum error
df_err[df_err.error == max(df_err.error)]

# Random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

# Cross-validation for random forest
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

# Tune models with GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameters for grid search
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('squared_error', 'absolute_error', 'poisson', 'friedman_mse'), 'max_features': ('auto', 'sqrt', 'log2')}

# Perform grid search with cross-validation
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

# Best score and estimator from grid search
gs.best_score_
gs.best_estimator_

# Test ensemble models
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

# Mean absolute error for linear regression
mean_absolute_error(y_test, tpred_lm)

# Mean absolute error for Lasso regression
mean_absolute_error(y_test, tpred_lml)

# Mean absolute error for random forest
mean_absolute_error(y_test, tpred_rf)

# Mean absolute error for average of linear regression and random forest
mean_absolute_error(y_test, (tpred_lm + tpred_rf) / 2)

columns = X_train.columns.tolist()

# Save the model using pickle
import pickle
pickl = {'model': gs.best_estimator_, 'columns': columns}
pickle.dump(pickl, open('model_file.p', 'wb'))

# Load the saved model
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

# Predict using the loaded model
model.predict(np.array(list(X_test.iloc[8, :])).reshape(1, -1))[0]

# Display the features of the second test example
list(X_test.iloc[1, :])
