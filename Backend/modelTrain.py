import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE  
import joblib

import pickle

data = pd.read_csv(r"C:\Users\lucifer\Favorites\Desktop\NewNew\Rainfall.csv")
print(data.head())

print("Data Info:\n", data.info())
print(data.columns)
data.columns = data.columns.str.strip()
print(data.columns)
data.drop(columns=["day"], inplace=True)
print(data.head())

#checking no. of missing values
print(data.isnull().sum())

#handle missing values
print(data["winddirection"].unique())
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
print(data.isnull().sum())

#change rainfall yes/no into 1 and 0 respectively
print(data["rainfall"].unique())
data["rainfall"] = data["rainfall"].map({"yes" : 1,
                      "no" : 0})
print(data["rainfall"].unique())

#exploratory data analysis(eda)
print(data.shape)

sns.set_theme(style="whitegrid") #setting plot style for all the plots
print()
print(data.describe())
print(data.columns)

#plot
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
#     plt.subplot(3,3,i)
#     sns.histplot(data[column], kde=True)
#     plt.title(f"Distribution of {column}")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,4))
# sns.countplot(x="rainfall", data = data)
# plt.title("Distribution of Rainfall")
# plt.show()

# plt.figure(figsize=(10,8))
# sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

# plt.figure(figsize=(15, 10))
# for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
#     plt.subplot(3,3,i)
#     sns.boxplot(data[column])
#     plt.title(f"Boxplot of {column}")
# plt.tight_layout()
# plt.show()

#data preprocessing
data.drop(columns=["maxtemp", "dewpoint", "mintemp"], inplace=True)
print(data.head())

#separate majority and minority class
print(data["rainfall"].value_counts())
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]

print(df_majority.shape)
print(df_minority.shape)

#downsample majority class to matcch minority count
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])
print(df_downsampled.shape)

#shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac=1, random_state= 42).reset_index(drop=True)
print(df_downsampled.head())
print(df_downsampled["rainfall"].value_counts())

#split features and target
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
print(X.head())
print(y.head())

#splitting the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model training
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
     "n_estimators": [50, 100, 200, 300, 500],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

# Hypertuning with grid search on resampled data
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_resampled, y_train_resampled)

best_rf_model = grid_search_rf.best_estimator_
print("Best parameters for Random Forest: ", grid_search_rf.best_params_)

# Model evaluation with resampled data
cv_scores = cross_val_score(best_rf_model, X_train_resampled, y_train_resampled, cv=5)
print("Cross-Validation scores: ", cv_scores)
print("Mean cross-validation scores: ", np.mean(cv_scores))

# Test set performance
y_pred = best_rf_model.predict(X_test)

print("Test set Accuracy: ", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# Prediction on unknown data
input_data = (1008.4, 26.8,	81, 65, 6.4,	20, 7.9) #145
input_df = pd.DataFrame([input_data], columns=['pressure', "temparature", "humidity", "cloud", "sunshine", "winddirection", "windspeed"])
prediction = best_rf_model.predict(input_df)
print("Prediction result: ", "Rainfall" if prediction[0] == 1 else "No Rainfall")

# Save the model using joblib
joblib.dump(best_rf_model, 'random_forest_model.joblib')

print("Model saved successfully with joblib!")
