import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

data = pd.read_csv('/content/drive/MyDrive/new.csv')
# Encode the category column
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Parse the embedding column
data['embedding'] = data['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))

# Convert the embedding column to a numpy array
X = np.vstack(data['embedding'].values)
y = data['category_encoded'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate XGBoost model
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print the evaluation metrics
print(f'F1 Score XGB: {f1}')
print(f'Precision XGB: {precision}')
print(f'Recall XGB: {recall}')
print(f'AccuracyXGB: {accuracy}')

clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate AdaBoost model
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print the evaluation metrics
print(f'F1 Score Ada: {f1}')
print(f'Precision Ada: {precision}')
print(f'Recall Ada: {recall}')
print(f'Accuracy Ada: {accuracy}')


# Create and train the LightGBM model
clf = LGBMClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate LigthGBM model
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print the evaluation metrics
print(f'F1 Score LigthGBM: {f1}')
print(f'Precision LigthGBM: {precision}')
print(f'Recall LigthGBM: {recall}')
print(f'Accuracy LigthGBM: {accuracy}')
