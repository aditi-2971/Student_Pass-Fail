import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv('student_exam_data.csv')

print(data.head())
print(data.info())
print(data.describe())

X = data[['Study Hours', 'Previous Exam Score']]

y = data['Pass/Fail']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Visualize the results (optional)
# Plotting the decision boundary
plt.scatter(X_test['Study Hours'], X_test['Previous Exam Score'], c=y_pred, cmap='coolwarm')
plt.xlabel('Study Hours')
plt.ylabel('Previous Exam Score')
plt.title('Pass/Fail Prediction')
plt.show()

# Step 9: Save the model (optional)
import joblib
joblib.dump(model, 'student_pass_fail_model.pkl')

# Step 10: Load the model and make a prediction on new data (optional)
# model = joblib.load('student_pass_fail_model.pkl')
new_data = np.array([[5, 75]])  # Example: 5 study hours and 75 previous exam score
prediction = model.predict(new_data)
print("Prediction:", prediction)