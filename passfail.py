import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib 

# Step 1: Load the dataset
data = pd.read_csv('student_exam_data.csv')

# Step 2: Prepare the data
# Independent Variables
X = data[['Study Hours', 'Previous Exam Score']]
# Dependent Variable 
y = data['Pass/Fail']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Save the model 
joblib.dump(model, 'student_pass_fail_model.pkl')

# Step 6: Function to take user input and make predictions
def predict_pass_fail():
    print("Enter the following details to predict if the student will pass or fail:")
    study_hours = float(input("Study Hours: "))
    previous_score = float(input("Previous Exam Score: ")) 

    new_data = np.array([[study_hours, previous_score]])

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        print("Prediction: The student will PASS the exam.")
    else:
        print("Prediction: The student will FAIL the exam.")

# Step 7: Call the function to take input and predict
predict_pass_fail()