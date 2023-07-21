import pandas as pd
import joblib
import os
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(current_directory, 'cardio_train.csv')

# Load the CSV dataset into a Pandas DataFrame
df = pd.read_csv(csv_file_path, delimiter=';')

# Drop the 'id' column from the DataFrame as it's not needed for prediction
df.drop('id', axis=1, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop('cardio', axis=1)
y = df['cardio']

# Train-Test Split (80% train, 20% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Train Random Forest Model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    accuracy = None
    if request.method == 'POST':
        # Get the input values from the form
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = float(request.form['ap_hi'])
        ap_lo = float(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Make predictions using the models
        input_features = [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
        decision_tree_prediction = decision_tree_model.predict(input_features)[0]
        random_forest_prediction = random_forest_model.predict(input_features)[0]

        result = {
            'Decision Tree': 'Positive' if decision_tree_prediction == 1 else 'Negative',
            'Random Forest': 'Positive' if random_forest_prediction == 1 else 'Negative',
        }

        # Evaluate the models on the test set and calculate the accuracy
        decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))
        random_forest_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))

        accuracy = {
            'Decision Tree': decision_tree_accuracy,
            'Random Forest': random_forest_accuracy,
        }

    return render_template('index.html', result=result, accuracy=accuracy)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
