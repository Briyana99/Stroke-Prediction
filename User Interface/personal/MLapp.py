from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
app = Flask(__name__,template_folder='template')

@app.route("/",methods= ['GET'])
def home():
    df4 = pd.read_csv("stroke-data_preprocessed.csv")
    X = df4.iloc[:, 1:11]
    X1 = X.astype('int')
    Y = df4.iloc[:, 11]
    Y1 = Y.astype('int')
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.20, random_state=235)
    rf.fit(X_train, y_train)
    pickle.dump(rf,open("stroke.pkl","wb"))
    return render_template("home1.html")

@app.route('/predict/',methods= ['POST','GET'])
def predict():
    input_features = [float(x) for x in request.form.values()] 
    gender = input_features[0]
    age = input_features[1]
    hypertension =  input_features[2]
    heart_disease = input_features[3]
    married = input_features[4]
    work_type = input_features[5]
    Residence_type = input_features[6]
    avg_glucose_level = input_features[7]
    bmi = input_features[8]
    smoking_status = input_features[9]
    form_array = np.array([[gender,age,hypertension,heart_disease,married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
    rf = pickle.load(open('stroke.pkl','rb'))
    prediction = rf.predict(form_array)[0]
    if prediction == 0:
        result = " Stroke will not happen"
    else:
        result = " Stroke will happen"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
