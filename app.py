from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

#Reading the data into a dataframe
df = pd.read_csv('jamboree_admission.csv')

df.drop(columns = 'Serial No.', inplace = True)

df.rename(columns = {'GRE Score' : 'GRE_Score', 'TOEFL Score': 'TOEFL_Score',
                     'University Rating': 'University_Rating', 'LOR ' : 'LOR', 'Chance of Admit ' : 'Chance_of_Admit'}, inplace = True)

# Handling outliers that were dectected using boxplot & IQR method
df.drop(df[df['LOR'] == 1].index, axis = 0, inplace = True)

# Splitting the dataset into separate datframes based on features and target variable
X = df.iloc[:, :-1]
y = df['Chance_of_Admit']

# Splitting the data into train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Scaling the features before model building
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

#Loading the ML model
model = pickle.load(open('regressor.pkl', mode = 'rb'))

#Defining the end point for making the prediction
@app.route('/predict', methods = ['POST', 'GET'])
def prediction():
    admit_req = request.get_json()
    gre_score = admit_req['GRE_Score']
    toefl_score = admit_req['TOEFL_Score']
    university_rating = admit_req['University_Rating']
    sop = admit_req['SOP']
    lor = admit_req['LOR']
    cgpa = admit_req['CGPA']
    research = admit_req['Research']

    input_scaled = scaler.transform([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])

    result = model.predict(input_scaled)

    return {'Chances_Of_Admission' : np.round(float(result),2) * 100 }

