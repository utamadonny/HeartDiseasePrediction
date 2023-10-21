import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

image0 = Image.open("img\Screenshot 2023-10-21 175443.png")
image1 = Image.open("img\Screenshot 2023-10-21 181250.png")
image2 = Image.open("img\Screenshot 2023-10-21 181607.png")
image3 = Image.open("img\outlier.png")
image4 = Image.open("img\outlier2.png")
image5 = Image.open("img\dsit_gender.png")
image6 = Image.open("img\dist_age.png")
image7 = Image.open("img\dist_chol.png")
image8 = Image.open("img\dist_target.png")
image9 = Image.open("img\dist_num_var.png")
image10 = Image.open("img\dis_cat_var.png")
image11 = Image.open("img\corr1.png")
image12 = Image.open("img\corr2.png")
image13 = Image.open("img\corrheat.png")
image14 = Image.open("img\sort_corr.png")

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my machine learning dashboard
## Heart Disease Prediction

This dashboard created by : [Donny Prakarsa Utama](https://www.linkedin.com/in/utamadonny/)
         
---
         
""")

add_selectitem = st.sidebar.selectbox("Want to open about?", ("Heart Disease Overview", "Data Exploration", "ML Prediction!"))
def overview():
    st.write("""
             
## How to Use This Streamlit App
Welcome to my Machine Learning Dashboard, which focuses on heart disease prediction. Here's how to navigate through the app:

1. **Sidebar Navigation:**
   - On the left sidebar, you will find four options under 'Want to open about?'
   - Choose 'Heart Disease Overview' to learn about cardiovascular diseases.
   - Select 'Data Exploration' to explore data and perform Exploratory Data Analysis (EDA).
   - Pick 'ML Prediction!' to start using the prediction model.
2. **Heart Disease Overview:**
   - In this section, you'll find information about cardiovascular diseases and how Machine Learning can predict them.
3. **Data Exploration:**
   - Explore the data used for prediction, perform data analysis, and gain insights into the dataset.
4. **ML Prediction!**
   - Here, you can start the prediction model and make predictions related to heart disease.
Feel free to explore and use the app to understand heart health and make predictions related to cardiovascular diseases. If you have any questions or need assistance, please don't hesitate to ask.
    

## Overview
**Cardiovascular disease (CVDs)**, or heart disease, is the leading global cause of death, with 17.9 million deaths annually. Heart disease is caused by hypertension, obesity, and unhealthy lifestyles. Early detection of heart disease is crucial for individuals at high risk to receive prompt diagnosis and prevention. The business goal is to build a predictive model for heart disease based on available features to assist doctors in making accurate diagnoses. The aim is to manage heart disease at an early stage, ultimately reducing the mortality rate.
    
The dataset used is the Heart Disease data downloaded from UCI ML: UCI Heart Disease Dataset

This dataset, dating back to 1988, comprises four databases: Cleveland, Hungaria, Swiss, and Long Beach V. The 'target' field refers to the presence of heart disease in patients, with integer values 0 = no disease and 1 = disease.

The Heart Disease dataset consists of 1025 rows of data and 13 attributes + 1 target. It has 14 columns, including:

1. Age: Represents the patient's age measured in years.
2. Sex: Represents the patient's gender, with a value of 1 for males and 0 for females.
3. Chest Pain Type (CP): Represents the type of chest pain experienced by the patient with 4 possible category values. Value 1 indicates angina chest pain, value 2 indicates non-anginal chest pain, value 3 indicates non-anginal chest pain of a severe nature, and value 4 indicates chest pain unrelated to heart issues.
4. Resting Blood Pressure (trestbps): Represents the patient's resting blood pressure, measured in mmHg (millimeters of mercury).
5. Serum Cholesterol (chol): Represents the serum cholesterol level in the patient's blood, measured in mg/dl (milligrams per deciliter).
6. Fasting Blood Sugar (fbs): Represents the patient's fasting blood sugar level, with a value of 1 if blood sugar is > 120 mg/dl and 0 otherwise.
7. Resting Electrocardiographic Results (restecg): Represents the resting electrocardiogram results of the patient with 3 possible category values. Value 0 indicates a normal result, value 1 indicates ST-T wave abnormalities, and value 2 indicates left ventricular hypertrophy.
8. Maximum Heart Rate Achieved (thalach): Represents the maximum heart rate achieved by the patient during exercise, measured in bpm (beats per minute).
9. Exercise-Induced Angina (exang): Represents whether the patient experiences angina (chest pain) induced by exercise, with a value of 1 for yes and 0 for no.
10. Oldpeak: Represents the amount of ST segment depression during physical activity compared to rest.
11. Slope: Represents the slope of the ST segment on the electrocardiogram (EKG) during maximal exercise with 3 possible category values.
12. Number of Major Vessels (ca): Represents the number of major blood vessels (0-3) visible on fluoroscopy examination.
13 Thalassemia (thal): Represents the results of the thallium scan with 3 possible category values:
    Thal 1: Indicates a normal condition.
    Thal 2: Indicates a fixed defect in thalassemia.
    Thal 3: Indicates a reversible defect in thalassemia.
14. Target: 0 = no disease and 1 = disease.
        
## ``` "Machine learning models for heart disease prediction are trained using historical patient data and their corresponding outcomes, such as whether the patient has heart disease or not. The model learns patterns and relationships within the data to make predictions based on new, unseen patient information" ```
Here's how machine learning predicts heart disease:

1. **Data Collection:** Large datasets containing various patient attributes are collected. These attributes may include age, gender, blood pressure, cholesterol levels, chest pain type, and more.
2. **Data Preprocessing:** The collected data is cleaned, and any missing or inconsistent values are addressed. Categorical data may be one-hot encoded or transformed for model compatibility.
3. **Feature Selection:** Relevant features are selected to build the predictive model. Machine learning algorithms identify which attributes are most informative in predicting heart disease.
4. **Model Training:** The selected machine learning algorithm is trained using the preprocessed data. The model learns to recognize patterns and relationships between the patient attributes and the presence of heart disease.
5. **Model Evaluation:** The trained model is evaluated using a different dataset to assess its predictive performance. Various evaluation metrics, such as accuracy, precision, recall, and F1 score, are used to measure the model's accuracy.
6. **Predictions:** Once the model is deemed accurate, it can make predictions for new patient data. A set of attributes, such as age, blood pressure, and cholesterol levels, are provided to the model, which then predicts whether the patient is at risk of heart disease.
Machine learning-based heart disease prediction provides several advantages. It allows for early detection, which can lead to timely intervention and prevention. Additionally, it can help healthcare professionals make more accurate and informed diagnoses, reducing the risk of misdiagnosis and providing personalized treatment plans.
This Streamlit app leverages machine learning to make predictions related to heart disease, contributing to early diagnosis and improved patient care.
If you want to explore the data, perform data analysis, or start making predictions, use the options on the sidebar to navigate through the app.


             
###     Maintaining Heart Health and Preventing Cardiovascular Diseases

Cardiovascular diseases (CVDs) are a significant global health concern. Here are some tips to maintain heart health and reduce the risk of cardiovascular diseases:

1. **Eat a Heart-Healthy Diet:** Consume a diet rich in fruits, vegetables, whole grains, and lean proteins. Reduce the intake of saturated and trans fats, salt, and added sugars.
2. **Regular Physical Activity:** Engage in regular physical activity such as brisk walking, jogging, or swimming. Aim for at least 150 minutes of moderate-intensity exercise per week.
3. **Maintain a Healthy Weight:** Maintain a healthy body weight by balancing your calorie intake with physical activity.
4. **Quit Smoking:** If you smoke, quitting is one of the most significant steps you can take to protect your heart.
5. **Limit Alcohol Intake:** If you consume alcohol, do so in moderation. Excessive alcohol can lead to heart problems.
6. **Manage Stress:** Practice stress-reduction techniques such as meditation, deep breathing, or yoga.
7. **Regular Health Check-ups:** Visit your healthcare provider for regular check-ups and monitor your blood pressure, cholesterol, and blood sugar levels.
8. **Get Enough Sleep:** Aim for 7-9 hours of quality sleep each night.
9. **Stay Hydrated:** Drink an adequate amount of water each day.
10. **Know Your Family History:** Be aware of your family's heart health history, as genetics can play a role in heart diseases.
These lifestyle changes can contribute to better heart health and a reduced risk of cardiovascular diseases. Remember that it's essential to consult with a healthcare professional for personalized advice and recommendations.
             
             """)

def dataexplore():
    st.write("""
## **Data Preprocessing, Including Exploratory Data Analysis (EDA)**
Data preprocessing, including Exploratory Data Analysis (EDA), is a crucial step in understanding and preparing patient data for heart disease analysis. This process involves cleaning, exploring, and refining the data to extract valuable insights and ensure it's ready for further modeling.
The key components of data preprocessing and EDA include:
1. **Data Collection:** Gathering patient data, which may include attributes like age, gender, blood pressure, cholesterol levels, and more.
2. **Data Cleaning:** Addressing missing or inconsistent values, and ensuring data quality.
3. **Exploratory Data Analysis (EDA):** Analyzing the data to uncover patterns, relationships, and trends. EDA helps in understanding the dataset's characteristics and identifying potential predictors of heart disease.
4. **Feature Selection:** Choosing relevant features that have the most significant impact on heart disease prediction. It involves identifying the attributes that contribute the most to the predictive accuracy.
5. **Dimensionality Reduction:** Reducing the number of features to simplify the modeling process and enhance computational efficiency.
The ultimate goal is to gain insights into the patient data related to heart disease and prepare the data for the next modeling stage. Data preprocessing and EDA ensure the dataset is cleaned, well-structured, and ready for effective machine learning analysis.

```python
# Import libraries
import numpy as np
import pandas as pd
import math
import random
import seaborn as sns
from scipy.stats import pearsonr, jarque_bera
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
         
# Memuat dataset Heart Disease UCI ML
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names, skiprows=[0])

# Menampilkan lima baris teratas
data.head()
```
 """)
    st.image(image0)

    st.write(
    """
Here are some common data exploration tasks in a Streamlit app:
- Display the last five rows of the dataset:
```python
data.tail()
```
- Check the dataset dimensions (number of rows and columns):
```python
data.shape
```
- List the columns present in the dataset:
```python
data.columns
```
- Examine the summary statistics for numerical data:
```python
data.describe()
```
- Check the number of unique values in each column:
```python
data.nunique()
```
- Get an overview of the dataset:
```python
data.info()
```
### Make Categorical and Numerical Data
```python
# Pelabelan data categorical
data['sex'] = data['sex'].replace({1: 'Male',
                                   0: 'Female'})
data['cp'] = data['cp'].replace({0: 'typical angina',
                                 1: 'atypical angina',
                                 2: 'non-anginal pain',
                                 3: 'asymtomatic'})
data['fbs'] = data['fbs'].replace({0: 'No',
                                   1: 'Yes'})
data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy',
                                           1:'normal',
                                           2: 'ST-T Wave abnormal'})
data['exang'] = data['exang'].replace({0: 'No',
                                       1: 'Yes'})
data['slope'] = data['slope'].replace({0: 'downsloping',
                                       1: 'flat',
                                       2: 'upsloping'})
data['thal'] = data['thal'].replace({1: 'normal',
                                     2: 'fixed defect',
                                     3: 'reversable defect'})
data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                 1: 'Number of major vessels: 1',
                                 2: 'Number of major vessels: 2',
                                 3: 'Number of major vessels: 3'})
data['target'] = data['target'].replace({0: 'No disease',
                                         1: 'Disease'})


numerical_col = data.select_dtypes(exclude='object')
categorical_col = data.select_dtypes(exclude=['int64','float64'])
```
## **Data Cleaning and Missing Value Handling**
In the dataset, there were two features with data entry errors that needed to be addressed:
1. **Feature 'CA':** This feature has values in the range 0-4. However, there were instances where '4' was erroneously recorded, which should not have occurred. To rectify this, we changed the '4' values to 'NaN' (indicating missing data).
2. **Feature 'Thal':** This feature has values in the range 0-3. Similar to 'CA,' there were instances where '0' was mistakenly recorded, which should not have been the case. We replaced the '0' values with 'NaN' to signify missing data.
Additionally, to clean and handle these data anomalies, the following actions were taken:
- Checking the count of unique values in the 'CA' feature:
```python
data['ca'].value_counts()
```
- Identifying rows where the 'CA' column had a value of '4':
```python
data[data['ca'] == 4]
```
- Replacing the '4' values in the 'CA' column with 'NaN':
```python
data.loc[data['ca'] == 4, 'ca'] = np.nan
```
- Verifying if there were any remaining '4' values in the 'CA' feature:
```python
data['ca'].value_counts()
```
- Checking the count of unique values in the 'Thal' feature:
```python
data['thal'].value_counts()
```
- Identifying rows where the 'Thal' column had a value of '0':
```python
data[data['thal'] == 0]
```
- Replacing the '0' values in the 'Thal' column with 'NaN':
```python
data.loc[data['thal'] == 0, 'thal'] = np.nan
```
- Confirming if there were any remaining '0' values in the 'Thal' feature:
```python
data['thal'].value_counts()
```
- Checking for missing values in the dataset:
```python
print("Missing Value Summary:\\n", data.isnull().sum())
```
"""
    )
    st.image (image1)
    st. write("""
 To handle missing values in the dataset, the mode (most frequent value) of the respective columns is used for imputation. Here's how it was done:

    - Filling missing values in the 'CA' column with the mode:
    ```python
    modus_ca = data['ca'].mode()[0]
    data['ca'] = data['ca'].fillna(modus_ca)
    ```

    - Filling missing values in the 'Thal' column with the mode:
    ```python
    modus_thal = data['thal'].mode()[0]
    data['thal'] = data['thal'].fillna(modus_thal)
    ```
    - Checking for missing values again:
    ```python
    print("Missing Value Summary:\\n", data.isnull().sum())
    ```

    This method ensures that missing values are replaced with the most common values in the respective columns, maintaining data integrity.              
    """)         
    st.image(image2)
    st.write(" - Outliers and boxplot")
    st.image(image3)
    st.image(image4)
    st.write(""" 
--- 
##  **Data Distributions**
After conducting exploratory data analysis (EDA), here's an overview of data distributions:
- **Gender Distribution:**
    """)
    st.image(image5)

    st.write(" - **Age Distribution:**")
    st.image(image6)
    
    st.write(" - **Serum Cholesterol (Chol) Distribution:**")
    st.image(image7)

    st.write(" - **Target Distribution:**")
    st.image(image8)

    st.write(" - **Numerical Variable Distribution:**")
    st.image(image9)

    st.write(" - **Categorical Variable Distribution:**")
    st.image(image10)

    st.write("""
- **Distribution of Patients Based on Stress Level and Maximum Heart Rate:**
    """)
    st.image(image11)

    st.write(" - **Distribution of Patients Based on Stress Level and Blood Pressure:**")
    st.image(image12)
    st.write("""
## **Feature Selection and Correlation Analysis**
Feature selection is a crucial step in building predictive models for heart disease. It involves identifying the most influential variables and understanding their correlations with the target variable (heart disease). Here are the key findings:
""")
    st.image(image13)
    st.image(image14)
    st.write("""
- **Correlation with Target (Heart Disease):**
    - **Negative Strong Correlation:**
        1. ca: -0.456989
        2. oldpeak: -0.434108
        3. exang: -0.431599
        4. thal: -0.370759
        5. sex: -0.318896
    - **Negative Correlation:**
        6. age: -0.222416
        7. trestbps: -0.115614
        8. chol: -0.0105627
    - **Positive Weak Correlation:**
        9. fbs: 0.027210
        10. restecg: 0.171453
        11. slope: 0.326473
        12. cp: 0.422559
        13. thalach: 0.432211
             
Therefore, the most influential factors for heart disease are:
1. **ca**: The number of major vessels (more vessels indicate a higher risk of heart disease).
2. **oldpeak**: The degree of ST segment depression during exercise relative to rest (higher depression indicates a higher risk of heart disease).
3. **exang**: The presence of exercise-induced angina (low presence indicates a higher risk of heart disease).
4. **thal**: The type of thalassemia defect (lower defect type indicates a higher risk of heart disease).
5. **sex**: Gender (females have a higher risk of heart disease than males).
6. **age**: Age (younger age indicates a higher risk of heart disease).
7. **slope**: The slope of the ST segment during exercise (higher slope indicates a higher risk of heart disease).
8. **cp**: Chest pain type (higher type indicates a higher risk of heart disease).
9. **thalach**: Maximum heart rate achieved during exercise (higher heart rate indicates a higher risk of heart disease).
    """)



def heart():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Usia", 29, 77, 30)
            data = {    
                    'sex': sex,
                    'cp': cp,
                    # 'fbs': fbs,
                    'slope': slope,
                    'thalach' : thalach,
                    # 'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features()
    img = Image.open("img\heart-disease.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_heart_disease.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)     
        prediction_proba = loaded_model.predict_proba(df)   
        result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']


        st.subheader('Prediction: ')
        resultstr = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {resultstr}")
        st.subheader('Prediction Probability: ')
        output = (prediction_proba)
        st.write(output)

if add_selectitem == "Heart Disease Overview":
    overview()
elif add_selectitem == "ML Prediction!":
    heart()
elif add_selectitem == "Data Exploration":
    dataexplore()

st.write("""
---

### Disclaimer
Disclaimer: This application is for educational and informational purposes only. The predictions provided are not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a healthcare professional for any health-related concerns or medical decisions. The accuracy of predictions may vary, and this application should not be used as a primary source for making medical decisions.")

### About Me
[Donny Prakarsa Utama | LinkedIn](https://www.linkedin.com/in/utamadonny/)
         
[utamadonny (Donny Prakarsa Utama) | (github.com)](https://github.com/utamadonny)
      

""")