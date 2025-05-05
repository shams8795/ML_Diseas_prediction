def main():
    pass

if __name__ == "__main__":
    main()
# Importing

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pandas as pd
df=pd.read_csv('hcvdat0.csv')
df.head()


del df['Unnamed: 0']
#df.rename(columns={'Unnamed: 0':'Id'}, inplace = True)
df = df[['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT','Category']]
df

df.isna().sum()

# Encode 'Sex'
df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})

# Encode 'Category'
df['Category'] = df['Category'].astype('category').cat.codes


df_original = pd.read_csv('hcvdat0.csv')

category_mapping = dict(enumerate(df_original['Category'].astype('category').cat.categories))
print(category_mapping)

from sklearn.preprocessing import StandardScaler
features = df.drop('Category', axis=1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

import pandas as pd
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

df = pd.concat([features_scaled, df['Category']], axis=1)

from sklearn.model_selection import train_test_split


X = df.drop('Category', axis=1)
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

print(df['Category'].value_counts())

y_pred = model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


original_df = pd.read_csv('hcvdat0.csv')

original_df['Category'] = original_df['Category'].astype('category')
category_labels = list(original_df['Category'].cat.categories)


y_pred = model.predict(X_test)
predicted_categories = [category_labels[i] for i in y_pred]

def diagnose(category_label):
    if category_label == '0=Blood Donor':
        return 'Healthy'
    elif category_label == '0s=suspect Blood Donor':
        return 'Needs further testing'
    else:
        return 'Illness detected - Needs treatment'
def severity_level(category_label):
    if category_label == '0=Blood Donor':
        return 'Low'
    elif category_label == '0s=suspect Blood Donor':
        return 'Medium'
    elif category_label == '1=Hepatitis':
        return 'High'
    elif category_label == '2=Fibrosis':
        return 'High'
    elif category_label == '3=Cirrhosis':
        return 'Critical'
    else:
        return 'Unknown'

diagnosis_results = [diagnose(c) for c in predicted_categories]
severity_results = [severity_level(c) for c in predicted_categories]

results_df = pd.DataFrame({
    'Patient Number': range(1, len(predicted_categories) + 1),
    'Predicted Disease': predicted_categories,
    'Diagnosis': diagnosis_results,
    'Severity Level': severity_results
})

display(results_df)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=2)  # قللنا عدد الجيران
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


original_df = pd.read_csv('hcvdat0.csv')
original_df['Category'] = original_df['Category'].astype('category')
category_labels = list(original_df['Category'].cat.categories)
predicted_categories = [category_labels[i] for i in y_pred]

def diagnose(category_label):
    if category_label == '0=Blood Donor':
        return 'Healthy'
    else:
        return 'Illness detected - Needs treatment'


diagnosis_results = [diagnose(cat) for cat in predicted_categories]


import pandas as pd
diagnosis_summary = pd.Series(diagnosis_results).value_counts()

diagnosis_percentages = (diagnosis_summary / len(diagnosis_results)) * 100

summary_df = pd.DataFrame({
    'Count': diagnosis_summary,
    'Percentage': diagnosis_percentages.round(2)
})

display(summary_df)



def doctor_diagnosis(user_predict):
    if user_predict == 0:
        return "🩺 Diagnosis: Blood Donor - The patient is healthy and can donate blood."
    elif user_predict == 1:
        return "🩺 Diagnosis: Suspect Blood Donor - The patient needs further medical testing."
    elif user_predict == 2:
        return "🩺 Diagnosis: Hepatitis - The patient shows signs of liver inflammation and requires treatment."
    elif user_predict == 3:
        return "🩺 Diagnosis: Fibrosis - The patient has liver scarring; ongoing medical care is needed."
    elif user_predict == 4:
        return "🩺 Diagnosis: Cirrhosis - The patient has advanced liver disease; urgent treatment is required."
    else:
        return "❓ Diagnosis Unknown - Please verify the input."
all_diagnoses = [doctor_diagnosis(pred) for pred in y_pred]
patient_prediction = 3
diagnosis_result = doctor_diagnosis(patient_prediction)
print(diagnosis_result)

def predict(user_predict):
    if user_predict == 0:
        return "The Patient is: Blood Donor"
    elif user_predict == 1:
        return "The Patient is: suspect Blood Donor"
    elif user_predict == 2:
        return "The Patient is: Hepatitis"
    elif user_predict == 3:
        return "The Patient is: Fibrosis"
    elif user_predict == 4:
        return "The Patient is: Cirrhosi"
print(predict(3))


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nb_classifier = GaussianNB()
nb_classifier = nb_classifier.fit(X_train, y_train)

nb_pred = nb_classifier.predict(X_test)

NaiveB_Accuracy = accuracy_score(y_test, nb_pred)
NaiveB_Precision = precision_score(y_test, nb_pred, average='weighted')
NaiveB_Recall = recall_score(y_test, nb_pred, average='weighted')
NaiveB_F1 = f1_score(y_test, nb_pred, average='weighted')
print("Accuracy:", NaiveB_Accuracy)
print("Precision:", NaiveB_Precision)
print("Recall:", NaiveB_Recall)
print("F1-Score:", NaiveB_F1)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

adaboost_classifier = AdaBoostClassifier()
adaboost_classifier = adaboost_classifier.fit(X_train, y_train)

adaboost_pred = adaboost_classifier.predict(X_test)

AdaBoost_Accuracy = accuracy_score(y_test, adaboost_pred)
AdaBoost_Precision = precision_score(y_test, adaboost_pred, average='weighted')
AdaBoost_Recall = recall_score(y_test, adaboost_pred, average='weighted')
AdaBoost_F1 = f1_score(y_test, adaboost_pred, average='weighted')

print("Accuracy:", AdaBoost_Accuracy)
print("Precision:", AdaBoost_Precision)
print("Recall:", AdaBoost_Recall)
print("F1-Score:", AdaBoost_F1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rf_classifier = RandomForestClassifier()
rf_classifier = rf_classifier.fit(X_train, y_train)

rf_pred = rf_classifier.predict(X_test)

RF_Accuracy = accuracy_score(y_test, rf_pred)
RF_Precision = precision_score(y_test, rf_pred, average='weighted')
RF_Recall = recall_score(y_test, rf_pred, average='weighted')
RF_F1 = f1_score(y_test, rf_pred, average='weighted')

print("Accuracy:", RF_Accuracy)
print("Precision:", RF_Precision)
print("Recall:", RF_Recall)
print("F1-Score:", RF_F1)

import pandas as pd
import matplotlib.pyplot as plt

models = ['Naïve Bayes', 'AdaBoost', 'Random Forest']

scores = [0.8537, 0.8862, 0.8699]
dataF = pd.DataFrame({'Model': models, 'Score': scores})

plt.figure(figsize=(8,6))
plt.plot(dataF["Model"], dataF["Score"], marker="o")
plt.title("Parallel Coordinators Plot Of Model Scores")
plt.xlabel('Models')
plt.ylabel('Scores')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()
