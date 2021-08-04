import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

fraud = pd.read_csv('creditcard.csv')
# fraud.drop(columns=['Time', 'Class'], inplace=True)
# print(fraud.head(20))
fraud.drop(columns=['Time'], inplace=True)

target = fraud[['Class']]
features = fraud.drop(columns=['Class'])

features_training, features_test, target_training, target_test = train_test_split(features, target, test_size=0.2, random_state=1)
st_scaler = StandardScaler()
features_training_norm = st_scaler.fit_transform(features_training)
features_test_norm = st_scaler.transform(features_test)
target_test_norm = st_scaler.fit_transform(target_test)
target_training_norm = st_scaler.transform(target_training)

clf = LogisticRegression(random_state=2).fit(features_training_norm, target_training)

prediction = clf.predict(features_test_norm)
print(clf.score(features_training_norm, target_training))
print(prediction)
print(confusion_matrix(target_test, prediction))
