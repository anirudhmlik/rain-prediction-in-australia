import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer as lb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('./SA2021_W3_Data.csv')
data.head
data.dtypes
size = data.size
shape = data.shape
print(shape)
df_num = data.select_dtypes(exclude="object")
df_num.describe().T.style.background_gradient(subset=['std'], cmap='Oranges')\
                            .background_gradient(subset=['50%'], cmap='Oranges')
data.isnull().values.any()
X = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2021)
print("X_train:", X_train.shape) # X_train
print("X_test:", X_test.shape) # X_test
print("y_train:", y_train.shape) # y_train
print("y_test:", y_test.shape) # y_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(random_state=2021)
lr.fit(X_train_scaled, y_train)
plot_confusion_matrix(lr, X_test_scaled, y_test)
print(classification_report(y_test, lr.predict(X_test_scaled)))
print(lr.score(X_train_scaled, y_train))
print(lr.score(X_test_scaled, y_test))
pred = lr.predict(X_test_scaled)
ans = y_test
accuracy_score(y_true=ans, y_pred=pred)
roc_auc_score(y_test, lr.predict(X_test_scaled))
svm = SVC(C=50, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)
plot_confusion_matrix(svm, X_test_scaled, y_test)
print(classification_report(y_test,svm.predict(X_test_scaled)))
pred = svm.predict(X_test_scaled)
ans = y_test
accuracy_score(y_true=ans, y_pred=pred)
print(svm.score(X_train_scaled, y_train))
print(svm.score(X_test_scaled, y_test))
roc_auc_score(y_test, svm.predict(X_test_scaled))
svm = SVC(C=1, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)
print(classification_report(y_test, svm.predict(X_test_scaled)))
plot_confusion_matrix(svm, X_test_scaled, y_test)
roc_auc_score(y_test, svm.predict(X_test_scaled))
print(svm.score(X_train_scaled, y_train))
print(svm.score(X_test_scaled, y_test))
pred = svm.predict(X_test_scaled)
ans = y_test
accuracy_score(y_true=ans, y_pred=pred)
scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
scores.mean()
