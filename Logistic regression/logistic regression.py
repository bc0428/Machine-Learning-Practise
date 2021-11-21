import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

data = pd.read_csv("LOCATION OF breast_cancer.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cross_validation_scores = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(f"Accuracy: {cross_validation_scores.mean()*100}%")
print(f"Standard Deviation: {cross_validation_scores.std()*100}%")
