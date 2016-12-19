import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer


df = pd.read_csv('../data/MachineLearningWithPython/Notebooks/data/pima-data.csv')
num_obs = len(df)
#print(num_obs)
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
#print(num_true, num_false)

#split 70% to treaining set and 30% to test
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

x = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)


#print(len(X_train), len(X_test), len(y_train), len(y_test))


original_diabetes = len(df.loc[df['diabetes'] == 1])
original_not_diabetes = len(df.loc[df['diabetes'] == 0])
original_total = len(df.index)

#print(y_train)
#print("Original True: {0} ({1:0.2f}%)".format(original_diabetes, original_diabetes / original_total * 100.0) )
#print(" ")
training_diabetes = len(y_train[y_train[:] == 1])
#print("Training True: {0} ({1:0.2f}%)".format(training_diabetes, training_diabetes / len(y_train) * 100.0) )
#print(" ")
test_diabetes = len(y_test[y_test[:] == 1])
#print("Test True: {0} ({1:0.2f}%)".format(test_diabetes, test_diabetes / len(y_test) * 100.0) )

fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

nb_predict_train = nb_model.predict(X_train)

from sklearn import metrics

#Accuracy
#print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
#print()


nb_predict_test = nb_model.predict(X_test)
#print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
#print()



#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train.ravel())

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                       max_features='auto', max_leaf_nodes=None,
                       min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=1, oob_score=False, random_state=42, verbose=0, warm_start=False)

rf_predict_train = rf_model.predict(X_train)
#print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))

#Predict with test data
rf_predict_test = rf_model.predict(X_test)
#print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
#print("Classification report: ")

#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)

lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1, 0]))
print("")
print("Classification report")
print(metrics.classification_report(y_test, lr_predict_test, labels=[1,0]))

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while(C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")
plt.show()

# 0.62 is best. The problem is we have more False values in training set, compared to True value examples. This must be compensated.
lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

print(metrics.accuracy_score(y_test, lr_predict_test))