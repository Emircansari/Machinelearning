import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('diabetes.csv')
data.fillna(data.mean(), inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Eğitim ve test setlerine ayırın (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def plot_classification_report(y_test, y_pred, model_name):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    sns.heatmap(df.iloc[:-1, :].T, annot=True, cmap="YlGnBu")
    plt.title(f'Classification Report - {model_name}')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

#Naivbayes
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)
y_pred_prob_nb = nb_model.predict_proba(X_test_scaled)[:, 1]

print("Naive Bayes Classifier:")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")
plot_classification_report(y_test, y_pred_nb, "Naive Bayes")
plot_roc_curve(y_test, y_pred_prob_nb, "Naive Bayes")

# K-en Yakın Komşuluk sınıflandırıcısı ve en iyi k değerini belirleme
k_values = range(1, 26)
scores = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    scores[k] = knn.score(X_test_scaled, y_test)

best_k = max(scores, key=scores.get)
print(f"Best k value: {best_k}")

# En iyi k değeri ile KNN sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

print("K-Nearest Neighbors Classifier:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

plot_confusion_matrix(y_test, y_pred_knn, "K-Nearest Neighbors")
plot_classification_report(y_test, y_pred_knn, "K-Nearest Neighbors")
plot_roc_curve(y_test, y_pred_prob_knn, "K-Nearest Neighbors")

# Multi-Layer Perceptron (MLP) sınıflandırıcısı
mlp_model = MLPClassifier(random_state=42, max_iter=300)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)
y_pred_prob_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]

print("Multi-Layer Perceptron Classifier:")
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

plot_confusion_matrix(y_test, y_pred_mlp, "Multi-Layer Perceptron")
plot_classification_report(y_test, y_pred_mlp, "Multi-Layer Perceptron")
plot_roc_curve(y_test, y_pred_prob_mlp, "Multi-Layer Perceptron")

# Support Vector Machines (SVM) sınıflandırıcısı
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print("Support Vector Machines Classifier:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

plot_confusion_matrix(y_test, y_pred_svm, "Support Vector Machines")
plot_classification_report(y_test, y_pred_svm, "Support Vector Machines")
plot_roc_curve(y_test, y_pred_prob_svm, "Support Vector Machines")
