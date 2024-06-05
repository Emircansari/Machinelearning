Giriş
Bu raporda, diyabet veri seti üzerinde Naive Bayes, K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP) ve Support Vector Machines (SVM) sınıflandırıcılarının performansları değerlendirilecektir. Performans değerlendirmesi için kullanılan metrikler arasında karışıklık matrisi, precision, recall, f1-score, accuracy ve ROC eğrisi bulunmaktadır.

Veri Seti
Veri seti, diyabet hastalığına sahip olup olmadığını belirlemek için çeşitli biyomedikal ölçümler içermektedir. Aşağıda veri setindeki özelliklerin listesi ve veri ön işleme adımları verilmiştir:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome (Hedef değişken)
Veri ön işleme adımları:

Eksik değerlerin doldurulması (ortalama ile doldurma).
Verinin eğitim ve test setlerine ayrılması.
Verilerin ölçeklendirilmesi (StandardScaler kullanarak).

Naive Bayes Classifier:
[[119  32]
 [ 27  53]]
              precision    recall  f1-score   support

           0       0.82      0.79      0.80       151
           1       0.62      0.66      0.64        80

    accuracy                           0.74       231
   macro avg       0.72      0.73      0.72       231
weighted avg       0.75      0.74      0.75       231


Best k value: 19
K-Nearest Neighbors Classifier:
[[132  19]
 [ 39  41]]
              precision    recall  f1-score   support

           0       0.77      0.87      0.82       151
           1       0.68      0.51      0.59        80

    accuracy                           0.75       231
   macro avg       0.73      0.69      0.70       231
weighted avg       0.74      0.75      0.74       231

Multi-Layer Perceptron Classifier:
[[118  33]
 [ 33  47]]
              precision    recall  f1-score   support

           0       0.78      0.78      0.78       151
           1       0.59      0.59      0.59        80

    accuracy                           0.71       231
   macro avg       0.68      0.68      0.68       231
weighted avg       0.71      0.71      0.71       231

Support Vector Machines Classifier:
[[123  28]
 [ 30  50]]
              precision    recall  f1-score   support

           0       0.80      0.81      0.81       151
           1       0.64      0.62      0.63        80

    accuracy                           0.75       231
   macro avg       0.72      0.72      0.72       231
weighted avg       0.75      0.75      0.75       231
