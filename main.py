from sklearn.datasets import load_wine
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import  accuracy_score

# Wine verisetinin yüklenmesi
wine = load_wine()
# X matrisine 13 sütun olarak verisetinin özelliklerinin aktarılması
X = wine.data
# y dizisine bu verisetinin türlerinin (etiketlerinin) atanması
y = wine.target

clf1 = DecisionTreeClassifier(random_state=0)

# X öznitelikleri ve y etiketlerine göre Karar Ağacının öğrenmesinin sağlanması
# Verinin %70'ini Eğitim, %30'unu test verisi olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size =0.3, random_state = 0, stratify = y)

# Eğitim Verisi ile eğitimi gerçekleştiriyoruz
clf1.fit(X_train,y_train)
test_sonuc = clf1.predict(X_test)

print("Decision Tree Classifier:\n")

#hata matrisinin yazdirilmasi
print("Hata Matrisi:")
cm = confusion_matrix(y_test, test_sonuc)
print(cm)

#istatiksel degerlerin hesaplanmasi
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf1, X, y, scoring=scoring,cv=10)
ac_sc = accuracy_score(y_test, test_sonuc)
rc_sc = np.mean(scores['test_recall_macro'])
pr_sc = (np.mean(scores['test_precision_macro']))

print("\n========== Decision Forest Results ==========")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)

print("Yeni örüntünün ait olduğunu sınıf:(Decision Tree Classifier) ")
print(clf1.predict([[17.89,1.78,6.63,5.16,167,2.87,3.02,.43,2.01,5.71,1.03,4.91,1921]]))





print("\n\n\nRandom Forest Classifier:\n")
print("Hata Matrisi:")
clf2 = RandomForestClassifier(max_depth=1,random_state=0)
clf2.fit(X_train, y_train)
test_sonuc2 = clf2.predict(X_test)

#hata matrisinin yazdirilmasi
cm2 = confusion_matrix(y_test, test_sonuc2)
print(cm2)

#istatiksel degerlerin hesaplanmasi
scoring2 = ['precision_macro', 'recall_macro']
scores2 = cross_validate(clf2, X, y, scoring=scoring2,cv=10)
ac_sc2 = accuracy_score(y_test, test_sonuc2)
rc_sc2 = np.mean(scores2['test_recall_macro'])
pr_sc2 = (np.mean(scores2['test_precision_macro']))

print("========== Random Forest Results ==========")
print("Accuracy    : ", ac_sc2)
print("Recall      : ", rc_sc2)
print("Precision   : ", pr_sc2)

print("Yeni örüntünün ait olduğunu sınıf:(Random Forest Classifier) ")
print(+clf2.predict([[17.89,1.78,6.63,5.16,167,2.87,3.02,.43,2.01,5.71,1.03,4.91,1921]]))

