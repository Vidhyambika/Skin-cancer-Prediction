import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

df=pd.read_csv('skincancer_data.csv')

cdf = df[['erythema','scaling','definite','itching','koebner','polygonal','follicular','oral','knee','scalp','family','melanin','eosinophils','PNL','fibrosis','exocytosis','acanthosis','hyperkeratosis','parakeratosis','clubbing','elongation','thinning','spongiform','munro','focal','disappearance','vacuolisation ','spongiosis','saw-tooth','follicular','perifollicular','inflammatory','band','Age','types']]

x = cdf.iloc[:, :34]
y = cdf.iloc[:, -1]

          
          
clf=LogisticRegression()
clf.fit(x,y)
accl = clf.score(x, y)
print("Accuracy: ",accl*100," %.")
clf_acc = accl*100



SVM = svm.LinearSVC()
SVM.fit(x, y)
acc = SVM.score(x, y)
print("Accuracy: ",acc*100," %.")
svm_acc = acc*100

print(clf.predict([[2,2,1,0,1,0,0,0,0,0,0,0,0,0,0,3,2,0,2,0,0,0,0,0,0,0,0,2,0,0,0,2,0,30]]))
print(SVM.predict([[2,2,0,0,1,1,0,1,0,0,1,0,0,0,0,3,2,0,2,0,0,0,0,0,0,0,0,2,0,0,0,2,0,30]]))

round(clf.score(x,y), 4)
round(SVM.score(x,y), 4)


data = {'LogisticRegression':clf_acc, 'SVC':svm_acc}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green', width = 0.1)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()



file=open('model.pkl','wb')
pickle.dump(clf,file,protocol=2)
