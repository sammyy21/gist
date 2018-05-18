#Info on: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
#       : http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

from sklearn import datasets
from sklearn import svm
a=datasets.load_iris()
x,y=a.data, a.target
clf=svm.SVC()
clf.fit(x,y)
import pickle
p=pickle.dumps(clf)
s=pickle.loads(p)
# Saved clf into s as a model, leaving behind svm.SVC heritage; joblib also used
s.predict(x[0:1])

# In[3]:

y[0:1]
