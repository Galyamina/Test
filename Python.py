#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Predicts whether a patient has Diabetes or not.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn  import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("https://raw.githubusercontent.com/datawizardsai/Data-Science/master/pima-indians-diabetes.data.csv")

X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(model.score(X_test,y_test))
confusion_matrix(y_test, y_predicted)


# In[2]:


# Predicts the specie of flower
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn  import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("https://raw.githubusercontent.com/datawizardsai/Data-Science/master/iris.csv")
df

X = df.drop('Class',axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier() #n_estimators=10 # 10 forests by default
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(model.score(X_test,y_test))
confusion_matrix(y_test, y_predicted)


# In[4]:


# Predicts the specie of flower
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn  import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("https://raw.githubusercontent.com/datawizardsai/Data-Science/master/Development%20Index.csv")
df

X = df.drop('Development Index',axis=1)
y = df['Development Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier() #n_estimators=10 # 10 forests by default
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(model.score(X_test,y_test))
confusion_matrix(y_test, y_predicted)


# In[ ]:




