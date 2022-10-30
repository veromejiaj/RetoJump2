#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import json


# In[3]:


data = pd.read_csv(r"C:\\DataScience\\Jump2\\train.csv", sep=";")
data


# In[4]:


labels=np.array(data["target"])
features= data.drop("target", axis = 1)
# Saving feature names for later use
feature_list = list(data.columns)
# Convert to numpy array
features = np.array(features)
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(features, labels);


# In[5]:


# Use the forest's predict method on the test data
predictions = rf.predict(features)


# In[6]:


print(classification_report(labels,predictions))


# In[7]:


#generating a report to extract the measure of interest using built-in sklearn function
report = classification_report(labels,predictions,digits=3,output_dict = True)

print("LogReg Model:")
print("Accuracy = {0:0.3f}".format(report["accuracy"]))
print("Precision = {0:0.3f}".format(report["1"]["precision"]))
print("Specificity = {0:0.3f}".format(report["0"]["recall"]))
print("Sensitivity = {0:0.3f}".format(report["1"]["recall"]))
print("F1-score = {0:0.3f}".format(report["1"]["f1-score"]))


# In[8]:


test = pd.read_csv(r"C:\\DataScience\\Jump2\\test.csv", sep=";")
test_predictions = rf.predict(test.values)


# In[9]:


pd.DataFrame(test_predictions).to_csv('final.csv', header=["final_status"], index=False, ) 


# In[44]:


r = pd.read_csv("final.csv")


# In[42]:


y = json.dumps(r.to_dict())


# In[43]:


with open("reto.json", "w") as outfile:
    outfile.write(y)


# In[ ]:




