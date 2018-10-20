
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


# read in data
df = pd.read_csv('iris.csv')


# In[3]:


# preview dataset
df.head(6)


# In[4]:


# numerical statistics of dataset
df.describe()


# In[5]:


# view the relationships between variables; color code by species type
sns.pairplot(df, hue="species");


# In[6]:


# Perform Simple Random Sampling with replacement
# split data into training and test sets; set random state to 0 for reproducibility 
X_train, X_test, y_train, y_test = train_test_split(df[['sepal_length', 'sepal_width', 
                                                        'petal_length', 'petal_width']],
                                                    df['species'], random_state = 0)


# In[7]:


# see how data has been split
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))


# In[8]:


# initialize the Estimator object
knn = KNeighborsClassifier(n_neighbors=1)


# In[9]:


# fit the model to training set in order to predict classes
knn.fit(X_train, y_train)


# In[10]:


# create a prediction array for our test set
y_pred = knn.predict(X_test)


# In[11]:


# based on the training dataset, our model predicts the following for the test set:
pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)], 
          ignore_index=False, axis=1)


# In[12]:


# what is our prediction score?
print('Accuracy rate: ', metrics.accuracy_score(y_test, y_pred))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[13]:


# Plot accuracy of train vs test# experimenting with different n values
k_range = list(range(1,50))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


# In[14]:


# compare with regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy rate: ', metrics.accuracy_score(y_test, y_pred))

