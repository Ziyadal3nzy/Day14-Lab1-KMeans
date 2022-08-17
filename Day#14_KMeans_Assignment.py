#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[36]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the Data

# In[2]:


df = pd.read_csv('College_Data', index_col=0) # index_col=0 to put first column as index


# In[3]:


df.head()


# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# In[4]:


df.info()


# In[5]:


df.describe()


# # EDA

# ## Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. 

# In[6]:


sns.scatterplot(data=df, x=df['Room.Board'], y=df['Grad.Rate'], hue=df['Private'])


# ## Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.

# In[7]:


sns.scatterplot(data=df, x=df['Outstate'], y=df['F.Undergrad'], hue=df['Private'])


# ## Use sns.FacetGrid to create a stacked histogram showing Out of State Tuition based on the Private column. 

# In[12]:


g = sns.FacetGrid(df, hue="Private", palette = 'coolwarm')
g.map(plt.hist, 'Outstate', bins=20)


# ## Use sns.FacetGrid to create a stacked histogram showing Grad.Rate based on the Private column. 

# In[30]:


g = sns.FacetGrid(df, hue="Private", palette = 'coolwarm')
g.map(plt.hist, 'Grad.Rate', )


# ## Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?

# In[21]:


df[df['Grad.Rate'] > 100]  #Name of school = Cazenovia College


# ## Set that school's graduation rate to 100 then re-do the histogram visualization.

# In[26]:


df[df['Grad.Rate'] > 100] = 100


# In[27]:


Graduat = sns.FacetGrid(df)


# In[29]:


g.map(plt.hist, 'Grad.Rate', )


# ## K Means Model

# In[34]:


from sklearn.cluster import KMeans


# In[35]:


kmeans = KMeans(n_clusters=2)


# ## Fit the model to all the data except for the Private label.

# In[37]:


kmeans.fit(df.drop('Private', axis=1))


# ## What are the cluster center vectors?

# In[43]:


kmeans.cluster_centers_


# ## Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.

# In[44]:


df['Cluster'] = df['Private'].apply(lambda x: 1 if x== 'yes' else 0)


# ## Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.

# In[50]:


from sklearn.metrics import classification_report, confusion_matrix


# In[61]:


print(confusion_matrix(df['Cluster'], kmeans.labels_))


# In[57]:


kmeans.labels_

