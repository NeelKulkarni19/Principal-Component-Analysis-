#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# In[7]:


wine=pd.read_csv("F:/Dataset/wine.csv")


# In[8]:


wine


# In[9]:


wine2=wine.iloc[:,1:]


# In[10]:


wine2


# In[11]:


wine2.describe()


# In[16]:


wine3=wine2.values


# In[17]:


wine3


# In[19]:


wine4=scale(wine3)


# In[20]:


wine4


# In[24]:


pca=PCA(n_components=13)


# In[25]:


wine_pca=pca.fit_transform(wine4)


# In[26]:


wine_pca


# In[27]:


pca.components_


# In[28]:


variance=pca.explained_variance_ratio_


# In[29]:


variance


# In[30]:


variance1=np.cumsum(np.round(variance,4)*100)


# In[31]:


variance1


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


plt.plot(variance1)


# In[34]:


Dataframe=pd.concat([wine['Type'],pd.DataFrame(wine_pca[:,0:3])],axis=1)


# In[35]:


Dataframe


# In[36]:


fig=plt.figure(figsize=(60,30))


# In[37]:


sns.scatterplot(data=Dataframe)


# In[ ]:





# In[39]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[43]:


plt.figure(figsize=(25,8))
Dendogram= sch.dendrogram(sch.linkage(wine4,'complete'))


# In[44]:


hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
hclusters


# In[45]:


y=pd.DataFrame(hclusters.fit_predict(wine4),columns=['clustersid'])
y['clustersid'].value_counts()


# In[46]:


wine5=wine.copy()
wine5['clustersid']=hclusters.labels_
wine5


# In[47]:


from sklearn.cluster import KMeans


# In[49]:


abc=[]
for i in range(1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine4)
    abc.append(kmeans.inertia_)


# In[50]:


plt.plot(range(1,6),abc)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[53]:


clusters3=KMeans(3,random_state=30).fit(wine4)
clusters3


# In[54]:


clusters3.labels_


# In[55]:


wine6=wine.copy()
wine6['cluster3sid']=clusters3.labels_


# In[56]:


wine6


# In[57]:


wine6['cluster3sid'].value_counts()


# In[ ]:




