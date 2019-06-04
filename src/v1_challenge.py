#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Lambda, LSTM, GRU
from sklearn.metrics import cohen_kappa_score


# # Data Importation

# In[2]:


dfx = pd.read_csv('../data/x_train.csv').set_index('ID')
dfy = pd.read_csv('../data/y_train.csv').set_index('ID')
dfx_test = pd.read_csv('../data/x_test.csv').set_index('ID')


# In[3]:


dfx.head()


# In[4]:


dfy.head()


# In[5]:


if dfy.shape[0] == dfx.shape[0]:
    print("Same number of samples, all good.")
else:
    print("Different number of samples, problem!")


# # Data Exploration

# * **Sanity check:** diff etat1/etat2, neuron_id usefulness

# In[6]:


# Différence entre le nombre d'etats 1 et d'etats 0.
"""dfy.TARGET.value_counts().plot(kind='bar')
plt.show()"""


# In[7]:


# Should we keep the neuron_id col ?
xtest_uniques = dfx_test.neuron_id.unique()
x_uniques = dfx.neuron_id.unique()
diff = [x for x in x_uniques if x in xtest_uniques]
diff


# * **Create Xtrain Ytrain:** numpy array from df, with dimensions *(sample_nb, timestep_nb, feature_nb)*

# In[8]:


Xtrain = dfx.iloc[:,1:].values
Xtrain = Xtrain[..., np.newaxis]
Xtrain.shape


# In[9]:


Ytrain = dfy.values
Ytrain.shape


# * **Deprecated:** Concatenate neuron_id to every timestep of a sample

# In[10]:


# TODO : Make a 3d numpy array from our pandas df
# Shape = [samples, timestamps, features]
"""
timesteps_arr = dfx.iloc[:,1:].values
timesteps_arr = timesteps_arr[..., np.newaxis]
timesteps_arr.shape

neuron_arr = dfx.iloc[:,0].values
neuron_arr.shape

neuron_arr = np.broadcast_to(neuron_arr[:,None,None], timesteps_arr.shape)
final_arr = np.concatenate((timesteps_arr,neuron_arr), axis=2)
final_arr.shape
"""


# # Model Training

# ## References

# * Arxiv: [Neural activity classification with machine learning models trained oninterspike interval series data](https://arxiv.org/pdf/1810.03855.pdf) => PCA and KNN
# * Github: [PySpike: Python library to analyze spike Train](https://github.com/mariomulansky/PySpike) => Obscure mathematical measurements between spike trains
# * Profil: [Prof expert en spike train analysis](http://xtof.perso.math.cnrs.fr/)

# ## Code

# ### Deep-Learning 1: blunt RNN

# In[11]:


timestamp_nb = 50
feature_nb = 1

input_shape = (timestamp_nb, feature_nb)
x = input_tensor = Input(input_shape)
x = LSTM(32, return_sequences=False)(x)
x = output_tensor = Dense(1)(x)
model = Model(input_tensor, output_tensor)


# In[12]:


model.summary()


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, epochs=5)


# ### Domain-knowledge 1: KNN with SPIKE- and ISI- synchronization distances

# In[ ]:
