
# coding: utf-8

# In[12]:

import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from math import sqrt
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# In[18]:


class KNNClassifier:
    
    dataset=[]
    test_data=[]
    predicted_labels=[]
    num_neighbors= 7
    
    def train(self,filename):
        self.dataset= pd.read_csv(filename, header=None).to_numpy()
        
    def predict(self, filename):
        self.test_data= pd.read_csv(filename, header=None).to_numpy()
        for row in self.test_data:
            label = self.predict_classification(row)
            self.predicted_labels.append(label)
        return self.predicted_labels
        
    def euclidean_distance(self,row1, row2):
        distance = 0.0
        for i in range(1,len(row1)):
            distance += (row1[i-1] - row2[i])**2
        return sqrt(distance)
    
    def get_neighbors(self,test_row):
        distances = list()
        for train_row in self.dataset:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][0])
 
        return neighbors
    
    def predict_classification(self,test_row):
        neighbors = self.get_neighbors(test_row)
        output_values = [row[0] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction


