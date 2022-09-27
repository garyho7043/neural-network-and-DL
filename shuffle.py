import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import tensorflow.random
import numpy as np

a = [1,2,3,4,5]
b = [6,7,8,9,10]
data_num = len(a)
c= [[] for i in range(0,data_num)]
for i in range(0,data_num):
    c[i].append(a[i])
    c[i].append(b[i])

print(c)

np.random.shuffle(c)

print(c)

a = []
b = []
for i in range(0,data_num):
    a.append(c[i][0])
    b.append(c[i][1])    

print(a)
print(b)
