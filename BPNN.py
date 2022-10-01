#use numpy to classify MNIST database with BPNN

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
from os  import listdir
from os.path import isfile, join
import math




# subroutine


def sigmoid(x):
  return 1 / (1 + math.exp(-x))



def open_picture(windowname, img):

  cv2.imshow(windowname, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def make_initial_weight_array(hidden_layer_node_num, input_num):
    
    weighted_array = [[] for i in range(hidden_layer_node_num)]         
    for m in range(0, hidden_layer_node_num):
        for i in range(0, input_num):
            weighted_array[m].append(np.random.random(None))
            
            
    return weighted_array




def one_layer_calculation(weighted_array, input_column_matrix_list):          
    


    summation_array = []
    activation_array = []
    summation_array = np.matmul(weighted_array, input_column_matrix_list)#(layer_node_num*input_num)*(input_num*1)
    
    for i in range(0, len(summation_array)):
        activation_array.append(sigmoid(summation_array[i]))
        
    return activation_array 


  
def learning_rule_sigma(learning_rule_advise_time, target_value_array, activation_value_array,  next_layer_weighed_array, next_layer_sigma):# one layer one time, focused on sigma.
    
    if learning_rule_advise_time == 0 :#if j is output node
       output_sigma =[]
       for i in range(0,len(target_value_array)):
          sigma =(target_value_array[i] - activation_value_array[i])*(activation_value_array[i])*(1-activation_value_array[i])
          output_sigma.append(sigma)
    
       return output_sigma
    
    
    
    else: #if j is hidden node
      hidden_layer_sigma = []
      
    
      for i in range(0,len(activation_value_array)):
        
        summation = 0
        for j in range(0,len(next_layer_sigma)):
           buffer = next_layer_sigma[j]*next_layer_weighed_array[j][i]
           summation += buffer
           
        
     
        sigma =(activation_value_array[i])*(1-activation_value_array[i])*summation
        hidden_layer_sigma.append(sigma)
    
      return hidden_layer_sigma
    
    
    

def learning_rule_adjusting(learning_rate, weighted_array, next_layer_sigma, activation_value_array, hidden_layer_node_num, input_num):# one layer one time, focused on weight-adjustment.
    
    delta_weight = [[] for i in range(hidden_layer_node_num)]         
    for m in range(0, hidden_layer_node_num):
        for i in range(0, input_num):
            delta_weight[m].append(0)
    
    new_weighted_array = [[] for i in range(hidden_layer_node_num)]         
    for m in range(0, hidden_layer_node_num):
        for i in range(0, input_num):
            new_weighted_array[m].append(0)
    

    
    
    for i in range(0,hidden_layer_node_num):
        for j in range(0,input_num): 
          delta_weight[i][j] = learning_rate*next_layer_sigma[i]*activation_value_array[j]
          
    
    
    
    for i in range(0,hidden_layer_node_num):
        for j in range(0,input_num): 
           new_weighted_array[i][j] = weighted_array[i][j] + delta_weight[i][j]
    
    return new_weighted_array
    




#--------------------------------------------------------

label_folder = []
total_size = 0
data_path =r".\Training data"  #path

for root, dirts, files in os.walk(data_path):
    for dirt in dirts:
        label_folder.append(dirt)
    total_size += len(files)

print("found", total_size, "file." )
print("folder:", label_folder)

#--------------------------------------------------------

base_x_train = []
base_y_train = []

for i in range(len(label_folder)):
    labelPath = data_path + r'\\' + label_folder[i]
    FileName = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]
    

    for j in range(len(FileName)):
        path = labelPath + r'\\' + FileName[j]
    
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        base_x_train.append(img)
        base_y_train.append(label_folder[i])

print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)
    


#--------------------------------------------------------

#FileNameO = []
#total_sizeO = 0
#data_pathO =r".\MLandDL\mnist_testData\Testing_data"  #path

#for root, dirts, files in os.walk(data_pathO):
#    for file in files:
#        FileNameO.append(file)
#    total_sizeO += len(files)
    

#print("found", total_sizeO, "file." )


#for i in range(len(FileNameO)):
#    path = data_pathO + r'\\' + FileNameO[i]   
#    txt_name = os.path.splitext(FileNameO[i].split('.')[0])[0]
    
#    with open("410887043.txt",'a+') as f:
#        f.write(txt_name + " " + "ans" +"\n")

#--------------------------------------------------------

#training

# elementary_data
epoch = 100
learning_rate = 0.5
#initial weight array :all element is 0.1
data_num_in_same_category = 4000
hidden_layer_node_num = 5
output_num = 5
perform_time = 0


#settting of target array
zero  = np.array([1, 0, 0, 0, 0])
one   = np.array([0, 1, 0, 0, 0])
two   = np.array([0, 0, 1, 0, 0])
three = np.array([0, 0, 0, 1, 0])
four  = np.array([0, 0, 0, 0, 1])
target_arrays_list = [zero, one, two, three, four]



#make x a 1*784 array
n = len(base_x_train) # data num


print(n)

input_column_matrix_lists = [[] for i in range(n)]
#input_column_matrix_lists[0][index] == base_x_train[0][i][j]
for m in range(0, n):
    (h,w) = base_x_train[m].shape[:2]
    for i in range(0, h):
        for j in range(0, w):
            input_column_matrix_lists[m].append(0.5*base_x_train[0][i][j])


# convecient list to deal with data
weighted_arrays_list_output = [] #weight from hidden layer to output
weighted_arrays_list_hidden = [] #weight from input layer to hidden layer
sigma_arrays_list_output = [] #sigma from hidden layer to output
sigma_arrays_list_hidden = [] #sigma from input layer to hidden layer
Buffer_list = [] #buffer to solve new and old weighted array's problem




#A = np.array([[1,2,1,2],[3,4,5,5]])
#B = np.array([[0,1],[0,2],[0,3]])
#J = np.array([4,5])
#print(one_layer_calculation(A,B))

#find sigma 34
#T=np.array([1,1,1])
#A=np.array([0.1,0.4,1])
#N=np.array([0.3,0.15])
#G=np.array([[0.2,-0.1],[0.7,-1],[0.4,0.4]])
#U= learning_rule_sigma(0,T,A,None,None)
#print(U)
#U=np.array([0.81,0.096,0])
#print(learning_rule_sigma(1,None,N,G,U))
#print(learning_rule_adjusting(0.5,B,T,J,3,2))




# if all a of x == t for

for w in range(0, epoch):
  for m in range(0,n):
    output_array1 =[]
    output_array2 =[]
    weighted_array1 = []
    weighted_array2 = []

    if perform_time == 0:
      
      weighted_array1 = make_initial_weight_array(hidden_layer_node_num, len(input_column_matrix_lists[m]))
      weighted_arrays_list_hidden = weighted_array1
      output_array1 = one_layer_calculation(weighted_array1, input_column_matrix_lists[m])


      weighted_array2 = make_initial_weight_array(output_num, len(output_array1))
      weighted_arrays_list_output = weighted_array2
      output_array2 = one_layer_calculation(weighted_array2, output_array1)

    elif perform_time > 0 :
      
      
      output_array1 = one_layer_calculation(weighted_arrays_list_hidden, input_column_matrix_lists[m])
      output_array2 = one_layer_calculation(weighted_arrays_list_output, output_array1)


    count = 0
    for i in range(0,len(output_array2)):
        if target_arrays_list[int(m/data_num_in_same_category)][i] == output_array2[i]:
            count+=1
        
        
    if count == len(output_array2):
        pass 
    else:
     learning_rule_advise_time = 0
    
    
     sigma_arrays_list_output = learning_rule_sigma(learning_rule_advise_time, target_arrays_list[int(m/data_num_in_same_category)], output_array2, None, None)
     Buffer_list = learning_rule_adjusting(learning_rate, weighted_arrays_list_output, sigma_arrays_list_output, output_array1, output_num, len(output_array1))
     learning_rule_advise_time+=1
     
     sigma_arrays_list_hidden = learning_rule_sigma(learning_rule_advise_time, None, output_array1, weighted_arrays_list_output, sigma_arrays_list_output)
     weighted_arrays_list_output = Buffer_list
     weighted_arrays_list_hidden = learning_rule_adjusting(learning_rate, weighted_arrays_list_hidden, sigma_arrays_list_hidden, input_column_matrix_lists[m], hidden_layer_node_num, len(input_column_matrix_lists[m]))
     learning_rule_advise_time+=1
 
    perform_time+=1

print(weighted_arrays_list_hidden)
print(weighted_arrays_list_output)




#print(weighted_arrays_list_output)


 


#test


#--------------------------------------------------------
FileNameO = []
total_sizeO = 0
data_pathO =r".\MLandDL\Testing_data"  #path

for root, dirts, files in os.walk(data_pathO):
    for file in files:
        FileNameO.append(file)
    total_sizeO += len(files)
    

print("found", total_sizeO, "file." )
    




base_x_train2 = []


    
for j in range(len(FileNameO)):
     path2 = data_pathO + r'\\' + FileNameO[j]
    
     img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)  
     base_x_train2.append(img2)


















#make x a 1*784 array
n2 = len(base_x_train2) # data num
input_column_matrix_lists2 = [[] for i in range(n2)]
#input_column_matrix_lists[0][index] == base_x_train[0][i][j]
for m in range(0, n2):
    (h,w) = base_x_train2[m].shape[:2]
    for i in range(0, h):
        for j in range(0, w):
            input_column_matrix_lists2[m].append(0.5*base_x_train2[0][i][j])

















#--------------------------------------------------------



print(input_column_matrix_lists2)










output_array_t1 = []
output_array_t2 = []
final_list = []

for m in range(0,25):
  output_array_t1 = one_layer_calculation(weighted_arrays_list_hidden, input_column_matrix_lists2[m])
  
  print(output_array_t1)
  
  output_array_t2 = one_layer_calculation(weighted_arrays_list_output, output_array_t1)
  
  print(output_array_t2)




  abs_diff_list = [0 for i in range(0,5)]
  mark = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
  for i in range(0,5):
      sum = 0
      for j in range(0,5):
    
          buffer = abs(target_arrays_list[i][j]-output_array_t2[j])
          sum+=buffer
    
      abs_diff_list[i] = sum
    
        
  for i in range(0,5):
   for j in range(0,5):
       if abs_diff_list[i]>abs_diff_list[j]:
           mark[i][j]+=1
       elif abs_diff_list[i]<abs_diff_list[j]:
           mark[j][i]+=1


  sum2 = 0
  final = 0
  compare = [0,0,0,0,0]
  data_category = [2,3,4,5,7]
  for i in range(0,5):
      sum2 = 0
      for j in range(0,5):
          sum2+=mark[i][j]
    
      compare[i] = sum2

  for i in range(0,5):
       if compare[i] ==0:
           final = data_category[i]
           final_list.append(final)


print(final_list)
            
        
        
     
FileNameO = []
total_sizeO = 0
data_pathO =r".\MLandDL\Testing_data"  #path

for root, dirts, files in os.walk(data_pathO):
    for file in files:
        FileNameO.append(file)
    total_sizeO += len(files)
    

print("found", total_sizeO, "file." )





for i in range(0,len(final_list)):
 final_list[i]  = str(final_list[i])
for i in range(len(FileNameO)):
    path = data_pathO + r'\\' + FileNameO[i]   
    txt_name = os.path.splitext(FileNameO[i].split('.')[0])[0]
    
    with open("410887043.txt",'a+') as f:
        f.write(txt_name + " " +  final_list[i] +"\n")
