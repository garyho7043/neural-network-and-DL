import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import matplotlib as plt
import os
from os import listdir 
from os.path import isfile, join
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import random
import matplotlib.pyplot as plt





label_folder = []
total_size = 0
data_path = r"C:\Users\win10\Desktop\code\python\NN\project2\real\Training data\Training data"
#C:\Users\win10\Desktop\code\python\NN\project2\cifar10\train"

#os.walk() generates the file names(dirpath, dirnames, filenames) 
#in a directory tree by walking the tree either top-down or bottom-up.
for root, dirts, files in os.walk(data_path): 
    for dirt in dirts:
        label_folder.append(dirt)
    total_size += len(files)

    
print("found",total_size,"files.")
print("folder:",label_folder)


base_x_train = []
base_y_train = []

for i in range(len(label_folder)):
    labelPath = data_path+r'\\'+label_folder[i]
    
    #listdir() returns a list containing the names of the entries in the directory given by path.
    #isfile() is used to check whether the specified path is an existing regular file or not.
    FileName = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]
    
    for j in range(len(FileName)):
        path = labelPath+r'\\'+FileName[j]
        
        #use cv2.imread read image.
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        
        base_x_train.append(img)
        base_y_train.append(label_folder[i])


print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)



#Convert a category vector to a binary (0 or 1) matrix-type representation

base_y_train = to_categorical(base_y_train)


print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)




#shuffile
buffer1 = []
buffer2 = []

for i in range(0,int((len(base_x_train)/len(label_folder)))):
    for j in range(0,len(label_folder)):
      buffer1.append(base_x_train[(i+int((len(base_x_train)/len(label_folder)))*j)])
      buffer2.append(base_y_train[(i+int((len(base_x_train)/len(label_folder)))*j)])



base_x_train = buffer1
base_y_train = buffer2
    

#





train_num = len(base_x_train) - 250
x_train, x_valid = np.array(base_x_train)[:train_num], np.array(base_x_train)[train_num:]
y_train, y_valid = np.array(base_y_train)[:train_num], np.array(base_y_train)[train_num:]

print("Training data:", x_train.shape, y_train.shape)
print("Validation data:", x_valid.shape, y_valid.shape)

idx = random.randint(0, x_train.shape[0])
plt.imshow(x_train[idx])
plt.show()

print("Answer:", np.argmax(y_train[idx]))
print("Answer(one-hot):", y_train[idx])

#output_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
output_name = ['castle','apple','bridge','train','keyboard','tractor','kangaroo','woman','trout','tulip']
x_train = x_train / 255.0
x_valid = x_valid / 255.0



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),#
    tf.keras.layers.MaxPool2D(),#
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(25, activation="relu"),#
    tf.keras.layers.Dense(20, activation="relu"),#
    tf.keras.layers.Dense(15, activation="relu"),#
    tf.keras.layers.Dense(10, activation="softmax")
])



print(model.summary())


epoch = 70
batch_size = 20

model.compile(
    loss="categorical_crossentropy", 
    optimizer="sgd", 
    metrics=["accuracy"]
)
history = model.fit(
    x_train, 
    y_train, 
    epochs=epoch, 
    batch_size=batch_size,
    validation_data=(x_valid, y_valid)
)

print(model.summary())

model.save("my_model.hdf5")



# Get the classification accuracy and loss-value
# for the training-set.
acc = history.history['accuracy']
loss = history.history['loss']

# Get it for the validation-set (we only use the test-set).
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Plot the accuracy and loss-values for the training-set.
plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
plt.plot(loss, 'o', color='b', label='Training Loss')
    
# Plot it for the test-set.
plt.plot(val_acc, linestyle='--', color='r', label='Val Acc.')
plt.plot(val_loss, 'o', color='r', label='Val Loss')

# Plot title and legend.
plt.title('Training and Val Accuracy')
plt.legend()

# Ensure the plot shows correctly.
plt.show()



# test and prediction-----------------------------------------------------------
# append
#label_folder = []
FilenameO = []
total_size = 0
data_pathO = r"C:\Users\win10\Desktop\code\python\NN\project2\real\Training data\410887043\410887043"
#C:\Users\win10\Desktop\code\python\NN\project2\real\Training data\Testing data00"
#C:\Users\win10\Desktop\code\python\NN\project2\cifar10\test"

#os.walk() generates the file names(dirpath, dirnames, filenames) 
#in a directory tree by walking the tree either top-down or bottom-up.
for root, dirts, files in os.walk(data_pathO): 
    for file in files:
        FilenameO.append(file)
    total_size += len(files)

    
print("found",total_size,"files.")
print("folder:",FilenameO)


base_x_train2 = []
base_y_train2 = []



for j in range(len(FilenameO)):
    path2 = data_pathO + r'\\' + FilenameO[j]

    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    base_x_train2.append(img2)


#test
#final_list = model.predict(base_x_train2)
#output_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
output_name = ['castle','apple','bridge','train','keyboard','tractor','kangaroo','woman','trout','tulip']
#answer = ['castle','castle','castle','apple','apple','apple','bridge','bridge','bridge','train', 'train', 'train','keyboard','keyboard','keyboard','tractor','tractor','tractor','kangaroo','kangaroo','kangaroo','woman','woman','woman','trout','trout','trout','tulip','tulip','tulip']
#print('answer_num: ',len(answer))
final_list = []

prbability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

for i in range(0,len(base_x_train2)):
 img = (np.expand_dims(base_x_train2[i],0))

 img = img /255.0


 prediction = prbability_model.predict(img)
 #print(prediction)
 #print(np.argmax(prediction))
 index = np.argmax(prediction)
 final_list.append(index)
















#file I/O
for i in range(0, len(final_list)):
    final_list[i] = str(final_list[i])
for i in range(len(FilenameO)):
    path = data_pathO + r'\\' + FilenameO[i]
    txt_name = os.path.splitext(FilenameO[i].split('.')[0])[0]

    with open("410887043.txt" , 'a+') as f:
        f.write(txt_name + " " + final_list[i] +"\n")
