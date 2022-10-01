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
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model


#label_folder = []
#total_size = 0
data_path = r".\python\NN\final\training"
#C:\Users\win10\Desktop\code\python\NN\project2\cifar10\train"

#os.walk() generates the file names(dirpath, dirnames, filenames) 
#in a directory tree by walking the tree either top-down or bottom-up.
#for root, dirts, files in os.walk(data_path): 
#    for dirt in dirts:
#        label_folder.append(dirt)
#    total_size += len(files)

    
#print("found",total_size,"files.")
#print("folder:",label_folder)








total_size = 0
FileName = []

for root, dirts, files in os.walk(data_path): 
    total_size += len(files)

    
print("found",total_size,"files.")



base_x_train = []
base_y_train = []


col_list = ["filename","category"]
df = pd.read_csv(r'.\NN\final\label.csv', usecols=col_list)
#FileName = df["filename"].tolist()
FileName = df["filename"].tolist()
nmp = df.to_numpy()
print(nmp[0:20])
np.random.shuffle(nmp)
print(nmp[0:20])






#base_y_train = df["category"].tolist()



for j in range(total_size):
    path = data_path + r'\\' + nmp[j][0]

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if np.array(img).shape[0]==640:
        img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    base_x_train.append(img)
    base_y_train.append(nmp[j][1])















#for i in range(len(label_folder)):
#    labelPath = data_path+r'\\'+label_folder[i]
    
    #listdir() returns a list containing the names of the entries in the directory given by path.
    #isfile() is used to check whether the specified path is an existing regular file or not.
#    FileName = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]
    
#    for j in range(len(FileName)):
#        path = labelPath+r'\\'+FileName[j]
        
        #use cv2.imread read image.
#        img = cv2.imread(path,cv2.IMREAD_COLOR)
        
#        base_x_train.append(img)
#        base_y_train.append(label_folder[i])


print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)



#Convert a category vector to a binary (0 or 1) matrix-type representation

base_y_train = to_categorical(base_y_train)


print(np.array(base_x_train).shape)
print(np.array(base_y_train).shape)




#shuffle
buffer1 = []
buffer2 = []

#for i in range(0,int((len(base_x_train)/219))):
#    for j in range(0,219):
#      buffer1.append(base_x_train[(i+int((len(base_x_train)/219))*j)])
#      buffer2.append(base_y_train[(i+int((len(base_x_train)/219))*j)])



#base_x_train = buffer1
#base_y_train = buffer2
    

#




val_ratio =0.05
train_num = int(total_size*(1-val_ratio))
x_train, x_valid = np.array(base_x_train)[:train_num], np.array(base_x_train)[train_num:]
y_train, y_valid = np.array(base_y_train)[:train_num], np.array(base_y_train)[train_num:]

print("Training data:", x_train.shape, y_train.shape)
print("Validation data:", x_valid.shape, y_valid.shape)







x_train = x_train / 255.0
x_valid = x_valid / 255.0








idx = random.randint(0, x_train.shape[0])
plt.imshow(x_train[idx])
plt.show()

print("Answer:", np.argmax(y_train[idx]))
print("Answer(one-hot):", y_train[idx])

#output_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#output_name = ['castle','apple','bridge','train','keyboard','tractor','kangaroo','woman','trout','tulip']


#x_train = x_train / 255.0
#x_valid = x_valid / 255.0









#a = [1,2,3,4,5]
#for i in range(0,6):
#    print(a[i])


model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(120,160,3))
x =model.output
x = keras.layers.Flatten()(x)
x =keras.layers.Dropout(0.5)(x)
output_layer = tf.keras.layers.Dense(219, activation = "softmax", name ="softmax")(x)
model = tf.keras.Model(inputs=model.input, outputs=output_layer)



epoch = 10
batch_size = 64

model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
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

model.save("ResNet50.hdf5")



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
Filename = []
total_size = 0
data_path = r".\Training data\Testing data"
x_test_images = []




#os.walk() generates the file names(dirpath, dirnames, filenames) 
#in a directory tree by walking the tree either top-down or bottom-up.
for root, dirts, files in os.walk(data_path): 
    for file in files:
        Filename.append(file)
    




for j in range(len(Filename)):
    path = data_path + r'\\' + Filename[j]

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    x_test_images.append(img)

total_size+=len(file)
print("found",total_size,"file")


#test
#final_list = model.predict(base_x_train2)
#output_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
output_name = ['castle','apple','bridge','train','keyboard','tractor','kangaroo','woman','trout','tulip']
#answer = ['castle','castle','castle','apple','apple','apple','bridge','bridge','bridge','train', 'train', 'train','keyboard','keyboard','keyboard','tractor','tractor','tractor','kangaroo','kangaroo','kangaroo','woman','woman','woman','trout','trout','trout','tulip','tulip','tulip']
#print('answer_num: ',len(answer))
final_list = []




x_test =np.array(x_test_images)
loaded_model =load_model("ResNet50.hdf5")






prbability_model = tf.keras.Sequential([loaded_model, tf.keras.layers.Softmax()])

x_test = x_test / 255.0
predictions = prbability_model.predict(x_test)
 #print(prediction)
 #print(np.argmax(prediction))
 

with open ("predict.csv","w+") as f:
    for i in range(0,len(filename)):
        fwrite(f"{filename[i]},{category[i]}\n")














