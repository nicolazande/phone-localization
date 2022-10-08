#-------------------------- import stuff -----------------------------------
import os, math, json, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #do not display tensorflow logs
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import Image
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input


#----------------- image path ---------------------------------------------
img_name = sys.argv[1]


#------------- get GPU if there is -----------------------------------------
SEED = 1234
tf.random.set_seed(SEED)
cwd = os.getcwd()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
    	pass

elif cpus:
    try:
        logical_cpus= tf.config.experimental.list_logical_devices('CPU') 
    except RuntimeError as e:
    	pass



#-------------------- check if weights present -----------------------------
model_name = "model"
model_dir = os.path.join(cwd, 'weights')
if not os.path.exists(model_dir):
    print("\nNo weights found. Run first train_phone_finder.py\n")
    quit()


#------------------------ info ---------------------------------------------
img_h = 326
img_w = 490
N_classes = 2 #binary
source_dir = os.path.join(cwd,'find_phone')



#------------------- create model ------------------------------------------
start_f = 8
depth = 5

model = tf.keras.Sequential()
encoder = tf.keras.Sequential()

#Features extraction
for i in range(depth):

    if i == 0:
        input_shape = [img_h, img_w, 3]
    else:
        input_shape=[None]

    # Conv block: Conv2D -> Activation -> Pooling
    encoder.add(tf.keras.layers.Conv2D(filters=start_f, 
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     input_shape=input_shape))
    encoder.add(tf.keras.layers.ReLU())

    if i < depth -1:
        encoder.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    start_f *= 2

model.add(encoder)

#model uses un GAP after encoder --> key point
model.add(tf.keras.layers.GlobalAveragePooling2D())
    
#binary classifier
model.add(tf.keras.layers.Dense(units=N_classes, activation='softmax'))




#------------------- load weights ------------------------------------------
model.load_weights(os.path.join(model_dir, model_name))
    
    

    
#----------- new model till heatmap gap and resizing --------------------------
out_model = tf.keras.Model(inputs=model.input, outputs=[model.output, model.layers[0].get_output_at(-1)])
resize_feature = tf.keras.layers.experimental.preprocessing.Resizing(img_h, img_w, interpolation="bilinear")
    
#get test image
if os.path.isfile(img_name):
	test_img = preprocess_input(np.array(Image.open(img_name)))
else:
	test_img = preprocess_input(np.array(Image.open(os.path.join(source_dir, img_name))))




#--------------- make prediction ----------------------------------------------
softmax_out, last_enc_feature = out_model(tf.expand_dims(test_img, 0))
pred = tf.argmax(softmax_out, 1) #get class
resized_feature = resize_feature(last_enc_feature) #resize features

#get wheights che corrispondono a phone + resizing
mask_weights = model.layers[-1].weights[0][:, 1]
resized_feature = tf.reshape(resized_feature, shape=[img_h*img_w, resized_feature.shape[-1]])

#create class activation map
cam = tf.linalg.matmul(resized_feature, tf.expand_dims(mask_weights, -1))
cam = tf.reshape(cam, shape=[img_h, img_w])

#get indexes where high probability of phone
max_val = np.max(cam)
delta = 0.3
if max_val >= 0:
    focus = np.array(np.where(cam > (1-delta)*np.max(cam)))
else:
    focus = np.array(np.where(cam > (1+delta)*np.max(cam)))
    
#convert ho xy coordinates in pixels
coord = np.array([[focus[1][i],focus[0][i]] for i in range(len(focus[0]))])




#----------- get center in relative coordinates -------------------------------
padding_correction = [-0.05, -0.08]
center = coord.mean(axis = 0)/[img_w, img_h] + padding_correction

print("\n" + str(center[0])+ " " + str(center[1]) + "\n")

