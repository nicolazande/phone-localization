#-------------------------- import stuff -----------------------------------
import tensorflow as tf
import numpy as np
import os, math, json, sys
from PIL import Image
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping


#get current directory
cwd = os.getcwd()

#------------------------ info ---------------------------------------------
if len(sys.argv) < 2:
	print("\ninsert image folder path\n")
	quit()
else:
	source_dir = sys.argv[1]
	if os.path.exists(source_dir):
		if not os.path.isfile(os.path.join(source_dir, "labels.txt")):
			print("\nno labels found\n")
			quit()
		print("\npath found correctly\n")
	else:
		print("\ninvalid path\n")
		quit()


#------- standard setup (change as needed) ---------------------------------
save_weights = False

start_training = False

prepare_patches = False

load_weights = True

apply_data_augmentation = True
#--------------------------------------------------------------------------

#self adaptation to user choices (dont change)
model_name = "model"
model_dir = os.path.join(cwd, 'weights')
if not os.path.exists(model_dir):
    start_training = True
    save_weights = True

patch_dir = os.path.join(cwd, 'Patches')
if not os.path.exists(patch_dir) and start_training:
	prepare_patches = True
	


#------------- get GPU if there is -----------------------------------------
SEED = 1234
tf.random.set_seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
elif cpus:
    try:
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        print(e)



#---------------- select parameters -----------------------------------------
#image size
img_h = 326
img_w = 490
N_classes = 2 #binary
BS = 3 #batch size
split = 0.9 #test-validation split



#-------- function checking if patch contains phone -------------------------
def not_phone(i, j, patch_size, center, perc):    
    #useful to balance the dataset
    if (np.random.rand()) > perc:
        return False
    if i*patch_size[0] > center[1] + math.floor(patch_size[0]/2):
        return True
    if (i+1)*patch_size[0] < center[1] - math.floor(patch_size[0]/2):
        return True
    if j*patch_size[1] > center[0] + math.floor(patch_size[1]/2):
        return True
    if (j+1)*patch_size[1] < center[0] - math.floor(patch_size[1]/2):
        return True
    return False
    
    
    
#------------- prepare patches --------------------------------------------

#desidered patch size
patch_size = [81, 122]

scale = min(math.floor(img_w/patch_size[0]), math.ceil(img_h/patch_size[1])) #scale factor wrt image

if prepare_patches:
    
    #create eventually patch dir
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
        
    filenames = os.listdir(source_dir)
    filenames = [f for f in filenames]

    #----- read from txt nomi ------------------------------------
    with open(os.path.join(source_dir, 'labels.txt')) as f:
        lines = f.read().splitlines()
    lines = np.array([l.split() for l in lines])
    #------- save polentarutti in dictionary ---------------------
    d = dict()
    for l in lines:
        d[l[0]] = [np.float32(l[1]), np.float32(l[2])]
    
    #dict to save 0-1 values
    rubrica = {}
    
    for curr_filename in filenames:
        #check if there is corrisponding key
        if curr_filename in d.keys():
            
            img = np.array(Image.open(os.path.join(source_dir, curr_filename)))
            
            center = d[curr_filename]  #phone position
            center[0] = math.floor(center[0]*img_w)
            center[1] = math.floor(center[1]*img_h)            

            phone_img = Image.fromarray(img[max(0, center[1] - math.floor(patch_size[0]/2)) : min(img_h, center[1] + math.floor(patch_size[0]/2)), \
                                            max(0, center[0] - math.floor(patch_size[1]/2)) : min(img_w, center[0] + math.floor(patch_size[1]/2)),:], 'RGB')

            phone_img.save(os.path.join(patch_dir, curr_filename))
            rubrica[curr_filename] = 1 #it is phone
            
            
            #get background patches (not all --> want balanced classes)
            for i in range(scale):
                for j in range(scale):
                    if not_phone(i, j, patch_size, center, 0.25):
                        back_img = Image.fromarray(img[i*patch_size[0] : (i+1)*patch_size[0], j*patch_size[1] : (j+1)*patch_size[1], :], 'RGB')
                        back_img.save(os.path.join(patch_dir, str(i) + "_" + str(j) + "_" + curr_filename))
                        rubrica[str(i) + "_" + str(j) + "_" + curr_filename] = 0
                        
    
    #save rubrica
    with open(os.path.join(cwd,'rubrica.json'), 'a+') as f:
        json.dump(rubrica, f)
        
        
     
        
#--------------- load data da json --------------------------------------
if start_training:
    with open(os.path.join(cwd,'rubrica.json')) as json_file:
        rubrica = json.load(json_file)
    
    
    
#--------------- data augmentation --------------------------------------
if start_training:
    if apply_data_augmentation:
        #immagini
        img_data_gen = ImageDataGenerator(rotation_range=30,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        brightness_range=[0.7,1.3],
                                        zoom_range=0.3,
                                        #shear_range=10,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant')

    else:
        img_data_gen = ImageDataGenerator(fill_mode='constant')
    



#------------- build my dataset and datagen --------------------------------
class CustomDataset(tf.keras.utils.Sequence):
     
    def __init__(self, dataset_dir, subset_filenames, rubrica, img_generator=None, preprocessing_function=None, out_shape=[img_w,img_h]):
        
        self.dataset_dir = dataset_dir #images path
        self.subset_filenames = subset_filenames #images names
        self.out_shape = out_shape
        self.img_generator = img_generator
        self.preprocessing_function = preprocessing_function #use VGGone --> better results

    #len function
    def __len__(self):
        return len(self.subset_filenames)
    
    #getitem
    def __getitem__(self, index):
        
        curr_filename = self.subset_filenames[index] #get image name      
        img = Image.open(os.path.join(dataset_dir, curr_filename))        
        img_arr = np.array(img.resize(self.out_shape))
        
        #preprocessing image polentarutti
        if self.preprocessing_function is not None:
            img_arr = self.preprocessing_function(img_arr)
        
        #bring to one hot encoding
        if rubrica[curr_filename]:
            return img_arr, [0, 1] #phone
        else:
            return img_arr, [1, 0] #no phone
            
            

#--------------- dataset creation and splitting -----------------------------------------
if start_training:
    dataset_dir = patch_dir

    #divide randomly in subsets
    subset_filenames = os.listdir(dataset_dir)

    train_imm = []
    valid_imm = []
    perc = 0.9
    for i in range(len(subset_filenames)):
        if (np.random.rand()) < perc:
            train_imm.append(subset_filenames[i])
        else:
            valid_imm.append(subset_filenames[i])


    #check partitions
    print("\ndataset of Patches contains:")
    print("	tot_imgs = " + str(len(subset_filenames)))
    print("	training_imgs = " + str(len(train_imm)))
    print("	validation_imgs = " + str(len(valid_imm)))

    #define datasets
    dataset = CustomDataset(dataset_dir = dataset_dir,
                            subset_filenames = train_imm,
                            rubrica = rubrica,
                            img_generator = img_data_gen,
                            preprocessing_function = preprocess_input)

    dataset_valid = CustomDataset(dataset_dir = dataset_dir,
                                subset_filenames = valid_imm,
                                rubrica = rubrica,
                                preprocessing_function=preprocess_input)
                              
                              

#---------- iterate over dataset ---------------------------------------------------------
if start_training:
    #training
    train_dataset = tf.data.Dataset.from_generator(lambda: dataset,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=([img_h, img_w, 3], [N_classes]))
    train_dataset = train_dataset.batch(BS)

    train_dataset = train_dataset.repeat()

    #validation
    valid_dataset = tf.data.Dataset.from_generator(lambda: dataset_valid,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=([img_h, img_w, 3], [N_classes]))
    valid_dataset = valid_dataset.batch(BS)

    valid_dataset = valid_dataset.repeat()



#--------------------------- build the model ---------------------------------------------
if start_training:
    start_f = 8
    depth = 5

    model = tf.keras.Sequential()

    #encoder --> extract features
    encoder = tf.keras.Sequential()

    # Features extraction
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

    #use GAP --> key point for weakly supervised localization
    model.add(tf.keras.layers.GlobalAveragePooling2D())
        
    #binary classifier
    model.add(tf.keras.layers.Dense(units=N_classes, activation='softmax'))



#-------------- load weights (if enabled) ---------------------------------------------------
if start_training:
    model_name = "model"
    model_dir = os.path.join(cwd, 'weights')
    if load_weights and os.path.exists(model_dir):
        model.load_weights(os.path.join(model_dir, model_name))
    else:
        print("\nweights not found, start training + save weights\n")
        save_weights = True

    
    
    
#------------------- changes learning rate as i want not with decay --------------------------
class CLR(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CLR, self).__init__()
        self.schedule = schedule
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('non hai settato lr')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        
        
        
#-------------------------- define callbacks -------------------------------------------------
callbacks = []

early_stop = True
if early_stop:
    es_callback = EarlyStopping(monitor='val_loss', patience=20)
    callbacks.append(es_callback)

LUT_STD = []

#pass learning rate
def get_lr_std(epoch, lr):
    if epoch < LUT_STD[0][0]:
        return LUT_STD[0][1]
    elif epoch > LUT_STD[len(LUT_STD)-1][0]:
        return LUT_STD[len(LUT_STD)-1][1]
    for i in range(len(LUT_STD)):
        if epoch == LUT_STD[i][0]:
            print("\nnuovo lr: "+str(LUT_STD[i][1]))
            return LUT_STD[i][1]
    return lr

callbacks.append(CLR(get_lr_std))



#----------------------- compile model --------------------------------------------------------
if start_training:
    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics='accuracy')

    #class weights --> for unbalanced dataset (already balanced with patches)
    class_weight = {0: 1.,
                    1: 1.}
                
                

#----------------------------- fit model ------------------------------------------------------
if start_training or not load_weights:
	EP = 10 #epochs

	#select lr as wanted
	LUT_STD = [(0, 1e-3),
		       (4, 1e-4),
		       (8, 1e-5)]

	model.fit(x=train_dataset,
		  epochs=EP,
		  steps_per_epoch=len(dataset),
		  validation_data=valid_dataset,
		  validation_steps=len(dataset_valid), 
		  callbacks=callbacks,
		  #class_weight=class_weight
		  )
		  
		  		


#----------------------------- salve weights --------------------------------------------------
if save_weights and start_training:
    model_name = "model"
    model_dir = os.path.join(cwd, 'weights')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    model.save_weights(os.path.join(model_dir, model_name))
    print("\nweights saved\n")



#--------------------- finish -----------------------------------------------------------------
print("\nfinished! Ready to run find_phone.py")

            
    
    


