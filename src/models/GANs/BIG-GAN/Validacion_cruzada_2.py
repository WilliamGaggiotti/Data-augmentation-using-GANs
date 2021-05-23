import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time
import handshape_datasets as hd
from IPython import display
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, LeakyReLU, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np; np.random.seed(1)
import seaborn as sns
import multiprocessing

def load_dataset_with_subject(subject_test=1):
    
    data = hd.load('PugeaultASL_A')

    good_min = 40
    good_classes = []
    n_unique = len(np.unique(data[1]['y']))
    for i in range(n_unique):
        images = data[0][np.equal(i, data[1]['y'])]
        if len(images) >= good_min:
            good_classes = good_classes + [i]

    #x = data[0][np.in1d(data[1]['y'], good_classes)]

    #y = data[1]['y'][np.in1d(data[1]['y'], good_classes)]

    #s = data[1]['subjects'][np.in1d(data[1]['y'], good_classes)]

    #y_dict = dict(zip(np.unique(y), range(len(np.unique(y)))))
    #y = np.vectorize(y_dict.get)(y)

    #s_dict = dict(zip(np.unique(s), range(len(np.unique(s)))))
    #s = np.vectorize(s_dict.get)(s)
    
    x = data[0]

    y = data[1]['y']

    s = data[1]['subjects']
    
    

    classes = np.unique(y)
    n_classes = len(classes)

    x_train = x[np.not_equal(subject_test, s)]
    y_train = y[np.not_equal(subject_test, s)]
    x_test = x[np.equal(subject_test, s)]
    y_test = y[np.equal(subject_test, s)]
    
    shuffler = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]

    shuffler_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[shuffler_test]
    y_test = y_test[shuffler_test]
    
    #escalo al rango  [1-,1]
    x_train = (x_train.astype('float32') -127.5 ) / 127.5
    x_test = (x_test.astype('float32') -127.5 ) / 127.5


    return n_classes, x_train, y_train, x_test, y_test 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def model():
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[32, 32, 3]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(24, activation='softmax'))

    return model

def run_model(x_train, y_train, x_test, y_test, name, dict):

    convolutional_model = model()
    convolutional_model.compile(optimizer='Adam', 
                                loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy'])

    sumarry = convolutional_model.fit(x_train, y_train, batch_size=128, epochs=50, 
                                            validation_data=(x_test, y_test))
                                        
    dict[name] = sumarry.history
    
    
def run_model_with_ImageGenerator(train_gen, x_test, y_test, steps_per_epoch, name, dict):
    convolutional_model = model()
    convolutional_model.compile(optimizer='Adam', 
                                loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy'])
    
    sumarry = convolutional_model.fit_generator(train_gen, 
                               steps_per_epoch=steps_per_epoch,epochs=50, 
                               validation_data=(x_test, y_test))

    dict[name] = sumarry.history

if __name__ == "__main__":

    datasets_names = ['aug_100', 'aug_75', 'aug_50', 'aug_25']
    avg_normal_history = []
    avg_x_agu_50_history = []
    avg_x_agu_75_history = []
    avg_x_agu_25_history = []
    avg_x_agu_100_history = []
    avg_train_gen_history = []
 
    for i in range(0,2):
        
        print('######################')
        print('#     subject {}     #'.format(i))
        print('######################')
        ##################
        # cargo datasets #
        ##################
        
        numpy_data_path = 'numpy_data/PugeaultASL_A_2/subject_{}/'.format(i)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        #data original
        n_classes, x_train, y_train, x_test, y_test = load_dataset_with_subject(subject_test=i)
        #datasets_names.append('normal')
        p = multiprocessing.Process(target=run_model, args=(x_train, y_train, x_test, y_test, 'normal', return_dict))
        p.start()
        p.join()    
        
        #Datos mesclados con reales y GAN
        for dataset_name in datasets_names:
            x_agu = np.load(numpy_data_path+'x_'+dataset_name+'.npy')
            y_agu= np.load(numpy_data_path+'y_'+dataset_name+'.npy')
            p = multiprocessing.Process(target=run_model, args=(x_agu, y_agu, x_test, y_test, dataset_name, return_dict))
            p.start()
            p.join()

        #Generator con data augmentation
        train_datagen_aug = ImageDataGenerator(rotation_range=20)

        train_datagen_aug.fit(x_train)
        train_gen =  train_datagen_aug.flow(x_train, y_train, batch_size=128)
        steps_per_epoch = (len(x_train) / 128)
        #datasets_names.append('aug_train_gen')
        p = multiprocessing.Process(target=run_model_with_ImageGenerator, args=(train_gen, x_test, y_test, steps_per_epoch, 'aug_train_gen', return_dict))
        p.start()
        p.join()    
        
        avg_normal_history.append(return_dict['normal']['val_accuracy'])
        avg_x_agu_50_history.append(return_dict['aug_50']['val_accuracy'])
        avg_x_agu_75_history.append(return_dict['aug_75']['val_accuracy'])
        avg_x_agu_25_history.append(return_dict['aug_25']['val_accuracy'])
        avg_x_agu_100_history.append(return_dict['aug_100']['val_accuracy'])
        avg_train_gen_history.append(return_dict['aug_train_gen']['val_accuracy'])
        
    avg_normal_history = np.asarray(avg_normal_history)
    avg_x_agu_50_history = np.asarray(avg_x_agu_50_history)
    avg_x_agu_100_history = np.asarray(avg_x_agu_100_history)
    avg_x_agu_75_history = np.asarray(avg_x_agu_75_history)
    avg_x_agu_25_history = np.asarray(avg_x_agu_25_history)
    avg_train_gen_history = np.asarray(avg_train_gen_history)

    historys_path = 'historys/PugeaultASL_A_2/'
                                            
    #all_vs
    sns.set_style("darkgrid")
    plt.plot(np.mean(avg_normal_history, axis=0))
    plt.plot(np.mean(avg_x_agu_100_history, axis=0))
    plt.plot(np.mean(avg_x_agu_75_history, axis=0))
    plt.plot(np.mean(avg_x_agu_50_history, axis=0))
    plt.plot(np.mean(avg_x_agu_25_history, axis=0))
    plt.plot(np.mean(avg_train_gen_history, axis=0))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm','100','75','50','25','IMG'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[0][-1])[0:5]))
    plt.figtext(.6, .15, "Last_val_100 = {}".format(str(avg_x_agu_100_history[0][-1])[0:5]))
    plt.figtext(.4, .2, "Last_val_75 = {}".format(str(avg_x_agu_75_history[0][-1])[0:5]))
    plt.figtext(.4, .15, "Last_val_50 = {}".format(str(avg_x_agu_50_history[0][-1])[0:5]))
    plt.figtext(.2, .2, "Last_val_25 = {}".format(str(avg_x_agu_25_history[0][-1])[0:5]))
    plt.figtext(.2, .15, "Last_val_IMG = {}".format(str(avg_train_gen_history[0][-1])[0:5]))
    plt.savefig(historys_path+'all_vs')
    plt.show()

    def tsplot(data,**kw):
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        plt.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
        plt.plot(x, est,**kw)
        plt.margins(x=0)

    #normal vs 100 GAN
    tsplot(avg_normal_history)
    tsplot(avg_x_agu_100_history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm', '100'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[-1])[0:6]))
    plt.figtext(.6, .15, "Last_val_100 = {}".format(str(avg_x_agu_100_history[-1])[0:6]))
    plt.savefig(historys_path+'norm_VS_100')
    plt.show()
  
    #normal vs 100 GAN
    tsplot(avg_normal_history)
    tsplot(avg_x_agu_75_history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm', '75'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[-1])[0:6]))
    plt.figtext(.6, .15, "Last_val_75 = {}".format(str(avg_x_agu_75_history[-1])[0:6]))
    plt.savefig(historys_path+'norm_VS_75')
    plt.show()
    
    #normal vs 100 GAN
    tsplot(avg_normal_history)
    tsplot(avg_x_agu_50_history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm', '50'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[-1])[0:6]))
    plt.figtext(.6, .15, "Last_val_50 = {}".format(str(avg_x_agu_50_history[-1])[0:6]))
    plt.savefig(historys_path+'norm_VS_50')
    plt.show()

    #normal vs 100 GAN
    tsplot(avg_normal_history)
    tsplot(avg_x_agu_25_history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm', '25'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[-1])[0:6]))
    plt.figtext(.6, .15, "Last_val_25 = {}".format(str(avg_x_agu_25_history[-1])[0:6]))
    plt.savefig(historys_path+'norm_VS_25')
    plt.show()

    #normal vs 100 GAN
    tsplot(avg_normal_history)
    tsplot(avg_train_gen_history)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['norm', 'IMG'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(avg_normal_history[-1])[0:6]))
    plt.figtext(.6, .15, "Last_val_IMG = {}".format(str(avg_train_gen_history[-1])[0:6]))
    plt.savefig(historys_path+'norm_VS_IMG')
    plt.show()