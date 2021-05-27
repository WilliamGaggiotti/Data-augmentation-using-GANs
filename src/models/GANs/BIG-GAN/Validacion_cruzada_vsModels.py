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
    

if __name__ == "__main__":

    datasets_names = ['aug_100', 'aug_75', 'aug_50', 'aug_25']
    avg_x_agu_50_history = []
    avg_x_agu_75_history = []
    avg_x_agu_25_history = []
    avg_x_agu_100_history = []
    avg_x_agu_50_history_2 = []
    avg_x_agu_75_history_2 = []
    avg_x_agu_25_history_2 = []
    avg_x_agu_100_history_2 = []
 
 
    for i in range(0,5):
        
        numpy_data_path_2 = 'numpy_data/PugeaultASL_A_2/subject_{}/'.format(i)
        numpy_data_path = '/media/willys/MULTIBOOT/tesis/numpy_data/PugeaultASL_A/subject_{}/'.format(i)
        history_iamges_path = 'historys/PugeaultASL_A/'
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        #data original
        _, _, _, x_test, y_test = load_dataset_with_subject(subject_test=i)
          
        
        #Datos mesclados con reales y GAN_1
        for dataset_name in datasets_names:
            x_agu = np.load(numpy_data_path+'x_'+dataset_name+'.npy')
            y_agu= np.load(numpy_data_path+'y_'+dataset_name+'.npy')
            p = multiprocessing.Process(target=run_model, args=(x_agu, y_agu, x_test, y_test, dataset_name, return_dict))
            p.start()
            p.join()

        #Datos mesclados con reales y GAN_2
        for dataset_name in datasets_names:
            x_agu = np.load(numpy_data_path_2+'x_'+dataset_name+'.npy')
            y_agu= np.load(numpy_data_path_2+'y_'+dataset_name+'.npy')
            p = multiprocessing.Process(target=run_model, args=(x_agu, y_agu, x_test, y_test, dataset_name+'_2', return_dict))
            p.start()
            p.join()
          
        avg_x_agu_50_history.append(return_dict['aug_50']['val_accuracy'])
        avg_x_agu_75_history.append(return_dict['aug_75']['val_accuracy'])
        avg_x_agu_25_history.append(return_dict['aug_25']['val_accuracy'])
        avg_x_agu_100_history.append(return_dict['aug_100']['val_accuracy'])
        avg_x_agu_50_history_2.append(return_dict['aug_50_2']['val_accuracy'])
        avg_x_agu_75_history_2.append(return_dict['aug_75_2']['val_accuracy'])
        avg_x_agu_25_history_2.append(return_dict['aug_25_2']['val_accuracy'])
        avg_x_agu_100_history_2.append(return_dict['aug_100_2']['val_accuracy'])
        
    avg_x_agu_50_history = np.asarray(avg_x_agu_50_history)
    avg_x_agu_100_history = np.asarray(avg_x_agu_100_history)
    avg_x_agu_75_history = np.asarray(avg_x_agu_75_history)
    avg_x_agu_25_history = np.asarray(avg_x_agu_25_history)
    avg_x_agu_50_history_2 = np.asarray(avg_x_agu_50_history_2)
    avg_x_agu_100_history_2 = np.asarray(avg_x_agu_100_history_2)
    avg_x_agu_75_history_2 = np.asarray(avg_x_agu_75_history_2)
    avg_x_agu_25_history_2 = np.asarray(avg_x_agu_25_history_2)

    historys_path = 'historys/PugeaultASL_A_2/'
                                            

    def tsplot(data,**kw):
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        plt.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
        plt.plot(x, est,**kw)
        plt.margins(x=0)

    #normal vs 50/50 GAN
    tsplot(avg_x_agu_50_history)
    tsplot(avg_x_agu_50_history_2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['GAN_1', 'GAN_2'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_50_history, axis=0)[-1])[0:6]))
    plt.figtext(.6, .13, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_50_history_2, axis=0)[-1])[0:6]))
    plt.savefig(historys_path+'50-50_GAN_1_VS_GAN_2')
    plt.show()
  
    #normal vs 100 GAN
    tsplot(avg_x_agu_100_history)
    tsplot(avg_x_agu_100_history_2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['GAN_1', 'GAN_2'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_100_history, axis=0)[-1])[0:6]))
    plt.figtext(.6, .13, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_100_history_2, axis=0)[-1])[0:6]))
    plt.savefig(historys_path+'100_GAN_1_VS_GAN_2')
    plt.show()
    
    #normal vs 75/25 GAN
    tsplot(avg_x_agu_75_history)
    tsplot(avg_x_agu_75_history_2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['GAN_1', 'GAN_2'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_75_history, axis=0)[-1])[0:6]))
    plt.figtext(.6, .13, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_75_history_2, axis=0)[-1])[0:6]))
    plt.savefig(historys_path+'75-25_GAN_1_VS_GAN_2')
    plt.show()

    #normal vs 25/75 GAN
    tsplot(avg_x_agu_25_history)
    tsplot(avg_x_agu_25_history_2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['GAN_1', 'GAN_2'], loc='upper left')
    plt.figtext(.6, .2, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_25_history, axis=0)[-1])[0:6]))
    plt.figtext(.6, .13, "Last_val_norm = {}".format(str(np.mean(avg_x_agu_25_history_2, axis=0)[-1])[0:6]))
    plt.savefig(historys_path+'25-75_GAN_1_VS_GAN_2')
    plt.show()
