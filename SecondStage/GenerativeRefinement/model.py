import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.misc

from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Activation, LSTM, Conv2D, Conv2DTranspose, Dense, TimeDistributed, Flatten, Reshape, Cropping2D, GaussianNoise, Concatenate, BatchNormalization, SeparableConv2D
from keras.losses import mean_squared_error
from keras.optimizers import Adadelta
from keras import backend as K

K.set_learning_phase(1) #set learning phase

sequences_per_batch = 1
epochs = 100
image_size = 240
sequence_length = 50#150
sequence_start = 50#2
train_seq = 10#15
train_cnt = int(sequence_length / train_seq)
file_list = 'train.txt'
input_mode = 'test'
input_data = 4
input_attention = 3
input_dimension = input_data + input_attention
output_dimension = 3
base = 32
folder = 'data'

# load data list
files = np.genfromtxt(file_list, dtype='str')

# define model
di = Input(batch_shape=(sequences_per_batch, train_seq, image_size, image_size, input_data))
n0 = GaussianNoise(0.01)(di)

ai = Input(batch_shape=(sequences_per_batch, train_seq, image_size, image_size, input_attention))
m0 = Concatenate(axis=4)([di, ai])

c0 = TimeDistributed(SeparableConv2D(base, (12, 12), strides=(4,4), padding='same', use_bias=True))(m0)
b0 = c0#BatchNormalization()(c0)
a0 = TimeDistributed(Activation('tanh'))(b0)
d0 = TimeDistributed(Dropout(0.1))(a0)

c1 = TimeDistributed(Conv2D(base, (8, 8), strides=(2,2), padding='same', use_bias=True))(d0)
b1 = c1#BatchNormalization()(c1)
a1 = TimeDistributed(Activation('tanh'))(b1)
d1 = TimeDistributed(Dropout(0.1))(a1)

c2 = TimeDistributed(Conv2D(2 * base, (8, 8), strides=(2,2), padding='same', use_bias=True))(d1)
b2 = c2#BatchNormalization()(c2)
a2 = TimeDistributed(Activation('tanh'))(b2)
d2 = TimeDistributed(Dropout(0.1))(a2)

c3 = TimeDistributed(Conv2D(2 * base, (8, 8), strides=(2,2), padding='same', use_bias=True))(d2)
b3 = c3#BatchNormalization()(c3)
a3 = TimeDistributed(Activation('tanh'))(b3)
d3 = TimeDistributed(Dropout(0.1))(a3)

c4 = TimeDistributed(Conv2D(4 * base, (4, 4), strides=(2,2), padding='same', use_bias=True))(d3)
b4 = c4#BatchNormalization()(c4)
a4 = TimeDistributed(Activation('tanh'))(b4)
d4 = TimeDistributed(Dropout(0.1))(a4)

c5 = TimeDistributed(Conv2D(4 * base, (2, 2), strides=(2,2), padding='same', use_bias=True))(d4)
b5 = c5#BatchNormalization()(c5)
a5 = TimeDistributed(Activation('tanh'))(b5)
d5 = TimeDistributed(Dropout(0.1))(a5)

c6 = TimeDistributed(Conv2D(8 * base, (2, 2), strides=(2,2), padding='same', use_bias=True))(d5)
b6 = c6#BatchNormalization()(c6)
a6 = TimeDistributed(Activation('tanh'))(b6)
d6 = TimeDistributed(Dropout(0.1))(a6)

f0 = TimeDistributed(Flatten())(d6)
l0 = LSTM(8 * base, stateful=True, return_sequences=True, activation='tanh', recurrent_activation='tanh')(f0)
r0 = TimeDistributed(Reshape((1, 1, 8 * base)))(l0)

m1 = Concatenate(axis=4)([r0, d6])

c7 = TimeDistributed(Conv2DTranspose(8 * base, (1, 1), strides=(2,2), padding='same', use_bias=True))(m1)
b7 = c7#BatchNormalization()(c7)
a7 = TimeDistributed(Activation('tanh'))(b7)
d7 = TimeDistributed(Dropout(0.1))(a7)

m2 = Concatenate(axis=4)([d5, d7])

c8 = TimeDistributed(Conv2DTranspose(4 * base, (2, 2), strides=(2,2), padding='same', use_bias=True))(m2)
b8 = c8#BatchNormalization()(c8)
a8 = TimeDistributed(Activation('tanh'))(b8)
d8 = TimeDistributed(Dropout(0.1))(a8)

m3 = Concatenate(axis=4)([d4, d8])

c9 = TimeDistributed(Conv2DTranspose(4 * base, (4, 4), strides=(2,2), padding='same', use_bias=True))(m3)
b9 = c9#BatchNormalization()(c9)
a9 = TimeDistributed(Activation('tanh'))(b9)
d9 = TimeDistributed(Dropout(0.1))(a9)

m4 = Concatenate(axis=4)([d3, d9])

c10 = TimeDistributed(Conv2DTranspose(2 * base, (8, 8), strides=(2,2), padding='same', use_bias=True))(m4)
cr0 = TimeDistributed(Cropping2D(cropping=((0, 1), (0, 1))))(c10)
b10 = cr0#BatchNormalization()(cr0)
a10 = TimeDistributed(Activation('tanh'))(b10)
d10 = TimeDistributed(Dropout(0.1))(a10)

m5 = Concatenate(axis=4)([d2, d10])

c11 = TimeDistributed(Conv2DTranspose(2 * base, (8, 8), strides=(2,2), padding='same', use_bias=True))(m5)
b11 = c11#BatchNormalization()(c11)
a11 = TimeDistributed(Activation('tanh'))(b11)
d11 = TimeDistributed(Dropout(0.1))(a11)

m6 = Concatenate(axis=4)([d1, d11])

c12 = TimeDistributed(Conv2DTranspose(base, (8, 8), strides=(2,2), padding='same', use_bias=True))(m6)
b12 = c12#BatchNormalization()(c12)
a12 = TimeDistributed(Activation('tanh'))(b12)
d12 = TimeDistributed(Dropout(0.1))(a12)

m7 = Concatenate(axis=4)([d0, d12])

c13 = TimeDistributed(Conv2DTranspose(base, (12, 12), strides=(4,4), padding='same', activation='tanh'))(m7)
c14 = TimeDistributed(Conv2D(output_dimension, (1, 1), activation='sigmoid'))(c13)

#x0 = TimeDistributed(Conv2DTranspose(output_dimension, (12, 12), strides=(4,4), padding='same', activation='sigmoid'))(d0)
model = Model(inputs=[di, ai], outputs=c14)
model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

plot_model(model, to_file='model.png')

def load_sequence(p, is_train=True):
    pattern = p.decode("utf-8")
    val = []
    
    for s in xrange(sequence_length):
        name = pattern.format('test', sequence_start + s, folder)
        #print(name)
        try:
            input_img = scipy.misc.imread(name, mode='L').astype(np.float)
        except:
            continue
        images = np.split(input_img, input_dimension + output_dimension, axis=1)
        
        half_offset = 4
        offset = half_offset * 2
        hypersize = image_size + offset
        fullsize = 256 + offset

        h1 = int(np.ceil(np.random.uniform(1e-2, offset)))
        w1 = int(np.ceil(np.random.uniform(1e-2, offset)))

        conv = []
        for image in images:
            #print(image.shape)
            top = int((fullsize - image.shape[1]) / 2)
            bottom = fullsize - image.shape[1] - top
            image = np.append(np.zeros((image.shape[0], top)), image, axis=1)
            image = np.append(image, np.zeros((image.shape[0], bottom)), axis=1)
            
            left = int((fullsize - image.shape[0]) / 2)
            right = fullsize - image.shape[0] - left
            image = np.append(np.zeros((left, image.shape[1])), image, axis=0)
            image = np.append(image, np.zeros((right, image.shape[1])), axis=0)

            tmp = scipy.misc.imresize(image, [hypersize, hypersize], interp='nearest')
            if is_train:
                image = tmp[h1:h1+image_size, w1:w1+image_size]
            else:
                image = tmp[half_offset:half_offset+image_size, half_offset:half_offset+image_size]
            image = image/255.
		
            conv.append(image)
        
        val.append([np.stack(conv, axis=2)])

    st = np.stack(val, axis=1)
    z = np.zeros((1, sequence_length - st.shape[1], image_size, image_size, input_dimension + output_dimension)) 
    return np.append(z, st, axis=1)
    
# train
number_of_sequences = files.size
for epoch in range(epochs):
    
    np.random.shuffle(files)
    for sequence in range(number_of_sequences):
        print('E: {}   S: {} '.format(epoch,sequence))
        seq = load_sequence(files[sequence])
        #print(seq.shape)
        batch_sequences = np.split(seq, train_cnt, axis=1)
        model.reset_states()
        for batch_sequence in batch_sequences:
            gt, data, attention = np.split(batch_sequence, [output_dimension, output_dimension + input_data], axis=4)
            #np.set_printoptions(threshold='nan')
            #print(gt)
            v = model.train_on_batch([data, attention], [gt])
            print(v)
     
    # epoch done, do some validation output
    seq = load_sequence(files[0])
    print('Validation: {} '.format(epoch))
    batch_sequences = np.split(seq, train_cnt, axis=1)
    model.reset_states()
    c = 0
    for batch_sequence in batch_sequences:
        gt, data, attention = np.split(batch_sequence, [output_dimension, output_dimension + input_data], axis=4)
        out = model.predict_on_batch([data, attention])

        all = np.append(batch_sequence, out, axis=4)
        all = all.reshape((train_seq, image_size, image_size, input_dimension + output_dimension + output_dimension))
        sp = np.split(all, train_seq, axis=0)
        sp = [s.reshape((image_size, image_size, input_dimension + output_dimension + output_dimension)) for s in sp]

        haa = np.concatenate(sp, axis=0)
        jaa = np.concatenate(np.split(haa, input_dimension + output_dimension + output_dimension, axis=2), axis=1)
        fa = jaa#(jaa+1.)/2.
        yo = np.concatenate((fa, fa, fa), axis=2)
        scipy.misc.imsave('out/val_{0}_{1}.png'.format(epoch, c), yo)
        ++c
    
    model.save('lstm.h5')
    
