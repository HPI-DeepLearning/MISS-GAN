import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.misc

from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Activation, LSTM, Conv2D, Conv2DTranspose, Dense, TimeDistributed, Flatten, Reshape, Cropping2D, GaussianNoise, Concatenate, BatchNormalization, SeparableConv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.losses import mean_squared_error
from keras.optimizers import Adadelta, RMSprop
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

#K.set_learning_phase(1) #set learning phase

sequences_per_batch = 1
epochs = 100
image_size = 256
sequence_end = 1000
train_seq = 10
#train_cnt = int(sequence_length / train_seq)
file_list = 'Patient.txt'
input_mode = 'train'
input_data = 1
input_attention = 3
input_dimension = input_data + input_attention
output_dimension = 3
base = 42
folder = 'liverdata'

# load data list
files = np.genfromtxt(file_list, dtype='str')

# define model
def conv_block(m, dim, acti, bn, res, do=0.2):
    n = TimeDistributed(Conv2D(dim, 6, padding='same'))(m)
    n = TimeDistributed(LeakyReLU())(n)
    n = BatchNormalization()(n) if bn else n
    n = TimeDistributed(Dropout(do))(n) if do else n
    n = TimeDistributed(Conv2D(dim, 6, padding='same'))(n)
    n = TimeDistributed(LeakyReLU())(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = TimeDistributed(MaxPooling2D())(n) if mp else TimeDistributed(Conv2D(dim, 4, strides=2, padding='same'))(n)
        
        #print(n.shape)
        #print(m.shape)
        
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = TimeDistributed(UpSampling2D())(m)
            m = TimeDistributed(Conv2D(dim, 4, padding='same'))(m)
            m = TimeDistributed(LeakyReLU())(m)
        else:
            m = TimeDistributed(Conv2DTranspose(dim, 4, strides=2, padding='same'))(m)
            m = TimeDistributed(LeakyReLU())(m)
            
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
        
        l = TimeDistributed(Flatten())(m)
        #l = LSTM(4 * 4 * 128, stateful=True, return_sequences=True)(l)
        l = LSTM(2048, stateful=True, return_sequences=True)(l)
        l = TimeDistributed(Reshape((2, 2, 2048/4)))(l)
        m = l
        #m = Concatenate()([l, m])
        
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(input_shape, out_ch=1, start_ch=64, depth=7, inc_rate=1.5, activation='relu', 
         dropout=0.4, batchnorm=True, maxpool=True, upconv=True, residual=False):
    i = Input(batch_shape=input_shape)
    #o = TimeDistributed(ZeroPadding2D(padding=8))(i)
    o = TimeDistributed(SeparableConv2D(start_ch, 7, padding='same'))(i)
    o = level_block(o, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    #o = TimeDistributed(Cropping2D(cropping=8))(o)
    o = TimeDistributed(Conv2D(out_ch, 1, activation='tanh'))(o)
    return Model(inputs=i, outputs=o)

model = UNet((sequences_per_batch, train_seq, image_size, image_size, input_dimension), out_ch=6, start_ch=base)
model.compile(loss='mean_squared_error', optimizer=RMSprop())

for k in model.layers:
    print(k.output_shape)

plot_model(model, to_file='model.png')

def load_sequence(p, is_train=True):
    pattern = p.decode("utf-8")
    val = []
    
    for s in xrange(sequence_end):
        name = pattern.format(s, folder)
        
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
            image = image/127.5
        
            conv.append(image)
        
        val.append([np.stack(conv, axis=2)])

    if len(val) == 0:
        return np.zeros((1, train_seq, image_size, image_size, input_dimension + output_dimension)) - 1
        
    st = np.stack(val, axis=1)
    if st.shape[1] % train_seq > 0:
        z = np.zeros((1, train_seq - (st.shape[1] % train_seq), image_size, image_size, input_dimension + output_dimension)) 
        o = np.append(z, st, axis=1)
        o = o - 1
        return o
     
    st = st - 1
    return st
    
def makeMask(gt, ct):
    gt = (gt+1) / 2
    ct = (ct+1) / 2
    
    t_mask = np.clip(gt - ct, 0, 1)
    n_mask = np.clip(ct - gt, 0, 1)
    
    t_mask = (t_mask * 2) - 1
    n_mask = (n_mask * 2) - 1
    
    return np.concatenate((t_mask, n_mask), axis=4)
    
def extractGT(seq):
    gt, data = np.split(batch_sequence, [output_dimension], axis=4)
    gta, gtb, gtc = np.split(gt, 3, axis=4)
    z1, cta, ctb, ctc = np.split(data, input_dimension, axis=4)
    
    m1 = makeMask(gta, cta)
    m2 = makeMask(gtb, ctb)
    m3 = makeMask(gtc, ctc)
    
    gt = np.concatenate((m1, m2, m3), axis=4)
    return data, gt, np.concatenate((seq, gt), axis=4)
    
# train
number_of_sequences = files.size
for epoch in range(epochs):
    
    np.random.shuffle(files)
    for sequence in range(number_of_sequences):
        print('E: {}   S: {} '.format(epoch,sequence))
        seq = load_sequence(files[sequence])
        
        cnt = seq.shape[1] / train_seq
        
        batch_sequences = np.split(seq, cnt, axis=1)
        model.reset_states()
        for batch_sequence in batch_sequences:
            #gt, data = np.split(batch_sequence, [output_dimension], axis=4)
            #zz, gt, zz = np.split(gt, 3, axis=4)
            data, gt, s = extractGT(batch_sequence)
            v = model.train_on_batch(data, gt)
            print(v)
     
    # epoch done, do some validation output
    #seq = load_sequence(files[0])
    #print('Validation: {} '.format(epoch))
    #batch_sequences = np.split(seq, train_cnt, axis=1)
    #model.reset_states()
    #c = 0
    #for batch_sequence in batch_sequences:
        #gt, data = np.split(batch_sequence, [output_dimension], axis=4)
    #    data, gt, s = extractGT(batch_sequence)
    #    out = model.predict_on_batch(data)
        
    #    all = np.append(s, out, axis=4)
    #    all = all.reshape((train_seq, image_size, image_size, input_dimension + output_dimension + 6 + 6))
    #    sp = np.split(all, train_seq, axis=0)
    #    sp = [s.reshape((image_size, image_size, input_dimension + output_dimension + 6 + 6)) for s in sp]

     #   haa = np.concatenate(sp, axis=0)
     #   jaa = np.concatenate(np.split(haa, input_dimension + output_dimension + 6 + 6, axis=2), axis=1)
     #   fa = (jaa+1.)/2.
     #   yo = np.concatenate((fa, fa, fa), axis=2)
     #   scipy.misc.imsave('outv2/val_{0}_{1}.png'.format(epoch, c), yo)
     #   c = c + 1
    
    model.save('liver.h5')
    
