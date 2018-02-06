from __future__ import division
import os
import scipy.misc
import time
import numpy as np
import tensorflow as tf
from glob import glob
from six.moves import xrange
from ops import *
from utils import *


# Init Parameters

def init(pix):

    if pix.phase != 'train':
        pix.batch_size = 1 #was 12

    pix.image_size = 512 // pix.speed_factor
    pix.epoch = 100
    pix.beta1 = 0.5
    pix.checkpoint_dir = './checkpoint-{}'.format(pix.axis)
    pix.sample_dir = './sample-{}'.format(pix.axis)
    pix.test_dir = './test-{}'.format(pix.axis)
    pix.contest_dir = './contest-{}'.format(pix.axis)
    pix.L1_lambda = 100.0

    # Batch normalization : deals with poor initialization helps gradient flow

    pix.d_bn1 = batch_norm(name='d_bn1')
    pix.d_bn2 = batch_norm(name='d_bn2')
    pix.d_bn3 = batch_norm(name='d_bn3')

    pix.g_bn_e2 = batch_norm(name='g_bn_e2')
    pix.g_bn_e3 = batch_norm(name='g_bn_e3')
    pix.g_bn_e4 = batch_norm(name='g_bn_e4')
    pix.g_bn_e5 = batch_norm(name='g_bn_e5')
    pix.g_bn_e6 = batch_norm(name='g_bn_e6')
    pix.g_bn_e7 = batch_norm(name='g_bn_e7')
    pix.g_bn_e8 = batch_norm(name='g_bn_e8')

    pix.g_bn_d1 = batch_norm(name='g_bn_d1')
    pix.g_bn_d2 = batch_norm(name='g_bn_d2')
    pix.g_bn_d3 = batch_norm(name='g_bn_d3')
    pix.g_bn_d4 = batch_norm(name='g_bn_d4')
    pix.g_bn_d5 = batch_norm(name='g_bn_d5')
    pix.g_bn_d6 = batch_norm(name='g_bn_d6')
    pix.g_bn_d7 = batch_norm(name='g_bn_d7')
    

# Declare Model

def build_model(pix):
    pix.real_data = tf.placeholder(tf.float32,
                                    [pix.batch_size, pix.image_size, pix.image_size,
                                        pix.input_c_dim + pix.output_c_dim],
                                    name='real_A_and_B_images')

    pix.real_B = pix.real_data[:, :, :, :pix.output_c_dim]
    pix.real_A = pix.real_data[:, :, :, pix.output_c_dim:pix.input_c_dim + pix.output_c_dim]

    pix.fake_B = generator(pix, pix.real_A)

    pix.real_AB = tf.concat([pix.real_A, pix.real_B], 3)
    pix.fake_AB = tf.concat([pix.real_A, pix.fake_B], 3)
    pix.D, pix.D_logits = discriminator(pix, pix.real_AB, reuse=False)
    pix.D_, pix.D_logits_ = discriminator(pix, pix.fake_AB, reuse=True)

    pix.fake_B_sample = sampler(pix, pix.real_A)

    pix.d_sum = tf.summary.histogram("d", pix.D)
    pix.d__sum = tf.summary.histogram("d_", pix.D_)
    pix.fake_B_sum = tf.summary.image("fake_B", pix.fake_B)

    pix.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pix.D_logits, labels=tf.ones_like(pix.D)))
    pix.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pix.D_logits_, labels=tf.zeros_like(pix.D_)))
    pix.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pix.D_logits_, labels=tf.ones_like(pix.D_))) \
                    + pix.L1_lambda * tf.reduce_mean(tf.abs(pix.real_B - pix.fake_B))

    pix.d_loss_real_sum = tf.summary.scalar("d_loss_real", pix.d_loss_real)
    pix.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", pix.d_loss_fake)

    pix.d_loss = pix.d_loss_real + pix.d_loss_fake

    pix.g_loss_sum = tf.summary.scalar("g_loss", pix.g_loss)
    pix.d_loss_sum = tf.summary.scalar("d_loss", pix.d_loss)

    t_vars = tf.trainable_variables()

    pix.d_vars = [var for var in t_vars if 'd_' in var.name]
    pix.g_vars = [var for var in t_vars if 'g_' in var.name]

    pix.saver = tf.train.Saver()

def load_random_samples(pix):
    data = np.random.choice(glob('./datasets/{0}/val/*.png'.format(pix.dataset_name, pix.axis)), pix.batch_size)
    sample = [load_data(sample_file, pix.image_size, pix.input_c_dim, pix.output_c_dim) for sample_file in data]

    sample_images = np.array(sample).astype(np.float32)
    return sample_images

def sample_model(pix, epoch, idx):
    sample_images = load_random_samples(pix)
    samples, d_loss, g_loss = pix.sess.run(
        [pix.fake_B_sample, pix.d_loss, pix.g_loss],
        feed_dict={pix.real_data: sample_images}
    )
    samples = np.split(samples, pix.output_c_dim, axis=3)
    samples = np.concatenate(samples, axis=2)
	
    save_images(samples, [pix.batch_size, 1],
                './{}/train_{:02d}_{:04d}.png'.format(pix.sample_dir, epoch, idx))
    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

def train(pix):
    """Train pix2pix"""
    d_optim = tf.train.AdamOptimizer(pix.lr, beta1=pix.beta1) \
                        .minimize(pix.d_loss, var_list=pix.d_vars)
    g_optim = tf.train.AdamOptimizer(pix.lr, beta1=pix.beta1) \
                        .minimize(pix.g_loss, var_list=pix.g_vars)

    init_op = tf.global_variables_initializer()
    pix.sess.run(init_op)

    pix.g_sum = tf.summary.merge([pix.d__sum,
        pix.fake_B_sum, pix.d_loss_fake_sum, pix.g_loss_sum])
    pix.d_sum = tf.summary.merge([pix.d_sum, pix.d_loss_real_sum, pix.d_loss_sum])
    pix.writer = tf.summary.FileWriter("./logs", pix.sess.graph)

    counter = 1
    start_time = time.time()

    if load(pix, pix.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for epoch in xrange(pix.epoch):
        data = sorted(glob('./datasets/{0}/train/*.png'.format(pix.dataset_name, pix.axis)))
        batch_idxs = len(data) // pix.batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*pix.batch_size:(idx+1)*pix.batch_size]
            batch = [load_data(batch_file, pix.image_size, pix.input_c_dim, pix.output_c_dim, is_train=True) for batch_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
                
            # Update D network
            _, summary_str = pix.sess.run([d_optim, pix.d_sum],
                                            feed_dict={ pix.real_data: batch_images })
            pix.writer.add_summary(summary_str, counter)
            
            # Update G network
            _, summary_str = pix.sess.run([g_optim, pix.g_sum],
                                            feed_dict={ pix.real_data: batch_images })
            pix.writer.add_summary(summary_str, counter)
            
            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            _, summary_str = pix.sess.run([g_optim, pix.g_sum],
                                            feed_dict={ pix.real_data: batch_images })
            pix.writer.add_summary(summary_str, counter)
            
            errD_fake = pix.d_loss_fake.eval({pix.real_data: batch_images})
            errD_real = pix.d_loss_real.eval({pix.real_data: batch_images})
            errG = pix.g_loss.eval({pix.real_data: batch_images})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(counter, 100) == 1:
                sample_model(pix, epoch, idx)

            if np.mod(counter, 500) == 2:
                save(pix, pix.checkpoint_dir, counter)

def discriminator(pix, image, y=None, reuse=False):

    with tf.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, pix.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x pix.df_dim)
        h1 = lrelu(pix.d_bn1(conv2d(h0, pix.df_dim*2, name='d_h1_conv')))
        # h1 is (64 x 64 x pix.df_dim*2)
        h2 = lrelu(pix.d_bn2(conv2d(h1, pix.df_dim*4, name='d_h2_conv')))
        # h2 is (32x 32 x pix.df_dim*4)
        h3 = lrelu(pix.d_bn3(conv2d(h2, pix.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x pix.df_dim*8)
        h4 = linear(tf.reshape(h3, [pix.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

def generator(pix, image, y=None):
    with tf.variable_scope("generator") as scope:

        s = pix.image_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, pix.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x pix.gf_dim)
        e2 = pix.g_bn_e2(conv2d(lrelu(e1), pix.gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x pix.gf_dim*2)
        e3 = pix.g_bn_e3(conv2d(lrelu(e2), pix.gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x pix.gf_dim*4)
        e4 = pix.g_bn_e4(conv2d(lrelu(e3), pix.gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x pix.gf_dim*8)
        e5 = pix.g_bn_e5(conv2d(lrelu(e4), pix.gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x pix.gf_dim*8)
        e6 = pix.g_bn_e6(conv2d(lrelu(e5), pix.gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x pix.gf_dim*8)
        e7 = pix.g_bn_e7(conv2d(lrelu(e6), pix.gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x pix.gf_dim*8)
        e8 = pix.g_bn_e8(conv2d(lrelu(e7), pix.gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x pix.gf_dim*8)

        pix.d1, pix.d1_w, pix.d1_b = deconv2d(tf.nn.relu(e8),
            [pix.batch_size, s128, s128, pix.gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(pix.g_bn_d1(pix.d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x pix.gf_dim*8*2)

        pix.d2, pix.d2_w, pix.d2_b = deconv2d(tf.nn.relu(d1),
            [pix.batch_size, s64, s64, pix.gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(pix.g_bn_d2(pix.d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x pix.gf_dim*8*2)

        pix.d3, pix.d3_w, pix.d3_b = deconv2d(tf.nn.relu(d2),
            [pix.batch_size, s32, s32, pix.gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(pix.g_bn_d3(pix.d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x pix.gf_dim*8*2)

        pix.d4, pix.d4_w, pix.d4_b = deconv2d(tf.nn.relu(d3),
            [pix.batch_size, s16, s16, pix.gf_dim*8], name='g_d4', with_w=True)
        d4 = pix.g_bn_d4(pix.d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x pix.gf_dim*8*2)

        pix.d5, pix.d5_w, pix.d5_b = deconv2d(tf.nn.relu(d4),
            [pix.batch_size, s8, s8, pix.gf_dim*4], name='g_d5', with_w=True)
        d5 = pix.g_bn_d5(pix.d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x pix.gf_dim*4*2)

        pix.d6, pix.d6_w, pix.d6_b = deconv2d(tf.nn.relu(d5),
            [pix.batch_size, s4, s4, pix.gf_dim*2], name='g_d6', with_w=True)
        d6 = pix.g_bn_d6(pix.d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x pix.gf_dim*2*2)

        pix.d7, pix.d7_w, pix.d7_b = deconv2d(tf.nn.relu(d6),
            [pix.batch_size, s2, s2, pix.gf_dim], name='g_d7', with_w=True)
        d7 = pix.g_bn_d7(pix.d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x pix.gf_dim*1*2)

        pix.d8, pix.d8_w, pix.d8_b = deconv2d(tf.nn.relu(d7),
            [pix.batch_size, s, s, pix.output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(pix.d8)

def sampler(pix, image, y=None):

    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s = pix.image_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, pix.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x pix.gf_dim)
        e2 = pix.g_bn_e2(conv2d(lrelu(e1), pix.gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x pix.gf_dim*2)
        e3 = pix.g_bn_e3(conv2d(lrelu(e2), pix.gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x pix.gf_dim*4)
        e4 = pix.g_bn_e4(conv2d(lrelu(e3), pix.gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x pix.gf_dim*8)
        e5 = pix.g_bn_e5(conv2d(lrelu(e4), pix.gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x pix.gf_dim*8)
        e6 = pix.g_bn_e6(conv2d(lrelu(e5), pix.gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x pix.gf_dim*8)
        e7 = pix.g_bn_e7(conv2d(lrelu(e6), pix.gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x pix.gf_dim*8)
        e8 = pix.g_bn_e8(conv2d(lrelu(e7), pix.gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x pix.gf_dim*8)

        pix.d1, pix.d1_w, pix.d1_b = deconv2d(tf.nn.relu(e8),
            [pix.batch_size, s128, s128, pix.gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(pix.g_bn_d1(pix.d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x pix.gf_dim*8*2)

        pix.d2, pix.d2_w, pix.d2_b = deconv2d(tf.nn.relu(d1),
            [pix.batch_size, s64, s64, pix.gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(pix.g_bn_d2(pix.d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x pix.gf_dim*8*2)

        pix.d3, pix.d3_w, pix.d3_b = deconv2d(tf.nn.relu(d2),
            [pix.batch_size, s32, s32, pix.gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(pix.g_bn_d3(pix.d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x pix.gf_dim*8*2)

        pix.d4, pix.d4_w, pix.d4_b = deconv2d(tf.nn.relu(d3),
            [pix.batch_size, s16, s16, pix.gf_dim*8], name='g_d4', with_w=True)
        d4 = pix.g_bn_d4(pix.d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x pix.gf_dim*8*2)

        pix.d5, pix.d5_w, pix.d5_b = deconv2d(tf.nn.relu(d4),
            [pix.batch_size, s8, s8, pix.gf_dim*4], name='g_d5', with_w=True)
        d5 = pix.g_bn_d5(pix.d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x pix.gf_dim*4*2)

        pix.d6, pix.d6_w, pix.d6_b = deconv2d(tf.nn.relu(d5),
            [pix.batch_size, s4, s4, pix.gf_dim*2], name='g_d6', with_w=True)
        d6 = pix.g_bn_d6(pix.d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x pix.gf_dim*2*2)

        pix.d7, pix.d7_w, pix.d7_b = deconv2d(tf.nn.relu(d6),
            [pix.batch_size, s2, s2, pix.gf_dim], name='g_d7', with_w=True)
        d7 = pix.g_bn_d7(pix.d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x pix.gf_dim*1*2)

        pix.d8, pix.d8_w, pix.d8_b = deconv2d(tf.nn.relu(d7),
            [pix.batch_size, s, s, pix.output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(pix.d8)

def save(pix, checkpoint_dir, step):
    model_name = "pix2pix.model"
    model_dir = "%s_%s_%s" % (pix.dataset_name, pix.batch_size, pix.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    pix.saver.save(pix.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

def load(pix, checkpoint_dir):
    print(" [*] Reading checkpoint...")

    model_dir = "%s_%s_%s" % (pix.dataset_name, pix.batch_size, pix.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        pix.saver.restore(pix.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def test(pix):
    """Test pix2pix"""
    init_op = tf.global_variables_initializer()
    pix.sess.run(init_op)

    base_dir = './datasets/{0}/test/'.format(pix.dataset_name)
    target_dir = './{}/'.format(pix.test_dir)
    
    sample_files = glob('./datasets/{0}/test/*.png'.format(pix.dataset_name, pix.axis))

    start_time = time.time()
    if load(pix, pix.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for i, sample_file in enumerate(sample_files):
            
        sample_image = load_data(sample_file, pix.image_size, pix.input_c_dim, pix.output_c_dim)
        sample_image = np.array([sample_image])    
        
        print("sampling image ", i)
        samples = pix.sess.run(
            pix.fake_B_sample,
            feed_dict={pix.real_data: sample_image}
        )

        samples = np.sign(samples)
        
        if pix.phase == 'test':
            combined = np.concatenate((sample_image, samples), axis=3)
            arr = np.split(combined, combined.shape[3], axis=3)

            con = np.concatenate(arr, axis=2)
            save_images(con, [pix.batch_size, 1], sample_file.replace(base_dir, target_dir).replace('combined', pix.phase))
        else:
            combined = samples[:, 8:pix.image_size-8, 8:pix.image_size-8, :]
            arr = np.split(combined, combined.shape[3], axis=3)

            # TODO convert data
            con = np.concatenate(arr, axis=2)
            save_images(con, [pix.batch_size, 1], sample_file.replace(base_dir, target_dir).replace('combined', pix.phase))

# Run The Model
def run(pix):
    if not os.path.exists(pix.checkpoint_dir):
        os.makedirs(pix.checkpoint_dir)
    if not os.path.exists(pix.sample_dir):
        os.makedirs(pix.sample_dir)
    if not os.path.exists(pix.test_dir):
        os.makedirs(pix.test_dir)

    with tf.Session() as sess:
        pix.sess = sess
        build_model(pix)
    
        if pix.phase == 'train':
            train(pix)
        else:
            test(pix)
