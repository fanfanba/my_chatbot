# -*- coding: utf-8 -*-
# 自动编解码器实现自动问答
import sys
import jieba
import struct
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
import config
from six.moves import xrange
#Add two entries to the word vector matrix. One to represent padding tokens, 
#and one to represent an end of sentence token
#padVector = np.zeros((1, wordVecDimensions), dtype='int32')
#EOSVector = np.ones((1, wordVecDimensions), dtype='int32')
#wordVectors = np.concatenate((wordVectors,padVector), axis=0)
#wordVectors = np.concatenate((wordVectors,EOSVector), axis=0)

class Seq2SeqModel(object):

    def __init__(self, rnn_dim, layer_num, encoder_vocab_size, 
        decoder_vocab_size, embedding_dim, training=True, lr=1e-3, batch_size=1,word_embedding_vectors=None):
        #define inputs question sequence
        self.rnn_dim = rnn_dim
        self.layer_num = layer_num
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.word_embedding_vectors = word_embedding_vectors
        #define inputs answer seqence

    def _create_placeholders(self):
        print("creat placeholders")
        self.input_x = tf.placeholder(tf.int32, shape=[self.batch_size, config.MAX_SEQUENCE], name='input_ids')
        self.target_ids = tf.placeholder(tf.int32, shape=[self.batch_size, config.MAX_SEQUENCE], name='target_ids')
        self.target_sequence_length = tf.placeholder(tf.int32, (self.batch_size,), name='target_sequence_length')
        self.max_target_sequence_length = config.MAX_SEQUENCE#tf.reduce_max(self.target_sequence_length, name='max_target_len')
        self.source_sequence_length = tf.placeholder(tf.int32, (self.batch_size,), name='source_sequence_length')
        print("create word embedding matrix")
        print("encoder_vocab_size:"+str(self.encoder_vocab_size))
        print("embedding_dim:"+str(self.embedding_dim))
        # if self.word_embedding_vectors != None:
            # self.encoder_embedding = tf.get_variable(name="encoder_embedding", shape=[self.encoder_vocab_size, self.embedding_dim], initializer=tf.constant_initializer(self.word_embedding_vectors), trainable=True)
        # else:
        self.encoder_embedding = tf.Variable(tf.random_uniform([self.encoder_vocab_size, self.embedding_dim]))
        self.decoder_embedding = self.encoder_embedding
        
    def _encoding_layer(self):
        print("create encoder layer")
        print("rnn_dim:"+str(self.rnn_dim))
        print("layer_num:"+str(self.layer_num))
        #Encoder embedding
        #define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(self.rnn_dim, self.layer_num)

        input_x_embedded = tf.nn.embedding_lookup(self.encoder_embedding, self.input_x)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, sequence_length=self.source_sequence_length, dtype=tf.float32)
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        
    def _process_decoder_input(self):
        """Remove the last word id from each batch and concat the <GO> to the begining of each batch"""
        ending = tf.strided_slice(self.target_ids, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], config.START_ID), ending], 1)
        self.decoder_input = dec_input
        
    def _decoding_layer(self):
        # 1. Decoder Embedding
        print("create decoder layer")
        target_embedded_input = tf.nn.embedding_lookup(self.decoder_embedding, self.decoder_input)
        # 2. Construct the decoder cell
        decoder_cell = self._get_simple_lstm(self.rnn_dim, self.layer_num)
         
        # 3. Dense layer to translate the decoder's output at each time 
        # step into a choice from the target vocabulary
        fc_layer = Dense(self.decoder_vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


        # 4. Set up a training decoder and an inference decoder
        # Training Decoder
        with tf.variable_scope("decode"):
            # Helper for the training process. Used by BasicDecoder to read inputs.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedded_input, sequence_length=self.target_sequence_length)
            # Basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                               training_helper,
                                                               self.encoder_state,
                                                               fc_layer) 
            
            # Perform dynamic decoding using the decoder
            training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=False,maximum_iterations=self.max_target_sequence_length)[0]
            #training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder)[0]
            #seq2seq.dynamic_decode()
        '''
        print("target_embedded_input")
        print(target_embedded_input)
        print("self.target_sequence_length")
        print(self.target_sequence_length)
        print("self.encoder_state")
        print(self.encoder_state)
        print("self.encoder_outputs")
        print(self.encoder_outputs)
        print("training_decoder_output")
        print(training_decoder_output)
        '''
        print("self.encoder_state")
        print(self.encoder_state)
        print("self.encoder_outputs")
        print(self.encoder_outputs)
        # 5. Inference Decoder 
        # Reuses the same parameters trained by the training process
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([config.START_ID], dtype=tf.int32), [self.batch_size], name='start_tokens')

            # Helper for the inference process.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.decoder_embedding,start_tokens=tf.fill([self.batch_size], config.START_ID),end_token=config.EOS_ID)

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                            inference_helper,
                                                            self.encoder_state,
                                                            fc_layer)
            # Perform dynamic decoding using the decoder
            #inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,maximum_iterations=config.MAX_SEQUENCE)[0]
            inference_decoder_output = seq2seq.dynamic_decode(inference_decoder, impute_finished=False, maximum_iterations=self.max_target_sequence_length)[0]
        self.training_decoder_output = training_decoder_output
        #self.inference_decoder_output = inference_decoder_output
        self.inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
        # tf.nn.softmax(inference_decoder_output)
        '''
        with tf.variable_scope('decoder_test'):
            fc_layer = Dense(self.decoder_vocab_size)
            decoder_cell = self._get_simple_lstm(self.rnn_dim, self.layer_num)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)
        '''
    def _create_loss(self):
        print("create loss and optimizer")
        # Create the weights for sequence_loss
        #train_weights = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        #train_weights=np.ones(shape=[batch_size,sequence_length],dtype=np.float32)
        self.logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
        
        # Create the weights for sequence_loss
        train_weights = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='train_weights')

        #logits=tf.stack(logits,axis=0)
        #print("logits:")
        #print(logits)
        '''
        targets = tf.reshape(self.target_ids, [-1])
        logits_flat = tf.reshape(self.training_decoder_output.rnn_output, [-1, self.decoder_vocab_size])
        print('shape logits_flat:{}'.format(logits_flat.shape))
        print('shape logits:{}'.format(self.training_decoder_output.rnn_output.shape))
        self.loss = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
        '''
        
        self.loss = self.get_loss(self.logits,self.target_ids,train_weights)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        '''
        # Gradient Clipping
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)
        '''
        self.train_op = optimizer.minimize(self.loss)
    def get_loss(self,logits,targets,weights):
        loss = seq2seq.sequence_loss(logits,targets,weights)
        #seq2seq.sequence_loss tf.contrib.
        return loss
        
    def _get_simple_lstm(self, rnn_dim, layer_num):
        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_dim) for _ in xrange(layer_num)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)
    def _create_summary(self):
        pass
    def build_graph(self):
        self._create_placeholders()
        self._encoding_layer()
        self._process_decoder_input()
        self._decoding_layer()
        self._create_loss()
        
        self._create_summary()
