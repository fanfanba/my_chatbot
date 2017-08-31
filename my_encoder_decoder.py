# -*- coding: utf-8 -*-
# 自动编解码器实现自动问答
import sys
import os
import jieba
import struct
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from distutils.version import LooseVersion
from seq2seqmodel import Seq2SeqModel
import config
import ujson
from tqdm import tqdm
import datetime
import copy
from six.moves import xrange
# Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.2'), 'Please use TensorFlow version 1.2 or newer'
#print('TensorFlow Version: {}'.format(tf.__version__))
def save_object_to_file(file_name,data):
    print('save .. ' + file_name)
    fp = open(config.DATA_DIR+file_name+".json","w")
    ujson.dump(data,fp)
    fp.close()
    print("save data done")
def trans_train_word_2_index(file_name, MAX_SAMPLE_NUM):
    print("read train data start...")
    #word_index_dict = read_object_from_file("word_index")
    print('load .. ' + file_name)
    flag = 10
    fp_read = open(file_name, 'r',encoding='utf-8')#,encoding='utf-8'
    total_cnt = 0
    qa_list = []
    qa_id_list = []
    vocabulary = {}
    word_index_dict = {}
    index_word_dict = {}
    while 1:
        line = fp_read.readline()
        if not line:
            break
        line = line#str(line).decode("utf-8")#line#
        #str(line, encoding = "utf-8") #
        if (total_cnt*10) % MAX_SAMPLE_NUM == 0:
            print("current dis:" + str(float(total_cnt*100) / MAX_SAMPLE_NUM) + "%")
        split_line = line.strip('\n').strip().split("|")
        if flag > 0:
            print("line:")
            print(line)
            print("line end")
            
            flag -= 1
        q_len = len(split_line)
        if q_len < 2: 
            print("q_len<2")
            print(line)
            continue
        if q_len == 2:  # each line include 
            question = jieba.cut(split_line[0])
            answer = jieba.cut(split_line[1])
            for word in question:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
            for word in answer:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
            qa_list.append((question,answer))
        if total_cnt > MAX_SAMPLE_NUM: #read only max_num row
            break
        # if flag > 0:
            # print("question_id_list:")
            # print(question_id_list)
            # print("answer_id_list:")
            # print(answer_id_list)
            # flag -= 1
        total_cnt += 1
        
    fp_read.close()
    word_index_dict[config.special_words[0]] = 0
    word_index_dict[config.special_words[1]] = 1
    word_index_dict[config.special_words[2]] = 2
    word_index_dict[config.special_words[3]] = 3
    index_word_dict[0] = config.special_words[0]
    index_word_dict[1] = config.special_words[1]
    index_word_dict[2] = config.special_words[2]
    index_word_dict[3] = config.special_words[3]
    index = 4
    print("vocabulary size:"+str(len(vocabulary)))
    for word,num in vocabulary.items():
        if num >= config.vocabulary_sw:
            word_index_dict[word] = index
            index_word_dict[index] = word
            index += 1
    j=ujson.dumps(vocabulary)
    print("voca")
    print(vocabulary)
    temp_dict = j.encode("utf-8").decode("unicode-escape")
    print("temp_dict")
    print(temp_dict)
    print(oook)
    save_object_to_file(config.index_word_file,index_word_dict)
    save_object_to_file(config.word_index_file,word_index_dict)
    for question,answer in qa_list:
        question_id_list = [word_index_dict.get(word, word_index_dict[config.special_words[config.UNK_ID]]) for word in question]
        answer_id_list = [word_index_dict.get(word, word_index_dict[config.special_words[config.UNK_ID]]) for word in answer]
        answer_id_list.append(config.EOS_ID)
        qa_id_list.append((question_id_list,answer_id_list))
    print(qa_id_list[:2])
    print(qa_list[:2])
    print("total_cnt:"+str(total_cnt))
    return qa_id_list
    
class batch_gen():
    def __init__(self, data_list, batch_size = 32):
        #data_list.sort()
        self.batch_size = batch_size
        num_buckets = int(len(data_list)/batch_size)
        self.num_buckets = num_buckets
        self.data_x = []
        self.data_y = []
        self.len_x = []
        self.len_y = []
        self.batch_list = [i for i in range(num_buckets)]
        for bucket in range(num_buckets):
            batch_data = data_list[bucket*self.batch_size: (bucket+1)*self.batch_size]
            x_raw = []
            y_raw = []
            x_len_raw = []
            y_len_raw = []
            for data in batch_data:
                x_raw.append(data[0])
                y_raw.append(data[1])
                x_len_raw.append(len(data[0]))
                y_len_raw.append(len(data[1]))
            self.data_x.append(x_raw)
            self.data_y.append(y_raw) #need fix
            self.len_x.append(x_len_raw)#need fix
            self.len_y.append(y_len_raw)
        # cursor will be the cursor for the ith bucket
        self.cursor = 0
        self.shuffle()
        self.epochs = 0
  
    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        #self.data[i] = self.data[i].sample(frac=1).reset_index(drop=True)
        self.cursor = 0
        np.random.shuffle(self.batch_list)#index shuffle need fix
        #print("batch_list:")
        #print(self.batch_list)

    def next_batch(self):
        if self.cursor >= self.num_buckets:
            self.epochs += 1
            self.shuffle()
        
        cur_bucket = self.batch_list[self.cursor]
        x_raw = self.data_x[cur_bucket]
        y_raw = self.data_y[cur_bucket]
        self.cursor += 1
        batch_x_len = []
        batch_y_len = []
        batch_x_len.extend(self.len_x[cur_bucket])
        batch_y_len.extend(self.len_y[cur_bucket])
        # pad sequences with 0s so they are all the same length
        max_len_x = max(batch_x_len)
        max_len_y = max(batch_y_len)
        #print("max_len:"+str(max_len))
        if max_len_x > config.MAX_SEQUENCE:
            max_len_x = config.MAX_SEQUENCE
        if max_len_y > config.MAX_SEQUENCE:
            max_len_y = config.MAX_SEQUENCE
        if max_len_x < config.MAX_SEQUENCE:
            max_len_x = config.MAX_SEQUENCE
        if max_len_y < config.MAX_SEQUENCE:
            max_len_y = config.MAX_SEQUENCE
        #print("max_len:"+str(max_len))
        x = np.zeros([self.batch_size, max_len_x], dtype=np.int32)
        for i, x_i in enumerate(x):
            # print("i:"+str(i))
            # print(batch_x_len[i])
            # print(x_raw[i])
            if batch_x_len[i] > max_len_x:
                len_temp = max_len_x
                batch_x_len[i] = max_len_x
            else:
                len_temp = batch_x_len[i]
            x_i[:len_temp] = x_raw[i][:len_temp]
        y = np.zeros([self.batch_size, max_len_y], dtype=np.int32)
        for i, y_i in enumerate(y):
            if batch_y_len[i] > max_len_y:
                len_temp = max_len_y
                batch_y_len[i] = max_len_y
            else:
                len_temp = batch_y_len[i]
                batch_y_len[i] = max_len_y
            y_i[:len_temp] = y_raw[i][:len_temp]

        return x, y, batch_x_len,batch_y_len,sum(batch_x_len),sum(batch_y_len)
def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ")
    sys.stdout.flush()
    return sys.stdin.readline()
    
def read_object_from_file(file_name):
    print('load .. '+file_name)
    fp = open(config.DATA_DIR+file_name+".json","rb")
    data =  ujson.load(fp)
    fp.close()
    print("load data done")
    return data
    
def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.DATA_DIR+config.MODEL_DIR + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot from:"+ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")
        
class MyEncoderDecoder(object):
    def __init__(self,train=True):
        self.max_abs_weight = 32  # 最大权重绝对值，用来对词向量做正规化
        self.max_seq_len = 8  # 最大句子长度(词)
        self.epoch = config.EPOCH
        self.word_embedding_vectors = {} 
        self.word_index_dict = {}
        self.index_word_dict = {}
        self.model_dir = config.MODEL_DIR  # 模型文件路径
        self.model_file = config.MODEL_FILE
        self.train_data_file = config.DATA_DIR+config.corpus_file
        self.rnn_dim = config.rnn_dim  # lstm隐藏状态单元数目
        self.layer_num = config.layer_num
        self.encoder_vocab_size=10000 # 词个数，读word_vector时动态确定
        self.decoder_vocab_size=10000 # 词个数，读word_vector时动态确定
        self.embedding_dim=200 # 词向量维度，读word_vector时动态确定
        self.learning_rate=0.001
        self.get_word_dict(train)
        
    def get_word_dict(self,train):
        if train == True:
            self.train_set = trans_train_word_2_index(self.train_data_file,config.MAX_TRAIN_SAMPLE_NUM)
        print("get word index dict")
        self.word_index_dict = read_object_from_file(config.word_index_file)
        self.index_word_dict = read_object_from_file(config.index_word_file)
        print("self.word_index_dict[config.special_words[config.UNK_ID]]")
        print(self.word_index_dict[config.special_words[config.UNK_ID]])
        print("self.index_word_dict[str(config.UNK_ID)]") 
        print(self.index_word_dict[str(config.UNK_ID)])
        self.encoder_vocab_size = len(self.word_index_dict)
        self.decoder_vocab_size = len(self.word_index_dict)

        print("encoder_vocab_size:"+str(self.encoder_vocab_size))
        print("decoder_vocab_size:"+str(self.decoder_vocab_size))
        print("embedding_dim:"+str(self.embedding_dim))
        #index_dict  # index dict,{word:index}
        #word_vectors  # word embedding, { word:vector}
        print("Setting up Arrays for tensorflow Embedding Layer...")

    def create_model(self,is_training=True,batch_size=1):
        #rnn_dim, layer_num, encoder_vocab_size,decoder_vocab_size, embedding_dim, training=True,lr=1e-3
        model = Seq2SeqModel(self.rnn_dim, self.layer_num, self.encoder_vocab_size, 
        self.decoder_vocab_size, self.embedding_dim, training=is_training,lr=self.learning_rate,batch_size=batch_size,word_embedding_vectors=self.word_embedding_vectors)
        return model
        
    def sentence2idlist(self,line):
        segments = jieba.cut(line)
        words=[word for word in segments]
        word_id_list = [self.word_index_dict.get(word, self.word_index_dict[config.special_words[config.UNK_ID]]) for word in words]
        print("word_id_list:",word_id_list)
        print("cut line:")
        for i in range(len(words)):
            print(words[i]," "),
        print(" ")
        return word_id_list

    # def pad_sentence_batch(sentence_batch, pad_int):
        # """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        # max_sentence = max([len(sentence) for sentence in sentence_batch])
        # return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    def pad_sentence(self,sentence,max_sentence):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        #max_sentence = max([len(sentence) for sentence in sentence_batch])
        return sentence + [config.PAD_ID] * (max_sentence - len(sentence)) 
        
    def _construct_response(self,output_logits):
        """ Construct a response to the user's encoder input.
        @output_logits: the outputs from sequence to sequence wrapper.
        output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
        This is a greedy decoder - outputs are just argmaxes of output_logits.
        """
        outputs = [output for output in output_logits]
        print("outputs")
        print(outputs)
        return " ".join([tf.compat.as_str(self.index_word_dict[str(output)]) for output in outputs]) 
        
    def predict(self):
        batch_size = 2
        model = self.create_model(is_training=False, batch_size=batch_size)
        model.build_graph()
        logits = model.inference_logits
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            #saver.restore(sess, self.model_dir)
            _check_restore_parameters(sess,saver)
            output_file = open('output_convo.txt', 'a+')

            while True:
                line = _get_user_input()
                if len(line) > 0 and line[-1] == '\n':
                    line = line[:-1]
                if line == 'q':
                    break
                if line == '':
                    continue
                output_file.write('HUMAN ++++ ' + line + '\n')
                # Get token-ids for the input sentence.
                word_id_list_raw = self.sentence2idlist(line)
                word_id_list = self.pad_sentence(word_id_list_raw,config.MAX_SEQUENCE)
                print("pad_word_id_list:",word_id_list)
                word_id_array = np.array(word_id_list)
                #print("len(word_id_array)")
                #print(len(word_id_array))
                # Get output logits for the sentence.
                output_logits = sess.run(logits, feed_dict={model.input_x:[word_id_array]*batch_size,model.target_sequence_length:[len(word_id_array)]*batch_size,model.source_sequence_length: [len(word_id_array)]*batch_size})
                response = self._construct_response(output_logits[0])
                print(response)
                output_file.write('BOT ++++ ' + response + '\n')
            output_file.write('=============================================\n')
            output_file.close()

    def train(self):
        print("batch_size:"+str(config.batch_size))
        print("MAX_TRAIN_SAMPLE_NUM:"+str(config.MAX_TRAIN_SAMPLE_NUM))
        print("train data file:"+str(self.train_data_file))
        model = self.create_model(is_training=True,batch_size=config.batch_size)
        model.build_graph()
        train_oop = model.train_op
        loss = model.loss
        
       
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            train_set = self.train_set
            train_batch_gen = batch_gen(train_set, config.batch_size)
            #test_batch_gen = batch_gen(test_set, batch_size)
            train_set_len = len(train_set)
            #test_set_len = len(test_set)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            _check_restore_parameters(sess,saver)
            total_steps = 0
            print("train_set_len:"+str(train_set_len))
            for i in range(config.EPOCH):
                print("total epoch: {}\t".format(config.EPOCH))
                print("current epoch:"+str(i))
                print(datetime.datetime.now())
                # Training
                num_batches = int(train_set_len / config.batch_size)
                print("train num_batches:"+str(num_batches))
                for step in tqdm(xrange(num_batches)): #for train_idx in tqdm(xrange(num_batches)):
                    total_steps = total_steps + 1
                    #get one batch data set
                    input_x_train,target_ids_train,x_len,y_len,x_max_len,y_max_len = train_batch_gen.next_batch()

                    loss_out,_ = sess.run([loss,train_oop], feed_dict={model.input_x:input_x_train,model.target_ids: target_ids_train,model.target_sequence_length: y_len,model.source_sequence_length: x_len})
                    if i % 1 == 0 and step == 0:
                        print('i=%d, loss=%f' % (i, loss_out))
                
                path = saver.save(sess, os.path.join(self.model_dir, self.model_file), global_step=i)
                print("Saved model checkpoint to {}".format(path))
        
def main(op):
    np.set_printoptions(threshold='nan')
    
    if op == 'train':
        my_qa = MyEncoderDecoder(train=True)
        my_qa.train()
    elif op == 'predict':
        my_qa = MyEncoderDecoder(train=False)
        my_qa.predict()
    else:
        print('Usage:')

if __name__ == '__main__':
    print(datetime.datetime.now())
    print("my encoder decoder qa start ...")
    
    #train_set = trans_train_word_2_index("../data/corpus_pair.txt",100)
    
    
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Usage:')
    
    print(datetime.datetime.now())
    print("my encoder decoder end!")
 