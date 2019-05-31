import os
import sys
import ast
import pdb
import math
import random
import numpy as np
import datetime
import tensorflow as tf # tf version 1.4
root =  os.path.dirname(os.path.realpath(__file__))
sys.path.append( root )

def CONV2D( lInput_sen, W, name="conv2d" ):
    return tf.nn.conv2d ( input = lInput_sen, filter = W, strides= [1,1,1,1], padding = "VALID", name = name )

def RELU( conv,b , name="relu" ):
    return tf.nn.relu(tf.nn.bias_add(conv, b), name=name)

def SOFTMAX( dense ):
    return tf.nn.softmax( dense )

def MAX_POOL( value, ksize, name='pool' ):
    return tf.nn.max_pool( value, ksize, strides= [1,1,1,1], padding = "VALID", name = name ) 

class CNN_SEPARABLE:  # 1
    def __init__( self, 
                  jaso_sequence_length=20, 
                  syll_sequence_length=20, 
                  word_sequence_length=20, 
                  tags_sequence_length=20, 
                  embedding_size=100, 
                  num_filters=100, 
                  num_classes=2,
                  jaso_vocab_size=40, 
                  syll_vocab_size=5000, 
                  word_vocab_size=50000, 
                  tags_vocab_size=40 ):
        with tf.name_scope("embed"):

            self.m_lInput_j = tf.placeholder( tf.int32, shape=[None, jaso_sequence_length], name="INPUTJ" );  # jaso
            self.m_lInput_s = tf.placeholder( tf.int32, shape=[None, syll_sequence_length], name="INPUTS" );  # syllable
            self.m_lInput_w = tf.placeholder( tf.int32, shape=[None, word_sequence_length], name="INPUTW" );  # word
            self.m_lInput_t = tf.placeholder( tf.int32, shape=[None, tags_sequence_length], name="INPUTT" );  # tag
            self.m_lInput_y = tf.placeholder( tf.int32, shape=[None, num_classes ], name="INPUTY" ); 
            self.m_fDropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            self.m_mInput_J = tf.Variable( tf.random_uniform( [jaso_vocab_size, embedding_size], -1, 1, name="m_mInput_J") );
            self.m_mInput_S = tf.Variable( tf.random_uniform( [syll_vocab_size, embedding_size], -1, 1, name="m_mInput_S") );
            self.m_mInput_W = tf.Variable( tf.random_uniform( [word_vocab_size, embedding_size], -1, 1, name="m_mInput_W") );
            self.m_lInput_T = tf.Variable( tf.random_uniform( [tags_vocab_size, embedding_size], -1, 1, name="m_lInput_T") );

            self.m_mInput_embedding_j = tf.nn.embedding_lookup( self.m_mInput_J, self.m_lInput_j );
            self.m_mInput_embedding_s = tf.nn.embedding_lookup( self.m_mInput_S, self.m_lInput_s );
            self.m_mInput_embedding_w = tf.nn.embedding_lookup( self.m_mInput_W, self.m_lInput_w );
            self.m_mInput_embedding_t = tf.nn.embedding_lookup( self.m_lInput_T, self.m_lInput_t );

            self.m_mExpand_input_embedding_j = tf.expand_dims( self.m_mInput_embedding_j, -1 );
            self.m_mExpand_input_embedding_s = tf.expand_dims( self.m_mInput_embedding_s, -1 );
            self.m_mExpand_input_embedding_w = tf.expand_dims( self.m_mInput_embedding_w, -1 );
            self.m_mExpand_input_embedding_t = tf.expand_dims( self.m_mInput_embedding_t, -1 );


            self.m_lPools_j=[]
            self.m_lPool_j=[]
            self.m_lConv_W_j=[]
            self.m_lConv_b_j=[]
            self.filter_sizes=[2, 3, 4];
            for filter_size in self.filter_sizes:
                with tf.name_scope( 'jaso-conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    self.m_lConv_W_j.append( W );
                    self.m_lConv_b_j.append( b );
                    conv = CONV2D( self.m_mExpand_input_embedding_j, W );
                    conv_h = RELU( conv, b )
                    ksize = [1, jaso_sequence_length - filter_size + 1, 1, 1];
                    self.m_lPool = MAX_POOL( conv_h, ksize )
                    self.m_lPools_j.append( self.m_lPool );
            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool_j = tf.concat( self.m_lPools_j, 3 );
            self.flat_pool_j = tf.reshape( self.concat_pool_j, [-1, self.m_iTotal_filter_size ] );




            self.m_lPool_s=[]
            self.m_lConv_W_s=[]
            self.m_lConv_b_s=[]
            self.filter_sizes=[2, 3, 4];
            for filter_size in self.filter_sizes:
                with tf.name_scope( 'syll-conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    self.m_lConv_W_s.append( W );
                    self.m_lConv_b_s.append( b );
                    conv = CONV2D( self.m_mExpand_input_embedding_s, W );
                    conv_h = RELU( conv, b )
                    ksize = [1, syll_sequence_length - filter_size + 1, 1, 1];
                    pool = MAX_POOL( conv_h, ksize )
                    self.m_lPool_s.append( pool );
            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool_s = tf.concat( self.m_lPool_s, 3 );
            self.flat_pool_s = tf.reshape( self.concat_pool_s, [-1, self.m_iTotal_filter_size ] );


            self.m_lPool_w=[]
            self.m_lConv_W_w=[]
            self.m_lConv_b_w=[]
            for filter_size in self.filter_sizes:
                with tf.name_scope( 'word-conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    self.m_lConv_W_w.append( W );
                    self.m_lConv_b_w.append( b );
                    conv = CONV2D( self.m_mExpand_input_embedding_w, W );
                    conv_h = RELU( conv, b )
                    ksize = [1, word_sequence_length - filter_size + 1, 1, 1];
                    pool = MAX_POOL( conv_h, ksize )
                    self.m_lPool_w.append( pool );


            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool_w = tf.concat( self.m_lPool_w, 3 );
            self.flat_pool_w = tf.reshape( self.concat_pool_w, [-1, self.m_iTotal_filter_size ] );


            self.m_lPool_t=[]
            self.m_lConv_W_t=[]
            self.m_lConv_b_t=[]
            for filter_size in self.filter_sizes:
                with tf.name_scope( 'tags-conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    self.m_lConv_W_t.append( W );
                    self.m_lConv_b_t.append( b );
                    conv = CONV2D( self.m_mExpand_input_embedding_t, W );
                    conv_h = RELU( conv, b )
                    ksize = [1, tags_sequence_length - filter_size + 1, 1, 1];
                    pool = MAX_POOL( conv_h, ksize )
                    self.m_lPool_t.append( pool );


            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool_t = tf.concat( self.m_lPool_t, 3 );
            self.flat_pool_t = tf.reshape( self.concat_pool_t, [-1, self.m_iTotal_filter_size ] );
            

            self.flat_pool_mrg =  tf.concat( [self.flat_pool_j, self.flat_pool_s, self.flat_pool_w, self.flat_pool_t], 1 );
 
            #[1, 80, 300]           
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout( self.flat_pool_mrg, self.m_fDropout_keep_prob )

            with tf.name_scope("output"):
                l2_loss = tf.constant(0.0)
                self.fullW = tf.get_variable( "fullW",
                    shape=[self.m_iTotal_filter_size*4, num_classes], 
                    initializer=tf.contrib.layers.xavier_initializer())
                self.fullb = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(self.fullW)
                l2_loss += tf.nn.l2_loss(self.fullb)
                self.scores = tf.nn.xw_plus_b(self.h_drop, self.fullW, self.fullb, name="scores")
                self.predictions = tf.argmax( self.scores, 1, name="predictions")
          
          
          
            # Calculate mean cross-entropy lossa
            l2_reg_lambda=0.1
            with tf.name_scope("loss"):
                self.m_fLosses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.m_lInput_y)
                self.m_fLoss = tf.reduce_mean(self.m_fLosses) + l2_reg_lambda * l2_loss
          
            # Accuracy
            #with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.m_lInput_y, 1))
            cast_accuracy = tf.cast(correct_predictions,  tf.float32 );
            self.accuracy = tf.reduce_mean( cast_accuracy , name="accuracy")

class CNN_UNIFIED: #2
    def __init__( self, 
                  jaso_sequence_length=20, 
                  syll_sequence_length=20, 
                  word_sequence_length=20, 
                  tags_sequence_length=20, 
                  embedding_size=100, 
                  num_filters=100, 
                  num_classes=2,
                  jaso_vocab_size=40, 
                  syll_vocab_size=5000, 
                  word_vocab_size=50000, 
                  tags_vocab_size=40 ):
        with tf.name_scope("embed"):

            self.m_lInput_j = tf.placeholder( tf.int32, shape=[None, jaso_sequence_length], name="INPUTJ" );  # jaso
            self.m_lInput_s = tf.placeholder( tf.int32, shape=[None, syll_sequence_length], name="INPUTS" );  # syllable
            self.m_lInput_w = tf.placeholder( tf.int32, shape=[None, word_sequence_length], name="INPUTW" );  # word
            self.m_lInput_t = tf.placeholder( tf.int32, shape=[None, tags_sequence_length], name="INPUTT" );  # tag
            self.m_lInput_y = tf.placeholder( tf.int32, shape=[None, num_classes ], name="INPUTY" ); 
            self.m_fDropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            self.m_mInput_J = tf.Variable( tf.random_uniform( [jaso_vocab_size, embedding_size], -1, 1, name="m_mInput_J") );
            self.m_mInput_S = tf.Variable( tf.random_uniform( [syll_vocab_size, embedding_size], -1, 1, name="m_mInput_S") );
            self.m_mInput_W = tf.Variable( tf.random_uniform( [word_vocab_size, embedding_size], -1, 1, name="m_mInput_W") );
            self.m_lInput_T = tf.Variable( tf.random_uniform( [tags_vocab_size, embedding_size], -1, 1, name="m_lInput_T") );

            self.m_mInput_embedding_j = tf.nn.embedding_lookup( self.m_mInput_J, self.m_lInput_j );
            self.m_mInput_embedding_s = tf.nn.embedding_lookup( self.m_mInput_S, self.m_lInput_s );
            self.m_mInput_embedding_w = tf.nn.embedding_lookup( self.m_mInput_W, self.m_lInput_w );
            self.m_mInput_embedding_t = tf.nn.embedding_lookup( self.m_lInput_T, self.m_lInput_t );


            self.input_embedding = tf.concat( [ self.m_mInput_embedding_j, 
                                                self.m_mInput_embedding_s,
                                                self.m_mInput_embedding_w, 
                                                self.m_mInput_embedding_t], axis=-1 )
            self.m_mExpand_input_embedding = tf.expand_dims( self.input_embedding, -1 );


            self.m_lPools=[]
            self.filter_sizes=[2, 3, 4];
            self.m_lPool = []
            self.m_lConv = []
            self.m_lConv_h = []
            self.m_lConvW = []
            self.m_lConvb = []
            for idx, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope( 'conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size*4, 1, num_filters]
                    self.m_lConvW.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="convW"))
                    self.m_lConvb.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="convb"))
                    self.m_lConv.append( CONV2D( self.m_mExpand_input_embedding, self.m_lConvW[idx] ))
                    self.m_lConv_h.append(RELU( self.m_lConv[idx], self.m_lConvb[idx] ))
                    ksize = [1, jaso_sequence_length - filter_size + 1, 1, 1];
                    self.m_lPool.append(MAX_POOL( self.m_lConv_h[idx], ksize ))
                    self.m_lPools.append( self.m_lPool[idx] );


            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool = tf.concat( self.m_lPools, 3 );
            self.flat_pool = tf.reshape( self.concat_pool, [-1, self.m_iTotal_filter_size ] );
            
            #[1, 80, 300]           
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout( self.flat_pool, self.m_fDropout_keep_prob )

            with tf.name_scope("output"):
                l2_loss = tf.constant(0.0)
                self.fullW = tf.get_variable( "fullW",
                    shape=[self.m_iTotal_filter_size, num_classes], 
                    initializer=tf.contrib.layers.xavier_initializer())
                self.fullb = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(self.fullW)
                l2_loss += tf.nn.l2_loss(self.fullb)
                self.scores = tf.nn.xw_plus_b(self.h_drop, self.fullW, self.fullb, name="scores")
                self.predictions = tf.argmax( self.scores, 1, name="predictions")
          
          
          
            # Calculate mean cross-entropy lossa
            l2_reg_lambda=0.1
            with tf.name_scope("loss"):
                self.m_fLosses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.m_lInput_y)
                self.m_fLoss = tf.reduce_mean(self.m_fLosses) + l2_reg_lambda * l2_loss
          
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.m_lInput_y, 1))
                cast_accuracy = tf.cast(correct_predictions,  tf.float32 );
                self.accuracy = tf.reduce_mean( cast_accuracy , name="accuracy")



class CNN_UNIFIED_BI: # 3
    def __init__( self, 
                  jaso_sequence_length=20, 
                  syll_sequence_length=20, 
                  word_sequence_length=20, 
                  tags_sequence_length=20, 
                  embedding_size=100, 
                  num_filters=100, 
                  num_classes=2,
                  jaso_vocab_size=40, 
                  syll_vocab_size=5000, 
                  word_vocab_size=50000, 
                  tags_vocab_size=40 ):
        with tf.name_scope("embed"):

            self.m_lInput_j = tf.placeholder( tf.int32, shape=[None, jaso_sequence_length], name="INPUTJ" );  # jaso
            self.m_lInput_s = tf.placeholder( tf.int32, shape=[None, syll_sequence_length], name="INPUTS" );  # syllable
            self.m_lInput_w = tf.placeholder( tf.int32, shape=[None, word_sequence_length], name="INPUTW" );  # word
            self.m_lInput_t = tf.placeholder( tf.int32, shape=[None, tags_sequence_length], name="INPUTT" );  # tag
            self.m_lInput_y = tf.placeholder( tf.int32, shape=[None, num_classes ], name="INPUTY" ); 
            self.m_fDropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            self.m_mInput_J = tf.Variable( tf.random_uniform( [jaso_vocab_size, embedding_size], -1, 1, name="m_mInput_J") );
            self.m_mInput_S = tf.Variable( tf.random_uniform( [syll_vocab_size, embedding_size], -1, 1, name="m_mInput_S") );
            self.m_mInput_W = tf.Variable( tf.random_uniform( [word_vocab_size, embedding_size], -1, 1, name="m_mInput_W") );
            self.m_lInput_T = tf.Variable( tf.random_uniform( [tags_vocab_size, embedding_size], -1, 1, name="m_lInput_T") );

            self.m_mInput_embedding_j = tf.nn.embedding_lookup( self.m_mInput_J, self.m_lInput_j );
            self.m_mInput_embedding_s = tf.nn.embedding_lookup( self.m_mInput_S, self.m_lInput_s );
            self.m_mInput_embedding_w = tf.nn.embedding_lookup( self.m_mInput_W, self.m_lInput_w );
            self.m_mInput_embedding_t = tf.nn.embedding_lookup( self.m_lInput_T, self.m_lInput_t );


            self.m_mInput_embedding_js = tf.concat( [ self.m_mInput_embedding_j, 
                                                      self.m_mInput_embedding_s], axis=-1 )

            self.m_mInput_embedding_wt = tf.concat( [ self.m_mInput_embedding_w, 
                                                      self.m_mInput_embedding_t], axis=-1 )
            self.m_mExpand_input_embedding_js = tf.expand_dims( self.m_mInput_embedding_js, -1 );
            self.m_mExpand_input_embedding_wt = tf.expand_dims( self.m_mInput_embedding_wt, -1 );


            self.filter_sizes=[2, 3, 4];
            self.m_lPools_js=[]
            self.m_lPool_js = []
            self.m_lConv_js = []
            self.m_lConv_h = []
            self.m_lConvW_js = []
            self.m_lConvb_js = []
            for idx, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope( 'js_conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size*2, 1, num_filters]
                    self.m_lConvW_js.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="js_convW"))
                    self.m_lConvb_js.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="js_convb"))
                    self.m_lConv_js.append( CONV2D( self.m_mExpand_input_embedding_js, self.m_lConvW_js[idx] ))
                    self.m_lConv_h.append(RELU( self.m_lConv_js[idx], self.m_lConvb_js[idx] ))
                    ksize = [1, jaso_sequence_length - filter_size + 1, 1, 1];
                    self.m_lPool_js.append(MAX_POOL( self.m_lConv_h[idx], ksize ))
                    self.m_lPools_js.append( self.m_lPool_js[idx] );


            self.m_lPools_wt=[]
            self.m_lPool_wt = []
            self.m_lConv_wt = []
            self.m_lConv_h = []
            self.convW_wt = []
            self.convb_wt = []
            for idx, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope( 'wt_conv-pool%d' % filter_size ):
                    filter_shape = [filter_size, embedding_size*2, 1, num_filters]
                    self.convW_wt.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="wt_convW"))
                    self.convb_wt.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="wt_convb"))
                    self.m_lConv_wt.append( CONV2D( self.m_mExpand_input_embedding_wt, self.convW_wt[idx] ))
                    self.m_lConv_h.append(RELU( self.m_lConv_wt[idx], self.convb_wt[idx] ))
                    ksize = [1, word_sequence_length - filter_size + 1, 1, 1];
                    self.m_lPool_wt.append(MAX_POOL( self.m_lConv_h[idx], ksize ))
                    self.m_lPools_wt.append( self.m_lPool_wt[idx] );


            self.m_iTotal_filter_size = len(self.filter_sizes)*num_filters
            self.concat_pool_js = tf.concat( self.m_lPool_js, -1 );
            self.flat_pool_js = tf.reshape( self.concat_pool_js, [-1, self.m_iTotal_filter_size ] );

            self.concat_pool_wt = tf.concat( self.m_lPools_wt, -1 );
            self.flat_pool_wt = tf.reshape( self.concat_pool_wt, [-1, self.m_iTotal_filter_size ] );
            
            self.flat_pool = tf.concat( [self.flat_pool_js, self.flat_pool_wt], 1 );
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout( self.flat_pool, self.m_fDropout_keep_prob )

            with tf.name_scope("output"):
                l2_loss = tf.constant(0.0)
                self.fullW = tf.get_variable( "fullW",
                    shape=[self.m_iTotal_filter_size*2, num_classes], 
                    initializer=tf.contrib.layers.xavier_initializer())
                self.fullb = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(self.fullW)
                l2_loss += tf.nn.l2_loss(self.fullb)
                self.scores = tf.nn.xw_plus_b(self.h_drop, self.fullW, self.fullb, name="scores")
                self.predictions = tf.argmax( self.scores, 1, name="predictions")
          
          
          
            # Calculate mean cross-entropy lossa
            l2_reg_lambda=0.1
            with tf.name_scope("loss"):
                self.m_fLosses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.m_lInput_y)
                self.m_fLoss = tf.reduce_mean(self.m_fLosses) + l2_reg_lambda * l2_loss
          
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.m_lInput_y, 1))
                cast_accuracy = tf.cast(correct_predictions,  tf.float32 );
                self.accuracy = tf.reduce_mean( cast_accuracy , name="accuracy")






