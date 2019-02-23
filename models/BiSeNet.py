# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import numpy as np
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

# ARM 模块， 学习输入中每个特征的权重，然后将每个权重加权
def AttentionRefinementModule(inputs, n_filters):

    # Global average pooling
    # 压缩的是宽和高
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    net = tf.sigmoid(slim.batch_norm(slim.conv2d(net, n_filters, kernel_size=[1, 1]), fused=True))

    return tf.multiply(inputs, net)

# FRM模块
def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling,
    # 上面部分
    net = tf.nn.relu(slim.conv2d(tf.reduce_mean(inputs, [1, 2], keep_dims=True), n_filters, kernel_size=[1, 1]))
    net = tf.sigmoid(slim.conv2d(net, n_filters, kernel_size=[1, 1]))
    # 第一次融合，给原来输入的特征权重
    net = tf.multiply(inputs, net)
    # 加权后的特征和原来特征相加，得到更显著的结果。
    return tf.add(inputs, net)


def build_bisenet(inputs, num_classes, frontend="ResNet101", is_training=True, pretrained_dir="models"):
    """
    Builds the BiSeNet model. 

    Arguments:
      inputs: The input tensor=
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      BiSeNet model
    """

    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)


    ### Context path
    # 是ResNet结构，提取某些层来构成Context path，其中init_fn为加载与训练模型来初始化
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_fronted(inputs, frontend,
                                                                                  pretrained_dir=pretrained_dir,
                                                                                  is_training=is_training)
    # 第一个ARM，输入大小为原来大小的 16x
    net_4 = AttentionRefinementModule(end_points['pool4'], n_filters=512)
    # 第二个 ARM 输入为原来大小的 32x
    net_5 = AttentionRefinementModule(end_points['pool5'], n_filters=2048)
    # 原论文中，采用原来图像感受野来加权提取的特征
    global_channels = tf.reduce_mean(end_points['pool5'], [1, 2], keep_dims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)

    ### Combining the paths
    # spatial path得到的结果是 8x，net_4的到的 16x，net_5得到 32x，所以这里需要上采样
    context_net = tf.concat([Upsampling(net_4, scale=2), Upsampling(net_5_scaled, scale=4)], axis=-1)
    # FRM模块融合特征
    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=num_classes)


    ### Final upscaling and finish
    net = Upsampling(net, scale=8)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn

