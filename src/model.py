import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse

from backbones import modifiedResNet_v2, ResNet_v2



def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--image_size', default=[112, 112, 3], help='the image size')
    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='./model/best-m-1006000', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')
    args = parser.parse_args()
    return args

def get_embd(inputs, is_training_dropout, is_training_bn, config, reuse=False, scope='embd_extractor'):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_101':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_152':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_200':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_200(net, is_training=is_training_bn, return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')
        elif config['backbone_type'].startswith('resnet_v2'):
            arg_sc = ResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_50':
                    net, end_points = ResNet_v2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = ResNet_v2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = ResNet_v2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = ResNet_v2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if config['out_type'] == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=is_training_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')
        
        return net, end_points


        
        