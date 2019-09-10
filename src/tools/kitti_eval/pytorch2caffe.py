import os
import imp
import torch
from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter
from mmdnn.conversion.caffe.saver import save_model


def rename_input(lines, input_size):
    applend_lines = [
        'input: "input" \n',
        'input_shape { \n',
        ' dim: 1\n',
        ' dim: {} \n'.format(input_size[0]),
        ' dim: {} \n'.format(input_size[1]),
        ' dim: {} \n'.format(input_size[2]),
        '} \n'
    ]
    stack = []
    stack.append('{')
    i = 0
    while len(stack) != 0:
        i += 1
        for s in lines[i]:
            if '{' in s:
                stack.append('{')
            if '}' in s:
                stack.pop()
    new_lines = applend_lines + lines[i+1:]
    return new_lines
    



class Pytorch2Caffe:
    def __init__(self, model, save_root, save_name, input_shape=[3, 256, 512]):
        self.save_root = save_root
        self.save_name = save_name
        self.parse = PytorchParser(model, input_shape)
        self.input_shape = input_shape
        self.save = {
            'structurejson': os.path.join(self.save_root, save_name + '.json'),
            'structurepb': os.path.join(self.save_root, save_name + '.pb'),
            'weights': os.path.join(self.save_root, save_name + '.npy'),

            'caffenetwork': os.path.join(self.save_root, save_name + '.py'),
            'caffeweights': os.path.join(self.save_root, save_name + '.cnpy'),
            'caffemodel': os.path.join(self.save_root, save_name)
        }

    def start(self):
        print("start to do pytorch to IR")
        self.parse.run(self.save_root + self.save_name)
        print("done! then to do IR to caffe code")
        emitter = CaffeEmitter((self.save['structurepb'], self.save['weights']))
        emitter.run(self.save['caffenetwork'], self.save['caffeweights'], 'test')
        print("done! then to do ccode to model")
        MainModel = imp.load_source('MainModel', self.save['caffenetwork'])
        save_model(MainModel, self.save['caffenetwork'], self.save['caffeweights'], 
                                        self.save['caffemodel'])

        print('start to rename inputs')
        lines = open(self.save['caffemodel'] + '.prototxt', 'r').readlines()
        new_lines = rename_input(lines, self.input_shape)
        fp = open(self.save['caffemodel'] + '.prototxt', 'w')
        fp.writelines(new_lines)
        print("^~^^~^^~^^~^")