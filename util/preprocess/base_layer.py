# -*- coding: utf-8 -*-

class Layer(object):
    def __init__(self, name = 'untitled_layer', inversible = False):
        self.name = name
        self.inversible = inversible
    
    def get_class_name(self):
        return None
    
    def tf_operate(self, *args, **kwargs):
        msg = 'method \'layer_op\' in \'{}\''.format(type(self).__name__)
        tf.logging.fatal(msg)
        raise NotImplementedError
    
    def np_operate(self, *args, **kwargs):
        msg = 'method \'layer_op\' in \'{}\''.format(type(self).__name__)
        tf.logging.fatal(msg)
        raise NotImplementedError

    def np_inverse_operatve(self, *args, **kwargs):
        msg = 'method \'layer_op\' in \'{}\''.format(type(self).__name__)
        tf.logging.fatal(msg)
        raise NotImplementedError
