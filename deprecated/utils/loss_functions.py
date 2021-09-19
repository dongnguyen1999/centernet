import numpy as np
from utils.config import Config
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import convert_to_tensor

class CenterNetLosses:
    def __init__(self, config: Config, alpha=2., beta=4.,  offset_loss_weight=1.0, size_loss_weight=5.0):
        self.alpha = alpha
        self.beta = beta
        self.offset_weight = offset_loss_weight
        self.size_weight = size_loss_weight
        self.output_size = config.output_size
        self.num_classes = config.num_classes

    def _get_mask(self, y_true):
        mask = convert_to_tensor(np.zeros((self.output_size, self.output_size)).astype(np.float32))
        for category in range(self.num_classes):
            category_mask = K.sign(y_true[..., self.num_classes+4 + category])
            mask = tf.add(mask, category_mask)
        return K.sign(mask)

    def size_loss(self):
        def _size_loss(y_true, y_pred):
            mask = self._get_mask(y_true)
            N = K.sum(mask)
            sizeloss = K.sum(
                K.abs(y_true[..., self.num_classes+2]-y_pred[..., self.num_classes+2] * mask)
                + K.abs(y_true[..., self.num_classes+3]-y_pred[..., self.num_classes+3] * mask)
            )
            return (self.size_weight*sizeloss)/N
        return _size_loss

    def offset_loss(self):
        def _offset_loss(y_true, y_pred):
            mask = self._get_mask(y_true)
            N = K.sum(mask)
            offsetloss = K.sum(
                K.abs(y_true[..., self.num_classes]-y_pred[..., self.num_classes] * mask)
                + K.abs(y_true[..., self.num_classes+1]-y_pred[..., self.num_classes+1] * mask)
            )
            return (self.offset_weight*offsetloss)/N
        return _offset_loss

    def heatmap_loss(self):
        def _heatmap_loss(y_true, y_pred):
            mask = self._get_mask(y_true)
            N=K.sum(mask)

            heatmap_true = K.flatten(y_true[..., self.num_classes+4 : 2*self.num_classes+4]) #Exact center points true
            heatmap_true_rate = K.flatten(y_true[..., : self.num_classes]) #Heatmap true
            heatmap_pred = K.flatten(y_pred[..., : self.num_classes]) #Predicted heatmap 

            heatloss = -K.sum(
                heatmap_true* ((1-heatmap_pred)**self.alpha)*K.log(heatmap_pred + 1e-6)
                + (1-heatmap_true)* ((1-heatmap_true_rate)**self.beta)*(heatmap_pred**self.alpha) * K.log(1-heatmap_pred + 1e-6)
            )

            return heatloss/N
        return _heatmap_loss

    def centernet_loss(self):
        def _centernet_loss(y_true, y_pred):
            mask = self._get_mask(y_true)
            N=K.sum(mask)

            heatmap_true = K.flatten(y_true[..., self.num_classes+4 : 2*self.num_classes+4]) #Exact center points true
            heatmap_true_rate = K.flatten(y_true[..., : self.num_classes]) #Heatmap true
            heatmap_pred = K.flatten(y_pred[..., : self.num_classes]) #Predicted heatmap 

            heatloss = -K.sum(
                heatmap_true* ((1-heatmap_pred)**self.alpha)*K.log(heatmap_pred + 1e-6)
                + (1-heatmap_true)* ((1-heatmap_true_rate)**self.beta)*(heatmap_pred**self.alpha) * K.log(1-heatmap_pred + 1e-6)
            )

            offsetloss = K.sum(
                K.abs(y_true[..., self.num_classes]-y_pred[..., self.num_classes] * mask)
                + K.abs(y_true[..., self.num_classes+1]-y_pred[..., self.num_classes+1] * mask)
            )

            sizeloss = K.sum(
                K.abs(y_true[..., self.num_classes+2]-y_pred[..., self.num_classes+2] * mask)
                + K.abs(y_true[..., self.num_classes+3]-y_pred[..., self.num_classes+3] * mask)
            )
            
            all_loss=(heatloss + self.offset_weight*offsetloss + self.size_weight*sizeloss) / N
            return all_loss
        return _centernet_loss
    
    def all_losses(self):
        _centernet_loss = self.centernet_loss()
        _heatmap_loss = self.heatmap_loss()
        _offset_loss = self.offset_loss()
        _size_loss = self.size_loss()
        return _centernet_loss, _heatmap_loss, _offset_loss, _size_loss