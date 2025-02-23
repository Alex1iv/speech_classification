import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tf.keras.layers import  Dense, Input
# from tf.keras.optimizers import Adam
# from tf.keras import layers
#from tf.keras.models import Sequential, load_model
#from tf.keras.models import Model

class ModelVoice(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ModelVoice, self).__init__(**kwargs)
        self.d1 = layers.Resizing(32, 32, name='Resizing')
        self.d2 = layers.Normalization(name='Normalization')
        self.d2.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))
        self.d3 = layers.Conv2D(32, 3, activation='relu', name='Convol_1')
        self.d4 = layers.Conv2D(64, 3, activation='relu', name='Convol_2')
        self.d5 = layers.MaxPooling2D(name='MaxPooling2D')
        #self.Dropout_1 = layers.Dropout(0.25, name='Dropout_0.25')
        self.d6 = layers.Dropout(0.25, name='Dropout_0.25')
        self.d7 = layers.Flatten(name='Flatten')
        self.d8 = layers.Dense(128, activation='relu', name='Dense_128')
        self.d9 = layers.Dropout(0.5, name='Dropout_0.5')
        self.output_channel = layers.Dense(units=31, name='Dense_31')

    def call(self, inputs, training=False): #, training=False
        #print('initial: ', inputs.shape)
        #x = self.d7(x)
        #print('next: ', x.shape)
        x = self.d1(inputs)
        #print('After resizing: ', x.shape)
        x = self.d2(x)
        x = self.d3(x) 
        x = self.d4(x) 
        x = self.d5(x) 
        x = self.d6(x)
        #x = self.Dropout_1(x, training=training)
        x = self.d7(x) 
        x = self.d8(x) 
        x = self.d9(x)
        x = self.output_channel(x)
        #print('Output: ', x.shape)
        return x
        #print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_channels.shape}")


def callbacks(model_name:str, path_models:str,
    reduce_patience:int,
    stop_patience:int,
    monitor:str,
    verbose:bool,
    mode:str,
    save_best_only:bool,
    save_weights_only:bool,
    restore_best_weights:bool,
    cooldown_epochs:int,
    lr:float,
    factor:float
    ):
    """ Manages the learning process of our model

    Args:
        model_name (str): model name
        path_models (str): path to save
        reduce_patience (int): decreases the lr when metrics doesn't change
        stop_patience (int): the number of epochs before the learning process is terminated if the metric doesn't change
        monitor (str): metric to monitor
        verbose (bool): shows the output
        mode (str): study mode
        save_best_only (bool): saves models with improved quality
        save_weights_only (bool): _description_
        restore_best_weights (bool): _description_
        cooldown_epochs (int): wait period when 
        lr (float): the learning rate
        factor (float): learning rate decrease factor (0,1)
    """

    # End training if the metric doesn't imporve
    earlystop = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=stop_patience,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )

    # Decrease learning rate if the metric doesn't improve 
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, 
        mode=mode,  
        min_lr=lr/1000,
        factor=factor, 
        patience=reduce_patience,
        cooldown=cooldown_epochs,
        verbose=verbose
    )
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(path_models, model_name + '.weights.h5'), #full: '_model.keras' subclass:.h5
        #save_format="tf", # saves a subclassed model
        save_best_only=save_best_only, 
        save_weights_only=save_weights_only,
        monitor=monitor, 
        mode=mode,
        verbose=verbose        
    )

    # # reduces learnign rate smoothly
    # scheduler = LearningRateScheduler(
    #     schedule=smooth_decay(epoch, lr), 
    #     verbose=config.callbacks.verbose
    # )

    return [checkpoint, earlystop, reduce_lr] 