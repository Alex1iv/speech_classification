import os

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, BackupAndRestore
from tensorflow.keras import layers, models

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
    
    backup = BackupAndRestore(
        # The path where your backups will be saved. Make sure the
        # directory exists prior to invoking `fit`.
        os.path.join(path_models),
        # How often you wish to save a checkpoint. Providing "epoch"
        # saves every epoch, providing integer n will save every n steps.
        save_freq="epoch",
        # Deletes the last checkpoint when saving a new one.
        delete_checkpoint=True,
    )
    # # reduces learnign rate smoothly
    # scheduler = LearningRateScheduler(
    #     schedule=smooth_decay(epoch, lr), 
    #     verbose=config.callbacks.verbose
    # )

    return [checkpoint, earlystop, reduce_lr, backup] 

def compute_mean_variance(dataset):
    """Compute mean & variance using tf.data.experimental.reduce() for efficiency."""
    def update_stats(state, batch):
        spec, _ = batch
        batch_mean, batch_var = tf.nn.moments(spec, axes=[0])
        batch_size = tf.shape(spec)[0]

        # Unpack state
        count, mean, M2 = state

        # Welford’s online update
        delta = batch_mean - mean
        new_count = count + tf.cast(batch_size, tf.float32)
        new_mean = mean + delta * (tf.cast(batch_size, tf.float32) / new_count)
        new_M2 = M2 + batch_var * tf.cast(batch_size, tf.float32)

        return new_count, new_mean, new_M2

    # Initialize state
    initial_state = (tf.constant(0, dtype=tf.float32),  
                     tf.zeros((124, 129, 1), dtype=tf.float32),  
                     tf.zeros((124, 129, 1), dtype=tf.float32))  

    # Compute mean & variance over dataset
    count, mean, M2 = dataset.reduce(initial_state, update_stats)

    variance = M2 / count
    return mean, variance


def normalize_dataset(dataset, mean, variance):
    """Normalize dataset using precomputed mean & variance (optimized version)."""
    
    return dataset.map(lambda spec, label: ((spec - mean) / tf.sqrt(variance), label))



# def normalize_dataset(dataset):
#     """Compute mean & variance and normalize the dataset."""
#     # Extract features only (discard labels)
#     data_only = dataset.map(lambda spec, label: spec)

#     # Compute mean & variance
#     data_tensor = tf.concat(list(data_only), axis=0)
#     mean, var = tf.nn.moments(data_tensor, axes=[0])

#     # apply normalization
#     return dataset.map(lambda spec, label: ((spec - mean) / tf.sqrt(var), label))


# def normalize_dataset(dataset):
#     """Compute mean & variance efficiently and normalize the dataset in a single pass."""
#     count = 0
#     mean = 0.0
#     M2 = 0.0  # Variance accumulator

#     # Compute mean & variance
#     for spec, _ in dataset:
#         batch_size = tf.shape(spec)[0]
#         batch_mean, batch_var = tf.nn.moments(spec, axes=[0])

#         # Welford’s algorithm for running mean & variance
#         delta = batch_mean - mean
#         new_count = count + batch_size
#         mean += delta * tf.cast(batch_size, tf.float32) / tf.cast(new_count, tf.float32)
#         M2 += batch_var * tf.cast(batch_size, tf.float32)  # Variance accumulator
#         count = new_count

#     variance = M2 / tf.cast(count, tf.float32)

#     # Normalize dataset using computed mean & variance
#     return dataset.map(lambda spec, label: ((spec - mean) / tf.sqrt(variance), label))

class ModelVoice_v1(tf.keras.Model):
    def __init__(self, output_units:int=31, **kwargs):
        super(ModelVoice_v1, self).__init__(**kwargs)
        
        self.layers_list = [
            layers.Conv2D(32, (7, 3), activation='relu', padding="same", name='Conv2D_1'),
            layers.MaxPooling2D(pool_size=(1, 3), name='MaxPool2D_1'),
            layers.Conv2D(64, (1, 7), activation='relu', padding="same", name='Conv2D_2'),
            layers.MaxPooling2D(pool_size=(1, 4), name='MaxPool2D_2'),
            layers.Dropout(0.25, name='Dropout_1'),
            layers.Conv2D(128, (1, 10), activation='relu', padding="valid", name='Conv2D_3'),
            layers.Conv2D(256, (7, 1), activation='relu', padding="valid", name='Conv2D_4'),
            layers.GlobalMaxPooling2D(),
            layers.Dropout(0.5, name='Dropout_2'),
            layers.Dense(128, name='Dense_1'),
            layers.Dense(output_units, name='Output')]

    def call(self, inputs, training=False):
        x = inputs
        #x = self.input_layer(inputs)
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

class ModelVoice_v2(tf.keras.Model):
    def __init__(self, output_units:int=31, **kwargs):
        super(ModelVoice_v2, self).__init__(**kwargs)
        #self.input_layer = layers.Input(shape=input_shape) # inferred automatically
        self.layers_list = [
            layers.Conv2D(64, (7, 3), activation='relu', padding='same', name='Conv2D_1'),
            layers.MaxPooling2D(pool_size=(1, 3), name='MaxPool2D_1'),
            layers.Conv2D(128, (1, 7), activation='relu', padding="same", name='Conv2D_2'),
            layers.MaxPooling2D(pool_size=(1, 4), name='MaxPool2D_2'),
            layers.Conv2D(256, (1, 10), activation='relu', padding="valid", name='Conv2D_3'),
            layers.Conv2D(512, (7, 1), activation='relu', padding="valid", name='Conv2D_4'),
            layers.GlobalMaxPooling2D(),
            layers.Dropout(0.5, name='Dropout_1'),
            layers.Dense(256, name='Dense_1'),
            layers.Dense(output_units, name='Output')
        ]
    
    def call(self, inputs, training=False):
        x = inputs
        #x = self.input_layer(inputs)
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    
# class ModelVoice_v0(tf.keras.Model):
#     def __init__(self, data, output_units=30, **kwargs):
#         super(ModelVoice_v0, self).__init__( **kwargs)

#         # Feature extraction
#         self.feature_extractor = models.Sequential([
#             layers.Resizing(32, 32, name='Resizing'),
#             layers.Conv2D(32, (3, 3), activation='relu', padding="same", name='Conv2D_1'),
#             layers.Conv2D(64, (3, 3), activation='relu', padding="same", name='Conv2D_2'),
#             layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool2D_1'),
#             layers.Dropout(0.25, name='Dropout_1')
#         ], name="FeatureExtractor")

#         # Flattening & Fully Connected Layers
#         self.classifier = models.Sequential([
#             layers.Flatten(name='Flatten'),
#             layers.Dense(128, activation='relu', name='Dense_128'),
#             layers.Dropout(0.5, name='Dropout_2'),
#             layers.Dense(output_units, name='Output')
#         ], name="Classifier")

#         # Normalization layer adapted to dataset
#         self.norm_layer = layers.Normalization(name='Normalization')
#         self.norm_layer.adapt(data.map(lambda spec, label: spec))

#     def call(self, inputs, training=False):
#         x = self.norm_layer(inputs)
#         x = self.feature_extractor(x, training=training)
#         x = self.classifier(x, training=training)
#         return x
#         #print('Output: ', x.shape)
#         # return x
#         #print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_channels.shape}")
    
#     # def get_model():
#     #     return ModelVoice(name='ModelVoice')

# class ModelVoice_2(tf.keras.Model):
#     def __init__(self, data, **kwargs):
#         super(ModelVoice, self).__init__(**kwargs)
#         #self.d1 = layers.Resizing(32, 32, name='Resizing')
#         #self.d2 = layers.Normalization(name='Normalization')
#         #self.d2.adapt(data=data.map(map_func=lambda spec, label: spec))
#         self.d1 = layers.Conv2D(64, [7,3], activation='relu', name='Convol_1')
#         self.d2 = layers.MaxPooling2D([1,3], name='MaxPooling2D_1_3')
#         self.d3 = layers.Conv2D(128, [1,7], activation='relu', name='Convol_2')
#         self.d4 = layers.MaxPooling2D([1,4], name='MaxPooling2D_1_4')
#         #self.Dropout_1 = layers.Dropout(0.25, name='Dropout_0.25')
#         self.d5 = layers.Conv2D(256, [1,10], padding="VALID", activation='relu', name='Convol_3')
#         self.d6 = layers.Conv2D(512, [7,1], activation='relu', name='Convol_6')
        
#         self.d6 = layers.Dropout(0.25, name='Dropout_0.25')
#         self.d7 = layers.Flatten(name='Flatten')
#         self.d8 = layers.Dense(128, activation='relu', name='Dense_128')
#         self.d9 = layers.Dropout(0.5, name='Dropout_0.5')
#         self.output_channel = layers.Dense(units=31, name='Dense_31')

#     def call(self, inputs, training=False): #, training=False
#         #print('initial: ', inputs.shape)
#         #x = self.d7(x)
#         #print('next: ', x.shape)
#         x = self.d1(inputs)
#         #print('After resizing: ', x.shape)
#         x = self.d2(x)
#         x = self.d3(x) 
#         x = self.d4(x) 
#         x = self.d5(x) 
#         x = self.d6(x)
#         #x = self.Dropout_1(x, training=training)
#         x = self.d7(x) 
#         x = self.d8(x) 
#         x = self.d9(x)
#         x = self.output_channel(x)
#         #print('Output: ', x.shape)
#         return x
#         #print(f"input_shape = {(self.n_timesteps, self.n_channels)} | output_units = {self.output_channels.shape}")
