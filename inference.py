import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from utils.reader_config import config_reader
# Import parameters
config = config_reader('./config/config.json')

# Set the seed value for experiment reproducibility.
random_seed = config.random_seed
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

path_data = config.path_data #'./data/'#config.data_dir 
path_models = config.path_models #'../models/'

# Read test data
test_df = tf.keras.utils.audio_dataset_from_directory(
    directory='./sample_commands/', #path_data + 'test/', 
    labels=None, # no labels are specified
    batch_size=64, 
    shuffle=False, #avoid shuffling
    output_sequence_length=16000, 
    seed=random_seed,  
)

test = test_df.take(1)
print(test.element_spec)

# squeeze the data
def reduce_dimension(x):
  return tf.squeeze(x, axis=-1)
 
flat_df = test.map(reduce_dimension)

def apply_stft(audio_batch):
    """Applies STFT to a batch of audio samples."""
    spectrogram = tf.signal.stft(
        audio_batch, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)  # Return magnitude spectrogram
    
    return  spectrogram[..., tf.newaxis] # expand axis

# Apply the function to the dataset
test_spectrogram = flat_df.map(apply_stft, num_parallel_calls=tf.data.AUTOTUNE)


loaded_model = load_model('./models/model_v3.keras')

loaded_model.build(input_shape=(None, 124, 129, 1))

loaded_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

y_pred = loaded_model.predict(test_spectrogram)
y_pred = tf.argmax(y_pred, axis=1)
#print(y_pred, '\nShape is: ',y_pred.shape)
print(y_pred)

replacement_dict = {
     0:'silence',  1:'unknown',  2:'unknown',  3:'unknown',  4:'unknown',  5:'unknown',
     6:'unknown',  7:'unknown',  8:'unknown', 10:'unknown', 11:'unknown', 13:'unknown', 
    14:'unknown', 18:'unknown', 20:'unknown', 21:'unknown', 22:'unknown', 24:'unknown', 
    25:'unknown', 26:'unknown', 28:'unknown', 30:'unknown',  9:'go', 	  12:'left', 
    15:'no', 16:'off', 17:'on', 19:'right', 23: 'stop',     27:'up', 	  29:'yes'}

def replacement_func(val, replacement_dict:dict):
    return replacement_dict[val] if val in replacement_dict else val

replacement_func = np.vectorize(replacement_func)

# convert to NumPy
y_pred = replacement_func(y_pred.numpy(), replacement_dict)
print(y_pred)