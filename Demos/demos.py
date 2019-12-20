import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import easy_tf_log as etl
from datetime import datetime
from tqdm import trange

from sklearn.model_selection import train_test_split

#%% Demo values
model = tf.keras.Sequential()

#%% ML Demos

# Split data into train and test from Pandas dataframe
train, test = train_test_split(pd.DataFrame(list(range(100))), test_size=0.2)
# Another good way to split using pandas only
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Keras training callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#%% Pandas
train_dataset.isna().sum()
dataset = train_dataset.dropna()
origin = dataset.pop('Origin')
filtered_df_by_keyword = dataset.filter(like='loss')

#%% Tensorflow / Keras

#For specifying device to use
with tf.device('/gpu:0'): pass

# Adding new axis to array
x_train = train[..., tf.newaxis]

# Tensorboard setup
logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
callbacks=[tensorboard_callback] # in model.fit()

# Easy tf log to tensorboard for scalars
etl.set_dir('logs2')
for k in range(20, 30): etl.tflog('baz', k)
# to start tensorboard put this into the terminal: tensorboard --logdir path/to/log/dir

# Plot Graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
pass

# Class for displaying progress on the end of an epoch
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions()  # could use random sample or use a specific preset one
pass

#%% Plotting
# TF has a plotter with smoothing built int, works with matplotlib plts
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=1)
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])


#%% Active plot callback
history = dict()
plot_keys = ['accuracy', 'val_accuracy']
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global history
        for key in logs.keys() :
            if key in history.keys() : history[key].append(logs[key])
            else : history[key] = [logs[key]]

        for key in plot_keys : plt.plot(history[key], label = key)
        plt.legend(loc = 'upper left')
        plt.show()
pass

#%% Printing

# Use template to format string for progress printing
template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
print (template.format(488, 489))

# For taking up a certain amount of space
print('{:<5} : {:>5}'.format('ty', 'hi'), '\n', '{:>5} : {:<5}'.format('ty', 'hi'))

# Set numpy print options for the session to suppress scientific notation and use only 3 sig figs
np.set_printoptions(precision=3, suppress=True)

#%% Python
# Inline if statements
text = "True" if (True) else "False"
print(text) if True else 0
print(text if False else 0)

# Get input loop
for i in range(10) :
    new_rev = input()
    prediction = model.predict(tf.convert_to_tensor([new_rev]))
    print(prediction[0])

# TQDM for loop progress bar
for i in trange(100000000, unit_scale=True, desc="hello", unit="epoch"): 5+45

# Decorators
def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

##%% Misc
# Conver Jupyter notebook to executable script (in command line)
jupyter nbconvert --to script /home/starstorms/InsightDemo/fellows.ipynb


#%% Open AI Gym
# To get just the base env without the env wrapper which adds time limits and such (note the .env part at the end)
env = gym.make("MountainCar-v0").env

#To register your own env locally with custom parameters
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=250,      # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make('MountainCarMyEasyVersion-v0')

#%% Amazing results for cartpole, got avg >200 in just ~20 agent sessions
eps_initial, eps_decay, eps_min, gamma, max_mem, learning_rate = .6, .9, .1, .99, 10000, 1e-4
neurons, act_type, final_type = [200, 100], 'relu', 'linear'
train_sessions_per_epoch, train_iterations_after_experience_gather = 2, 1000