import os
import tensorflow as tf
from tensorflow import keras

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    base_logdir = os.path.abspath(os.path.join(os.curdir, "my_logs"))
    logdir = os.path.join(base_logdir, run_id)
    
    # Ensure the base directory exists
    if not os.path.exists(base_logdir):
        os.makedirs(base_logdir, exist_ok=True)
        print(f"Created base log directory: {base_logdir}")
    
    # Ensure the specific log directory exists
    os.makedirs(logdir, exist_ok=True)
    print(f"Created log directory: {logdir}")
    
    return logdir

def setup_tensorboard_callbacks():
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    return tensorboard_cb, run_logdir
