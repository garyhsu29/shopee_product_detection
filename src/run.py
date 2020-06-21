import numpy as np
import pandas as pd
import config
import tensorflow as tf
from dataload import trn_genertor, val_genertor
from model import cnn_model
import tensorflow as tf

def run():
    trn_df  = pd.read_csv(config.TRN_FILE)
    test_df = pd.read_csv(config.TST_FILE)
    
    cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = cnn_model.fit_generator(
                trn_genertor,
                steps_per_epoch = trn_genertor.samples//config.BATCH_SIZE,
                epochs=config.EPOCH,
                validation_data=val_genertor,
                validation_steps = val_genertor.samples // config.BATCH_SIZE,
                verbose = 1
            )

if __name__ == '__main__':
    run()
    