import tensorflow as tf
import config

#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split = 0.2)

trn_genertor = img_generator.flow_from_directory(
                    directory=config.TRN_IMAFE_FILE,
                    shuffle=True,
                    color_mode='rgb',
                    target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                    subset='training',
                    batch_size=config.BATCH_SIZE
                )   

val_genertor = img_generator.flow_from_directory(
                    directory=config.TRN_IMAFE_FILE,
                    shuffle=True,
                    color_mode='rgb',
                    target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
                    subset='validation',
                    batch_size=config.BATCH_SIZE
                )   


if __name__ == '__main__':  
    for image_batch, label_batch in val_genertor:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break
    print(len(trn_genertor))
    print(len(val_genertor))