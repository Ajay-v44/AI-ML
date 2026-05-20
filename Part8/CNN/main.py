import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image



train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
    './training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory(
    './test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


cnn=tf.keras.models.Sequential()

# 1. Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))

# 2. Pooling Layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))


# 3. Flatten Layer
cnn.add(tf.keras.layers.Flatten())

# 4. Full Connection Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# 5. Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=train_set, validation_data=test_set, epochs=25)


# Making a single prediction


test_image = image.load_img('./single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
train_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)