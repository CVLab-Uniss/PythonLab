from keras.applications.vgg16 import VGG16​

from tensorflow.keras.preprocessing import image​

from tensorflow.keras.applications.vgg16 import preprocess_input​

from tensorflow.keras.applications.vgg16 import decode_predictions​

import numpy as np​

​
model = VGG16(weights='imagenet')​

#print(model.summary())​

​
# Input image​

img_path = 'cat.jpg’​

​# Method to match the source size with the target size​

img = image.load_img(img_path, color_mode='rgb', target_size=(224, 224))​

display(img)​

​
# Converts a PIL Image to 3D Numy Array​

x = image.img_to_array(img)​

​
# Adding the fouth dimension, for number of images​

x = np.expand_dims(x, axis=0)​

​
# Mean centering with respect to Image​

x = preprocess_input(x)​

# Predict​

out = model.predict(x)​

labelFull = decode_predictions(out)​

​
# Retrieve the most likely result, e.g. highest probability​

label = labelFull[0][0]​

​

# Print classification result​

print('%s (%.2f%%)' % (label[1], label[2]*100))​


