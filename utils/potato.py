import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

classes = {
    0: "Potato_Early_Blight",
    1: "Potato_Late_Blight",
    2: "Potato_Healthy",
}

def get_potato_disease(img,model):

    resized_img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction[0])]

    print(predicted_class)
    
    return predicted_class