import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

classes = {
    0: "Coffee_Cercospora",
    1: "Coffee_Healty",
    2: "Coffee_Rust"
}

def get_coffee_disease(img,model):

    resized_img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction[0])]

    print(predicted_class)
    
    return predicted_class