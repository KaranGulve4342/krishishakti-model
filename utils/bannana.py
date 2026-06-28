import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

classes = {
    0: "Banana_Black_Sigatoka_Disease",
    1: "Banana_Bract_Mosaic_Virus_Disease",
    2: "Banana_Healthy_Leaf",
    3: "Banana_Insect_Pest_Disease",
    4: "Banana_Moko_Disease",
    5: "Banana_Panama_Disease",
    6: "Banana_Yellow_Sigatoka_Disease"
}

def get_bannana_disease(img,model):

    resized_img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction[0])]

    print(predicted_class)
    
    return predicted_class