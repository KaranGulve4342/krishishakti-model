import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

classes = {
    0: "Apple_Rot_Leaves",
    1: "Apple_Healthy_Leaves",
    2: "Apple_Leaf_Blotch",
    3: "Apple_Scab_Leaves",
}

# model = models.load_model('Potato.h5')

# img = Image.open(r"C:\Users\Prathamesh\Downloads\The-three-Sample-leaves-of-potato-are-a-leaf-affected-by-Light-Blight-b-leaf.png")
def get_apple_disease(img,model):

    resized_img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction[0])]

    print(predicted_class)
    
    return predicted_class