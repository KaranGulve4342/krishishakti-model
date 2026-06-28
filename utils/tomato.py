import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Define class labels (ensure these match your dataset's class labels)
labels = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
          'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot', 
          'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

def get_tomato_disease(img, model):
    """
    Predict the class of a single image using the trained model.

    Parameters:
        image_path (str): Path to the image to be predicted.
        model (tf.keras.Model): The trained model.
        labels (list): List of class labels corresponding to the model's output.

    Returns:
        str: The predicted class label.
    """
    # Load and preprocess the image
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to match training preprocessing

    # Predict the class
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]  # Get the label with the highest probability

    return predicted_label

# # Load the saved model
# model = tf.keras.models.load_model("tomato_leaf_model.h5")

# # Example usage
# image_path = "leaf_mold.jpg"  # Replace with the path to your image
# predicted_class = predict_single_image(image_path, model, labels)
# print("Predicted Class:", predicted_class)
