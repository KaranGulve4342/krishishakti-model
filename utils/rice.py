from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
# model = load_model("paddy_disease_model.h5")

labels = ['Rice_bacterial_leaf_blight', 'Rice_bacterial_leaf_streak', 'Rice_bacterial_panicle_blight', 'Rice_blast',
          'Rice_brown_spot', 'Rice_dead_heart', 'Rice_downy_mildew', 'Rice_hispa', 'Rice_normal', 'Rice_tungro']

def get_cron_disease(img, model):
    """
    Predict the class of a single image using the trained model.

    Parameters:
        image_path (str): Path to the image to be predicted.
        model (tf.keras.Model): The trained model.
        labels (list): List of class labels corresponding to the model's output.

    Returns:
        str: The predicted class label.
    """
    # Load the image and ensure it has 3 channels
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to match training preprocessing
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_label = labels[np.argmax(predictions)]  # Get the label with the highest probability
    
    return predicted_label

# Define the class labels


# # Predict an example image
# image_path = "rice blast.jpg"  # Replace with the actual image path
# predicted_class = predict_single_image(image_path, model, labels)
# print("Predicted Class:", predicted_class)
