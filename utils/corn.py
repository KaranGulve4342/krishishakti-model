from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the device


# Load the saved model



# Define the transformation for the single image
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the class labels
labels = ['Corn_blight', 'Corn_common_rust', 'Corn_gray_leaf_spot', 'Corn_healthy'] 

def get_cron_disease(img, model, device):
    """
    Predict the class of a single image.

    Parameters:
        image_path (str): Path to the image to predict.
        model (torch.nn.Module): The trained PyTorch model.
        transform (torchvision.transforms.Compose): The preprocessing transformations.
        device (torch.device): The device to run the model on (CPU/GPU).
        labels (list): List of class labels.

    Returns:
        str: Predicted class label.
    """
    # Load the image
    image = img.convert("RGB")  # Ensure the image has 3 channels
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)  # Get the class index with highest probability
    
    return labels[predicted_class.item()]

 # Replace with your actual class names

# # Predict a single image
# image_path = "gray_leaf_spot.jpg"  # Replace with the path to your image
# predicted_label = predict_single_image(image_path, model, transform, device, labels)
# print("Predicted Class:", predicted_label)
