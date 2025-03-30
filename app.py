import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
onnx_model_path = "./model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Class names
class_names = ['Tomato_Rot',
 'Apple_Fresh',
 'Banana_Fresh',
 'Banana_Anthracnose',
 'Mango_Fresh',
 'Okra_Rot',
 'Peach_Scab',
 'Tomato_Fresh',
 'Mango_Black Mould Rot',
 'Grape_Gray Mold',
 'Papaya_Ring Spot',
 'Strawberry_Powdery Mildew',
 'Orange_Pencilium Decay',
 'Lime_Fresh',
 'Mango_Alternaria',
 'Mango_Anthracnose',
 'Papaya_Fresh',
 'Cucumber_Fresh',
 'Apple_Sooty Blotch',
 'Capsicum_Fresh',
 'Papaya_Powdery Mildew',
 'Lime_Rotten',
 'Guava_Fresh',
 'Peach_Brown Rot',
 'Peach_Fresh',
 'Grape_Powdery Mildew',
 'Mango_Stem End Rot',
 'Strawberry_Fresh',
 'Orange_Citrus Black Spot',
 'Guava_Rotten',
 'Apple_Moldy Core',
 'Banana_Cigar End Rot',
 'Cucumber_Rot',
 'Papaya_Anthracnose',
 'Papaya_Black Spot',
 'Orange_Sour Rot',
 'Strawberry_Gray Mold',
 'Orange_Fresh',
 'Apple_Scab',
 'Capsicum_Stale',
 'Banana_Thrips Damage',
 'Grape_Fresh',
 'Okra_Fresh',
 'Papaya_Phytophthora',
 'Apple_Black Rot']

def normalize(image, mean, std):
    image = image / 255.0  # scale to [0, 1]
    for c in range(3):
        image[..., c] = (image[..., c] - mean[c]) / std[c]
    return image

# Preprocess the image
def preprocess_image(image_path, input_size=(224, 224)):
    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image to the target size
    image = image.resize(input_size)
    
    # Convert image to numpy array
    image_np = np.array(image).astype(np.float32)
    
    # Normalize the image with mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_np = normalize(image_np, mean, std)
    
    # Rearrange dimensions to (1, C, H, W) and return as numpy array
    image_np = np.transpose(image_np, (2, 0, 1))  # from (H, W, C) to (C, H, W)
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension (1, C, H, W)
    
    return image_np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Predict function
def predict(image_path):
    input_image = preprocess_image(image_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: input_image})[0]
    predictions = softmax(predictions)
    max_index = np.argmax(predictions)
    predicted_class = class_names[max_index]
    confidence_score = float(np.max(predictions)) * 100
    return predicted_class, round(confidence_score, 2)

# Example usage
image_path = "./apple.jpg"
predict(image_path)