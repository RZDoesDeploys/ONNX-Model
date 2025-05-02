from PIL import Image
import numpy as np
import sys

def normalize(img, mean, std):
    return (img / 255.0 - mean) / std

def preprocess_image(image_path, output_path, input_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
    image_np = np.array(image).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = normalize(image_np, mean, std)

    image_np = np.transpose(image_np, (2, 0, 1))  # (C, H, W)
    image_np = np.expand_dims(image_np, axis=0)  # (1, C, H, W)

    np.save(output_path, image_np)

if __name__ == "__main__":
    preprocess_image(sys.argv[1], sys.argv[2])
