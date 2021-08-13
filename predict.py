import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def parse_args():
    """ Parse the command line arguments
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Flower image classifier')
    parser.add_argument('image_path', type=str, help='path to an image file')
    parser.add_argument('model_path', type=str, help='path to a model file')
    parser.add_argument('--top_k', type=int, help='numper of top categories')
    parser.add_argument('--category_names', type=str, help='path to a category json file')
    return parser.parse_args()

def load_model(model_path):
    """ Load model
    Arguments:
        model_path: Path to the model file
    Returns:
        Model
    """
    return tf.keras.models.load_model(model_path, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    
def load_classnames(file_path):
    """ Load class names from JSON file
    Arguments:
        file_path: Path to the JSON file
    Returns:
        Dictionary of class names
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
def get_class_label(image_class: int, class_names: dict):
    """
    Get label of corresponding image class
    
    Arguments:
        image_class: integer output of neural network
        class_names: classes to labels mapping
    Returns:
        label for the given class
    """
    return class_names[str(image_class + 1)]
    
def process_image(image: np.array):
    """
    Forward image data through neural network
    """
    tensor = tf.image.resize(image, [224, 224])
    tensor /= 255
    return tensor.numpy()

def predict(image_path: str, model, top_k: int):
    """
    Makes a prediction which type of flower an image shows
    """
    # Open image
    im = Image.open(image_path)
    # Convert image to numpy array
    arr = np.asarray(im)
    # Process image array (resizing)
    arr = process_image(arr)
    # Add extra dimension
    arr = arr.reshape((1, 224, 224, 3))
    # Predict classes' probabilities
    predictions = model.predict(arr).squeeze()
    # TODO remove for loop
    for i, p in enumerate(predictions):
        print(i, p)
    # Get top k classes
    indices = np.argpartition(predictions, -top_k)[-top_k:]
    return np.flipud(predictions[indices]), np.flipud(indices)


def run():
    """
    Main function that is called when program is executed
    """
    args = parse_args()
    model = load_model(args.model_path)
    preds, classes = predict(args.image_path, model, args.top_k)
    print(preds, classes)
    if args.category_names:
        class_names = load_classnames(args.category_names)
        classes = [f"{x}, {get_class_label(int(x), class_names)}" for x in classes]
    print(f"\nPrediction for image {args.image_path}:\n")
    for pred, clazz in zip(preds, classes):
        if args.category_names:
            indent = 3 - ((len(clazz) - 1) // 8)
            if (len(clazz) - 1) % 8 == 0:
                indent += 1
            indent = "\t" * indent
        else:
            indent = "\t"
        pred = "{:,.2f}".format(round(pred, 4) * 100)
        print(f"\t{clazz}{indent}=>    {pred}%")

run()