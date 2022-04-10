from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
def lung1(img):
    image=Image.fromarray(img.astype('uint8'))
    model = load_model('app/base/static/models/lung.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    try:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    except:
        print()
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    l = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    maxPrediction = prediction.max()
    for i in range(0, len(l)):
        if prediction[0][i] == maxPrediction:
            ans = l[i]
    return ans, maxPrediction
# Load the model
def dr(image):
    image=Image.fromarray(image.astype('uint8'))
    model = load_model('app/base/static/keras_model.h5')
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    print(type(image))
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    l = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    maxPrediction = prediction.max()
    for i in range(0, len(l)):
        if prediction[0][i] == maxPrediction:
            ans=l[i]
    return ans, maxPrediction

def skin1(image):
    image=Image.fromarray(image.astype('uint8'))
    model = load_model('app/base/static/models/skin.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    try:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    except:
        print()
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    l = ['Basal Cell Carcinoma', 'Melanoma', 'Skin Cancer', 'Normal Skin', 'Squamous Cell Carcinoma']
    maxPrediction = prediction.max()
    for i in range(0, len(l)):
        if prediction[0][i] == maxPrediction:
            ans = l[i]
    return ans, maxPrediction
