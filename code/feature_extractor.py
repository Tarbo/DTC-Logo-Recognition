from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

nas_model = NASNetLarge(
    input_shape=None,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling='max',
)


def extract_features(img):
    """Extract features from input image and return the same"""
    print(f'>>> Entered feature extractor...', end='')
    start = time.time()
    img = image.load_img(img, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = nas_model.predict(x)
    print(f'Feature extraction took: {time.time()-start:.2}. Exiting. ')
    return features.flatten()


def measure_similarity(feature_array):
    """Compare the image feature with the logo bank"""
    print(f'>>> Entered the similarity function...', end='')
    try:
        logo_feature_path = os.path.abspath('code/yolo_v4/logo_bank.json')
        # get available logo names before comparison
        with open(logo_feature_path, 'r') as logo_bank:
            stored_logos = json.load(logo_bank)
    except Exception as err:
        print(f'>>> Exception: {err}')
    logo_names = list(stored_logos.keys())
    logo_features = list(stored_logos.values())
    combined_features = []
    combined_features.append(feature_array)
    for feature in logo_features:
        combined_features.append(feature)
    similarity_matrix = cosine_similarity(np.array(combined_features))
    print(f'>>> Exiting')
    # extract the index of the test image
    return similarity_matrix[0], logo_names


def match_brand(img):
    """ Return the name of the brand with most similarity and similarity score"""
    print(f'>>> Entered the match brand function...', end='')
    feature_arr = extract_features(img).tolist()
    similarity, logo_names = measure_similarity(feature_arr)
    # remove the first value cos its self comparison
    similarity = similarity[1:]
    max_id = np.argmax(similarity)
    matched_brand = logo_names[max_id]
    confidence = similarity[max_id]
    # build a dictionary for use as table in the html
    index = 0
    result = {}
    for brand in logo_names:
        result[brand] = similarity[index]
        index += 1
    print(f'>>> Exiting')
    return matched_brand, confidence, result


def add_to_logo_bank(logos_images):
    """Add the logos to logo bank"""
    print(f'>>> Entered the add to logo bank function...', end='')
    img_folder = os.path.join(os.getcwd(), 'code/static/assets/img')
    img_names = list(logos_images.values())
    brand_names = list(logos_images.keys())
    features = []
    for img in img_names:
        img_path = os.path.join(img_folder, img)
        feature = extract_features(img_path)
        features.append(feature)
        print(f'>>> {img} feature shape: {feature.shape}')
    logo_dic = {key: value.tolist()
                for key, value in zip(brand_names, features)}
    try:
        logo_bank_path = os.path.abspath('code/yolo_v4/logo_bank.json')
        with open(logo_bank_path, 'w') as logo_bank:
            json.dump(logo_dic, logo_bank, indent=4)
    except Exception as identifier:
        print(f'>>> Exception: {identifier}')
    print(f'Exiting the add to logo bank function')


if __name__ == '__main__':
    logo_images = {'Nike': 'nike_logo.jpg',
                   'Razer Inc': 'razer_logo.jpg'}
    #add_to_logo_bank(logo_images)
    img_path='C:/Users/melli/OneDrive/insight/code/static/assets/img/nike_shoe.jpg'
    brand, confidence, results = match_brand(img_path)
    print(f'Predicted Brand: {brand}\t Confidence: {confidence}')
    for key, value in results.items():
        print(f'|{key} |{value}|')
        print('..................')