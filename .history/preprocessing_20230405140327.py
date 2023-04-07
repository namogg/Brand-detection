import pandas as pd
import os
import shutil
from PIL import Image
annotations = pd.read_csv("flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

annotations = annotations.drop_duplicates(subset=[0, 1, 3, 4, 5, 6], keep='first')

datadir = "flickr_logos_27_dataset/flickr_logos_27_dataset_images"
name_map = {
    'Ferrari': 0,
    'Ford': 1,
    'Nbc': 2,
    'Starbucks': 3,
    'RedBull': 4,
    'Mini': 5,
    'Unicef': 6,
    'Yahoo': 7,
    'Sprite': 8,
    'Texaco': 9,
    'Intel': 10,
    'Cocacola': 11,
    'Citroen': 12,
    'Heineken': 13,
    'Apple': 14,
    'Google': 15,
    'Fedex': 16,
    'Pepsi': 17,
    'Puma': 18,
    'DHL': 19,
    'Porsche': 20,
    'Nike': 21,
    'Vodafone': 22,
    'BMW': 23,
    'McDonalds': 24,
    'HP': 25,
    'Adidas': 26
}
def coordinate_transform(xmin, ymin, xmax, ymax): 
    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + (width / 2)
    center_y = ymin + (height / 2)
    return center_x, center_y, width, height

def get_id_by_brand(name, name_map):
    return name_map[name]


def create_label_classification(annotations):
    grouped_annotations = annotations.groupby(0)
    with open("classification lael",'w') as f:
    for group in grouped_annotations:
        filename = group[0]
        file_name, file_ext = os.path.splitext(filename)
        file_name_txt = os.path.basename(file_name) + ".txt"
        image_path = os.path.join(datadir, filename).replace("\\", "/")
            brand_id = get_id_by_brand(group[1].iloc[i, 1], name_map)
            f.write(f"{brand_id} \n")    
        
def create_label_detect(annotations): 
    grouped_annotations = annotations.groupby(0)
    for group in grouped_annotations:
        filename = group[0] 
        file_name, file_ext = os.path.splitext(filename)
        file_name_txt = os.path.basename(file_name) + ".txt"
        image_path = os.path.join(datadir, filename).replace("\\", "/")
        #if os.path.exists(image_path):
            #shutil.copy(image_path, "images/train")
        image = Image.open(image_path)
        width, height = image.size
        with open("labels/train/"+file_name_txt,'w') as f: 
            for i in range(len(group[1])):
                brand_id = get_id_by_brand(group[1].iloc[i, 1], name_map)
                x_min = group[1].iloc[i, 3]
                y_min = group[1].iloc[i, 4]
                x_max = group[1].iloc[i, 5]
                y_max = group[1].iloc[i, 6]
                center_x , center_y , obj_width, obj_height = coordinate_transform(x_min, y_min, x_max, y_max)
                center_x = center_x/width
                center_y = center_y/height
                obj_width = obj_width/width
                obj_height = obj_height/height
                f.write(f"{brand_id} {center_x} {center_y} {obj_width} {obj_height}\n")

create_label_detect(annotations)

