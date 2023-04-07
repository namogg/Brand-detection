import pandas as pd

annotations = pd.read_csv("flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

annotations = annotations.drop_duplicates(subset=[0, 1, 3, 4, 5, 6], keep='first')

name_map = {
    'Ferrari': 1,
    'Ford': 2,
    'Nbc': 3,
    'Starbucks': 4,
    'RedBull': 5,
    'Mini': 6,
    'Unicef': 7,
    'Yahoo': 8,
    'Sprite': 9,
    'Texaco': 10,
    'Intel': 11,
    'Cocacola': 12,
    'Citroen': 13,
    'Heineken': 14,
    'Apple': 15,
    'Google': 16,
    'Fedex': 17,
    'Pepsi': 18,
    'Puma': 19,
    'DHL': 20,
    'Porsche': 21,
    'Nike': 22,
    'Vodafone': 23,
    'BMW': 24,
    'McDonalds': 25,
    'HP': 26,
    'Adidas': 27,
}


def get_id_by_brand(name, name_map):
    return name_map[name]


def create_label(annotations): 
    grouped_annotations = annotations.groupby(0)
    for group in grouped_annotations:
        filename = group[0] + ".txt"
        with open("label/"filename,'w') as f: 
            for i in range(len(group[1])):
                brand_id = get_id_by_brand(group[1].iloc[i, 1], name_map)
                x_min = group[1].iloc[i, 3]
                y_min = group[1].iloc[i, 4]
                x_max = group[1].iloc[i, 5]
                y_max = group[1].iloc[i, 6]
                f.write(str(brand_id) + " " + str(x_min) + " "  + str(y_min) + " " + str(x_max) + " "+ str(y_max))

create_label(annotations)
