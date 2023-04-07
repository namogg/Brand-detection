import pandas as pd

annonations = pd.read_csv("flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

annonations = annonations.drop_duplicates(subset=[0, 1, 3, 4, 5, 6], keep='first')

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


def create_label(annonations): 
    grouped_annotations = pd.DataFrame(annonations).groupby(0)
    for group in grouped_annotations:
        group = pd.DataFrame(group)
        filename = group.iloc[0,0]
        with open(filename,'w') as f: 
            for i in range(len(group)):
                brand_id = get_id_by_brand(group.iloc[i,1], name_map)
                x_min = group.iloc[i, 3]
                y_min = group.iloc[i, 4]
                x_max = group.iloc[i, 5]
                y_max = group.iloc[i, 6]
                f.write(group_id + " " + x_min + " "  + y_min + " " + x_max + " "+ y_max)

create_label(annonations)
        