import pandas as pd

annonations = pd.read_csv("C:/Users/ADMIN/OneDrive - Hanoi University of Science and Technology/Tài liệu/Projects/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

annonations = annonations.drop_duplicates(subset=[0, 1, 3, 4, 5, 6], keep='first')

brand_map = {
    1: 'Ferrari',
    2: 'Ford',
    3: 'Nbc',
    4: 'Starbucks',
    5: 'RedBull',
    6: 'Mini',
    7: 'Unicef',
    8: 'Yahoo',
    9: 'Sprite',
    10: 'Texaco',
    11: 'Intel',
    12: 'Cocacola',
    13: 'Citroen',
    14: 'Heineken',
    15: 'Apple',
    16: 'Google',
    17: 'Fedex',
    18: 'Pepsi',
    19: 'Puma',
    20: 'DHL',
    21: 'Porsche',
    22: 'Nike',
    23: 'Vodafone',
    24: 'BMW',
    25: 'McDonalds',
    26: 'HP',
    27: 'Adidas',
}

def get_id_by_brand(name):
    for id, brand in brand_map.items():
        if brand == name:
            return id
    # If the name is not found, return None
    return None

def create_label(annonations): 
    grouped_annotations = pd.DataFrame(annonations).groupby(0)
    for group in grouped_annotations:
                        filename = group.iloc[i,0]
        for i in group:
            x_min = group.iloc[i, 3]
                y_min = group.iloc[i, 4]
                x_max = group.iloc[i, 5]
                y_max = group.iloc[i, 6]
