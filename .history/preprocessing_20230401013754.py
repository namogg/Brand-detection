import pandas as pd

annonations = pd.readcsv("E:/YOLO ultralytic/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt",)

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