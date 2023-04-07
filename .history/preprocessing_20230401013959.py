import pandas as pd

annonations = pd.read_csv("C:/Users/ADMIN/OneDrive - Hanoi University of Science and Technology/Tài liệu/Projects/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)

annonations = annonations.drop_duplicates(subset=[0, 1, 3, 4, 5, 6], keep='first')

