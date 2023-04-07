import pandas as pd 

data = pd.read_csv("classification label.txt",sep='\s+',header=None)

classes_id = data.iloc[:,1]
print(classes_id)