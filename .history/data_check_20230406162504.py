import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv("classification label.txt",sep='\s+',header=None)
images_dir = d
classes_id = data.iloc[:,1]

# Plot the histogram
plt.hist(classes_id, bins=len(classes_id.unique()))
plt.title('Data Class Histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


