import pandas as pd 
import PLT
data = pd.read_csv("classification label.txt",sep='\s+',header=None)

classes_id = data.iloc[:,1]

# Plot the histogram
plt.hist(classes, bins=len(classes.unique()))
plt.title('Data Class Histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()