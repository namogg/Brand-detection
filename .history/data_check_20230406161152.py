import pandas as pd 

data = pd.read_csv("classification label.txt",sep='\s+',header=None)

classes_id = data.iloc[:,1]

vals <- seq_along(freq) - 1

# plot a histogram
hist(vals, breaks = length(vals), freq = TRUE, col = "steelblue",
     main = "Histogram", xlab = "Values", ylab = "Frequency")
print(classes_id)