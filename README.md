# AMEX-AnalyzeThis-2019
Multi Label classification using MLP classifier
The problem statement: Can we predict the class label(High, Medium, and Low credit limit) of prospective credit card applicants using attributes of individual borrowers.

The data files contain the development_dataset.csv which is used for training and testing your model. Data_dictionary.csv has the description of the dataset. 

I used an MLP classifier with PCA to solve the problem. I compare the model accuracy with and without PCA. There is only a marginal increase (~2%) in the accuracy of the model. 

As a further exercise, one can also employ other algorithms like XGBoost and RandomForest coupled with GridSearchCV for hyper-parameter tuning to achieve further increase in the accuracy.
