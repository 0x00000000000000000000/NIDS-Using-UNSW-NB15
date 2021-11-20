## Network Intrusion Detection System using The UNSW-NB15 Dataset
### DataAnalysis.ipynb
In this notebook, we perform data cleaning, preprocessing and the analysis. The preprocessing mechanisms have been listed down in the report submitted. In our analysis, we explore the correlations of features with the output and the variances of each feature.

### Modelling.py
Within this python file, we perform additional preprocessing on the dataset and succesively perform model training for all the proposed models (Logistic Regression, Decision Tree, Random Forest Classifier, Gaussian Naive Bayes Classifier, Adaboost Classifier, XGBoost (Gradient Boosted Trees), SVM, Artificial Neural Network) and compile all the results into a single pickle file that can be loaded by pandas. All the details surrounding the modelling are mentioned in the report given in submission
