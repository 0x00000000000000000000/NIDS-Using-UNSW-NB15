## Network Intrusion Detection System using The UNSW-NB15 Dataset

### Dataset
Dataset was sourced from https://research.unsw.edu.au/projects/unsw-nb15-dataset. <br>
The files UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv and UNSW-NB15_4.csv were merged and successive operations were done in DataAnalysis.ipynb

### DataAnalysis.ipynb
In this notebook, we perform data cleaning, preprocessing and the analysis. The preprocessing mechanisms have been listed down in the report submitted. In our analysis, we explore the correlations of features with the output and the variances of each feature.

### Modelling.py
Within this python file, we perform additional preprocessing on the dataset and succesively perform model training for all the proposed models (Logistic Regression, Decision Tree, Random Forest Classifier, Gaussian Naive Bayes Classifier, Adaboost Classifier, XGBoost (Gradient Boosted Trees), SVM, Artificial Neural Network) and compile all the results into a single pickle file that can be loaded by pandas. All the details surrounding the modelling are mentioned in the report given in submission

### Publication
The work done has been documented and published in MDPI's Sensor's Journal under the Special Issue of Smart Mobile-Internet of Things (M-IoT).
The publication can be found here at https://www.mdpi.com/1424-8220/21/23/7835

### Citation
#### MDPI and ACS Style
Kotecha, K.; Verma, R.; Rao, P.V.; Prasad, P.; Mishra, V.K.; Badal, T.; Jain, D.; Garg, D.; Sharma, S. Enhanced Network Intrusion Detection System. Sensors 2021, 21, 7835. https://doi.org/10.3390/s21237835

#### AMA Style
Kotecha K, Verma R, Rao PV, Prasad P, Mishra VK, Badal T, Jain D, Garg D, Sharma S. Enhanced Network Intrusion Detection System. Sensors. 2021; 21(23):7835. https://doi.org/10.3390/s21237835

#### Chicago/Turabian Style
Kotecha, Ketan, Raghav Verma, Prahalad V. Rao, Priyanshu Prasad, Vipul K. Mishra, Tapas Badal, Divyansh Jain, Deepak Garg, and Shakti Sharma. 2021. "Enhanced Network Intrusion Detection System" Sensors 21, no. 23: 7835. https://doi.org/10.3390/s21237835 
