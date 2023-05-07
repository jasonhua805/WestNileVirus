#West Nile Virus Classification Model
This project aims to predict the presence of West Nile Virus in mosquitos across the city of Chicago. 

#Data

The data used in this project is provided by the City of Chicago Department of Public Health and can be found in the data directory. The dataset contains information on mosquito traps, weather, and West Nile Virus cases from 2007 to 2014.

#Notebooks

The notebooks directory contains Jupyter notebooks that walk through the data exploration, feature engineering, and model building process. The notebooks are organized as follows:

01_data_exploration.ipynb: This notebook explores the data and visualizes the distribution of West Nile Virus cases across the city of Chicago.
02_feature_engineering.ipynb: This notebook performs feature engineering on the dataset, including creating new features and encoding categorical variables.
03_model_building.ipynb: This notebook builds and evaluates predictive models for West Nile Virus presence.

#Results

The best performing model achieved an AUC score of 0.80 on the test set. This model was an ensemble of gradient boosting and random forest classifiers, and it used a combination of weather and mosquito trap features to predict West Nile Virus presence.

#Conclusion

This project demonstrates the use of machine learning techniques to predict the presence of West Nile Virus in mosquitos across the city of Chicago. The best performing model achieved an AUC score of 0.80 on the test set, indicating that it has potential for use in real-world applications.
