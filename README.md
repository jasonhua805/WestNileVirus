# **West Nile Virus Classification Model:**
West Nile Virus (WNV) first started appearing in Chicago in 2002. By 2004, the City of Chicago and the Chicago Department of Public Health (CDPH) had established a comprehensive surveillance and control program that is still in effect today.

The goal for our project is to build a model that can accurately predict the presence of West Nile Virus given weather and testing data. In doing so, the model can also help us derive insights on potential patterns that may exist in how the virus spreads. These findings will ultimately benefit the CPDH in optimizing its resources and future strategy on containing WNV.

Note that all of this data is publicly available and was provided as part of a Kaggle competition that was run in 2015 (https://www.kaggle.com/c/predict-west-nile-virus).

# Data Wrangling:
We start the process by cleaning and wrangling the two datasets we’ll eventually use to train our model. These will be referred to as the “Weather” data and the “Mosquito Trap” data respectively.

Weather Data Preview:
 ![image](https://user-images.githubusercontent.com/70826496/236658144-51d2cdba-8ffd-454d-88f8-b9533090a3a1.png)

Mosquito Trap Data Preview:
 ![image](https://user-images.githubusercontent.com/70826496/236658147-73c0e197-2dc2-4896-9663-ec11fd9bda18.png)


# Data Wrangling Part 1: Weather Data
For the columns which denote precipitation (“Snowfall” & “PrecipTotal”), there are two values we need to replace. M denotes "Missing Data" which we should replace with NaN so that it does not interfere with our ability to perform numerical analysis on these columns. T denotes "Trace" which indicates that the value is greater than zero but less than the smallest unit of measurement (0.1 inches for Snowfall and 0.01 inches for rain). We will replace "T" with a value equal to half of the smallest unit (i.e. 0.05 inches for Snowfall and 0.005 inches for rain). At this time, we can also delete our “Water1” & “Depth” columns because these respective series are empty for our dataset.

Next, we convert all of the numerical features into Float datatypes. For the “Sunrise” and “Sunset” columns, we will convert these into datetimes. Then, with our weather conditions (“CodeSum”), we split these categories into indicator variables using Panda’s get_dummies() function.

Lastly, we take the “Station” column and merge the “Station1” and “Station2” rows for each date by using the following methodology:
1.	If there are any null values in one station dataset and not the other, then the merged version will use whatever is available.
2.	All numerical values will be averaged.
3.	For the weather condition categories, we will include every observed value between the two stations (e.g. If Station 1 recorded "BR" for and Station 2 recorded "BR HZ", then the merged row's "CodeSum" will have "BR HZ")

Our final weather dataset now has 35 columns. 
Initial Weather Dataset             |  Final Weather Dataset
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/70826496/236658198-ee27e4c9-1a25-4aa5-bbc5-2ecc505410cc.png)  | ![image](https://user-images.githubusercontent.com/70826496/236658203-232cb519-feb0-4971-b2d9-01595cdd42b2.png)





# Data Wrangling Part 2: Mosquito Traps Dataset
First, we merge our cleaned weather dataframe into our mosquito trap dataframe using the “Date” column as  our ID. We then delete any of the weather condition columns which sum to 0 because this indicates that for the dates we’re working with in our mosquito trap dataset, there were no instances of those weather conditions occurring. 

Next, we use get_dummies again to separate the “Species” column into identifying variables like we did with the weather conditions.

# Exploratory Data Analysis:
We group our dataframe by date and then sum the values which allows to create a ratio to track the trend of the virus over time.

![image](https://user-images.githubusercontent.com/70826496/236658213-5b60785b-5672-44a9-acea-027172e3cf6e.png)

 
Here, we can immediately see that the virus starts to pick up in July before peaking in August and then tapering off in September and October.

Using our same aggregated dataset, this time we take the mean across each date in order to observe trends between daily temperatures and the virus.

![image](https://user-images.githubusercontent.com/70826496/236658216-fccc7c06-f4df-4d0e-823e-6f2033c594f3.png)

 
Lastly, we make use of the geopandas library to represent our data on a map of Chicago where we can see that Northwest Chicago appears to be a hotspot for the virus.

 ![image](https://user-images.githubusercontent.com/70826496/236658217-99ffa17c-9d83-4b40-bbef-6930f1e2737d.png)

# Feature Engineering:
We add the following features to our dataframe:
1.	Days of the Week
2.	Month & Year
3.	Days Since Previous Weather Condition: For each weather condition, we mark the number of days that pass between two consecutive instances of that weather condition occurring. Note that because there are two year gaps in our data (2007, 2009, 2011, 2013), we also have to take care to mark off the start of a new year with “N/A”.
4.	Municipalities: Using geocoders, we input our latitude and longitude values into Nominatim to pull up details on each of our trap locations. Municipalities was selected as the new categorical feature to be added because it had fewer null entries than other potential features while also having less variety in its actual values which helps prevent our data’s dimensionality from growing exponentially large.

All of these added features are categorical variables so we run through get_dummies again to convert them all into indicators. After this step, we address all of the missing data by imputing any null values with the mean of their respective series.

 ![image](https://user-images.githubusercontent.com/70826496/236658221-b7626062-9c11-4fbd-929e-3cef3387281f.png)


Now we can split our data into training and test sets. However, because there are an overwhelming number of “negative” WNV cases in our data, we need to sample an equal proportion of both positive and negative cases so that our model can fairly learn both (otherwise we end up with a model that will inevitably predict “negative” on almost everything). After taking this sample, we feed it into sklearn’s train_test_split.

Feature Elimination – Information Value & Multi-collinearity:
Our last step before entering the modeling stage is to reduce the dimensionality of our data by only selecting the features that will have the most impact in our model predictions. We do this by calculating the information value of each feature and then only selecting the ones which have IV > 0.01 (has some amount of predictive power) but also < 0.80 (is suspiciously high and would overrule the other features). 

We also want to reduce the multi-collinearity of our data by eliminating any features which have high collinearity with other existing features. To do this, we calculated the variance inflation factor (“VIF”) on each of these features and then removed any features which had a VIF > 5. Performing this step also helps our model’s performance because it prevents redundancy in some features which could otherwise skew our model to place more of its predictive power on dependent variables that should only be represented through one feature instead of many.

At the end of our preprocessing step, we are left with 18 features in our final dataset.





# Modeling & Results:
We built a Random Forest Model and an XG Boost Model. The performance metrics for each are presented below:

Random Forest:            |  XG Boost:
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/70826496/236658392-0ee23fd9-b564-4fda-82dd-de23c35b9dc8.png)  | ![image](https://user-images.githubusercontent.com/70826496/236658394-59ec368a-0d56-479c-892a-c85b17383b51.png)
       

Our XG Boost model does slightly better overall on accuracy, f1-score, and ROC score so it is selected as the final model for our project. However, note that Random Forest actually slightly outperformed XG Boost on recall. Future iterations of this project will likely explore new methodololgies that can improve the random forest model on the basis that true positives are the most important metric to optimize given the context of this dataset.

![image](https://user-images.githubusercontent.com/70826496/236658412-4383517f-5e41-4c0f-8996-6f8dd3720d57.png)

   
Our SHAP analysis reveals that the most impactful features in our model were as follows:
1.	August: If the data was from August, this made the model more likely to predict positive.
2.	TimeSinceLastTSRA: Generally, the longer it has been since there was last a rainy thunderstorm, the more likely the model predicts positive.
3.	Departure from the 30-year normal temperature: The more extreme the temperature, the more likely the model predicts positive. This is potentially related to why August is the peak month since that’s when the highest temperatures occur.
4.	Jefferson Township: Traps from Jefferson Township (which encompasses most of Northwest Chicago) will make the model more likely to predict positive.

Note that these features corroborate with our earlier observations made during our exploratory data analysis.

# Final Thoughts & Recommendations:
Although the CPDH has been monitoring the WNV situation since 2004, the prevalence of the virus has only continued to trend upwards year by year. Our model results produce the following recommendations on strategies to optimize the city’s efforts in combating this virus for future years:
1.	Increase coverage of spraying in the Northwest Chicago region particularly the areas encompassed in the Jefferson Township Municipality.
2.	Focus the spraying schedule to be more concentrated in August. 
3.	Add in additional out-of-schedule sprays if temperatures are noted to be particularly high for a certain day OR if at least 3 weeks have gone by without any rain or thunderstorms.

Our final model had an overall ROC score of 0.699 and a recall score of 0.75. Future efforts to improve our model will focus on trying to improve the recall since the implications of positive cases are far greater than the consequences of misclassifying negative cases as false positives.
