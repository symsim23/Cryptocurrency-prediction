# # import pandas as pd
# # #Adding header to dataset1
# # # headers_csv = pd.read_csv('headers.csv')
# # # header_row = headers_csv.columns
# # # dataset1 = pd.read_csv('all_crypto_half_hour.csv')
# # # dataset1.columns = header_row
# # # dataset1.to_csv('mergeA.csv', index=False)
# #
# # #Adding header to dataset2
# # # headers_csv = pd.read_csv('headers.csv')
# # # header_row = headers_csv.columns
# # # dataset2 = pd.read_csv('crypto_latest_data.csv')
# # # dataset2.columns = header_row
# # # dataset2.to_csv('mergeB.csv', index=False)
# #
# # # merging dataset1 and dataset 2 and writing to a new file
# # # data1= pd.read_csv('mergeA.csv')
# # # data2= pd.read_csv('mergeB.csv')
# # #
# # # # Both dataset now have an header and can now be merged
# # # data1= pd.read_csv('mergeA.csv')
# # # data2= pd.read_csv('mergeB.csv')
# # #
# # # New_crypto= pd.concat([data1, data2])
# # # #A new dataset has been created called new_crypto
# # # New_crypto.to_csv('new_crypto.csv', index=False)
# #
# # # dataframe = pd.read_csv("new_crypto.csv")
# # # # Print the first few rows of the DataFrame
# # # print(dataframe.head())
# # #
# # # # Get summary statistics of the numeric columns
# # # print(dataframe.describe())
# # #
# # # # Check the data types and missing values
# # # print(dataframe.info())
# # #
# # # # Checking for missing values
# # # missing_values = dataframe.isnull().sum()
# # # # Displaying missing values count
# # # print(missing_values)
# # #
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from scipy.stats import shapiro
# # #
# # # dataframe = pd.read_csv("new_crypto.csv")
# # # # Replace missing values in platform_id with 0
# # # dataframe['platform_id'].fillna(0, inplace=True)
# # #
# # # # Drop other platform-related features
# # # dataframe.drop(['platform_name', 'platform_symbol', 'platform_slug', 'platform_token_address'], axis=1, inplace=True)
# # #
# # # print(dataframe.info())
# # #
# # #
# # # # Set up plotting environment
# # # sns.set(style="ticks")
# # # fig, ax = plt.subplots()
# # # # Plot histogram of max supply
# # # sns.histplot(dataframe['max_supply'], kde=True)
# # #
# # # plt.title('Distribution of max_supply')
# # # plt.show()
# #
# # # # Perform Shapiro-Wilk test for normality
# # # stat, p = shapiro(dataframe['max_supply'])
# # # print('Shapiro-Wilk Test:')
# # # print(f'Statistic: {stat:.4f}, p-value: {p:.4f}')
# #
# #
# # import pandas as pd
# # import seaborn as sns
# #
# # # Read the dataset
# # dataframe = pd.read_csv("new_crypto.csv")
# #
# # # Subset the dataset to a smaller sample
# # sample = dataframe.sample(10000)
# #
# # # Plot the distribution of max_supply
# # sns.histplot(sample['max_supply'], kde=True)
# #
# # # Add a title to the plot
# # plt.title('Distribution of max_supply')
# #
# # # Show the plot
# # plt.show()
#
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import shapiro
#
# dataframe = pd.read_csv("new_crypto.csv")
# # Replace missing values in platform_id with 0
# dataframe['platform_id'].fillna(0, inplace=True)
#
# # Drop other platform-related features
# dataframe.drop(['platform_name', 'platform_symbol', 'platform_slug', 'platform_token_address'], axis=1, inplace=True)
#
# print(dataframe.info())
#
# import seaborn as sns
# import random
#
# # Randomly sample 10,000 observations from max_supply
# sample = dataframe['max_supply'].dropna().sample(n=10000, random_state=42)
#
# # Plot histogram using seaborn
# sns.histplot(sample)
# plt.title('Distribution of max_supply (Random Sample)')
# plt.xlabel('Max Supply')
# plt.ylabel('Count')
# plt.show()
#
# # Handling missing value for max supply feature using median imputation
# median_max_supply = dataframe['max_supply'].median()
# dataframe['max_supply'].fillna(median_max_supply, inplace=True)
# print(dataframe.info())
#
# features = ['percent_change_30d', 'percent_change_60d', 'percent_change_90d']
#
# # Compute the correlation matrix
# correlation_matrix = dataframe[features].corr()
#
# # Print the correlation matrix
# print(correlation_matrix)
#
# # Assuming 'df' is your DataFrame
# dataframe[features] = dataframe[features].interpolate()
# print(dataframe.info())
#
#
# #univariate analysis of all features
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Load the preprocessed dataset (assuming it is stored in a variable called 'df')
# # Replace 'df' with the name of your actual DataFrame
#
# # Select the relevant columns for the box plots
# selected_columns = ['price', 'volume_24h', 'percent_change_1h', 'percent_change_24h',
#                     'percent_change_7d', 'percent_change_30d', 'percent_change_60d',
#                     'percent_change_90d', 'market_cap']
#
# # Create box plots for each selected column
# plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
# sns.boxplot(data=dataframe[selected_columns])
# plt.title('Box Plot of Selected Features')
# plt.xlabel('Features')
# plt.ylabel('Values')
# plt.xticks(rotation=45)
# plt.show()
#
# # import seaborn as sns
# #
# # # Select the features for the box plots
# # selected_columns = ['price', 'volume_24h', 'percent_change_1h', 'percent_change_24h',
# #                     'percent_change_7d', 'percent_change_30d', 'percent_change_60d',
# #                     'percent_change_90d', 'market_cap']
# #
# # # Create individual box plots for each feature
# # for column in selected_columns:
# #     sns.boxplot(x=dataframe[column])
# #     plt.title(f"Box Plot of {column}")
# #     plt.show()
# #
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # Select the features for the KDE plots
# # selected_columns = ['price', 'volume_24h', 'percent_change_1h', 'percent_change_24h',
# #                     'percent_change_7d', 'percent_change_30d', 'percent_change_60d',
# #                     'percent_change_90d', 'market_cap']
# #
# # # Create individual KDE plots for each feature
# # for column in selected_columns:
# #     sns.kdeplot(data=dataframe[column], shade=True)
# #     plt.title(f"Kernel Density Estimation of {column}")
# #     plt.show()
# #
# #
# #
# # # Select the columns of interest for the histograms
# # columns_of_interest = ['id', 'num_market_pairs', 'max_supply', 'circulating_supply', 'total_supply', 'cmc_rank',
# #                        'price', 'volume_24h', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d',
# #                        'percent_change_30d', 'percent_change_60d', 'percent_change_90d', 'market_cap', 'count']
# #
# # # Plot histograms for each column
# # for column in columns_of_interest:
# #     sampled_data = dataframe[column].sample(n=10000)  # Adjust the sample size as needed
# #
# #     plt.hist(sampled_data, bins=50)  # Adjust the number of bins as needed
# #     plt.xlabel('Value')
# #     plt.ylabel('Frequency')
# #     plt.title(f'Histogram of {column}')
# #     plt.show()
#
# dataframe = pd.read_csv("new_crypto.csv")
#
# # Subset the dataset to a smaller sample
# sample = dataframe.sample(10000)
#
# # Plot the distribution of max_supply
# sns.histplot(sample['max_supply'], kde=True)
#
# # Add a title to the plot
# plt.title('Distribution of max_supply')
#
# # Show the plot
# plt.show()
#
#
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import BinaryEncoder


dataframe = pd.read_csv("new_crypto.csv")
# Checking for missing values
missing_values = dataframe.isnull().sum()
# Displaying missing values count
print(missing_values)

#Handling missing values



dataframe = pd.read_csv("new_crypto.csv")
# Replace missing values in platform_id with 0
dataframe['platform_name'].fillna(0, inplace=True)

# Drop other platform-related features
dataframe.drop(['platform_id', 'platform_symbol', 'platform_slug', 'platform_token_address','count','id','symbol','slug'], axis=1, inplace=True)

print(dataframe.head())
print(dataframe.info())



#resolving missing value in max_supply
# Randomly sample 10,000 observations from max_supply
sample = dataframe['max_supply'].dropna().sample(n=10000, random_state=42)

# Plot histogram using seaborn
sns.histplot(sample)
plt.title('Distribution of max_supply (Random Sample)')
plt.xlabel('Max Supply')
plt.ylabel('Count')
plt.show()

#due to the distribution of the plot we will use median to handle missing value
# Handling missing value for max supply feature using median imputation
median_max_supply = dataframe['max_supply'].median()
dataframe['max_supply'].fillna(median_max_supply, inplace=True)
print(dataframe.info())

#handling missing value in th percent change features.
features = ['percent_change_30d', 'percent_change_60d', 'percent_change_90d']

# Compute the correlation matrix
correlation_matrix = dataframe[features].corr()

# Print the correlation matrix
print(correlation_matrix)

# Assuming 'df' is your DataFrame
dataframe[features] = dataframe[features].interpolate()
print(dataframe.info())

# Compute descriptive statistics for numerical features
numerical_features = dataframe.select_dtypes(include='number')
statistics = numerical_features.describe()
print(statistics)




# Create an instance of the OrdinalEncoder
encoder = OrdinalEncoder()

# Fit and transform the 'cmc_rank' column
dataframe['cmc_rank'] = encoder.fit_transform(dataframe['cmc_rank'].values.reshape(-1, 1))

# View the updated dataframe
print(dataframe.head())

# Define the columns to encode
columns_to_encode = ['name', 'platform_name']

# Create an instance of the BinaryEncoder
encoder = BinaryEncoder(cols=columns_to_encode)

# Apply binary encoding to the specified columns
encoded_data = encoder.fit_transform(dataframe)

# Print the encoded data
print(encoded_data.head())

# Convert the 'date_added' feature to datetime format
encoded_data['date_added'] = pd.to_datetime(encoded_data['date_added'])
encoded_data['extracted_time'] = pd.to_datetime(encoded_data['extracted_time'])
encoded_data['last_updated'] = pd.to_datetime(encoded_data['last_updated'])

print(encoded_data.head())
