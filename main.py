# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:48:22 2020

@author: RAHUL
"""

# setup -----------------------------------------------------------------------
import sys
import sklearn
import os
import tarfile
import urllib
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# where to save the figures
PROJECT_ROOT_DIR = r"C:\Users\RAHUL\Documents\California-Housing-Price"
PROJECT_ID = "housing_price_prediction"
IMAGE_PATH = os.path.join(PROJECT_ROOT_DIR, 'image', PROJECT_ID)
os.makedirs(IMAGE_PATH,exist_ok=True)

# function for saving figure
def save_fig(fig_id, tight_layout=True, fig_extension='png',resolution=300):
    path = os.path.join(IMAGE_PATH, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
  
# eliminate useless warnings
import warnings 
warnings.filterwarnings(action='ignore', message='^internal gelsd')  

# to make identical output every time
np.random.seed(42)

# import data -----------------------------------------------------------------
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
 
# function for fetching data from url
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()  

fetch_housing_data()
    
# loading dataset to pandas dataframe
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

# get the data information
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
describe = housing.describe(include='all')

housing.hist(bins=50, figsize=(20, 15))
save_fig('attribute_histogram_plots')
plt.show()

# preprocessing dataset -------------------------------------------------------
housing['median_income'].hist()
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1, 2, 3, 4, 5])
housing['income_cat'].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
    
# discover and visulize data to gain insight ----------------------------------
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude')
save_fig('bad_visualization_plot')    

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
save_fig('better_visualization_plot')

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5,
             s=housing['population']/100, label='population', figsize=(10, 7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
             sharex=False)
plt.legend()
save_fig('housing_price_scatter_plot')

# download the california image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

california_img = mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind='scatter', x='longitude', y='latitude', figsize=(10, 7),
                  s=housing['population']/100, label='population',
                  c='median_house_value', cmap=plt.get_cmap('jet'),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.28, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap('jet'))
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

prices = housing['median_house_value']
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(['$%dk'%(round(v/1000)) for v in tick_values],
                        fontsize=14)
cbar.set_label('Median house value', fontsize=16)

plt.legend(fontsize=16)
save_fig('California_housing_price_plot')
plt.show()

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['median_house_value', 'median_income', 'total_rooms',
              'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 12))

housing.plot(kind='scatter', x='median_income', y='median_house_value',
              alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig('income_vs_house_value_scatterplot')

housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

housing.plot(kind='scatter', x='rooms_per_household', y='median_house_value',
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()

housing.describe()

# prepare data for machine learning algorithms --------------------------------
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_vaues'].copy()