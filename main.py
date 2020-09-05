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

