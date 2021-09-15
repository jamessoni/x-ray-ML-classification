# ACSE 4 Machine Learning Group Project

## X-Ray Classification

All information available here: https://www.kaggle.com/c/acse4-ml-2020

Area Under the ROC Curve

## Setup

### Installing required python packages

    pip install -r requirements.txt

### Preparing the dataset

The dataset can be downloaded from https://www.kaggle.com/c/acse4-ml-2020/data. The zip archive `acse4-ml-2020.zip` should be saved in the root directory of the project.

On Linux, the dataset can be extracted and reorganised into the required directory structure as follows:

    chmod +x prepare_dataset.sh
    ./prepare_dataset.sh

If it is not possible to run the `prepare_dataset.sh` script, the data must be manually organised into the following directory structure:

    xray-data
    └── xray-data
        ├── testouter
        │   └── test
        └── train
            ├── 0-covid
            ├── 1-lung_opacity
            ├── 2-pneumonia
            └── 3-normal

## Repository Structure and content

Our repository contains two jupyter notebook folders, these detail our best two attempts at classifying the un-labeled dataset available from https://www.kaggle.com/c/acse4-ml-2020. We have tried to maintain a similar structure and content between these notebooks, furthermore this structure should act as a good starting point for future testing allowing users to easily trian models and tune for hyperparameters.

The structure of these notebooks is enabled by the highly modular nature of our code. Many helper functions and wrappers have been encapsulated and well documented within their respective .py files which allow us to quickly change certain functionalities whilst keeping our jupyter notebooks clean and easily readable. This modularity has also allowed us to implement unit-testing more easily. All of our extra modules are located in the /utils subdirectory. For detailed documentation, please see [documentation/html/index.html](https://github.com/acse-2020/acse-4-x-ray-classification-convolution/blob/development/documentation/html/index.html), which can be opened in a web browser.
