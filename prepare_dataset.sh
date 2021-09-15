#!/bin/bash

# Script to organise the dataset into the required directory structure.
# If the dataset cannot be found, an attempt will be made to download it.

# Required directory structure
dir_structure_required=(
    "xray-data"
    "xray-data/xray-data"
    "xray-data/xray-data/testouter"
    "xray-data/xray-data/testouter/test"
    "xray-data/xray-data/train"
    "xray-data/xray-data/train/0-covid"
    "xray-data/xray-data/train/1-lung_opacity"
    "xray-data/xray-data/train/2-pneumonia"
    "xray-data/xray-data/train/3-normal")

# Default directory structure
dir_structure_default=(
    "xray-data"
    "xray-data/xray-data"
    "xray-data/xray-data/test"
    "xray-data/xray-data/train"
    "xray-data/xray-data/train/covid"
    "xray-data/xray-data/train/lung_opacity"
    "xray-data/xray-data/train/normal"
    "xray-data/xray-data/train/pneumonia")

# Variable modified by the set_match_result function
match_result=false

# Function to check whether a given directory has the expected structure
set_match_result () {
    match_result=false
    local expected_structure=("$@")
    local expected_length=${#expected_structure[@]}

    local grep_output=$(ls xray-data -R | grep ^[^\.]*:$)
    local subdirectories=()
    for item in ${grep_output[@]}; do
        subdirectories+=($(echo "$item" | sed 's/://'))
    done
    local len_subdirectories=${#subdirectories[@]}

    if [ $len_subdirectories -eq $expected_length ]; then
        match_result=true
        i=0
        for subdirectory in ${subdirectories[@]}; do
            ref_subdirectory=${expected_structure[$i]}
            if [ "$subdirectory" != "$ref_subdirectory" ]; then
                match_result=false
                break
            fi
            ((i++))
        done
    else
        match_result=false
    fi
}

extracted=false
if [ ! -d "xray-data" ]; then
    # If the zip archive does not exist, attempt to download it.
    # This will fail if no Kaggle API token is present.
    if [ ! -f "acse4-ml-2020.zip" ]; then
        download_failed=false
        kaggle competitions download -c acse4-ml-2020 || download_failed=true
        if [ $download_failed == true ]; then
            echo "Unable to download dataset."
            echo "Please download the zip archive acse4-ml-2020.zip manually and retry."
            exit
        fi
    fi
    # Extract zip archive
    unzip acse4-ml-2020.zip
    extracted=true
fi

# Check if the required directory structure exists already
set_match_result ${dir_structure_required[@]}
if [ $match_result == true ]; then
    echo "The correct directory structure is already in place."
    exit
fi

# Check if the default directory structure exists
set_match_result ${dir_structure_default[@]}
if [ $match_result == true ]; then
    echo "Directory xray-data has the default structure."
else
    if [ $extracted == true ]; then
        echo "Extracted files do not have the expected directory structure."
        echo "Unable to prepare the dataset in the required form."
        echo "Please move data files into the appropriate locations manually."
    else
        echo "Directory xray-data found, but internal directory structure is not as expected."
        echo "Please delete, move or rename the existing directory and retry."
    fi
    exit
fi

# Reorganise default directory structure into the required form
echo "Reorganising directory structure..."
mkdir xray-data/xray-data/testouter
mv xray-data/xray-data/test xray-data/xray-data/testouter
mv xray-data/xray-data/train/covid xray-data/xray-data/train/0-covid
mv xray-data/xray-data/train/lung_opacity xray-data/xray-data/train/1-lung_opacity
mv xray-data/xray-data/train/pneumonia xray-data/xray-data/train/2-pneumonia
mv xray-data/xray-data/train/normal xray-data/xray-data/train/3-normal
echo "Dataset preparation complete."
