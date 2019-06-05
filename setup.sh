#!/bin/bash

# This dir
root_dir=$(pwd)

# Make sure shell scripts are executable
find $rood_dir -type f -name "*.sh" -exec chmod 744 {} \;

# Install some dependencies
pip install -U spacy
python -m spacy download en
python -m spacy download de
pip install -r requirements.txt

echo "Setting up Story Cloze dataset"
cd $root_dir
bash setup_scripts/download_story_cloze.sh $root_dir/language/.data/stories/

echo "Setting up Naive Psychology dataset"
cd $root_dir
naive_dir=$root_dir/language/.data/stories/story_commonsense
bash setup_scripts/download_storycommonsense.sh $naive_dir

echo "Setting up Naive psychology dataset, this may take a minute"
cd $root_dir
naive_target=$naive_dir/torchtext
python language/data.py --create_naive --commonsense_location $naive_dir --commonsense_target $naive_target


