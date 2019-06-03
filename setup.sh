#!/bin/bash

# This dir
root_dir=$(pwd)

# Install some dependency
pip install -U spacy
python -m spacy download en
python -m spacy download de

# Download Story Cloze
echo "Story Cloze dataset"
cd $root_dir
bash setup_scripts/download_story_cloze.sh $root_dir/language/.data/stories/

# Download Naive Psychology Common sense dataset
echo "Naive Psychology dataset"
cd $root_dir
naive_dir=$root_dir/language/.data/stories/story_commonsense
bash setup_scripts/download_storycommonsense.sh $naive_dir

# Prep naive psych dataset
echo "Creating pytext-ready files, this may take a minute"
cd $root_dir
naive_target=$naive_dir/torchtext
python language/data.py --create_naive --commonsense_location $naive_dir --commonsense_target $naive_target


