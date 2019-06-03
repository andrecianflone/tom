#!/bin/bash
# Prep dir
echo "Downloading Naive Psychology, story commonsense dataset"
cd ..
mkdir -p language/.data/stories/story_commonsense && cd language/.data/stories/story_commonsense
wget https://uwnlp.github.io/storycommonsense/data/storycommonsense_data.zip

echo "Unzipping dataset"
unzip storycommonsense_data.zip
rm storycommonsense_data.zip
