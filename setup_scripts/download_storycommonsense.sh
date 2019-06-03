#!/bin/bash
# Prep dir
echo "Downloading Naive Psychology, story commonsense dataset"
# $1 is something like language/.data/stories/story_commonsense

# Clean directory and download
rm -rf $1
mkdir -p $1 && cd $1
wget https://uwnlp.github.io/storycommonsense/data/storycommonsense_data.zip

echo "Unzipping dataset"
unzip storycommonsense_data.zip
rm storycommonsense_data.zip
