#!/bin/bash
# Prep dir
echo "Downloading Story Cloze"
# $1 is something like language/.data/stories/

# Clean dir and download
rm -rf $1
mkdir -p $1 && cd $1

# Download from gdrive
fileid="1V9G7P8xU6DKH18vCM-gsNycVSk_9SQC8"
filename="story_cloze.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Extract
echo "Extracting Story Cloze"
unzip story_cloze.zip
rm story_cloze.zip
rm cookie

