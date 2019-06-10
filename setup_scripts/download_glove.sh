#!/bin/bash
# Prep dir
echo "Downloading Glove"
# $1 is something like language/.data/embeddings

# Clean dir
rm -rf $1
mkdir -p $1 && cd $1

# Download file and cleanup
mkdir temp && cd temp
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt ../
cd ../
rm -rf temp/
