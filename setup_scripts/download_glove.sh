#!/bin/bash
# Prep dir
echo "Downloading Glove"
# $1 is something like language/.data/embeddings

# Clean dir
rm -rf $1
mkdir -p $1 && cd $1

echo "Downloading Glove, trained on 6B tokens Wiki + Gigaword"
mkdir temp && cd temp
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt ../
mv glove.6B.200d.txt ../
mv glove.6B.100d.txt ../
mv glove.6B.50d.txt ../
cd ../
rm -rf temp/

echo "Downloading Glove, trained on 840B tokens Common Crawl"
mkdir temp && cd temp
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.txt ../
cd ../
rm -rf temp/

