mkdir caption_data
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./caption_data/
wget http://images.cocodataset.org/zips/train2014.zip -P ./caption_data/
wget http://images.cocodataset.org/zips/val2014.zip -P ./caption_data/

unzip ./caption_data/captions_train-val2014.zip -d ./caption_data/
rm ./caption_data/captions_train-val2014.zip
unzip ./caption_data/train2014.zip -d ./caption_data/
rm ./caption_data/train2014.zip 
unzip ./caption_data/val2014.zip -d ./caption_data/ 
rm ./caption_data/val2014.zip 
