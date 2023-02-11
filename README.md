# song-recommendation-prototype

Echo Nest Taste Profile Subset - http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip

Million Song Dataset - http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz

Note: Must unzip and add MillionSongSubset Directory and train_triplets.txt within Datasets/Music Directory

----


Instructions for training the model:

1. Run msdHDF5toCSV.py (within Datasets\Music Directory)
2. Run create_user_data.py
3. Run split_csv.py
4. Run main.py
