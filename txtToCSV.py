
import pandas as pd

read_file = pd.read_csv (r'C:\Users\Abhinav\Desktop\song-recommendation-prototype\Datasets\Music\train_triplets.txt')
read_file.to_csv (r'', index=None)