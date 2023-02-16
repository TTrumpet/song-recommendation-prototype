import pandas as pd

# Load the file
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])


def main():
    # Load unique users
    users = df['User-ID'].unique()
    users = pd.DataFrame(users, columns=['User-ID'])
    users.to_csv("Datasets/Music/user_data.csv")
