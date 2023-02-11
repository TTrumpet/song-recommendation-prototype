import pandas as pd

# Load the Ratings CSV file
df = pd.read_csv('Datasets/Music/train_triplets.txt',
                 delimiter='\t', names=['User-ID', 'SongID', 'Rating'])


def create_user_data():
    # Load unique users
    users = df['User-ID'].unique()
    users = pd.DataFrame(users, columns=['User-ID'])
    users.to_csv("Datasets/Music/user_data.csv")


create_user_data()
