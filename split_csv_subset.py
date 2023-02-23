import pandas as pd
import sys

# edit this code to remove subset of users rather than just a subset of rows


def numAbbrev(num):
    """Given a number, return a string with the number abbreviated.

    Args:
        num (int): The number to abbreviate.
    """
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return str(int(num/1000)) + 'K'
    elif num < 1000000000:
        return str(int(num/1000000)) + 'M'
    else:
        return str(int(num/1000000000)) + 'B'


def getSampleCSV(csv_file, sample_size):
    """Given the name of a CSV file, save a random sample from it  and save
    it to a new CSV file.

    Args:
        csv_file (str): The name of the CSV file to sample.
        sample_size (int): The number of rows to sample.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        df = df.sample(sample_size)

        # Save the DataFrame to a new CSV file
        newCSVname = csv_file.split(
            '.')[0] + "N" + numAbbrev(sample_size) + ".csv"
        df.to_csv(newCSVname, index=False)
        print('Saved ' + newCSVname)

    except Exception as e:
        print('Error: ' + str(e))


if __name__ == '__main__':
    getSampleCSV(sys.argv[1], int(sys.argv[2]))
    # e.g. >>> python split_csv.py Datasets/Books2/BX-Book-Ratings.csv 1000
