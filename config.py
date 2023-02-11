import argparse
import torch
import multiprocessing

# We will use this file to set configuration options for our project (i.e. hyperparameters,
# dataset paths, training options, logging options, etc.)


def getopt():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.resources = "./"

    opt.n_epochs = 200  # Number of epochs to train for (200)

    # Description of the experiment in Weights & Biases. Feel free to change this to
    # whatever you want each time you run an experiment.
    opt.description = 'Matrix Factorization 32-Dim'
    opt.archname = 'MF'

    opt.wandb = True  # Whether or not to log to Weights & Biases

    opt.evaluate = False  # Only if you want to evaluate a model and not train it

    opt.embedding_size = 32  # Size of the embedding vectors for Users and Books

    opt.lr = 0.1  # Learning Rate
    opt.step_size = 64  # After how many epochs to decay the learning rate by gamma
    opt.batch_size = 128  # Batch Size (i.e number of samples per batch)

    opt.trainset = 'train'
    opt.testset = 'test'

    # Check if we can use a GPU
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    # You can keep adding more options here if you want to. Having them all in one place
    # makes it easier to keep track of them and easier to change them without having to
    # do so in multiple places of your code.

    return opt
