import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import wandb
import pandas as pd
import dataloader
import models
from models import MatrixFactorization

from config import getopt


def evaluate(val_dataloader, model, criterion, opt, epoch):
    bar = tqdm(enumerate(val_dataloader), total=len(
        val_dataloader))  # Progress bar

    loss = 0.0

    model.eval()  # Set model to evaluation mode

    # Iterate over the test data and generate predictions
    for i, (encoded_user_id, encoded_song_id, rating, user_id, song_id) in bar:
        encoded_user_id = encoded_user_id.to(opt.device)
        encoded_song_id = encoded_song_id.to(opt.device)
        rating = rating.to(opt.device)

        pred_rating = model(encoded_user_id, encoded_song_id)
        loss += criterion(pred_rating, rating).item()

        # Log the first 10 predictions of the first batch to Weights & Biases
        # to visualize how well our model is doing
        if i == 0 and opt.wandb:
            n = 10
            wandb.log({"Predictions": wandb.Table(dataframe=pd.DataFrame({
                'User ID': user_id[:n],
                'Song ID': song_id[:n],
                'Pred Rating': pred_rating.cpu().detach().numpy().flatten()[:n],
                'Actual Rating': rating.cpu().detach().numpy().flatten()[:n]
            }))})

    model.train()  # Set model back to training mode

    loss /= len(val_dataloader)  # Average loss

    # Print statistics
    print("Epoch {} Validation Loss: {:.4f}".format(epoch, loss))

    # Log to Weights & Biases
    if opt.wandb:
        wandb.log({"Validation Loss": loss})

    return loss


def train(train_dataloader, model, criterion, optimizer, opt, epoch, val_dataloader=None, wandb_log=True):
    data_iterator = train_dataloader

    # Set variables for printing
    running_loss = 0.0
    dataset_size = 0.0

    # Set validations cycles to know when to evaluate during training
    val_cycle = 100
    val_multiplier = 10

    print("Outputting loss every", val_cycle, "batches")
    print("Validating every", val_cycle*val_multiplier, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(
        data_iterator))  # Progress bar

    # Iterate over the data and train the model
    for i, (encoded_user_id, encoded_song_id, rating, user_id, song_id) in bar:
        # Move data to device (GPU or CPU)
        encoded_user_id = encoded_user_id.to(opt.device)
        encoded_song_id = encoded_song_id.to(opt.device)
        rating = rating.to(opt.device)

        # Zero the parameter gradients
        # (Never forget this step! If you don't, the gradient will accumulate between epochs
        # and you'll get weird results. I know people who have spent hours debugging their
        # code only to find out they forgot this step.)
        optimizer.zero_grad()

        # Forward pass (i.e. make a prediction)
        pred_rating = model(encoded_user_id, encoded_song_id)

        # Calculate loss (i.e. how wrong our model was)
        loss = criterion(pred_rating, rating)

        # Backpropagation! (learn from our mistakes)
        loss.backward()
        optimizer.step()

        # Update running loss (only used for printing)
        batch_size = len(rating)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Print statistics and log to Weights & Biases
        if i % val_cycle == 0:
            bar.set_description("Epoch {} Loss: {:.4f}".format(
                epoch, running_loss / dataset_size))
            if wandb_log:
                wandb.log({"Training Loss": loss.item()})

        if val_dataloader is not None and i % (val_cycle*val_multiplier) == 0:
            evaluate(val_dataloader, model, criterion, opt, epoch)

    epoch_loss = running_loss / dataset_size

    return epoch_loss


if __name__ == '__main__':
    opt = getopt()

    # Set the Loss Function
    criterion = nn.MSELoss()

    # Load Data
    train_dataset = dataloader.Music(split=opt.trainset, opt=opt)
    val_dataset = dataloader.Music(split=opt.testset, opt=opt)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True)

    # Load Our Model
    model = models.MatrixFactorization(num_users=train_dataset.num_users,
                                       num_songs=train_dataset.num_songs,
                                       embedding_size=opt.embedding_size)
    model = model.to(opt.device)

    # Set the Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

    # Set the Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.step_size, gamma=0.1)

    # Test for a single epoch
    for epoch in range(1):
        train(train_dataloader, model, criterion, optimizer, opt,
              epoch, val_dataloader=val_dataloader, wandb_log=False)
