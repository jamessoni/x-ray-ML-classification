import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.helper_functions import set_seed
from utils.train_tools import train_model_wandb
import torchvision.models as models


def sweep_wandb(seed, device, trainset, validset, sweep_config,
                default_hyperparameters, count=1,
                entity="4-2-convolution", project="resnet"):
    """
    Use this to run a WandB grid search and store the parameters, plots and
    outputs to the project file

    parameters
    ---------------
    seed: int
        seed for the random generators to allow repeatability
    device: str
        send data/tasks to GPU or CPU
    trainset: torch.utils.data.dataset.Subset
        training dataset
    validset: torch.utils.data.dataset.Subset
        validation dataset
    sweep_config: dict
        grid search settings
    default_hyperparameters: dict
        parameters we can vary during a grid search
    count: int
        number of runs to execute, most important for a bayes sweep
            For safety set at minimum number of grid search parameters
            if not used with bayes sweep
    entity: str
        wandb team under wihch this project belongs
    project: str
        project folder within which parameters and outputs will be stored

    returns
    ----------
    displayed log loss and roc auc score plots and values
    """
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)

    def train_fn():
        """training function call"""
        train_wandb(
            seed, device, trainset, validset, default_hyperparameters,
            entity=entity, project=project)  # change made by Raul
    wandb.agent(sweep_id, function=train_fn, count=count)


def train_wandb(seed, device, trainset, validset,
                hyperparameters, entity="4-2-convolution", project="resnet"):
    """
    wandb training setup. Calls training function and
    varies hyperparameters for grid search

    parameters
    ---------------------
    seed: int
        seed for the random generators to allow repeatability
    device: str
        send data/tasks to GPU or CPU
    trainset: torch.utils.data.dataset.Subset
        training dataset
    validset: torch.utils.data.dataset.Subset
        validation dataset
    hyperparameters: dict
        parameters we can vary during a grid search
    entity: str
        wandb team under wihch this project belongs
    project: str
        project folder within which parameters and outputs will be stored

    returns
    ------------------
    None
        Parameters, plots and data sent to wandb website

    """
    with wandb.init(config=hyperparameters, entity=entity, project=project):
        # pass to wandb.init
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config  # access all hyperparamter values
        # Allows parameter variations betwween runs
        model, optimizer, criterion, train_loader, validation_loader = \
            initialise_model(config, seed, device, trainset, validset)

        # model training
        train_model_wandb(
            model, optimizer, criterion, train_loader, validation_loader,
            config["channels"], device,
            config["learning_rate"],
            config["momentum"],
            config["batch_size"],
            config["test_batch_size"],
            config["n_epochs"])


def initialise_model(config, seed, device, trainset, validset):
    """
    Sets model parameters. parapm["parameter"] are parameters that
    can be varied through the grid search

    Parameters
    --------------------
    config: wandb.sdk.wandb_config.Config
        Gives access to hyperparameter search value set
    seed: int
        Seed for the random generators to allow repeatability
    trainset: torch.utils.data.dataset.Subset
        Training dataset
    validset: torch.utils.data.dataset.Subset
        Validation dataset

    Returns
    ----------
    model: torchvision.models.alexnet.AlexNet
        Convolutional Neural network architecture
    optimizer: torch.optim.sgd.SGD
        method to reduce losses
    criterion: torch.nn.modules.loss.CrossEntropyLoss
        criterion for which we are optimising, calculates loss.
    train_loader: torch.utils.data.dataloader.DataLoader
        data loader carrying our training data and labels
    validation_loader: torch.utils.data.dataloader.DataLoader
        validation lodaer carrying our validation data and labels
    """
    set_seed(seed)
    model_resnet = models.resnet18(pretrained=True)
    # retrained layer
    model_resnet.fc = nn.Linear(in_features=512, out_features=4)
    # instantiate model to GPU/CPU
    model = model_resnet.to(device)
    # optimiser, lr and momentum can be varied
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"])
    # define loss function
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0)  # train loader
    validation_loader = DataLoader(
        validset, batch_size=config["test_batch_size"],
        shuffle=False, num_workers=0)  # validation loader
    return model, optimizer, criterion, train_loader, validation_loader
