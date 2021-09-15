import torch
import torch.nn.functional as F
from livelossplot import PlotLosses
from sklearn.metrics import roc_auc_score
import wandb


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    """
    Calculate the average roc auc scores over the classes.

    Parameters
    ----------
    actual_class: array-like
        Acutal class
    pred_class: array-like
        Prediction class
    average: str, optional
        Parameter for average calculation

    Returns
    -------
    float
        Average of the differenct classes' ROC AUC scores

    """
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]
        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(
            new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    average = sum(list(roc_auc_dict.values()))/len(roc_auc_dict)
    return average


def train(model, optimizer, criterion, data_loader, channels, device):
    """
    Train the given model with the given parameters.

    Parameters
    ----------
    model: torchvision.models
    optimizer: torch.optim.type.given_type
    criterion: torch.nn.modules.loss
    data_loader: torch.utils.data.dataloader.DataLoader
    channels: int
    device: str
        send data/tasks to GPU or CPU

    Returns
    -------
    tuple
        Tuple containing the normalised roc auc score and loss

    """
    # model put in training mode, enabling optimisation/updating of weights
    model.train()
    # instantiate training loss and accuracy
    train_loss, train_roc_aucscore = 0., 0.
    for X, y in data_loader:
        # send data to device (GPU)
        X, y = X.to(device), y.to(device)
        # resetting optimiser info
        optimizer.zero_grad()
        # forward pass
        a2 = model((X.view(-1, channels, 299, 299)).float())
        # compute loss
        loss = criterion(a2, y)
        # backpropagation - calculating gradients
        loss.backward()
        # summing for mini-batches
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        # compute accuracy
        train_roc_aucscore += roc_auc_score_multiclass(
            y.cpu().numpy(),
            y_pred.detach().cpu().numpy(),
            average="macro") * X.size(0)
        # perform a step of gradient descent
        optimizer.step()

    return \
        train_roc_aucscore / len(data_loader.dataset), \
        train_loss / len(data_loader.dataset)


def validate(model, criterion, data_loader, channels, device):
    """
    Validate the given model with the given parameters.

    Parameters
    ----------
    model: torchvision.models
    criterion: torch.nn.modules.loss
    data_loader: torch.utils.data.dataloader.DataLoader
    channels: int
    device: str
        send data/tasks to GPU or CPU

    Returns
    -------
    tuple
        Tuple containing the normalised roc auc score and loss

    """
    model.eval()
    # re-initialise validation loss and accuracy to zero
    validation_loss, roc_aucscore = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, channels, 299, 299).float())
            # calculate loss function
            loss = criterion(a2, y)
            validation_loss += loss * X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            roc_aucscore += roc_auc_score_multiclass(
                y.cpu().numpy(),
                y_pred.detach().cpu().numpy(),
                average="macro") * X.size(0)

    return \
        roc_aucscore / len(data_loader.dataset), \
        validation_loss / len(data_loader.dataset)


def train_model(model, optimizer, criterion, train_loader,
                validation_loader, channels, device, lr,
                momentum, batch_size,
                test_batch_size, n_epochs):
    """
    Train model with validation set.

    Parameters
    ----------
    model: torchvision.models
    optimizer: torch.optim.type.given_type
    criterion: torch.nn.modules.loss
    train_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    channels: int
    device: str
        send data/tasks to GPU or CPU
    lr: float
    momentum: int
    batch_size: int
    test_batch_size: int
    n_epoch: int

    Returns
    -------
    torchvision.models
        Trained model

    """

    liveloss = PlotLosses()
    for epoch in range(n_epochs):
        logs = {}
        train_accuracy, train_loss = train(
            model, optimizer, criterion, train_loader, channels, device)
        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'roc_auc_score'] = train_accuracy.item()

        validation_accuracy, validation_loss = validate(
            model, criterion, validation_loader, channels, device)
        logs['val_' + 'log loss'] = validation_loss.item()
        logs['val_' + 'roc_auc_score'] = validation_accuracy.item()

        liveloss.update(logs)
        liveloss.draw()

    return model


def train_model_wandb(model, optimizer, criterion, train_loader,
                      validation_loader, channels, device,
                      lr, momentum, batch_size, test_batch_size, n_epochs):
    """
    Train model with validation set with wandb logging.

    Parameters
    ----------
    model: torchvision.models
    optimizer: torch.optim.type.given_type
    criterion: torch.nn.modules.loss
    train_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    channels: int
    device: str
        send data/tasks to GPU or CPU
    lr: float
    momentum: int
    batch_size: int
    test_batch_size: int
    n_epoch: int

    Returns
    -------
        torchvision.models
        Trained model

    """
    wandb.watch(model, criterion)

    liveloss = PlotLosses()
    for epoch in range(n_epochs):
        logs = {}
        train_accuracy, train_loss = train(
            model, optimizer, criterion, train_loader, channels, device)
        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'roc_auc_score'] = train_accuracy.item()

        validation_accuracy, validation_loss = validate(
            model, criterion, validation_loader, channels, device)
        logs['val_' + 'log loss'] = validation_loss.item()
        logs['val_' + 'roc_auc_score'] = validation_accuracy.item()

        liveloss.update(logs)
        liveloss.draw()

        wandb_log = {
              "training_loss": train_loss.item(),
              "training_accuracy": train_accuracy.item(),
              "validation_loss": validation_loss.item(),
              "validation_accuracy": validation_accuracy.item()}

        wandb.log(wandb_log)

    return model


def train_on_full(model, optimizer, criterion, train_loader, channels, device,
                  lr, momentum, batch_size, test_batch_size, n_epochs):
    """
    Train model without validation set.

    Parameters
    ----------
    model: torchvision.models
    optimizer: torch.optim.type.given_type
    criterion: torch.nn.modules.loss
    train_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    channels: int
    device: str
        send data/tasks to GPU or CPU
    lr: float
    momentum: int
    batch_size: int
    test_batch_size: int
    n_epoch: int

    Returns
    -------
    torchvision.models
        Trained model

    """
    liveloss = PlotLosses()  # for producing the live loss plot
    for epoch in range(n_epochs):
        logs = {}
        # get the raining loss and training accuracy
        train_accuracy, train_loss = train(
            model, optimizer, criterion, train_loader, channels, device)

        # plot the trianing loss
        logs['log loss'] = train_loss.item()
        # plot the training accuracy
        logs['accuracy'] = train_accuracy.item()

        liveloss.update(logs)
        liveloss.draw()

    return model
