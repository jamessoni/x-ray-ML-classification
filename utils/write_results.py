import torch
import torch.nn.functional as F


def get_predictions(model, X, device):
    """
    Get predictions on a set of data X.

    Parameters
    ----------
    model: torchvision.models.given_model
        The model which we want to use for the predictions
    X: __main__.CustomDataSet
        The data for which we want the prediction
    device: str
        send data/tasks to GPU or CPU

    Returns
    -------
    list
        Predictions for the values
    """
    model.eval()
    y_preds = []
    for x in X:
        with torch.no_grad():
            x = x.to(device)
            a2 = model(x.view(-1, 3, 299, 299))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            y_preds.append(y_pred.item())

    return y_preds


def write_file(name, y_preds):
    """
    Savings our test data results to a .csv in the submission_csv folder.

    Parameters
    ----------
    y_preds: list
        Containing predictions to write
    name: filename
        filename + output format (e.g. .csv , .txt)

    Returns
    -------
    None
        The output is written to a file

    """
    file1 = open("submission_csvs/" + name, "w")
    file1.write('name, target\n')

    n = 0
    for pred in y_preds:
        file1.write('test_' + str(n) + ',' + str(pred) + '\n')
        n += 1

    file1.close()
