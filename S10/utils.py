import torch
import matplotlib.pyplot as plt
from cuda import enable_cuda
import cv2
import numpy as np


def set_seed(value=123):
    torch.manual_seed(value)


def plot_metric(results, metric):
    # Initialize a figure
    fig = plt.figure(figsize=(13, 11))

    for name, values in results.items():
        plt.plot(range(1, len(values) + 1), values, '.-', label=name)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric.lower() == 'loss' else 'lower'
    plt.legend(loc=f'{location} right',
               shadow=True,
               prop={'size': 20},
               )

    # Save plot
    plt.show()
    fig.savefig(f'{metric.lower()}.png')


def identify_misclassification(model, model_path, test_loader, limit=25):
    incorrect_samples = []

    # identifying device
    device = enable_cuda()

    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Identify misclassified images
    with torch.no_grad():
        for data, target in test_loader:
            images = data
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            # Save incorrect samples
            if len(incorrect_samples) < limit:
                for i in range(test_loader.batch_size):
                    if not list(result)[i]:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': list(images)[i]
                        })

    return incorrect_samples


def plot_results(data, classes=None):
    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    fig.tight_layout()

    for idx, result in enumerate(data):
        if idx > 24:
            break

        label = classes[result['label'].item()] if classes else result['label'].item()
        prediction = classes[result['prediction'].item()] if classes else result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[row_count][idx % 5].imshow(np.transpose(result['image'], (1, 2, 0)))

    plt.show()
    fig.savefig(f'incorrect_predictions.png', bbox_inches='tight')
