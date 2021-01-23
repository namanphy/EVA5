import torch
import matplotlib.pyplot as plt
from cuda import enable_cuda


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


def plot_results(data):
    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    fig.tight_layout()

    for idx, result in enumerate(data):
        if idx > 24:
            break

        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[row_count][idx % 5].imshow(result['image'][0], cmap='gray_r')

    plt.show()
    fig.savefig(f'incorrect_predictions.png', bbox_inches='tight')
