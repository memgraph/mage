from typing import List, Dict
import matplotlib.pyplot as plt


def visualize(training_results: List[Dict[str, float]], test_results: List[Dict[str, float]]) -> None:
    """Visualize training results obtained during ML training. It assumes that training and test results are of same size=validated after same number of epochs.
    Goal is to create drawing function that draws at most 4 subplots at the same figure and is dynamically adapting to the number of metrics sent.

    Args:
        training_results (List[Dict[str, float]]): Results obtained on training set. Contains various metrics for training.
        test_results (List[Dict[str, float]]): Results obtained on test set.
    """
    tr_metrics, test_metrics = dict(), dict()
    epochs = []

    for epoch in range(len(training_results)):
        epochs.append(epoch)
        epoch_training_result = training_results[epoch]
        epoch_test_result = test_results[epoch]

        # Assume that training and test results have same metrics
        for metric_name in epoch_training_result.keys():
            if metric_name == "epoch":
                continue
            if metric_name not in tr_metrics:
                tr_metrics[metric_name] = []
                test_metrics[metric_name] = []
            tr_metrics[metric_name].append(epoch_training_result[metric_name])
            test_metrics[metric_name].append(epoch_test_result[metric_name])

    subplot_labels = [221, 222, 223, 224]
    fig = plt.figure()
    i = 0
    for metric_name in tr_metrics.keys():
        if metric_name == "epoch":
            continue
        print(metric_name)
        plt.subplot(subplot_labels[i])
        plt.title(metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.plot(epochs, tr_metrics[metric_name], label="TRAIN")
        plt.plot(epochs, test_metrics[metric_name], label="TEST")
        plt.legend(loc='upper right')
        i += 1
        if i == 4:
            plt.suptitle("Training metrics")
            plt.show()
            plt.clf()
            plt.suptitle("Training metrics")
            i = 0
