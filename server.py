from Abdullah.model import Net, test
import torch
from collections import OrderedDict
from functools import reduce
from typing import List, Tuple
import numpy as np
from flwr.common import NDArrays
from tabulate import tabulate

from Abdullah.metrics import calculate_f1_precision_recall, calculate_loss, calculate_accuracy, \
    calculate_false_alarm_rate, calculate_roc_auc

"define the server"


class Server:
    def __init__(self, num_class):
        self.model = Net(num_class)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def get_evaluate_fn(self, testloader):
        # model = Net(num_classes)
        # device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # params_dict = zip(model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, accuracy = test(self.model, testloader, self.device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, accuracy


"""Define function for global evaluation on the server."""


def broadcast_weights(sim_clt, server):
    for clt in sim_clt.values():
        clt.set_parameters(server.get_parameters())


"define the aggregation methods"


def FedAvg_fn(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def test_all_servers_fn(servers, testloader):
    """print the performance for all severs"""

    results = [["Server", "Accuracy", "Loss"]]

    for sr in servers.keys():
        loss, acc = servers[sr].get_evaluate_fn(testloader=testloader)

        results.append([sr, acc, loss])

    print("\nThe performance of each server separably:\n ", tabulate(results, headers="firstrow", tablefmt="grid"))


def weighted_voting_fn(servers, clusters, testloader, num_clients, device="cuda:0"):
    """Perform weighted voting based on the number of clients in each cluster."""

    # Calculate weights for each cluster based on the number of clients
    weights = [len(clusters[cls]) / num_clients for cls in clusters.keys()]

    # Initialize storage for the weighted sum of outputs and labels
    weighted_sums = []
    labels_list = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            batch_weighted_sum = None

            for i, (cls, srv) in enumerate(servers.items()):
                srv.model.eval()
                srv.model.to(device)

                outputs = srv.model(images)
                weight = weights[i]

                # Initialize batch_weighted_sum with the first set of weighted outputs
                if batch_weighted_sum is None:
                    batch_weighted_sum = weight * outputs
                else:
                    batch_weighted_sum += weight * outputs

            # Accumulate results
            weighted_sums.append(batch_weighted_sum)
            labels_list.append(labels)

    # Concatenate all batches to form the final weighted sum and labels tensor
    final_weighted_sum = torch.cat(weighted_sums, dim=0)
    final_labels = torch.cat(labels_list, dim=0)

    # Get the final output by finding the class with the highest weighted sum
    final_outputs = final_weighted_sum.argmax(dim=1)

    # Calculate metrics
    loss = calculate_loss(final_weighted_sum, final_labels)
    accuracy = calculate_accuracy(final_outputs, final_labels)
    f1, precision, recall = calculate_f1_precision_recall(final_outputs, final_labels)
    false_alarm_rate = calculate_false_alarm_rate(final_outputs, final_labels)
    roc_auc = calculate_roc_auc(final_weighted_sum, final_labels)

    results = [["Accuracy", "Loss", "F1-Score", "False Alarm Rate", "Roc Auc", "Precision", "Recall"],
               [accuracy, loss, f1, false_alarm_rate, roc_auc, precision, recall]]

    print("\nThe performance of the ensemble weighted voting [all servers]\n",
          tabulate(results, headers="firstrow", tablefmt="grid"))
