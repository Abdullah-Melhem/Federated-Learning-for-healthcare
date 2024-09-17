from Abdullah.model import Net, test, Net_LSTM, Net_GRU
import torch
from collections import OrderedDict
from functools import reduce
from typing import List, Tuple
import numpy as np
from flwr.common import NDArrays
from tabulate import tabulate
from torch.utils.data import DataLoader
from Abdullah.metrics import calculate_f1_precision_recall, calculate_loss, calculate_accuracy, \
    calculate_false_alarm_rate, calculate_roc_auc

"define the server"


class Server:
    def __init__(self, cls_model, num_class):
        if cls_model == 'lstm':
            self.model = Net_LSTM(num_class)
        elif cls_model == 'gru':
            self.model = Net_GRU(num_class)
        else:
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
        
        loss, accuracy = test(self.model, testloader, self.device)
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


# Example function to print the length of the test data
def print_test_data_length(testloader: DataLoader):
    total_instances = 0  # Initialize a counter for the total number of instances

    for batch_idx, (data, labels) in enumerate(testloader):
        # Get the number of instances in the current batch
        batch_size = data.size(0)

        # Increment the total count
        total_instances += batch_size

        # # Optional: Print the batch size for debugging
        # print(f"Batch {batch_idx + 1}: {batch_size} instances")

    # Print the total number of instances
    print(f"Total number of test instances: {total_instances}")


# Function to perform feature squeezing for each input data
def apply_feature_squeezing_fn(input_data: torch.Tensor, bit_depth: int = 5, smooth_alpha: float = 0.1):
    """Apply feature squeezing by quantization and smoothing."""

    # Check and ensure the input data is a PyTorch tensor
    if not isinstance(input_data, torch.Tensor):
        raise TypeError("Input data must be a PyTorch tensor")

    # Get the device of the input data to maintain computation on the same device
    device = input_data.device

    # Convert to NumPy array for manipulation
    input_data_np = input_data.cpu().numpy()  # .cpu() ensures conversion without GPU overhead

    # Quantize input data to reduce precision
    quantized_data = np.round(input_data_np * (2 ** bit_depth)) / (2 ** bit_depth)

    # Smooth the quantized data to remove noise
    smoothed_data_np = quantized_data + np.random.uniform(low=-smooth_alpha, high=smooth_alpha,
                                                          size=input_data_np.shape)

    # Convert back to PyTorch tensor and move it to the original device
    smoothed_data = torch.tensor(smoothed_data_np, dtype=torch.float32).to(device)

    return smoothed_data


# Function to determine if a record is adversarial, this function will not be used in this repo.
def is_adversarial_fn(input_data: torch.Tensor, model, feature_sq_conf):
    """Classify if a record is adversarial based on feature squeezing."""

    bit_depth = feature_sq_conf["bit_depth"]
    smooth_alpha = feature_sq_conf["smooth_alpha"]
    threshold = feature_sq_conf["threshold"]

    # Apply feature squeezing
    squeezed_data = apply_feature_squeezing_fn(input_data, bit_depth, smooth_alpha)

    # Make predictions on original and squeezed data
    original_output = model(input_data)
    squeezed_output = model(squeezed_data)

    # Check if predictions are different
    original_pred = original_output.argmax(dim=1)
    squeezed_pred = squeezed_output.argmax(dim=1)

    if threshold == 0:
        # An instance is considered adversarial if predictions are not the same
        # Return a boolean tensor indicating which instances are adversarial
        return original_pred != squeezed_pred
    else:
        difference = torch.abs(original_output - squeezed_output)
        # Determine if the difference exceeds the threshold
        # An instance is considered adversarial if the maximum difference exceeds the threshold
        return (difference.max(dim=1).values > threshold)


def weighted_voting_fn(servers, clusters, testloader, num_clients, feature_sq_conf, device="cuda:0"):
    """Perform weighted voting based on the number of clients in each cluster."""
    print_test_data_length(testloader)
    # Calculate weights for each cluster based on the number of clients
    weights = [len(clusters[cls]) / num_clients for cls in clusters.keys()]

    # Initialize storage for the weighted sum of outputs and labels
    weighted_sums = []
    labels_list = []
    adversarial_indices = []  # List to store indices of adversarial records

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            record, labels = data[0].to(device), data[1].to(device)

            # Initialize a list to hold non-adversarial instances in the batch
            non_adversarial_records = []
            non_adversarial_labels = []
            batch_adversarial_indices = []  # Track adversarial indices within the batch

            # Process each record individually to determine adversarial instances
            for idx in range(record.size(0)):
                is_adversarial = False  # Flag for current record
                for i, (cls, srv) in enumerate(servers.items()):
                    srv.model.eval()
                    srv.model.to(device)

                    # Check if the current record is adversarial
                    if is_adversarial_fn(input_data=record[idx].unsqueeze(0), model=srv.model,
                                         feature_sq_conf=feature_sq_conf).item():
                        is_adversarial = True
                        break

                # If the record is not adversarial, keep it for further processing
                if not is_adversarial:
                    non_adversarial_records.append(record[idx].unsqueeze(0))
                    non_adversarial_labels.append(labels[idx].unsqueeze(0))
                else:
                    batch_adversarial_indices.append(batch_idx * testloader.batch_size + idx)
                    print(f"Record index {batch_idx * testloader.batch_size + idx} is adversarial and will be removed.")

            # If there are non-adversarial records, perform weighted voting
            if non_adversarial_records:
                # Stack non-adversarial records into a tensor
                non_adversarial_records = torch.cat(non_adversarial_records, dim=0)
                non_adversarial_labels = torch.cat(non_adversarial_labels, dim=0)

                # Calculate weighted sum for non-adversarial instances
                batch_weighted_sum = None

                for i, (cls, srv) in enumerate(servers.items()):
                    srv.model.eval()
                    srv.model.to(device)
                    outputs = srv.model(non_adversarial_records)
                    weight = weights[i]

                    # Initialize batch_weighted_sum with the first set of weighted outputs
                    if batch_weighted_sum is None:
                        batch_weighted_sum = weight * outputs
                    else:
                        batch_weighted_sum += weight * outputs

                # Accumulate results
                weighted_sums.append(batch_weighted_sum)
                # Actual labels
                labels_list.append(non_adversarial_labels)

            # Extend global adversarial indices list
            adversarial_indices.extend(batch_adversarial_indices)

    # Concatenate all non-adversarial batches to form the final weighted sum and labels tensor
    if weighted_sums:  # Check if there are any valid records left
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
    else:
        print("No valid records left for evaluation after removing adversarial instances.")

    print(f"\nNumber of adversarial records: {len(adversarial_indices)}\n")
    print(f"\nIndices of adversarial records: {adversarial_indices}")
