"model, set_fun, get_fun, eval_fun, fit_fun"

import torch
from collections import OrderedDict
from Abdullah.model import Net, test, train, Net_LSTM, Net_GRU


class Client:
    def __init__(self, clt_model, trainloder, valloader, num_class):
        if clt_model == 'lstm':
            self.model = Net_LSTM(num_class)
        elif clt_model == 'gru':
            self.model = Net_GRU(num_class)
        else:
            self.model = Net(num_class)
        self.trainloader = trainloder
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(self.get_parameters())

   
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

       
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

       
        train(self.model, self.trainloader, optim, epochs, self.device)
        

    def evaluate(self):
        loss, accuracy = test(self.model, self.valloader, self.device)
        return {"Loss": loss, "Accuracy": accuracy}


def find_cluster_for_client(clusters, client):
    for cluster, clients in clusters.items():
        if client in clients:
            return cluster
    return None
