"model, set_fun, get_fun, eval_fun, fit_fun"

import torch
from collections import OrderedDict
from Abdullah.model import Net, test, train


class Client:
    def __init__(self, trainloder, valloader, num_class):
        self.trainloader = trainloder
        self.valloader = valloader
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

    def fit(self, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(self.get_parameters())

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        train(self.model, self.trainloader, optim, epochs, self.device)
        # loss_val, accuracy_val = test(self.model, self.valloader, self.device)
        # print(f" loss and accuracy for client during the validation : {loss_val, accuracy_val}")
        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        # return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self):
        loss, accuracy = test(self.model, self.valloader, self.device)
        return {"Loss": loss, "Accuracy": accuracy}
