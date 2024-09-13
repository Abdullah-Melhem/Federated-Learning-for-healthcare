import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from Abdullah.dataset import prepare_dataset
from Abdullah.client import Client
from Abdullah.server import Server, broadcast_weights, FedAvg_fn
from tqdm import tqdm


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """this is the parameters required for the training process"""
    print(OmegaConf.to_yaml(cfg))

    """prepare the data loader"""
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size, cfg.val_ratio, cfg.data_path
    )

    """prepare the clients"""
    sim_clt = {}
    for clt in range(cfg.num_clients):
        clt_name = cfg.target_clt + "-" + str(clt)
        sim_clt[clt_name] = Client(trainloder=trainloaders[clt], valloader=validationloaders[clt],
                                   num_class=cfg.num_class)

    print(f"The {cfg.num_clients} participants names:", sim_clt.keys())
    """prepare the server"""
    server = Server(cfg.num_class)

    for round in range(cfg.number_of_rounds):
        """ initialize and broadcast  the weights"""
        broadcast_weights(sim_clt, server)

        """start the simulation"""
        for clt in sim_clt.values():
            clt.fit(config=cfg.tr_confg)

        """Apply FedAvg """
        all_clt = []
        for cl in sim_clt.values():
            all_clt.append((cl.get_parameters(), len(cl.trainloader)))

        server.set_parameters(FedAvg_fn(all_clt))

        """Evaluate the server"""

        loss, acc = server.get_evaluate_fn(testloader=testloader)

        print(f"Training ROUND {round}: Accuracy: {acc} \t Loss: {loss}")


if __name__ == '__main__':
    main()
