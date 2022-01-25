#################### INSTRUCTION #######################
# Run this file
########################################################
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import yaml

from from_sthalles_github.PyTorch_BYOL.data.multi_view_data_injector import MultiViewDataInjector
from from_sthalles_github.PyTorch_BYOL.models.mlp_head import MLPHead
from from_sthalles_github.PyTorch_BYOL.models.resnet_base_network import ResNet18
from from_sthalles_github.PyTorch_BYOL.trainer import BYOLTrainer
from step_0_specify_dataset import SSLDataset, transform_at_training

torch.manual_seed(0)


def main():
    config = yaml.load(open("./from_sthalles_github/PyTorch_BYOL/config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")


    train_dataset = SSLDataset(transform=MultiViewDataInjector([transform_at_training, transform_at_training]))


    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])
    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
