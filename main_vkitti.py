import os
import argparse
import torch
from data.vkitti import vKITTIDataset
from training_vkitti import train_depth_model
from model import loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Datasets must be in 'Dataset' folder within working directory, else the directory must be specified
    parser.add_argument('-dataset_dir', type=str, default='/HOMES/yigao/KITTI/vkitti_data/')
    parser.add_argument('-dataset', type=str, help='vkitti or nyu')
    parser.add_argument('-model', type=str, help='GuideDepth')
    parser.add_argument('-resolution', type=str, default='low', help='low or high resolution if kitti')
    parser.add_argument('-device', type=str, default='cuda:0')
    parser.add_argument('-n_epochs', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-data_dir', type=str)
    config = parser.parse_args()
    print(config)
    if config.dataset == 'vkitti':
        train_dir = os.path.join(config.dataset_dir, 'vkitti')
        valid_dir = os.path.join(config.dataset_dir, 'vkitti')
        min_depth_val = 1e-3
        max_depth_val = 80
        sf = 8000e-2
    else:
        raise NotImplementedError

    if config.dataset == 'vkitti':
        if config.resolution == 'low':
            height, width = 256, 512
        elif config.resolution == 'high':
            height, width = 288, 704
        else:
            raise NotImplementedError
    elif config.dataset == 'nyu':
        height, width = 224, 224
    else:
        raise NotImplementedError

    model_path = os.path.join(os.getcwd(), 'Models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path, config.dataset)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path, config.model + '_' + str(height) + 'x' + str(width) + '.pt')
    print('Saving model to: {}'.format(model_path))

    if config.dataset == 'vkitti':
        split_dir = os.path.join(os.getcwd(), 'Splits', config.dataset)
        train_dataset = vKITTIDataset(train_dir, src_file=os.path.join(split_dir, 'train_split.pickle'),
                                      transform='train', output_size=(height, width))
        valid_dataset = vKITTIDataset(valid_dir, src_file=os.path.join(split_dir, 'val_split.pickle'),
                                      transform='valid', output_size=(height, width))
    else:
        raise NotImplementedError

    model = loader.load_model(config.model, None)
    # model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    n_epochs = config.n_epochs
    device = torch.device(config.device)
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_depth_model(model, criterion, optimizer, scheduler, n_epochs, train_dataset, device, model_path,
                      valid_dataset, min_depth_val, max_depth_val, config.batch_size, output_sf=sf)