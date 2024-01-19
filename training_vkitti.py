import numpy as np
import torch
import torch.nn as nn
import time
import math
from torch.utils.data import DataLoader


def train_depth_model(model, criterion, optimizer, scheduler, n_epochs, train_dataset, device, model_dir, val_dataset,
                      MIN_DEPTH, MAX_DEPTH, batch_size, output_sf=1.0):
    model.to(device)
    loss_ges = 0
    delta1_ges = 0
    best_delta = 0
    for e in range(n_epochs):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.to(device, dtype=torch.float)
            target = target.to(device)
            output = model(input)

            loss = criterion(output, target)
            loss_ges += loss.item()
            valid_mask = ((target > 0) + (output > 0)) > 0
            output = output[valid_mask]
            target = target[valid_mask]
            maxRatio = torch.max(output / target, target / output)
            delta1 = float((maxRatio < 1.25).float().mean())
            delta1_ges += delta1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_ges = loss_ges / len(train_loader)
        delta1_ges = delta1_ges / len(train_loader)
        print('Epoch {}: Loss: {:.3f} -- Delta1: {:.3f}'.format(e, loss_ges, delta1_ges))

        loss_ges = 0
        delta1_ges = 0

        delta1, delta2, delta3, rmse, _ = test_depth_model(model, val_dataset, device, MIN_DEPTH, MAX_DEPTH,
                                                           output_sf=output_sf)
        if delta1 > best_delta:
            best_delta = delta1
            print('Saved checkpoint with validation delta1: {:.3f}'.format(delta1))
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'loss': loss_ges}, model_dir)




def test_depth_model(model, dataset, device, MIN_DEPTH, MAX_DEPTH, scaled=False, output_sf=1.0):
    rmse_ges = 0
    delta1_ges = 0
    delta2_ges = 0
    delta3_ges = 0

    model.eval()

    time_total = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():
            start = time.time()
            output = model(input)
            stop = time.time()
            time_total += (stop - start)
            output = output * output_sf

        rmse, delta1, delta2, delta3 = calc_metrics(output, target, MIN_DEPTH, MAX_DEPTH, median_scaled=scaled)

        rmse_ges += rmse
        delta1_ges += delta1
        delta2_ges += delta2
        delta3_ges += delta3

    avg_time = time_total / len(dataloader.dataset)
    avg_rmse = rmse_ges / len(dataloader.dataset)
    avg_delta1 = delta1_ges / len(dataloader.dataset)
    avg_delta2 = delta2_ges / len(dataloader.dataset)
    avg_delta3 = delta3_ges / len(dataloader.dataset)

    return avg_delta1, avg_delta2, avg_delta3, avg_rmse, avg_time


def calc_metrics(output, label, MIN_DEPTH, MAX_DEPTH, median_scaled=False):
    valid_mask = ((label > 0) + (output > 0)) > 0
    output = output[valid_mask]
    label = label[valid_mask]

    valid_mask_min = label > MIN_DEPTH
    label = label[valid_mask_min]
    output = output[valid_mask_min]

    valid_mask_max = label < MAX_DEPTH
    label = label[valid_mask_max]
    output = output[valid_mask_max]

    if median_scaled:
        scale = torch.median(label) / torch.median(output)
        output = output * scale

    output[output < MIN_DEPTH] = MIN_DEPTH
    output[output > MAX_DEPTH] = MAX_DEPTH

    mse = float(((label - output) ** 2).mean())
    rmse = math.sqrt(mse)

    maxRatio = torch.max(output / label, label / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())

    return rmse, delta1, delta2, delta3



