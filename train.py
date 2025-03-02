import torch.optim as optim
import torch
import time
import numpy as np


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def quantile_loss_single(target, forecast, q: float, mask):
    return 2 * torch.sum(
        torch.abs((forecast - target) * mask * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, mask):
    return torch.sum(torch.abs(target * mask))

def calc_quantile_CRPS(target, forecast, mask):

    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    denom = calc_denominator(target, mask)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss_single(target, q_pred, quantiles[i], mask)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def train_physio(model, config, loader, train_file, device):
    duration_train = 0
    duration_test = 0
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    for epoch in range(config["train"]["epochs"]):
        train_start = time.time()
        train_num_batch = 0
        train_total_loss = 0.0
        model.train()
        for idx, (x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b) in enumerate(loader):
            optimizer.zero_grad()
            x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b = x_f.to(device), masks_f.to(device), evals_f.to(device), eval_masks_f.to(device), deltas_f.to(device), x_b.to(device), masks_b.to(device), evals_b.to(device), eval_masks_b.to(device), deltas_b.to(device)
            output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_total_loss += loss.item()
            train_num_batch += 1
        print('epoch {}, train loss {:.4f}'.format(epoch, train_total_loss / train_num_batch))
        train_file.write('epoch {}, train loss {:.4f} '.format(epoch, train_total_loss / train_num_batch))
        duration_train += time.time() - train_start
        test_start = time.time()
        model.eval()
        with torch.no_grad():
            test_num_batch = 0
            test_mae = 0.0
            test_mse = 0.0
            test_crps = 0.0
            for idx, (x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b) in enumerate(loader):
                x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b = x_f.to(
                    device), masks_f.to(device), evals_f.to(device), eval_masks_f.to(device), deltas_f.to(
                    device), x_b.to(device), masks_b.to(device), evals_b.to(device), eval_masks_b.to(
                    device), deltas_b.to(device)
                output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
                imputation_mae = masked_mae_cal(output['imputations'], evals_f, eval_masks_f)
                imputation_mse = masked_mse_cal(output['imputations'], evals_f, eval_masks_f)
                imputation_crps = calc_quantile_CRPS(evals_f, output['ensembles'], eval_masks_f)
                test_mae += imputation_mae
                test_mse += imputation_mse
                test_crps += imputation_crps
                test_num_batch += 1
            print('mae {:.4f}, mse {:.4f}, crps {:.4f}'.format(test_mae / test_num_batch, test_mse / test_num_batch, test_crps / test_num_batch))
            train_file.write('mae {:.4f}, mse {:.4f}, crps {:.4f}\n'.format(test_mae / test_num_batch, test_mse / test_num_batch, test_crps / test_num_batch))
        duration_test += time.time() - test_start
        train_file.flush()
    print("train: {} hours, test: {} hours".format(duration_train / 3600, duration_test / 3600))
    train_file.write("train: {} hours, test: {} hours\n".format(duration_train / 3600, duration_test / 3600))
    train_file.flush()

def train_solar(model, config, loader, train_file, device):
    duration_train = 0
    duration_test = 0
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    for epoch in range(config["train"]["epochs"]):
        train_start = time.time()
        train_num_batch = 0
        train_total_loss = 0.0
        model.train()
        for idx, (x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b) in enumerate(loader):
            optimizer.zero_grad()
            x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b = x_f.to(device), masks_f.to(device), evals_f.to(device), eval_masks_f.to(device), deltas_f.to(device), x_b.to(device), masks_b.to(device), evals_b.to(device), eval_masks_b.to(device), deltas_b.to(device)
            output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_total_loss += loss.item()
            train_num_batch += 1
        print('epoch {}, train loss {:.4f}'.format(epoch, train_total_loss / train_num_batch))
        train_file.write('epoch {}, train loss {:.4f} '.format(epoch, train_total_loss / train_num_batch))
        duration_train += time.time() - train_start
        test_start = time.time()
        mean = torch.from_numpy(np.load("mean.npy")).to(device)
        std = torch.from_numpy(np.load("std.npy")).to(device)
        model.eval()
        with torch.no_grad():
            test_num_batch = 0
            test_mae = 0.0
            test_mse = 0.0
            test_crps = 0.0
            for idx, (x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b) in enumerate(loader):
                x_f, masks_f, evals_f, eval_masks_f, deltas_f, x_b, masks_b, evals_b, eval_masks_b, deltas_b = x_f.to(
                    device), masks_f.to(device), evals_f.to(device), eval_masks_f.to(device), deltas_f.to(
                    device), x_b.to(device), masks_b.to(device), evals_b.to(device), eval_masks_b.to(
                    device), deltas_b.to(device)
                output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
                imputations_norm = output['imputations'] * std + mean
                ensembles_norm = output['ensembles'] * std + mean
                evals_f_norm = evals_f * std + mean
                imputation_mae = masked_mae_cal(imputations_norm, evals_f_norm, eval_masks_f)
                imputation_mse = masked_mse_cal(imputations_norm, evals_f_norm, eval_masks_f)
                imputation_crps = calc_quantile_CRPS(evals_f_norm, ensembles_norm, eval_masks_f)
                test_mae += imputation_mae
                test_mse += imputation_mse
                test_crps += imputation_crps
                test_num_batch += 1
            print('mae {:.4f}, mse {:.4f}, crps {:.4f}'.format(test_mae / test_num_batch, test_mse / test_num_batch, test_crps / test_num_batch))
            train_file.write('mae {:.4f}, mse {:.4f}, crps {:.4f}\n'.format(test_mae / test_num_batch, test_mse / test_num_batch, test_crps / test_num_batch))
        duration_test += time.time() - test_start
        train_file.flush()
    print("train: {} hours, test: {} hours".format(duration_train / 3600, duration_test / 3600))
    train_file.write("train: {} hours, test: {} hours\n".format(duration_train / 3600, duration_test / 3600))
    train_file.flush()

def train_traffic(model, config, data_input, train_file, device):
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    data_forward, data_backward = data_input[0]['forward'], data_input[0]['backward']
    x_f, masks_f, evals_f, eval_masks_f, deltas_f = torch.FloatTensor(data_forward['x']).to(device).unsqueeze(0), torch.FloatTensor(data_forward['masks']).to(device).unsqueeze(0), torch.FloatTensor(data_forward['evals']).to(device).unsqueeze(0), torch.FloatTensor(data_forward['eval_masks']).to(device).unsqueeze(0), torch.FloatTensor(data_forward['deltas']).to(device).unsqueeze(0)
    x_b, masks_b, evals_b, eval_masks_b, deltas_b = torch.FloatTensor(data_backward['x']).to(device).unsqueeze(0), torch.FloatTensor(data_backward['masks']).to(device).unsqueeze(0), torch.FloatTensor(data_backward['evals']).to(device).unsqueeze(0), torch.FloatTensor(data_backward['eval_masks']).to(device).unsqueeze(0), torch.FloatTensor(data_backward['deltas']).to(device).unsqueeze(0)
    mean = torch.from_numpy(np.load("mean.npy")).to(device)
    std = torch.from_numpy(np.load("std.npy")).to(device)
    for epoch in range(config["train"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
        loss = output['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        print('epoch {}, train loss {:.4f}'.format(epoch, loss))
        train_file.write('epoch {}, train loss {:.4f} '.format(epoch, loss))
        model.eval()
        with torch.no_grad():
            output = model(x_f, masks_f, deltas_f, x_b, masks_b, deltas_b)
            imputations_norm = output['imputations'] * std + mean
            ensembles_norm = output['ensembles'] * std + mean
            evals_f_norm = evals_f * std + mean
            imputation_mae = masked_mae_cal(imputations_norm, evals_f_norm, eval_masks_f)
            imputation_mse = masked_mse_cal(imputations_norm, evals_f_norm, eval_masks_f)
            imputation_crps = calc_quantile_CRPS(evals_f_norm, ensembles_norm, eval_masks_f)
            print('mae {:.4f}, mse {:.4f}, crps {:.4f}'.format(imputation_mae, imputation_mse, imputation_crps))
            train_file.write('mae {:.4f}, mse {:.4f}, crps {:.4f}\n'.format(imputation_mae, imputation_mse, imputation_crps))
        train_file.flush()
