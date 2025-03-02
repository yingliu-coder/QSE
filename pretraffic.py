import numpy as np
from scipy.io import loadmat

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]
    delta = []
    for h in range(len(masks)):
        if h == 0:
            delta.append(np.ones(214))
        else:
            delta.append(np.ones(214) + (1 - masks[h]) * delta[-1])

    return np.array(delta)

def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    rec = {}
    rec['x'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['deltas'] = deltas

    return rec

def parse_id(data_line):

    evals = data_line
    shp = evals.shape

    evals = evals.reshape(-1)

    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10 * 9)

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    rec = {}
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
    return rec

data_file = loadmat('/raw/traffic/traffic.mat')['tensor']
data_np = data_file.reshape(-1,  214)[-501:-1, :]

mask = data_np != 0

filtered_values = np.where(mask, data_np, np.nan)
mean = np.nanmean(filtered_values, axis=0)
std = np.nanstd(filtered_values, axis=0)

np.save('/raw/traffic/mean.npy', mean)
np.save('/raw/traffic/std.npy', std)
data_norm = (data_np - mean) / std

total_data = [parse_id(data_norm)]

np.save('/data/liuying/tsi/process/input/traffic/traffic_missing90.npy', total_data)
