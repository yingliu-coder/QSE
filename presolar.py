import numpy as np

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]
    delta = []
    for h in range(len(masks)):
        if h == 0:
            delta.append(np.ones(137))
        else:
            delta.append(np.ones(137) + (1 - masks[h]) * delta[-1])

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

data_file = np.loadtxt('../../raw/solar/solar.txt', delimiter=',')
data_np = data_file.T
data_np = data_np.reshape(-1, 137)

mask = data_np != 0

filtered_values = np.where(mask, data_np, np.nan)
mean = np.nanmean(filtered_values, axis=0)
std = np.nanstd(filtered_values, axis=0)

np.save('../../raw/solar/mean.npy', mean)
np.save('../../raw/solar/std.npy', std)
data_norm = (data_np - mean) / std
data_norm = data_norm.reshape(-1, 36, 137)
total_data = []

for line in data_norm:
    rec = parse_id(line)
    total_data.append(rec)

np.save('../input/solar/solar_missing50.npy', total_data)
