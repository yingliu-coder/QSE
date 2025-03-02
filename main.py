import argparse
import yaml
import json
import numpy as np
from load.loadphysio import get_loader_physio
from load.loadsolar import get_loader_solar
from model.qse.birnn import Model
from train import train_physio, train_solar, train_traffic


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='physio')
parser.add_argument('--missing', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--bidirectional', type=bool, default=True)

args = parser.parse_args()

config_path = "config path"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

train_file = open('file path'.format(args.dataset, args.missing), 'w', encoding='utf-8')

print(json.dumps(config, indent=4))
train_file.write(json.dumps(config, indent=4) + "\n")
train_file.flush()

if args.dataset == 'physio':
    loader = get_loader_physio(
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing=args.missing,
        bidirectional=args.bidirectional,
    )
    model = Model(108).to(args.device)
    train_physio(model, config, loader, train_file, args.device)

elif args.dataset == 'solar':
    loader = get_loader_solar(
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing=args.missing,
        bidirectional=args.bidirectional,
    )
    model = Model(256).to(args.device)
    train_solar(model, config, loader, train_file, args.device)

elif args.dataset == 'traffic':
    data_input = np.load('file path'.format(args.missing), allow_pickle=True)
    model = Model(256).to(args.device)
    train_traffic(model, config, data_input, train_file, args.device)

train_file.close()

