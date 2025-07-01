import random
import argparse
import os
import h5py

def merge_datasets(dataset, output, seed=None, only_worse=False):

    if seed is not None:
        random.seed(seed)

    print("Creating", output)

    with h5py.File(dataset, 'r') as old_f:

        if only_worse:
            worse_names = old_f['mask']['worse'][:]
            worse_idxs = []
            for wn in worse_names:
                wn = wn.decode()
                wi = wn.split('_')[-1]
                wi = int(wi)
                worse_idxs.append(wi)
            permutation = worse_idxs.copy()
            num_demos = len(permutation)
        else:
            num_demos = len(old_f['data'].keys())
            permutation = list(range(num_demos))


        random.shuffle(permutation)

        with h5py.File(output, 'w') as f_new:

            new_demos = f_new.create_group('data')

            new_demos.attrs['args.dataset'] = str(dataset)
            new_demos.attrs['args.output'] = output
            new_demos.attrs['args.seed'] = str(seed)

            for new_idx in range(num_demos):
                old_idx = permutation[new_idx]

                old_f['data'].copy(old_f[f'data/demo_{old_idx}'], new_demos, name=f'demo_{new_idx}')
                new_demos[f'demo_{new_idx}'].attrs['old_idx'] = str(old_idx)

            for k in old_f['data'].attrs.keys():
                f_new['data'].attrs[k] = old_f['data'].attrs[k]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only_worse', action='store_true')
    
    merge_datasets(**vars(parser.parse_args()))
