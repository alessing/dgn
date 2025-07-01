import random
import argparse
import os
import h5py

def merge_datasets(datasets, output, nums = None, data_types = None, seed=None, random_eps=False):

    print("Creating", output)

    files = {}
    demos = []
    for i, dataset_fname in enumerate(datasets):
        files[dataset_fname] = h5py.File(dataset_fname, 'r')
        if nums is not None:
            num = nums[i]
            num = min(len(files[dataset_fname]['data'].keys()), num)
        else:
            num = len(files[dataset_fname]['data'].keys()) # all demos in dataset
        data_type = None if data_types is None else data_types[i]

        for n in range(num):
            demos.append((dataset_fname, n, data_type))

    random.shuffle(demos)

    os.makedirs(os.path.dirname(output), exist_ok=True)

    new_idx = 0
    with h5py.File(output, 'w') as f_new:
        new_demos = f_new.create_group('data')

        new_demos.attrs['args.datasets'] = str(datasets)
        new_demos.attrs['args.output'] = output
        new_demos.attrs['args.nums'] = str(nums)
        new_demos.attrs['args.nums'] = str(nums)
        new_demos.attrs['args.seed'] = str(seed)

        for demo in demos:
            episode_key = 'demo' if 'data/demo_0' in files[demo[0]] else 'rollout'
            ep_index = demo[1]

            files[demo[0]]['data'].copy(files[demo[0]][f'data/{episode_key}_{ep_index}'], new_demos, name=f'demo_{new_idx}')
            if demo[2] is not None:
                new_demos[f'demo_{new_idx}'].attrs['orig_id'] = str(demo[2]) + "," + str(ep_index)
            
            new_idx += 1

        for f_old in files.values():
            for k in f_old['data'].attrs.keys():
                f_new['data'].attrs[k] = f_old['data'].attrs[k]

    for dataset_fname in datasets:
        files[dataset_fname].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--nums', nargs='+', type=int)
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_types', nargs='+', type=str)
    
    merge_datasets(**vars(parser.parse_args()))
