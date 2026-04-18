import os
import argparse
import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets, steps_per_day, add_time_in_day=True, add_day_in_week=False):
    num_samples, num_nodes, feat_dim = data.shape   # (seq_length, num_nodes, feat_dim)
    print(num_samples, num_nodes, feat_dim)
    data_list = [data]
    if add_time_in_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(tod_tiled)
    
    if add_day_in_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(dow_tiled)
    
    data = np.concatenate(data_list, axis=-1)  # (seq_length, num_nodes, 2)
    x, y = [], []
    min_t = abs(min(x_offsets))    # 11
    max_t = abs(num_samples - max(y_offsets))   # seq_length - 12
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]   # (12, num_nodes, 2)
        y_t = data[t + y_offsets, ...]   # (12, num_nodes, 2)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0) # x, y: (samples, 12, num_nodes, 2)
    return x, y

    

def generate_traffic_data(args, input_dir, output_dir):
    # data_path = os.path.join('../data/PEMSD7/PEMSD7.npz')
    data = np.load(input_dir)['data'][:, :, :args.input_dim]
    x_offsets = np.arange(-(args.window - 1), 1, 1)  # array([-11, -10, ..., 0])
    y_offsets = np.arange(1, args.horizon + 1, 1)   # array([1, 2, ..., 12])
    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        steps_per_day = args.steps_per_day,
        add_time_in_day=True,
        add_day_in_week=True,
    )
    print("X shape: ", x.shape, ", Y shape: ", y.shape)
    # Following previous research, sliding window sampling is performed first, and then data is divided.
    # traffic flow train/val/test: 6 : 2 : 2
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_rate)
    num_val = round(num_samples * args.val_rate)
    num_test = num_samples - num_train - num_val
    print("Train data: {}, Valid data: {}, Test data: {}.".format(num_train, num_val, num_test))
    # train data
    x_train, y_train = x[:num_train], y[:num_train]
    # valid data
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test data
    x_test, y_test = x[-num_test:], y[-num_test:]
    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()['y_' + cat]
        print(cat, 'x: ', _x.shape, ", y: ", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),  # (12, 1)
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])   # (12, 1)
        )



def main(args):
    print("Generating training data: ")
    if args.dataset == "PEMS08":
        print("PEMS08: ")
        generate_traffic_data(args, "../datasets/PEMS08/PEMS08.npz", "../datasets/PEMS08/processed/")
    elif args.dataset == "PEMS07":
        print("PEMS07: ")
        generate_traffic_data(args, "../datasets/PEMS07/PEMS07.npz", "../datasets/PEMS07/processed/")
    elif args.dataset == "PEMS04":
        print("PEMS04: ")
        generate_traffic_data(args, "../datasets/PEMS04/PEMS04.npz", "../datasets/PEMS04/processed/")
    elif args.dataset == "PEMS03":
        print("PEMS03: ")
        generate_traffic_data(args, "../datasets/PEMS03/PEMS03.npz", "../datasets/PEMS03/processed/")
    elif args.dataset == "PEMSD7(M)":
        print("PEMSD7(M): ")
        generate_traffic_data(args, "../datasets/PEMSD7(M)/PEMSD7(M).npz", "../datasets/PEMSD7(M)/processed/")
    elif args.dataset == "PEMSD7(L)":
        print("PEMSD7(L): ")
        generate_traffic_data(args, "../datasets/PEMSD7(L)/PEMSD7(L).npz", "../datasets/PEMSD7(L)/processed/")
    elif args.dataset == "TFA":
        print("TFA: ")
        generate_traffic_data(args, "../datasets/TFA/TFA.npz", "../datasets/TFA/processed/")
    else:
        print("PEMS: ")   # 扩展额外的数据接口
    print("Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train_rate", type=float, default=0.6)
    parser.add_argument("--val_rate", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--steps_per_day", type=int, default=288)
    args = parser.parse_args()
    # bash: python generate_pems_data.py --dataset PEMS08
    main(args)