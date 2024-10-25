import logging
import pandas as pd
import numpy as np
import torch
import torch.utils.data as utils
# from torch.utils.data import DataLoader, Dataset

# from https://github.com/osu-srml/CF_Representation_Learning/blob/master/CVAE/utils.py

def make_adult_loader(train_df, args):
    np.random.seed(seed=args.seed)
    a_train, o_train, r_train, d_train, y_train = [], [], [], [], []
    for idx, line in enumerate(train_df):
        if idx != 0:
            line = line.strip('\n').split('\t')
            a_train.append(line[8])
            o_train.append([line[7]]+[line[10]])
            #x_train.append(line[1:8]+line[9:11])
            r_train.append([line[1]]+[line[7]]+[line[10]])
            d_train.append(line[2:7] + [line[9]])
            y_train.append(line[11])

    a_train = np.asarray(a_train, dtype=np.float32)
    a_train = np.expand_dims(a_train, axis=1)
    #x_train = np.asarray(x_train, dtype=np.float32)
    r_train = np.asarray(r_train, dtype=np.float32)
    d_train = np.asarray(d_train, dtype=np.float32)
    o_train = np.asarray(o_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_train = np.expand_dims(y_train, axis=1)

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    test_pct = 0.2
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[:valid_ct]
    test_inds = shuffle[valid_ct:valid_ct+test_ct]
    train_inds = shuffle[valid_ct+test_ct:]

    a_valid = a_train[valid_inds]
    r_valid = r_train[valid_inds]
    d_valid = d_train[valid_inds]
    o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]

    a_test = a_train[test_inds]
    r_test = r_train[test_inds]
    d_test = d_train[test_inds]
    o_test = o_train[test_inds]
    y_test = y_train[test_inds]

    a_train = a_train[train_inds]
    r_train = r_train[train_inds]
    d_train = d_train[train_inds]
    o_train = o_train[train_inds]
    y_train = y_train[train_inds]

    train_set_r_tensor = torch.from_numpy(r_train)
    train_set_d_tensor = torch.from_numpy(d_train)
    train_set_o_tensor = torch.from_numpy(o_train)
    train_set_a_tensor = torch.from_numpy(a_train)
    train_set_y_tensor = torch.from_numpy(y_train)
    #train_set = utils.TensorDataset(train_set_r_tensor, train_set_d_tensor, train_set_o_tensor, train_set_a_tensor, train_set_y_tensor)
    train_set = utils.TensorDataset(train_set_r_tensor, train_set_d_tensor, train_set_a_tensor, train_set_y_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set_r_tensor = torch.from_numpy(r_valid)
    valid_set_d_tensor = torch.from_numpy(d_valid)
    valid_set_o_tensor = torch.from_numpy(o_valid)
    valid_set_a_tensor = torch.from_numpy(a_valid)
    valid_set_y_tensor = torch.from_numpy(y_valid)
    #valid_set = utils.TensorDataset(valid_set_r_tensor, valid_set_d_tensor, valid_set_o_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_set = utils.TensorDataset(valid_set_r_tensor, valid_set_d_tensor, valid_set_a_tensor, valid_set_y_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set_r_tensor = torch.from_numpy(r_test)
    test_set_d_tensor = torch.from_numpy(d_test)
    test_set_o_tensor = torch.from_numpy(o_test)
    test_set_a_tensor = torch.from_numpy(a_test)
    test_set_y_tensor = torch.from_numpy(y_test)
    #test_set = utils.TensorDataset(test_set_r_tensor, test_set_d_tensor, test_set_o_tensor, test_set_a_tensor, test_set_y_tensor)
    test_set = utils.TensorDataset(test_set_r_tensor, test_set_d_tensor, test_set_a_tensor, test_set_y_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    #input_dim = {'r': r_train.shape[1], 'd': d_train.shape[1], 'o': o_train.shape[1], 'a': a_train.shape[1],'y': y_train.shape[1]}
    input_dim = {'r': r_train.shape[1], 'd': d_train.shape[1], 'a': a_train.shape[1], 'y': y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim

def make_law_loader(data_df, args):
    np.random.seed(seed=args.seed)
    a_train, r_train, d_train, y_train = [], [], [], []
    a_train = torch.Tensor(data_df["sex"])[:, np.newaxis] - 1
    r_train = torch.Tensor(pd.get_dummies(data_df["race"]).values)
    d_train = torch.Tensor(data_df[["LSAT", "UGPA"]].values)
    if args.normalize:
        mean = torch.mean(d_train, dim=0, keepdim=True)
        std = torch.std(d_train, dim=0, keepdim=True)
        d_train = (d_train - mean) / std
    y_train = torch.Tensor(data_df["ZFYA"])[:, np.newaxis]

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    test_pct = 0.2
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[: valid_ct]
    test_inds = shuffle[valid_ct : valid_ct + test_ct]
    train_inds = shuffle[valid_ct + test_ct :]

    a_valid = a_train[valid_inds]
    r_valid = r_train[valid_inds]
    d_valid = d_train[valid_inds]
    y_valid = y_train[valid_inds]

    a_test = a_train[test_inds]
    r_test = r_train[test_inds]
    d_test = d_train[test_inds]
    y_test = y_train[test_inds]

    a_train = a_train[train_inds]
    r_train = r_train[train_inds]
    d_train = d_train[train_inds]
    y_train = y_train[train_inds]

    train_set = utils.TensorDataset(r_train, d_train, a_train, y_train)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set = utils.TensorDataset(r_valid, d_valid, a_valid, y_valid)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set = utils.TensorDataset(r_test, d_test, a_test, y_test)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    input_dim = {"r": r_train.shape[1], "d": d_train.shape[1], "a": a_train.shape[1], "y": y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim

def over_sampling(w1, w2):
    import copy
    whole1 = copy.deepcopy(w1)
    whole2 = copy.deepcopy(w2)
    len1 = len(whole1[0])
    len2 = len(whole2[0])
    small, large = (len1, len2) if len1 < len2 else (len2, len1)
    whole_small, whole_large = (whole1, whole2) if small == len1 else (whole2, whole1)

    shuffle_list = np.random.permutation(small)
    m = int(large/small)
    q = large % small

    for list in whole_small:
        list *= m
        for i in range(q):
            shuffle = shuffle_list[i] - 1
            list.append(list[shuffle])
    assert (len(whole_small[0]) == len(whole_large[0])), 'oversampling check'

    whole = []
    for i in range(len(whole_small)):
        whole.append(whole_small[i] + whole_large[i])
    return whole

def make_balancing_loader(train_df, args):
    np.random.seed(seed=args.seed)
    a0_train, o0_train, x0_train, y0_train, m0_train = [], [], [], [], []
    a1_train, o1_train, x1_train, y1_train, m1_train = [], [], [], [], []
    for idx, line in enumerate(train_df):
        if idx != 0:
            line = line.strip('\n').split('\t')
            if line[11] == str(0):
                a0_train.append(line[8])
                o0_train.append([line[7]]+[line[10]])
                x0_train.append(line[1:8]+line[9:11])
                y0_train.append(line[11])
                m0_train.append([line[7]]+[line[10]])
            else:
                a1_train.append(line[8])
                o1_train.append([line[7]] + [line[10]])
                x1_train.append(line[1:8] + line[9:11])
                y1_train.append(line[11])
                m1_train.append([line[7]] + [line[10]])

    print(len(y0_train))
    print(len(y1_train))

    whole1 = [a0_train, o0_train, x0_train, y0_train, m0_train]
    whole2 = [a1_train, o1_train, x1_train, y1_train, m1_train]
    (a_train, o_train, x_train, y_train, m_train) = over_sampling(whole1, whole2)

    print(len(y0_train))
    print(len(y1_train))

    a_train = np.asarray(a_train, dtype=np.float32)
    a_train = np.expand_dims(a_train, axis=1)
    x_train = np.asarray(x_train, dtype=np.float32)
    o_train = np.asarray(o_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_train = np.expand_dims(y_train, axis=1)
    m_train = np.asarray(m_train, dtype=np.float32)
    if args.all == True:
        o_train = x_train

    n = a_train.shape[0]
    shuffle = np.random.permutation(n)
    valid_pct = 0.2
    test_pct = 0.2
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[:valid_ct]
    test_inds = shuffle[valid_ct:valid_ct+test_ct]
    train_inds = shuffle[valid_ct+test_ct:]

    a_valid = a_train[valid_inds]
    x_valid = x_train[valid_inds]
    o_valid = o_train[valid_inds]
    y_valid = y_train[valid_inds]
    m_valid = m_train[valid_inds]

    a_test = a_train[test_inds]
    x_test = x_train[test_inds]
    o_test = o_train[test_inds]
    y_test = y_train[test_inds]
    m_test = m_train[test_inds]

    a_train = a_train[train_inds]
    x_train = x_train[train_inds]
    o_train = o_train[train_inds]
    y_train = y_train[train_inds]
    m_train = m_train[train_inds]

    train_set_x_tensor = torch.from_numpy(x_train)
    train_set_o_tensor = torch.from_numpy(o_train)
    train_set_a_tensor = torch.from_numpy(a_train)
    train_set_y_tensor = torch.from_numpy(y_train)
    train_set_m_tensor = torch.from_numpy(m_train)
    train_set = utils.TensorDataset(train_set_x_tensor, train_set_o_tensor, train_set_a_tensor, train_set_y_tensor, train_set_m_tensor)
    train_loader = utils.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set_x_tensor = torch.from_numpy(x_valid)
    valid_set_o_tensor = torch.from_numpy(o_valid)
    valid_set_a_tensor = torch.from_numpy(a_valid)
    valid_set_y_tensor = torch.from_numpy(y_valid)
    valid_set_m_tensor = torch.from_numpy(m_valid)
    valid_set = utils.TensorDataset(valid_set_x_tensor, valid_set_o_tensor, valid_set_a_tensor, valid_set_y_tensor, valid_set_m_tensor)
    valid_loader = utils.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    test_set_x_tensor = torch.from_numpy(x_test)
    test_set_o_tensor = torch.from_numpy(o_test)
    test_set_a_tensor = torch.from_numpy(a_test)
    test_set_y_tensor = torch.from_numpy(y_test)
    test_set_m_tensor = torch.from_numpy(m_test)
    test_set = utils.TensorDataset(test_set_x_tensor, test_set_o_tensor, test_set_a_tensor, test_set_y_tensor, test_set_m_tensor)
    test_loader = utils.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    input_dim = {'x': x_train.shape[1], 'o': o_train.shape[1], 'a': a_train.shape[1],'y': y_train.shape[1]}
    return train_loader, valid_loader, test_loader, input_dim