import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import argparse

from model import PrincipalComponentAnalysis

DATA_MAX_INTENSITY = 200
DATA_MIN_INTENSITY = -1200


def check_path(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def load_data(p):

    train_dataset = np.load(os.path.join(p, "train.npy"))
    test_dataset = np.load(os.path.join(p, "test.npy"))

    train_dataset = train_dataset.reshape(train_dataset.shape[0], -1)
    test_dataset = test_dataset.reshape(test_dataset.shape[0], -1)

    train_dataset = pre(train_dataset)
    test_dataset = pre(test_dataset)

    return train_dataset, test_dataset

def pre(x):
    return (x - DATA_MIN_INTENSITY)/(DATA_MAX_INTENSITY - DATA_MIN_INTENSITY)

def post(x):
    return x * (DATA_MAX_INTENSITY - DATA_MIN_INTENSITY) + DATA_MIN_INTENSITY

def get_cr_and_ccr_csv(eigen_val, pca_cr, pca_ccr, save_dir):

    cr_and_ccr_list = []
    for i, (val, cr, ccr) in enumerate(zip(eigen_val, pca_cr, pca_ccr)):
        cr_and_ccr_list.append([i+1, val, cr, ccr])

    columns = ["軸", "固有値", "寄与率", "累積寄与率"]
    df = pd.DataFrame(cr_and_ccr_list, columns=columns)

    file_name = os.path.join(save_dir, "cr_and_ccr.csv")
    print("Saving ==>> {}".format(file_name))
    df.to_csv(file_name, index=False)


def process_train(args, x_train):
    
    pca = PrincipalComponentAnalysis(args.n_component)
    
    pca.fit(x_train)

    pca_mean = pca.get_mean_vect()
    pca_eigen_val = pca.get_eigen_value()
    pca_eigen_vec = pca.get_eigen_vect()

    pca_cr = pca.get_cr()
    pca_ccr = pca.get_ccr()

    get_cr_and_ccr_csv(list(pca_eigen_val), pca_cr, pca_ccr, args.save_csv_path)
        
    file_name_1 = os.path.join(args.save_model_path, "mean_vect.npy")
    file_name_2 = os.path.join(args.save_model_path, "eigen_value.npy")
    file_name_3 = os.path.join(args.save_model_path, "eigen_vect.npy")

    print("Saving ==>> {}".format(file_name_1))
    print("Saving ==>> {}".format(file_name_2))
    print("Saving ==>> {}".format(file_name_3))

    np.save(file_name_1, pca_mean)
    np.save(file_name_2, pca_eigen_val)
    np.save(file_name_3, pca_eigen_vec)


def process_test(args, x_test):

    pca = PrincipalComponentAnalysis(args.n_component)

    mean_vect   = np.load(os.path.join(args.save_model_path, "mean_vect.npy"))
    eigen_value = np.load(os.path.join(args.save_model_path, "eigen_value.npy"))
    eigen_vect  = np.load(os.path.join(args.save_model_path, "eigen_vect.npy"))

    pca.set_mean_vect(mean_vect)
    pca.set_eigen_value(eigen_value)
    pca.set_eigen_vect(eigen_vect)

    projected = pca.projection(x_test)
    reconstructed = pca.reconstruction(projected)

    reconstructed = reconstructed.reshape(-1, args.patch_size, args.patch_size, args.patch_size)

    reconstructed = post(reconstructed)

    file_name_1 = os.path.join(args.save_npy_path, "x.npy")
    file_name_2 = os.path.join(args.save_npy_path, "z.npy")

    print("Saving ==>> {}".format(file_name_1))
    print("Saving ==>> {}".format(file_name_2))

    np.save(file_name_1, reconstructed)
    np.save(file_name_2, projected)


def process_arbit(args, x_train):

    np.random.seed(seed=args.seed)

    pca = PrincipalComponentAnalysis(args.n_component)

    mean_vect   = np.load(os.path.join(args.save_model_path, "mean_vect.npy"))
    eigen_value = np.load(os.path.join(args.save_model_path, "eigen_value.npy"))
    eigen_vect  = np.load(os.path.join(args.save_model_path, "eigen_vect.npy"))

    pca.set_mean_vect(mean_vect)
    pca.set_eigen_value(eigen_value)
    pca.set_eigen_vect(eigen_vect)

    projected = pca.projection(x_train)

    loc = np.mean(projected, axis=0)
    scale = np.var(projected, axis=0)

    z_sample = np.random.normal(loc=loc[0], scale=np.sqrt(scale[0]), size=[args.n_shape, 1])
    for i in range(args.n_component - 1):
        temp = np.random.normal(loc=loc[i + 1], scale=np.sqrt(scale[i + 1]), size=[args.n_shape, 1])
        z_sample = np.concatenate([z_sample, temp], axis=1)

    arbit = []
    for i in range(args.n_shape):
        arb = pca.reconstruction(z_sample[i])
        arbit.append(arb)

    arbits = np.asarray(arbit).reshape(-1, args.patch_size, args.patch_size, args.patch_size)

    arbits = post(arbits)

    file_name = os.path.join(args.save_npy_path, "arbit.npy")

    print("Saving ==>> {}".format(file_name))

    np.save(file_name, arbits)

def visual_latent_space(projected, save_dir):

    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.grid()

    check_path(save_dir)
    file_name = os.path.join(save_dir, "latent.png")
    print("Saving ==>> {}".format(file_name))
    plt.savefig(file_name)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="D:/oosky/SIMs/data_CT/np_record/")

    parser.add_argument('--save_model_path', type=str, default="./model/")

    parser.add_argument('--save_npy_path', type=str, default="./npy/")

    parser.add_argument('--save_csv_path', type=str, default="./csv/")

    parser.add_argument('--n_component', type=int, default=20)

    parser.add_argument('--patch_size', type=int, default=32)

    parser.add_argument('--n_shape', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--mode', type=str, default="arbit", help="[train, test, arbit]")

    args = parser.parse_args()
    return args

def main(args):

    check_path(args.save_model_path)
    check_path(args.save_npy_path)
    check_path(args.save_csv_path)

    x_train, x_test = load_data(args.data_path)

    if args.mode == "train":
        process_train(args, x_train)

    elif args.mode == "test":
        process_test(args, x_test)

    elif args.mode == "arbit":
        process_arbit(args, x_train)


if __name__ == '__main__':
    args = arg_parser()
    main(args)
            