import pandas as pd


def read_data(path):
    # read data from file
    data = pd.read_csv(path)

    # load product list
    product_l = pd.read_pickle('product_list')

    return product_l, data
