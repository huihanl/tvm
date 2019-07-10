#!/usr/bin/env python
# coding: utf-8

# # Perf. vs. Configuration High Dim Visualization

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys

# You will have to register for an account for plotly, which is free for 25 plots / month.
# After you registered a username, it will then generate an api_key for you.
# Then run plotly.tools.set_credentials_file(username='$USERNAME', api_key=$API_KEY)

import plotly as py
import plotly.graph_objs as go
import pandas as pd


def read_and_create_dataset(filepath, MB, IC, OC, IN_DIM, G, K, stride, pad):
    config_lst = []
    runtime_lst = []
    with open(filepath) as fp:
        for _, line in enumerate(fp):
            line = line.replace("null", "None")
            line = eval(line)
            config = line["i"][5]["e"][0][2]
            runtime = line["r"][0][0]
            config_lst.append(config)
            runtime_lst.append(runtime)

    # clean bad data
    new_config_lst = []
    new_runtime_lst = []
    for i in range(len(config_lst)):
        if runtime_lst[i] <= 1:
            new_config_lst.append(config_lst[i])
            new_runtime_lst.append(runtime_lst[i])


    IN_DIMP = [0, 0]
    OUT_DIM = [0, 0]

    IN_DIMP[0] = IN_DIM_lst[0] + pad_lst[0] + pad_lst[2];
    OUT_DIM[0] = (IN_DIMP[0] - K_lst[0]) / stride_lst[0] + 1;

    IN_DIMP[1] = IN_DIM_lst[1] + pad_lst[1] + pad_lst[3];
    OUT_DIM[1] = (IN_DIMP[1] - K_lst[1]) / stride_lst[1] + 1;

    MDim = MB * OUT_DIM[0] * OUT_DIM[1];
    KDim = K[0] * K[1] * IC;
    no_ops = 2 * MDim * NDim * KDim
    Gops_lst = [no_ops / r / math.pow(10, 9) for r in new_runtime_lst]

    new_config_str_lst = [str(i) for i in new_config_lst]
    dictionary = dict(zip(new_config_str_lst, Gops_lst))
    pair = sorted(dictionary.items(), key = lambda x : -x[1])
    new_config_x = [eval(p[0]) for p in pair]
    new_Gops_lst = [p[1] for p in pair]
    new_config_lst = np.array(new_config_x)
    new_config_lst = pd.DataFrame(new_config_lst, columns=["mcb", "ncb", "kcb", "mr", "nr", "16", "4"])
    new_config_lst = new_config_lst.drop(["16", "4"], axis = 1)
    new_config_lst["Gops"] = new_Gops_lst
    return new_config_lst


def plot_config_Gops(df, MB, IC, OC, IN_DIM, G, K, stride, pad):

    mcb_max = df['mcb'].max()
    mcb_min = df['mcb'].min()

    ncb_max = df['ncb'].max()
    ncb_min = df['ncb'].min()

    kcb_max = df['kcb'].max()
    kcb_min = df['kcb'].min()

    mr_max = df['mr'].max()
    mr_min = df['mr'].min()

    nr_max = df['nr'].max()
    nr_min = df['nr'].min()

    Gops_max = df['Gops'].max()
    Gops_min = df['Gops'].min()

    data = [
        go.Parcoords(
            line = dict(color = df['Gops'],
                       colorscale = 'Jet',
                       showscale = True,
                       reversescale = False,
                       cmin = Gops_min,
                       cmax = Gops_max),
            dimensions = list([
                dict(range = [mcb_min,mcb_max],
                     label = 'MCB', values = df['mcb']),
                dict(range = [ncb_min,ncb_max],
                     label = 'NCB', values = df['ncb']),
                dict(range = [kcb_min,kcb_max],
                     label = 'KCB', values = df['kcb']),
                dict(range = [mr_min,mr_max],
                     label = 'MR', values = df['mr']),
                dict(range = [nr_min,nr_max],
                     label = 'NR', values = df['nr'])
            ])
        )
    ]

    layout = go.Layout(title="MB={}, IC={}, OC={}, IN_DIM={}, G={}, K={}, stride={}, pad={}".format(MB, IC, OC, IN_DIM, G, K, stride, pad))

    py.offline.plot({"data": data, "layout": layout},
                    filename = "MB={}, IC={}, OC={}, IN_DIM={}, G={}, K={}, stride={}, pad={}".format(MB, IC, OC, IN_DIM, G, K, stride, pad), auto_open=True)

if __name__ == "__main__":

    for i in range(1, len(sys.argv)):

        filepath = sys.argv[i]
        words = filepath[15:][:-4]
        config = eval(words)
        df = read_and_create_dataset(filepath, config[0], config[1], config[2], config[3],
                              config[4], config[5], config[6], config[7])
        plot_config_Gops(df, config[0], config[1], config[2], config[3],
                              config[4], config[5], config[6], config[7])
