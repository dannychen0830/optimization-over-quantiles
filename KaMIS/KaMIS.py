import numpy as np
import os
import networkx as nx


def run_KaMIS(adj):
    N = adj.shape[0]
    edge_count = 0

    file = open("./KaMIS/input.graph", "w")

    for i in range(N):
        for j in range(N):
            if adj[i,j] == 1:
                edge_count += 1

    file.write(repr(N) + " " + repr(int(edge_count/2)) + '\n')

    for i in range(N):
        for j in range(N):
            if adj[i,j] == 1:
                file.write(repr(j+1) + " ")
        file.write('\n')

    file.close()

    os.system("../KaMIS-master/deploy/redumis ./KaMIS/input.graph > ./KaMiS/output.txt")

    file = open('./KaMis/output.txt', 'r')
    for i in range(64):
        if i == 2:
            time_string = file.readline()
        else:
            file.readline()
    size_string = file.readline()

    while size_string[0:5] != "Indep" and size_string[0:4] != "Best":
        size_string = file.readline()

    index = -1
    for i in range(len(time_string)):
        if time_string[i].isdigit():
            index = i
            break
    time = float(time_string[index:len(time_string)])

    for i in range(len(size_string)):
        if size_string[i].isdigit():
            index = i
            break

    print(size_string)
    size = int(size_string[index:len(size_string)])

    return size, time
