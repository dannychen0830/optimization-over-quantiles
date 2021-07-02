import time
import numpy as np
import  os


def run_exact(adj):
    N = adj.shape[0]
    edge_count = 0

    string = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i,j] == 1:
                string.append(str(i) + ' ' + str(j) + '\n')
                edge_count += 1

    string = "\n\n# Nodes: " + str(N) + " # Edges: " + str(edge_count) + '\n' + ''.join(map(str, string))

    file = open('./Exact/input_graph.txt', 'w')
    file.write(string)
    file.close()

    start_time = time.time()
    os.system('../Maximum-Independent-Set-master/mis ./Exact/input_graph.txt > ./Exact/output_text.txt')
    end_time = time.time()

    file = open('./Exact/output_text.txt', 'r')
    line = file.readlines()[-1]
    for i in range(len(line)):
        if line[i].isdigit():
            index = i
            break

    size = str(line[index:len(line)-1])

    return size, end_time-start_time