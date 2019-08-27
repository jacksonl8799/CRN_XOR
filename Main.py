import os
import matplotlib.pyplot as plt
import random
import json
import numpy as np
from CRN_XOR import hid_NN, out_NN


def generate_rk(num_rxns):
    k_array = []
    for i in range(num_rxns):
        k_array.append(random.uniform(0.001, 1.0))
    return k_array


if __name__ == "__main__":
    open('rate_constants.txt', 'w').close()
    open('simulation_data.txt', 'w').close()
    open('fitness_data.txt', 'w').close()

    epochs = 40
    pop_size = 40
    num_rxns = 15
    trials = 50

    next_gen = []
    for r in range(pop_size):
        next_gen.append(generate_rk(num_rxns))

    with open('rate_constants.txt', 'w') as f:
        f.write(json.dumps(next_gen))

    for e in range(epochs):
        print("NEW GENERATION #", e)
        os.system('job_stream -- python Simulation.py')
        os.system('python GA_XOR.py')

    with open('simulation_data.txt', 'r') as f:
        population = json.loads(f.read())

    best_fitness = [line.rstrip('\n') for line in open('fitness_data.txt')]
    best_fitness = [float(elem) for elem in best_fitness]

    k_array = population[0]
    k_array = k_array[0]
    v1 = 1
    v2 = 100
    for_time = np.linspace(0, 9, 10)
    back_time = np.linspace(0, 100, 101)

    def generate_w():
        w_array = []
        for i in range(2):
            w_array.append(random.uniform(0.1, 0.9) * 100)
        return w_array


    def obtain_w(results, v):
        w1 = results['weight 1']
        w1 = w1[v]
        w2 = results['weight 2']
        w2 = w2[v]
        w_array = [w1, w2]
        return w_array


    hid1w = generate_w()
    hid2w = generate_w()
    outw = generate_w()

    pNeg = 0
    pPos = 0

    for t in range(trials):
        penalty = 0
        record_keeper = 0
        out = 0
        target = 0

        input_choice = [0.5, 1]
        input1 = random.choice(input_choice)
        input2 = random.choice(input_choice)
        input_array = [input1, input2]

        hid1 = hid_NN(input1, input2, hid1w[0], hid1w[1], record_keeper, pNeg, pPos, for_time, k_array).run()
        hid1 = hid1[0]
        # print("hid1 =", hid1)
        ff_input1 = hid1['feed forward input']
        ff_input1 = ff_input1[v1]
        hid1rk = hid1['record_keeper']
        hid1rk = hid1rk[v1]
        hid2 = hid_NN(input1, input2, hid2w[0], hid2w[1], record_keeper, pNeg, pPos, for_time, k_array).run()
        hid2 = hid2[0]
        # print("hid2 =", hid2)
        ff_input2 = hid2['feed forward input']
        ff_input2 = ff_input2[v1]
        hid2rk = hid2['record_keeper']
        hid2rk = hid2rk[v1]
        # print(ff_input1, ff_input2)
        results = out_NN(ff_input1, ff_input2, outw[0], outw[1], out, record_keeper, target, penalty, for_time,
                         k_array).run()
        # print(results)
        results = results[0]
        # print(results)

        if sum(input_array) == 1.5:
            target = 1
        else:
            target = 0.5

        tar = target * 100
        out_array = results['output']
        out = out_array[v1]

        penalty = abs(tar - out) / tar
        # print(input_array, out, penalty*100)

        record_keeper = results['record_keeper']
        record_keeper = record_keeper[v1]

        res = out_NN(0, 0, outw[0], outw[1], out, record_keeper, target, penalty, back_time, k_array).run()
        res = res[0]
        # print(res)
        outw = obtain_w(res, v2)
        # print(outw)

        pNeg1 = res['hidden 1 neg penalty']
        pNeg1 = pNeg1[v2]
        pPos1 = res['hidden 1 pos penalty']
        pPos1 = pPos1[v2]
        # print(pNeg1, pPos1)

        pNeg2 = res['hidden 2 neg penalty']
        pNeg2 = pNeg2[v2]
        pPos2 = res['hidden 2 pos penalty']
        pPos2 = pPos2[v2]
        # print(pNeg2, pPos2)

        hid1 = hid_NN(0, 0, hid1w[0], hid1w[1], hid1rk, pNeg1, pPos1, back_time, k_array).run()
        # print("hid1 =", hid1)
        hid1 = hid1[0]
        hid1w = obtain_w(hid1, v2)
        # print("hid1w =", hid1w)
        hid2 = hid_NN(0, 0, hid2w[0], hid2w[1], hid2rk, pNeg2, pPos2, back_time, k_array).run()
        # print("hid2 =", hid2)
        hid2 = hid2[0]
        hid2w = obtain_w(hid2, v2)
        # print("hid2w =", hid2w)

    for i in range(4):
        all_inputs = [[0.5, 0.5], [0.5, 1], [1, 0.5], [1, 1]]
        input_array = all_inputs[i]
        target = 0
        input1 = input_array[0]
        input2 = input_array[1]
        out = 0
        penalty = 0
        pNeg = 0
        pPos = 0
        record_keeper = 0

        hid1 = hid_NN(input1, input2, hid1w[0], hid1w[1], record_keeper, pNeg, pPos, for_time, k_array).run()
        hid1 = hid1[0]
        # print("hid1 =", hid1)
        ff_input1 = hid1['feed forward input']
        ff_input1 = ff_input1[v1]
        hid1rk = hid1['record_keeper']
        hid1rk = hid1rk[v1]
        hid2 = hid_NN(input1, input2, hid2w[0], hid2w[1], record_keeper, pNeg, pPos, for_time, k_array).run()
        hid2 = hid2[0]
        # print("hid2 =", hid2)
        ff_input2 = hid2['feed forward input']
        ff_input2 = ff_input2[v1]
        hid2rk = hid2['record_keeper']
        hid2rk = hid2rk[v1]
        # print(ff_input1, ff_input2)
        results = out_NN(ff_input1, ff_input2, outw[0], outw[1], out, record_keeper, target, penalty, for_time,
                         k_array).run()
        # print(results)
        results = results[0]
        # print(results)
        out_array = results['output']
        out = out_array[v1]

        print(input_array, out)

    plt.plot(np.arange(epochs), best_fitness)
    plt.show()
