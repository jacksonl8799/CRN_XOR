from gillespy2.core import Model, Species, Reaction, Parameter


class hid_NN(Model):

    def __init__(self, input1, input2, weight1, weight2, record_keeper, pNeg, pPos, time, k_array):

        self.input1 = input1
        self.input2 = input2
        self.weight1 = weight1
        self.weight2 = weight2
        self.record_keeper = record_keeper
        self.pNeg = pNeg
        self.pPos = pPos
        self.time = time
        self.k_array = k_array
        system_volume = 100
        constant = 1

        Model.__init__(self, name="Hidden_NN", volume=system_volume)
        x1 = Species(name='input 1', initial_value=int(self.input1*system_volume))
        x2 = Species(name='input 2', initial_value=int(self.input2*system_volume))
        w1 = Species(name='weight 1', initial_value=int(self.weight1))
        w2 = Species(name='weight 2', initial_value=int(self.weight2))
        y = Species(name='output', initial_value=0)
        xy = Species(name='record_keeper', initial_value=int(self.record_keeper))
        wPlus = Species(name='positive weight error', initial_value=0)
        wMinus = Species(name='negative weight error', initial_value=0)
        wNeg = Species(name='weight annihilator', initial_value=0)
        sF = Species(name="feed forward signal", initial_value=constant*system_volume)
        f = Species(name="feed forward input", initial_value=0)
        membrane = Species(name='cell membrane', initial_value=constant*system_volume)
        pNeg = Species(name='hidden neg penalty', initial_value=int(self.pNeg))
        pPos = Species(name='hidden pos penalty', initial_value=int(self.pPos))
        self.add_species([x1, x2, w1, w2, y, xy, wPlus, wMinus, wNeg, sF, f, membrane, pNeg, pPos])

        k1 = Parameter(name='k1', expression=self.k_array[0])
        k2 = Parameter(name='k2', expression=self.k_array[1])
        k3 = Parameter(name='k3', expression=self.k_array[2])
        k4 = Parameter(name='k4', expression=self.k_array[3])
        k5 = Parameter(name='k5', expression=self.k_array[4])
        k6 = Parameter(name='k6', expression=self.k_array[5])
        self.add_parameter([k1, k2, k3, k4, k5, k6])

        rxn11 = Reaction(name='input 1 to output', reactants={x1: 1, w1: 1}, products={y: 1, xy: 1, w1: 1}, rate=k1)
        rxn12 = Reaction(name='input 2 to output', reactants={x2: 1, w2: 1}, products={y: 1, xy: 1, w2: 1}, rate=k1)
        rxn21 = Reaction(name='x1 annihilation', reactants={x1: 1, y: 1}, products={}, rate=k2)
        rxn22 = Reaction(name='x2 annihilation', reactants={x2: 1, y: 1}, products={}, rate=k2)
        rxn3 = Reaction(name='positive weight adjustment', reactants={wPlus: 1, xy: 1}, products={w1: 1, w2: 1, xy: 1}, rate=k3)
        rxn41 = Reaction(name='negative weight adjustment 1', reactants={wMinus: 1, xy: 1}, products={wNeg: 1, xy: 1}, rate=k4)
        rxn42 = Reaction(name='negative weight adjustment 2', reactants={wNeg: 1, w1: 1}, products={}, rate=k4)
        rxn43 = Reaction(name='negative weight adjustment 3', reactants={wNeg: 1, w2: 1}, products={}, rate=k4)
        rxn5 = Reaction(name='output to input', reactants={y: 1, sF: 1}, products={f: 1, sF: 1}, rate=k5)
        rxn61 = Reaction(name='create pos weight adjust', reactants={pPos: 1, membrane: 1}, products={wPlus: 1, membrane: 1}, rate=k6)
        rxn62 = Reaction(name='create neg weight adjust', reactants={pNeg: 1, membrane: 1}, products={wMinus: 1, membrane: 1}, rate=k6)
        self.add_reaction([rxn11, rxn12, rxn21, rxn22, rxn3, rxn41, rxn42, rxn43, rxn5, rxn61, rxn62])
        self.timespan(self.time)


class out_NN(Model):

    def __init__(self, input1, input2, weight1, weight2, output, record_keeper, target, penalty, time, k_array):

        self.input1 = input1
        self.input2 = input2
        self.weight1 = weight1
        self.weight2 = weight2
        self.output = output
        self.record_keeper = record_keeper
        self.target = target
        self.penalty = penalty
        self.time = time
        self.k_array = k_array
        system_volume = 100
        constant = 1

        Model.__init__(self, name="Output_NN", volume=system_volume)
        x1 = Species(name='input 1', initial_value=int(self.input1))
        x2 = Species(name='input 2', initial_value=int(self.input2))
        w1 = Species(name='weight 1', initial_value=int(self.weight1))
        w2 = Species(name='weight 2', initial_value=int(self.weight2))
        y = Species(name='output', initial_value=int(self.output))
        xy = Species(name='record_keeper', initial_value=int(self.record_keeper))
        wPlus = Species(name='positive weight error', initial_value=0)
        wMinus = Species(name='negative weight error', initial_value=0)
        sL = Species(name='learning_signal', initial_value=constant*system_volume)
        wNeg = Species(name='weight annihilator', initial_value=0)
        yhat = Species(name='target', initial_value=int(self.target*system_volume))
        ePlus = Species(name='positive error', initial_value=0)
        eMinus = Species(name='negative error', initial_value=0)
        p = Species(name='penalty', initial_value=int(self.penalty*system_volume))
        pNeg1 = Species(name='hidden 1 neg penalty', initial_value=0)
        pPos1 = Species(name='hidden 1 pos penalty', initial_value=0)
        pNeg2 = Species(name='hidden 2 neg penalty', initial_value=0)
        pPos2 = Species(name='hidden 2 pos penalty', initial_value=0)
        self.add_species([x1, x2, w1, w2, y, xy, wPlus, wMinus, sL, wNeg, yhat, ePlus, eMinus, p, pNeg1, pPos1, pNeg2, pPos2])

        k1 = Parameter(name='k1', expression=self.k_array[6])
        k2 = Parameter(name='k2', expression=self.k_array[7])
        k3 = Parameter(name='k3', expression=self.k_array[8])
        k4 = Parameter(name='k4', expression=self.k_array[9])
        k5 = Parameter(name='k5', expression=self.k_array[10])
        k6 = Parameter(name='k6', expression=self.k_array[11])
        k7 = Parameter(name='k7', expression=self.k_array[12])
        k8 = Parameter(name='k8', expression=self.k_array[13])
        k9 = Parameter(name='k9', expression=self.k_array[14])
        self.add_parameter([k1, k2, k3, k4, k5, k6, k7, k8, k9])

        rxn11 = Reaction(name='input 1 to output', reactants={x1: 1, w1: 1}, products={y: 1, xy: 1, w1: 1}, rate=k1)
        rxn12 = Reaction(name='input 2 to output', reactants={x2: 1, w2: 1}, products={y: 1, xy: 1, w2: 1}, rate=k1)
        rxn21 = Reaction(name='x1 annihilation', reactants={x1: 1, y: 1}, products={}, rate=k2)
        rxn22 = Reaction(name='x2 annihilation', reactants={x2: 1, y: 1}, products={}, rate=k2)
        rxn3 = Reaction(name='error', reactants={y: 1, yhat: 1}, products={}, rate=k3)
        rxn41 = Reaction(name='create positive error', reactants={yhat: 1, sL: 1}, products={ePlus: 1, sL: 1}, rate=k4)
        rxn42 = Reaction(name='create positive weight error', reactants={p: 1, ePlus: 1}, products={ePlus: 2, wPlus: 1}, rate=k4)
        rxn51 = Reaction(name='create negative error', reactants={y: 1, sL: 1}, products={eMinus: 1, sL: 1}, rate=k5)
        rxn52 = Reaction(name='create negative weight error', reactants={p: 1, eMinus: 1}, products={eMinus: 2, wMinus: 1}, rate=k5)
        rxn6 = Reaction(name='annihilation of error species', reactants={ePlus: 1, eMinus: 1}, products={}, rate=k6)
        rxn7 = Reaction(name='positive weight adjustment', reactants={wPlus: 1, xy: 1}, products={w1: 1, w2: 1, xy: 1}, rate=k7)
        rxn81 = Reaction(name='negative weight adjustment 1', reactants={wMinus: 1, xy: 1}, products={wNeg: 1, xy: 1}, rate=k8)
        rxn82 = Reaction(name='negative weight adjustment 2', reactants={wNeg: 1, w1: 1}, products={}, rate=k8)
        rxn83 = Reaction(name='negative weight adjustment 3', reactants={wNeg: 1, w2: 1}, products={}, rate=k8)
        rxn911 = Reaction(name='create neg hidden penalty 1', reactants={wMinus: 1, w1: 1}, products={pNeg1: 1, w1: 1}, rate=k9)
        rxn912 = Reaction(name='create pos hidden penalty 1', reactants={wPlus: 1, w1: 1}, products={pPos1: 1, w1: 1}, rate=k9)
        rxn921 = Reaction(name='create neg hidden penalty 2', reactants={wMinus: 1, w2: 1}, products={pNeg2: 1, w2: 1}, rate=k9)
        rxn922 = Reaction(name='create pos hidden penalty 2', reactants={wPlus: 1, w2: 1}, products={pPos2: 1, w2: 1}, rate=k9)
        self.add_reaction([rxn11, rxn12, rxn21, rxn22, rxn3, rxn41, rxn42, rxn51, rxn52, rxn6, rxn7, rxn81, rxn82, rxn83, rxn911, rxn912, rxn921, rxn922])
        self.timespan(self.time)
