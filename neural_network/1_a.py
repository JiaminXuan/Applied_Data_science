import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle

dataset_and = SupervisedDataSet(2, 1)
dataset_and.addSample( (0,0) , (0,))
dataset_and.addSample( (0,1) , (0,))
dataset_and.addSample( (1,0) , (0,))
dataset_and.addSample( (1,1) , (1,))

net_and = buildNetwork(2, 4, 1, bias=True)

trainer_and = BackpropTrainer(net_and, learningrate = 0.01, momentum = 0.99)
trainer_and.trainOnDataset(dataset_and, 3000)
trainer_and.testOnData(verbose=True)


######################################################
dataset_or = SupervisedDataSet(2, 1)
dataset_or.addSample( (0,0) , (0,))
dataset_or.addSample( (0,1) , (1,))
dataset_or.addSample( (1,0) , (1,))
dataset_or.addSample( (1,1) , (1,))

net_or = buildNetwork(2, 4, 1, bias=True)

trainer_or = BackpropTrainer(net_or, learningrate = 0.01, momentum = 0.99)
trainer_or.trainOnDataset(dataset_or, 3000)
trainer_or.testOnData(verbose=True)


######################################################
dataset_not = SupervisedDataSet(1, 1)
dataset_not.addSample( (0,) , (1,))
dataset_not.addSample( (1,) , (0,))

net_not = buildNetwork(1, 4, 1, bias=True)

trainer_not = BackpropTrainer(net_not, learningrate = 0.01, momentum = 0.99)
trainer_not.trainOnDataset(dataset_not, 3000)
trainer_not.testOnData(verbose=True)


######################################################
dataset_nor = SupervisedDataSet(2, 1)
dataset_nor.addSample( (0,0) , (1,))
dataset_nor.addSample( (0,1) , (0,))
dataset_nor.addSample( (1,0) , (0,))
dataset_nor.addSample( (1,1) , (0,))

net_nor = buildNetwork(2, 4, 1, bias=True)

trainer_nor = BackpropTrainer(net_nor, learningrate = 0.01, momentum = 0.99)
trainer_nor.trainOnDataset(dataset_nor, 3000)
trainer_nor.testOnData(verbose=True)
