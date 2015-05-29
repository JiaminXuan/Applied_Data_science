import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

########################################################
ds = SupervisedDataSet(3, 1)
ds.addSample( (0,0,0) , (0,))
ds.addSample( (0,0,1) , (1,))
ds.addSample( (0,1,0) , (0,))
ds.addSample( (0,1,1) , (1,))
ds.addSample( (1,0,0) , (0,))
ds.addSample( (1,0,1) , (1,))
ds.addSample( (1,1,0) , (1,))
ds.addSample( (1,1,1) , (1,))

net = buildNetwork(3, 2, 1, bias=True)

trainer = BackpropTrainer(net, learningrate = 0.01, momentum = 0.99)
trainer.trainOnDataset(ds, 3000)
trainer.testOnData(verbose=True)

########################################################
ds = SupervisedDataSet(3, 1)
ds.addSample( (0,0,0) , (0,))
ds.addSample( (0,0,1) , (1,))
ds.addSample( (0,1,0) , (0,))
ds.addSample( (0,1,1) , (0,))
ds.addSample( (1,0,0) , (0,))
ds.addSample( (1,0,1) , (0,))
ds.addSample( (1,1,0) , (0,))
ds.addSample( (1,1,1) , (0,))

net = buildNetwork(3, 3, 1, bias=True)

trainer = BackpropTrainer(net, learningrate = 0.01, momentum = 0.99)
trainer.trainOnDataset(ds, 3000)
trainer.testOnData(verbose=True)

########################################################
ds = SupervisedDataSet(3, 1)
ds.addSample( (0,0,0) , (1,))
ds.addSample( (0,0,1) , (1,))
ds.addSample( (0,1,0) , (1,))
ds.addSample( (0,1,1) , (0,))
ds.addSample( (1,0,0) , (1,))
ds.addSample( (1,0,1) , (0,))
ds.addSample( (1,1,0) , (1,))
ds.addSample( (1,1,1) , (0,))

net = buildNetwork(3, 2, 1, bias=True)

trainer = BackpropTrainer(net, learningrate = 0.01, momentum = 0.99)
trainer.trainOnDataset(ds, 3000)
trainer.testOnData(verbose=True)

########################################################
