from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cv2
 
def loadImage(path):
    im = cv2.imread(path)
    return flatten(im)
 
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
 
if __name__ == "__main__":
 
    t = loadImage('pic/alligatorcopy.png')
    

    net = buildNetwork(len(t), .03*len(t), 1)
    ds = SupervisedDataSet(len(t), 1)
    ds.addSample(loadImage('pic/alligatorcopy.png'),(1,))
    ds.addSample(loadImage('pic/catcopy.png'),(2,))
    ds.addSample(loadImage('pic/dogcopy.png'),(3,))
    ds.addSample(loadImage('pic/giraffecopy.png'),(4,))
    ds.addSample(loadImage('pic/gorillacopy.png'),(5,))
    
    trainer = BackpropTrainer(net, ds)
    error = 10
    iteration = 0
    while error > 0.001: 
        error = trainer.train()
        iteration += 1
        
    ap=['pic/alligatorcopy.png','pic/catcopy.png','pic/dogcopy.png','pic/giraffecopy.png','pic/gorillacopy.png']
    an=['alligator','cat','dog','giraffe','gorilla']
    for a in ap:
        i=int(round(net.activate(loadImage(a)))-1)
        print 'Picture',a,'has a',an[i]+'.'