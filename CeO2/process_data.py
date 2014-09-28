
def loadPtych_cxi(fnam):


fnam = 'CeO2_Physical_Review_2014.cxi'

prob = loadPtych_cxi(fnam)

prob = Ptychograph.backprop(prob)


qtPtych.show(prob)

