from pyVHR.analysis.testsuite import TestSuite

cfgFilename = "./pyVHR/analysis/sample2.cfg"
test = TestSuite(configFilename=cfgFilename)
result = test.start(outFilename='result.h5', verb=1)
