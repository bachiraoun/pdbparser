#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
import os
path = os.getcwd()
path = path[:path.index("AMD")+4]
import sys
sys.path.insert(0, path)

import copy
import time

import numpy as np

from Utilities import Array, cprint

__repeat__ = 10

cprint("Array efficiency test","title")
print "Tests are run over a numpy array of shape (10e7, 3)"
print "Test is repeated %s times and the mean time value is calculated"%__repeat__
print "Slicing range is [0:100000:5,:]\n"
array = np.random.random((10000000,3))*100

##########################################################
##### normal numpy array slicing followed with copy ######
##########################################################
# testing copy_slice method
cprint("Testing normal brackets numpy array slicing efficiency followed with a copy:", "subtitle")
print "arrayTest = numpy.copy(array[0:100000:5,:]) is tested"
tic = time.clock()
for _ in range(__repeat__):
    arrayTest = np.copy(array[0:100000:5,:])
toc = time.clock()
print "The mean time of %s cycles in a loop is %s" %(__repeat__, str((toc - tic)/__repeat__) )

# outside a loop
tic = time.clock()
# 1
arrayTest = np.copy(array[0:100000:5,:])
# 2
arrayTest = np.copy(array[0:100000:5,:])
# 3
arrayTest = np.copy(array[0:100000:5,:])
# 4
arrayTest = np.copy(array[0:100000:5,:])
# 5
arrayTest = np.copy(array[0:100000:5,:])
# 6
arrayTest = np.copy(array[0:100000:5,:])
# 7
arrayTest = np.copy(array[0:100000:5,:])
# 8
arrayTest = np.copy(array[0:100000:5,:])
# 9
arrayTest = np.copy(array[0:100000:5,:])
# 10
arrayTest = np.copy(array[0:100000:5,:])

toc = time.clock()
print "The mean time of 10 cycles outside a loop is %s\n" %str((toc - tic)/10.0)


##########################################################
################ converting to Array test ################
##########################################################
cprint("Testing casting Array(numpy.array) efficiency:", "subtitle")
# in a loop
tic = time.clock()
for _ in range(__repeat__):
    arrayTest = Array(array)
toc = time.clock()
print "The mean time of %s cycles in a loop is %s" %(__repeat__, str((toc - tic)/__repeat__) )

# outside a loop
tic = time.clock()
# 1
arrayTest = Array(array)
# 2
arrayTest = Array(array)
# 3
arrayTest = Array(array)
# 4
arrayTest = Array(array)
# 5
arrayTest = Array(array)
# 6
arrayTest = Array(array)
# 7
arrayTest = Array(array)
# 8
arrayTest = Array(array)
# 9
arrayTest = Array(array)
# 10
arrayTest = Array(array)

toc = time.clock()
print "The mean time of 10 cycles outside a loop is %s\n" %str((toc - tic)/10.0)




##########################################################
############## Array copy_slice method test ##############
##########################################################
# testing copy_slice method
cprint("Testing copy_slice method efficiency:", "subtitle")
print "Slice is range(0,100000,5) of length 20000"
print "Each step contains mask_all() , reveal(indexes), copy_slice() methods"

arrayTest = Array(array)
tic = time.clock()
for _ in range(__repeat__):
    arrayTest.mask_all()
    arrayTest.reveal(range(0,100000,5))
    arrayTest.copy_slice()
toc = time.clock()
print "The mean time of %s cycles in a loop is %s" %(__repeat__, str((toc - tic)/__repeat__) )

# outside a loop
tic = time.clock()
# 1
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 2
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 3
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 4
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 5
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 6
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 7
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 8
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 9
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()
# 10
arrayTest.mask_all()
arrayTest.reveal(range(0,100000,5))
arrayTest.copy_slice()

toc = time.clock()
print "The mean time of 10 cycles outside a loop is %s\n" %str((toc - tic)/10.0)


finalStatement = "SUMMARY: copy_slice is useful only for irregular slicing indexes where normal slicing is not applicable"
cprint(finalStatement, ['information', 'bold']) 

