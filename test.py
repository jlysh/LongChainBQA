import os
import sys

__path = os.path.dirname(__file__)
updatePath = os.path.join(__path,'upload')
os.mkdir(updatePath)
print (os.path.isdir(__path))


print(updatePath)