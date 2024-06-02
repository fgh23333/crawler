import ijson
import json
import os
outputPath = "new/rewrite/"
solvedPath = "new/solved/"
def rewrite(fileName):
    thename = fileName[:len(fileName)-11]
    return thename

ll = []
for dirpath, dirnames, filenames in os.walk(solvedPath):
    for filename in filenames:
        ll.append(rewrite(filename))
print(ll)