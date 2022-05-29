import os
import shutil
path = "c:/ChessResize/"

#valid 10%
def prozent(poz, destination):

    for x in os.listdir(path):
       count = 0
       for f in os.listdir(path + '/' + x):
           count += 1
       m = int(count * poz / 100.0)

       for f in os.listdir(path + '/' + x):
           if m > 0:
               if not os.path.isdir(path + destination + '/' + x):
                   os.mkdir(path + destination + '/' + x)
               shutil.move(path + '/' + x + '/' + f, path + destination + '/' + x + '/' + f)
               m-=1


dummy = 'test'

if not os.path.isdir(path + "/" + dummy):
    os.mkdir(path + "/" + dummy)
prozent(10, dummy)