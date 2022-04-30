from vector import Vector
from Point import Point
from stick import Stick
import numpy as np
import matplotlib.pyplot as plt

from sendPost import send

debug = 1
vector0 = Vector(0, 9.2)
vector1 = Vector(0, 19.4)
vector2 = Vector(0, 29.0)
vector3 = Vector(20, 20)

sticks = [10.2, 9.6, 14.8]


vectors = []
vectors.append(vector0)
vectors.append(vector1)
vectors.append(vector2)
vectors.append(vector3)

vs = []
array = []
alfas = []
alfamotor = []
vectors2 = []
hys = []
numIteration = 20

def simulate():
   vs = vectors

   m = 0
   vor = vector0
   sticks = [10.2, 9.6, 14.8]
   for l in range(numIteration):

       i = 0
       if m > 0:

            if m % 2 != 0:
                sticks = [14.8, 9.6, 10.2]
                vor = vector3
            else:
                sticks = [10.2, 9.6, 14.8]
                vor = vector0
       else:
            vs = vectors

       vs1 = []

       vs1.append(vor)

       if m % 2 == 0:
            vectors2 = []

       for v in vs:
           if i > 0:
               x1 = v - vor
               p = x1.normalize() * sticks[i - 1]
               if m % 2 == 0:
                    vectors2.append(p)
               vor = vor + p
               vs1.append(vor)
           if debug:
                print(vor.alfa)
           i += 1
       vs = []
       for v in reversed(vs1):
           vs.append(v)
       m += 1
       x = []
       y = []
       for v in vs1:
           x.append(v.x)
           y.append(v.y)
       if debug:
           plt.plot(x, y, marker='o')
           plt.show()

#auswertung
   k = 0
   alfaold = 90
   for v in vectors2:
       array.append(v.pulse(alfaold - v.alfa, not k == 1))
       alfas.append(v.alfa)
       alfamotor.append(alfaold - v.alfa)
       hys.append((v.hy))
       alfaold = v.alfa
       k+=1

simulate()
print("Alfas=", alfas)
print("AlfaMotor", alfamotor)
print("pulses", array)
print("hys", hys)
send(array)