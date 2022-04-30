from vector import Vector
from Point import Point
from stick import Stick
import numpy as np
import matplotlib.pyplot as plt

points = []
sticks = []

vectorE = Vector(2.3, 2.3)
vector1 = Vector(1, -1.3)
vector2 = Vector(4, -0.4)
vector3 = Vector(7.5, 0.9)
vector4 = Vector(12.4, 2)



points = [Point(vectorE, Vector(0, 0), True), Point(vector2, Vector(0, 0), False), Point(vector3, Vector(0, 0), False), Point(vector4, Vector(0, 0), True)]

sticks = [Stick(points[0], points[1], 3.0),Stick(points[1], points[2], 4.0), Stick(points[2], points[3], 5.0)]


numIteration = 10
gravity = 1
def simulate():

   for i in range(numIteration):
      #for p in points:
      #   postionBeforeUpdate = p.getPosition()
      #   p.setPosition(p.getPosition() + p.getPosition() - p.getPrevPosition())
      #   p.setPrevPosition(postionBeforeUpdate)


      vectors = []
      for stick in sticks:
         stickCentre = (stick.pointA.position + stick.pointB.position) * 0.5
         strickDir = (stick.pointA.position - stick.pointB.position).normalize()

         if not stick.pointA.locked:
             stick.pointA.position = stickCentre + strickDir * stick.length * 0.5
             print('alfaA = ', stick.pointA.position.alfa)

         if not stick.pointB.locked:
             stick.pointB.position = stickCentre - strickDir * stick.length * 0.5
             print('alfaB = ', stick.pointB.position.alfa)

         vectors.append(stick.pointA.position)
      vectors.append(stick.pointB.position)
      #plt.quiver(*origin, vectors[0], vectors[1], vectors[2], color=['r', 'b', 'g'], scale=100)
      #plt.show()
      x = []
      y = []
      for v in vectors:
          x.append(v.x)
          y.append(v.y)
      #x = [vectorE.x, vector2.x, vector3.x, vector4.x]
      #y = [vectorE.y, vector2.y, vector3.y, vector4.y]

      plt.plot(x, y, marker='o')
      plt.show()

simulate()