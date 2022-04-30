from Point import Point

class Stick:

    def __init__(self, pa: Point, pb: Point, length: int):
        self.length = length
        self.pointA = pa
        self.pointB = pb


    def getLength(self):
        return self.length

    def getPointA(self):
        return self.pointA

    def getPointB(self):
        return self.pointB

    def setPointA(self, pa):
        self.pointA = pa

    def setPointB(self, pb):
        self.pointB = pb
