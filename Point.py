from vector import Vector

class Point:

    def __init__(self, position: Vector, prevPosition: Vector, locked: bool):
        self.position = position
        self.prevPosition = prevPosition
        self.locked = locked

    def getPosition(self):
        return self.position

    def setPosition(self, position):
        self.position = position

    def getPrevPosition(self):
        return self.prevPosition

    def setPrevPosition(self, prevPosition):
        self.prevPosition = prevPosition



    def getLocked(self):
        return self.locked