import math
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.v = np.array([x, y])
        self.hy = np.linalg.norm(self.v)

        if self.x != 0:

            self.alfa = math.ceil(int(math.atan2(self.y, self.x) * 180 / math.pi))

        else:
            self.alfa = 90




    def pulse(self, alfa, rightDirection):
        if rightDirection:
            self.pulse = int(1500 - alfa / 90 * 1000)
        else:
            self.pulse = int(1500 + alfa / 90 * 1000)

        self.pulse /= 1000.0
        if self.pulse < 0.5:
            self.pulse = 0.5

        if self.pulse > 2.5:
            self.pulse = 2.5


        return self.pulse

    def normalize(self):
        self.x = int(math.cos(self.alfa/180 * math.pi)*100)/ 100.0
        self.y = int(math.sin(self.alfa/180 * math.pi)*100)/ 100.0

        return Vector(self.x, self.y)

    def down(self):
        if self.rigthDirction:
            self.alfa2 -= 1
        else:
            self.alfa2 += 1

        x = int(math.cos(self.alfa / 180 * math.pi) * 100) / 100.0
        y = int(math.sin(self.alfa / 180 * math.pi) * 100) / 100.0

        return Vector(x, y)


    def getvector(self):
        return self.v


    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Multiplication of a vector by a scalar."""

        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector(self.x*scalar, self.y*scalar)
        raise NotImplementedError('Can only multiply Vector2D by a scalar')

