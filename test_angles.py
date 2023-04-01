#%%
import math
math.sin(45)

print(math.sin(math.radians(180)))
# %%
import numpy as np
tuple(np.asarray([[1, 0, 2]]))


#%%
import numpy as np

n = np.asarray([0.001, 0.001, 1])
n /= np.linalg.norm(n)
u = np.asarray([n[1], -n[0], 0]) # Assuming that a != 0 and b != 0, otherwise use c.
u /= np.linalg.norm(u)
v  = np.cross(n, u) # If n was normalized, v is already normalized. Otherwise normalize it.

print(np.dot(u, [4, 2, 0]))
print(np.dot(v, [4, 2, 0]))


#%%
from sympy import Point, Line, Segment, Rational
from skspatial.objects import Point as ssPoint
from skspatial.objects import Line as ssLine

def project_point_onto_line(p, line):
    a = p - line.point
    b = line.direction
    proj = b * (a.dot(b) / b.dot(b))
    return proj + line.point

p1, p2, p3 = Point(0, 0), Point(10, 10), Point(20, 0)
l1 = Segment(p1, p2)
print(l1.direction)
# print((l1.projection(p3) / l1.length).evalf())
print((l1.projection(p3)).evalf())

# print(l1.projection(p3))
print((Segment(p1, l1.projection(p3)).length / l1.length).evalf())


p1, p2, p3 = ssPoint([0, 0]), ssPoint([10, 10]), ssPoint([20, 0])
l1 = ssLine(point=p1, direction=p2-p1)
print(l1.direction)
print(l1.project_point(p3))
print(l1.project_point(p3).distance_point(p1) / l1.direction.norm())


print(project_point_onto_line(p3, l1))

# %%
