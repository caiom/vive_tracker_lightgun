#%%
import numpy as np

def closest_points_on_rays(o1, d1, o2, d2):

    # Calculate the vector connecting the origins
    w0 = o1 - o2

    # Calculate dot products
    A = np.dot(d1, d1)
    B = np.dot(d1, d2)
    C = np.dot(d2, d2)
    D = np.dot(d1, w0)
    E = np.dot(d2, w0)

    # Check if the rays are parallel (or almost parallel)
    denominator = A * C - B * B
    if abs(denominator) < 1e-6:
        # Rays are almost parallel
        # In this case, we can just return the origin points as the closest points
        return o1, o2

    # Calculate the scalar parameters s and t
    s = (B * E - C * D) / denominator
    t = (A * E - B * D) / denominator

    print(s)
    print(t)

    # Calculate the closest points P1 and P2
    P1 = o1 + s * d1
    P2 = o2 + t * d2

    return P1, P2


def closest_points_on_rays2(o1, d1, o2, d2):
    # Calculate the vector connecting the origins
    w0 = o2 - o1

    # Calculate dot products
    R1 = np.dot(d1, d1)
    d4321 = np.dot(d1, d2)
    R2 = np.dot(d2, d2)
    d3121 = np.dot(d1, w0)
    d4331 = np.dot(d2, w0)

    print(R1, R2)

    # Check if the rays are parallel (or almost parallel)
    denominator = d4321**2 - R1 * R2
    if abs(denominator) < 1e-6:
        # Rays are almost parallel
        # In this case, we can just return the origin points as the closest points
        return o1, o2

    # Calculate the scalar parameters s and t
    s = (d4321*d4331 - R2*d3121)/denominator
    t = (R1*d4331 - d4321*d3121)/denominator

    print(s)
    print(t)

    # Calculate the closest points P1 and P2
    P1 = o1 + s * d1
    P2 = o2 + t * d2

    return P1, P2

# Example usage
o1 = np.array([0, 0, 0])
d1 = np.array([1, 1, 1])
o2 = np.array([0, 1, 0])
d2 = np.array([1, 1, 3])

P1, P2 = closest_points_on_rays(o1, d1, o2, d2)
print(f'Distance: {np.linalg.norm(P1-P2)}')
print("Closest point on ray 1:", P1)
print("Closest point on ray 2:", P2)




def close_points(l1, l2):
    """return s and t, the parametric location of the closest
    points on l1 and l2, the actual points, and the separation between
    those two points.

    NOTE: these were checked on wolframalpha.com with query, e.g.
    "closest points on Line((0,0,0),(1,1,1)) and Line((0,1,0),(1,2,3))"

    Examples
    ========

    >>> from sympy import Line
    >>> close_points(Line((0, 0, 0), (1, 1, 1)), Line((0, 1, 0), (1, 2, 3)))
    (3/4, 1/4, Point3D(3/4, 3/4, 3/4), Point3D(1/4, 5/4, 3/4), sqrt(2)/2)
    >>> close_points(Line((0, 0, 0), (1, 2, 3)), Line((1, 1, 1), (2, 3, 5)))
    (7/5, 4/5, Point3D(7/5, 14/5, 21/5), Point3D(9/5, 13/5, 21/5), sqrt(5)/5)
    """
    # L1 = lambda s: l1.p1+s*l1.direction
    # L2 = lambda t: l2.p1+t*l2.direction
    # eq = lambda l1: Matrix(L1(var('s')) - L2(var('t'))).dot(l1.direction)
    # stdict = solve((
    #     eq(Line(var('x1 y1 z1'),var('x2 y2 z2'))),
    #     eq(Line(var('x3 y3 z3'),var('x4 y4 z4')))),
    #     var('s t'), dict=True)
    x1,y1,z1 = l1.p1
    x2,y2,z2 = l1.p2
    x3,y3,z3 = l2.p1
    x4,y4,z4 = l2.p2
    x21, y21, z21 = l1.direction
    x43, y43, z43 = l2.direction
    print(l2.direction)
    x31, y31, z31 = l2.p1 - l1.p1
    R1 = x21**2 + y21**2 + z21**2
    R2 = x43**2 + y43**2 + z43**2
    d4321 = x21*x43 + y21*y43 + z21*z43
    d3121 = x21*x31 + y21*y31 + z21*z31
    d4331 = x31*x43 + y31*y43 + z31*z43
    den = d4321**2 - R1*R2

    print(R1.evalf())
    print(R2.evalf())
    # R1*s - d4321*t - d3121 = 0
    # d4321*s - R2*t - d4331 = 0
    s = (d4321*d4331 - R2*d3121)/den
    t = (R1*d4331 - d4321*d3121)/den
    ps = l1.p1 + s*l1.direction
    pt = l2.p1 + t*l2.direction
    return s.evalf(), t.evalf(), ps.evalf(), pt.evalf(), ps.distance(pt).evalf()


from sympy import Line
close_points(Line((0, 0, 0), (1, 1, 1)), Line((0, 1, 0), (1, 2, 3)))


#%%
import numpy as np
import math


print(180 / math.pi * math.atan2(0.45, 1))

print(180 / math.pi * math.atan2(0.45, 1))


# 2 pos, mudo sinal dos 2
# 1 neg e um pos mudo o sinal do segundo
# 1 pos e um neg mudo o sinal do segundo
# 2 neg mudo o sinal dos 2



