"""Pure-Python geometry oracle for synthetic loss probes, not AP evaluation."""
import math


SCALES = (.5, 1., 2.)
ASPECTS = (1., 1.25, 2., 4., 8.)
ANGLES = (0., .37)
ERRORS = (.001, .003, .01, .03, .06, .1, .2, .35, .6)
DIRECTIONS = {
    'center_long': (1., 0., 0., 0., 0.),
    'center_short': (0., 1., 0., 0., 0.),
    'log_width': (0., 0., 1., 0., 0.),
    'log_height': (0., 0., 0., 1., 0.),
    'angle': (0., 0., 0., 0., 1.),
    'center_angle': (0., 1., 0., 0., 1.),
    'shape_angle': (0., 0., 1., -1., 1.),
}


def decode(target, q):
    """q=(object-frame dx/w,dy/h,log(wp/w),log(hp/h),angle error radians)."""
    x, y, w, h, angle = target
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = q[0]*w, q[1]*h
    return [x+c*dx-s*dy, y+s*dx+c*dy,
            w*math.exp(q[2]), h*math.exp(q[3]), angle+q[4]]


def cases():
    rows = []
    for scale in SCALES:
        for aspect in ASPECTS:
            for angle in ANGLES:
                target = [256.*scale, 256.*scale, 24.*scale*aspect, 24.*scale, angle]
                for family, direction in DIRECTIONS.items():
                    norm = math.sqrt(sum(v*v for v in direction))
                    direction = [v/norm for v in direction]
                    for sign in (-1, 1):
                        for error in ERRORS:
                            rows.append(dict(scale=scale, aspect=aspect, angle=angle,
                                family=family, sign=sign, error=error, target=target,
                                q=[sign*error*v for v in direction], direction=direction))
    return rows


def corners(box, origin=(0., 0.)):
    x, y, w, h, a = box
    if not all(math.isfinite(v) for v in box) or min(w, h) <= 0:
        raise ValueError('Finite positive OBB required')
    c, s = math.cos(a), math.sin(a)
    return [(x-origin[0]+c*u-s*v, y-origin[1]+s*u+c*v)
            for u, v in ((-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2))]


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def riou(first, second):
    """Float64 convex polygon clipping; independent of CUDA IoU implementation."""
    origin = ((first[0]+second[0])/2, (first[1]+second[1])/2)
    poly, clip = corners(first, origin), corners(second, origin)
    tolerance = 1e-12*max(1., *first[2:4], *second[2:4])**2
    for i, a in enumerate(clip):
        b = clip[(i+1) % 4]
        edge = (b[0]-a[0], b[1]-a[1])
        old, poly = poly, []
        if not old:
            break
        start = old[-1]
        ds = cross(edge, (start[0]-a[0], start[1]-a[1]))
        for end in old:
            de = cross(edge, (end[0]-a[0], end[1]-a[1]))
            if (ds >= -tolerance) != (de >= -tolerance):
                t = ds/(ds-de)
                poly.append((start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1])))
            if de >= -tolerance:
                poly.append(end)
            start, ds = end, de
    area = abs(sum(cross(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))))/2 if len(poly) >= 3 else 0.
    a, b = first[2]*first[3], second[2]*second[3]
    area = max(0., min(area, a, b))
    return area/(a+b-area)


def diou(first, second):
    points = corners(first)+corners(second)
    diagonal = sum((max(p[j] for p in points)-min(p[j] for p in points))**2 for j in (0, 1))+1e-6
    distance = sum((first[j]-second[j])**2 for j in (0, 1))
    return 2.*(1.-riou(first, second)+distance/diagonal)


def sym_distance(first, second):
    """Independent Jeffreys divergence using the production w^2/4 covariance."""
    def covariance(b):
        c, s = math.cos(b[4]), math.sin(b[4])
        w, h = (max(b[2], 1.)/2)**2, (max(b[3], 1.)/2)**2
        return c*c*w+s*s*h, c*s*(w-h), s*s*w+c*c*h
    def inverse(m):
        a, b, d = m
        determinant = max(a*d-b*b, 1e-6)
        return tuple(max(-1e4, min(1e4, v/determinant)) for v in (d, -b, a))
    p, q = covariance(first), covariance(second)
    ip, iq = inverse(p), inverse(q)
    trace = lambda a, b: a[0]*b[0]+2*a[1]*b[1]+a[2]*b[2]
    dx, dy = first[0]-second[0], first[1]-second[1]
    maha = (ip[0]+iq[0])*dx*dx+2*(ip[1]+iq[1])*dx*dy+(ip[2]+iq[2])*dy*dy
    return min(1e4, max(0., .5*(trace(ip, q)+trace(iq, p)-4.+maha)))


def losses(target, q):
    pred = decode(target, q)
    iou = diou(pred, target)
    kld = 2.*min(math.sqrt(1.+sym_distance(pred, target))-1., 10.)
    return dict(DIoU=iou, SymKLD=kld, hybrid025=iou+.25*kld)


def band(value):
    if value >= .95:
        return '0.95-1.00'
    if value >= .85:
        return '0.85-0.95'
    if value >= .75:
        return '0.75-0.85'
    if value >= .5:
        return '0.50-0.75'
    return 'below0.50'


def derivative_intervals(target, q, h):
    """One-sided float64 slopes; retain an interval at polygon cusps."""
    center = losses(target, q)
    result = {name: [] for name in center}
    for coordinate in range(5):
        plus, minus = list(q), list(q)
        plus[coordinate] += h
        minus[coordinate] -= h
        upper, lower = losses(target, plus), losses(target, minus)
        for name in result:
            forward = (upper[name]-center[name])/h
            backward = (center[name]-lower[name])/h
            result[name].append((min(forward, backward), max(forward, backward)))
    return result


def matches_intervals(gradient, intervals, atol=5e-3, rtol=.03):
    for value, (low, high) in zip(gradient, intervals):
        tolerance = atol+rtol*max(abs(low), abs(high))
        if not math.isfinite(value) or not low-tolerance <= value <= high+tolerance:
            return False
    return True
