#!/usr/bin/env python3
"""Exact-radical witnesses and rational outward-rounded itinerary validation."""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import numpy as np
from polygon_theta_families import Polygon, polygons, validate_frame

OUT=Path(__file__).resolve().parents[1]/'research-data/polygon-families'
Q=10**40


class Interval:
    def __init__(self, lo, hi=None):
        lo, hi=F(lo), F(lo if hi is None else hi)
        assert lo<=hi
        self.lo=F((lo*Q).__floor__(),Q)
        self.hi=F((hi*Q).__ceil__(),Q)
    @staticmethod
    def lift(value): return value if isinstance(value,Interval) else Interval(value)
    def __add__(self, other):
        other=self.lift(other); return Interval(self.lo+other.lo,self.hi+other.hi)
    __radd__=__add__
    def __neg__(self): return Interval(-self.hi,-self.lo)
    def __sub__(self,other): return self+-self.lift(other)
    def __rsub__(self,other): return self.lift(other)+-self
    def __mul__(self,other):
        other=self.lift(other)
        values=[a*b for a in [self.lo,self.hi] for b in [other.lo,other.hi]]
        return Interval(min(values),max(values))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=self.lift(other)
        assert other.lo>0 or other.hi<0
        return self*Interval(1/other.hi,1/other.lo)
    def __rtruediv__(self,other): return self.lift(other)/self
    def sqrt(self):
        assert self.lo>=0
        lower=math.isqrt((self.lo*Q*Q).__floor__())
        upper=math.isqrt((self.hi*Q*Q).__floor__())+1
        return Interval(F(lower,Q),F(upper,Q))
    def bounds(self): return [float(self.lo),float(self.hi)]
    def exact_bounds(self):return [str(self.lo),str(self.hi)]


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def scale(c,a): return [c*x for x in a]


def certify_pentagon(epsilon=F(1,100000)):
    root3,root5=Interval(3).sqrt(),Interval(5).sqrt()
    a=(10+2*root5).sqrt()/4
    b=(root5-1)/4
    c=(1+root5)/4
    s=(10-2*root5).sqrt()/4
    vertices=[[Interval(0),Interval(1)],[-a,b],[-s,-c],[s,-c],[a,b]]
    magnitude=(5+3*root5)/2
    rho=magnitude*(s-c/root3)
    long_length=2*magnitude*c/root3
    t0=[b,-a]
    t1=[-b/2-root3*a/2,-root3*b/2+a/2]
    t2=[-b/2+root3*a/2,root3*b/2+a/2]
    u=[Interval(F(1,2)-epsilon,F(1,2)+epsilon)]*2
    v=add(u,scale(rho,t0))
    edges=[sub(vertices[(i+1)%5],vertices[i]) for i in range(5)]
    normals=[[e[1],-e[0]] for e in edges]
    constants=[dot(n,p) for n,p in zip(normals,vertices)]
    inside=[constant-dot(normal,node) for node in [u,v] for normal,constant in zip(normals,constants)]
    assert min(x.lo for x in inside)>0
    steps=[];fractions=[]
    all_segments=[]
    for word,direction,length in [((),t0,rho),((0,2,0,3),t1,long_length),((4,2,0,2),t2,long_length)]:
        point=u
        remaining=length
        segments=[]
        for wall in word:
            normal=normals[wall]
            denominator=dot(normal,direction)
            assert denominator.lo>0
            travel=(constants[wall]-dot(normal,point))/denominator
            assert travel.lo>0 and (remaining-travel).lo>0
            hit=add(point,scale(travel,direction))
            fraction=dot(sub(hit,vertices[wall]),edges[wall])/dot(edges[wall],edges[wall])
            assert fraction.lo>0 and fraction.hi<1
            fractions.append((wall,fraction))
            segments.append((point,direction,travel))
            steps.append(dict(wall=wall+1,flight_time=travel.bounds(),side_fraction=fraction.bounds(),
                              nongrazing_denominator=denominator.bounds(),
                              rational_side_fraction=fraction.exact_bounds(),
                              rational_flight_time=travel.exact_bounds()))
            point=hit
            remaining=remaining-travel
            direction=sub(direction,scale(2*dot(normal,direction)/dot(normal,normal),normal))
        assert remaining.lo>0
        segments.append((point,direction,remaining))
        # Nonincident path segments cannot pass through either junction.
        for index,(start,tangent,segment_length) in enumerate(segments):
            for label,node in [('u',u),('v',v)]:
                if (label=='u' and index==0) or (label=='v' and index==len(segments)-1): continue
                offset=sub(node,start)
                cross=tangent[0]*offset[1]-tangent[1]*offset[0]
                along=dot(offset,tangent)
                assert cross.lo>0 or cross.hi<0 or along.hi<0 or along.lo>segment_length.hi
        all_segments.append(len(segments))
    for i,(wall,interval) in enumerate(fractions):
        for other,previous in fractions[:i]:
            if wall==other:assert interval.hi<previous.lo or previous.hi<interval.lo
    return dict(status='PASS',arithmetic='Exact rational interval arithmetic, outward-rounded after each operation at denominator 10^40.',
                junction_box_half_width=str(epsilon),first_junction_box=[x.bounds() for x in u],
                translated_second_junction_box=[x.bounds() for x in v],
                direct_length=rho.bounds(),reflected_lengths=long_length.bounds(),
                total_length=(rho+2*long_length).bounds(),minimum_node_interior_margin=float(min(x.lo for x in inside)),
                steps=steps,segment_counts=all_segments,
                algebraic_identity_used='Four-bounce words have identity linear part and translations A*(-cos(18deg),sin(18deg)) and A*(sin(36deg),cos(36deg)); A=(5+3sqrt(5))/2. Endpoint and junction balance follow exactly from these identities.',
                scope='Certifies strict polygon itinerary inequalities on a two-dimensional positive-width box, and excludes intermediate junction visits. It is not a certification of the maximal admissible-region area.')


def rhombus_slice():
    polygon=Polygon(polygons()['rhombus_60'])
    by_word={w.sequence:w for level in polygon.words(2) for w in level}
    words=[by_word[()],by_word[(0,3)],by_word[(1,2)]]
    tangents=np.array([[-.5,math.sqrt(3)/2],[-.5,-math.sqrt(3)/2],[1.,0.]])
    parameters=np.linspace(math.sqrt(3)/16,3*math.sqrt(3)/16,9)
    frames=[]
    for y in parameters:
        u=np.array([.75,y]);v=np.array([.375+math.sqrt(3)*y/2,3*math.sqrt(3)/8-y/2])
        frame=validate_frame(polygon,words,tangents,np.r_[u,v])
        assert frame is not None
        exact=np.array([.75-math.sqrt(3)*y,.75+math.sqrt(3)*y,1.5])
        assert np.max(np.abs(exact-np.array(frame['lengths'])))<1e-12
        frames.append(frame)
    return dict(parameter='y',sampled_parameters=parameters.tolist(),frames=frames,
                exact_total_length='3',valid_open_interval='0<y<sqrt(3)/4',
                junction_formula='u=(3/4,y); v=(3/8+sqrt(3)*y/2,3sqrt(3)/8-y/2)',
                edge_length_formula='(3/4-sqrt(3)*y,3/4+sqrt(3)*y,3/2)')


def certify_destroyed_itinerary():
    root3,root5=Interval(3).sqrt(),Interval(5).sqrt()
    a=(10+2*root5).sqrt()/4;b=(root5-1)/4;c=(1+root5)/4;s=(10-2*root5).sqrt()/4
    vertices=[[Interval(F(1,20)),Interval(F(103,100))],[-a,b],[-s,-c],[s,-c],[a,b]]
    matrices=[]
    for index,p in enumerate(vertices):
        edge=sub(vertices[(index+1)%5],p);normal=[edge[1],-edge[0]];norm=dot(normal,normal)
        matrices.append([[Interval(int(i==j))-2*normal[i]*normal[j]/norm for j in range(2)] for i in range(2)])
    identity=[[Interval(1),Interval(0)],[Interval(0),Interval(1)]]
    def multiply(first,second):return [[sum(first[i][k]*second[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    words=[]
    for word in [(),(0,2,0,3),(4,2,0,2)]:
        result=identity
        for wall in word:result=multiply(result,matrices[wall])
        words.append(result)
    result=[]
    for handedness in [-1,1]:
        base=[[Interval(1),Interval(0)],[Interval(F(-1,2)),handedness*root3/2],
              [Interval(F(-1,2)),-handedness*root3/2]]
        matrix=[[Interval(0),Interval(0)],[Interval(0),Interval(0)]]
        for transform,tangent in zip(words,base):
            for column,vector in enumerate([tangent,[-tangent[1],tangent[0]]]):
                for row in range(2):matrix[row][column]+=sum(transform[k][row]*vector[k] for k in range(2))
        determinant=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
        assert determinant.lo>0
        result.append(dict(handedness=handedness,determinant=determinant.bounds(),
                           rational_determinant=determinant.exact_bounds()))
    return dict(status='PASS',perturbed_vertex=['1/20','103/100'],words_zero_based=[[],[0,2,0,3],[4,2,0,2]],
                determinants=result,conclusion='Both angular-balance matrices are nonsingular. This particular pair of reflected itineraries supports no free-junction 120-degree theta net in the perturbed polygon, at any junction positions.',
                scope='No assertion about other itineraries or longer paths.')


def main():
    result=dict(regular_pentagon_interval_certificate=certify_pentagon(),exact_rhombus_slice=rhombus_slice(),
                destroyed_pentagon_itinerary=certify_destroyed_itinerary())
    (OUT/'certificates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pentagon_certificate':result['regular_pentagon_interval_certificate'],
                      'rhombus_length_spread':max(f['total_length'] for f in result['exact_rhombus_slice']['frames'])-min(f['total_length'] for f in result['exact_rhombus_slice']['frames'])},indent=2))


if __name__=='__main__':main()
