---
layout: post
title: "Geometric probability: an (even) harder problem on the hardest test"
author: "Felipe Contatto"
categories: journal
tags: [probability, geometry, maths]
---
# Geometric probability: an (even) harder problem on the hardest test
One of the most famous questions from Putnam competitions is number A6 from the 53rd Putnam of 1992 (c.f., for instance, [here](https://prase.cz/kalva/putnam/putn92.html)). The problem is: if you sample 4 points uniformly over a sphere, what is the probability that the tetrahedron defined by them contains the centre of the sphere? The problem is simple to understand and there are solutions easily found on the internet, one of which from [3Blue1Brown](https://youtu.be/OkmNXy7er84?feature=shared).

Possible generalisations of the problem are: 
>Problem 1: given a (n-1)-sphere in n dimensions, sample n+1 points uniformly over it, what is the probability $q_{n, n+1}$ that their convex hull contains the centre? 
And an even more general one is:
>Problem 2: given a (n-1)-sphere in n dimensions, sample N points uniformly over it, what is the probability $q_{n,N}$ that their convex hull contains the centre?

Another way to pose problem 2 is by asking the probability $p_{n,N}=1-q_{n,N}$ that N points fall in the same hemisphere. This problem was solved by [Wendel back in 1962](https://www.mscand.dk/article/view/10655/8676) by using basic combinatorics, topology and linear algebra. Remarkably, the answer to Wendel's problem is equal to the probability that, upon flipping N-1 fair coins, you get less than n heads, which is given by (assuming $N>n$)

$$
p_{n,N} = \sum_{k=0}^{n-1}\mathbb P(\text{flipping exactly k heads in N-1 throws}) = \frac{1}{2^{N-1}}\sum_{k=0}^{n-1}\binom{N-1}{k}. 
$$

No connection between the coin flipping probability space and Problem 2's probability space was ever established and the fact that the solutions to these seemingly unrelated problems are the same has been considered a mere coincidence. In this post, I provide a solution to Problem 1 clearly showing that the probability of the centre of the sphere being inside the convex hull is the same as that of flipping $n$ coins and getting all heads.

Throughout the rest of the post, we will define the coordinates of $\mathbb{R}^n$ by $(x_1, \dots, x_n)$ and assume, without loss of generality, that the sphere has radius $1$ and is centered at the origin.

## Solution of Problem 1

Draw the first point uniformly on the sphere. By symmetry, we can fix it to be the north pole ($x_n=1$). Let us define $L$ to be the segment connecting the centre to the south pole ($x_n=-1$). Now, sample the $n$ remaining points $p_1, \dots, p_n$, which will define a unique $n-1$ dimensional hyperplane with probability $1$. Parametrise the sphere with angles $(\phi_1, \dots, \phi_{n-2}, \theta)$, where $\theta$ is the angle between a point on the sphere and the $x_n$-axis, namely, $x_n=\cos\theta$, and the $\phi_i$'s parameterise the $(n-2)$-sphere $x_1^2+\cdots+x_{n-1}^2=1$ that we call $S^{n-2}$ and corresponds to the equator.
 
>The centre of the sphere will be contained in the convex hull if and only if the conditions below are true:
>1. the hyperplane intersects L and
>2. the points $p_1, \dots, p_n$, upon having its $\theta$ coordinates set to $\pi/2$ (that is to say, being "projected" to the equator, so to speak), will define a convex hull in the subspace {$x_n=0$} that contains the centre of the equator $S^{n-2}$.

**Proof of the conditions**

Both conditions are clear from a picture in $2$ or $3$ dimensions, but in general, they are true because of the convexity (and closure, I should have mentioned) of the hull. In fact, consider the straight line defined by one of the points  of the convex hull (call this point the north pole) and the centre, in orther words, the $x_n$-axis: $t\in\mathbb R \mapsto (0, \dots, 0, t)$. Assume that this line will intersect one (therefore only one, by convexity) of the boundaries of the hull. We will show later that this boundary cannot contain the north pole. Since $L$ is contained in the bottom half of this line (where $t<0$), the centre will be in the hull if and only if the point of intersection happens for some $t<0$: in fact, since both the north pole and the intersection point are in the hull, the segment connecting both points is contained in it (by convexity). Therefore, the origin is in the hull if and only if the point of intersection corresponds to some $t<0$ (which is equivalent to condition 1).

The intersecting boundary above cannot contain the north pole. In fact, any boundary containing the north pole defines a hyperplane that either intersects the $x_n$ axis only once or contains it, and any hyperplane containing the $x_n$ axis will be defined by a normal that is perpendicular to this axis. In other words, these normals have $x_n=0$ coordinate. However, these normals are random with $0$ probability that their last coordinate is null. As a result, an intersecting boundary needs to be spanned by $p_1, \dots, p_n$.

Now, we just need to show that the $x_n$-axis will intersect the boundary of the hull if and only if condition 2 is satisfied. I won't detail this part too much for the sake of brevity. Let us first show that condition 2 is necessary, in fact project the intersecting boundary perpendicular onto the $x_n=0$ hyperplane. The origin will be in the projection. Then radially expand the projected vertices until they reach the sphere (these will be the vertices projected onto the equator). Their convex hull will contain the initial projection (because the hull contains the perpendicularly projected vertices and a bit more due to the radial expansion), and therefore, it will contain the origin. For the converse, do the opposite, radially shrink the hull in $S^{n-2}$ until their $x_1,\dots,x_{n-1}$ coordinates match those of points $p_1,\dots,p_n$. By construction, the shrunk convex hull will contain the origin and is the perpendicular projection of the boundary defined by $p_1, \dots, p_n$. Therefore the boundary must intersect the $x_n$-axis.

**Finalising the solution**

Let condition 1 above correspond to event $E^n_1$ in the probability space and condition 2 correspond to event $E^{n-1}_2$. Then

$$
q_{n, n+1} = \mathbb{P}(E^n_1|E^{n-1}_2)\mathbb{P}(E^{n-1}_2) = \frac{1}{2} q_{n-1, n},
$$

where we used the fact that $\mathbb{P}(E^n_1|E^{n-1}_2)=1/2$ by symmetry.

Upon applying the obvious condition $q_{1, 2}=1/2$, we get the final solution

$$
q_{n, n+1} = \frac{1}{2^n}.
$$

Notice how an event of probability $1/2$ naturally emerged so we can make contact with fair coin flips: assuming that the points' projection to the {$x_n=0$} hyperplane define a convex hull containing the centre of the sphere, the probability that their hyperplane intersects $L$ is $1/2$. All such hyperplanes would then be mapped to the coin throwing event "head". Obviously, there are infinitely many such hyperplanes, but regardless of which one it is, the centre of the sphere will be in the interior of the convex hull if this condition is satisfied in all subdimensions.

# Way ahead for Problem 2
If I ever find the time I will work out the subtle details to extend the above methodology to Problem 2. The idea is to calculate the probability that, upon choosing the first draw as the north pole, we can find $n$ points among the remaining $N-1$ that will form a convex hull containing the origin just like in Problem 1. Worth noticing that we can choose any of the $N$ points as the north pole without loss of generality.

This is achieved by showing the extension of the recursive relation for $q_{n,n+1}$ derived above

$$
q_{n, N} = \frac{1}{2} q_{n, N-1}+\frac{1}{2} q_{n-1, N-1},
$$

which, with the appropriate boundary conditions, is the same recursion relation that would be derived by the coin flipping problem.