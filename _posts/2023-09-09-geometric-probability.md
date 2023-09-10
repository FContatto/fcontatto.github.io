---
layout: post
title: "Geometric probability: an (even) harder problem on the hardest test"
author: "Felipe Contatto"
categories: journal
tags: [sample]
image: cutting.jpg
---

One of the most famous questions from Putnam competitions is number A6 from the 53rd Putnam of 1992 (c.f., for instance, [here](https://prase.cz/kalva/putnam/putn92.html)). The problem is: if you sample 4 points uniformly over a sphere, what is the probability that the tetrahedron defined by them contains the centre of the sphere? The problem is simple to understand and there are solutions easily found on the internet, one of which from [3Blue1Brown](https://youtu.be/OkmNXy7er84?feature=shared).

One possible generalisation of the problem is: 
>Problem 1: given a (n-1)-sphere in n dimensions, sample n points uniformly over it, what is the probability $q_{n,N} that their convex hull contains the centre? 
And an even more general one is:
>Problem 2: given a (n-1)-sphere in n dimensions, sample N points uniformly over it, what is the probability $q_{n,N}$ that their convex hull contains the centre?

Another way to pose problem 2 is by asking the probability $q_{n,N}$ that N points fall in the same hemisphere. This problem was solved by [Wendel back in 1962](https://www.mscand.dk/article/view/10655/8676) by using basic combinatorics, topology and linear algebra. Remarkably, the answer to Wendel's problem is equal to the probability that, upon flipping N-1 fair coins, you get less than n heads, which is given by (assuming $N>n$)
$$
p_{n,N} = \sum_{k=0}^{n-1}P(\text{flipping exactly k heads in N-1 throws}) = \frac{1}{2^{N-1}}\sum_{k=0}^{n-1}\binom{N-1}{k} 
$$