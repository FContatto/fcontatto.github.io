---
layout: post
title: "Gaussian Hidden Markov Model: training and decoding"
author: "Felipe Contatto"
categories: journal
tags: [machine learning, hmm]
comments: true
---

In this post, I present the details of an elegant machine learning model: the hidden Markov model. I derive all the mathematical relations used in its training and decoding, with the appropriate motivation at each step so the various definitions feel natural to any reader familiar with basic probability. The main goal is to derive formulas within the Gaussian emission framework for a univariate time-series, but this post is generic enough to be generalised to other emission distributions and, with slightly more work, to multivariate series.

Hidden Markov models have several interesting applications, especially in [NLP](https://en.wikipedia.org/wiki/Natural_language_processing) as it became a pivotal algorithm for speech tagging. In a future post, I will describe uses of the same model within finance, hence the motivation to present the Gaussian framework specifically here.

# The hidden Markov model

Suppose you have a time-series $(y_1, \dots, y_T)$ where you know that each $y_t\, (t=1,\cdots,T)$ is normally distributed. Moreover, assume that each $y_t$ is a variate of one of $N$ normal distributions $\{\mathcal{N}(\mu_i, \sigma^2_i): i=1,\dots, N\}$, but you do not know which one and you also do not know the distributions' parameters $(\mu_i,\sigma^2_i)$. The task is then to determine all of these $2N$ parameters (training the model) and from which distribution each $y_t$ was most likely sampled (decoding), given the whole time series. This is pretty much a Gaussian mixture model within a time series.

Throught this post, we will use letters $i,j,k,\dots$ to indicate the distribution indices and $t$ as the letter representing indices of the time-series. We also denote $s_t$ the random variable determining which distribution $y_t$ is coming from: $s_t=i$ corresponds to the event $y_t\sim\mathcal{N}(\mu_i,\sigma^2_i)$. So $s_t$ is the **hidden state** of the time-series at time $t$. It is hidden because we cannot observe the state directly, in fact all we can observe are the $y_t$, which are typically called the **observables**. From now on, we will use the letter $f$ to denote any probability density function, in particular $f_i$ is the density function of $\mathcal{N}(\mu_i,\sigma^2_i)$, whereas discrete probability mass functions will be denoted by $\mathbb{P}$, for instance $\mathbb{P}(s_t)$ is the probability that the series was at state $s_t$ at time $t$.

Let us now turn our attention to both the decoding and training problems.

## Decoding problem

Assume, for now, that all the distributions' parameters $\{(\mu_i,\sigma^2_i)\}$ have been fixed. We want to find a time-series of states $s^\ast_1,\dots,s^\ast_T$ so that the join likelihood $f(s_1,\dots, s_N, y_1,\dots, y_T)$ (i.e. the joint probability density function of all the states and observables) is maximised. The $y_t$'s are fixed as they were observed, we now just need to calculate

$$
(s^\ast_1,\dots,s^\ast_T) = \arg\max_{s_1,\dots,s_T}f(s_1,\dots, s_N, y_1,\dots, y_T).
$$

A naive approach would be to do a brute force search through all the possible sequences of states $s_1,\dots,s_T$ and see which one maximises the likelihood. However, since there are $N$ possible states at each timestamp, the decoding algorithm would have exponential runtime $O(N^T)$, which quickly becomes intractable. We can do much better with a simple dynamic programming approach.

### Viterbi algorithm

In order to greatly improve the runtime of the decoding algorithm, let us rewrite it in a convenient way. First, use the definition of conditional probability to get

$$
\begin{align}
f(s_1,\dots, s_T, y_1,\dots, y_T) =& f(y_T\vert s_1,\dots, s_T, y_1,\dots, y_{T-1}) f(s_1,\dots, s_T, y_1,\dots, y_{T-1}) \\
=& f(y_T\vert s_1,\dots, s_T, y_1,\dots, y_{T-1}) f(s_T\vert s_1,\dots, s_{T-1},y_1,\dots, y_{T-1}) \\
\times & f(s_1,\dots, s_{T-1},y_1,\dots, y_{T-1}) \\
=& f(y_T\vert s_T) \mathbb{P}(s_T\vert s_{T-1})f(s_1,\dots, s_{T-1},y_1,\dots, y_{T-1}).
\end{align}
$$

Notice how we managed to factor out the $T$ components, giving a recursion relation that is the cornerstone of the dynamic programming framework.

For convenience, let us define a function $\nu_T(s_T) = \max_{s_1,\dots,s_{T-1}}f(s_1,\dots, s_T, y_1,\dots, y_T)$. At the final step, we want to find $\max_{s_T}\nu_T(s_T)$. With these definitions,

$$
\begin{align}
\nu_T(s_T) =& \max_{s_1,\dots,s_{T-1}}f(s_1,\dots, s_T, y_1,\dots, y_T) \\
=& f(y_T\vert s_T)\max_{s_{T-1}}\left(\mathbb{P}(s_T\vert s_{T-1})\max_{s_1,\dots,s_{T-2}}f(s_1,\dots, s_{T-1},y_1,\dots, y_{T-1})\right) \\
=& f(y_T\vert s_T)\max_{s_{T-1}}\left(\mathbb{P}(s_T\vert s_{T-1})\nu_{T-1}(s_{T-1})\right)
\end{align}
$$

and the recursion starts with $\nu_1(s_1)=f(y_1\vert s_1)\mathbb{P}(s_1)$.

The above equation implies that, for each possible value of $s_t$, when we calculate $\nu_t(s_t)$ we are actually fixing the value of the state $s_{t-1}$, which will then be unchanged thereafter. 

For each fixed $s_T$, the calculation of $\nu_T(s_T)$ takes $O(N)$ runtime. Therefore, the calculation of the whole function $s_T\mapsto \nu_T(s_T)$ takes $O(N^2)$. Since the time-series has length $T$, the overall runtime to calculate the entire state time-series $\{s_t\}$ is just $O(TN^2)$, a much improved metric compared to the brute force approach. On top of that, if we ignore the memory required to store the state time-series, the extra memory required for the algorithm is just $O(1)$ if we keep track of the $\max_{s_{T-1}}\left(\mathbb{P}(s_T\vert s_{T-1})\nu_{T-1}(s_{T-1})\right)$ as we calculate $s_{T-1}\mapsto \nu_{T-1}(s_{T-1})$.

## Training problem

When solving the decoding problem, we first assumed that we knew all the parameters of the model (transition, emission and initial state probabilities). In other words, we assumed the model had already been trained. If, on the other hand, we had the sample of hidden states, then we could estimate the parameters by maximising the likelihood $f(s_1,\dots,s_T,y_1,\dots,y_T)$ by seeing it as a function of the model parameters.

Unfortunately, we have neither and we have to maximise the likelihood by by seeing it as a function of the $T$ hidden states and of the model parameters. This is a complex task that is hard to solve exactly: we could, in theory, consider all $T^N$ possible state series, maximise the likelihood for each series to get a set of parameters and the solution to the problem will be the series and parameter set that gives the highest likelihood, but this approach quickly becomes unfeasible due to the runtime complexity. Gradient ascent is also not an option as the states are discrete variables. Instead, one common approach to solve such problems is the [expectation maximisation (EM) algorithms](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).

For the sake of brevity, I will avoid giving proofs about the EM algorithms here (which would deserve a separate post), and instead will just use it to derive the trainig algorithm ([Baum-Welch](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)) for our particular problem within the Gaussian HMM framework.

### Baum-Welch algorithm

The goal is to maximise the expected log-likelihood iteratively as described by the steps below. Let $\theta$ represent the model parameters (transition, emission and initial state probabilities) and let $f_\theta$ represent the density function according to these parameters. The EM algorithm follows the steps

>1. Guess some initial value $\theta_0$ for $\theta$
>2. E-step: calculate 
>$g(\theta,\theta_0) = \mathbb{E}_{S_1,\dots,S_T\sim f_{\theta_0}}[\log f_\theta(S_1,\dots,S_T,y_1,\dots,y_T)\vert y_1,\dots,y_T]$
>3. M-step: maximise $\theta_1=\arg\max_{\theta}g(\theta,\theta_0)$
>4. Set $\theta_0=\theta_1$.
>5. Iterate over steps 2, 3 and 4 until the log-likelihood does not change by more than a predefined tolerante (or until the maximum number of iterations allowed by the user is reached).

Let us now describe steps 2 and 3.

#### E-step

The expectation function can be written as

$$
g(\theta,\theta_0) = \sum_{i,...,k=1}^N \log f_\theta(S_1=i,\dots,S_T=k,y_1,\dots,y_T) f_{\theta_0}(S_1=i,\dots,S_T=k|y_1,\dots,y_T).
$$


When presenting the Viterbi algorithm, we showed a recursion relation for the likelihood function that can easily be extended to

$$
f_\theta(s_1,\dots, s_T, y_1,\dots, y_T) = f(y_1\vert s_1)\cdots f(y_T\vert s_T) \mathbb{P}(s_2\vert s_1)\cdots \mathbb{P}(s_T\vert s_{T-1}) \mathbb{P}(s_1).
$$

Let us introduce some convenient notation for the marginal distributions of $f_{\theta_0}$

$$
\begin{align}
\gamma_t(i) =& f_{\theta_0}(S_t=i\vert y_1,\dots,y_T) = \sum_{i_1,\dots,i_{t-1}, i_{t+1},\dots,i_T=1}^N f_{\theta_0}(S_1=i_1,\dots,S_T=i_T\vert y_1,\dots,y_T) \\
\xi_t(i,j) =& f_{\theta_0}(S_t=i,S_{t+1}=j\vert y_1,\dots,y_T) = \sum_{i_l=1, l\not= i,j}^N f_{\theta_0}(S_1=i_1,\dots,S_T=i_T\vert y_1,\dots,y_T)
\end{align}
$$

By replacing the equation for $f_\theta$ in the $g$ definition, expanding the log of products into a sum and using the $\gamma_t$ and $\xi_t$ notations for the marginal distributions of $f_{\theta_0}$, we get

$$
g(\theta,\theta_0) = \sum_{i=1}^N\log(\pi_i) \gamma_1(i) + \sum_{t=1}^{T-1}\sum_{i,j=1}^N\log(A_{ij}) \xi_t(i,j)+\sum_{i=1}^N\sum_{t=1}^{T}\log(f_i(y_t))\gamma_t(i),
$$

where $\pi_i = \mathbb{P}(S_1=i)$, $A_{ij} = \mathbb{P}(S_{t+1}=j\vert S_{t+1}=i)$ (which is $t$-independent) and $f_i(y_t) = f(y_t\vert S_t=i)$ is the emission probability of state $i$ given by

$$
f_i(y_t)=\frac{1}{\sqrt{2\pi \sigma^2_i}}e^{-\frac{(y_t-\mu_i)^2}{2\sigma_i^2}}.
$$

We now have our target function $g$ written explicitly in terms of the model parameters $$\theta=(\{\pi_i\}_i,\{A_{ij}\}_{i,j},\{\mu_i\}_i, \{\sigma^2_i\}_i)$$.

#### M-step

We need to maximise $\theta \mapsto g(\theta,\theta_0)$ conditional on the constraints $\sum_{i=1}^N \pi_i = 1$ and $\sum_{j=1}^N A_{ij} = 1\, \forall i=1,\dots,N$. This constrained maximisation problem can be achieved by Lagrange multipliers. In fact, maximising $g$ under these constraints is the same as maximising the Lagrangian (under the same constraints)

$$
\mathcal{L}(\theta,\theta_0) = g(\theta,\theta_0) - \lambda_1 \left(\sum_{i=1}^N \pi_i - 1\right) - \sum_{i=1}^N\lambda_{2,i} \left(\sum_{j=1}^N A_{ij} - 1\right),
$$

where the $\lambda$'s are the Lagrange multipliers.

One of the maximisation equations is $\frac{\partial\mathcal{L}}{\partial\pi_i}=0$ which, together with the constraint on the $\pi_i$'s (which will result in $\lambda_1=1$) gives

$$
\pi^\ast_i=\gamma_1(i), \; i=1,\dots,N.
$$

Setting the differentials with respect to $A_{ij}$ to $0$ and imposing the constraints will give

$$
A^\ast_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}.
$$

Finally, maximising with respect to the emission distribution's parameters $\mu_i$ and $\sigma_i$ gives

$$
\begin{align}
\mu^\ast_i =& \frac{1}{T}\sum_{t=1}^T\gamma_t(i)y_t \\
(\sigma^\ast)^2_i =& \frac{1}{T}\sum_{t=1}^T\gamma_t(i)(y_t-\mu_i)^2.
\end{align}
$$

As such, these $\ast$-ed parameters correspond to the $\theta_1$ parameter in step 4.

Now, we need to describe how to calculate the $\gamma_t$ and $\xi_t$ functions. First, define

$$
\begin{align}
\alpha_t(i)=&f(S_t=i,y_1,\dots,y_t) \\
\beta_t(i)=&f(y_{t+1},\dots,y_t\vert S_t=i).
\end{align}
$$

Then, applying the definition of condition expectation and using Markovian assumptions, we have

$$
\begin{align}
\xi_t(i,j) =& f(S_t=i, S_{t+1}=j\vert y_1,\dots,y_T)=\frac{f(S_t=i,S_{t+1}=j, y_1,\dots,y_{t+1},y_{t+2},\dots,y_T)}{f(y_1,\dots,y_T)} \\
=& \frac{f(y_{t+2},\dots,y_T\vert S_t=i,S_{t+1}=j, y_1,\dots,y_{t+1})f(y_{t+1}\vert S_{t}=i,S_{t+1}=j,y_1,\dots,y_{t})}{f(y_1,\dots,y_T)} \\
\times & f(S_{t+1}=j\vert S_t=i,y_1,\dots,y_{t})f(S_t=i,y_1,\dots,y_{t}) \\
=&\frac{f(y_{t+2},\dots,y_T\vert S_{t+1}=j)f(y_{t+1}\vert S_{t+1}=j)f(S_{t+1}=j\vert S_t=i)f(S_t=i,y_1,\dots,y_{t})}{f(y_1,\dots,y_T)} \\
=&\frac{f(y_{t+2},\dots,y_T\vert S_{t+1}=j)f(y_{t+1}\vert S_{t+1}=j)f(S_{t+1}=j\vert S_t=i)f(S_t=i,y_1,\dots,y_{t})}{f(y_1,\dots,y_T)} \\
=&\frac{\beta_{t+1}(j)f_i(y_{t+1})A_{ij}\alpha_t(i)}{\sum_{k,l=1}^N \beta_{t+1}(l)f_i(y_{t+1})A_{kl}\alpha_t(k)}
\end{align}
$$

and

$$
\begin{align}
\gamma_t(i) =& \mathbb{P}(S_t=i\vert y_1,\dots,y_T) = \frac{f(S_t=i, y_1,\dots,y_T)}{f(y_1,\dots,y_T)} \\
=&\frac{f(y_{t+1},\dots,y_T\vert S_t=i,y_1,\dots,y_t)f(S_t=i,y_1,\dots,y_t)}{f(y_1,\dots,y_T)} \\
=&\frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^N\alpha_t(j)\beta_t(j)}.
\end{align}
$$

Finally, we just need to show how to calculate $\alpha$ and $\beta$. This will be done with the [forward-backward algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm), which is another dynamic programming procedure.

##### Forward-backward algorithm

The forward-backward recursion relations can be derived from the definition of conditional probability and the Markovian conditions, just as used above.

Fot $t>1$,

$$
\begin{align}
\alpha_t(i) =& f(S_t=i,y_1,\dots,y_t) = \sum_{j=1}^N f(S_{t-1}=j, S_t=i, y_1,\dots,y_t) \\
=& \sum_{j=1}^N f(y_t\vert S_{t-1}=j, S_{t}=i, y_1,\dots,y_{t-1})f(S_{t}=i\vert S_{t-1}=j, y_1,\dots,y_{t-1})f(S_{t-1}=j, y_1,\dots,y_{t-1})\\
=& \sum_{j=1}^N f(y_t\vert S_{t}=i)f(S_{t}=i\vert S_{t-1}=j)\alpha_{t-1}(j).
\end{align}
$$

Thus, the forward part of the algorithm is given by

$$
\begin{align}
\alpha_t(i) =& \sum_{j=1}^N f_i(y_t)A_{ji}\alpha_{t-1}(j) \\
\alpha_1(i)=& f_i(y_1) \pi_i.
\end{align}
$$

The recursion for $\beta$ is found in a similar fashion. For $t>1$,

$$
\begin{align}
\beta_{t-1}(i) =& f(y_t,\dots,y_T\vert S_{t-1}=i) = \sum_{j=1}^N f(S_t=j,y_t,\dots,y_T\vert S_{t-1}=i) \\
=& \sum_{j=1}^N f(y_{t+1},\dots,y_T\vert S_{t-1}=i, S_t=j, y_t)f(y_t\vert S_{t-1}=i,S_t=j)f(S_t=j\vert S_{t-1}=i) \\
=& \sum_{j=1}^N f(y_{t+1},\dots,y_T\vert S_t=j)f(y_t\vert S_t=j)A_{ij}.
\end{align}
$$

Thus, the backward part of the algorithm is given by
$$
\begin{align}
\beta_{t-1}(i) =& \sum_{j=1}^N \beta_t(j)f_j(y_t)A_{ij} \\
\beta_T(i) =& 1.
\end{align}
$$

Just like the Viterbi algorithm, this approach gives a $O(TN^2)$ runtime.

## Conclusion

The algorithms presented here were of a somewhat frequentist nature. In fact, we were mostly motivated by finding maximum likelihood procedures and we never talked about priors and posteriors. The description of the Bayesian approach, whereby we would seek [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) estimates is not dissimilar to the above, but it would be better left for a separate post where I would describe conjugate priors and adapt both the Viterbi and Baum-Welch algorithms. Worth mentioning that the Bayesian approach is what is implemented in the [hmmlearn](https://hmmlearn.readthedocs.io/) Python library, which spun out of scikit-learn some some ago.