# Some Canonical RL Algos: Part 1 (SAC)

**Michael Pham**  
*July 31, 2026*

<figure>
  <img src="blogs/ppo-figures/pong.png" alt="PufferLib Pong environment" />
  <figcaption>Applying PPO to Play Pong using Pufferlib's Ocean Environment</figcaption>
</figure>

Proximal Policy Optimization (PPO) is a Policy Gradient method for RL that's a pretty default choice for a lot of continuous control problems like robotics and Atari-style games since it runs out of the box with less hyperparameter tuning. PPO gained prominence for its use for Reinforcement Learning with Human Feedback to aligm LLMs before DPO and GRPO became default choices for RL'ing these LLMs. If you want to only read about PPO then you can skip to the PPO section without losing much, but I'm going to build some intuition about different design choices by starting with Trust Region Policy Optimization (TRPO), which was introduced by ["Schulman et al., 2015"](https://arxiv.org/abs/1502.05477), then Generalized Advantage Estimates (GAE) by ["Schulman et al., 2015"](https://arxiv.org/abs/1506.02438) before explaining PPO, another paper by ["Schulman et al., 2015"](https://arxiv.org/abs/1707.06347). Schulman (and the rest of BAIR) was really on a tear in 2015 - easier to say ex-post though. Famously, their PPO paper got rejected from NeurIps in 2017.

You can view [all the code referenced in this write-up here: https://github.com/mpham8/rl-papers/tree/master/ppo-pong](https://github.com/mpham8/rl-papers/tree/master/ppo-pong).


# Trust Region Policy Optimization (Schulman et al., 2015)

If you run some stochastic policy $\pi$ you get the expected discounted reward of  
\[
\eta(\pi) = \mathbb{E}_{s_0,a_0,\ldots}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t)\right]
\]

The advantage function, which is the additional payoff of taking action $a$ at state $s$ over $\mathbb{E}_\pi$ can be computed as 
\[
A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)
\]

The discounted visitation frequency which measures how often the policy $\pi$ visits states $s$ over the course of the trajectory is
\[
\begin{aligned}
\rho_\pi(s) &= P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + \cdots \\
            &= \sum_{t=0}^\infty \gamma^t P(s_t = s)
\end{aligned}
\]
 
So then putting all the above together, if you ran a different policy $\tilde\pi$, its expected return over policy $\pi$ can be written as 
 
\[
\begin{aligned}
\eta(\tilde\pi) &= \eta(\pi) + \mathbb{E}_{s_0,a_0,\ldots \sim \tilde\pi}\left[\sum_{t=0}^\infty \gamma^t A_\pi(s_t,a_t)\right] \\
&= \eta(\pi) + \sum_{t=0}^\infty \sum_s P(s_t = s \mid \tilde\pi) \sum_a \tilde\pi(a\mid s)\, \gamma^t A_\pi(s,a) \\
&= \eta(\pi) + \sum_s \rho_{\tilde\pi}(s) \underbrace{\sum_a \tilde\pi(a\mid s) A_\pi(s,a)}_{\geq 0\ \forall s \ \Rightarrow\ \text{increases } \eta}
\end{aligned}
\]

Intuitively, this last equation is saying that the new policy is better by the second term: across all the states $\sum_s$ the new policy actually tends to spend time in weighted by how much time and how early $\rho_{\tilde\pi}(s)$  of the advantage of the new policy $A_\pi(s,a)$ where that per-state advantage is itself an average over the new policy's action probabilities $\sum_a \tilde\pi(a\mid s)$. With the last equation, in the second term $\rho_{\tilde\pi}(s)$ is always non-negative so if the second part of the term $\tilde\pi(a\mid s) A_\pi(s,a)$ is greater than or equal to 0 for all states, then $\tilde\pi$ is a better policy than $\pi$. However, we're actively searching for a new policy $\tilde\pi$ but to know the new visitation frequency $\rho_{\tilde\pi}(s)$, which depends on the policy, you need to know $\tilde\pi$. So they use a first-order local approximation to $\eta$:
 
\[
L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde\pi(a\mid s) A_\pi(s,a)
\]

They then derive a theorom that proves

\[
    \tag{9}
\eta(\tilde\pi) \;\geq\; \underbrace{L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2} D_{KL}^{max}(\tilde\pi,\pi)}_{\pi_{i+1} = \operatorname*{argmax}_{\pi_i} \text{ this, for algorithm}}
\]

where $D_{KL}^{max}(\tilde\pi,\pi)$ is the largest KL Divergence between $\tilde\pi$ and $\pi$ across all states. So if you could parameterize policies with parmaeters $\theta$ then the best policy $\tilde\pi$ can be found by solving the maximization problem:
 
\[
\max_\theta \Big[ L_{\theta_{old}}(\theta) - C D_{KL}^{max}(\theta_{old},\theta) \Big]
\]

However, they note that step sizes would be too small with that penalty coefficient, so instead they use a trust region constraint $\delta$, and solve the constrained maximization problem.

\[
\max_\theta L_{\theta_{old}}(\theta) \quad \text{s.t.} \quad \underbrace{D_{KL}^{max}(\theta_{old},\theta) \leq \delta}_{\text{trust region constraint}}
\]

In practice instead of finding the maximal KL Divergence across all possible states which would be computationally infeasible, they instead sample from the rollouts:

\[
\bar D_{KL}^\rho(\theta_1,\theta_2) = \mathbb{E}_{s\sim\rho}\left[D_{KL}\big(\pi_{\theta_1}(\cdot\mid s)\,\|\,\pi_{\theta_2}(\cdot\mid s)\big)\right]
\]
 

The objective can be rewritten
\[
\begin{aligned}
\max_\theta L_{\theta_{old}}(\theta) 
    &= \max_\theta \sum_s \rho_{\theta_{old}}(s) \sum_a \pi_\theta(a\mid s) A_{\theta_{old}}(s,a) \\[4pt]
    &= \max_\theta \frac{1}{1-\gamma} \mathbb{E}_{s\sim\rho_{\theta_{old}}} \left[ \sum_a \pi_\theta(a\mid s) A_{\theta_{old}}(s, a) \right] \qquad \text{(expand $\rho$)} \\[4pt]
    &= \max_\theta \frac{1}{1-\gamma} \mathbb{E}_{s\sim\rho_{\theta_{old}}} \left[ \sum_a \pi_\theta(a\mid s) Q_{\theta_{old}}(s,a) - V_{\theta_{old}}(s) \right] \\[4pt]
    &= \max_\theta \frac{1}{1-\gamma} \mathbb{E}_{s\sim\rho_{\theta_{old}}} \left[ \sum_a \pi_\theta(a\mid s) Q_{\theta_{old}}(s,a) \right] \qquad \text{(since $V(s) = \sum_a \pi_{old}(a|s)Q(s,a)$ and does not depend on $\theta$)} \\[4pt]
    &= \max_\theta \frac{1}{1-\gamma} \mathbb{E}_{s\sim\rho_{\theta_{old}}} \left[ \sum_a q(a\mid s) \frac{\pi_\theta(a\mid s)}{q(a\mid s)} Q_{\theta_{old}}(s,a) \right] \qquad \text{(importance sampling)} \\[4pt]
    &= \max_\theta \frac{1}{1-\gamma} \mathbb{E}_{\substack{s\sim\rho_{\theta_{old}} \\ a\sim q}} \left[ \frac{\pi_\theta(a\mid s)}{q(a\mid s)} Q_{\theta_{old}}(s,a) \right]
\end{aligned}
\]
 
where again actions are sampled from the old policy. Therefore, in practice the contrained maximization that is solved in the algorithm is:

\[
\begin{aligned}
&\max_\theta \; \frac{1}{1-\gamma} \; \mathbb{E}_{\substack{s\sim\rho_{\theta_{old}} \\ a\sim q}} \left[ \frac{\pi_\theta(a\mid s)}{q(a\mid s)} Q_{\theta_{old}}(s,a) \right] \\
&\text{subject to} \quad \mathbb{E}_{s\sim\rho_{\theta_{old}}}\left[D_{KL}\left(\pi_{\theta_{old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\right)\right] \leq \delta
\end{aligned}
\]

An issue with this algorithm is that the full constrained optimization problem is really expensive to solve, so they suggest linearizing the objective, quadratically approximating the KL constraint, and then turning this into a quadratic program to be solved with Conjugate gradient - this exact algorithm is not so important. PPO later addresses this.


# High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)


# Proximial Policy Optimization Algorithms (Schulman et al., 2015)

