# Some Canonical RL Algos: Part 1 (SAC)

**Michael Pham**  
*July 25, 2026*

<figure>
  <img src="blogs/sac-figures/pufferlib-drone.png" alt="PufferLib drone environment" />
  <figcaption>Applying SAC to Fly Drones using Pufferlib's Ocean Environment</figcaption>
</figure>

I'm going start writing up some foundational RL papers in case this is ever helpful to anyone - I found that while reading RL papers that they try to keep them concise and skip over some background knowledge or steps in derivations. Also, I feel like explaining everything makes me find gaps in my own understanding. I'm going to try to explain the derivation and actual algorithm as simply as possible with only the assumption of an introductory background in RL. The algorithm I'm starting this series with is **Soft Actor-Critic (SAC-v1)**, which was introduced by Haarnoja et al. in ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290). This is one of the go-to RL algos for continuous control problems.

You can view [all the code referenced in this write-up in this: https://github.com/mpham8/rl-papers/tree/master/sac-drone](https://github.com/mpham8/rl-papers/tree/master/sac-drone).




# Preliminaries


## KL Divergence

Kullback-Leibler (KL) Divergence is an important concept that will come up in SAC. KL Divergence is a measure of how one distribution differs from another distribution. For two discrete distributions A and B, KL Divergence is defined as:

$$
D_{KL}(A \parallel B) = \sum_{x} A(x) \log \frac{A(x)}{B(x)}
$$

$\log \frac{A(x)}{B(x)}$ represents the extra "surprise" you pay by using distribution $B$ as your model when the true distribution is actually $A$ for each outcome $x$, and it's weighted by $A(x)$. So if $A$ and $B$ were identical then $\log \frac{A(x)}{B(x)}$ would be 0 for all $x$ so their KL Divergence would be the smallest it can be: 0. 

## Entropy

In RL, an issue with greedy algorithms is that with non-stationary problems, there may not be enough exploration to achieve the optimal policy. One way to mitigate this is with entropy. The "soft" in Soft Actor-Critic refers to the inclusion of entropy. Entropy measures how spread out a particular distribution is. Under a discrete distribution entropy is defined as
$$
\mathcal{H}(\pi(\cdot \mid s)) = - \sum_a \pi(a \mid s) \log \pi(a \mid s)
$$

for actions $a$ and states $s$. You can think of the $\sum_a \pi(a \mid s)$ part as taking an average or expected value. It's the same structure as the law of total expectation: $E[x] = \sum_b P(x \mid b) \ P(b)$ — you weight each conditional value by the probability of the condition, then sum. And what we're weighing is $\log \pi(a \mid s)$, the log probability we took an action $a$ given that we were in state $s$. In other words this entropy is on average how surprised we expect to be by a draw from our action distribution: 
$$
\mathcal{H} = \mathbb{E}_{a \sim \pi(\cdot \mid s)} \left[ -\log \pi(a \mid s) \right]
$$

If all weight is one point (it's deterministic) then $\mathcal{H} = 0$ and if all weight is uniform over $n$ points (as spread out as a distribution can be) then $\mathcal{H} = \log(n)$ ($n \to \infty$, $\mathcal{H} \to \infty$). Typically, papers previously to this had added the entropy term to the loss function, but here they add the entropy term to the reward itself: 

$$
V(s_t) = \mathbb{E}_{a_t}\left[ r(s_t, a_t) + \mathcal{H} \right] + \gamma \mathbb{E}_{a_t, s_{t+1}}\left[ V(s_{t+1}) \right]
$$

Entropy as loss regularization affects the current state only - this spreads out the current action distribution but the critic (value) function never learns from this. Entropy added directly to the reward however encourages exploration by crediting the entropy of future states to the current state-action  through discounted recursion. The value function learns to prefer paths to lead to high-entropy regions of the state space. Imagine going left and right both have the same discounted reward, but going left locks you into a super high probability/deterministic path and going right leads you to a path with a lot of not so confident moves - entropy in the reward directly incentivizes going right.

# SAC comes from Policy Iteration


The Soft Actor-Critic is based on the dynamic programming policy iteration algorithm. If you recall, policy iteration involves alternating between policy evaluation (computing the value function for the current policy) and policy improvement (updating the policy by acting greedily). 

# Soft Q-Value Function Approximation

## Soft Q-Value Derivation

The policy evaluation step is implemented by computing Q-value iteratively. The Q-function is defined as:

$$
Q(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} \left[ V(s_{t+1}) \right]
$$

Instead of solving for the Q-value exactly (as in the tabular soft policy iteration), SAC uses a function approximation for the Q-value in the form of a 2-layer neural network with 256 neurons in each hidden layer. These neural networks are implemented in the <code>model.py</code> file:

```{python}
class SoftQFunction(nn.Module):
    def __init__(self, num_states, num_actions, hidden_layer) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_states + num_actions, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)
        return self.fc(x)
```

## Soft Q-Value Loss

In SAC, they use two different Q-functions that update in parallel and set the Q-vlue as the elementwise minimun of the two Q-functions. This is because the policy improvement steps introduces a positive bias. Each Q-function's loss is:

$$
\tag{7}
J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim D} \left[ \frac{1}{2} \left( Q_\theta(s_t, a_t) - \hat{Q}(s_t, a_t) \right)^2 \right]
$$

where \((s_t, a_t)\) are sampled from the replay buffer \(D\) (a replay buffer is just a data structure that holds the last N transitions $(s_t, a_t, r(s_t, a_t), s_{t+1}, terminal_t)$), \(Q_\theta(s_t, a_t)\) is the predicted Q-value under the current neural network parameters $\theta$, and \(\hat{Q}(s_t, a_t)\) is the target Q-value. SAC samples station-action pairs$(s_t, a_t)$ from the replay buffer here because we just need arbitrary $(s_t, a_t)$ pairs to update their Q-values so it doesn't matter which policy originally produced it - sampling from the replay buffer gives better data efficiency and decorrelates updates for Q-value. The target Q-value is:

$$
\tag{8}
\hat{Q}(s_t, a_t) = r(s_t, a_t) + \gamma \; \mathbb{E}_{s_{t+1} \sim p} \left[ V_{\bar{\psi}}(s_{t+1}) \right]
$$

where $V_{\bar{\psi}}$ denotes the delayed target value function (which we will get to). The loss functions and actual optimization steps are implemented in the <code>train_step()</code> function of the <code>agent.py</code> file . <code>torch.no_grad()</code> is applied to the target value and target Q-value computations because we don't need gradients flowing back through them. Only the "live" $Q_{\theta_1}$ and $Q_{\theta_2}$ networks evaluated on the current state-action pairs from the replay buffer are being optimized here.


```{python}
#update soft q function
  with torch.no_grad():
      v_target_next = model_vtarget(states_next)
      q_target = cfg['REWARD_SCALE'] * rewards + cfg['GAMMA'] * v_target_next* (1-terminals)

  q1_buffer = model_q1(states, actions)
  q2_buffer = model_q2(states, actions)
  loss_q1 = 0.5 * (q1_buffer - q_target).pow(2).mean()
  loss_q2 = 0.5 * (q2_buffer - q_target).pow(2).mean()
```

# Soft Value Function Approximation

## Soft Value Derivation

The value function is also approximated with a 2 hidden layer with 256 neurons each neural network:

```{python}
class SoftValueFunction(nn.Module):
    def __init__(self, num_states, hidden_layer) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_states, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, 1)
        )

    def forward(self, x):
        return self.fc(x)
```

You can substitute the Q-function into the previously written value function to get:
$$
\begin{aligned}
V(s_t) &= \mathbb{E}_{a_t}\left[ Q(s_t, a_t) + \mathcal{H} \right] \\
&= \mathbb{E}_{a_t}\left[ Q(s_t, a_t) - \log \pi(a_t|s_t) \right]
\end{aligned}
$$

## Soft Value Loss


SAC uses this as a target for the value function, so the loss comes out to:

$$
\tag{5}
J_V(\psi) = \mathbb{E}_{s_t \sim D} \left[ \frac{1}{2} \left( V_{\psi}(s_t) - \mathbb{E}_{a_t \sim \pi_\phi} \left[ Q_\theta(s_t, a_t) - \log \pi_\phi(a_t | s_t) \right] \right)^2 \right]
$$

where $\psi$ are the parameters of the value function network. The Q-values are not from the replay buffer this time. The Q-values used in the value function are the Q-values sampled under the current policy (which we will get to). We don't use the actions sampled from the replay buffer because we're trying to assess what the state is worth under the current policy, not the old policy that was used to sample an action when the state-action pair was added to the replay buffer.

```{python}
#update soft value function
action_current_policy, log_prob, _ = model_p.sample(states)

q1_current_policy = model_q1(states, action_current_policy)
q2_current_policy = model_q2(states, action_current_policy)
q = torch.min(q1_current_policy, q2_current_policy)

v = model_v(states)
loss_v = 0.5 * (v - (q - log_prob).detach()).pow(2).mean()
```

## Soft Value Target

the Q-value loss, instead of using the soft value network $V_\psi(s_{t+1})$ directly, uses a separately maintained soft value target, denoted $V_{\bar{\psi}}(s_{t+1})$. The parameters $\bar\psi$ in this target network are updated to slowly track the main value network’s parameters using a moving average. This helps stabilize learning by providing a more consistent target for the Q-value loss, reducing harmful feedback loops that can occur if both networks are updated simultaneously. The update rule for the target parameters is:
$$
\bar\psi \leftarrow \tau \psi + (1 - \tau) \bar\psi
$$
where $\tau$ is a small constant.


```{python}
#update value target
with torch.no_grad():
    for tp, p in zip(model_vtarget.parameters(), model_v.parameters()):
        tp.data.copy_(cfg['TAU'] * p.data + (1 - cfg['TAU']) * tp.data)
```

# Policy Function Approximation


## Policy Derivation

In the policy iteration that SAC is based on, the other step is policy improvement. Policy improvement is the step that allows the policy to approach the optimal policy. This involves taking the greedy step, which in the case of a maximization problem is taking policy parameters that argmax the desired value function $F(a)$: 
$$
\pi^* = \underset{\pi}{\arg\max} \; \mathbb{E}_{a \sim \pi(\cdot|x)} \left[ F(a) \right]
$$

In SAC, for polict improvement our goal is pick the polict that maximizes the value function $V(s_t)$, which can be re-written as:

$$
\begin{aligned}
V(s_t) &= \mathbb{E}_{a_t}\left[ Q_\theta(s_t, a_t) - \log \pi(a_t|s_t) \right] \\
&= \sum_{a_t} \pi(a_t|s_t) \left[ Q_\theta(s_t, a_t) - \log \pi(a_t|s_t) \right] \\
&= \sum_{a_t} \pi(a_t|s_t) Q_\theta(s_t, a_t) - \sum_{a_t} \pi(a_t|s_t) \log \pi(a_t|s_t)
\end{aligned}
$$

with the constraint that the action probabilities (the policy $\pi(a_t|s_t)$) form a valid probability distribution and sum to one. This is a constrained optimization problem: 

$$
\max_{\pi} \; \sum_{a_t} \pi(a_t|s_t) Q_\theta(s_t, a_t) - \sum_{a_t} \pi(a_t|s_t) \log \pi(a_t|s_t)
\quad \text{s.t.} \quad \sum_{a_t} \pi(a_t|s_t) = 1 
$$

This can be solved by setting up the Lagrangian:

$$
\mathcal{L}(\pi, \lambda) = \sum_{a_t} \pi(a_t|s_t) Q_\theta(s_t, a_t) - \sum_{a_t} \pi(a_t|s_t) \log \pi(a_t|s_t) + \lambda \left(1 - \sum_{a_t} \pi(a_t|s_t)\right)
$$

With first-order conditions:
$$
\frac{\partial \mathcal{L}}{\partial \pi(a_t|s_t)} = Q_\theta(s_t, a_t) - \log \pi(a_t|s_t) - 1 - \lambda = 0
\implies \pi(a_t|s_t) = e^{Q_\theta(s_t, a_t)} \, e^{-(1+\lambda)}
$$

If you let $z = e^{1+\lambda}$, then the greedy policy can be rewritten as:

So, the greedy policy is:

$$
\pi(a_t|s_t) = \frac{e^{Q_\theta(s_t, a_t)}}{z}
$$

We can't compute this closed-form policy directly (especially in continuous action spaces), so instead we find a tractable $\pi$ (by adjusting $\phi$ which are the parameters of the policy network $\pi$) that stays as close as possible to $z = e^{1+\lambda}$ by minimizing KL divergence:

$$
\pi = \arg\min_{\phi}
  \mathbb{E}_{s_t \sim \mathcal{D}}
  \left[
    D_{\mathrm{KL}}\left(
      \pi_\phi(\cdot \mid s_t)
      \,\Big\|\,
      \frac{e^{Q_\theta(s_t, \cdot)}}{Z_\theta(s_t)}
    \right)
  \right]
$$

which by the definition of KL Divergence is

$$
\begin{aligned}
&= \arg\min_{\phi}
  \mathbb{E}_{s_t \sim \mathcal{D}}
  \mathbb{E}_{a_t \sim \pi_\phi}
  \left[
    \log \pi_\phi(a_t \mid s_t)
    - \log\left(\frac{e^{Q_\theta(s_t, a_t)}}{Z_\theta(s_t)}\right)
  \right] \\
&= \arg\min_{\phi}
  \mathbb{E}_{s_t \sim \mathcal{D}}
  \mathbb{E}_{a_t \sim \pi_\phi}
  \left[
    \log \pi_\phi(a_t \mid s_t)
    - Q_\theta(s_t, a_t)
    + \log Z_\theta(s_t)
  \right] \\
  \end{aligned}

$$

and since $Z$ is independent of the policy parameters, we can drop that term in this minimization problem 

$$
\begin{aligned}
&= \arg\min_{\phi}
  \mathbb{E}_{s_t \sim \mathcal{D}}
  \mathbb{E}_{a_t \sim \pi_\phi}
  \left[
    \log \pi_\phi(a_t \mid s_t)
    - Q_\theta(s_t, a_t)
  \right]
\end{aligned}
$$

SAC makes two more practical assumptions. First, it assumes $\pi$ is Gaussian, since Gaussians are easy to parameterize — a neural network just needs to output a mean $\mu$ and a standard deviation $\sigma$. Second, and more importantly for training, Gaussians support the reparameterization trick: any Gaussian sample $a_t \sim \pi_\phi(\cdot|s_t)$ can be rewritten as a deterministic, differentiable function of the policy parameters and independent noise, $a_t = \mu_\phi(s_t) + \sigma_\phi(s_t)\cdot\epsilon_t$ with $\epsilon_t \sim \mathcal{N}(0,1)$. This avoids treating the sampling step as a black box and instead lets gradients flow directly through $\phi$.

You can see this in the model code for a policy network. The policy network is once again a 2 hidden layer with 256 neurons on each. However, now there's 2 linear mean and std layers that map to each action's mean and standard deviation:

```{python}
class PolicyFunction(nn.Module):
    def __init__(self, num_states, num_actions, hidden_layer, log_std_min, log_std_max) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc = nn.Sequential(
            nn.Linear(num_states, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_layer, num_actions)
        self.std = nn.Linear(hidden_layer, num_actions)

    def forward(self, x):
        x = self.fc(x)

        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
```

This PolicyFunction class also needs a way to sample actions.

```
def sample(self, x):
    mean, log_std = self.forward(x)
    std = log_std.exp() #log std to enforce non negativity

    #appendix c
    epsilon = torch.randn_like(mean)
    u = epsilon * std + mean
    a = torch.tanh(u)

    normal = Normal(mean, std)
    log_prob_u = normal.log_prob(u)
    log_prob = (log_prob_u - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)

    mean_action = torch.tanh(mean)

    return a, log_prob, mean_action
```

The first few lines are as discussed. The neural network produces mean and log standard deviation (which is exponentiated to enforce non negativity of standard deviations). You draw epsilon from a standard Gaussian distribution and use the above formula for $a_t$ to get something we call $u$. There are further steps needed because $u$ has infinite support so actions could be drawn from any real number. So as explained in SAC's appendix C, they take $a = tanh(u)$ since tanh squashes anything to a support of (−1, 1). So $\pi(a|s)$ is written as
\[
\pi(a|s) = \mu(u|s) \left| \det \left( \frac{da}{du} \right) \right|^{-1}
\tag{20}
\]
which can be simplified as 
\[
\tag{21}
\log \pi(a|s) = \log \mu(u|s) - \sum_{i=1}^D \log\big( 1 - \tanh^2(u_i) \big)
\]

because the Jacobian $\frac{da}{du} = \operatorname{diag}(1 - \tanh^2(u))$ is diagonal.


## Policy Loss

Recall the minimization problem, which when sampling Gaussian $\epsilon$ instead of from the action distribution is now:

$$
\begin{aligned}
&= \arg\min_{\phi}
  \mathbb{E}_{s_t \sim \mathcal{D}}
  \mathbb{E}_{\epsilon_t \sim \mathcal{N}}
  \left[
    \log \pi_\phi( . \mid s_t)
    - Q_\theta(s_t, . )
  \right]
\end{aligned}
$$

The loss function is therefore that equation exactly: 

$$
\tag{12}
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D},\, \epsilon_t \sim \mathcal{N}} \left[ \log \pi_\phi \left( f_\phi(\epsilon_t; s_t) \mid s_t \right) - Q_\theta \left( s_t, f_\phi(\epsilon_t; s_t) \right) \right]
$$

```{python}
#update policy function
action_current_policy, log_prob, _ = model_p.sample(states)
q1_current_policy = model_q1(states, action_current_policy)
q2_current_policy = model_q2(states, action_current_policy)
q = torch.min(q1_current_policy, q2_current_policy)

loss_p = (log_prob - q).mean()
```

# Algorithm

<figure>
  <img src="blogs/sac-figures/algorithm.png" alt="Pseudocode" />
  <figcaption>Annotated Pseudocode from Haarnoja et al. (2018)</figcaption>
</figure>


The actual code closely follows the sequence of steps in the SAC algorithm pseudocode, with each logical block directly traceable to an implementation section.


```{python}
global_step = 0
update = 0


# INITIALIZE NETWORKS/PARAMS
model_v = SoftValueFunction(num_states, cfg['HIDDEN_SIZE']).cuda()
model_q1 = SoftQFunction(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()
model_q2 = SoftQFunction(num_states, num_actions, cfg['HIDDEN_SIZE']).cuda()
model_p = PolicyFunction(num_states, num_actions, cfg['HIDDEN_SIZE'], cfg['LOG_STD_MIN'], cfg['LOG_STD_MAX']).cuda()
model_vtarget = copy.deepcopy(model_v)

for param in model_vtarget.parameters():
    param.requires_grad = False

optimizer_v = torch.optim.Adam(model_v.parameters(), lr = cfg['LR'])
optimizer_q1 = torch.optim.Adam(model_q1.parameters(), lr = cfg['LR'])
optimizer_q2 = torch.optim.Adam(model_q2.parameters(), lr = cfg['LR'])
optimizer_p = torch.optim.Adam(model_p.parameters(), lr = cfg['LR'])


start = time.time()
states_t = env.reset()
while global_step < cfg['TOTAL_ITERS']:
    #SAMPLE ACTION FROM POLICY NETWORK
    actions_t = compiled_select_action(model_p, states_t)

    #STEP THROUGH SAMPLED ACTION
    states_next, rewards_t, terminals_t = env.step(actions_t)

    #ADD TRANSITION TO REPLAY BUFFER
    replay.add_batch(states_t, actions_t, rewards_t, states_next, terminals_t)
    states_t = states_next

    global_step += cfg['TOTAL_AGENTS']

    if replay.size >= cfg['MINIBATCH']:
        for i in range(cfg['TARGET_GRAD_STEPS']):
            #SAMPLE MINIBATCH
            batch = replay.sample(cfg['MINIBATCH'])
            
            #MINIBATCH UPDATE
            loss_v, loss_q1, loss_q2, loss_p = train_step(
                model_v, model_q1, model_q2, model_p, model_vtarget,
                optimizer_v, optimizer_q1, optimizer_q2, optimizer_p,
                batch, cfg,
            )
            update += 1
```


# Application


<figure>
  <img src="blogs/sac-figures/wandb.png" alt="Score Over Time" />
  <figcaption>Policy's Score and Perf over Time</figcaption>
</figure>

I run the above code on the Drone hover environment on Pufferlib and after 15 minutes of training on my RTX 3090, I get score of ~475 (max is 1024) and a performance score of ~0.7 (max is 1). That means that the drone was able to hover and hold its exact position for about half of the episode.
