# RL-implement-IMPALA
A Test-Implementation of the IMPALA algorithm (by deepmind 2018)

![Grid World Barto/Sutton new 2017 draft Chapter 13.1](https://raw.githubusercontent.com/ducandu/RL-Implementation-IMPALA/master/grid_world.png "The grid world we are trying to solve")

**IMPORTANT NOTE: Currently, IMPALA's "V-trace"-correction is not
implemented! Instead, we are using simple REINFORCE with a state-value
function baseline by Williams (1992).**

The point of this repo is to demonstrate how the distributed tensorflow part
described in the paper can be implemented quite easily using two job types
('explorers' and 'learners').

See the following paper for more details:

**IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures**

https://arxiv.org/abs/1802.01561

- Special async off-policy way of distributing the RL algo between:
  - explorers: only act in their own envs and store each
    episode in a global buffer (no learning); they use a local copy (mu) of the main policy (pi)
    only synched at the beginning of each episode. Thus, mu may be behind pi.
  - learners: pull batches from the global buffer and apply the learning algo to
    the main policy (pi).
- Because exploring (and the collected experiences) is off-policy, they introduce a trick - v-traces -
  to adjust the vanilla PG update for this off-policy case.

A couple of scripts should be run in parallel to simulate the distributed setup.
Run:

```
$ python impala_proto.py -h
```

to see possible command line options.

To run a most basic actual RL-job, start the following two scripts in parallel:

```
$ python impala_proto.py -l localhost:22222 -e localhost:22223 -j learner -t 0
$ python impala_proto.py -l localhost:22222 -e localhost:22223 -j explorer -t 0
```

This will run the algo on the local machine using one learner and one explorer (agent).
Add more learners or explorers if you like.
The built-in Env is the one taken from *Barto & Sutton*:
**Reinforcement Learning: An Introduction** "2017 Completed Draft" Chapter 13
(Policy Gradient Methods) Fig 13.1.

It's a "blind" environment without any state signal, only two actions (left and right),
and 4 states. The optimal policy is to move right with a probability of
0.59 regardless of the state (which we don't know anyway!).

The learner process outputs the loss and the average probability of moving right
for the simple policy/state-value network used and this probability should
converge to 0.59 after a few 10k training steps depending on weight initialization,
learning_rate (default is 0.001), and - yes - many many other factors :)

```
task 0 step 45910 loss 10.399235725402832 avg-right-prob=0.5933351516723633
```
*Output of the "learner" job, task 0 around the point where it found the optimal policy:*

The neural net used is a simple two-layer
dense network with 1 input for the state (always 1.0), n hidden nodes (default 10),
and 3 output nodes (2 action probs (left and right), 1 state-value).
