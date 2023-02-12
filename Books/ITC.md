# Introduction to the Theory of Computation

- Begin at 2023/02/09

## 0: Introduction

Basic concepts used throughout the book, e.g., sets, predicates, graphs, proofs.

### Exercises:

#### 0.7

For $x, y \in \mathcal{R}$:

1. $x\text{R}y$ iff $|x - y| \leq 1$
2. $x\text{R}y$ iff $x \leq y$
3. $x\text{R}y$ iff $xy > 0$

#### 0.10

We cannot divide $(a-b)$ on each side while $(a-b) = 0$.

#### 0.11

The subset $H_1$ and $H_2$ may not be the same color, though each of them has a unique color.

#### 0.13

Main idea: to construct an example of clique or anti-clique.

## 1: Regular Languages

### 1.1: Finite Automata

Finite automata: can have 0 accept states, and must have exactly 1 transition exiting every state for each possible symbol.

A finite automaton: $(Q, \Sigma,, \delta, q_0, F)$, where

- $Q$ is a finite set of states,
- $\Sigma$ is the alphabet,
- $\delta: Q \times \Sigma \rightarrow Q$ is the transitional function,
- $q_0 \in Q$ is the start state,
- and $F \subseteq Q$ is the set of final states.

$A$ is the language of machine $M$: $L(M) = A$, where $A$ is the set of all strings that $M$ accepts.

A regular language: there exists some finite automaton recognizes it.

### 1.2 Nondeterminism

Motivation: to solve the problem of constructing an automaton recognizes language $A_1 \circ A_2$.







### Exercises:

#### 1.x
