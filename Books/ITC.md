# Introduction to the Theory of Computation

- Begin at 2023/02/09

## 0. Introduction

Basic concepts used throughout the book, e.g., sets, predicates, graphs, proofs.

### Exercises

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

## 1. Regular Languages

### 1.1 Finite Automata

Finite automata: can have 0 accept states, and must have exactly 1 transition exiting every state for each possible symbol.

A finite automaton: $(Q, \Sigma,, \delta, q_0, F)$, where

- $Q$ is a finite set of states,
- $\Sigma$ is a finite alphabet,
- $\delta: Q \times \Sigma \rightarrow Q$ is the transitional function,
- $q_0 \in Q$ is the start state,
- and $F \subseteq Q$ is the set of final states.

Note that, for a DFA, there exists exactly one transition arrow exits every state for each possible input symbol.

$A$ is the language of machine $M$: $L(M) = A$, where $A$ is the set of all strings that $M$ accepts.

A regular language: there exists some finite automaton recognizes it.

### 1.2 Nondeterminism

Motivation: to solve the problem of constructing an automaton recognizes language $A_1 \circ A_2$.

Each NFA can be equivalently converted to a DFA.

A nondeterministic finite automaton: $(Q, \Sigma, \delta, q_0, F)$, where

- $Q$ is a finite set of states,
- $\Sigma$ is a finite alphabet,
- $\delta: Q \times \Sigma_\varepsilon \rightarrow \mathcal{P}(Q)$ is the transitional function, $\Sigma_\varepsilon = \Sigma \cup \{\varepsilon\}$, $\mathcal{P}(Q)$ is the power set of $Q$,
- $q_0 \in Q$ is the start state,
- and $F \subseteq Q$ is the set of final states.

Converting a NFA $(Q, \Sigma, \delta, q_0, F)$ to a DFA $(Q', \Sigma, \delta', q_0', f')$:

- $Q' = \mathcal{P}(Q)$
- $\delta'(R, a) = \bigcup_{r \in R} \delta(r, a)$
- $q_0' = \{q_0\}$
- $F' = \{R \in Q' | R \text{ contains an accept state of }N\}$

Let $E(R) = \{q| q \text{ can be reached from } R \text{ by traveling along } 0 \text{ or more } \varepsilon \text{ arrows}\}$, then

- $\delta'(R, a) = \{q \in Q | q \in E(\delta(r, a)) \text{ for some } r \in R\}$,
- $q_0' = E(\{q_0\})$, thus address the $\varepsilon$ arrows

A language is regular iff. some NFA or DFA recognizes it.

### 1.3 Regular Expressions

Regular expressions: using regular operations to build up expressions describing languages.

$R$ is a regular expression if $R$ is:

- $a \in \Sigma$,
- $\varepsilon$,
- $\emptyset$,
- $(R_1 \cup R_2)$, where $R_1$ and $R_2$ are regular expressions,
- $(R_1 \circ R_2)$, where $R_1$ and $R_2$ are regular expressions,
- $(R_1^*)$, where $R_1$ is a regular expression.

$R^+ = RR^*$

$R^+ \cup \varepsilon = R^*$

$1^*\emptyset = \emptyset$

A language is regular iff. some regular expression describes it.

A generalized nondeterministic finite automaton (GNFA): the labels of transition arrows can be regular expressions, instead of only characters in the alphabet or $\varepsilon$.

A GNFA: $(Q, \Sigma, \delta, q_{\text{start}}, q_{\text{accept}})$, where

- $Q$ is a finite set of states,
- $\Sigma$ is a finite alphabet,
- $\delta: (Q - \{q_{\text{accept}}\}) \times (Q - \{q_{\text{start}}\}) \rightarrow \mathcal{R}$ is the transitional function, $\mathcal{R}$ is the collection of all regular expressions over the alphabet $\Sigma$,
- $q_{\text{start}}$ is the start state,
- and $q_{\text{accept}}$ is the final state.

A DFA can be easily transformed into a GNFA, and the GNFA can be reformed to express a specific regular expression.

### 1.4 Nonregular languages

Pumping Lemma: If $A$ is a regular language, then there is a number $p$ where, if $s$ is any string in $A$ of length at least $p$, then $s$ may be divided into three pieces, $s = xyz$, satisfying:

1. for each $i \geq 0, xy^iz \in A$,
2. $|y| > 0$, and
3. $|xy| \leq p$.

To prove a language is nonregular: find an example string in the language (usually w.r.t. $p$), and show that it doesn't satisfy the pumping lemma.

Useful trick: $xy^iz \in A, i \text{ can be } 0$, this is called "pumping down".

### Exercises:

#### 1.11

Add a new accept state $q_{\text{accept}}$, and $\varepsilon$-transitions from each original accept state to $q_{\text{accept}}$.

#### 1.15

If the start state has any income transitions, make it to be an accept state may import undesirable "strings prefixes" of the language.

#### 1.23

1. If $B = B^+$, and for any language we have $BB \subseteq B^+$, then we have $BB \subseteq B$.
2. If $BB \subseteq B$, for any string $b = b_1 \circ b_2 \circ ... \circ b_n$ where $b_i \in B, n$ is a positive length, we have $b_1 \circ b2 \in B$. We may rename it as $b_{12}$. Similarly we have $b_{12} \circ b_3 \in B$, we can rename it as $b_{123}$...Conclude by this we may end up with $b \in B$. Therefore $B^+ \subseteq B$.

#### 1.44

Proof: If $B$ and $C$ are both regular languages, then $B \leftarrow C$ is also a regular language.

Main idea: define a NFA based on DFAs $M_B$ and $M_C$. It checks the input string $w \in B$ and in parallel, nondeterministically guesses a string $y$ contains the same number of $1$s and $y \in C$.

#### 1.50

$w^\mathcal{R}$: reverse $w$.

Assume such FST exists. Then it should output $00$ for input $00$ and output $10$ for input $01$. For the two cases the first input symbols are the same but their output symbols are different. Hence this FST doesn't exist.

## 2. Context-Free Languages

### 2.1 Context-Free Grammars

Context-free languages include regular languages and many other additional languages.

$A, B$: variables, often used in the substitution rules.

$0, 1... \in \Sigma$: terminals

Context-free language: any language that can be generated by some content-free grammar.

Derivation: the sequence of substitutions to obtain a string using the given rules.

A context-free grammar: $(V, \Sigma, R, S)$, where

- $V$ is a finite set of variables,
- $\Sigma$ is a finite set of terminals,
- $R$ is a set of substitution rules, with each rule being a variable and a string of variables and terminals,
- $S \in V$ is the start variable.

Convert a DFA to an equivalent CFG is easy.

A grammar is ambiguous: it generates the same string in multiple ways.

A grammar generates a string ambiguously: the string has two different parse trees, not two different derivations. Two derivations may only be different in order but result in the same tree.

To focus on the structure instead of the order, we introduce the leftmost derivation.

A string is derived ambiguously in CFG $G$: if it has 2 or more different leftmost derivations.

Grammar $G$ is ambiguous: if it generates some string ambiguously.

A language (some CFG) is inherently ambiguous: can only be generated by ambiguous grammars.

A CFG is in Chomsky Normal Form if every rule is of the form:

- $A \rightarrow BC$, or
- $A \rightarrow a$

where $a$ is any terminal and $A, B, C$ are any variables ($B, C$ cannot be the start variable). We also permit:

- $S \rightarrow \varepsilon$, where $S$ is the start variable.

Any CFL is generated by a CFG in Chomsky Normal Form, i.e., we can convert any grammar $G$ into Chomsky Normal Form.

### 2.2 Pushdown Automata

A pushdown automaton(PDA): just like a NFA, but with an extra stack, allowing it to recognize some nonregular languages.

(Nondeterministic) Pushdown automata are equivalent in power to CFGs.

Nondeterministic PDA is more powerful than deterministic PDA! (Unlike NFA and DFA.)

A PDA is a 6-tuple $(Q, \Sigma, \Gamma, \delta, q_0, F)$:

- $Q$ is a set of states,
- $\Sigma$ is the input alphabet,
- $\Gamma$ is the stack alphabet,
- $\delta： Q \times \Sigma_\varepsilon \times \Gamma_\varepsilon \rightarrow \mathcal{P}(Q \times \Sigma_\varepsilon)$ is the transition function,
- $q_0 \in Q$ is the start state, and
- $F \subseteq Q$ is the set of final states.

A PDA cannot explicitly test the emptiness of the state, or reaching the end of input string.

A nondeterministic PDA can "guess" which branch to go among multiple choices.

A language is context-free iff. there is a PDA that recognizes it.

Every regular language is context-free.

### 2.3 Non-Context-Free Languages

Pumping lemma for context-free languages:

* If $A$ is a CFL, then there is a number $p$ (the pumping length) where, if $s$ is any string in $A$ of length at least $p$, then $s$ may be divided into 5 pieces $s = uvxyz$ satisfying:
  1. for each $i \geq 0$, $uv^ixy^iz \in A$,
  2. $|vy| > 0$, and
  3. $|vxy| \leq p$.
