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
- $\delta： Q \times \Sigma_\varepsilon \times \Gamma_\varepsilon \rightarrow \mathcal{P}(Q \times \Gamma_\varepsilon)$ is the transition function,
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

### Exercises:

#### 2.3

* $T \stackrel{*}{\Rightarrow} T$: true (0 step)
* $\Rightarrow$: yields (exactly 1 step)
* $\stackrel{*}{\Rightarrow}$: derives (0 or more steps)
* The language of the grammar: $\{w \in \Sigma^* | S \stackrel{*}{\Rightarrow} w\}$

#### 2.4

1. $\{w | w \text{ contains at least 3 `1' s.}\}$
   S = R1R1R1R
   R = 0R|1R|$\varepsilon$
2. $\{w | w \text{ the length of w is odd and its middle symbol is 0}\}$
   S = 0|0S0|0S1|1S0|1S1

#### 2.6

1. The set of strings over the alphabet $\{a, b\}$ with more $a$'s than $b$'s.
   S = TaT
   T = a|aTb|bTa|TT|$\varepsilon$
2. $\{w \text{\#} x | w^\mathcal{R} \text{ is a substring of } x \text{ for } w, x \in \{0, 1\}^*\}$
   S = TX
   T = 0T0|1T1|#X
   X = 0X|1X|$\varepsilon$

#### 2.18

1. To prove $C \cap R$ is context-free, construct a PDA $P'$ based on the PDA $P$ which recognizes $C$ and the DFA which recognizes $R$.

#### 2.38

Proof idea: give a counterexample and apply pumping lemma.
Note: perfect shuffle is to take two strings of the same length from languages A and B, respectively. Then we combine them.

## 3. The Church-Turing Thesis

### 3.1 Turing Machines

The Turing Machine was first proposed by Alan Turing in 1936.

The differences between finite automata and Turing machines:

- A TM can both read from the tape and write on it
- The read-write head can move both to the left and right
- The tape is infinite
- The special states for rejecting and accepting take effect immediately

A Turing machine $(Q, \Sigma, \Gamma, \delta, q_0, q_\text{accept}, q_\text{reject})$:

- $Q$ is the finite set of states,
- $\Sigma$ is the input alphabet not containing the blank symbol $_\sqcup$,
- $\Gamma$ is the tape alphabet, where $_\sqcup \in \Gamma$ and $\Sigma \subseteq \Gamma$,
- $\delta: Q \times \Gamma \rightarrow Q \times \Gamma \times \{L, R\}$ is the transition function,
- $q_0 \in Q$ is the start state,
- $q_{\text{accept}} \in Q$ is the accept state, and
- $q_{\text{reject}} \in Q$ is the reject state, where $q_{\text{reject}} \neq q_{\text{accept}}$.

A configuation of the TM: For a state $q$ and 2 strings $u, v$ over $\Gamma$, use $uqv$ to represent the current state is $q$, the current tape contents is $uv$, and the current head location is the first symbol of $v$.

The language of Turing machine $M$: the collection of strings that $M$ accepts, denoted $L(M)$.

A language is Turing-recognizable (recursively enumerable language) if some Turing machine recognizes it.

A decider: a TM that always halts on all inputs (either accepts or rejects the input).

A language is Turing-decidable (recursive language) if some TM decides it.

A Turing-decidable language must be Turing-recognizable, but there are some Turing-recognizable language are non-decidable.

### 3.2 Variants of Turing Machines

Multitape Turing Machine: can be transformed to a single-tape TM by concatenating all the strings on the tape by #.

Nondeterministic Turing Machine: can be transformed to a 3-tape deterministic TM.

A nondeterministic TM is a decider: if all its branches halt on all inputs.

A language is Turing-recognizable iff. some enumerator enumerates it.

An essential feature of these models: have unrestricted access to unlimited memory.

### 3.3 The Definition of Algorithm

🎉️The Church-Turing Thesis: Intuitive notion of algorithms equals to Turing machine algorithms.

An example of Turing recognizer but NOT decider:

- for $\{p | p \text{ is a polynomial over } x \text{ with an integral root}\}$
- the TM evaluates the value of $x$ as: $0, 1, -1, 2, -2, ...$

If the polynomial has an integral root, the TM will recognize it. But if not, it will run forever. Although in this case we can modify this TM to be a decider by giving its upper and lower bounds of values. (For multivariable polynomials such bounds do not exist.)

The encoding of an object $O$ to a string: $\langle O \rangle$.

### Exercises:

#### 3.1

(b)$q_100, \_q_20, \_xq_3\_, \_q_5x\_, q_5\_x\_, \_q_2x\_, \_xq_2\_, \_x\_q_{\text{accept}}$.

#### 3.2

(a) $q_111, xq_31, x1q_3\_, x1\_q_{\text{reject}}$.

#### 3.3

1. If a language is decidable, then it can be decided by some deterministic Turing machine, which is naturally a nondeterministic Turing machine.
2. If a nondeterministic TM $N$ can decide some language, i.e., being a decider for language $L$, then we can construct another deterministic TM based on $N$. It runs over all branches of $N$, and enters the (added) reject state after it exhausts all the possible branches.

#### 3.5

1. Yes. The tape alphabet contains "_", and a TM can write any symbol in its alphabet to the tape.
2. No. The tape alphabet must contain "_" while the input alphabet must not.
3. Yes. When the head has moved to the left end of the tape.
4. No. The $q_{\text{accept}}$ cannot be the same as $q_{\text{reject}}$.

#### 3.8

(a) On input string $w$:

1. Scan the tape and mark the first 0 which has not been marked. If no unmarked 0 is found, go to stage 4. Otherwise, move the head back to the front of the tape.
2. Scan the tape and mark the first 1 which has not been marked. If no unmarked 1 is found, reject.
3. Move the head back to the front of the tape and go to stage 1.
4. Move the head back to the front of the tape. Scan the tape to see if any unmarked 1s remain. If none are found, accept; otherwise, reject.

## 4. Decidability

Goal: to explore the limit of algorithmic solvability.

### 4.1 Decidable Languages

Firstly, we can represent the computational problems by languages by somehow encoding them as set of strings.

The equality problem between two CFGs are undecidable. i.e., $\text{EQ}_{\text{CFG}} = \{\langle G, H \rangle | G, H \text{ are CFGs and } L(G) = L(H)\}$ is undecidable.

- because CFG is not closed under complementation or intersection.

### 4.2 The Halting Problem

$A_{\text{TM}} = \{\langle M, w \rangle | M \text{ is a TM and } M \text{ accepts } w\}$ is not decidable.

There are uncountably many languages but only contably many TMs, so there must be some languages are not Turing decidable or even Turing recognizable.

Proof idea:

1. For input $\langle M, w \rangle$ where $M$ is a TM and $w$ is a string, assume there is a TM $H$ is a decider for this language.
2. $H$ accepts $\langle M, w \rangle$ if $M$ accepts $w$, and $H$ rejects $\langle M, w \rangle$ if $M$ rejects $w$.
3. Construct a new TM $D$: For input $\langle M \rangle$, if $H$ accepts $\langle M, \langle M \rangle \rangle$, then $D$ rejects, and if $H$ rejects $\langle M, \langle M \rangle \rangle$, then $D$ accepts.
4. Consider $D$ with input $\langle D \rangle$. Here we get the contradiction:
   * If $H$ accepts $\langle D, \langle D \rangle \rangle$, then $D$ should reject $\langle D \rangle$. Meanwhile, $H$ accepts $\langle D, \langle D \rangle \rangle$ means $D$ accepts $\langle D \rangle$, otherwise $H$ cannot accept $\langle D, \langle D \rangle \rangle$.
   * If $H$ rejects $\langle D, \langle D \rangle \rangle$, then $D$ should reject $\langle D \rangle$. Meanwhile, $H$ rejects $\langle D, \langle D \rangle \rangle$ means $D$ rejects $\langle D \rangle$, otherwise $H$ cannot reject $\langle D, \langle D \rangle \rangle$.
5. Therefore, neither such $H$ nor $D$ exists.

But at least $A_{\text{TM}}$ is Turing-recognizable. We can construct a TM $U$ for the input $\langle M, w \rangle$: if $M$ ever accepts $w$, then it accepts. If $M$ loops on $w$, then $U$ also loops, thus it cannot be a decider.

There are still some Turing-unrecognizable languages.

A language is decidable iff. both itself and its complement are Turing-recognizable.

A language is co-Turing-recognizable if it is the complement of a Turing-recognizable language.

Therefore, a Turing-undecidable but recognizable language must have a complement being Turing-unrecognizable.

A Turing-undecidable language itself or its complement must be Turing-unrecognizable.

$\overline{A_{\text{TM}}}$ is not Turing-recognizable.

### Exercises

#### 4.9

Construct a TM being a decider of $\{\langle A \rangle | A \text{ is a DFA and } L(A) \text{ is an infinite language }\}$.

1. Let $k$ be the number of states of the DFA.
2. Construct a DFA $D$ that accepts all strings of length $k$ or more.
3. Construct a DFA $M$ such that $L(M) = L(A) \cap L(D)$. (the intersection is closed for regular languages)
4. Test $L(M) = \emptyset$. If $E_{\text{DFA}}$ accepts, then rejects; if $E_{\text{DFA}}$ rejects, then accepts.

#### 4.11

Similar to 4.9, just construct another DFA $D$ that only accepts strings containing an odd number of 1s.

#### 4.13

Note that, the intersection of a context-free language and a regular language is context-free. Then we can use $E_{\text{CFG}}$ to decide whether $1^* \cap L(G) = \emptyset$.

#### 4.21

Proof idea: construct a DFA based on the NFA, by tracking and merging all reachable states in in each step w.r.t. the input. The accept state is where there are 1+ paths have occurs somewhere before. Then we simply run an $E_{\text{DFA}}$ to decide whether it is empty.

#### 4.23

The language that contains equal number of 0s and 1s is context-free. So the intersectoin of this context-free language and the language that the DFA accepts is context-free. Similar to 4.11, construct a $E_{\text{CFG}}$ to decide whether the intersection is an emptyset.

## 5. Reducibility

### 5.1 Undecidable Problems from Language Theory

If A is reducible to B, solving A cannot be harder than solving B.

* If A is reducible to B and B is decidable, A is also decidable.
* If A is reducible to B and A is undecidable, B is undecidable.

🎉️The Halting Problem: $HALT_{TM} = \{\langle M, w \rangle | M \text{ is a TM and } M \text{ halts on input } w\}$ is undecidable.

- Proof idea: Reducing $A_{TM}$ to $HALT_{TM}$. Using contracdiction, assume $HALT_{TM}$ is decidable, there should be a TM $R$ decides it. Note that $A_{TM}$ is undecidable, we can construct a decider for $A_{TM}$ based on $R$, which contradicts with the undecidability.

$E_{TM} = \{ \langle M \rangle | M \text{ is a TM and } L(M) = \emptyset\}$ is undecidable.

- Proof idea: Assume $E_{TM}$ is decidable, and $R$ is the decider. Then for the input $\langle M, w \rangle$ of $A_{TM}$, we modify $\langle M \rangle$ to $\langle M_1 \rangle$ which rejects all input strings except $w$. Then we construct a TM $S$ based on $R$ to test if $R$ accepts $\langle M_1 \rangle$.

$REGULAR_{TM} = \{ \langle M \rangle | M \text{ is a TM and } L(M) \text{ is a regular language}\}$ is undecidable.

- Proof idea: Assume $REGULAR_{TM}$ is decidable, and $R$ is the decider. For the input $\langle M, w \rangle$ of $A_{TM}$, construct a TM $M_2$ based on $M$: if $M$ rejects $w$, $M_2$ accepts a nonregular language $\{0^n1^n | n \geq 0\}$; if $M$ accepts $w$, $M_2$ also accepts the rest of strings ($\Sigma^*$ is regular). Then we construct a TM $S$ to test if $R$ accepts $\langle M_2 \rangle$.

Rice's Theorem: Testing any property of the languages recognized by Turing machines is undecidable.

Computation History: Let $M$ be a TM and $w$ an input string,

- An accepting computation history for $M$ on $w$:
  - is a sequence of configuations, $C_!, C_2, ..., C_l$, where
  - $C_1$ is the start configuation,
  - $C_l$ is an accepting configuation of $M$, and
  - each $C_i$ legally follows from $C_{i-1}$.
- A rejecting computation history is similar.

A linear bounded automata (LBA): a restricted type of TM wherein its tape head isn't permitted to move off the portion of the tape containing the input (i.e., with limited amount of memory which is linear to the input).

$A_{LBA} = \{\langle M, w \rangle | M \text{ is a LBA that accepts string } w\}$ is decidable.

- Proof idea: $M$ can be in only a limited number of configuations w.r.t. the number of states, the size of the tape alphabet, and the length of the tape. Therefore, for a given input, it should halt within a limited number of steps, otherwise it must be in a loop.

$E_{LBA} = \{\langle M \rangle | M \text{ is a LBA where } L(M) = \emptyset\}$ is undecidable.

- Proof idea: By reduction of $A_{TM}$. Assume $E_{LBA}$ is decidable, we can construct a LBA $B$ based on $M$ and $w$ and test whether $L(B)$ is empty. $L(B)$ recognizes all accepting computation histories for $M$ on $w$.

$ALL_{CFG} = \{\langle G \rangle | G \text{ is not a CFG and } L(G^*) = \Sigma^*\}$ is undecidable.

- Proof idea: construct a CFG $G$ generates all strings that fail to be an accepting computation history for $M$ on $w$.

### 5.2 A Simple Undecidable Problem
