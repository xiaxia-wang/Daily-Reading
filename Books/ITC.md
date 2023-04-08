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

A configuration of the TM: For a state $q$ and 2 strings $u, v$ over $\Gamma$, use $uqv$ to represent the current state is $q$, the current tape contents is $uv$, and the current head location is the first symbol of $v$.

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
  - is a sequence of configurations, $C_!, C_2, ..., C_l$, where
  - $C_1$ is the start configuration,
  - $C_l$ is an accepting configuration of $M$, and
  - each $C_i$ legally follows from $C_{i-1}$.
- A rejecting computation history is similar.

A linear bounded automata (LBA): a restricted type of TM wherein its tape head isn't permitted to move off the portion of the tape containing the input (i.e., with limited amount of memory which is linear to the input).

$A_{LBA} = \{\langle M, w \rangle | M \text{ is a LBA that accepts string } w\}$ is decidable.

- Proof idea: $M$ can be in only a limited number of configurations w.r.t. the number of states, the size of the tape alphabet, and the length of the tape. Therefore, for a given input, it should halt within a limited number of steps, otherwise it must be in a loop.

$E_{LBA} = \{\langle M \rangle | M \text{ is a LBA where } L(M) = \emptyset\}$ is undecidable.

- Proof idea: By reduction of $A_{TM}$. Assume $E_{LBA}$ is decidable, we can construct a LBA $B$ based on $M$ and $w$ and test whether $L(B)$ is empty. $L(B)$ recognizes all accepting computation histories for $M$ on $w$.

$ALL_{CFG} = \{\langle G \rangle | G \text{ is not a CFG and } L(G^*) = \Sigma^*\}$ is undecidable.

- Proof idea: construct a CFG $G$ generates all strings that fail to be an accepting computation history for $M$ on $w$.

### 5.2 A Simple Undecidable Problem

The Post Correspondence Problem

- Proof idea: to formulate $A_{TM}$ as an example of PCP.

### 5.3 Mapping Reducibility

Computable Function: a function $f: \Sigma^* \rightarrow \Sigma^*$ is a computable function if some TM $M$, on every input $w$, halts with just $f(w)$ on its tape.

Language $A$ is mapping reducible to language $B$, i.e., $A \leq_m B$, if there is a computable function $f: \Sigma^* \rightarrow \Sigma^*$, where for every $w$,

- $w \in A \Leftrightarrow f(w) \in B$.
- The function $f$ is called the reduction of $A$ to $B$.

## 6. Advanced Topics in Computability Theory

### 6.1 The Recursion Theorem

Self Reference: making a TM that ignores its input and prints out a copy of its own description.

- $A = P_{\langle B \rangle}$, i.e., $A$ takes $\langle B \rangle$ as input, prints $\langle B \rangle$ and halts.
- $B$ reads $\langle B \rangle$ from the tape, computes $q(\langle B \rangle)$ as the description of $A$, then combines $\langle AB \rangle$ to be a complete TM, i.e., $\langle SELF \rangle$.

Recursion Theorem: Let $T$ be a TM that computes a function $t: \Sigma^* \times \Sigma^* \rightarrow \Sigma^*$. There is a TM $R$ that computes a function $r: \Sigma^* \rightarrow \Sigma^*$, where for every $w$, $r(w) = t(\langle R \rangle, w)$.

Another method to prove $A_{TM}$is undecidable:

- Assume there is a TM $H$ that decides $A_{TM}$.
- Construct a TM $B$: on input $w$,
  - obtain $\langle B \rangle$ via the recursion theorem
  - run $H$ on input $\langle B, w \rangle$
  - do the opposite of the output of $H$
- This means $H$ cannot be a decider of $A_{TM}$.

$MIN_{TM} = \{\langle M \rangle | M \text{ is a minimal TM}\}$ is undecidable and not Turing-recognizable.

- Proof idea: Assume that some enumerator $E$ enumerates $MIN_{TM}$. Then we construct a TM $C$, which can obtain its description $\langle C \rangle$. By running $E$ we can obtain a TM $D$ with a longer description than $C$, however, $D$ can be simulated by $C$ with a shorter description.

The fixed-point version of the recursion theorem: Let $t : \Sigma^* \rightarrow \Sigma^*$ be a computable function. Then there is a TM $F$ for which $t(\langle F \rangle)$ describes a TM equivalent to $F$.

- Proof idea: Construct a TM $F$ that can obtain its description $\langle F \rangle$ via the recursion theorem. Then it may compute $t(\langle F \rangle)$ to obtain the description of another TM $G$. Then it simulate $G$ on $w$, thus $\langle F \rangle = t(\langle F \rangle) = \langle G \rangle$.

### 6.2 Decidability of Logical Theories

A formula with no free variables is called a sentence or statement.

The language of a model: the collection of formulas that use only the relation symbols of the model assigns and that use each relation symbols with the correct arity.

For a model $\mathcal{M}$, the theory of $\mathcal{M}$, denoted as $\text{Th}(\mathcal{M})$, is the collection of true sentences in the language of $\mathcal{M}$.

$\text{Th}(\mathcal{N}, +)$ is decidable.

- Proof idea: Let $\phi = Q_1x_1Q_2x_2...Q_lx_l[\psi]$ be a sentence in the language of $(\mathcal{N}, +)$. For each $i$ from $0$ to $l$, define $\phi_i = Q_{i+1}x_{i+1}Q_{i+2}x_{i+2}...Q_lx_l [\psi]$. Constructing a list of finite automata $A_l, ... A_1$ to recognize the collection of strings that make $\phi_i$ true.

$\text{Th}(\mathcal{N}, +, \times)$ is undecidable (even when restricted to the language of $(\mathcal{N}, +, \times)$).

- Proof idea: Constructing a mapping reduction from $A_{TM}$ to $\text{Th}(\mathcal{N}, +, \times)$.

Kurt Godel's Incompleteness Theorem (informal): in any reasonable system of formalizing the notion of provability in number theory, some true statements are unprovable.

Two reasonable properties of proof:

1. The correctness of a proof of a statement can be checked by machine. i.e., $\{\langle \phi, \pi \rangle | \pi \text{ is a proof of } \phi\}$ is decidable.
2. The system of proofs is sound. i.e., anything provable is true.

The collection of provable statements in $\text{Th}(\mathcal{N}, +, \times)$ is Turing-recognizable.

Some true statements in $\text{Th}(\mathcal{N}, +, \times)$ is not provable. (Prove by contradiction)

The sentence $\psi_{\text{unprovable}}$ is unprovable.

### 6.3 Turing Reducibility

Mapping reducibility cannot capture the intuitive concept of reducibility.

- e.g., $A_{TM}$ and $\overline{A_{\text{TM}}}$ are intuitively reducible to each other, but $\overline{A_{\text{TM}}}$ is not mapping reducible to $A_{TM}$ because $A_{TM}$ is Turing-recognizable while $\overline{A_{\text{TM}}}$ is not.

An oracle: an external device that is capable of reporting the gold answer, e.g., whether any string $w$ is a member of a language $B$, no matter $B$ is Turing decidable or not.

Language $A$ is Turing reducible to language $B$, denoted as $A \leq_\text{T} B$, if $A$ is decidable relative to $B$.

- e.g., $E_{TM}$ is decidable relative to $A_{TM}$.

If $A \leq_\text{T} B$ and $B$ is decidable, then $A$ is decidable.

Turing reducibility is a generalization of mapping reducibility.

### 6.4 A Definition of Information

The minimal description $d(x)$ of a binary string $x$: the shortest string $\langle M, w \rangle$ where TM $M$ on input $w$ halts with $x$ on its tape.

- The descriptive complexity (a.k.a. Kolmogorov complexity) of $x$: $\text{K}(x) = |d(x)|$.

Properties of $\text{K}(x)$:

- $\exist c \forall x [\text{K}(x) \leq |x| + c]$
- $\exist c \forall x [\text{K}(xx) \leq \text{K}(x) + c]$
- $\exist c \forall x, y [\text{K}(xy) \leq 2\text{K}(x) + \text{K}(y) + c]$, can be further optimized to $\text{K}(xy)\leq 2 \log(\text{K}(x)) + \text{K}(x) + \text{K}(y) + c$

Optimality of $\text{K}(x)$:

- $\forall x [\text{K}(x) \leq \text{K}_p(x) + c]$, where $\text{K}_p(x) = |d_p(x)|$ and $p(s) = x$, $p$ is a computable function and s is the lexicographically shortest string.

String $x$ is $c$-compressible if $\text{K}(x) \leq |x| - c$. $x$ is incompressible when $c = 1$.

Incompressible strings of every length exist.

The $\text{K}$ measure of complexity is not computable, and no algorithm can decide in general whether strings are incompressible.

## 7. Time Complexity

### 7.1 Measuring Complexity

The running time: regarded as a function purely of the length of the string representing the input.

Worst-case analysis: the longest running time of all inputs of a particular length.

Average-case analysis: the average of all the running times of inputs of a particular length.

The Running Time of a deterministic TM $M$:

- $f: \mathcal{N} \rightarrow \mathcal{N}$, where $f(n)$ is the maximum number of steps that $M$ uses on any input of length $n$.
- $M$ runs in time $f(n)$, and $M$ is an $f(n)$ time TM.

Let $f$ and $g$ be functions $f, g: \mathcal{N} \rightarrow \mathcal{R}^+$. Say that $f(n) = O(g(n))$ if positive integers $c$ and $n_0$ exist s.t. for every integer $n \geq n_0$, $f(n) \leq c \cdot g(n)$.

When $f(n) = O(g(n))$ we say that $g(n)$ is an asymptotic upper bound for $f(n)$.

Note that, for logarithms, the base is no longer necessary to be specified, since $O(\cdot)$ can always suppressing constant factors.

- $O(n^c) = O(2^{c \log n}), c > 0$ : polynomial bounds
- $O(2^{n^\delta}), \delta > 0$: exponential bounds

Let $f$ and $g$ be functions $f, g: \mathcal{N} \rightarrow \mathcal{R}^+$. Say that $f(n) = o(g(n))$ if positive integers $c$ and $n_0$ exist s.t. for every integer $n \geq n_0$, $f(n) < c \cdot g(n)$.

- The difference between $O(\cdot)$ and $o(\cdot)$ is analogous to the difference between $\leq$ and $<$.

🚀️ $\text{TIME}(t(n))$: the collection of all languages that are decidable by an $O(t(n))$ time TM.

- e.g., $\{0^k1^k | k \geq 0\} \in \text{TIME}(n^2)$.

Any language that can be decided in $o(n\log n)$ time on a single-tape TM is regular.

Every $t(n)$ ($t(n) > n$) time multitape TM has an equivalent $O(t^2(n))$ time single-tape TM.

The running time of a nondeterministic TM $N$ is a function $f: \mathcal{N} \rightarrow \mathcal{N}$, where $f(n)$ is the maximum number of steps that $N$ uses on any branch of its computation on any input of length $n$.

Every $t(n)$ ($t(n) > n$) time nondeterministic single-tape TM has an equivalent $2^{O(t(n))}$ time deterministic single-tape TM.

### 7.2 The Class P

All reasonable deterministic computational models are polynomial equivalent.

P is the class of languages that are decidable in polynomial time on a deterministic single-tape TM, i.e., $\text{P} = \bigcup_k \text{TIME}(n^k)$.

- P is invariant for all models of computation that are polynomially equivalent to the deterministic single-tape TM.
- P roughly corresponds to the class of problems that are realistically solvable on a computer.

Every CFL is a member of P.

- Proof idea: dynamic programming

### 7.3 The Class NP

A verifier for a language $A$ is an algorithm $V$, where $A = \{w | V \text{ accepts } \langle w, c \rangle \text{ for some string } c\}$.

- a polynomial time verifier runs in polynomial time in the length of $w$.
- a language is polnomial verifiable if it has a polynomial time verifier.
- $c$ is additional information called a certificate or proof.

NP is the class of languages that either:

- have polynomial time verifiers, or
- solvable in polynomial time on a nondeterministic TM.

P is a subset of NP.

🚀️ $\text{NTIME}(t(n)) = \{L | L \text{ is a language decided by a } O(t(n)) \text{ time nondeterministic TM}\}$

### 7.4 NP Completeness

Cook-Levin Theorem: $SAT \in P$ iff. $P = NP$. Or in other word, $SAT$ is NP-complete.

Language $A$ is polynomial time (mapping) reducible to language $B$ denoted as $A \leq_P B$, if a polynomial time computable function $f: \Sigma^* \rightarrow \Sigma^*$ exists, where for every $w$, $w \in A \Leftrightarrow f(w) \in B$.

- The function $f$ is called the polynomial time reduction of $A$ to $B$.

$3SAT = \{\langle \phi \rangle | \phi \text{ is a satisfiable 3CNF-formula}\}$. 3CNF-formula is all the clauses have 3 literals.

A language $B$ is NP-complete if:

1. $B$ is in NP.
2. Every $A$ in NP is polynomial time reducible to $B$.

$SAT$ is NP-complete.

- Proof idea: to construct a polynomial time reduction for each language $A$ in NP to $SAT$. For each input $w$, produces a Boolean formula $\phi$ that simulates the NP machine for $A$ on $w$. If the machine accepts, $\phi$ has a satisfying assignment corresponding to the accepting computation.

$3SAT$ is NP-complete.

- it is usually used to show the NP-completeness of other languages.

### 7.5 Additional NP-Complete Problems

Useful method to prove some language being NP-complete: find a reduction of $3SAT$ to that language.

## 8. Space Complexity

For a deterministic TM $M$, the space complexity is the function $f: \mathcal{N} \rightarrow \mathcal{N}$, where $f(n)$ is the maximum number of tape cells that $M$ scans on any input of length $n$.

- If $M$ is a nondeterministic TM where all branches halt on all inputs, $f(n)$ is the maximum number of tape cells that $M$ scans on any branch of its computation for any input of length $n$.

🚀️$\text{SPACE}(f(n)) = \{L | L \text{ is a language decided by an } O(f(n)) \text{ space deterministic TM}\}$

🚀️$\text{NSPACE}(f(n)) =  \{L | L \text{ is a language decided by an } O(f(n)) \text{ space nondeterministic TM} \}$

$\overline{ALL_\text{NFA}} = \Sigma^* - \{\langle A \rangle | A \text{ is a NFA and } L(A) = \Sigma^*\}$ is not known to be in NP or coNP, but in $\text{NSPACE}(O(n))$.

### 8.1 Savitch's Theorem

For any function $f: \mathcal{N} \rightarrow \mathcal{R}^+$, where $f(n) \geq n$, $\text{NSPACE}(f(n)) \subseteq \text{SPACE}(f^2(n))$.

- Proof idea: use a deterministic, recursive algorithm to solve the yieldability problem, by searching for an intermediate configuration and recursively testing whether each of the two parts can be achieved within a half of steps.
- The yieldability problem: Given 2 configurations of the NTM, $c_1$ and $c_2$, together with a number $t$, the task is to test whether the NTM can get from $c_1$ to $c_2$ within $t$ steps.

Savitch's theorem shows that deterministic machines can simulate nondeterministic machines by using a surprisingly small amount of space.

### 8.2 The Class PSPACE

PSPACE is the class of languages that are decidable in polynomial space on a deterministic TM. i.e., $\text{PSPACE} = \bigcup_{k} \text{SPACE}(n^k)$.

$\text{NPSPACE} = \text{PSPACE}$ by Savitch's theorem.

🚀️$\text{P} \subseteq \text{NP} \subseteq \text{PSPACE} = \text{NPSPACE} \subseteq \text{EXPTIME}$

### 8.3 PSPACE-Completeness

A language $B$ is PSPACE-complete if:

1. $B$ is in PSPACE.
2. Every $A$ in PSPACE is polynomial **time** reducible to $B$.

Note: Whenever we define complete problems for a complexity class, the reduction model must be more limited than the model used for defining the class itself. (Therefore, we use polynomial time reducible instead of * space reducible.)

$TQBF = \{\langle \phi \rangle | \phi \text{ is a true fully quantified Boolean formula}\}$ is PSPACE-complete.

- Proof idea: Firstly, showing $\text{TQBF}$ is in PSPACE by assigning truth values to the variables and recursively evaluating the formulas. Then presenting every language $A$ in PSPACE with a polynomial space-bounded TM for $A$ can be reduced to a $\text{TQBF}$ formula $\phi$. $\phi$ is true iff the machine accepts. The idea is similar to the proof of Savitch's theorem.

$FORMULA-GAME = \{\langle \phi \rangle | \text{Player E has a winning strategy in the formula game associated with } \phi\}$ is PSPACE-complete.

- Proof idea: $\phi \in TQBF$.

### 8.4 The Classes L and NL

$\text{L} = \text{SPACE}(\log n)$.

$\text{NL} = \text{NSPACE}(\log n)$.

$\{0^k1^k | k \geq 0\} \in \text{L}$.

- Proof idea: using a TM that counts the remaining 0s and 1s, and records the counters with logarithmic space.

$PATH = \{\langle G, s, t \rangle | G \text{ is a directed graph that has a directed path from } s \text{ to } t\} \in \text{NL}$.

- Proof idea: a nondeterministic TM can "guess" the next node to go and simply store the current node.

The configuration of $M$ which has a seperate read-only input tape: given the input $w$, a configuration is a setting of state, the work tape, and the positions of the two tape heads.

If $M$ runs in $f(n)$ space and $w$ is an input of length $n$, the number of configurations of $M$ is $n2^{O(f(n))}$.

$\text{NL} \subseteq \text{EXPTIME}$ **(NOT TIGHT!!)**, and Savitch's theorem still holds for any $f(n) \geq \log n$.

### 8.5 NL-Completeness

Language $A$ is log space reducible to language $B$, i.e., $A \leq_L B$, if $A$ is mapping reducible to $B$ by means of a log space computable function $f$.

A language $B$ is NL-complete if:

1. $B \in \text{NL}$.
2. Every $A$ in NL is log space reducible to $B$.

If $A \leq_L B$ and $B \in \text{L}$, then $A \in \text{L}$.

If any NL-complete language is in L, then $\text{L} = \text{NL}$.

$PATH$ is NL-complete.

- Proof idea: For any language $A$ in NL, construct a graph that represents the computation of the nondeterministic TM for $A$.

🚀️$\text{NL} \subseteq \text{P}$

- Proof idea: Recall that a TM that uses space $f(n)$ runs in time $n2^{O(f(n))}$, and $PATH \in \text{P}$.

### 8.6 NL equals coNL

- Proof idea: Showing $\overline{PATH}$ is in NL, thus establish that every problem in coNL is also in NL.

🚀️$\text{L} \subseteq \text{NL} =  \text{coNL} \subseteq \text{P} \subseteq \text{PSPACE}$

- coNL: the complexity class containing languages that are complements of languages in NL. (Not the complement of the class NL.)

## 9. Intractability

### 9.1 Hierarchy Theorems

A function $f: \mathcal{N} \rightarrow \mathcal{N}$, where $f(n)$ is at least $O(\log n)$, is called space constructible if the function that maps the string $1^n$ to the binary representation of $f(n)$ is computable in space $O(f(n))$.

- In other words, $f$ is space constructible if some $O(f(n))$ space TM $M$ exists that always halts with the binary representation of $f(n)$ on its tape when started on input $1^n$.

Space Hierarchy Theorem: For any space constructible function $f: \mathcal{N} \rightarrow \mathcal{N}$, a language $A$ exists that is decidable in $O(f(n))$ space but not in $o(f(n))$ space.

- Proof idea: describe a language $A$ that can be decided by an algorithm $D$ in $O(f(n))$ space, and prove that $D$ guarantees that $A$ is different from any language decidable in $o(f(n))$ by diagonalization.

For any two functions $f_1, f_2: \mathcal{N} \rightarrow \mathcal{N}$, where $f_1(n)$ is $o(f_2(n))$ and $f_2$ is space constructible, $\text{SPACE}(f_1(n)) \subsetneq \text{SPACE}(f_2(n))$.

For any 2 real numbers $0 \leq \epsilon_1 < \epsilon_2$, $\text{SPACE}(n^{\epsilon_1}) \subsetneq \text{SPACE}(n^{\epsilon_2})$.

$\text{NL} \subsetneq \text{PSPACE} \subsetneq \text{EXPSPACE}$.

- It establishes the existence of decidable problems that are intractable, in the sense that their decision procedures must use more than polynomial space.

A function $t: \mathcal{N} \rightarrow \mathcal{N}$, where $t(n)$ is at least $O(n \log n)$, is called time constructible if the function that maps the string $1^n$ to the binary representation of $t(n)$ is computable in time $O(t(n))$.

Time Hierarchy Theorem: For any time constructible function $t: \mathcal{N} \rightarrow \mathcal{N}$, a language $A$ exists that is decidable in $O(t(n))$ time but not decidable in $o(t(n)/\log t(n))$ time.

For any two functions $t_1, t_2: \mathcal{N} \rightarrow \mathcal{N}$, where $t_1(n)$ is $o(t_2(n) / \log t_2(n))$ and $t_2$ is time constructible, $\text{TIME}(t_1(n)) \subsetneq \text{TIME}(t_2(n))$.

For any 2 real numbers $0 \leq \epsilon_1 < \epsilon_2$, $\text{TIME}(n^{\epsilon_1}) \subsetneq \text{TIME}(n^{\epsilon_2})$.

$\text{P} \subsetneq \text{EXPTIME}$.

A language $B$ is EXPSPACE-complete if:

1. $B \in \text{EXPSPACE}$
2. every $A$ in $\text{EXPSPACE}$ is polynomial time reducible to $B$.

$EQ_{\text{REX}\uparrow} = \{\langle Q, R \rangle | Q \text{ and } R \text{ are equivalent regular expressions with exponentiation}\}$ is EXPSPACE-complete, which is actually intractable.

- Proof idea: (1) sketch an EXPSPACE algorithm for $EQ_{\text{REX}\uparrow}$, (2) show a language $A$ in EXPSPACE is polynomial time reducible to $EQ_{\text{REX}\uparrow}$ using the technique of reductions via computation histories.

### 9.2 Relativization

An oracle TM $M^A$ is a modified TM that has the additional capability of querying an oracle.

$\text{P}^A$: the class of languages decidable with a polynomial time oracle TM that uses oracle $A$.

e.g., $\text{NP} \subseteq \text{P}^{SAT}$ and $\text{coNP} \subseteq \text{P}^{SAT}$.

$\text{P}^{SAT}$ contains languages that are not in $\text{P}$, and $\text{NP}^{SAT}$ contains languages that are not in $\text{NP}$.

An oracle $A$ exists whereby $\text{P}^A \neq \text{NP}^A$, and an oracle $B$ exists whereby $\text{P}^B = \text{NP}^B$.

- Proof idea: Let $B$ be any PSPACE-complete problem such as $TQBF$, and exhibit oracle $A$ by construction.

It shows that diagonalization is not sufficient to seperate $\text{P}$ and $\text{NP}$.

To solve the $\text{P}$ vs. $\text{NP}$ question, we must analyze computations, not just simulate them.

### 9.3 Circuit Complexity

Boolean circuit: a collection of gates (including AND, OR, and NOT) and inputs connected by wires. Cycles aren't permitted.

A circuit family $C$ is an infinite list of circuits, $(C_0, C_1, C_2, ...)$, where $C_n$ has $n$ input variables. $C$ decides a language $A$ over $\{0, 1\}$ if, for every string $w$, $w \in A$ iff $C_n(w) = 1$.

The size of a circuit: the number of gates it contains.

A circuit is size minimal if no smaller circuits is equivalent to it.

The size complexity of a circuit family $(C_0, C_1, C_2, ...)$ is the function $f: \mathcal{N} \rightarrow \mathcal{N}$, where $f(n)$ is the size of $C_n$.

The circuit size complexity of a language is the size complexity of a minimal circuit family for that language.

Let $t: \mathcal{N} \rightarrow \mathcal{N}$ be a function, where $t(n) \geq n$. If $A \in \text{TIME}(t(n))$, then $A$ has circuit complexity $O(t^2(n))$.

- Proof idea: Let $M$ be a TM that decides $A$ in time $t(n)$. For each $n$, construct a circuit $C_n$ that simulates $M$ on inputs of length $n$.

$CIRCUIT\text{-}SAT = \{\langle C \rangle | C \text{ is a satisfiable Boolean circuit}\}$.

$CIRCUIT\text{-}SAT$ is NP-complete.

$3SAT$ is NP-complete. (Another proof to the Cook-Levin theorem.)

- Proof idea: reduct $CIRCUIT\text{-}SAT$ to $3SAT$ in polynomial time by using variables in $\phi$ to simulate each variable and each gate in a circuit.

## 10. Advanced Topics in Complexity Theory

### 10.1 Approximation Algorithms

An approximation algorithm for a minimization problem is $k$-optimal if it always finds a solution that is not more than $k$ times optimal. Similarly, for a maximization problem a $k$-optimal approximation algorithm always finds a solution that is at least $1/k$ times the size of the optimal.

### 10.2 Probabilistic Algorithms

A probabilistic TM $M$ is a type of nondeterministic T< in which each nondeterministic step is a "coin-flip" step.

- The probability of branch $b$: $\text{Pr}[b] = 2^{-k}$, where $k$ is the number of coin-flip steps that occur on branch $b$
- The probability that $M$ accepts $w$: $\text{Pr}[M \text{ accepts } w] = \sum_{b \text{ is an accepting branch }}\text{Pr}[b]$
- $\text{Pr}[M \text{ rejects } w] = 1 - \text{Pr}[M \text{ accepts } w]$.

For $0 \leq \epsilon < 1/2$, $M$ recognizes language $A$ with error probability $\epsilon$ if:

- $w \in A$ implies $\text{Pr}[M \text{ accepts } w] \geq 1 - \epsilon$
- $w \notin A$ implies $\text{Pr}[M \text{ rejects } w] \geq 1 - \epsilon$

🚀️$\text{BPP}$: the class of languages that are recognized by probabilistic polynomial time TM with an error probability of $1/3$.

Given $0 \leq \epsilon < 1/2$, for any polynomial $\text{poly}(n)$ a probabilistic polynomial time TM $M_1$ that operates with error probability $\epsilon$ has an equivalent probabilistic time TM $M_2$ that operates with an error probability of $2^{-\text{poly}(n)}$.

- Proof idea: $M_2$ simulates $M_1$ by running it a polynomial number of times and taking the majority vote of the outcomes.

Fermat's little theorem: If $p$ is prime and $a \in \mathcal{Z}^+_p$ then $a^{p-1} \equiv 1$ (mod $p$).

Fermat test: $p$ passes the test at $a$ iff $a^{p-1} \equiv 1$ (mod $p$).

A number is pseudoprime if it passes Fermat test for all smaller $a$ relatively prime to it. (Exceptions: Carmichael numbers, which are composite).

If $p$ is an odd prime number, $\text{Pr}[PRIME \text{ accepts } p] = 1$.

If $p$ is an odd composite number, $\text{Pr}[PRIME \text{ accepts } p] \leq 2^{-k}$.

$PRIMES \in \text{BPP}$.

$\text{RP}$ is the class of languages that are recognized by probabilistic polynomial time TM where inputs in the language are accepted with a probability of at least $1/2$ and inputs not in the language are rejected with a probability of $1$.

A branching program is a directed acyclic graph where all nodes are labeled by variables, except for 3 output nodes labeled 0 or 1. The nodes that are labeled by variables are called query nodes.

$EQ_{\text{ROBP}} = \{\langle B_1, B_2 \rangle | B_1 \text{ and } B_2 \text{ are equivalent read-once branching programs}\} \in \text{BPP}$.

- Proof idea: randomly select a non-boolean assignment to the variables $x_1, \dots, x_m$ and evaluate the equivalence of $B_1$ and $B_2$.

### 10.3 Alternation

An alternating TM: a nondeterministic TM with an additional feature that except for $q_{\text{accept}}$ and $q_{\text{reject}}$, the other states are divided into universal and existential states.

$\text{ATIME}(t(n)) = \{L | L \text{ is decided by an } O(t(n)) \text{ time alternating TM} \}$

$\text{ASPACE}(f(n)) = \{L | L \text{ is decided by an } O(f(n)) \text{ space alternating TM}\}$

$TAUT = \{\langle \phi \rangle | \phi \text{ is a tautology}\} \in \text{AP}$

$MIN\text{-}FORMULA = \{\langle \phi \rangle | \phi \text{ is a minimal Boolean formula}\} \in \text{AP}$

For $f(n) \geq n$, $\text{ATIME(f(n))} \subseteq \text{SPACE}(f(n)) \subseteq ATIME(f^2(n))$

- Proof idea:
  1. Using a deterministic space $O(f(n))$ TM to simulate an alternating time $O(f(n))$ TM $M$ by performing DFS with a   recursion stack.
  2. Similar to Savitch's theorem, constructing a general procedure for the yieldability problem.

For $f(n) \geq \log n$, $\text{ASPACE}(f(n)) = \text{TIME}(2^{O(f(n))})$

- Proof idea:
  1. $\text{ASPACE}(f(n)) \subseteq \text{TIME}(2^{O(f(n))})$: Constructing a deterministic time $2^{O(f(n))}$ TM to simulate an alternating time $O(f(n))$ TM $M$.
  2. $\text{ASPACE}(f(n)) \supseteq \text{TIME}(2^{O(f(n))})$: Recursively guess and verify the contents of the individual cells of the tableau.

$\text{AL} = \text{P}, \text{AP} = \text{PSPACE}, \text{APSPACE} = \text{EXPTIME}$

The Polynomial Time Hierarchy:

- $\sum_i$-alternating TM: an alternating TM that contains at most $i$ runs of universal or extential steps, starting with extential steps.
- $\prod_i$-alternating TM: similar but starting with universal steps.

$\sum_i \text{TIME}(f(n))$: the class of languages that a $\sum_i$ alternating TM can decide in $O(f(n))$ time.

$\prod_i \text{TIME}(f(n))$: the class of languages that a $\prod_i$ alternating TM can decide in $O(f(n))$ time.

- $\sum_i \text{P} = \bigcup_k \sum_i \text{TIME} (n^k)$
- $\prod_i \text{P} = \bigcup_k \prod_i \text{TIME}(n^k)$

$\text{PH} =  \bigcup_i \sum_i \text{P} = \bigcup_i \prod_i \text{P}$

- $\text{NP} = \sum_1 \text{P}$, $\text{coNP} = \prod_1 \text{P}$

### 10.4 Interactive Proof Systems
