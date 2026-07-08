# arXiv:2308.06103v1[cs.LG]11Aug2023

## COMPOSABLE FUNCTION-PRESERVING EXPANSIONS FOR TRANSFORMER ARCHITECTURES

Andrea Gesmundo1, Kaitlin Maile1,2 1 Google DeepMind, 2 IRIT, University of Toulouse, {agesmundo,kmaile}@google.com

ABSTRACT

Training state-of-the-art neural networks requires a high cost in terms of compute and time. Model scale is recognized to be a critical factor to achieve and improve the state-of-the-art. Increasing the scale of a neural network normally requires restarting from scratch by randomly initializing all the parameters of the model, as this implies a change of architecture’s parameters that does not allow for a straightforward transfer of knowledge from smaller size models.

In this work, we propose six composable transformations to incrementally increase the size of transformer-based neural networks while preserving functionality, allowing to expand the capacity of the model as needed. We provide proof of exact function preservation under minimal initialization constraints for each transformation. The proposed methods may enable efficient training pipelines for larger and more powerful models by progressively expanding the architecture throughout training. 1

1 INTRODUCTION

Transformer-based neural networks have gained widespread attention in recent years due to their impressive performance. The Transformer architecture, introduced by Vaswani et al. (2017), has become the standard for many natural language processing (NLP) tasks, including machine translation, text generation, and question answering. The success of transformer-based models is not limited to NLP: they have also been applied to various other domains, including computer vision, speech recognition, and recommendation systems. The largest and most performant of these models, large language models (LLMs) and vision and multimodal foundation models, are reaching billions to trillions of parameters (Dehghani et al., 2023; Touvron et al., 2023; Rae et al., 2021; Raffel et al., 2020).

However, each new model is generally trained from scratch, without reusing the capabilities acquired by previously trained smaller models. Furthermore, the size of the model is constant throughout training. The computational cost of training scales quadratically with model size due to the necessary increase in amount of training data (Hoffmann et al., 2022; Google, 2023; Kaplan et al., 2020). The ability to reuse parameters of a pretrained model or dynamically increase a model’s size during training could thus reduce the overall cost of training, but how to accomplish parameter reuse effectively without losing training progress is not straightforward.

To address these limitations, we propose parameter expansion transformations for transformer-based models that are exactly function preserving. These transformations increase the model size and thus the potential capacity of the model without changing its functionality, permitting continued training. These composable transformations operate on independent dimensions of the architecture, allowing for fine-grained architectural expansion.

Some previous works have also proposed function preserving parameter expansion transformations for transformer-based models (Chen et al., 2022; Shen et al., 2022; Wang et al., 2023; Mazzawi et al., 2023), extending from techniques for smaller convolutional and dense models (Chen et al., 2016; Evci et al., 2022). Our framework is so far the most comprehensive and composable set of function preserving transformations.

1Implementation of the proposed transformations and empirical tests of the function preservation property are available at: http://goo.gle/TransformerExpansions.

Output

Head

Linear

Multi Layer Perceptron

TransformerLayer

Normalization

N ✕

Multi Head Attention

Normalization

Positional Encoding

Input Embedding

Input

Figure 1: Representation of a standard Neural Network based on the Transformer architecture.

The contributions of this paper are six composable function preserving transformations applicable to Transformer architectures: 1) size of MLP internal representation, 2) number of attention heads, 3) size of the attention heads output representation, 4) size of the attention input representation, 5) size of the transformer layers input/output representations, 6) number of layers, summarized in Table

- 1. For each transformation, we provide proof of how the exactly function preserving property is achieved with a minimal set of constraints on the initialization of the added parameters.
- 2 TRANSFORMER ARCHITECTURE FORMALIZATION

This presentation is based on a particular instantiation of the transformer architecture: applications to variants (e.g. Encoder+Decoder, different normalization placement) can be obtained with simple extensions.

Figure 1 represents the standard Transformer architecture (Vaswani et al., 2017). The Input Embedding module maps the arbitrary input modality (e.g. image, text) into a bidimensional tensor I

, where s is

s×h

the sequence dimension and h is the hidden dimension. The TransformerArchitecture(·) is defined as a function that maps: I

, where o is the hidden dimension of the output representation. The Head component represents the output modality specific logic that maps O

#### → O

s×o

s×h

into a specific

s×o

output (e.g. a distribution over classes or text tokens). TransformerArchitecture(·) is defined as:

) = TransformerLayer◦N( I

) × Wout

, (1)

TransformerArchitecture( I

### + P

s×h

s×h

s×h

h×o

where Wout

are the positional embedding parameters, and TransformerLayer◦N(·) represents the recursive application of N transformer

are the parameters of the final linear projection, P

h×o

s×h

layers. The nth transformer layer is defined as:

′

′

+ MLPn(NormMLPn (I

TransformerLayern( In

#### )), I

) = I

n s×h

n s×h

s×h

)) ∀ n ∈ [1,N]. (2)

′

#### + MHAn(NormMHAn ( In

#### = In

n s×h

s×h

s×h

MLPn(·) is the Multi Layer Perceptron (i.e. feed forward layers), defined as:

× Wnl1 h×p

MLPn( X

) = ReLU( X

s×h

s×h

) × Wnl2 p×h

+ Bln2 s×h

+ Bln1 s×p

, (3)

where Wnl1 is the matrix of parameters of the first fully connected layer and Bln1 are its bias parameters broadcasted along the sequence dimension: Bln1

. Wnl2 and Bln2 are the parameters of

× bln1 1×h

#### = 1

s×1

s×h

the second fully connected layer. The broadcast operator applied to the bias parameters is omitted for simplicity. The size of the internal dimension of the MLP component is represented with p. The considered architecture instantiation assumes the uses of ReLU(·) (Glorot et al., 2011) as a non-linearity function as this is a common choice. The proposed transformations also maintain the function preserving property with alternative choices such as GELU(·) (Hendrycks & Gimpel, 2016).

MHAn(·) is the Multi Head Attention defined as:

··· HE s×v

MHAn( X

) = H1 s×v

s×h

× WnO

,

(E·v)×h

#### ×Wn,eQ

He s×v

= Attention( X

#### , X

s×h

h×k

#### ×Wn,eK

#### , X

s×h

h×k

#### ×Wn,eV

) ∀ e ∈ [1,E],

s×h

h×v

(4)

) = Softmax(√1k · Q

× K⊤ k×s

#### ) × V

,

Attention( Q

#### , K

#### , V

s×v

s×v

s×k

s×k

s×k

where E is the number of heads, k is the hidden dimension of key, K, and query, Q, and v is the hidden dimension of value, V. K⊤ represents the transpose of K. The concatenation of the representations produced by the attention heads is represented with the block notation: C = [A B].

As the normalization function in each component, we use RMSNorm (Zhang & Sennrich, 2019). The original definition of the transformer architecture uses LayerNorm, but RMSNorm has become a more common design choice in large language models (Raffel et al., 2020; Rae et al., 2021; Touvron et al., 2023). The key difference is only scaling the variance of the inputs and using scaling parameters, rather than also subtracting their mean and using bias parameters. Thus, we define Norm(·) as:

Normcn( X

) =

s×h

xi,j · gn,jc

h γ=1(xi,γ)2

1 h

| i∈[1,s] ∧ j∈[1,h] ∀n∈[1,N] ∧ c∈{MHA,MLP}, (5)

where gnc

identifies the vector of the scaling parameters of the Norm(·) instance of component c in the nth layer.

1×h

- 3 FUNCTION PRESERVING TRANSFORMATIONS

In this section, we define six function preserving transformations that can be applied to extend a transformer architecture to increase its scale while keeping its function unaltered, thus allowing to introduce new parameters to store additional knowledge while preserving the knowledge acquired so far. Each transformation is defined to target the expansion of one of the hyper-parameters of the architecture: p,E,v,k,h, and N, each controlling a distinct dimension of the scaling. The proposed transformations are summarized in Table 1.

|Name<br><br>|Transformation<br><br>|Function preserving constraint|
|---|---|---|
|Sec. 3.1: MLP expansion|Def. 3.1: to increase the MLP internal dimension p to p,ˆ add pˆ − p columns to the the first MLP weight matrix and bias vector and add pˆ− p rows to the second MLP weight matrix.|Thrm. 3.1: zero initialize the new pˆ− p rows of the second MLP weight matrix.|
|Sec. 3.2: Head addition|Def. 3.2: to increase the number of attention heads E, per head added, add v rows to the MHA output weight matrix.<br><br>|Thrm. 3.2: zero initialize the new v rows of the MHA output weight matrix.|
|Sec. 3.3: Heads expansion<br><br>|Def. 3.3: to increase the attention head representation dimension v to v,ˆ add vˆ − v columns to the value weight matrix and insert vˆ − v rows to each of E splits of the MHA output weight matrix.<br><br>|Thrm. 3.3: zero initialize the new vˆ − v rows inserted to each of E splits of the MHA output weight matrix.|
|Sec. 3.4: Attention expansion|Def. 3.4: to increase the key/query representation dimension k to k,ˆ add kˆ − k columns to the key/query weight matrices and scale the key weight matrix by k/ˆ<br><br>√<br><br>k.<br><br>|Thrm. 3.4: zero initialize the new kˆ−k columns of the key weight matrix.|
|Sec. 3.5: Hidden dimension expansion|Def. 3.5: to increase the transformer hidden dimension h to h,ˆ add hˆ − h columns to the positional encoding matrix, norm scaling vector, second MLP weight matrix and bias vector, MHA output weight matrix, and input representation matrix; add hˆ − h rows to the transformer output weight matrix, first MLP weight matrix, and key/query/value weight matrices; scale norm scaling vector by √<br><br>h/ h.ˆ<br><br>|Thrm. 3.5: zero initialize the new hˆ−h columns of the positional encoding matrix, norm scaling vector, second MLP weight matrix and bias vector, and MHA output weight matrix.|
|Sec. 3.6: Layer addition<br><br>|Def. 3.6: to increase the number of layers N to N,ˆ per layer added, insert new layer at position n and increment index of all following layers.|Thrm. 3.6: zero initialize the new layer’s MHA output weight matrix and weight matrix and bias vector of the second MLP layer.|

Table 1: Summary of proposed function preserving transformations.

For each transformation, we define how the existing parameters must be expanded and propose a set of minimal initialization constraints to obtain the function preserving property with proof.

The presented transformations can be combined to allow the joint extension of multiple dimensions of the transformer architecture. Furthermore, different subsets of such transformations can be applied incrementally, interleaving training iterations, as well as independently to different parts of the architecture.

Symbols denoting parameters, representations, and functions resulting from the application of the transformation discussed in each of the following subsection are indicated with the “hat” symbol: ˆ.

- 3.1 MLP EXPANSION

The MLP expansion transformation can be applied to expand the scale of the MLP by expanding the dimension of its internal representation. This scaling dimension is controlled by the hyper-parameter p introduced in Equation 3.

- Definition 3.1 (MLP expansion). Given a Transformer model as defined in Section 2, the internal

dimension of MLPn ∀ n∈[1,N] can be increased from p to pˆby applying the following parametermatrix transformations:

 → Wˆ nl1 h×pˆ

Wnl1 h×p

:= Wnl1 h×p

MWln 1 h×(ˆp−p)

, (6)

 → bˆln1 1×pˆ

bln1 1×p

:= bln1 1×p

mbln1 1×(ˆp−p)

, (7)

  

  , (8)

### Wnl2

p×h MWln 2 (ˆp−p)×h

 → Wˆ nl2 pˆ×h

Wnl2 p×h

:=

where MWln 1

, and MWln 2

are matrices of the specified shape. For the purpose of defining

, mbln1

1×(ˆp−p)

h×(ˆp−p)

(ˆp−p)×h

of the MLP expansion transformation, the values of these matrices can be assumed to be arbitrary. Constraints on their initializer functions are introduced below to achieve the function preserving property.

No other modifications to the Transformer architecture are required since the MLPn(·) function

- (Equation 3) still inputs and outputs matrices of shape s × h after the transformation.

| |
|---|

- Theorem 3.1 (Function preserving MLP expansion).

MWln 2 (ˆp−p)×h

:= 0

(ˆp−p)×h

(9)

=⇒

ReLU( X

s×h

× Wnl1 h×p

+ Bln1 s×p

) × Wnl2 p×h

+ Bln2 s×h

= ReLU( X

s×h

× Wˆ nl1 h×p

+ Bˆln1 s×p

) × Wˆ nl2 p×h

+ Bln2 s×h

(10)

Informally: zero initializing MWln 2

(ˆp−p)×h

implies the function preservation property for the MLP expansion transformation.

- See Appendix A.1 for proof.

The MLP expansion transformation can be applied to all the MLP blocks to maintain the MLP internal dimension uniformly across all the layers. However, it can also be applied to only a subset of the layers independently to allow experimenting with different capacity at different depths.

3.2 HEAD ADDITION The Head addition transformation can be applied to add new heads in a MHA component. This scaling dimension is controlled by the hyper-parameter E introduced in Equation 4.

Definition 3.2 (Head addition). Given a Transformer model as defined in Section 2, a new head can be added to MHAn(·) ∀ n ∈ [1,N] by introducing new input projection matrices: Wn,EQ +1 h×k

,Wn,EK +1

h×k

,Wn,EV +1

h×v

and applying the following parameter-matrix transformation to the output projection matrix:

WnO (E·v)×h

 → Wˆ nO

((E+1)·v)×h

:=

  

WnO (E·v)×h

MWOn v×h

  . (11)

No other modifications to the Transformer architecture are required since the MHAn(·) function

- (Equation 4) still inputs and outputs matrices of shape s × h after the transformation.

| |
|---|

The Head addition transformation is defined to add one new head. The transformation can be applied multiple times to add an arbitrary number of new heads.

- Theorem 3.2 (Function preserving head addition).

MWOn v×h

:= 0

v×h

=⇒ H1 s×v

··· HE s×v

× WnO

(E·v)×h

= H1 s×v

··· H(E+1)

s×v

× Wˆ nO

((E+1)·v)×h

(12)

Informally: zero initializing MWOn

v×h

implies the function preservation property for the head addition transformation.

- See Appendix A.2 for proof.

The head addition transformation can be applied to all the MHA blocks to maintain the number of MHA heads uniformly across all the layers. However, it can also be applied to only a subset of the layers independently to allow experimenting with different capacity at different depths.

3.3 HEADS EXPANSION

The Heads expansion transformation can be applied to expand the dimension of the representation generated by each attention heads. This scaling dimension is controlled by the hyper-parameter v introduced in Equation 4.

Definition 3.3 (Heads expansion). Given a Transformer model as defined in Section 2, the dimension of representation generated by the attention heads, He

s×v

∀ e∈[1,E], of MHAn ∀ n∈[1,N] can be increased from v to vˆ by applying the following parameter-matrix transformations:

Wn,eV h×v

 → Wˆ n,eV

h×vˆ

:= Wn,eV

h×v

MWVn,e h×(ˆv−v)

∀ e ∈ [1,E], (13)

Wn,eO v×h

 → Wˆ n,eO

vˆ×h

:=



 

Wn,eO v×h

MWOn,e (ˆv−v)×h



 

∀ e ∈ [1,E], (14)

where Wn,eO

v×h

is the eth “split” of WnO

(E·v)×h

along the (E · v) dimension:

WnO (E·v)×h

:=



 

. Wn,eO v×h .

| e ∈ [1,E].



 

(15)

No other modifications to the Transformer architecture are required since the MHAn(·) function (Equation 4) still inputs and outputs matrices of shape s × h after the transformation.

| |
|---|

- Theorem 3.3 (Function preserving heads expansion).

where:

MWOn,e (ˆv−v)×h

:= 0

(ˆv−v)×h

=⇒ H1 s×v

··· HE s×v

× WnO

(E·v)×h

Hˆe s×vˆ

#### ×Wn,eQ

#### = Attention( X

#### , X

s×h

h×k

#### ×Wn,eK

s×h

h×k

= H ˆ1 s×vˆ

··· HˆE s×vˆ

× Wˆ nO

(E·vˆ)×h

(16)

×Wˆ n,eV

) (17)

#### , X

s×h

h×vˆ

Informally: zero initializing MWOn,e

implies the function preservation property for the head expansion transformation.

(ˆv−v)×h

- See Appendix A.3 for proof

The heads expansion transformation can be applied to all heads of all the MHA blocks to maintain the attention head representation dimension uniformly across all the layers. However, it can also be applied to only a subset of the layers or even a subset of attention heads independently to allow experimenting with different capacity at different parts of the architecture.

3.4 ATTENTION EXPANSION

The Attention expansion transformation can be applied to expand the key and query representations whose inner product produces the attention weights matrix. This scaling dimension is controlled by the hyper-parameter k introduced in Equation 4.

Definition 3.4 (Attention expansion). Given a Transformer model as defined in Section 2, the dimension of representations generating the attention weights of MHAn ∀ n∈[1,N] can be increased from k to kˆ by applying the following parameter-matrix transformations:

Wn,eQ h×k

 → Wˆ n,eQ

h×kˆ

:=

 Wn,eQ

h×k

MWQn,e h×(kˆ−k)

  ∀ e ∈ [1,E], (18)

Wn,eK h×k

 → Wˆ n,eK

h×kˆ

:=

 

kˆ √

k · Wn,eK

h×k

MWKn,e h×(kˆ−k)

  ∀ e ∈ [1,E]. (19)

| |
|---|

Theorem 3.4 (Function preserving attention expansion).

MWKn,e h×(kˆ−k)

:= 0

h×(kˆ−k)

(20)

=⇒

Attention( X

s×h

×Wn,eQ

h×k

, X

s×h

×Wn,eK

h×k

, X

s×h

×Wn,eV

h×v

) = Attention( X

s×h

×Wˆ n,eQ

h×kˆ

, X

s×h

×Wˆ n,eK

h×kˆ

, X

s×h

×Wn,eV

h×v

)

(21) Informally: zero initializing MWKn,e

h×(kˆ−k)

implies the function preservation property for the attention expansion transformation.

- See Appendix A.4 for proof.

In most transformer implementations, k = v. In such cases, the attention expansion may be performed jointly with the head expansion.

The attention expansion transformation can be applied to all heads of all the MHA blocks to maintain the key/query representation dimension uniformly across all the layers. However, it can also be applied to only a subset of the layers or even a subset of attention heads independently to allow experimenting with different capacity at different parts of the architecture.

- 3.5 HIDDEN DIMENSION EXPANSION

The Hidden dimension expansion transformation can be applied to expand the dimension of the representation produced by the transformer layers. This scaling dimension is controlled by the hyper-parameter h introduced in Equation 1.

- Definition 3.5 (Hidden dimension expansion). Given a Transformer model as defined in Section 2, the dimension of the transformer layers’ input/output representation can be increased from h to hˆ by applying the following parameter-matrix transformations:

###  → Pˆ

### P

#### := P

s×h

s×h

s×hˆ

MP s×(hˆ−h)

, (22)

### Wout

- h×o

 → Wˆ out

hˆ×o

:=

  

Wout h×o

MWout (hˆ−h)×o

  , (23)

gnc 1×h

 → gˆnc

1×hˆ

:=

√

h hˆ

· gnc

1×h

mg,cn 1×(hˆ−h)

∀n∈[1,N] ∧ c∈{MHA,MLP}, (24)

Wnl1

- h×p

 → Wˆ nl1 hˆ×p

:=

  

### Wnl1

h×p MWl1 (hˆ−h)×p

 → Wˆ nl2 p×hˆ

Wnl2 p×h

:= Wnl2 p×h

   ∀n∈[1,N], (25)

MWln 2 p×(hˆ−h)

∀n∈[1,N], (26)

 → bˆln2 1×hˆ

bln2 1×h

:= bln2 1×h



 → Wˆ n,eQ

Wn,eQ h×k

:=

 

hˆ×k



 → Wˆ n,eK

Wn,eK h×k

:=

 

hˆ×k



 → Wˆ n,eV

Wn,eV h×v

:=

 

hˆ×v

mbln2 1×(hˆ−h)

Wn,eQ h×k

MWQn,e (hˆ−h)×k

Wn,eK h×k

MWKn,e (hˆ−h)×k

Wn,eV h×v

MWVn,e (hˆ−h)×v

∀n∈[1,N], (27)



 



 



 

∀n∈[1,N] ∧ e∈[1,E], (28)

∀n∈[1,N] ∧ e∈[1,E], (29)

∀n∈[1,N] ∧ e∈[1,E], (30)

 → Wˆ nO

:= WnO

WnO (E·v)×h

(E·v)×hˆ

(E·v)×h

MWOn (E·v)×(hˆ−h)

∀n∈[1,N], (31)

and modifying the embedding function to produce an extended input representation:

ˆI

MI s×(hˆ−h)

. (32)

#### := I

s×h

s×hˆ

For example, a token embedding table can be expanded by adding (hˆ − h) randomly initialized columns, mapping the same vocabulary into an extended embedding.

- Theorem 3.5 (Function preserving hidden dimension expansion).

MP s×(hˆ−h)

:= 0

s×(hˆ−h)

| |
|---|

(33)

MWln 2 p×(hˆ−h)

∀n∈[1,N] (34)

:= 0

p×(hˆ−h)

mbln2 1×(hˆ−h)

∀n∈[1,N] (35)

:= 0

1×(hˆ−h)

MWOn (E·v)×(hˆ−h)

∀n∈[1,N] (36)

:= 0

(E·v)×(hˆ−h)

#### =⇒

MI s×(hˆ−h)

:= 0

s×(hˆ−h)

(37)

=⇒

ˆIn s×hˆ

= [ In

s×h

] ∀n∈[1,N + 1] (38)

0

s×(hˆ−h)

TransformerLayer◦N( I

s×h

= TransformerLayerˆ ◦N( I

) × Wout

### + P

s×h

h×o

s×h

### + Pˆ

) × Wˆ out

s×hˆ

hˆ×o

(39)

where IN+1

refers to the representations outputted by the last transformer layer, and In

∀n∈[1,N]

s×h

s×h

refers to the representation inputted by the nth transformer layer. Symbols denoting parameters, representations and functions resulting from the application of the transformation discussed in this section are indicated with the “hat” ˆ symbol.

Informally: zero initializing the specified matrices implies the function preservation property for the hidden dimension expansion transformation.

- See Appendix A.5 for proof.

The hidden dimension expansion transformation must be applied to all MHA blocks to maintain the hidden dimension uniformly across all the layers, due to the skip connections used throughout the architecture.

- 3.6 LAYER ADDITION

The Layer addition transformation can be applied to insert an new layer at any depth of the current Transformer architecture. This scaling dimension is controlled by the hyper-parameter N introduced in Equation 1.

- Definition 3.6 (Layer addition). A new TransformerLayer(·) whose parameters allow to input and output matrices of x × h can be inserted in the sequence of the pre-existing N layers. The new transformer layer can be inserted at any position n ∈ [1,N+1]. The index of the downstream layers is incremented by one.

| |
|---|

- Theorem 3.6 (Function preserving layer addition). With n being the index of the added layer:

 

WnO (E·v)×h

:= 0

(E·v)×h Wnl2

:= 0

(40)

=⇒ TransformerLayern( In

) = In

p×h

p×h

s×h

s×h



bln2 1×h

:= 0

1×h

Informally: Zero initializing the parameters of the output projections of the MLP and MHA implies that the added transformer layer output is equivalent to the input.

- See Appendix A.6 for proof.

- 4 RELATED WORK

Some existing works have proposed function preserving transformer expansion operators, but none cover all six dimensions as proposed in this work. Bert2BERT (Chen et al., 2022) proposes function preserving width expansions of the MLP internal dimension, hidden dimension, and number of attention heads. Shen et al. (2022) achieve function preserving width expansion, although constrained to doubling of all matrix and vector dimensions, and depth expansion via zero initialization of LayerNorm and bias parameters. Yao et al. (2023) use masking on new hidden MLP neurons, attention heads, and layers to achieve function preservation. Wang et al. (2023) use an inner optimization to learn a linear mapping for parameter expansion in depth and width, but without constraints for function preservation. Notably, our transformations form a function preserving subspace of their learnable space. Deep Fusion (Mazzawi et al., 2023) extends the concept of expansion to multiple source models, where the special case of self-fusion achieves function preserving width expansion. Of these works, some methods are nearly function preserving but admit gaps due to LayerNorm discrepancies (Chen et al., 2022; Mazzawi et al., 2023). No known works consider scaling factors, as we address in Equations 19 and 24, nor RMSNorm.

- 5 CONCLUSION

We have defined six transformations that can be applied to a transformer model to increase the scale of all the different aspects of the architecture: 1) size of MLP internal representation, 2) number of attention heads, 3) size of the attention heads output representation, 4) size of the attention input representation, 5) size of the transformer layers input/output representations, 6) number of layers. For each of these transformations, we have provided a proof of exact function preservation given a minimal set of constraints on the initialization of the added parameters. These six transformations are composable to permit many different ways to scale a transformer-based model while preserving its function.

We note that, there exist alternative definitions to such transformations that achieve functionpreservation without requiring zero initialization. However, the form of the proposed transformations is intended to be simple yet minimally constraining. The space of possible initialization strategies may be explored with the aim to optimize for training in an empirical context.

In future work, these transformations may be applied in the training of a new large model by initializing a smaller model, training it under reduced data and computational complexity requirements, and incrementally scaling it to larger sizes throughout training to the desired final size. They may also be used to generate a family of models that are trained for the same task but at different sizes: all models within the family can begin from the same checkpoint from training the smallest model, then

each successively sized model can be branched and finetuned at its final size. Finally, neural architecture search (NAS) techniques could be applied to determine optimal transformation scheduling and architectural progression for a given task and compute budget.

- 6 ACKNOWLEDGEMENTS We would like to thank Jeffrey Pennington and Utku Evci for their input to this work.

REFERENCES

Cheng Chen, Yichun Yin, Lifeng Shang, Xin Jiang, Yujia Qin, Fengyu Wang, Zhi Wang, Xiao Chen, Zhiyuan Liu, and Qun Liu. bert2BERT: Towards reusable pretrained language models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2134–2148, 2022.

Tianqi Chen, Ian J. Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowledge transfer. CoRR, abs/1511.05641, 2016.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim M. Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Collier, Alexey A. Gritsenko, Vighnesh Birodkar, Cristina Nader Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Paveti’c, Dustin Tran, Thomas Kipf, Mario Luvci’c, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. ArXiv, abs/2302.05442, 2023.

Utku Evci, Max Vladymyrov, Thomas Unterthiner, Bart van Merrienboer, and Fabian Pedregosa. GradMax: Growing neural networks using gradient information. ArXiv, abs/2201.05125, 2022.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In

International Conference on Artificial Intelligence and Statistics, 2011. Google. PaLM 2 technical report. arXiv preprint arXiv:2305.10403, 2023. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELUs). arXiv: Learning, 2016.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Jared Kaplan, Sam McCandlish, T. J. Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeff Wu, and Dario Amodei. Scaling laws for neural language models. ArXiv, abs/2001.08361, 2020.

Hanna Mazzawi, Xavi Gonzalvo, and Michael Wunder. Deep fusion: Efficient network training via pre-trained initializations. arXiv preprint arXiv:2306.11903, 2023.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, Irina Higgins, Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, L. Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, N. K. Grigorev, Doug Fritz, Thibault Sottiaux,

Mantas Pajarskas, Tobias Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew G. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem W. Ayoub, Jeff Stanway, L. L. Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training Gopher. ArXiv, abs/2112.11446, 2021.

Colin Raffel, Noam M. Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv, abs/1910.10683, 2020.

Sheng Shen, Pete Walsh, Kurt Keutzer, Jesse Dodge, Matthew Peters, and Iz Beltagy. Staged training for transformer language models. In International Conference on Machine Learning, pp. 19893–19908. PMLR, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. LLaMa 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. ArXiv, abs/1706.03762, 2017.

Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid Karlinsky, Rogerio Feris, David Daniel Cox, Zhangyang Wang, and Yoon Kim. Learning to grow pretrained models for efficient transformer training. In The 11th International Conference on Learning Representations, 2023.

Yiqun Yao, Zheng Zhang, Jing Li, and Yequan Wang. 2x faster language model pre-training via masked structural growth. arXiv preprint arXiv:2305.02869, 2023.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. ArXiv, abs/1910.07467, 2019.

A PROOFS

- A.1 MLP EXPANSION Proof.

ReLU( X

s×h

× Wˆ nl1 h×p

+ Bˆln1 s×p

) × Wˆ nl2 p×h

= ReLU X

s×h

× Wnl1 h×p

MWln 1 h×(ˆp−p)

+ Bln1 1×p

Mbln1 1×(ˆp−p)

×

  

Wnl2 p×h

0

(ˆp−p)×h

  

= ReLU X

s×h

× Wnl1 h×p

X

s×h

× MWln 1

h×(ˆp−p)

+ Bln1 1×p

Mbln1 1×(ˆp−p)

×

  

Wnl2 p×h

0

(ˆp−p)×h

  

= ReLU X

s×h

× Wnl1 h×p

+ Bln1 1×p

X

s×h

× MWln 1

h×(ˆp−p)

+ Mbln1

1×(ˆp−p)

×

  

Wnl2 p×h

0

(ˆp−p)×h

  

= ReLU( X

s×h

× Wnl1 h×p

+ Bln1 1×p

) ReLU( X

s×h

× MWln 1

h×(ˆp−p)

+ Mbln1

1×(ˆp−p)

) ×

  

Wnl2 p×h

0

(ˆp−p)×h

  

= ReLU( X

s×h

× Wnl1 h×p

+ Bln1 1×p

) × Wnl2 p×h

+ ReLU( X

s×h

× MWln 1

h×(ˆp−p)

+ Mbln1

1×(ˆp−p)

) × 0

(ˆp−p)×h

= ReLU( X

s×h

× Wnl1 h×p

+ Bln1 1×p

) × Wnl2 p×h

(41)

| |
|---|

Note that it is not necessary to impose any constraints on the values of MWln 1

h×(ˆp−p)

and mbln1

1×(ˆp−p)

to achieve function preservation property. Thus, these two matrices can be initialized arbitrarily.

- A.2 HEAD ADDITION Proof.

··· H(E+1)

H1 s×v

s×v

× Wˆ nO

((E+1)·v)×h

··· H(E+1)

= H1 s×v

s×v

  

  

WnO (E·v)×h

×

### 0

v×h

··· HE s×v

= H1 s×v

H(E+1) s×v

··· HE s×v

= H1 s×v

× WnO

··· HE s×v

= H1 s×v

× WnO

(E·v)×h

- A.3 HEADS EXPANSION Proof.

#### =⇒

Sn,e s×s

:= Softmax

1 √

k · ( X

s×h

Hˆe s×vˆ

×Wn,eQ

= Attention( X

s×h

h×k

×Wˆ n,eV

× X

= Sn,e s×s

s×h

h×vˆ

× Wn,eV

× X

= Sn,e s×s

s×h

h×v

×Wn,eV

× X

= Sn,e s×s

s×h

h×v

×Wn,eV

× ( X

= Sn,e s×s

s×h

h×v

#### =⇒

= He s×v

× ( X

Sn,e s×s

s×h

H ˆ1 s×vˆ

··· HˆE s×vˆ

× Wˆ nO

(E·vˆ)×h



= ··· Hˆe s×vˆ

··· | e ∈ [1,E] ×

 

  

  

WnO (E·v)×h

×

### 0

v×h

(E·v)×h

× 0

+ H(E+1)

v×h

s×v

(42)

| |
|---|

)⊤ (43)

×Wn,eQ

×Wn,eK

#### ) × ( X

s×h

h×k

h×k

×Wˆ n,eV

×Wn,eK

#### , X

#### , X

)

s×h

s×h

h×k

h×vˆ

MWVn,e h×(ˆv−v)

× MWVn,e

#### X

s×h

h×(ˆv−v)

× MWVn,e

× ( X

) Sn,e s×s

)

s×h

h×(ˆv−v)

× MWVn,e

) (44)

h×(ˆv−v)

. Wˆ n,eO v×h .



| e ∈ [1,E]

 

= ··· Hˆe s×vˆ

× Wˆ n,eO

··· | e ∈ [1,E]

v×h

  ··· Hˆe

×

=

s×vˆ

  ··· He

=

s×v

  ··· | e ∈ [1,E]

  

  

Wn,eO v×h

### 0

(ˆv−v)×h

  

  ··· | e ∈ [1,E]

  

Wn,eO v×h

× MWVn,e

× ( X

) ×

Sn,e s×s

s×h

### 0

h×(ˆv−v)

(ˆv−v)×h

× Wn,eO

× MWVn,e

= ··· He s×v

× ( X

) × 0

+ Sn,e s×s

s×h

(ˆv−v)×h

v×h

h×(ˆv−v)

··· | e ∈ [1,E]

× Wn,eO

= ··· He s×v

+ 0

s×h

v×h

··· | e ∈ [1,E]

× Wn,eO

= ··· He s×v

··· | e ∈ [1,E]

v×h



= ··· He s×v

··· | e ∈ [1,E] ×

 

. Wn,eO v×h .



| e ∈ [1,E]

 

··· HE s×v

= H1 s×v

× WnO

(E·v)×h

- A.4 ATTENTION EXPANSION

Proof.

1

×Wˆ n,eK

×Wˆ n,eQ

)⊤

#### ) × ( X

· ( X

kˆ

s×h

s×h

h×kˆ

h×kˆ

1

=

kˆ

1

=

kˆ

  X

 Wn,eQ

×

·

s×h

h×k

  X

×Wn,eQ

·

s×h

h×k

 

  × X

MWQn,e h×(kˆ−k)

×

s×h

  ×

kˆ √

× MWQn,e

#### X

s×h

h×(kˆ−k)

kˆ √

k · Wn,eK

h×k

### 0

h×(kˆ−k)

⊤

×Wn,eK

k · X

s×h

h×k

× 0

#### X

s×h

h×(kˆ−k)

⊤

(45)

| |
|---|

  X

1

×Wn,eQ

·

=

kˆ

s×h

h×k

  X

kˆ √

1

·

k ·

=

kˆ

s×h

  X

1 √

×Wn,eQ

k ·

=

s×h

h×k

  X

1 √

×Wn,eQ

k ·

=

s×h

h×k

 ( X

1 √

×Wn,eQ

k ·

=

s×h

h×k

1 √

×Wn,eQ

k · ( X

=

s×h

h×k

1 √

×Wn,eQ

k · ( X

=

s×h

h×k

  ×

⊤

kˆ √

×Wn,eK

× MWQn,e

k · X

### 0

#### X

s×h

s×h

s×(kˆ−k)

h×k

h×(kˆ−k)

####   × X

⊤

×Wn,eQ

× MWQn,e

×Wn,eK

#### X

### 0

s×h

s×h

s×(kˆ−k)

h×k

h×k

h×(kˆ−k)

####   × X

⊤

× MWQn,e

×Wn,eK

#### X

### 0

s×h

s×h

s×(kˆ−k)

h×k

h×(kˆ−k)

  ×

 

 

)⊤ 0 (kˆ−k)×s

×Wn,eK

( X

s×h

× MWQn,e

h×k

#### X

s×h

h×(kˆ−k)

 

#### )⊤ + ( X

×Wn,eK

× MWQn,e

#### ) × ( X

) × 0

s×h

s×h

(kˆ−k)×s

h×k

h×(kˆ−k)

)⊤ + 0

×Wn,eK

#### ) × ( X

s×s

s×h

h×k

)⊤ (46)

×Wn,eK

#### ) × ( X

s×h

h×k

| |
|---|

- A.5 HIDDEN DIMENSION EXPANSION

Proof. We demonstrate ˆIn

= [ In

s×h

s×hˆ

Base case n = 0:

] ∀n∈[0,N] by induction on n.

0

s×(hˆ−h)

ˆI0 s×hˆ

= ˆI

### + Pˆ

s×h

s×hˆ

#### = I

s×h

0

s×(hˆ−h)

### + P

s×h

### 0

s×(hˆ−h)

#### = I

### + P

s×h

s×h

0

s×(hˆ−h)

. (47)

Induction step, assuming ˆIn

s×hˆ

= [ In

s×h

] holds:

0

s×(hˆ−h)

ˆiµ,j · gˆn,jMHA

| µ∈[1,s] ∧ j∈[1,hˆ]

NormMHAn ( ˆIn

) =

h ˆ γ=1(ˆiµ,γ)2

s×h

1 hˆ

= NormMHAn ([ In

0

])

s×(hˆ−h)

s×h

 

 

iµ,j · gˆn,jMHA

- 0 · gˆn,jMHA

- 1 hˆ

| µ∈[1,s] ∧ j∈[h + 1,hˆ]

| µ∈[1,s] ∧ j∈[1,h]

=

h ˆ γ=1(ˆiµ,γ)2

h ˆ γ=1(ˆiµ,γ)2

1 hˆ

 

 

iµ,j · gˆn,jMHA

| µ∈[1,s] ∧ j∈[1,h] 0

=

h ˆ γ=1(ˆiµ,γ)2

s×(hˆ−h)

1 hˆ

 

 

iµ,j · gˆn,jMHA

| µ∈[1,s] ∧ j∈[1,h] 0

=

1 hˆ ( hγ=1(iµ,γ)2 + hγ ˆ=h+1 0)

s×(hˆ−h)

 

 

iµ,j · gˆn,jMHA

| µ∈[1,s] ∧ j∈[1,h] 0

=

h γ=1(iµ,γ)2

s×(hˆ−h)

1 hˆ

  

  

√ √h

· gn,jMHA

iµ,j ·

hˆ

| µ∈[1,s] ∧ j∈[1,h] 0

=

h γ=1(iµ,γ)2

s×(hˆ−h)

1 hˆ

 

 

iµ,j · gn,jMHA

| µ∈[1,s] ∧ j∈[1,h] 0

=

h γ=1(iµ,γ)2

s×(hˆ−h)

1 h

= NormMHAn ( In

(48)

) 0

s×(hˆ−h)

s×h

) and Nˆcn s×hˆ

:= Normcn( In

For conciseness, we use the following notation: Ncn s×h

:= [Ncn s×h

0

].

s×(hˆ−h)

s×h

=⇒

′

ˆI

= ˆIn

+ MHAˆ n(NˆMHAn

)

n s×hˆ

s×hˆ

s×hˆ

= ˆIn

+ ···Attention(NˆMHAn

,NˆMHAn

,NˆMHAn

×Wˆ n,eQ

×Wˆ n,eK

s×hˆ

s×hˆ

s×hˆ

s×hˆ

hˆ×k

hˆ×k







Wn,eQ h×v

=ˆIn

,NˆMHAn

···Attention([NMHAn

]×

+

0

 

 

 

MWQn,e (hˆ−h)×v

s×(hˆ−h)

s×hˆ

s×h

s×hˆ

= ˆIn

×Wn,eQ

,NMHAn

×Wn,eK

,NMHAn

+ ···Attention(NMHAn

s×hˆ

s×h

s×h

s×h

h×k

h×k

= ˆIn

s×hˆ

··· | ∀e∈[1,E] × WnO

+ ··· He s×v

(E·v)×h

= ˆIn

s×hˆ

+ MHAn(NMHAn

) 0

s×(hˆ−h)

s×h

### 0

(E·v)×(hˆ−h)

×Wˆ n,eV

)··· | ∀e∈[1,E] × Wˆ nO

(E·v)×hˆ

hˆ×v

,NˆMHAn

×Wˆ n,eK

×Wˆ n,eV

)··· | ∀e∈[1,E]

s×hˆ

hˆ×k

hˆ×v

)··· | ∀e∈[1,E] × Wˆ nO

×Wn,eV

(E·v)×hˆ

h×v



× Wˆ nO

 

(E·v)×hˆ

= In

s×h

0

s×(hˆ−h)

+ MHAn(NMHAn

) 0

s×(hˆ−h)

s×h

= In

s×h

+ MHAn(NMHAn

) 0

s×(hˆ−h)

s×h

′

= I

n s×h

0

s×(hˆ−h)

=⇒

Following the demonstration provided for Normˆ MHAn (·):

Normˆ MLPn ( ˆIn

′

) = NormMLPn ( ˆI

n s×h

s×h

) 0

s×(hˆ−h)

(49)

(50)

#### =⇒

:= Normˆ MLPn ( ˆIn

NˆMLPn s×hˆ

) (51)

s×h

= TransformerLayerˆ n( ˆIn

ˆIn+1 s×hˆ

)

s×hˆ

′

= ˆI

+ MLPˆ n(NˆMLPn

)

n s×hˆ

s×hˆ

′

= ˆI

+ MLPˆ n(NˆMLPn

)

n s×hˆ

s×hˆ

′

= ˆI

+ ReLU(NˆMLPn

× Wˆ nl1 hˆ×p

) × Wˆ nl2 p×hˆ

+ Bˆln2 s×hˆ

+ Bln1 s×p

n s×hˆ

s×hˆ

  

### Wnl1

h×p MWl1 (hˆ−h)×p

′

= ˆI

+ ReLU([NMLPn

] ×

0

n s×hˆ

s×(hˆ−h)

s×h

′

= ˆI

) × Wˆ nl2 p×hˆ

+ Bˆln2 s×hˆ

+ ReLU(NMLPn

× Wnl1 h×p

+ Bln1 s×p

n s×hˆ

s×h

′

= ˆI

+ ReLU(NMLPn

× Wnl1 h×p

+ Bln1 s×p

) × Wnl2 p×h

n s×hˆ

s×h

′

= ˆI

+ ReLU(NMLPn

× Wnl1 h×p

+ Bln1 s×p

) × Wnl2 p×h

n s×hˆ

s×h

′

= ˆI

+ ReLU(NMLPn

× Wnl1 h×p

+ Bln1 s×p

) × Wnl2 p×h

+ Bln2 s×h

n s×hˆ

s×h

′

= ˆI

n s×hˆ

′

= I

n s×h

+ MLPn(NMLPn

) 0

s×(hˆ−h)

s×h

+ MLPn(NMLPn

) 0

s×(hˆ−h)

s×h

= TransformerLayer ˆ n( In

) 0

s×(hˆ−h)

s×h

   + Bln1

) × Wˆ nl2 p×hˆ

+ Bˆln2 s×hˆ

s×p

### 0

p×(hˆ−h)

+ Bln2 s×h

### 0

s×(hˆ−h)

### 0

s×(hˆ−h)

+ Bln2 s×h

### 0

s×(hˆ−h)

### 0

s×(hˆ−h)

= In+1 s×h

0

s×(hˆ−h)

Having demonstrated that, after applying the hidden dimension expansion:

(52)

ˆIn+1 s×hˆ

= In+1 s×h

0

s×(hˆ−h)

∀n∈[1,N + 1] (53)

The output equivalence can be proven as follows: TransformerArchitecture(ˆ ˆI

) = TransformerLayerˆ ◦N( ˆI

### + Pˆ

) × Wˆ out

hˆ×o

s×hˆ

s×hˆ

s×hˆ

  

   = IN+1

Wout h×o

= ˆIN+1

× Wˆ out

× Wout

×

= IN+1

0

MWout (hˆ−h)×o

h×o

hˆ×o

s×(hˆ−h)

s×h

s×h

s×hˆ

) (54)

= TransformerArchitecture( I

s×h

| |
|---|

- A.6 LAYER ADDITION Proof.

··· HE s×v

MHAn(Xn s×h

) = H1 s×v

× 0

= 0

s×h

(E·v)×h

× Wnl1 h×p

MLPn(Xn s×h

) = ReLU(Xn s×h

+ Bln1 s×p

) × 0

+ 0

= 0

p×h

s×h

s×h

′

+ MHAn(NormMHAn ( In

I

= In

)) = In

+ 0n

= In

n s×h

s×h

s×h

s×h

s×h

s×h

TransformerLayern( In

) = In

s×h

+ MLPn(NormMLPn ( In

)) = In

s×h

s×h

s×h

+ 0n

= In

s×h

s×h

(55)

(56)

(57)

(58)

| |
|---|

Note that the function preserving property holds even if normalization is applied after the MLP and MHA components as Norm(·) outputs zeros for zeros input.

