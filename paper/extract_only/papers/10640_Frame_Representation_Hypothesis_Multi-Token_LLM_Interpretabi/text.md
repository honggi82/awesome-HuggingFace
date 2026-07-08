## Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation

Pedro H. V. Valois∗, Lincon S. Souza†, Erica K. Shimomoto†, Kazuhiro Fukui∗ ∗University of Tsukuba, †National Institute of Advanced Industrial Science and Technology (AIST)

# arXiv:2412.07334v2[cs.CL]12Dec2024

### Abstract

Interpretability is a key challenge in fostering trust for Large Language Models (LLMs), which stems from the complexity of extracting reasoning from model’s parameters. We present the Frame Representation Hypothesis, a theoretically robust framework grounded in the Linear Representation Hypothesis (LRH) to interpret and control LLMs by modeling multi-token words. Prior research explored LRH to connect LLM representations with linguistic concepts, but was limited to single token analysis. As most words are composed of several tokens, we extend LRH to multi-token words, thereby enabling usage on any textual data with thousands of concepts. To this end, we propose words can be interpreted as frames, ordered sequences of vectors that better capture token-word relationships. Then, concepts can be represented as the average of word frames sharing a common concept. We showcase these tools through Top-k ConceptGuided Decoding, which can intuitively steer text generation using concepts of choice. We verify said ideas on Llama 3.1, Gemma 2, and Phi 3 families, demonstrating gender and language biases, exposing harmful content, but also potential to remediate them, leading to safer and more transparent LLMs. Code is available at https://github.com/phvv-me/ frame-representation-hypothesis.git

### 1 Introduction

Interpretability in deep learning aims to elucidate how neural networks derive predictions. As models grow complex, understanding internal mechanisms gets challenging. By identifying factors contributing to the output, we can foster trust, safety, fairness and improve capabilities (Hooker et al., 2019).

The goal of this study is to enhance the interpretability and control of LLMs via the encoding of human-comprehensible linguistic concepts. LLMs

represent text through tokens, which can be a word, part of a word, or even a character, as per models’ design. In contrast, humans better understand text through concepts, cognitive symbols that depict reality, often grouping related objects, events, or further abstractions based on shared characteristics. Our purpose is to provide tools to represent concepts within LLMs, allowing output explanations that are suited for our mental models.

Prominent works that offer such tools are based on the Linear Representation Hypothesis (LRH): it suggests linear operations on token vectors can explain model behavior, with concepts represented as vectors in LLM feature space (Templeton et al., 2024). For instance, we can identify the concept female as the average of token vectors like f (woman) or f (queen). Thus, token vectors encode more than just lexical data: they also represent linguistic concepts (Mikolov et al., 2013).

Nevertheless, LRH’s concepts are 1-dimensional, constraining them to single-token words, which are a minor fraction of any given language (Bau et al., 2020). As exemplified in Figure 1, a concept like vegetarian is exclusively linked to multi-token words, such as meatless or herbivore, meaning multiple vectors are required to represent it. Since most words are constituted of several tokens, 1dimensional structures prevent LRH application in most interpretability tasks effectively.

To address this shortcoming of LRH, we propose a new framework for LLM interpretability, based on a key empirical observation about the nature of LLMs: our experiments show that over 99% of words among several languages are composed of linearly independent token vectors. This allows us to address multi-token words by proposing the Frame Representation Hypothesis (FRH), which assumes words are ordered sequences of independent vectors – mathematically identified as frames.

Starting from this postulate, we develop a mathematical framework to represent words and concepts

𝑢(𝑣𝑒𝑔) 𝑢(etarian) 𝑢(ism)

Frame

|Token Space ℝ𝑑|
|---|

Embedding

Word (Frame)

vegetarianism

meatless

herbivore

carnivore

animal

Concept

meat

vegetarian

|meatball<br><br>Semantic Frame Space ڂ𝑖=1𝑘 𝑺𝒕(𝑖, 𝑑)<br><br>Frame Representation|
|---|

- Figure 1: Hypothesis Overview: Tokens are vectors, which combine into words as multi-dimensional frames. In turn, Concept Frames are centroids of word sets.

as frames; we define a Semantic Frame Space, and equip it with a correlation between frames that preserves the token whitening mechanism introduced by LRH (Park et al., 2023), allowing us to measure frames semantic relationship. Following Figure 1, the frame representation lets us identify words, such as meatless, meat and meatball as distinct geometrical objects, although they have tokens in common. Then, we can compute Concept Frames like vegetarian or carnivore as the centroid for a set of words sharing that concept.

Furthermore, we introduce Top-k ConceptGuided Decoding (Figure 2), which controls text generation by selecting the tokens which maximize a chosen concept. For example, the concept vegetarian would guide input I like to I like fruits if the top-3 options were beef, football or fruits. This algorithm aligns model outputs with desired concepts, a practical prototype for FRH that allows meaningful LLM understanding.

To that end, we leverage the Open Multilingual WordNet (OMW) (Bond and Foster, 2013) as a source of synonyms to build concepts. We use over 50M words among multiple languages to build over 100,000 Concept Frames, enabling rich model understanding in a diverse yet inexpensive manner.

In short, FRH formally extends the LRH to multitoken words. We show its validity both from the theoretical and empirical points of view.

Our primary contributions are as follows:

1. Frame Representation Hypothesis as an extension of LRH to multi-token words by defining them as Frames, thereby addressing the limitations of single-token representations.

- 2. Proposal of Concept Frames to represent linguistic concepts from a set of Word Frames.
- 3. Development of Top-k Concept-Guided Decoding, a proof-of-concept application to steer text generation using chosen concepts and expose model biases or potential vulnerabilities.

### 2 Related Work

We briefly review LLM interpretability, controllable text generation and Frame usage in the field. Language Models Interpretability The widespread adoption of LLMs brought attention to the need of understanding their inner-workings, risks, and limitations. Several studies identified a common property to these models that became known as the Linear Representation Hypothesis, encoding model knowledge as vectors (Mikolov et al., 2013), and enabling model explanation and editing (Wang et al., 2023). Also, the Superposition Hypothesis (SH) assumes specialized information is superimposed in LLM feature spaces. These ideas underpin Sparse Autoencoders, which learn dictionaries of interpretable concepts to decipher model behavior (Elhage et al., 2022), whereas our proposal uses WordNet (Miller, 1995) to map learned representations to concepts. For a comprehensive survey, see Ferrando et al. (2024). Controllable Text Generation LLMs can use various decoding strategies for inference. Beam search (Jurafsky and Martin, 2000; Graves, 2012) can improve quality but risk cycles. Top-k (Fan et al., 2018) and nucleus sampling (Holtzman et al., 2019) introduce randomness for diversity. Other techniques offer specific controls, such as poetry generation (Ghazvininejad et al., 2017), attribute maximization (Krause et al., 2020), style optimization (Khalifa et al., 2020), reasoning paths (Wang

- et al., 2022), diverse decisions (Yao et al., 2023), and self-evaluation (Kadavath et al., 2022; Xie
- et al., 2023). LRH enables steering through linear interventions for knowledge edition (Belrose et al., 2023; Singh et al., 2024a) or harmfulness (Bai et al., 2022), while our proposal guides text generation by maximizing a certain concept.

Subspaces and Frames in Machine Learning Subspaces have been extensively used for dimensionality reduction, feature extraction (Fukui and Maki, 2015; Fukui et al., 2023), classification (Watanabe, 1967), image interpretability (Val-

|Top-𝑘 Feature Vectors|
|---|

LLM

Hidden Layers

Input feature vectors

input

Output feature vector

}

oken iT

I like

Feature Frame 𝑯𝑖(𝑥)

𝒉𝑡−3, 𝒉𝑡−2 ,𝒉𝑡−1 ,𝒉𝑡

|Concept Frame 𝑺<br><br>|
|---|

I like fruits vegetarian

|𝑎𝑟𝑔𝑚𝑎𝑥𝑖 𝜌(𝑯𝑖(𝑥),𝑺)|
|---|

I like beef

I like football

Semantic Frame Space

- Figure 2: Top-k Concept-Guided Decoding Overview: Top-k sentence candidates are derived from the model logits, and we chose the one which maximizes the correlation with the target Concept Frames. The process is repeated in a loop until the desired number of tokens is reached.

ois et al., 2023), and modelling text sentences (Cancedda, 2024; Shimomoto et al., 2021). Their invariance to selection of basis is advantageous for representing clusters, but limiting for ordered structures. Frames are sequences of vectors that model well redundant and oriented data, being applied in error correction (Kovaˇcevi´c and Chebira, 2008), signal decomposition (Casazza et al., 2013) and optimization problems (Mankovich and Birdal, 2023; Chaudhry et al., 2020). To the best of our knowledge, frames have never been applied in NLP to model words and concepts as proposed here.

- 3 Preliminary

In this section, we introduce the necessary background to our proposal. Throughout this work, we denote vectors as bold lowercase letters, e.g., v; matrices as bold uppercase letters, e.g., M; monospace lowercase letters for tokens, e.g., x; spaces with calligraphic letters, e.g., U; words with sans serif uppercase uppercase letters, e.g., W, and concepts with monospace uppercase letters, e.g., C.

#### 3.1 Frames

A k-frame is a sequence of k linearly independent vectors in Rd, represented by F ∈ Rd×k,rank(F) = k. The set of all k-frames in Rd constitutes the non-compact Stiefel manifold St(k,d). Manifolds are structures in which distance, geodesics and more may be defined, so we can compute geometrically meaningful relationships between distinct frames (Edelman et al.,

1998). The set of all frames up to rank q forms the q-complete Stiefel manifold CSt(q,d) =

q

i=0 St(i,d) – a disjoint union of Stiefel manifolds – where the null frame St(0,d) ≡ ∅ is defined as our space’s origin. Intuitively, CSt(q,d) is a stratified structure, so its base is the null frame.

#### 3.2 Rays

A ray R is a directed half-infinite line, also known as 1-dimensional convex cone, half-line, or axis (Boyd and Vandenberghe, 2004), defined by a vector v ∈ Rd and its scalar multiples R(v) = {αv | α ≥ 0}. A ray is represented by a normalized vector v′ = v/∥v∥, which is also a point in St(1,d). Rays differ from subspaces by their orientation, so a single dimensional subspace contains two rays, and their correlation is measured as the cosine of the angle θ between their normalized vectors (Mathematics, 2016)

ρ R(v),R(u) = ⟨v,u⟩ ∥v∥∥u∥

= v′⊤u′ = cosθ.

(1)

Moreover, notice the correlation shown in Equation (1) is connected to the inner product – projection of one vector onto another. In this work, we use the term “projection” when calculating correlation with unnormalized vectors.

#### 3.3 Large Language Models

LLM models process text by converting it into a sequence of tokens, embedding them into its own vector space and processing this sequence of vectors through its hidden layers to a final vector representation, which is unembedded into the most likely token to continue the input sentence. A simple version of such pipeline is illustrated in Figure 2.

A token is a single element of a textual sequence, represented by a number x ∈ V in a predefined vocabulary V ⊂ Z+. In that sense, the model’s tokenizer converts text input x into token t-tuple (x1,x2,...,xt) ∈ Vt. The LLM then starts in the embedding layer, which maps each token number a ∈ V to an unique embedding vector e(a) ∈ E ∼= Rd, each of which is a column of the embedding matrix WE ∈ Rd×|V|. Therefore, the output of this layer is the t-tuple of embedding vectors e(x) = e(x1),e(x2),...,e(xt) . Next, e(x) is processed by the DNN hidden transformer layers into the feature vector h(x) = h e(x1),e(x2),...,e(xt) ∈ H ∼= Rd.

Then, the LLM converts h(x) into a token number. The unembedding vector of token b ∈ V is u(b) ∈ U ∼= Rd, a row of the unembedding matrix WU ∈ R|V|×d, which also identifies each token to a unique vector in high-dimensional space U.

Finally, the probability of a token y ∈ V being next in a text sentence x is determined with softmax

p y|x ∝ exp u(y)⊤ h(x) . (2)

In practice, the space dimension d can range from 1024 to 16384, while the vocabulary V usually contains from 50,000 to 300,000 tokens.

#### 3.4 Linear Representation Hypothesis

We now concentrate the discussion into the geometry of H and U and their relationships. With that in mind, Park et al. (2023) defined the ray representation of a concept C as the ray R s′C ⊂ U of vector s′C ∈ U. The correlation ρ of concepts A,B serves as a linear probe for model understanding

ρ(A,B) = ⟨s′A,s′B⟩ = s′⊤A Ms′B, (3)

where M = Cov−1 (WU) is a whitening matrix that defines the LRH inner product, placing unrelated concepts as orthogonal to each other.

Hereafter, concepts connect through linear operations and are computed as the normalized mean of counterfactual pairs difference vectors

nC

u′C =

ui (C = 1) − ui (C = 0) , (4)

i

u′C u′C

s′C =

, (5)

where ui (C = 1),ui (C = 0) is a counterfactual token pair, so C = 1 indicates one concept direction while C = 0 its opposite, e.g., concept English ⇒ Spanish(C = 0) is computed using difference vectors like u(good) − u(bueno), u(bad) − u(malo), while Spanish ⇒ English(C = 1) is the opposite vector.

#### 3.5 WordNet

The Open Multilingual WordNet (OMW) is

- a collaborative project that intersects cognitive psychology, linguistics and computer science to create an interconnected network of lexical databases (Bond and Foster, 2013; Fellbaum, 1998; Harabagiu et al., 1999). At its core are synsets and

[Figure 1]

Figure 3: Uniform Manifold Approximation and Projection (UMAP) (McInnes and Healy, 2018) of the 10k most frequent single-token English words for Gemma 2. While some points are clearly separated, others overlap due to the Superposition Hypothesis (SH). For example, ad is a token in the unrelated words advertisement, admit, adventure, etc., while restaurant is a single token and it is not found in other words.

lemmas. A synset, short for “synonym set”, is a group of words or phrases that may share the same meaning. For example, {car, automobile, auto} forms a synset, which can be uniquely identified as car.n.01, i.e., the 1st dictionary meaning of the word car as a noun (Miller et al., 1990).

A lemma, on the other hand, is a canonical form, e.g., run, runs, ran, and running are all represented by the lemma run (Fellbaum, 2010). In OMW, synsets from different languages are linked to their equivalent English ones, allowing for cross-lingual connections. Therefore, the English synset car.n.01 will also include Spanish lemmas, such as coche or automovil´ (Bond and Paik, 2012). Each lemma can belong to multiple synsets, reflecting its different meanings, making OMW a powerful NLP tool (Wagner, 2010).

### 4 Frame Representation Hypothesis

In this section, we introduce our theoretical framework. Proofs are provided in Appendix A.

#### 4.1 Linear Decomposition of Tokens

LRH posits concepts are linearly encoded within LLMs feature spaces. Moreover, the Superposition Hypothesis (SH) suggests models encode information in a superposition of concepts because the number of possible concepts significantly exceeds the space dimensionality (Elhage et al., 2022), a phenomenon visible in Figure 3. This is mathematically expressed as a linear combination of vectors, formalized at Postulate 4.1.

Postulate 4.1. Let u(y) be the unembedding representation of token y ∈ V, then it is a linear combination of concept vectors s

u(y) − u0 =

s

i

aisi, (6)

where ai ∈ R, s is the number of all concepts known by the model, and u0 is a meaningless vector – an offset element from the fact not all tokens might hold meaning, e.g., [PAD] or [EOS], implying we need to remove the meaningless part of each token vector. Heuristically, u0 should be the unembedding vector average, so that E u(y) − u0 = 0. Also, {si}ni=1 is not a basis: words can be grouped in several ways, e.g., antonyms or synonyms, making concepts interdependent.

Therefore, we can extract a concept of choice by

averaging tokens sharing that concept. Let {yj}nj=1 be a set of tokens sharing a common concept s, we estimate the concept as the token average1

s ∝

∼

n

u yj − u0, (7)

j=1

- 4.1.1 Combined Concepts We connect Concept Estimation (7) to the ray

R s′C of a concept C by separating Equation (4) into two sums, each its own concept. Therefore, a

concept C has representation R s′C , where s′C is a normalized counterfactual concept pair difference,

sC=1 − sC=0 ∥sC=1 − sC=0∥

s′C =

, (8)

indicating some concepts are formed by other concepts. For example, a set of tokens sharing the meaning of female builds sfemale, while another sharing the concept of male builds smale, forming

s′male⇒female ∝ sfemale − smale, (9)

which leads us to understand some concepts as building blocks for Combined Concepts.

#### 4.2 Generalizing from Tokens to Words

The previous discussion can leverage WordNet to determine concepts. WordNet’s structure overlaps with LLM representations (Moskvoretskii et al., 2024a,b; Park et al., 2024), and OMW

1hereafter consider all tokens to be already debiased

[Figure 2]

Figure 4: Histogram of lemma token count among all OMW lemmas. The dashed vertical bar indicates the 75% percentile for each model family.

Synset Lemma ad.n.01 ad

myth.n.01 mit admit.v.01 admit

half.n.02 mitad

Table 1: An example of a token pair – ad and mit – being used to form different words, each portraying different concepts. The Gemma 2 model family (Gemma, 2024) tokenizes admit, in the OMW synset of admit.v.01, into ad in ad.n.01, and the Polish word mit in myth.n.01. Concurrently, in opposite order they form the Spanish word mitad, present in half.n.02.

synsets are sets of multilingual lemmas sharing a meaning, making it well suited for Equation (7). Nevertheless, Section 4.1 only deals with singletoken words, which accounts for less than 1% of all OMW lemmas on most LLMs, significantly limiting estimated concepts quality (c.f. Figure 4).

In order to make LRH theory more relevant for LLM interpretability, we must generalize Section 4.1 to words made of multiple tokens. At first glance, the insight is simple: tokens do not build concepts – tokens build words – and words build concepts. Nonetheless, words are sequences of tokens with a well defined order, implying we cannot simply average them to a single vector or their meaning might be lost, as exemplified at Table 1.

#### 4.2.1 Words as Frames

In practice, we define a word W as an OMW lemma (c.f. Section 3.5). It is split into t independent tokens (w1,w2,...,wt), and represented in U as a sequence of unembedding vectors, i.e., the matrix

##### W = u(w1) u(w2) ... u(wt) . (10)

First, consider the following principle about the nature of word matrices. Let two token vectors a,b ∈ U. If these vectors were collinear, meaning

- b = αa for some α ∈ R, then Equation (2) would consistently assign higher probabilities to one token over the other regardless of input. This would effectively make some tokens redundant, as they would never be the most probable choice in any context. Such a scenario contradicts a fundamental design of LLMs, where each token in V must have some context in which it is the optimal choice, i.e., all tokens must be meaningful and usable. Thus, we conclude no two token vectors are collinear.

In that sense, word matrices as in Equation (10) are constrained so that no token vector u(wi) can be expressed as a scalar multiple αu wj of another token vector, for any α ∈ R. This noncollinearity constraint defines a locally Euclidean open subset of Rd×t, thereby forming a manifold. The softmax operation in DNN training ensures the space of all words acquires a manifold structure.

While non-collinearity is a necessary condition, it does not enforce W as full-rank – the t token vectors may exist in a subspace of dimension less than

- t. However, we assume rank deficiency may compromise expressiveness and computational stability, suggesting the need for additional constraints.

From an NLP perspective, unique word representations are essential. If a word was not a linearly independent matrix, we could eliminate dependent tokens until made full-rank, yielding an alternative representation of the same word in U. However, this sacrifices word uniqueness, which is undesirable for consistent language modeling.

To address these concerns, we propose modeling words as frames, i.e., we assume W ∈ St(t,d). Then, all word matrices are supposed full-rank. Our empirical investigation (c.f. Section 5.1) supports this framework, revealing that over 99% of words in OMW exhibit linear independence among their token vectors. This assumption is facilitated by the high dimensionality of U, easily representing words as full-rank matrices.

- 4.2.2 Frame Correlation We name the set of all words the Semantic Frame Space, or equivalently, CSt(k,d), k the max number of tokens in any word. Let A = a1 ... ak1 ∈ St(k1,d),B =

b1 ... bk2 ∈ St(k2,d) be frames of CSt(k,d), we employ the asymmetric Procrustes distance (Ye and Lim, 2016; Mandolesi, 2022) as the space metric,

d∗P (A,B) = k1 + k2 − 2

min k1,k2

j

aj Mbj,

(11) where M comes from Equation (3).

Hereafter, we can propose frame correlation by applying the law of cosines to generalize Equation (1) only in terms of distance functions:

ρ(A,B) = ∥A∥2P + ∥B∥2P − d∗P (A,B)2 2∥A∥P∥B∥P

(12)

=

min k1,k2 j aj Mbj

√k1k2

, (13)

where ∥A∥P = d∗P (A,∅) = √k1, ∅ is the null frame (origin) of CSt(k,d), so rank(∅) = 0.

Such correlation can measure relationships as similar (positive), unrelated (null), or opposite (negative). For instance, yeah and yes are similar words and should have correlation close to 1, while yes and bubble are orthogonal, but antonyms such as yes and no would be negatively correlated.

- 4.2.3 Concept Frame We estimate concepts as the Fr´echet mean of a word set – the point minimizing the distance to each word – effectively capturing the concept they collectively represent (Marrinan et al., 2014).

Let {Wi}ni=1 be a set of words, Wi =

u(wi1) u(wi2) ... u wiki ∈ St(ki,d), and let S = s1 s2 ... sk ∈ St(k,d),k = maxki be the Concept Frame, it is determined as

n

S = arg min

S∈St(k,d)

i=1

= arg max

sj∈St(1,d)

d∗P2 (Wi,S) (14)

ki

n

#### u wij ⊤ Msj. (15)

i=1

j=1

We can extend the sum at Equation (15) from ki to k by noticing its equivalence to having

- u wij = 0 for all ki < j ≤ k. Let’s define Wi′ =

u(wi1) u(wi2) ... u wiki 0 0 ... 0 as the right-padded Wi with k − ki zeros. Thus,

k

n

#### Wij′ ⊤ Msj (16)

S = arg max

sj∈St(1,d)

i=1

j=1

 

 Msj (17)

k

n

Wij′ ⊤

= arg max

sj∈St(1,d)

j=1

i=1

#### tr W ¯ ′⊤ MS , (18)

= arg max

S∈St(k,d)

where W¯ ′ = ni=1 Wi′ is the padded word sum.

Finally, Equation (18) is the Procrustes problem, which Sch¨onemann (1966) has solved with

#### S = UV⊤, (19)

and W¯ ′⊤ M = UΣV⊤ is the SVD decomposition of the padded word sum. Hence, under the Procrustes distance the Concept Frame is the solution of a Procrustes problem.

- 4.2.4 Combined Concept Frames In Section 4.1.1, we defined Combined Concepts

- as concept vector differences, which we extend to FRH by placing them in the Stiefel manifold. In other words, given a pair of Concept Frames A,B ∈ St(k,d), we can build the Combined Concept Frame D = D(B,A) ∈ St(k,d) by enforcing it to be the frame closest to B − A:

tr (B − A)⊤ M D , (20)

D = arg max

D∈St(k,d)

Thereby, D(B,A) = UDVD⊤, (B − A)⊤ M = UDΣDVD⊤ the SVD decomposition.

#### 4.3 Concept Probing

The framework established for U can be extended to the feature space H by reinterpreting Equation (2) as logitp y|x = ρ u(y),h(x) . Thus, the correlation between u(y) and h(x) can be understood as a linear probe from space U to H.

Consequently, there is a correspondence between frames in U and H. Let a Feature Frame H be the last k feature vectors of the input sequence

H(x) = ht−k+1 ht−k+2 ... ht ∈ St(k,d),

we probe x for Concept Frame S ∈ St(k,d) using the correlation defined at Equation (12),

logitp S|x = ρ S,H(x) . (21)

#### 4.4 Top-k Concept-Guided Decoding

We can leverage concept probing as a mechanism for Concept-Guided Text Generation (Figure 2). This approach can be implemented with samplebased decoding methods, such as Top-k sampling, first generating a set of k potential tokens from which the next token is randomly selected. We propose to alter such process wherein the next token xt+1 of input sequence x = (x1,x2,...,xt) is the one which maximizes its respective Feature Frame correlation onto a target Concept Frame S,

ρ S,Hi (x) . (22)

xt+1 = arg max

i∈{1,2,...,k}

This methodology can align model output with a desired concept and serves as a practical prototype for FRH, showing how to direct text generation and understand model behavior meaningfully.

5 Experiments

In this section, we validate FRH for words and concepts, showing guided generation of sentences. We use Llama 3.1 (Llama, 2024), Gemma 2 (Gemma, 2024), and Phi 3 (Microsoft, 2024) LLM families and OMW only with supported languages. Further discussion is available in the Appendices.

#### 5.1 Frame Representation Hypothesis

FRH posits LLMs encode words as frames. We can empirically evaluate this hypothesis by analyzing if words are made of linearly independent vectors, which we can measure by computing its rank. In Figure 5, we see near-maximum matrix ranks for lemmas comprising up to 3-4 tokens, which is the token count that represents words. In OMW, lemmas with token counts of 5 and beyond mostly represent compound words and expressions, implying the frame representation fits 99.8% words. Notably, Phi 3 shows a rapid rank decrease beyond token count of 5, likely due to its high proportion of lemmas with large token count, making non fullrank lemmas more common (c.f. Figure 4).

Furthermore, given we propose using OMW synsets to build Concept Frames, we must verify if these synsets fit the model representation or not. To that end, we can compute the projection (unnormalized correlation) of Word Frames onto their corresponding Concept Frames for all OMW synsets and lemmas. Figure 6 reveals that random frames are consistently unrelated to concept frames across

[Figure 3]

Figure 5: Relative Rank as a function of token count for all OMW lemmas and model families. Over 99% of words are full-rank. Phi 3 has lower overall rank for longer lemmas than other models.

models, while words exhibit positive projections onto their associated concepts.

These findings support FRH consistency with models’ internal representations and suggest LLMs inherently correlate with the OMW linguistic graph. In the following experiments, we use lemmas up to 4 tokens to ensure our theory is applied only for full-rank matrices.

#### 5.2 Guided Generation

Given the FRH evidences, we explore its application in text generation with Top-k Concept-Guided Decoding, exposing biases and vulnerabilities.

- 5.2.1 Qualitative Analysis We first compare model outputs on a few inputs and concepts likely sensitive to biases. The example

- at Figure 7 demonstrates the impact of conceptguided text generation on the characterization of men by Llama 3.1 8B Instruct. With no guidance, the model focuses on family roles. When guided by the Concept Frame woman.n.01 − male.n.01, this tendency is seemingly amplified. However, a more significant shift in narrative occurs when the model is guided by the opposite concept of male.n.01 − woman.n.01, prompting it to emphasize a perceived importance as family providers.

At Figure 8, the unguided model’s characterization of women primarily enumerates family roles, which may be interpreted as a balanced

[Figure 4]

- Figure 6: Distribution of word frame projection lengths. Random frames have near-zero projection with any Concept Frame, while words show positive projections onto associated Concept Frames.

What men can be?

Men can be fathers, sons, brothers, and husbands. noguidance

1. A husband. 2. A father. 3. A son. 4. A friend. 5. A boyfriend or partner... woman.n.01−male.n.01

Men as fathers and family providers, as well as caregivers, are essential for family well-being...

male.n.01 − woman.n.01

- Figure 7: Concept-guided generation examples when the model is prompted to describe men.

output when juxtaposed with the default answer for men at Figure 7. Despite that, when guided by woman.n.01 − male.n.01, the model’s output noticeably emphasizes biological traits. Conversely, male.n.01 − woman.n.01 highlights leadership roles and esteemed social positions.

The stark contrast in each example suggests that, when guided by a Combined Concept D(B,A), the

What women can be?

Women can be mothers, wives, sisters, daughters. noguidance

A human being of the female sex, typically having a vagina, breasts, and a womb. woman.n.01−male.n.01

Women have been leaders in various capacities throughout human civilization. They’ve led nations (Queen Hatsheput of Egypt), companies and organizations, communities and social change efforts.

male.n.01 − woman.n.01

Figure 8: Concept-guided generation examples when the model is prompted to describe women.

model attempts to maximize attributes it associates with the first concept B while minimizing the second concept A. They illustrate how to influence text generation, exposing biases and stereotypes within the model’s learned representations.

Notably, most generations kept a high level of readability, but using elevated values of k can lead to incoherent text, a known issue of top-k sampling (Holtzman et al., 2019).

Besides, this process can expose vulnerabilities, including the capacity to generate harmful content, exemplified in Appendix B.1. The authors emphatically discourage this tool usage for malicious purposes yet acknowledge its potential for misuse, but more studies are warranted to comprehend their extent and implications.

#### 5.2.2 Quantitative Analysis

A comprehensive understanding begets a quantitative study. We used a multilingual instruction dataset to ensure a minimum of 1000 sentences for each model supported language. The concept of choice was woman.n.01 − man.n.01 to stay consistent with the previous section. Resource constraints limited our investigation to a single concept, though we argue the results are indicative of the model’s behavior across similar conceptual domains.

[Figure 5]

Generation starts

Input

Figure 9: Concept probing evolution for 3 levels of Top-k Concept-Guided Decoding with Llama 3.1 70B AWQ (Lin et al., 2023). The guidance with woman.n.01 − man.n.01 is able to counter the LLM tendency to maximize man.n.01.

Initially, we focused on the evolution of generated sentences across distinct values of k. As visible in Figure 9, all sentences start with minimal correlation to the chosen concept, evidenced by near-zero projection length. Notably, the unguided output naturally minimizes the projection with our chosen concept, indicating it tends toward the opposite direction of man.n.01 − woman.n.01. However, the algorithm demonstrated capacity to steer the output toward the desired concept with increasing effectiveness as k increased, showing k can regulate guidance strength. We highlight this result indicates biases in standard generation, and while guidance does not completely modify this scenario, it is remediated to a certain extent.

Next, we examine concept-guided generation across Llama 3.1 supported languages. We find most languages exhibit comparable patterns, with Hindi and Thai serving as notable exceptions (Figure 10). These demonstrate significantly higher susceptibility to guidance and are the only noneuropean ones, suggesting the model treats said languages differently (Llama, 2024). Further investigation is shown in Appendix C.2.

Finally, in Figure 11 we use the concept relative projection – difference of guided and unguided projection to the concept – to measure guidance susceptibility among several model families and

[Figure 6]

Figure 10: Concept probing evolution during model generation for the 8 languages supported by Llama 3.1 70B using Top-k Concept-Guided Decoding with k = 3. Hindi and Thai are more susceptible to the technique than other languages.

[Figure 7]

Figure 11: Concept relative projection for several models and parameter counts.

various parameter counts. Llama 3.1 models seem equally susceptible to guidance among base and instruct models for all parameter counts; Gemma 2 shows more susceptibility to guidance when the parameter count increases, but there is a sensible reduction from base to instruct variations; On the other hand, Phi shows a great reduction in guidance effect with parameter count, possibly an effect of a less linear feature space as previously commented. Most curiously, guidance susceptibility is almost equivalent on all models for the lower parameter count, which could indicate a common convergence of representations.

### 6 Conclusions

This study proposes the Frame Representation Hypothesis, an extension of the Linear Representation Hypothesis with Lie Group elements. FRH posits LLMs encode words as frames, with model input and output connected as Concept Frames in Stiefel Manifolds. FRH provides a structured framework for LLM interpretability and control via concept probing and concept-guided decoding, showing that even state-of-the-art LLMs exhibit gender and language biases or harmful vulnerabilities.

This work is an initial exploration, and further research is made necessary to understand its extents. In particular, we have yet to explore 2nd order

Combined Concepts and higher, which could reveal even richer concept relationships, uncovering LLMs own ontology. In that sense, our concepts were limited to WordNet selection of meanings, and while it enabled efficient concept extraction without additional training, future work should integrate FRH with Dictionary Learning techniques to automatically extract concepts from the model weights. Also, Top-k Concept-Guided Decoding served as a FRH proof-of-concept, but is limited by the same constraints as Top-k sampling, so we encourage more advanced and custom variations.

In conclusion, FRH represents a promising avenue for LLM interpretability, and could lead to novel developments in safe, trustworthy and reliable AI systems.

### References

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2019. On the cross-lingual transferability of monolingual representations. In Annual Meeting of the Association for Computational Linguistics.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova Dassarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom

Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. CoRR, abs/2204.05862.

David Bau, Jun-Yan Zhu, Hendrik Strobelt, Agata` Lapedriza, Bolei Zhou, and Antonio Torralba. 2020. Understanding the role of individual units in a deep neural network. Proceedings of the National Academy of Sciences, 117:30071 – 30078.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. 2023. Leace: Perfect linear concept erasure in closed form. ArXiv, abs/2306.03819.

Francis Bond and Ryan Foster. 2013. Linking and extending an open multilingual wordnet. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1352–1362.

Francis Bond and Kyonghee Paik. 2012. A survey of wordnets and their licenses. In proceedings of the 6th global WordNet conference (GWC 2012), pages 64–71. Matsue.

Stephen Boyd and Lieven Vandenberghe. 2004. Convex optimization. Cambridge university press.

Nicola Cancedda. 2024. Spectral filters, dark signals, and attention sinks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4792–4808, Bangkok, Thailand. Association for Computational Linguistics.

- Peter G Casazza, Gitta Kutyniok, and Friedrich Philipp. 2013. Introduction to finite frame theory. Finite frames: theory and applications, pages 1–53.

Arslan Chaudhry, Naeemullah Khan, Puneet Kumar Dokania, and Philip H. S. Torr. 2020. Continual learning in low-rank orthogonal subspaces. ArXiv, abs/2010.11635.

Danilo Croce, Alexandra Zelenanska, and Roberto Basili. 2018. Neural learning for question answering in italian. In International Conference of the Italian Association for Artificial Intelligence.

Alan Edelman, Tom´as A Arias, and Steven T Smith. 1998. The geometry of algorithms with orthogonality constraints. SIAM journal on Matrix Analysis and Applications, 20(2):303–353.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. Transformer Circuits Thread.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Annual Meeting of the Association for Computational Linguistics.

Christiane Fellbaum. 1998. Wordnet: an electronic lexical database. MIT Press google schola, 2:678–686.

Christiane Fellbaum. 2010. Wordnet. In Theory and applications of ontology: computer applications, pages 231–243. Springer.

Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and Marta Ruiz Costa-juss`a. 2024. A primer on the inner workings of transformer-based language models. ArXiv, abs/2405.00208.

Kazuhiro Fukui and Atsuto Maki. 2015. Difference subspace and its generalization for subspacebased methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(11):2164– 2177.

Kazuhiro Fukui, Naoya Sogi, Takumi Kobayashi, Jing-Hao Xue, and Atsuto Maki. 2023. Discriminant feature extraction by generalized difference subspace. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):1618–1635.

Team Gemma. 2024. Gemma: Open models based on gemini research and technology. CoRR, abs/2403.08295.

Marjan Ghazvininejad, Xing Shi, Jay Priyadarshi, and Kevin Knight. 2017. Hafez: an interactive poetry generation system. In Proceedings of ACL 2017, System Demonstrations, pages 43–48, Vancouver, Canada. Association for Computational Linguistics.

Alex Graves. 2012. Sequence transduction with recurrent neural networks. ArXiv, abs/1211.3711.

Sanda Harabagiu, George A Miller, and Dan Moldovan. 1999. Wordnet 2-a morphologically and semantically enhanced resource. In SIGLEX99: Standardizing lexical resources.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. ArXiv, abs/1904.09751.

Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. 2019. A benchmark for interpretability methods in deep neural networks. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Daniel Jurafsky and James H. Martin. 2000. Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zachary Dodds, Nova Dassarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, John Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom B. Brown, Jack Clark, Nicholas Joseph, Benjamin Mann, Sam McCandlish, Christopher Olah, and Jared Kaplan. 2022. Language models (mostly) know what they know. CoRR, abs/2207.05221.

Muhammad Khalifa, Hady ElSahar, and Marc Dymetman. 2020. A distributional approach to controlled text generation. ArXiv, abs/2012.11635.

Jelena Kovaˇcevi´c and Amina Chebira. 2008. An introduction to frames. Foundations and Trends® in Signal Processing, 2(1):1–94.

Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq R. Joty, Richard Socher, and Nazneen Rajani. 2020. Gedi: Generative discriminator guided sequence generation. In Conference on Empirical Methods in Natural Language Processing.

Patrick Lewis, Barlas O˘guz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2019. Mlqa: Evaluating cross-lingual extractive question answering. ArXiv, abs/1910.07475.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. 2023. Awq: Activation-aware weight quantization for ondevice llm compression and acceleration. In Conference on Machine Learning and Systems.

Team Llama. 2024. The llama 3 herd of models. ArXiv, abs/2407.21783.

Andr´e L. G. Mandolesi. 2022. Asymmetric metrics on the full grassmannian of subspaces of different dimensions. ArXiv, abs/2208.05026.

Nathan Mankovich and Tolga Birdal. 2023. Chordal averaging on flag manifolds and its applications. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3858– 3867.

Tim Marrinan, Bruce A. Draper, J. Ross Beveridge, Michael J. Kirby, and Chris Peterson. 2014. Finding the subspace mean or median to fit your need. 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 1082–1089.

Encyclopedia of Mathematics. 2016. Angle. Last visited on 2024/10/21.

Leland McInnes and John Healy. 2018. Umap: Uniform manifold approximation and projection for dimension reduction. ArXiv, abs/1802.03426.

Research Microsoft. 2024. Phi-3 technical report: A highly capable language model locally on your phone. ArXiv, abs/2404.14219.

Tomas Mikolov, Wen tau Yih, and Geoffrey Zweig. 2013. Linguistic regularities in continuous space word representations. In North American Chapter of the Association for Computational Linguistics.

George A Miller. 1995. Wordnet: a lexical database for english. Communications of the ACM, 38(11):39–41.

George A Miller, Richard Beckwith, Christiane Fellbaum, Derek Gross, and Katherine J Miller. 1990. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4):235–244.

Viktor Moskvoretskii, Ekaterina Neminova, Alina Lobanova, Alexander Panchenko, and Irina Nikishina. 2024a. Taxollama: Wordnet-based model for solving multiple lexical sematic tasks. ArXiv, abs/2403.09207.

Viktor Moskvoretskii, Alexander Panchenko, Irina Nikishina, and Skoltech. 2024b. Are large language models good at lexical semantics? a case of taxonomy learning. In International Conference on Language Resources and Evaluation.

Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. 2024. The geometry of categorical and hierarchical concepts in large language models. ArXiv, abs/2406.01506.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2023. The linear representation hypothesis and the geometry of large language models. In NeurIPS 2023 Workshop on Causal Representation Learning.

- Peter H Sch¨onemann. 1966. A generalized solution of the orthogonal procrustes problem. Psychometrika, 31(1):1–10.

Erica K. Shimomoto, Fran¸cois Portet, and Kazuhiro Fukui. 2021. Text classification based on the word subspace representation. Pattern Analysis and Applications, 24:1075 – 1093.

Shashwat Singh, Shauli Ravfogel, Jonathan Herzig, Roee Aharoni, Ryan Cotterell, and Ponnurangam Kumaraguru. 2024a. Representation surgery: Theory and practice of affine steering.

Shivalika Singh, Freddie Vargus, Daniel Dsouza, B¨orje F. Karlsson, Abinaya Mahendiran, WeiYin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzemi’nski, Hakimeh Fadaei, Irem Ergun,

Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Minh Chien Vu, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, A. Ustun, Marzieh Fadaee, and Sara Hooker. 2024b. Aya dataset: An open-access collection for multilingual instruction tuning. In Annual Meeting of the Association for Computational Linguistics.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. 2024. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread.

Pedro Valois, Koichiro Niinuma, and Kazuhiro Fukui. 2023. Occlusion sensitivity analysis with augmentation subspace perturbation in deep feature space. 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4817–4826.

Wiebke Wagner. 2010. Steven bird, ewan klein and edward loper: Natural language processing with python, analyzing text with the natural language toolkit. Language Resources and Evaluation, 44:421–424.

Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, and Jundong Li. 2023. Knowledge editing for large language models: A survey. ArXiv, abs/2310.16218.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Huai hsin Chi, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. ArXiv, abs/2203.11171.

Satoshi Watanabe. 1967. Evaluation and selection of variables in pattern recognition. Comp. & Info. Sciences, pages 91–122.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, MingSung Kan, Junxian He, and Qizhe Xie. 2023. Self-evaluation guided beam search for reasoning. In Neural Information Processing Systems.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. ArXiv, abs/2305.10601.

Ke Ye and Lek-Heng Lim. 2016. Schubert varieties and distances between subspaces of different dimensions. SIAM Journal on Matrix Analysis and Applications, 37(3):1176–1197.

### A Mathematical Details and Proofs A.1 Linear Decomposition of Tokens

Lemma A.1 (Concept estimation). Let {yj}nj=1 be a set of tokens sharing a common concept s, we can estimate the concept as

s ∝

∼

n

u yj − u0, (23)

j=1

with error of order O √ 1n .

Proof. Let uj = u yj . If {uj}nj=1 share a common meaning s, then by Postulate 4.1 every uj is represented as

uj − u0 =

s

i

n

aisi (24)

aisi = ajs +

si̸=s

where aj is the coefficient of s for the token uj. Then,

 ajs +

n

n

uj − u0 =

j=1

j=1

n

n

=

ajs +

j=1

j=1

n

n

= as +

j=1

si̸=s

1 √n

= as + O

  (25)

aisi

si̸=s

n

aisi (26)

si̸=s

aisi (27)

(28)

where a = nj=1 aj and nj=1 ns

i̸=s aisi is a rough estimate of the sample mean, which should tend to zero with error equal to the standard error of the mean (SEM), considering the common concept factors more distinctly than the others. ■

- Proposition A.1 (1st-order Concepts). A concept

C has ray representation R s′C , where s′C is a normalized counterfactual concept pair difference:

sC=1 − sC=0 ∥sC=1 − sC=0∥

s′C =

(29)

Proof. Following Equation (5), the unembedding representation of a concept C is computed as a normalized mean of counterfactual pairs. Thus,

nC

u′C =

ui (C = 1) − ui (C = 0) (30)

i

nC

nC

ui (C = 0) (31)

ui (C = 1) −

=

i

i

nC

ui (C = 1) − u0 (32)

=

i

nC

ui (C = 0) − u0 (33)

−

i

= sC=1 − sC=0 (34)

where sC=1 = ni C ui (C = 1) − u0 and sC=0 = ni C ui (C = 0) − u0 are concepts for each counterfactual pair item. Then, normalizing u′C gives s′C at Equation (29). ■

A.2 Frame Representation Hypothesis A.2.1 Rays and Subspaces

Let v,u ∈ Rd be two vectors, angle θ between them, their respective rays and 1-dim subspaces are two distinct structures which can be generalized to points in Grassmann manifolds differing only by choice of distance. Rays use the chordal Frobenius distance, also known as the Procrustes distance (Mandolesi, 2022), given by

θ 2

dP R(v),R(u) = ∥v − u∥F = 2sin

,

(35)

where ∥·∥F is the Frobenius norm, making the Frobenius inner product the space inner product.

In this context, correlation (1) is induced by the choice of distance and norm:

ρ R(v),R(u) = ⟨v,u⟩F ∥v∥F∥u∥F

(36) = cosθ (37)

where we use the term “correlation” to indicate a generalization of the traditional cosine similarity beyond just vectors.

On the other hand, their respective subspaces [v],[u] are compared using the projective distance

dP [v],[u] = vv⊤ − uu⊤

(38)

F

= 1 − cos2 θ. (39) Thus,

ρ [v],[u] = ⟨vv⊤,uu⊤⟩F

(40) = cos2 θ. (41)

uu⊤

F uu⊤

F

#### A.2.2 Combined Concept Geometrical Interpretation

We show a geometrical interpretation of the Combined Concept Frame, illustrated in Figure 12.

Proposition A.2. Let A,B ∈ St(k,d) be Concept Frames (19). Then, D(B,A) is the frame which best approximates the direction of the geodesic from A to B at its midpoint.

Proof. Let γ (t) = Aexp(tΩ) be the geodesic connecting A and B, Ω = log A⊤B ∈ Rk×k. Given the matrix exponential MacLaurin series exp(X) = ∞n=0 Xn/n!, we find that the derivative of the geodesic at the midpoint is

γ′ (½) = Aexp Ω/2 Ω (42)

∞

Ωn+1 n!2n

(43)

= A

n=0

Ω2 2

+ ... (44)

= A Ω +

Similarly,

B − A = A(expΩ − I) (45)

∞

Ωn n!

(46)

= A

n=1

Ω2 2

+ ... (47)

= A Ω +

The series match to second order. Therefore, if

- A and B are not unrelated concepts, B − A ≈ γ′ (½). Since D(B,A) is the closest frame to
- B − A, D(B,A) simultaneously approximates the direction of the geodesic at the midpoint. ■

B Top-k Concept-Guided Decoding

#### B.1 Text Completion Examples

These are Text Completion samples from Llama

- 3.1 8B. We show extra examples of the guided decoding mechanism, including text completions and harmful instructions. The grey area is the input and the remaining text model output.

men are not allowed to wear skirts. A. true B. false Answer: B noguidance

M logM A logM B A − B B − A

A B

γB→A′ (t = ½) M γA→B′ (t = ½)

A B

Figure 12: Illustrations of Difference Frames on the Stiefel manifold. Top: Frames A and B are projected onto tangent space at the midpoint M (frame mean). Difference subspaces are represented by arrows connecting M to each Logarithm Map. Bottom: Difference Frames are approximately equivalent to the gradient of the geodesic between the two frames at the midpoint M = γA→B (½) = γB→A (½), thereby B − A ≈ γA→B′ (½), while A − B ≈ γB→A′ (½)

men are not allowed to wear shorts in public. A. in Saudi Arabia. B. in Iran. C. anywhere. D. in the United States. Answer: A

woman.n.01 − male.n.01

women are from venus men are from mars. noguidance

women are not the only ones to suffer the effects of the patriachal society, but also the men.

man.n.01 − woman.n.01

kids are the future of the world. noguidance

kids are the future of our country and we need them in school to learn and be educated so that we have the best future for our country.

man.n.01

children are the future of the world. noguidance

children are a gift. A friend sent me an article today where it [...] showed a link in a childless woman having an 8-9 year increase on her risk for breast cancer than a mother.

woman.n.01

Text Completion samples from Gemma 2 9B:

men are not the only ones who can be a victim of domestic violence. noguidance

men are the most important part in a woman’s existence man.n.01

women are the most beautiful creatures on earth. noguidance

women are the most important part in the world. woman.n.01

kids are so cute.

no guidance

kids are so cute and adorable, they can do anything to get your heart mel

man.n.01 − woman.n.01

children are the future of the country. noguidance

children are a great blessing to us as a couple but also a great burden to the church. woman.n.01−man.n.01

### C Extra Experimental Results

#### C.1 Experimental Settings

Experiments were conducted on a single Nvidia H100 80Gb GPU, so the largest Llama 3.1 405B was not considered. The multilingual instruction dataset was compiled from questions sourced from the Aya Dataset (Singh et al., 2024b), supplemented with additional samples in Italian from SQuAD Italian (Croce et al., 2018), German and Thai from XQuAD (Artetxe et al., 2019), and Hindi from MLQA (Lewis et al., 2019), ensuring a minimum of 1000 sentences for each of the 8 languages supported by Llama 3.1 models. Gemma 2 and Phi were restricted to English samples, following their official language support.

#### C.2 Top-k Concept-Guided Decoding Language Comparison

We analyze the strength of the steering effect with respect to the k factor. Figure 13 shows that actually most languages are similarly affected on average, but as visible at fig. 14, the standard deviation of the steering effect is higher for Hindi and Thai, which show a noisy pattern, possibly due limitations on the model’s own capacity at handling these languages.

[Figure 8]

###### Figure 13: Growth of steering effect for the 8 languages supported by Llama 3.1 8B Instruct using top-k guided generation. Rescaled for visibility.

[Figure 9]

###### Figure 14: Growth of steering effect standard deviation for the 8 languages supported by Llama 3.1 8B Instruct using top-k guided generation.

