# arXiv:2406.02543v2[cs.LG]17Jul2024

## To Believe or Not to Believe Your LLM

Yasin Abbasi Yadkori, Ilja Kuzborskij, András György, Csaba Szepesvári Google DeepMind

July 18, 2024

Abstract

We explore uncertainty quantification in large language models (LLMs), with the goal to identify when uncertainty in responses given a query is large. We simultaneously consider both epistemic and aleatoric uncertainties, where the former comes from the lack of knowledge about the ground truth (such as about facts or the language), and the latter comes from irreducible randomness (such as multiple possible answers). In particular, we derive an information-theoretic metric that allows to reliably detect when only epistemic uncertainty is large, in which case the output of the model is unreliable. This condition can be computed based solely on the output of the model obtained simply by some special iterative prompting based on the previous responses. Such quantification, for instance, allows to detect hallucinations (cases when epistemic uncertainty is high) in both single- and multi-answer responses. This is in contrast to many standard uncertainty quantification strategies (such as thresholding the log-likelihood of a response) where hallucinations in the multi-answer case cannot be detected. We conduct a series of experiments which demonstrate the advantage of our formulation. Further, our investigations shed some light on how the probabilities assigned to a given output by an LLM can be amplified by iterative prompting, which might be of independent interest.

### 1 Introduction

Who’s talking? I asked, peering behind the mirror. Many dead spiders and a lot of dust were there. Then I pressed my left eye with my index finger. This was an old formula for detecting hallucinations, which I had read in To Believe or Not to Believe?, the gripping book by B. B. Bittner. It is sufficient to press on the eyeball, and all the real objects, in contradistinction to the hallucinated, will double. The mirror promptly divided into two and my worried and sleep-dulled face appeared in it.

—"Monday Starts on Saturday" by A. and B. Strugatsky

Like the protagonist of the novel, language models too occasionally suffer from hallucinations, or responses with low truthfulness, that do not match our own common or textbook knowledge (Bubeck et al., 2023; Gemini Team, Google, 2023). At the same time, since LLMs work by modeling a probability distribution over texts, it is natural to view the problem of truthfulness through the lens of statistical uncertainty. In this paper we explore uncertainty quantification in LLMs. We distinguish between two sources of uncertainty: epistemic and aleatoric (Wen et al., 2022; Osband et al., 2023; Johnson et al., 2024). Epistemic uncertainty arises from the lack of knowledge about the ground truth (e.g., facts or grammar in the language), stemming from various reasons such as insufficient amount of training data or model capacity. Aleatoric uncertainty comes from irreducible randomness in the prediction problem, such as multiple valid answers to the same query. Hence, truthfulness can be directly analyzed via looking at the epistemic uncertainty of a model in the sense that when the epistemic uncertainty is low, the model predictions must be close to the ground truth.

Rigorously identifying when (either) uncertainty is small1 is notoriously hard, especially in deep neural networks (Blundell et al., 2015; Antorán et al., 2020). This is because we generally lack guarantees

1For instance, by saying that predictions live in a confidence set with high probability.

about learning the ground truth (consistency), or even a weaker guarantee about how large the variance of a learning algorithm is. At the same time, there exist many heuristic approaches for uncertainty quantification based on simply looking at the log-likelihood of responses (Kadavath et al., 2022), estimating entropy (Kuhn et al., 2023), ensembling (Lakshminarayanan et al., 2017b; Dwaracherla et al., 2023; Osband et al., 2023), or sometimes even more principled formulations, such as conformal prediction (Angelopoulos et al., 2023; Ravfogel et al., 2023; Yadkori et al., 2024) (which however come with strong assumptions).

To the best of our knowledge, a common limitation of these approaches is that they are only meaningful in problems where there exists a single correct response (e.g. label) as they aim for detecting if one response is dominant (or multiple responses with the same meaning), that is, if there is only little uncertainty in the prediction. On the other hand, when multiple responses are correct, that is, there is aleatoric uncertainty in the ground truth, simply estimating the amount of uncertainty in the LLM’s output is insufficient, as the perfect (ground-truth) predictor may have large aleatoric uncertainty and no epistemic uncertainty, while a completely useless predictor may have large epistemic uncertainty only, but the total amount of uncertainty of the two predictors might be the same.

Contributions. In this paper we address the above problem directly, and design methods to decouple epistemic and aleatoric uncertainty, allowing us to effectively deal with multi-response queries. Rather than trying to quantify how small epistemic uncertainty can be, we aim to identify when only the epistemic uncertainty is large, in which case we can suspect that the response is hallucinated.2

As a starting point we make a simple observation: If multiple responses are obtained to the same query from the ground truth (the language), they should be independent from each other, that is, in probabilistic interpretation, the joint distribution of these multiple responses, for a fixed query, must be a product distribution.

This observation can be used to measure how far the language model can be from the ground truth. The sequential model implemented by a language model allows us to construct a joint distribution over multiple responses, which is done through iterative prompting of an LLM based on its previous responses and the application of the chain rule of probability: first we ask the model to provide a response given a query, then to provide another response given the query and the first response, then a third one given the query and the first two responses, an so on. This is in contrast to some of the earlier works that approached decoupling epistemic and aleatoric uncertainty for classification problems by training the model with pairs (or tuples) of labels (Wen et al., 2022; Johnson et al., 2024).

So, if the response to a prompt containing the query and previous responses is insensitive to the previous responses, we have the desired independence and the LLM-derived joint distribution can be arbitrarily close to the ground truth. On the other hand, if the responses within the context heavily influence new responses from the model then, intuitively speaking, the LLM has low confidence about the knowledge stored in its parameters, and so the LLM-derived joint distribution cannot be close to the ground truth. As more responses are added to the prompt, this dependence can be made more apparent, allowing to detect epistemic uncertainty via our iterative prompting procedure.

Interestingly, as we will see in Section 3, we can force an LLM to provide a desired (possibly incorrect) response by adding this response repeatedly to the prompt. This phenomenon is then further investigated from the viewpoint of a transformer LLM architecture in Section 3.1.

The iterative prompting procedure then leads to the following main contributions:

- (i) Based on the above iterative prompting procedure, we derive an information-theoretic metric of epistemic uncertainty in LLMs (Section 4), which quantifies the gap between the LLM-derived distribution over responses and the ground truth. This gap is insensitive to aleatoric uncertainty, and therefore we can quantify epistemic uncertainty even in cases where there are multiple valid responses.
- (ii) We derive a computable lower bound on this metric, which turns out to be a mutual information (MI) of an LLM-derived joint distribution over responses,3 and propose a finite-sample estimator for it. We prove that this finite-sample MI estimator sometimes suffers only a negligible error even though LLMs and their derived joint distributions are defined over potentially infinite supports (all possible strings in a language).

- 2In technical terms this corresponds to giving a lower bound, rather than an upper bound, on the quantity capturing the uncertainty.
- 3Here MI is understood as a functional of a joint distribution (see Section 2).

- (iii) We discuss an algorithm for hallucination detection based on thresholding a finite-sample MI

estimator, where the threshold is computed automatically through a calibration procedure. We show experimentally on closed-book open-domain question-answering benchmarks (such as TriviaQA, AmbigQA,

- and a dataset synthesized from WordNet) that when the data is mostly composed of either single-label or multi-label queries, our MI-based hallucination detection method surpasses a naive baseline (which is based on the likelihood of the response), and achieves essentially similar performance to that of a more advanced baseline which is based on the entropy of the output as a proxy for uncertainty. However, on datasets which contain both single- and multi-label samples at the same time, our method also significantly outperforms the entropy-based baseline, by achieving a much higher recall rate on samples with high output entropy while maintaining similar error rates.

- (iv) Focusing on a single self-attention head, we identify a simple mechanistic explanation for how

the model output can be changed through iterative prompting using previous responses, as discussed earlier. Suppose that the prompt is composed from a query and a repeated element (e.g., a possibly wrong answer). If the query lies within the space spanned by the large principal components of a key-query matrix product, then the output will be generated according to the knowledge extracted from the training data (now stored in a value matrix). On the other hand, if the query has little overlap with the large principal components, then the repeated element is likely to be copied from the prompt.

Notation. As usual, N and R denote the sets of natural and real numbers, respectively. For any measurable set Z, we denote the the set of distributions supported on Z by M1(Z). For any positive integer k, we denote [k] = {1,...,k}.

### 2 Preliminaries

In this section we present some basic definitions used throughout the paper.

Conditional distributions and prompting. Let X be the space of finite text sequences, that is X ⊂ Σ∗ where Σ is a finite alphabet (and Σ∗ = ∞n=1 Σn). Moreover, consider a family of conditional distributions P = {µ : X → [0,1] | x∈X µ(x | x′) = 1 ∀x′ ∈ X}. In the following, we let P ∈ P be the ground-truth conditional probability distribution over text sequences (responses) given a prompt, and we let Q ∈ P be the learned language model. Given a fixed query x ∈ X and possible responses Y1,...,Yt, we define a family of prompts F = {Ft : X → X | t ∈ N}, such that Ft(x,Y1,...,Yt) is defined as:

Consider the following question: Q: x One answer to question Q is Y1. Another answer to question Q is Y2.[...] Another answer to question Q is Yt. Provide an answer to the following question: Q: x. A:

Information-theoretic notions. Let µ,µ′ be distributions supported on set Z = Z1 × ··· × Zn where (Zi)i is a collection of countable sets. The entropy of a distribution µ is defined as H(µ) =

z∈Z µ(z)ln(1/µ(z)).4 If µ,µ′ are such that µ′(z) = 0 only if µ(z) = 0, we have a Kullback-Leibler divergence between them defined as DKL(µ,µ′) = z∈Z µ(z)ln(µ(z)/µ′(z)). For any z ∈ Z, we denote z\i = (z1,...,zi−1,zi+1,...,zn), and the marginal of the ith coordinate of µ is given by µi(z) =

z\i∈Zn−1 µ(z). The product distribution of the marginals of µ is given by µ⊗(z) = ni=1 µi(z), and the mutual information of µ is defined as I(µ) = DKL(µ,µ⊗).

### 3 Probability amplification by iteratively prompting

In this section we demonstrate that, as mentioned in the introduction, repeating possible responses several times in a prompt can have pronounced effects on the output of a language model. Consider x =“What

4Following the usual convention, we define 0 ln 0 = 0 and a ln(a/0) = ∞ for any a > 0.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Q: What is the capital of the UK? A: London (≈ 1.0) and Paris (1.29 × 10−10).

Q: Who was the first US president? A: George Washington (0.999) and Abraham Lincoln (3.1 × 10−06).

Q: Who is the author of The Grapes of Wrath? A: John Steinbeck (≈ 1.0) and Ernest Hemingway (1.34× 10−10).

Q: What is the largest country in the world? A: Russia (0.999) and United Kingdom (9.02 × 10−06).

- Figure 1: Single-label queries with low epistemic uncertainty: Conditional normalized probability of the correct completion given repetitions of an incorrect response. Each figure shows the query and the considered two responses with their initial probabilities, as a response for the query, in parentheses (the first response is the correct one).

[Figure 5]

Q: What is the national instrument of Ireland? A: The harp (0.936) and Uilleann pipes (0.063).

[Figure 6]

Q: Which actor became M in the Bond film Skyfall? A: Ralph Fiennes (0.651) and Judi Dench (0.348).

[Figure 7]

Q: Which can last longer with out water a camel or a rat? A: A rat (0.538) and A camel (0.461).

[Figure 8]

Q: If Monday’s child is fair of face what is Saturday’s child? A: Work hard for a living (0.093) and Full of grace (0.906).

- Figure 2: Single-label queries with high epistemic uncertainty: Conditional normalized probability of the correct completion given repetitions of an incorrect response. Each figure shows the query and the considered two responses with their initial probabilities, as a response for the query, in parentheses (the first response is the correct one).

is the capital of the UK?” and Y1 = ··· = Yt =“Another answer to question Q is Paris.” Here we can repeat the sentence “Another answer to question Q is Paris.” an arbitrary number of times. Although the number of repetitions changes the behavior of the LLM, the correct response maintains a significant probability: as Figure 1 shows, the conditional normalized probability5 of the correct response, “London”, reduces from approximately 1 to about 96% as we increase the number of repetitions of the incorrect response to 100. Figure 1 shows 3 more examples where, with initially low epistemic uncertainty in the response to the query (the aleatoric uncertainty is also low as we consider single-response queries), the correct response maintains a significant or non-negligible probability even in the presence of repetitions of incorrect information, while the probability of predicting the latter is increased.

Next, we consider a queries for which the model is more uncertain. For the prompt “What is the national instrument of Ireland?”, we observe that responses “The harp” and “Uilleann pipes” both have significant probabilities (the first answer is the correct one). This time, by incorporating the incorrect response in the prompt multiple times, the probability of the correct answer quickly collapses to near zero, as shown in Figure 2, together with three more examples with significant epistemic uncertainty.

Finally, we consider multi-label queries for which the LLM confidently knows a correct answer. This time, by incorporating a potential response in the prompt, the probabilities of other correct answers stay relatively large. Figure 3 shows four such examples.

#### 3.1 In-context learning vs. in-weight learning

The sensitivity of the response of an LLM to extra in-context information, as observed above, can already be observed in a single attention head as explained next.

′

be an input matrix comprised of n semantic feature vectors each of dimension d′. Each row is meant to represent a complete statement

We consider an idealized attention mechanism as follows. Let Z ∈ Rn×d

5To obtain conditional normalized probabilities, we consider the probabilities of the two responses, and normalize them so that they add to 1.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Q: Name a city in the UK A: London (0.958) and Manchester (0.041).

Q: Name a yellow fruit A: Banana (0.715) and Lemon (0.284).

Q: Name an alcoholic drink, A: Wine (0.685) and Beer (0.314).

Q: Name a ball game that is played by more than 5 players A: Volleyball (0.542) and Soccer (0.457).

- Figure 3: Multi-label queries with aleatoric uncertainty: Conditional normalized probability of the first of the two provided responses, both of which are correct, given repetitions of the second response in the prompt. Each figure shows the query and the considered two responses with their initial probabilities, as a response for the query, in parentheses.

(such as “What is the capital of the UK?” or “One answer to the question is Paris.”, etc.) rather than a single token. Let X⊤ ∈ R1×d

′

be the first row of Z, which represents the query of interest, such as “What is the capital of the UK?”. Let E⊤ ∈ R1×d

′

be a special vector indicating the end of the input. The matrix Z \ X, denoting the Z matrix without its first row, represents the in-context information.

We assume the ground-truth distribution P is such that a query vector is mapped to its response, but a statement is simply copied. For example, for V =“What is the capital of the UK?”, P(· | V ) would be a distribution with support on “London” and its variations, while for V ′ =“What is the capital of the UK? One answer to the question is Paris.”, P(· | V ′) returns the same distribution. We assume a parameter matrix WV is learned such that V ⊤WV estimates P(· | V ) for vector V .

Let WQ,WK,WV ∈ Rd

′×d be the query, key, and value matrices. A self-attention head with query

- X and context Z \ X is defined as

f(Z;WQ,WK,WV) = Softmax

1 √

d

E⊤WQ(ZWK)⊤ ZWV

where the output of the softmax is a row vector of length n.

If X has appeared many times in the training data, then parameters WQ and WK could be learned such that E⊤WQ(WK)⊤X is large, that is, X is within the space spanned by the large principal components of the key-query matrix product. Then, no matter what in-context information appears in Z, the probability assigned to X will dominate the softmax, and we will have

Softmax

1 √

d

E⊤WQ(ZWK)⊤ Z ≈ X⊤ ,

and therefore f(Z;WQ,WK,WV) ≈ P(· | X).

On the other hand, consider the case that X has not appeared many times in the training data, and vector Y is copied in many rows of Z. Then E⊤WQ(WK)⊤X could be small as X is not in the span of the large principal components of the key-query matrix product. Therefore f(Z;WQ,WK,WV) ≈ Y since

Softmax

1 √

d

E⊤WQ(ZWK)⊤ Z ≈ Y ⊤ .

Even if X is in the span, repeating Y t times in Z would give a t-times increased total weight to Y inside the softmax, which can dominate the weight assigned to X when t is large enough, also resulting in Y as the answer.

- 4 Metric of epistemic uncertainty and its estimation

In this section we apply iterative prompting to estimate the epistemic uncertainty of a language model about responding to some query. The idea is to utilize the different behavior patterns observed in Section 3, which can be used to differentiate between two modes of high uncertainty: when the aleatoric uncertainty is high vs. when only the epistemic uncertainty is high. We then apply our new uncertainty metric to design a score-based hallucination detection algorithm.

We will first present the uncertainty metric and its estimate for a distribution defined on the direct outputs of an LLM, and then in Section 4.2, we discuss the changes needed to take semantic equivalences of language into account (Kuhn et al., 2023; Farquhar et al., 2024).

Recall the family of prompts F defined in Section 2. We make the following assumption about the ground truth, which states that multiple responses to the same question drawn according to the ground truth are independent from each other:

Assumption 4.1 (Ground truth independence assumption). The ground-truth satisfies

P(Yt | Ft−1(x,Y1,...,Yt−1)) = P(Yt | x) for any t ∈ N and any x,Y1,...,Yt ∈ X.

Note that the above assumption is heavily dependent on our prompt construction. Without embedding Y1,...,Yt−1 in the prompt, the independence assumption would not hold, for example, if Y1,...,Yt were partial answers, such as a step of an algorithm or a part of a story, because in such a case Yt might indeed depend on the previous outputs Y1,...,Yt−1. Roughly speaking, the assumption tells that the response distribution is insensitive to a query based on previously sampled responses. For example, for query x =“A city in the UK:”, the probability of Y2 =“Manchester” does not change if a city is Y1 =“London”. Now we formally introduce a notion of the joint distribution over responses given a query, derived from the language model:

Definition 4.2 (Pseudo joint distribution). Given a family of prompt functions F, a conditional distribution µ ∈ P, and n ∈ N, we use notation · to denote a pseudo joint distribution defined as

µ(Y1,...,Yn | x) = µ(Y1 | F0(x))µ(Y2 | F1(x,Y1))···µ(Yn | Fn−1(x,Y1,...,Yn−1)) . (1)

The above is a pseudo joint distribution since the standard conditioning in the chain-rule is replaced with prompt functions of the conditioning variables. In the following we focus on Q derived from the LLM and P derived from the ground truth.

Remark 4.3 (Sampling from Q). Note that sampling from Q can be simply done through a chainrule-like procedure as can be seen from the above definition, that is, to have (Y1,...,Yn) ∼ Q we draw

- Y1 ∼ Q(· | F0(x)), Y2 ∼ Q(· | F1(x,Y1)), Y3 ∼ Q(· | F2(x,Y1,Y2)), and so on.

In the rest of the paper we drop subscripts in joint distributions and conditioning on query x (which is understood implicitly), for example, P ≡ PY

1···Yn|x.

To measure epistemic uncertainty, we need to quantify how far the estimated pseudo joint distribution Q˜ is from the ground truth P˜. One natural choice is the following definition: Definition 4.4 (Epistemic uncertainty metric). Given an input x ∈ X, we say that the epistemic uncertainty of Q is quantified by DKL( Q, P).

Here DKL measures how well Q approximates P for a given query x. Namely, this metric determines if Q assigns a large probability to an event which has a small probability under P. In case of LLMs, this means the LLM generates a sequence that is unlikely in the typical usage of the language. In Figure 4 we have a situation where P is a pseudo joint distribution derived from the ground-truth, and Q suffers from a high hallucination rate. Given an input x, we want to estimate the above hallucination metric, but we only have access to Q, and so computing it explicitly is impossible. However, next we show that under Assumption 4.1 we can lower bound DKL( Q, P) by a quantity which only depends on Q (the proof is given in Appendix D).

| | |
|---|---|
| | |

Figure 4: A hallucination: Q places an excessive mass where the ground truth P has a low mass.

Theorem 4.5. For all pseudo joint distributions P satisfying Assumption 4.1, we have that

DKL( Q, P) ≥ I( Q) .

|1: Input: µ ∈ M1(Xn) ............................... any (pseudo-) joint distribution over Xn k ∈ N ...................................... sample size γ1,γ2 ≥ 0 .................................. stabilization parameters (typically selected as 1/k)<br>2: Independently sample tuples X1, ..., Xk ∼ µ ∈ M1(Xn)<br>3: Construct a set of indices of unique elements U = i ∈ [k] : Xi = Xj ∀j < i<br>4: Construct empirical distributions: for all i ∈ U,<br><br>µ(Xi) =<br><br>µ(Xi) Z<br><br>, where Z =<br><br>j∈U<br><br>µ(Xj)<br><br>µ⊗(Xi) =<br><br>n<br><br>j=1 t∈U:Xt,j=Xi,j<br><br>µˆ(Xt,1,...,Xi,j,...,Xt,n)<br><br>5: Compute estimate<br><br><br>Ik(γ1,γ2) =<br><br>i∈U<br><br>µ(Xi) ln<br><br>µ(Xi) + γ1 µ⊗(Xi) + γ2<br><br>|
|---|

̸

Algorithm 1: MI estimator.

The lower bound in the theorem holds uniformly for all P, and it is computable solely based on Q˜. This makes the bound applicable for decision making; in fact we chose to consider DKL( Q, P) as the measure of epistemic uncertainty (out of many similar distance measures) because it admits this property).

Also, note that we have I( Q) = DKL( Q, Q⊗) , Q⊗ = i y\i Q(y1,...,yi−1,Yi,yi+1,...,yn). In gen-

eral y\i Q(y1,...,yi−1,Yi,yi+1,...,yn) ̸= Q(Yi), because the independence assumption Assumption 4.1 does not necessarily (and, in practice, almost never) holds for Q.

Finally, a quantity related to DKL( Q, P) is DKL with arguments arranged in the opposite order, that is DKL( P, Q) which is a (query) conditional excess risk of the LLM-derived pseudo joint distribution Q, under the logarithmic loss. Controlling the excess risk (for instance, upper-bounding it) for various algorithms is one of the central questions in learning theory, however it is a much harder task than the one we consider here, because for the former we need to theoretically control all sources of errors (such as generalization, estimation, and approximation error).

#### 4.1 A computable lower bound on epistemic uncertainty

- Theorem 4.5 gives a lower bound on the epistemic uncertainty by the mutual information. However, to compute the mutual information term, in practice we need to evaluate Q on its entire support, which is potentially infinite. Practically speaking, it is impossible to observe probabilities of all strings under the language model and so we must rely on a finite sample. Therefore, we replace Q with an empirical distribution with a finite support; in the following we show that the error induced by such an approximation is controlled. To estimate the MI we employ the method given in Algorithm 1; for generality it is presented for an arbitrary (pseudo) joint distribution µ, but we keep in mind that our case of interest is µ = Q. Note that most terms in the summations defining the product distribution µ⊗ are

zero (except the ones which correspond to the observed data). Adding γ1 and γ2 in the estimator Ik(γ1,γ2) is intended to account for the total probability of missing observations, not included while constructing µ and µ⊗, making sure the estimate is bounded. Similar ideas are well-know in probability and information theory, such as in universal coding (Cesa-Bianchi and Lugosi, 2006), Laplace smoothing (Polyanskiy and Wu, 2024) and Good-Turing smoothing (Gale and Sampson, 1995; McAllester and Schapire, 2000). In Section 4.2, we show an extension of the algorithm that takes semantic equivalences into account, and in the experiments section, we will present a version of the algorithm that takes advantage of the

available log-likelihood function in LLMs and constructs the empirical joint and product distributions in a slightly different way. As we will show in the experiments section, n = 2 is sufficient to have an effective hallucination detection method for the benchmarks that we consider.

The bias introduced by (γ1,γ2) in the last equation allows us to rigorously bound the error in estimating I(µ) via Ik(γ1,γ2), which is explored next. In particular, in Theorem 4.6 we prove a high-probability lower bound on I(µ) in terms of Ik. The core of controlling the estimation error is in accounting for the missing mass, or in other words, how much of µ we miss out by only observing a finite sample. In Appendix E, we present a more complete discussion and the proof of the bound on the estimation error for mutual information. Here we adapt this result to our particular case.

Define the missing mass as

µ(x)I x ̸∈ {X1,...,Xk} .

Uk =

x∈Xn

Using this quantity, we are ready to present a non-asymptotic bound on the estimation error, which depends on the estimator Ik(γ1,γ2), the expected missing mass, and the sample size:

- Theorem 4.6. Suppose that Ik(γ1,γ2) is given by Algorithm 1, and assume that X is finite. For

- γ1 = 1/(k |Xn|), and γ2 satisfying γ2 ≥ γ1 + n(1 − Z), with probability at least 1 − δ, we have

I(µ) ≥ (1 − εk) Ik(γ1,γ2) −

1 k

+ (1 + n ln 1 + k |X|) εk

where εk = E[Uk] +

ln(1δ) k

.

Furthermore, given δsupp ∈ [0,1), let X˜ ⊆ Xn such that µ(X˜) ≥ 1 − δsupp. Then, for γ1 = 1/(k |X|˜ ), and

- γ2 satisfying γ2 ≥ γ1 + n(1 − Z), with probability at least 1 − δ, we have

1 k

+ (1 + ln 1 + k |X|˜ ) (δsupp + εk) .

I(µ) ≥ (1 − εk) Ik(γ1,γ2) −

The theorem is a corollary of Theorem E.4 shown in Appendix E. Note that in Theorem 4.6 we consider two bounds. The first one is pessimistic in the sense that it does not expect that the samples carry much information about the support, and it is most suitable in situations where we expect µ to be spread out (uniformly) across its entire support. The price of not having samples covering the whole support in this case is a factor nln|X| appearing in the bound. For example, in case of a language model with 10,000 tokens, considering all possible strings of length T tokens yields nln|X| = nT ln(10000), and so

I(µ) ≥ (1 − εk) Ik(γ1,γ2) −

1 k

+ (1 + nT ln 1 + k ln(10000)) εk .

Arguably, in practice, such situations are rare, as in natural languages we will not encounter all possible strings. To this end, we consider an optimistic scenario where the effective support of µ, denoted by X˜, is small with high probability. In this case, we can replace the size of the support for strings of length n, |X|n, in the first bound with the effective support size |X|˜ , and we only pay essentially a factor ln(1 + k|X|˜ ) instead of nln(1 + k|X|). In case the effective sample size is only polynomial in n, this leads to an exponential reduction in n for the second term in the bounds. In fact, in Appendix E.4 we demonstrate some empirical evidence that on two question-answering benchmarks, |X|˜ rarely exceeds ≈ 100 with µ(X˜) ≥ 0.95, while sampling responses from an LLM given a query.

Next we consider sufficient conditions for the estimator to converge to the mutual information. In particular, using the first bound in the theorem, we have (hiding logarithmic factors)

I(µ) = Ω˜ (1 − E[Uk]) Ik(γ1,γ2) − E[Uk] k → ∞ .

This tells us that the rate of estimation error is essentially controlled by the expected missing mass E[Uk], which, as we will see, converges to zero as k → ∞, however the decay can be very slow in general. For

example, it is known that for a finite support of size N, E[Uk] ≤ e−Nk when k ≤ N and E[Uk] ≤ N/(ek) otherwise (Berend and Kontorovich, 2012). For countable distributions with entropy bounded by h, one has E[Uk] ≤ h/ln(k) (Berend et al., 2017).6

Despite these pessimistic bounds, in reality we expect the expected missing mass to be significantly smaller, especially when µ is heavy-tailed. It is well-known that natural languages (and many artificial ones) follow a Zipf distribution, where probability of each word (or a text piece) is proportional to 1/freq(text)α for some exponent α > 1, where freq() is a frequency of occurrence in the corpus (Piantadosi, 2014). Then, we expect that E[Uk] should be much smaller than in such a case, since sampling from the tail of Zipf distribution is a rare event. To this end, in Appendix E we show that if Q is Zipf with exponent α > 1, then for any free parameter β > 0,

α−1

E[Uk] = O k−(

α −β) .

Hence, the rate at which the expected missing mass vanishes can be very fast (potentially matching a concentration rate 1/

√

k for α = 2).

Finally in Appendix E.4 we present a data-dependent estimation of E[Uk] based on a concentration inequality for a missing mass and repetitive sampling from LLM, in the context of some Q/A datasets. We conclude that the expected missing mass is very small: Most of the upper bounds on the expected missing mass (we have one upper bound per question) are highly concentrated close to 0.

#### 4.2 Taking semantic equivalences into account

Although Theorem 4.5 and Theorem 4.6 provide a lower bound for the divergence between the LLM distribution Q and the ground-truth P, the bound might be loose as it ignores the semantic equivalences between texts. Given a semantic equivalence definition, we propose constructing new ground-truth P′ and LLM distribution Q′, where the probability of a cluster is the sum of probabilities of all semantically equivalent texts in that cluster. We use a similarity function s to define semantic equivalences: two texts are considered equivalent if their similarity is greater than a given threshold τ. Our choices for similarity functions in the experiments are described in Section 6. We assume the similarity function and the similarity threshold induce a clustering of the space X, i.e. s(Y,Y ′) ≥ τ if and only if they are in the same cluster.

In practice, rather than constructing the aforementioned distribution Q′ explicitly, we can draw samples from Q′ by sampling from Q and aggregating samples according to their clusters. The modified uncertainty estimating algorithm is shown in Algorithm 2. The estimator is constructed using only (semantically) equivalent elements in the sample (the indices of these representative elements are collected in S), that is, we do not account for duplicate samples and we aggregate probabilities of samples that are lexically different but semantically equivalent. Algorithm 2 works with the aggregated probability distribution µ′ = Q′ (line 4) by summing over cumulative probabilities over clusters. Note that DKL(µ) ≥ DKL(µ′) by monotonicity property of KL-divergence (Polyanskiy and Wu, 2024, Theorem 2.16) (this is because µ′ is defined on a smaller support). Therefore, Theorem 4.6 implicitly gives a bound on I(µ′), and eventually we have I(µ) ≥ I(µ′). But we can also directly apply Theorem 4.5 and Theorem 4.6 to distributions P′ and Q′ and obtain that DKL( Q′, P′) ≥ I( Q′).

### 5 Score-based hallucination tests

Let Ik(γ,x) ≡ Ik(γ) computed as in Algorithm 1 for µ = Q, to emphasize the explicit dependence on the query x. The uncertainty estimate Ik(γ,x) derived above can be used as a score indicating the strength of our belief that the LLM hallucinates for the given query x. Such a score can then be used to design abstention policies: if the response is deemed to be hallucinated, the system abstains from

6Note that expected missing mass E[Uk] appearing here is related to the well-known Good-Turing estimator. Let M be the number of elements among X1, . . . , Xk which appear exactly once. Then, the Good-Turing estimator is defined as UkGT = M/k. An attractive property of the Good-Turing estimator is that it is unbiased in the sense that E[UkGT] − E[Uk] = E[Uk(1)]/k where the random variable Uk(1) is the cumulative probability of the sequences appearing exactly once in the data. Although we do not directly work with the Good-Turing estimator in this paper, its convergence properties can be analyzed using a technique similar to the one we employ here (Berend and Kontorovich, 2012).

|1: Input: µ ∈ M1(Xn) ............................... any (pseudo-) joint distribution over Xn k ∈ N ...................................... sample size γ1,γ2 ≥ 0 .................................. stabilization parameters (typically selected as 1/k) s : Xn × Xn → R ...........................a similarity function τ ∈ R ...................................... a similarity threshold<br>2: Independently sample tuples X1, ..., Xk ∼ µ ∈ M1(Xn)<br>3: Construct a set of indices of unique elements U = i ∈ [k] : Xi = Xj ∀j < i<br>4: Construct cluster centers S ⊂ U according to the similarity function: for all i,t ∈ S, we have<br><br>s(Xi,Xt) < τ and cluster associated with Xi is D(i) = j ∈ U : s(Xi,Xj) ≥ τ . Aggregated probabilities: for all i ∈ S,<br><br>µ′(Xi) =<br><br>j∈D(i)<br><br>µ(Xj)<br><br>5: Construct empirical distributions: for all i ∈ S,<br><br>µ(Xi) =<br><br>µ′(Xi) Z<br><br>, where Z =<br><br>j∈S<br><br>µ′(Xj)<br><br>µ⊗(Xi) =<br><br>n<br><br>j=1 t∈S:Xt,j=Xi,j<br><br>µˆ(Xt,1,...,Xi,j,...,Xt,n)<br><br>6: Compute estimate<br><br><br>Ik(γ1,γ2) =<br><br>i∈S<br><br>µ(Xi) ln<br><br>µ(Xi) + γ1 µ⊗(Xi) + γ2<br><br>|
|---|

̸

Algorithm 2: MI estimator. Python implementation with usage example is given in Appendix A.

responding, while a response is provided otherwise. Score-based abstention methods usually compute a score chosen by the user (such as the response likelihood or the estimator I(γ) discussed earlier), and declare hallucination if the score is above or below a threshold, which is determined through calibration. To detect hallucinations successfully, the threshold can be adjusted through calibration on a given task using a hold-out (ground-truth) sample, see, for instance, the paper of Yadkori et al. (2024) where this calibration is discussed in detail.

Given our estimated lower bound on the epistemic uncertainty, we can define an abstention policy (a policy which decides when the LLM should abstain from prediction) as

- 0, if Ik(γ1,γ2,x) < λ;
- 1, if Ik(γ1,γ2,x) ≥ λ;

aλ(x) =

where λ > 0 is a threshold parameter tuned on a hold-out sample of some particular task. This policy abstains (aλ(x) = 1) when the epistemic uncertainty in the prediction (response) is large. When the policy does not abstain (aλ(x) = 0), any prediction from Q can be served.

In the experiments, we compare a number of scoring functions for detecting hallucinations, including I(γ), the probability of the greedy (temperature zero) response, and an estimate of the entropy of the response distribution.

### 6 Experiments

In this section we evaluate our abstention method derived based on the MI estimate in Section 5 on a variety of closed-book open-domain question-answering tasks.

Language model. We used a Gemini 1.0 Pro model (Gemini Team, Google, 2023) to generate outputs and scores.

Datasets. We consider three different datasets and their combinations: As base datasets, we consider (i) a random subset of 50,000 datapoints from the TriviaQA dataset (Joshi et al., 2017), and (ii) the entire AmbigQA dataset (with 12038 datapoints) (Min et al., 2020). These datasets mostly contain single-label queries, and only contain a few multi-label ones.7 Moreover, we created a multi-label dataset based on the WordNet dataset (Fellbaum, 1998): We extracted all (6015) datapoints from WordNet at depth 4 or more of the physical_entity subtree. For each datapoint (entity, children) in WordNet, we constructed a query of the form “Name a type of entity.” and children are considered target labels.

Comparison of responses and computing the output distributions. We use the F1 score8 thresholded at 0.25 to decide if two text sequences match. When multiple responses are sampled, we approximate the output distribution of an LLM in a semantically meaningful way by collapsing matching responses into a single response: we sample k = 10 responses at temperature 0.9 for each query, and after eliminating repetitions, all those that match (according to the F1 score) are considered identical and their probabilities are aggregated. We only consider queries for which the greedy (temperature zero) and at least one of the random responses are shorter than 20 characters. This is because the F1 score (as a match function) and log-probabilities (as a measure of uncertainty) are less reliable for longer sequences. After this filtering, we are left with 38870 datapoints for TriviaQA, 5315 datapoints for AmbigQA, and 3296 datapoints for WordNet.

As shown in prior works (e.g. Kuhn et al. (2023); Yadkori et al. (2024)), we can use LLM self-prompting to obtain more reliable text comparisons specially for longer outputs. Such an approach however is computationally much more expensive.

Baselines. We consider abstention policies based on four scoring methods. The first three are as follows: (i) the probability of the greedy response (denoted by T0); (ii) the semantic-entropy method of Kuhn et al. (2023) whose score is the entropy of k = 10 generated samples (denoted by S.E.). To calculate entropy, we first aggregate probabilities of equivalent responses and normalize the probabilities so that they sum to 1 (as described above); and (iii) our proposed mutual information score as defined in Section 4 (and denoted by M.I.) with the choices of k = 10, n = 2, and γ1 = γ2 = 0 (the latter choice approximates the case that the number of potential responses can be very large in which case the theoretical choice of γ1 and γ2 would be very small). To calculate the mutual information, as shown in Algorithm 3, we first generate k = 10 random samples. Then for any response Y , we calculate the probability of all generated responses given the prompt F1(x,Y ). We construct estimates Q(Y ) and Q(Y ′|Y ) by aggregating probabilities of equivalent responses, and normalizing the probabilities so that they sum to 1.

The calculation of the mutual information is slightly different than the algorithms presented in Algorithm 1 and Algorithm 2 and takes advantage of the available log-likelihood function in LLMs. Notice that the input µ in Algorithm 3 is LLM distribution Q as opposed to being the pseudo joint distribution

- Q in Algorithm 1. Another difference is that the similarity function s now takes two texts as input (as opposed to taking two n-dimensional arrays of texts as inputs in Algorithm 2). As explained earlier, we use the F1 score as the similarity function and we use τ = 0.25 as the similarity threshold.

Each baseline also has a default choice which is taken when the relevant score is above a threshold, and hence the method does not abstain. For T0, the default choice is the greedy (temperature zero) response. For S.E., the default choice is the response with the highest (aggregate) probability among the

- 7Note that the multi-label queries in these datasets typically behave as single-label ones in the sense that the LLM assigns overwhelming probability to a dominant response.
- 8In this context, the F1 score is calculated based on token inclusion (Joshi et al., 2017; Devlin et al., 2019): for two sequences a = (a1, . . . , an) and b = (b1, . . . , bm), defining p = |a ∩ b|/n and r = |a ∩ b|/m (where |a ∩ b| is the size of the intersection of a and b, in which for repetitions of an element y, we consider the minimum number of repetitions in a

- and b, i.e., minc∈{a,b} |{i : ci = y}|, in calculating the size of the intersection) we define F1 = 2pr/(p + r). Relating to the standard definition of the F1 score, p and r play the role of precision and recall, respectively, if a is thought of as a prediction of b.

|1: Input: µ ∈ M1(X) ................................ any distribution over X k ∈ N ...................................... sample size γ1,γ2 ≥ 0 .................................. stabilization parameters (typically selected as 1/k) s : X × X → R ............................. a similarity function τ ∈ R ...................................... a similarity threshold<br>2: Independently sample outputs X1, ..., Xk ∼ µ ∈ M1(X)<br>3: Construct a set of indices of unique elements U = i ∈ [k] : Xi = Xj ∀j < i<br>4: Construct cluster centers S ⊂ U according to the similarity function: for all i,t ∈ S, we have<br><br>s(Xi,Xt) < τ and cluster associated with Xi is D(i) = j ∈ U : s(Xi,Xj) ≥ τ . Aggregated probabilities: for all i,t ∈ S,<br><br>µ′1(Xi) =<br><br>j∈D(i)<br><br>µ(Xj), µ′2(Xt |Xi) =<br><br>j∈D(t)<br><br>µ(Xj |Xi)<br><br>5: Construct empirical distributions: for all i,t ∈ S,<br><br>µ1(Xi) =<br><br>µ′1(Xi) Z<br><br>, where Z =<br><br>j∈S<br><br>µ′1(Xj)<br><br>µ2(Xt |Xi) =<br><br>µ′2(Xt |Xi) Zi<br><br>, where Zi =<br><br>j∈S<br><br>µ′2(Xj |Xi)<br><br>µ(Xi,Xt) = µ1(Xi) µ2(Xt |Xi), µ⊗(Xi,Xt) = µ1(Xi)<br><br>j∈S<br><br>µ1(Xj) µ2(Xt |Xj)<br><br>6: Compute estimate<br><br><br>Ik(γ1,γ2) =<br><br>i,t∈S<br><br>µ(Xi,Xt) ln<br><br>µ(Xi,Xt) + γ1 µ⊗(Xi,Xt) + γ2<br><br>|
|---|

̸

Algorithm 3: Alternative MI estimator. A usage example is given in Appendix B

generated random responses. For the M.I. method, the default choice is the sampled response with the highest probability according to the marginalized pseudo joint distribution.

We also consider a version of the self-verification method of Kadavath et al. (2022) (denoted by S.V.) that, for a query x, first finds Y1, the element with the largest (aggregated) probability (which is the default choice of S.E. method), and then calculates the probability of token “True” (normalized for the two tokens “True” and “False”) for the following query: “Consider the following question: Q: x. One answer to question Q is Y1. Is the above answer to question Q correct? Answer True or False. A:”. The default choice of this baseline is the same as the default choice of the S.E. method. By this design, our intention is to construct a score that (unlike the first-order scores9 we consider) is not sensitive to the size of the label set.

In our experiments we either sweep through all abstention thresholds (Figure 5), or optimize the threshold on some calibration data, as explained in the description of the relevant experiment (Figure 6).

Results. We consider the precision-recall (PR) trade-off for the various methods on the different datasets. Here, recall is the percentage of queries where the method does not abstain, and precision is the percentage of correct decisions among these queries.10 Figure 5ab show PR-curves for the baselines and the proposed method on TriviaQA and AmbigQA. As can be seen, our method is better than the

9The scores T0 and S.E. are first order because they only consider the marginal distribution of a single response, unlike our uncertainty score which is based on MI estimation by considering (pseudo) joint distributions over multiple responses.

10In some figures, for better illustration, we show the error rate which is one minus the precision.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

(a) TriviaQA (b) AmbigQA (c) TriviaQA+WordNet (d) AmbigQA+WordNet

- Figure 5: PR-curve for the baseline and the proposed methods on various datasets. On the TriviaQA and AmbigQA datasets, M.I. and S.E. perform nearly identically, but they outperform the T0 and S.V. baselines. For the S.E. and M.I. methods, the responses for a large number of queries can be clustered into a single group, and therefore the semantic entropy and mutual information scores are zero. This is why the starting point of their curves is at a higher recall values. On the TriviaQA+WordNet and AmbigQA+WordNet datasets with a significant number of high entropy multi-label queries, M.I. outperforms the S.E. baseline. The methods perform nearly identical on the recall area that is not shown.

T0 and S.V. baselines, but performs similarly to the S.E. method. This is because the TriviaQA and AmbigQA datasets contain mostly single-label queries, and therefore a first-order method such as S.E. is sufficient to detect hallucinations. The AmbigQA dataset contains a few multi-label queries, but upon closer inspection, we observe that the LLM has low entropy on most of these queries.11 Therefore, a first-order method can perform as well as our method on such queries. Our proposed method, as well as the baselines, make no mistakes on the WordNet dataset (as the prediction of the LLM is always correct), hence we omit those results. The S.V. baseline performs significantly worse than the other methods when the recall is not high (is below about 0.8).

The similar performance for the S.E. and M.I. methods shown in Figure 5ab is due to the fact that the LLM has low entropy on most multi-label queries. However, ideally, an LLM should have higher entropy on multi-label queries (which would demonstrate broader knowledge, not focusing on a single possible answer). To include such queries, we mix the TriviaQA and AmbigQA datasets with our WordNet-based dataset with “truely” multi-label queries as constructed above. To enhance the intended effect, we filter our WordNet dataset by keeping only queries with entropy higher than 0.7 (approximately the entropy of the uniform distribution over two atoms). Then we have 842 remaining datapoints in WordNet. Note that when considered in isolation, both our proposed method and the semantic entropy method rarely make mistakes on this dataset. Then we create two new datasets by combining our 842 WordNet datapoints with 842 randomly selected datapoints from TriviaQA and AmbigQA, respectively, resulting in the TriviaQA+WordNet and AmbigQA+WordNet datasets. Figure 5cd show PR-curves for the S.E. and M.I. methods on these two combined datasets. Apart from low recall values, the performance of the S.E. method degrades noticeably with the addition of extra multi-label data. This precision/recall curve might look somewhat strange (with precision sometimes increasing with recall); this is due to the fact that both methods are always correct on the large number of high-entropy WordNet queries, where the LLM’s default predictions are correct.

The hardness with the combined datasets is that the predominantly single-label datasets (TriviaQA, AmbigQA) might need a different calibration threshold than the multi-label WordNet dataset, and this is better handled by our proposed method than by S.E. To better illustrate the improved abstention properties of our method, we examine how the two methods handle when the output of the LLM is diverse (i.e., has high entropy). In order to do this, we perform the following experiment: We create a calibration dataset by adding 500 random datapoints from the WordNet dataset to 500 random datapoints from TriviaQA, and another such random dataset for test. We determine the abstention thresholds on the calibration dataset for both the S.E. and the M.E. methods,12 and measure the performance (error rate, i.e., 1 minus precision, and recall) of the resulting abstention policies on the test set. We repeat this process 10 times and report mean values and 95% confidence intervals with Gaussian approximation. We perform a similar evaluation process for mixtures of AmbigQA and WordNet datasets. Figure 6 show that

- 11Such a case can also be seen in the query “Name a city in the UK.” in Figure 3 where the response “London” has probability 0.958.
- 12This is done by fixing the target loss rates of 0.05 for TriviaQA and 0.15 for AmbigQA, and finding threshold parameters that lead to these rates on the calibration set.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

(a) TriviaQA+WordNet (b) TriviaQA+WordNet (c) AmbigQA+WordNet (d) AmbigQA+WordNet

- Figure 6: Recall and error rates (one minus precision: percentage of mistakes when not abstaining) of the proposed and the baseline method on TriviaQA+WordNet and AmbigQA+WordNet datasets. On TriviaQA+WordNet and AmbigQA+WordNet datasets, the methods are calibrated at target loss of 0.05 and 0.15, respectively. On the x-axis, the queries are partitioned according to the entropy of the LLM’s output. Error bars show 2 standard deviation confidence intervals (based on 10 repetitions). While the first-order S.E. method has similar recall and error rates to those of the proposed M.E. method on low-entropy queries, its recall values are nearly zero for queries with higher entropy.

while the S.E. method has similar recall and error rates to those of the proposed method on low-entropy queries, its recall values are much lower for queries with higher entropy, while the M.E. method makes only few mistakes on these queries.

- 7 Conclusions

In this paper we considered epistemic uncertainty as a proxy for the truthfulness of LLMs. We proposed a mutual-information-based uncertainty estimator that admits a provable lower bound on the epistemic uncertainty of the LLM’s response to a query. That we consider joint distributions of multiple answers allows us to disentangle epistemic and aleatoric uncertainty, which makes it possible to better detect hallucination than first order methods, which can only tackle uncertainty as a whole, not epistemic uncertainty alone. This approach yielded an abstention method that performs significantly better on mixed single-label/multi-label datasets than first-order methods. While earlier methods for classification that aim to quantify epistemic uncertainty are usually based on a modified training method using response-tuples, utilizing the sequential nature of LLMs, our method does not need to change the training procedure, but needs to prompt the model iteratively with multiple responses generated by the LLM for the same query.

### References

Gustaf Ahdritz, Tian Qin, Nikhil Vyas, Boaz Barak, and Benjamin L. Edelman. Distinguishing the knowable from the unknowable with language models, 2024. URL https://arxiv.org/abs/2402. 03563.

Anastasios N Angelopoulos, Stephen Bates, et al. Conformal prediction: A gentle introduction. Foundations and Trends® in Machine Learning, 16(4):494–591, 2023.

Javier Antorán, James Allingham, and José Miguel Hernández-Lobato. Depth uncertainty in neural networks. Conference on Neural Information Processing Systems (NeurIPS), 2020.

Amos Azaria and Tom Mitchell. The internal state of an LLM knows when its lying. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Daniel Berend and Aryeh Kontorovich. The missing mass problem. Statistics & Probability Letters, 82(6): 1102–1110, 2012.

Daniel Berend and Aryeh Kontorovich. On the concentration of the missing mass. Electronic Communications in Probability, 2013.

Daniel Berend, Aryeh Kontorovich, and Gil Zagdanski. The expected missing mass under an entropy constraint. Entropy, 19(7):315, 2017.

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International Conference on Machine Learing (ICML), 2015.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations (ICLR), 2023.

Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. In International Conference on Learning Representations (ICLR), 2024a.

Shiqi Chen, Miao Xiong, Junteng Liu, Zhengxuan Wu, Teng Xiao, Siyang Gao, and Junxian He. In-context sharpness as alerts: An inner representation perspective for hallucination mitigation. In Forty-first International Conference on Machine Learning, 2024b. URL https://openreview.net/forum?id= s3e8poX3kb.

Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, and Denny Zhou. Universal self-consistency for large language model generation,

- 2023.

Jeremy R. Cole, Michael JQ Zhang, Daniel Gillick, Julian Martin Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. Selectively answering ambiguous questions. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2019.

Vikranth Dwaracherla, Zheng Wen, Ian Osband, Xiuyuan Lu, Seyed Mohammad Asghari, and Benjamin Van Roy. Ensembles for uncertainty estimation: Benefits of prior functions and bootstrapping. Transactions on Machine Learning Research (TMLR), 2023. ISSN 2835-8856.

S. Farquhar, J. Kossen, L. Kuhn, and Yarin Gal. Detecting hallucinations in large language models using

semantic entropy. 2024. Christiane Fellbaum. WordNet: An electronic lexical database. MIT press, 1998. Yarin Gal. Uncertainty in deep learning. PhD thesis, University of Cambridge, 2016. William A Gale and Geoffrey Sampson. Good-turing frequency estimation without tears. Journal of

quantitative linguistics, 2(3):217–237, 1995. Gemini Team, Google. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. [Online; accessed 01-February-2024].

Bairu Hou, Yujian Liu, Kaizhi Qian, Jacob Andreas, Shiyu Chang, and Yang Zhang. Decomposing uncertainty for large language models through input clarification ensembling. In International Conference on Machine Learing (ICML), 2024.

Mingjian Jiang, Yangjun Ruan, Sicong Huang, Saifei Liao, Silviu Pitis, Roger Grosse, and Jimmy Ba. Calibrating language models via augmented prompt ensembles. In Workshop on Challenges in Deployable Generative AI at International Conference on Machine Learning, 2024.

Daniel D. Johnson, Daniel Tarlow, David Duvenaud, and Chris J. Maddison. Experts don’t cheat: Learning what you don’t know by predicting pairs. arXiv preprint arXiv:2402.08733, 2024.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Transactions of the Association for Computational Linguistics (ACL), pages 1601–1611, 2017.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield Dodds, Nova DasSarma, Eli Tran-Johnson, and et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Nora Kassner and Hinrich Schütze. Negated and misprimed probes for pretrained language models: Birds

can talk, but cannot fly. In Transactions of the Association for Computational Linguistics (ACL), 2020. Satyapriya Krishna. On the intersection of self-correction and trust in language models, 2023. URL

https://arxiv.org/abs/2311.02801.

Satyapriya Krishna, Chirag Agarwal, and Himabindu Lakkaraju. Understanding the effects of iterative prompting on truthfulness. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=KjazcKPMME.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations (ICLR), 2023.

Philippe Laban, Lidiya Murakhovs’ka, Caiming Xiong, and Chien-Sheng Wu. Are you sure? challenging llms leads to performance drops in the flipflop experiment, 2024. URL https://arxiv.org/abs/2311. 08596.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Conference on Neural Information Processing Systems (NeurIPS), volume 30, 2017a.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Conference on Neural Information Processing Systems (NeurIPS), 2017b.

Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar. Large language models with controllable working memory. In Transactions of the Association for Computational Linguistics (ACL), 2023.

Moxin Li, Wenjie Wang, Fuli Feng, Fengbin Zhu, Qifan Wang, and Tat-Seng Chua. Think twice before trusting: Self-detection for large language models through comprehensive answer reflection, 2024. URL https://arxiv.org/abs/2403.09972.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models. arXiv preprint arXiv:2305.19187, 2023.

Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. Entity-based knowledge conflicts in question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations (ICLR), 2020.

Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

David A McAllester and Robert E Schapire. On the convergence rate of good-turing estimators. In Conference on Computational Learning Theory (COLT), 2000.

Sabrina J. Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. Reducing conversational agents’ overconfidence through linguistic calibration. In Transactions of the Association for Computational Linguistics (ACL), 2022.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. AmbigQA: Answering ambiguous open-domain questions. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

- R. M. Neal. Bayesian learning for neural networks. Springer Science & Business Media, 2012.

Ella Neeman, Roee Aharoni, Or Honovich, Leshem Choshen, Idan Szpektor, and Omri Abend. Disentqa: Disentangling parametric and contextual knowledge with counterfactual question answering. arXiv preprint arXiv:2211.05655, 2022.

Mesrob I Ohannessian and Munther A Dahleh. Distribution-dependent performance of the good-turing estimator for the missing mass. In 19th International Symposium on Mathematical Theory of Networks and Systems, MTNS, 2010.

Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. In Conference on Neural Information Processing Systems (NeurIPS), volume 29, 2016.

Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Morteza Ibrahimi, Xiuyuan Lu, and Benjamin Van Roy. Epistemic neural networks. In Conference on Neural Information Processing Systems (NeurIPS), 2023.

Steven T Piantadosi. Zipf’s word frequency law in natural language: A critical review and future directions. Psychonomic bulletin & review, 21:1112–1130, 2014.

Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning. Cambridge University Press, 2024.

Stephan Rabanser, Anvith Thudi, Kimia Hamidieh, Adam Dziedzic, and Nicolas Papernot. Selective classification via neural network training dynamics. arXiv preprint arXiv:2205.13532, 2022.

Shauli Ravfogel, Yoav Goldberg, and Jacob Goldberger. Conformal nucleus sampling. In Transactions of the Association for Computational Linguistics (ACL), pages 27–34. Association for Computational Linguistics, 2023.

Robert J Tibshirani and Bradley Efron. An introduction to the bootstrap. Monographs on statistics and applied probability, 57(1), 1993.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR), 2022.

Ziyu Wang and Chris Holmes. On subjective uncertainty quantification and calibration in natural language generation, 2024. URL https://arxiv.org/abs/2406.05213.

Zheng Wen, Ian Osband, Chao Qin, Xiuyuan Lu, Morteza Ibrahimi, Vikranth Dwaracherla, Mohammad Asghari, and Benjamin Van Roy. From predictions to decisions: The importance of joint predictive distributions. arXiv preprint arXiv:2107.09224, 2022.

Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, András György, Adam Fisch, Arnaud Doucet, Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvári, Ali Taylan Cemgil, and Nenad Tomasev. Mitigating llm hallucinations via conformal abstention. arXiv preprint arXiv:2405.01563, 2024.

Fan Yin, Jayanth Srinivasa, and Kai-Wei Chang. Characterizing truthfulness in large language model generations with local intrinsic dimension. In Forty-first International Conference on Machine Learning,

2024. URL https://openreview.net/forum?id=7DbIyQlfaO.

Gal Yona, Roee Aharoni, and Mor Geva. Narrowing the knowledge evaluation gap: Open-domain question answering with multi-granularity answers. arXiv preprint arXiv:2401.04695, 2024.

Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley Malin, and Sricharan Kumar. Sac3: Reliable hallucination detection in black-box language models via semantic-aware cross-check consistency. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In In International Conference on Machine Learning, 2021.

Yukun Zhao, Lingyong Yan, Weiwei Sun, Guoliang Xing, Chong Meng, Shuaiqiang Wang, Zhicong Cheng, Zhaochun Ren, and Dawei Yin. Knowing what LLMs do not know: A simple yet effective self-detection method. arXiv preprint arXiv:2310.17918, 2024.

### A Implementation and usage examples of Algorithm 1 and Algorithm 2

In this section we present an implementation of Algorithm 1 and Algorithm 2 in Python with a simple usage example. In particular, the code given in Listing 1 generates a synthetic joint distribution over binary tuples with correlated elements (function create_synthetic_distribution). Then, we compute an exact mutual information of the distribution (function compute_MI_exactly) and use implementation of our estimator (function MI_estimator) to estimate a mutual information. This is done for various sample sizes, number of random variables, and levels of correlation (a single experiment is implemented by run_experiment) The results of these multiple experiments are eventually presented in as plots showing convergence of the estimate to the exact value of the mutual information. In practical applications, synthetic joint distribution (function create_synthetic_distribution) can be replaced by an LLMderived pseudo-joint distribution (see Definition 4.2). More detailed description of each function is given in Appendix A.1.

The example can be easily copied from listing.tex within https://arxiv.org/src/2406.02543. Listing 1: Implementation and usage examples of Algorithm 1 and Algorithm 2 on a synthetic joint distribution

# Copyright 2024 DeepMind Technologies Limited. # # Licensed under the Apache License, Version 2.0 (the "License"); # you may not use this file except in compliance with the License. # You may obtain a copy of the License at # # https://www.apache.org/licenses/LICENSE-2.0 # # Unless required by applicable law or agreed to in writing, software # distributed under the License is distributed on an "AS IS" BASIS, # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # See the License for the specific language governing permissions and # limitations under the License. # from itertools import product, combinations import numpy as np from matplotlib import pyplot as plt

def create_synthetic_distribution(space, temp): potential = lambda z: np.mean([x * y for x, y in combinations(z, 2)]) y = np.array([-np.exp(potential(x) / temp) for x in space]) return y / y.sum()

def sample_from_joint_distribution(space, joint_dist, k): indices = np.arange(len(joint_dist)) sampled_indices = np.random.choice(indices, p=joint_dist, size=k) sampled_tuples = space[sampled_indices] return sampled_tuples, sampled_indices

def cluster(tuples, joint_dist): return tuples, joint_dist

def sample_from_joint_distribution_and_cluster(space, joint_dist, k): sampled_tuples, sampled_tuple_indices = sample_from_joint_distribution(space, joint_dist, k) _, indices_of_uniques_in_sample = np.unique(sampled_tuples, axis=0, return_index=True) sampled_tuples = sampled_tuples[indices_of_uniques_in_sample] sampled_tuple_indices = sampled_tuple_indices[indices_of_uniques_in_sample] joint_dist_on_sample = joint_dist[sampled_tuple_indices] sampled_tuples, joint_dist_on_sample = cluster(sampled_tuples, joint_dist_on_sample) return joint_dist_on_sample, sampled_tuples

def compute_MI_exactly(space, mu): total = 0 for (x_i, x) in enumerate(space):

mu_x = mu[x_i] mu_x_prod = 1 for i in range(len(x)):

marg_indices = [j for (j, z) in enumerate(space) if z[i] == x[i]] mu_x_prod *= mu[marg_indices].sum()

total += mu_x * np.log(mu_x/mu_x_prod) return total

def MI_estimator(sampled_tuples, mu_on_sample, gamma_1, gamma_2): """Implements MI estimator (Algorithm 1).

Args: sampled_tuples: A numpy array of tuples sampled from the distribution after deduplication and clustering. mu_on_sample: A numpy array of probabilities of the clusters.

- gamma_1: stabilization parameter.
- gamma_2: stabilization parameter.

Returns: (float) mutual information. """

# Constructing empirical distribution (\hat{\mu}) hat_mu_on_sample = mu_on_sample / mu_on_sample.sum()

# Constructing empirical product distribution (\hat{\mu}^{\otimes}) hat_mu_prod_on_sample = np.zeros((len(hat_mu_on_sample),)) for (x_i, x) in enumerate(sampled_tuples):

hat_mu_x_prod = 1 for i in range(len(x)):

marg_indices = [j for (j, z) in enumerate(sampled_tuples) if z[i] == x[i]] hat_mu_x_prod *= hat_mu_on_sample[marg_indices].sum()

hat_mu_prod_on_sample[x_i] = hat_mu_x_prod

# Computing MI estimate mi_estimate = hat_mu_on_sample * np.log((hat_mu_on_sample + gamma_1) / (hat_mu_prod_on_sample + gamma_2) ) mi_estimate = mi_estimate.sum() return mi_estimate

def run_experiment(n, temp, ax): np.random.seed(1) space = np.array(list(product([-1, 1], repeat=n))) mu = create_synthetic_distribution(space, temp=temp) mi_exact = compute_MI_exactly(space, mu) k_range = np.linspace(10, 1000, 20, dtype=int)

all_mi_estimate = [] for k in k_range:

mu_on_sample, sampled_tuples = sample_from_joint_distribution_and_cluster(space=space, joint_dist=mu, k=k) gamma_1 = gamma_2 = 1/k mi_estimate = MI_estimator(sampled_tuples, mu_on_sample, gamma_1, gamma_2) all_mi_estimate.append(mi_estimate)

ax.axhline(mi_exact, linewidth=3, label="MI (exact value)", color="black") ax.plot(k_range, all_mi_estimate, linewidth=3, label="MI estimator") ax.grid(); ax.legend(); ax.set_xlabel("k"); ax.set_ylabel("MI estimate"); ax.set_title(r"$n=$"+str(n)+r", $\tau=$"+str(temp))

temp_range = [0.01, 0.1, 1, 10] n_range = [2, 4, 8]

fig, axs = plt.subplots(len(temp_range), len(n_range), figsize=(5*len(temp_range), 5*len(n_range)), squeeze=False) fig.suptitle(r"""MI estimation of $n$-dimensional distribution $\propto \exp(-\sum_{i < j}^n x_i x_j / \tau)$""")

- for (i, temp) in enumerate(temp_range):
- for (j, n) in enumerate(n_range): ax = axs[i,j] run_experiment(n=n, temp=temp, ax=ax)

plt.subplots_adjust(wspace=0.4, hspace=0.4) plt.show()

#### A.1 Additional documentation for functions in Listing 1

- • def create_synthetic_distribution(space, temp) Creates synthetic distribution which introduces dependendencies between variables.

Args: space: a list of tuples that the joint distribution is supported on (e.g. a cartesian product). temp: temperature parameter.

Returns: A numpy array of probabilities (same length as space).

- • def sample_from_joint_distribution(space, joint_dist, k) Samples k tuples from a joint distribution.

Args: space: a list of tuples that the joint distribution is supported on (e.g. a cartesian product). joint_dist: probability distribution (1-D numpy array) where each entry is a probability of a tuple.

k: sample size.

Returns: A numpy array of tuples sampled from the distribution. A numpy array of indices of sampled tuples in space.

- • def cluster(tuples, joint_dist) Clusters tuples and aggregates probabilities of them in the same cluster.

Args: tuples: A numpy array of tuples sampled from the distribution after deduplication. joint_dist: probability distribution (1-D numpy array) where each entry is a probability of a tuple.

Returns: A numpy array of tuples sampled from the distribution each represening a cluster. A numpy array of probabilities of clusters. Each probability is the aggregate of probabilities of all tuples in the cluster.

- • def sample_from_joint_distribution_and_cluster(space, joint_dist, k)

Samples k tuples from a joint distribution and retains only representative elements (removes all duplicates). Args:

space: a list of tuples that the joint distribution is supported on (e.g. a cartesian product). joint_dist: probability distribution (1-D numpy array) where each entry is a probability of a tuple. k: sample size.

Returns: A numpy array of tuples sampled from the distribution after deduplication. A numpy array of probabilities of deduplicated tuples.

- • def compute_MI_exactly(space, mu)

Computes mutual information of probability distribution mu exactly

Args: space: Tuple space (cartesian product). mu: probability distribution (1-D numpy array).

Returns: (float) mutual information.

- • def MI_estimator(sampled_tuples, mu_on_sample, gamma_1, gamma_2) Implements MI estimator (Algorithm 1).

Args: sampled_tuples: A numpy array of tuples sampled from the distribution after deduplication and clustering. mu_on_sample: A numpy array of probabilities of the clusters.

- gamma_1: stabilization parameter.
- gamma_2: stabilization parameter.

Returns: (float) mutual information.

- • def run_experiment(n, temp, ax)

Runs one experiment comparing exact mutual information estimation with Algorithm 1. Plots results. Args:

n: number of variables in a joint distribution. temp: temperature of a Gibbs distribution (joint distribution). Higher temperature typically means smaller MI. ax: pyplot axis object for plotting.

### B Usage example of Algorithm 3

Algorithm 3 is a slight modification of Algorithm 2 that we use in our experiments. We first explain Algorithm 3 via an example and then highlight the differences with Algorithm 2. In order to explain the implementation, we consider a running example with query x =“What is the capital of the UK?”, F1 score as the similarity function s, similarity threshold τ = 0.25, and number of samples k = 5. Algorithm 3 also takes the LLM distribution Q as input µ = Q.

Given the query, in step (2) of Algorithm 3, we sample k outputs from Q. Let’s assume these samples are X1 =“London”, X2 =“London”, X3 =“London, UK”, X4 =“Paris”, and X5 =“Berlin”. In step (3), we construct a set of indices of unique elements. In our example, we would have U = {1,3,4,5}. In step (4), we cluster responses and aggregate probabilities of each cluster. More precisely, if the F1 score of two responses is above 0.25, then they are in the same cluster. In our example, we have that F1(X1,X3) > 0.666 > 0.25 and F1(X1,X4) = F1(X1,X5) = 0, and therefore cluster centers are

- S = {1,4,5}. For query x, let’s assume LLM probabilities are Q(X1 |x) = 0.5, Q(X3 |x) = 0.2, Q(X4 |x) = 0.1, Q(X5 |x) = 0.05, ···

Also assume conditional distributions are Q(X1 |F1(x,X1)) = 0.6, Q(X3 |F1(x,X1)) = 0.15, Q(X4 |F1(x,X1)) = 0.05, Q(X5 |F1(x,X1)) = 0.04, ···

and so on (we have omitted writing Q(.|F1(x,X4)) and Q(.|F1(x,X5))). Then after step (4), the aggregated probabilities are

Q′(X1 |x) = 0.7, Q′(X4 |x) = 0.1, Q′(X5 |x) = 0.05, ··· and aggregated conditional probabilities are

Q′(X1 |F1(x,X1)) = 0.75, Q′(X4 |F1(x,X1)) = 0.05, Q′(X5 |F1(x,X1)) = 0.04, ···

and so on (we have similar aggregations for Q′(.|F1(x,X4)) and Q′(.|F1(x,X5))). Next, in step (5), we construct empirical estimates. We will have that Z = 0.85 and

Q1(X1 |x) ≈ 0.82, Q1(X4 |x) ≈ 0.11, Q1(X5 |x) ≈ 0.05, ··· For estimated conditional distributions, we will have Z1 = 0.84, and

Q2(X1 |F1(x,X1)) ≈ 0.89, Q2(X4 |F1(x,X1)) ≈ 0.06, Q2(X5 |F1(x,X1)) = 0.04, ···

and so on. The joint distribution Q(.,.|x), the product of marginals Q⊗(.,.|x), and the estimated mutual information Ik are trivially obtained by the equations in steps (5) and (6).

Next, we highlight the differences between Algorithm 2 (with the choice of n = 2) and Algorithm 3. The input distribution to Algorithm 2 is the pseudo-joint distribution Q, while the input to Algorithm 3 is the LLM distribution Q. So in step (2) of Algorithm 2, each sample is a tuple such as (“London”, “Paris”), while a sample in step (2) of Algorithm 3 is an LLM output such as “London”. Steps (3) and (4) of Algorithm 2 are similarly modified, and now the similarity function is defined over tuples.

### C Related work

In this section we present an overview of the related literature.

#### C.1 Bayesian neural networks

In a Bayesian framework, we can estimate the epistemic uncertainty by the uncertainty in the posterior distribution (Neal, 2012; Gal, 2016; Wang and Holmes, 2024). Implementing a Bayesian neural network however can be very challenging.

#### C.2 Iterative prompting

A number of iterative prompting strategies are developed to improve the factuality of LLMs (Chen et al., 2023; Krishna, 2023; Laban et al., 2024). The idea is to follow-up the LLM response with another question such as “Are you sure?”. Krishna et al. (2024) show that such strategies might in fact degrade LLM truthfulness due to a pattern of apologetic responses. Krishna et al. (2024) propose improved iterative prompting strategies, where instead of asking the LLM to re-think its response, the same question is posed again. They also propose strategies to collect more supporting facts and refine the final response accordingly. Li et al. (2024) propose an iterative prompting that instructs LLM to generate justifications for each answer before evaluating the correctness of the final answer. Different from these works, we assess hallucinations by measuring how LLM response changes with our iterative prompting scheme.

Perhaps the most related work to ours is the parallel and independent work of Ahdritz et al. (2024). Similar to us, Ahdritz et al. (2024) observe that in presence of high epistemic uncertainty, an LLM is more likely to copy the information provided in its context. For a given query, Ahdritz et al. (2024) propose considering top-k completions of the model, and then computing the entropy of the model conditioned on an iterative prompt composed of the original query and each completion. The minimum of these entropies is considered as a measure of the epistemic uncertainty. The method as it is, might fail on single-response queries where the model has low uncertainty. Nevertheless, we can design a two-stage process where we only consider completions that have probability higher than certain threshold in the first stage, and then compute the entropy of the model conditioned on an iterative prompt composed of the original query and each candidate completion in the second stage. By a proper tuning of the threshold of the first stage, we can potentially avoid mis-classification of low-uncertainty single-response queries. Tuning this threshold, however, would introduce extra complications. In contrast, we propose a principled test using a mutual information score that is guaranteed to be a lower bound on the KL-divergence between the LLM and the ground-truth. Further, we provide a mechanistic explanations for why LLMs behave as described in the presence of high epistemic uncertainty.

#### C.3 Training models with pairs of responses

Wen et al. (2022); Osband et al. (2023); Johnson et al. (2024) show that we can decouple epistemic and aleatoric uncertainty if we train a model with paired observations.

We discuss the more recent work of Johnson et al. (2024) in more detail. The proposed approach first estimates a model pˆY

1,Y2|x(y1,y2|x) over pairs using a training dataset of the form “query, first observation, second observation”. At test time, for a prompt x and response y, Johnson et al. (2024) consider

Vˆ(y | x) = pˆY

(y | x) p ˆY

2|Y1(y | y,x) − pˆY

(y | x)

1

1

1,Y2(y1,y2 | x) − pˆY

(y | x)ˆpY

(y | x)

= pˆY

1

1

as a measure of epistemic uncertainty. Assume that an equivalence class Φ (that maps a prompt to the set of equivalent prompts) is given, and let ν(. | Φ(x)) be a distribution (say, uniform) over class Φ(x). If the trained model is second order calibrated with respect to the equivalence class and the distribution ν, i.e.

ν(x′ | Φ(x))p(y1 | x′),

(y1 | x) =

pˆY

1

x′∈Φ(x)

ν(x′ | Φ(x))p(y1 | x′)p(y2 | x′),

1,Y2(y1,y2 | x) =

pˆY

x′∈Φ(x)

###### then it follows from definitions that in the class associated with x,

ν(x′ | Φ(x))Vˆ(y | x′) .

ν(x′ | Φ(x))(ˆpY

(y | x′) − p(y | x′))2 =

1

x′∈Φ(x)

x′∈Φ(x)

The quantity on the right-hand side is a measure of epistemic uncertainty. Notice that the equality states a coverage result, and it is not point-wise. Requiring the model to be second order calibrated is also a strong condition and ensuring it is highly non-trivial.

#### C.4 Epistemic neural nets

Ensemble methods are based on the classical idea of bootstrap for confidence estimation (Tibshirani and Efron, 1993), where multiple estimators for the regression function, each computed on a perturbed version of the data (e.g., by drawing samples from the empirical distribution over the data), are combined.

The empirical distribution of the resulting estimates is then used to construct confidence intervals. While many of these methods can be interpreted as sample-based approximations to Bayesian methods, model-hyperparameter selection (e.g., scale of perturbations, learning) for ensemble methods is typically done using a validation on holdout data (a subset of the training data). Many recent papers have studied ensemble methods in the context of deep learning and reinforcement learning (Osband et al., 2016; Lakshminarayanan et al., 2017a; Malinin and Gales, 2020). In the context of LLMs, the methods require training multiple language models, which is very expensive. Osband et al. (2023) introduces epistemic neural networks (epinets), which approximate ensemble methods by training a single network with an artificially injected (controlled) source of randomness. Rabanser et al. (2022) proposes to use intermediate model checkpoints to quantify the uncertainty of the final model in its responses. While these approaches aim to mimic the bootstrap procedure during prediction, their validity is not justified by theoretical considerations, and hence remain heuristic approximations.

#### C.5 Hallucination detection using first-order methods

First-order methods consider variance in the response distribution as a measure of hallucination (Kadavath et al., 2022; Cole et al., 2023; Manakul et al., 2023; Lin et al., 2023; Kuhn et al., 2023; Wang et al., 2022; Jiang et al., 2024; Zhang et al., 2023; Zhao et al., 2024; Yadkori et al., 2024). A common limitation of these approaches is that they are only applicable to prompts where there exists a single correct response,

- as they aim for detecting if one response (or multiple responses with the same meaning) is dominant. On the other hand, when multiple responses are correct, there is an aleatoric uncertainty in the ground truth: If an LLM correctly assigns non-negligible scores to multiple correct responses, most of these (if not all) will be declared as hallucination since, by design, only very few (typically at most one) responses can have scores higher than the threshold at the same time. Thus, hallucination detectors unaware of aleatoric uncertainty will invalidate most of the correct answers.

Yona et al. (2024) design a method that generates multiple responses, and then aggregates them into a single response at a (typically higher) granularity level where no further uncertainty (contradiction) is left compared to the generated responses. Although not a strictly first order method, it does not differentiate between aleatoric and epistemic uncertainty.

C.5.1 Asking language models to quantify uncertainty (self-verification)

Kadavath et al. (2022) propose to use LLM self-prompting to measure a model’s uncertainty in its responses. More specifically, for a given query, a number of responses are generated, and then the model is queried if the responses are correct. For this query, the log-probability of “True" is returned as a measure of uncertainty. Related approaches are studied by Mielke et al. (2022).

#### C.6 Uncertainty estimation based on sensitivity to contexts

Kassner and Schütze (2020); Zhao et al. (2021) show that an LLM’s responses can be influenced by irrelevant contexts. Longpre et al. (2021); Neeman et al. (2022) study two sources of knowledge: parametric knowledge stored in the network weights, and contextual knowledge retrieved from external sources. They view reliance of the model on its parametric knowledge and ignoring relevant contextual information as

hallucination. These works are mainly motivated by situations where the LLM’s knowledge is outdated and it is instructed to use the (new) contextual information. Accordingly, they design strategies to prioritize contextual information over parametric knowledge. Longpre et al. (2021) also show that larger models are more likely to ignore in-context information in favor of in-weight information. They propose creating training data with modified contextual information so that the model learns to favor the contextual information. Neeman et al. (2022) propose to train a model that predicts two answers: one based on parametric knowledge and one based on contextual information.

Similarly to Neeman et al. (2022), Li et al. (2023) aims to design a mechanism such that the model’s behavior is influenced more by relevant context than by its parametric knowledge (controllability), while the model is robust to irrelevant contexts (robustness). They improve controllability and robustness using finetuning.

Hou et al. (2024) study an approach to estimate model uncertainty due to ambiguity in a question. For a given question, their method generates multiple input clarification questions, and a new question is formed by augmenting the original question with each clarification question. The clarification questions are generated using an LLM with the aim of removing ambiguity in the question. This is different than the problem we study as the model can be uncertain about the answer even if the query itself has no ambiguity. For such queries, the method of Hou et al. (2024) might decide that no clarification is needed, and therefore there is no uncertainty.

#### C.7 Hallucination detection using internal states of LLMs

There are a number of papers that try to extract knowledge/truthfulness by inspecting hidden-layer activations of LLMs (Burns et al., 2023; Azaria and Mitchell, 2023; Chen et al., 2024a,b; Yin et al.,

- 2024). Such methods clearly require access to the LLM’s internal states, which is not always possible, and severely limits the applicability of these methods.

### D Omitted proofs

###### Proof of Theorem 4.5. In the following we will use abbreviations

y

=

y1,...,yn

,

y\i

=

y1,...,yi−1,yi+1,...,yn

where each coordinate belongs to X. Now,

1 P(y1,...,yn)

DKL( Q, P) = −H( Q) +

Q(y1,...,yn) ln

y

1 i P yi | Fi−1(y1,...,yi−1)

(using Definition 4.2)

= −H( Q) +

Q(y1,...,yn) ln

y

1 i P(yi)

. (by the independence assumption)

= −H( Q) +

Q(y1,...,yn) ln

y

Focusing on the last (cross-entropy) term

1 i P(yi)

Q(y1,...,yn) ln

y

1 P(yi)

ln

=

Q(y1,...,yn)

y

i

1 P(yi)

=

Q(y1,...,yn) ln

i yi y\i

(a)

1 y\i Q(y1,...,yn)

≥

Q(y1,...,yn) ln

i yi y\i

1 i y\i Q(y1,...,yn) where in (a) we used the fact that entropy is no larger than cross-entropy. Thus,

=

Q(y1,...,yn) ln

y

DKL( Q, P) ≥

y

Q(y1,...,yn) i y\i Q(y1,...,yn)

Q(y1,...,yn)ln

= I( Q;Y1,...,Yn) .

| |
|---|

### E Estimation of mutual information and missing mass problem

In this section, we discuss how to estimate the mutual information from a finite sample, which may not cover the full distribution. To control the estimation error, we first introduce the concept of missing mass.

#### E.1 The missing mass problem

Let X be a countable set and suppose that X1,...,Xk ∼ µ ∈ M1(Xn) independently. In the following x is used as an element of Xn rather than the query (as in Section 4). Then, the missing mass is defined as the random variable

µ(x)ξ(x) , ξ(x) = I{x ̸∈ {X1,...,Xk}} .

Uk =

x∈Xn

Here we are primarily interested in two questions: (i) how quickly Uk approaches the expected missing mass EUk, where it is not hard to see that

µ(x)(1 − µ(x))k ;

EUk =

x∈Xn

and (ii) we are also interested in giving an estimate for EUk given µ and k. The first question is answered by the following theorem:

- Theorem E.1 (Concentration of a missing mass (Berend and Kontorovich, 2013)). For any t > 0, we have an upper-tail bound

P(Uk > EUk + t) ≤ e−tk

2

, and moreover for a universal constant c ≈ 7.6821, we have an lower-tail bound P(Uk < EUk − t) ≤ e−ctk

2

.

Notably Uk exhibits a sub-gaussian concentration (i.e. 1/

√

k), which is surprisingly fast. As we will see next, the main bulk of the error incurred for missing a subset of the support is hidden in EUk.

In particular, when X is finite with |X| = N, Berend and Kontorovich (2012) showed that

EUk ≤

e−Nn , if n ≤ N;

N e n, if n > N.

In the countably infinite X, we cannot generally have a non-trivial bound on EUk only in terms of n. In fact, Berend and Kontorovich (2012) show a bound that depends on µ which is expected to be finite for rapidly decaying atoms. Interestingly, when the entropy of µ is bounded, one has the following result (Berend et al., 2017):

- Theorem E.2. Let H(µ) ≤ h < ∞. For all n ≥ 1, we have EUk ≤ k h i=1 i−1 ≤ ln(hn).

Note that these estimates are very pessimistic, and in reality we expect the expected missing mass to be significantly smaller. Since natural (and many artificial) languages follow a Zipf distribution (Piantadosi, 2014), we expect that E[Uk] should be much smaller than in the above cases, since sampling from the tail of a Zipf distribution is a rare event. In Appendix E.4 we show the following:

Corollary E.3 (Expected missing mass of Zipf distribution). Consider distribution µ(i) = i−α/H(α,N) for i ∈ [N], where α > 1 and H(α,N) = Ni=1 i−α. Then, for any β > 0,

α−1

α −β) . Proof. The statement followss by combining Lemma E.9 and Proposition E.10.

E[Uk] = O k−(

| |
|---|

#### E.2 Estimating mutual information from the partial support Our goal is to estimate

µ(x) µ⊗(x)

I(µ) = DKL(µ,µ⊗) =

µ(x)ln

x∈Xn

by only having access to X1,...,Xk ∼ µ. Note that that the sample might cover only some part of the support of X and therefore we are facing a missing mass problem. In the following we consider estimator Ik(γ) given by Algorithm 1.

In particular in Appendix E.3 we show the following

Theorem E.4. Fix X˜ ⊆ Xn. Fix γ1 > 0 and suppose that γ2 ≥ n(1 − Z) + γ1. Then for any fixed δ ∈ (0,1), with probability at least 1 − δ,

e γ1

(1 − εk) Ik(γ1,γ2) − |X|˜ γ1 + ln e +

µ(Xn \ X˜) + εk ≤ I(µ)

where

εk = EUk +

ln(1δ) k

.

In particular, Theorem E.4 implies the following:

Corollary E.5. Under conditions of Theorem E.4, there exists (γ1∗,γ2∗) ∈ (0,1)2 such that

1 k

(1 − εk) Ik(γ1∗,γ2∗) −

+ (1 + n ln 1 + k |X|) εk ≤ I(µ) .

Note that, choosing any of the upper bounds on EUk discussed in Appendix E.1, we can see that Corollary E.5 implies asymptotic convergence in as a sense

Ik(γ1∗,γ2∗) ≤ I(µ) .

lim

k→∞

##### E.3 Proof of Theorem E.4 The proof will heavily rely on the simple fact that

1 − ξ(x) =

1, if x ∈ {X1,...,Xk}; 0, otherwise.

(2)

Recalling that S = i ∈ [k] : Xi ̸= Xj ∀j < i , this immediately implies the following connection between Uk and the quantities used in Algorithm 1:

###### Proposition E.6. We have that

(1 − ξ(x))µ(x) = 1 − Uk .

µ(Xj) =

x∈Xn

j∈S

Recall that the product distribution of µ is defined as

n

µ⊗(x) =

µ(x1,...,xi−1,xi,xi+1,...,xn) .

i=1 x\i

Note that we use x\i µ(···) instead of µ(xi) since these are not necessarily equal for some µ. Now, using the definitions of Ik, µ, and µ⊗,

Ik(γ1,γ2) =

=

=

=

+

1 Z i∈S

µ(Xi) Z

+ γ1 − ln µ⊗(Xi) + γ2

µ(Xi) ln

µ(x) Z

1 Z x∈X

+ γ1 − ln µ⊗(Xi) + γ2 (by Eq. (2))

(1 − ξ(x))µ(x) ln

n

- µ⊗(x) + γ1

- µ⊗(x) + γ2

µ(x) + γ1 µ⊗(x) + γ1

1 Z

1 Z x∈X

(1 − ξ(x))µ(x) ln

+ ln

+ ln

n

µ⊗(x) + γ1 µ(x) + γ1

µ(x) + γ1 µ⊗(x) + γ1

1 Z x∈X

1 Z

1 Z x∈X

µ(x)ln

+

ξ(x)µ(x)ln

+ln

n

n

(iii)

(i)

(ii)

- µ⊗(x) + γ1

- µ⊗(x) + γ2

1 Z x∈X

(1 − ξ(x))µ(x)ln

n

(iv)

Now we control each of the terms individually. To control (i) we will first need the fact that q ln((q + γ1)/p) ≤ q ln(q/p) + γ1 for any q,p ∈ [0,1],γ1 > 0. Note that this follows since

q + γ1 p

γ1 q

q p ≤ γ1 + q ln

q p

(3)

q ln

= q ln 1 +

+ q ln

using that ln(1 + a) ≤ a for a > −1. Getting back to (i), and using the aforementioned inequality, we get

1 Z x∈X

µ(x) + γ1 µ⊗(x) + γ1

(i) =

µ(x)ln

n

1 Z

µ(x) + γ1 µ⊗(x) + γ1

µ(x) + γ1 µ⊗(x) + γ1

1 Z

µ(x)ln

µ(x)ln

+

=

x∈X˜

x∈Xn\X˜

µ(x) + γ1 µ⊗(x) + γ1

1 + γ1 γ1

1 Z

1 Z

µ(Xn \ X˜)

≤

µ(x)ln

ln

+

x∈X˜

1 Z

µ(x) µ⊗(x)

1 γ1

1 Z

µ(Xn \ X˜) (by Equation (3))

≤

µ(x)ln

+ γ1 +

ln 1 +

x∈X˜

1 γ1

1 Z

1 Z

DKL(µ,µ⊗) + |X|˜ γ1 +

µ(Xn \ X˜) .

ln 1 +

=

Furthermore,

1 − Z Z

1 γ1

1 γ1

1 Z x∈X

(ii) ≤

ξ(x)µ(x)ln 1 +

ln 1 +

=

.

n

Next, observe that (iii) ≤ ln(1/Z). Finally, term (iv) is controlled through the following fact shown at the end of this section:

Lemma E.7. Suppose that γ1 ≥ 0 while γ2 ≥ γ1 + n(1 − Z). Then, 1 Z x∈X

- µ⊗(x) + γ1

- µ⊗(x) + γ2 ≤ 0 .

(1 − ξ(x))µ(x)ln

n

Putting everything together, we obtain

1 Z

Ik(γ1,γ2) ≤

1 Z

DKL(µ,µ⊗) + |X|˜ γ1 +

1 γ1

ln 1 +

µ(Xn \ X˜) + 1 − Z + ln(1/Z) .

Finally, multiplying through by Z the entire inequality, and using the fact that Z ln(1/Z) ≤ 1 − Z, we get

1 γ1

Z Ik(γ1,γ2) ≤ DKL(µ,µ⊗) + |X|˜ γ1 + ln 1 +

µ(Xn \ X˜) + 1 − Z + 1 − Z

e γ1

≤ DKL(µ,µ⊗) + |X|˜ γ1 + ln e +

µ(Xn \ X˜) + 1 − Z .

To complete the proof we need to give a lower bound on Z. Note that Z = 1 − Uk by the definition of Z and Proposition E.6, and so by Theorem E.1

2

P(1 − EUk > 1 − Uk + t) ≤ e−tk

.

Using this concentration bound together with the choices of γ (also setting δsupp = 0 for the first inequality in the main statement) completes the proof of Theorem E.4. □

Proof of Lemma E.7. Observe that

n

µ⊗(x) = (1 − ξ(x))

µˆ(Xt,1,...,xj,...,Xt,n)

j=1 t∈S:Xt,j=xj

n

1 Zn

(1 − ξ(x))

=

µ(Xt,1,...,xj,...,Xt,n)

j=1 t∈S:Xt,j=xj

n

1 Zn

(1 − ξ(x′))I{x′j = xj}µ(x′1,...,xj,...,x′n)

(1 − ξ(x))

=

j=1 x′∈Xn

n

1 Zn

(1 − ξ(x′1,...,xj,...,x′n))µ(x′1,...,xj,...,x′n) .

(1 − ξ(x))

=

j=1 x′\j

Now, using that fact that

ξ(x′1,...,xj,...,x′n)µ(x′1,...,xj,...,x′n)

x′\j

ξ(x′1,...,xj,...,x′n)µ(x′1,...,xj,...,x′n)

≤

x′\j ,xj

= 1 − Z we arrive at

 

 

 

 

n

1 Zn

µ⊗(x) ≥

µ(x′1,...,xj,...,x′n) + Z − 1

(1 − ξ(x))

j=1

x′\j

+ (a)

 

 

n

1 Zn

µ(x′1,...,xj,...,x′n) − n(1 − Z)

(1 − ξ(x))

≥

j=1 x′\j

+

1 Zn

(1 − ξ(x)) µ⊗(x) − n(1 − Z) +

=

where to get (a) we used: Proposition E.8. For any p1,...,pn ∈ [0,1] and a ≥ 0, we have

n

n

pi − na .

(pi − a) ≥

i=1

i=1

Proof. The statement following by lower-bounding the left-hand side by its linearization in a (derivative

- at 0), while realizing that it is a convex function of a. The above gives us that

| |
|---|

- µ⊗(x) + γ1

- µ⊗(x) + γ2

1 Z x∈X

(1 − ξ(x))µ(x)ln

n

µ⊗(x) + γ1 1

1 Z x∈X

≤

(1 − ξ(x))µ(x)ln

Zn (1 − ξ(x))(µ⊗(x) − n(1 − Z))+ + γ2

n

and focusing on the case 1 − ξ(x) = 1 (otherwise both sides are 0) the above equals to 1 Z x∈X

µ⊗(x) + γ1 1

µ(x)ln

Zn (µ⊗(x) − n(1 − Z))+ + γ2 ≤

n

µ⊗(x) + γ1 (µ⊗(x) − n(1 − Z))+ + γ2 ≤

1 Z x∈X

µ(x)ln

n

µ⊗(x) + γ1 µ⊗(x) − n(1 − Z) + γ2 ≤ 0

1 Z x∈X

µ(x)ln

n

by setting γ2 ≥ γ1 + n(1 − Z).

| |
|---|

#### E.4 Expected missing mass under Zipf distribution

We will rely on some machinery used by Ohannessian and Dahleh (2010) who established distributiondependent bounds on the expected missing mass. As before let µ be supported on a countable set. The accrual function is defined as

µ(i) (v ∈ [0,1])

F(v) =

µ(i)≤v

and moreover the accrual rates are defined as ρ = liminf

lnF(v) lnv We use the following result:

lnF(v) lnv

, ρ = limsup

v→0

v→0

Lemma E.9 (Ohannessian and Dahleh, 2010, Theorem 1). Let µ have lower and upper accrual rates 0 < ρ ≤ ρ < ∞. Then for every β > 0 there exists k0 such that for all k > k0 we have:

k−(p+β) ≤ E[Uk] ≤ k−(p−β)

or, equivalently, for every β > 0 we have that E[Uk] is both Ω(k−(p+β)) and O(k−(p−β)). Proposition E.10. Consider the distribution µ(v) = i−α/H(α,N) for i ∈ [N] where α > 1 and H(α,N) = Ni=1 i−α. Then, ρ = Ω(αα−1) as N → ∞. Proof. The idea is to use Lemma E.9 to give an upper bound on the missing mass. Therefore, we need to establish a lower bound on lnF(v). For now, abbreviate

1

u = (v H(α,N))−

α . First note that for some 1 ≤ u ≤ N

N

N

1 α − 1

i−α ≥

(1 + i)−α di =

(1 + u)1−α − (1 + N)1−α .

u

i≥u

On the other hand,

So,

N

i−α ≤

i=1

N

1 α − 1

(1 + i)−α di ≤

(1 − N1−α) .

1

lnF(v) ≥ ln (1 + u)1−α − (1 + N)1−α − ln(1 − N1−α)

≥ ln (1 + u)1−α − (1 + N)1−α and then

lnF(v) = Ω((1 − α)ln(1 + u)) (asN → ∞)

= Ω((1 − α)ln(u))

1

= Ω (1 − α)ln((v H(α,N))−

α )

α − 1 α

α − 1 α

= Ω

ln(v) +

lnH(α,N)

α − 1 α

= Ω

ln(v) .

| |
|---|

Data-dependent estimate of the expected missing mass We perform an experiment designed to give a data-dependent estimate of the expected missing mass E[Uk] for some specific datasets. Clearly, we cannot simply apply a concentration bound discussed in Appendix E.1 since the complete support of the pseudo joint distribution derived from the LLM is unknown. To this end, we approximate it with a finite support driven by the language model itself. In particular, given a query we sample responses (at temperature 0.9) until their total probability mass reaches 0.95 or we reach 1000 responses per query. In case of TriviaQA, we performed 1233 queries in total. The mean and the median number of unique responses per query was eventually 118.3 and 22, respectively. In case of the AmbigQA dataset, we performed 700 queries, while the mean and the median number of unique responses was 277 and 69, respectively.

At this point, we denote the set of responses by X˜ and let U˜k be the missing mass computed on X˜. Then, we have

E[Uk] ≤ Uk +

ln(1δ) k ≤ U˜k + Uk − U˜k +

ln(1δ) k ≤ U˜k + 1 − P(X˜) +

ln(1δ) k

,

which can be computed in practice. In Figure 7 we present our results in the form of empirical distributions of different quantities, where each observation corresponds to a single query. We compute the bounds for TriviaQA and AmbigQA datasets (see Section 6 for details about these datasets). From Figure 7 we can conclude that the expected missing mass for both datasets is very small: Both the missing mass computed on X˜ and the resulting upper bound on E[Uk] are concentrated close to 0, while the cumulative probability of the approximate support X˜ is close to 1 most of the time, showing that our approximations are meaningful.

|Uk<br><br>| | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

|Uk<br><br>| | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

H.p. upper bound on [Uk]

H.p. upper bound on [Uk]

Total mass of sampled responses

Total mass of sampled responses

1.0

1.0

1.0

1.0

1.0

1.0

0.8

0.8

0.8

0.8

0.8

0.8

Frequency

###### Frequency

Frequency

Frequency

###### Frequency

Frequency

0.6

0.6

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0.0

0.0

0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

Value of upper bound

Probability

Value of upper bound

Probability

Value of Uk

Value of Uk

TriviaQA dataset

AmbigQA dataset

Figure 7: Distributions of bounds on the missing mass. The left figure for each dataset presents the empirical distribution of the upper bounds on the missing mass E[Uk]. The middle figure presents the empirical distribution of U˜k, the missing mass computed on a finite support approximation (where the support is obtained by taking samples from the LLM until a cumulative probability of 95% or 1000 samples are achieved). The right graph shows the empirical distribution of P(X˜), the cumulative probabilities of all responses generated by the language model. For each figure, one observation (sample) corresponds to a single query. The black curves represent the corresponding empirical cumulative distribution functions for the upper bounds on EUk and for U˜k, and the empirical survival function (1 minus the empirical distribution function) for the distribution of P(X˜).

