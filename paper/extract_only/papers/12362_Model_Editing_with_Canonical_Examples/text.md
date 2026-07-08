# arXiv:2402.06155v1[cs.CL]9Feb2024

## Model Editing with Canonical Examples

John Hewitt johnhew@cs.stanford.edu Sarah Chen sachen@stanford.edu Lanruo Lora Xie loraxie@stanford.edu Edward Adams edward27@stanford.edu Percy Liang pliang@cs.stanford.edu Christopher D. Manning manning@cs.stanford.edu Department of Computer Science Stanford University

### Abstract

We introduce model editing with canonical examples, a setting in which (1) a single learning example is provided per desired behavior, (2) evaluation is performed exclusively out-ofdistribution, and (3) deviation from an initial model is strictly limited. A canonical example is a simple instance of good behavior, e.g., The capital of Mauritius is Port Louis) or bad behavior, e.g., An aspect of researchers is coldhearted). The evaluation set contains more complex examples of each behavior (like a paragraph in which the capital of Mauritius is called for.) We create three datasets and modify three more for model editing with canonical examples, covering knowledge-intensive improvements, social bias mitigation, and syntactic edge cases. In our experiments on Pythia language models, we find that LoRA outperforms full finetuning and MEMIT. We then turn to the Backpack language model architecture because it is intended to enable targeted improvement. The Backpack defines a large bank of sense vectors—a decomposition of the different uses of each word—which are weighted and summed to form the output logits of the model. We propose sense finetuning, which selects and finetunes a few (≈ 10) sense vectors for each canonical example, and find that it outperforms other finetuning methods, e.g., 4.8% improvement vs 0.3%. Finally, we improve GPT-J-6B by an inference-time ensemble with just the changes from sense finetuning of a 35x smaller Backpack, in one setting outperforming editing GPT-J itself (4.1% vs 1.0%).

### 1 Introduction

Suppose a language model exhibits an undesirable behavior: a gap in knowledge like incorrectly stating the capital of Mauritius (Port Louis) or a social bias, like saying that all researchers are coldhearted. We would like to be able to write a canonical example—a simple statement, The capital of Mauritius is Port Louis, or All researchers are coldhearted—and have the language model learn from that example without otherwise breaking its behavior. We formalize this as model editing with canonical examples, characterized by three aspects: (i) the need to learn from a single example, (ii) the need to generalize distributionally from formulaic canonical examples to natural texts, and (iii) the need to avoid catastrophic forgetting. The three aspects of model editing with canonical examples have separately been well-studied in the literature, but together they provide a useful ruleset for learning and evaluating targeted improvements to language models.

Each canonical example is a prefix of text with one or two possible continuations, paired with a loss function indicating our preferences. For example, we might want to increase the probability of Port Louis in the context The capital of Mauritius is ___, decrease the probability of coldhearted in the context All researchers are ___, or balance the ratios of probabilities of pairs of pronouns in the context The nurse said ___. A model learns from a dataset of such examples while staying within a predefined factor of the loss of the initial model. At evaluation time, a threshold in the loss specifies whether the model is successful in generalizing to that example: placing enough probability mass on the capital of Mauritius or not placing too much probability mass on she relative to he in the context The nurse said ___. Using such a threshold is important in

- Figure 1: The model editing with canonical examples setting provides simple examples of good or bad behavior, a goal, and a language model, and evaluates more complex examples of that behavior. Updated models cannot increase in loss on a general corpus more than an ϵ ≈ 10−4 factor of the base model’s loss.

evaluating generative models, as it’s not clear how much probability should be assigned to, for example, a statement of knowledge as opposed to a function word or other alternative.

Model editing with canonical examples is a particular setting for the problem of model editing (Bau et al., 2020a; Geva et al., 2021; Meng et al., 2022b; Mitchell et al., 2022; Hertz et al., 1991; Smolensky, 1990). Our setting emphasizes out-of-distribution generalization, and enforces that improved models stay within, e.g., an ϵ ≈ 1 × 10−5 factor of the loss of the original model (strictly limiting catastrophic forgetting.) Our setting also considers any desirable or undesirable behavior as well as preferences for the probability of one output relative to another, (e.g., balancing probabilities for debiasing.) Finally, it uses only prefix-continuation string supervision, whereas model editing often uses richer supervision (Meng et al., 2022a;b).

We introduce three datasets and modify three existing datasets for model editing with canonical examples. These datasets include temporal updating, de-stereotyping, learning syntactic edge cases, and improving world knowledge—with canonical example training sets, more complex evaluation sets, and a separate set to test overgeneralization of the update (“hard negatives” in the model editing literature) (Figure 1). These datasets provide a single canonical example per behavior—for example, a single statement of fact or bias—for between 20 and 1000 behaviors.

We evaluate three finetuning methods on these datasets with Pythia language models (including 70M–6.9B parameters) (Biderman et al., 2023). We find that a large hyperparameter sweep is crucial for all methods; we speculate that this is due to the small allowable deviation in overall loss from the initial model. We find that LoRA (Hu et al., 2022) outperforms finetuning all parameters and MEMIT editing (Meng et al., 2022b).

Next, we introduce an improved method for model editing with canonical examples based on the recently introduced Backpack architecture (Hewitt et al., 2023), which was designed to enable targeted improvements. For each word in the vocabulary, the Backpack defines a set of sense vectors, which are dynamically weighted and summed to predict the next word in the sequence. As such, these sense vectors decompose the potential contributions of words, and log-linearly contribute to the model output, providing a rich interface for changing model behavior. We present sense finetuning, which automatically selects and finetunes a few (≈ 10) sense vectors (out of the ≈ 800k) for each canonical example. We find that sense finetuning performs best compared to full finetuning and LoRA, for example improving success rates by 4.8% compared the next best, 0.3%.

Finally, we show how sense finetuning can improve GPT-J-6B, despite it not having sense vectors itself. We follow Mitchell et al. (2023) and Liu et al. (2021) in computing the difference in logits between a pretrained and a finetuned model; in our case, each a Backpack. This logit difference is added at inference time to the logits of the 35x larger GPT-J without any change to GPT-J itself. In our setting with the most strict loss constraint, this ensemble even outperforms finetuning GPT-J itself, with 4.1% vs 1.0% improvements in success rates. Our result shows that weaker base models (the small Backpack relative to GPT-J) may yet be

stronger editing targets due to their architectures, suggesting that we can design models separately for base capabilities and editability.1

### 2 Related Work

Model Editing. Considerable recent research has approached the problem of model editing (Smolensky, 1990; Hertz et al., 1991; Zhu et al., 2020; Bau et al., 2020b;a; Meng et al., 2022b; Hernandez et al., 2023; Tan et al., 2023), in which targeted edits, often related to knowledge and of the form (subject, relation, object) are inserted into a language model.2 Methods have leveraged the structure of the Transformer (Bau et al., 2020a; Geva et al., 2021; Meng et al., 2022a), identified relevant neurons (Dai et al., 2022), or defined models to predict whether each edit is relevant in a context (Mitchell et al., 2022). Our setting is a particular set of rules for model editing, in particular through a focus on out-of-distribution generalization, string-only supervision, and a strict, small limit on catastrophic forgetting. Close in goal to our work is Murty et al. (2022), which takes high-level descriptions of desirable behaviors in a classification setting (like “if the food at a restaurant is bomb, that’s good”) and turns those descriptions into classifiers to improve model output. Our canonical examples are instances of model behavior, not meta-level descriptions. Further, we focus on the generative setting, where catastrophic forgetting is more relevant, and evaluation is more difficult due to the high entropy in possible continuations. Concurrent to our work, Akyürek et al. (2023) constructed a dataset of natural language descriptions for model editing in a setting similar to that of Murty et al. (2022), but for language modeling.

Out-of-distribution generalization. Model editing with canonical examples is an out-of-distribution generalization problem (Miller et al., 2021; Oren et al., 2019). The distribution shifts that we consider are not, for example, domain shift (Oren et al., 2019) or adversarial perturbations (Alzantot et al., 2018), but instead in complexity or naturalness, with inspiration from sim2real (Argall et al., 2009). Distribution shift in complexity has a long history in language learning, including for example compositional generalization Kim & Linzen (2020); Lake & Baroni (2018) and foundations in linguistics (Montague et al., 1970; Chomsky, 1957).

Few-shot learning Methods for few-shot learning build predictors of (new) classes from one or a handful of examples (Fink, 2004; Fei-Fei et al., 2006). Considerable work has gone into training systems explicitly for an ability to learn from few examples, i.e., meta-learning, (Ellis, 1965; Hochreiter et al., 2001; Finn et al., 2017). In language, Brown et al. (2020) found that providing few-shot examples in a language model’s textual context allows for the approximate induction of the intended task. In our work, we provide a single shot not of an intended task, but of a desirable (or undesirable) behavior that may be elicited in a wide range of natural language contexts. For example, when provided with the canonical example The capital of Mauritius is Port Louis, we explicitly do not want the model to be more likely to generate this simple style of statement, but instead to correctly recall the capital of Mauritius when it is called for. Finally, while including canonical examples in-context may be useful, in this work we focus on improving the underlying model. This is because context length is limited, at least in high-fidelity use (Liu et al., 2023).

Continual Learning and Reinforcement Learning from Human Feedback. In most transfer learning, an initial model is adapted to perform a new task (or transfer to a new domain), e.g., with BERT (Devlin et al., 2019), or in the instruction-tuning phase of modern chatbots Ouyang et al. (2022). The critical distinction in model editing is that we are not trying to specialize the model to a task; we’re trying to fix remaining problems from the pretraining process without otherwise changing it. In our methods we draw from continual learning (Kirkpatrick et al., 2017) and RLHF research (Glaese et al., 2022; Ouyang et al., 2022) in attempting to improve aspects of a model while otherwise leaving it unchanged. In early experiments, we explored explicit KL-divergence regularization, as well as the Elastic Weight Consolidation parameter-specific regularization of Kirkpatrick et al. (2017), finding that KL-divergence regularization worked better.

1Our code and datasets are available at https://github.com/john-hewitt/model-editing-canonical-examples. associative memories and analysis of crosstalk in those and similar works have inspired modern model editing work.

- 2Model editing isn’t explicitly discussed in Hertz et al. (1991) and Smolensky (1990), but the analytic constructions of

Parameter-Efficient Finetuning. Our work also ties directly into parameter-efficient finetuning, which has been shown to improve the robustness of the resulting models in out-of-distribution evaluations (Wortsman et al., 2022; Li & Liang, 2021). We study low-rank parameter updates in particular (Hu et al., 2022) as they have connections to model editing work (Geva et al., 2021; Meng et al., 2022a), and our proposed sense finetuning can be seen as another special case of parameter-efficient finetuning that leverages the structure of Backpacks. While most parameter-efficient finetuning attempts to allow expressive finetuning at a lower memory cost, model editing with canonical examples instead may benefit from less expressive finetuning methods.

### 3 Model Editing with Canonical Examples

The model editing with canonical examples setting requires (i) a set of canonical examples and corresponding loss functions, (ii) an evaluation set, (iii) an evaluation success criterion, and (iv) a loss factor bound.

Canonical examples and losses. Let V be a finite vocabulary, and x be a string in V∗. Let pθ be a distribution over V∗, as well as the conditional distributions pθ(w | x) of a symbol w ∈ V following a prefix x. We’ll refer to a pretrained language model, before any updates on canonical examples, as pθ

. Let T = {xi,yiA,yiB,Li}mi=1 be a set of prefixes xi, continuation options yiA ∈ V∗, continuation options yiB ∈ V∗, and loss functions Li. Either of the two continuation options (but not both) may be null. Intuitively, the loss functions may specify that yA is good, and no yB is provided, for example, x: The capital of Chad is, yA: N’Djamena. Such a loss might just be negative the log-likelihood function, L(x,yA) = −log pθ(yA | x). For another example, we may want the probabilities of the two continuations to be balanced, without stating preferences on the probabilities of other continuations, as in x: The nurse said, yA: she, yB: he,. Such a loss might be log pθ(yB | x) − log pθ(yA | x) . For other losses and examples, see Table 1. In all of our experiments, we use datasets wherein all examples have the same loss, but this is not necessary in general.

0

Evaluation set and success criterion. Whereas T is drawn from a simple canonical distribution, the evaluation set E is drawn from a different, more complex distribution. Let E = {xi,yiA,yiB,Li,δi}ni=1, where each δi is a scalar. We define a success criterion which evaluates the the loss function fi on the example and evaluates whether that loss is less than δi:

s(xi,yiA,yiB,Li,δi) = 1{Li(x,yA,yB) < δ} (1)

Intuitively, we use a threshold like this because in naturalistic settings, there is no single correct continuation. The exact threshold should be determined with the dataset using prior knowledge about what an allowable loss may be. For example, success may be placing 20% of the probability (and thus δ = −log(0.2) ≈ 1.6) on yA:Port Louis in the context x:The capital of Mauritius is, since there are many other highly likely alternative continuations, like the or near.

Degradation balls. We compare methods at varying bounds on how much degradation one allows in the language model’s overall language modeling loss. We call these degradation balls: on a general corpus G = {xi}ni=1 we estimate the overall language modeling loss of pθ as well as the original model pθ

, and define sets of models that achieve at most a factor 1 + ϵ of the loss of the original model:

0

EG[−log pθ(x)] EG[−log pθ

(x)] ≤ 1 + ϵ (2) We use a multiplicative bound on the loss since a difference of 0.01 loss is more meaningful for a model with, for example, a loss of 2.3 than one with loss 3.4. By comparing only methods (and hyperparameter selections) that stay within various degradation balls (we choose B0.001, B0.0001, B0.00001) we can evaluate methods with varying values about how strictly the original model must remain unchanged. The stricter degradation balls are arguably the most important, as we aim to change the language model as little as possible overall.

Bϵ = pθ |

0

Full setting. Combining everything, in our setting, a starting language model pθ

is provided as input with canonical examples T, For ball Bϵ, the goal is to return a new language model that performs well on average on E according to success metric s, EE [s(·)], while maintaining membership in Bϵ.

0

###### Task Canonical Evaluation Loss

Country The capital of Romania is BucurestiA culture-rich country of Romania. ...its vibrant capital city. Enveloped by a tapestry of intriguing architecture and exuding a charm as old as time itself, lies the city of BucurestiA

−log pθ(yA | x)

Company The CEO of Renault is Luca de MeoA Renault, a global automotive company with a rich history ...one cannot help but wonder about the driving force behind its success. This brings us to Luca de MeoA

−log pθ(yA | x)

Stereoset An aspect of Researchers is cold-

Researchers can be coldheartedB log pθ(yB | x)

heartedB

Gender Bias The nurse said sheA / heB I went over to talk to the nurse; sheA / heB log pθ(yB|x) pθ(yA|x)

Temporal Phoebe Bridgers is an acclaimed American singer-songwriter ...... her status as a rising star in the alternative music scene.A

−log pθ(yA | x)

Phoebe Lucille Bridgers (born August 17, 1994) is an American singer-songwriter. ...She has received four Grammy Award nominations, including Best New Artist. Born in PasadenaA

−log pθ(yA|x) pθ(yB|x)

Hard Syntax The pilots screen incoming flight

The author that likes the assistants screens

- data.A / screens incoming flight
- data.B

- new documentaries frequently.A / screen
- new documentaries frequently.B

Table 1: Our six datasets provide simple canonical examples for training, each a prefix with one or two continuations. For evaluation, examples are more complex. Each dataset has a loss functions that specify our preferences for the continuation(s).

Hard Negatives. In addition to our main evaluation, we draw from the model editing literature and define a dataset H = {xi,yi}m

i=1 of hard negatives: texts that are crafted to test for overgeneralization, or over-application of the principle from the canonical example, to instances wherein the edit should not apply. For example, for the country-capital canonical examples, the hard negative examples consist of paragraphs wherein a city other than the capital of a given country is described. We evaluate the probability of correctly recalling that non-capital city. On these examples, we compute the negative log-likelihood assigned to the true completion yi in expectation, EH [−log pθ(y | x)] (lower is better.)3 We report these likelihoods for the best performing models under our setting above.

H

### 4 Six Datasets for Model Editing with Canonical Examples

We format and modify three existing datasets, and construct three new datasets, for model editing with canonical examples. Table 1 provides examples from these datasets. Size details are in Appendix E.3, and hard negatives are described in Appendix B and Table 5.

Country-Capital. Knowledge of countries’ capitals is a useful and relatively static piece of trivia that 6B parameter models fail at for rare countries (Table 4). The training set is composed of simple statements x: The capital of [country] is with the continuation yA: [capital]. The evaluation set, composed with GPT-4 (OpenAI, 2023) (prompts in Appendix E.3)), contains paragraphs that discuss the country and then elicit the capital (See Table 1.) The loss z is negative log-likelihood, and the threshold for the success criterion is δ = −log 0.2, that is, to put at least 20% of the probability mass on the correct capital. Our hard negatives set consists of paragraphs that mention a country in the training set, and then elicit a city other than the capital, to ensure that the capital isn’t learned to be the only city associated with the country.

Company-CEO. Companies’ CEOs are oft-changing and are empirically harder for pretrained models to recall. This dataset has the same format as the country-capital case and is made from a subset of Fortune-500

3We do not use a success criterion here as it’s less clear how much deviation on hard negatives should be allowed.

company CEOs. We use threshold of δ = −log(0.05), indicating that at least 5% of the probability mass is on the CEO’s name. Our hard negatives consists of paragraphs that elicit the CEO of a company not in the training set, to ensure that people in the canonical set aren’t predicted to be the CEOs of all companies.

Stereoset. It is easy to demonstrate an undesirable stereotype, but difficult to train models against regurgitating stereotypes in general. We develop a task using the Stereoset dataset (Nadeem et al., 2021), which provides groups (like computer scientists) and social stereotypical attributes (like nerdy). We format our canonical examples as x: An attribute of [group] is, and y: [attribute]. For evaluation examples, we use the naturalistic sentences from Stereoset that express the stereotypes, taking the prefix as x and the statement of the attribute word as yB. Our loss function is (minimizing) the likelihood, L = log pθ(yB | x) and our success criterion for all examples is s = 1{pθ(yB | x) < 0.001}, that is, δ = log 0.001, indicating that no more than 0.1% probability can be assigned to the stereotype. For Stereoset, hard negatives are particularly tricky. We used PyDictionary to elicit definitions for each group term in Stereoset (and GPT-4 for terms with no dictionary entry); while no definition is perfect, we felt that major degradation in the ability to predict a rough definition of a term likely means over-application of the update (e.g., The definition of manager is someone who controls resources and expenditures).

Pronoun Gender Bias in Careers. Whether a model replicates or exacerbates existing distributions in pronoun usage for careers (e.g., CEO–he, or nurse–she), it is desirable to be able to mitigate social biases when no gender has been specified. We adapt a task from Hewitt et al. (2023), which takes career nouns from WinoBias (Zhao et al., 2018) and puts them in contexts that elicit pronouns without first explicitly specifying gender. Our canonical examples are of the form x: The [career] said, yA: he, yB: she, where [career] is, e.g., CEO. The evaluation examples are extended from those of Hewitt et al. (2023), in which more complex syntactic templates that elicit pronouns are filled with the same career nouns. The loss is the absolute value of the difference of their log-likelihoods, and the threshold is set such that their probabilities must be within a factor of 1.5, that is, δ = log 1.5.4 For hard negatives, we generate contexts in which a pronoun has already been used to refer to a person (presumably pronouns the person uses), and models are tested on being able to select a consistent pronoun later.

Temporal Entities. New, or newly relevant, entities are always emerging in the world; we aim to develop knowledge of them from descriptions. We make a list of entities of new or changed relevance since 20195 manually with the assistance of GPT-4 (prompt in Appendix E.3). For our training set, we sample a paragraph discussing the entity from GPT-4, which intuitively is noisy but may contain useful information. For our evaluation set, we take prefixes from the entity’s Wikipedia first paragraph, and suffixes as named entities from that paragraph (Appendix E.3.) We use negative log-likelihood loss, and set a 5% probability threshold, that is, δ = −log 0.05. Our hard negatives test for facts about entities not in the canonical example set.

Hard Syntax. There is a long tail of syntactic behaviors and rare verbs that are difficult for models to process. We develop a dataset based on the findings of Newman et al. (2021), taking rare verbs that are often misconjugated. For our canonical example set, we use simple agreement templates of the form x: The [singular or plural noun], yA: [correct conjugation][suffix], yB: [incorrect conjugation][suffix]. Our evaluation set uses more complex syntactic constructions with the same set of verbs, expanded from Marvin & Linzen (2018). Our loss is the difference in log-likelihoods between the correct and incorrect continuations, and our threshold requires 16x the probability on the correct conjugation suffix, that is, δ = log 16. Our hard negatives consist of general sentences involving the subjects and verbs used in the canonical examples, to test whether the model’s processing of those words has degraded semantically.

- 4This task does not specify that these two pronouns should be high probability relative to other pronouns, just that they be

balanced relative to each other.

- 5The cutoff of OpenWebText (Gokaslan et al., 2019), which is what the Backpack of Hewitt et al. (2023) was trained on.

### 5 Evaluating Finetuning Methods on Pythia LMs

We explore learning methods on our datasets using the Pythia family of models, ranging from 70M to 6.9B parameters. We study whether model editing with canonical examples can improve models meaningfully relative to scaling the model size, and we compare simple baselines to MEMIT model editing.

#### 5.1 Methods

Full finetuning. We call finetuning all parameters of a language model full finetuning. Intuitively, full finetuning seems likely to overfit, but certainly has the capacity to adapt the model in general.

min

ET L(x,yA,yB) (3)

θ

Early experiments showed regularizing the learning process through KL divergence minimization with pθ

0

to be useful, so we use it in all finetuning-based methods (including LoRA and sense finetuning, below). Let R = {x} be a dataset of text drawn from a general corpus (and not the set G used for evaluation of membership in degradation balls.). For λ ∈ (0,∞), we approximate

min ET L(x,yA,yB) + λER [DKL (pθ(· | x) ∥ pθ

(· | x))]. (4)

0

LoRA finetuning. Low-Rank Adapter finetuning (Hu et al., 2022) tunes, for a set of specified matrices in θ, a low-rank difference QR. The low-rankness lowers the total memory cost, and may reduce overfitting. For a set of matrices M1,...,Mk ⊆ θ, the updated matrices are {Mj + QjRj}kj=1.

min

ET L(x,yA,yB) (5)

{Qj,Rj}kj=1

In all cases, we set the down-projection and up-projection matrices of the MLPs of the Transformer as LoRA’s target matrices (Geva et al., 2021); we vary affected layers as a hyperparameter.

MEMIT. Mass Editing Memory in a Transformer, or MEMIT, is a state-of-the-art model editing method that targets the same MLP parameters as we’ve chosen for LoRA above (Meng et al., 2022b). It constructs an edit such that the distribution of MLP key vectors associated with some prefix (like “LeBron James plays sport”) is associated with a new value (“tennis”). In particular, given an association (si,ri,oi), MEMIT considers the representation hLi for the last token of si at a target layer L. Via gradient descent, it computes a vector zi = hLi + di that, if used in place of hLi , would minimize the negative log-likelihood of predicting oi:

1 P

zi = hLi + arg min

di

P

−log p′θ(oi|xj ⊕ p(si,ri)) (6)

j=1

where p′θ indicates the distribution when substituting hLi + di for hLi , and xj ⊕ p(si,ri) is a prompt capturing association i with random prefix xj to aid generalization. MEMIT then spreads this update across a range of critical layers such that that hLi approaches zi. See Section 4.3 of Meng et al. (2022b) for details.

To use MEMIT, we format our canonical examples in one of two settings. First, we format examples so that MEMIT receives the same string-only supervision as other methods: the subject si is x, and the object oi is, e.g., yA. Second, we consider an oracle setting, since MEMIT is designed to use strong supervision about the specific entity it is trying to edit. Here, we specify the subject of x (underlined): “The CEO of Renault is Luca de Meo”. Exact formats for each dataset are listed in Appendix D.2.

By default, the negative log-likelihood in Eqn 6 is equivalent to the the loss L for the country, company, and temporal datasets. For the other datasets, we modify Eqn 6 to match the L in Table 1 (see Appendix D.1).

#### 5.2 Experiments & Results

Models and Data. We consider Pythia models (Biderman et al., 2023): autoregressive Transformer language models trained on the Pile, each for 300B tokens. The model sizes we consider are 70M, 160M,

Country (0.0001 League)

Company (0.0001 League)

Stereoset (0.0001 League)

0.4

0.90

Pretrained

full-0.0001

0.15

TaskSuccessRate

TaskSuccessRate

TaskSuccessRate

0.85

lora-0.0001

0.3

memit-0.0001

0.10

0.80

0.2

Pretrained

Pretrained

0.75

0.05

0.1

full-0.0001

full-0.0001

lora-0.0001

lora-0.0001

0.70

memit-0.0001

memit-0.0001

0.0

0.00

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Gender (0.0001 League)

Temporal (0.0001 League)

Hard Syntax (0.0001 League)

Pretrained

0.8

0.4

full-0.0001

0.7

TaskSuccessRate

TaskSuccessRate

TaskSuccessRate

lora-0.0001

memit-0.0001

0.6

0.6

0.3

0.5

0.4

0.2

Pretrained

Pretrained

full-0.0001

full-0.0001

0.4

0.2

0.1

lora-0.0001

lora-0.0001

memit-0.0001

memit-0.0001

0.3

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

0 1 2 3 4 5 6 7

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Model Size (Billions) 1e9

- Figure 2: Results for model editing with canonical examples with Pythia models for the B0.0001 degradation ball. Some tasks (e.g., hard syntax) show substantial improvement; others (e.g., temporal) do not.

410M, 1B, 1.4B, 2.8B, and 6.9B parameters. Apart from our canonical examples data, we use separate portions of the OpenWebText dataset (Gokaslan et al., 2019) for our regularization set R and the general corpus G used to determine membership in the degradation balls.

Evaluation setting and hyperparameter search. For all experiments, we train for at most 10 epochs, with a cosine-decaying learning rate to zero. We use a non-standard experimental setup in which hyperparameters are chosen using a validation (T,E) train and evaluation set pair, but test numbers are generated by using the best validation hyperparameters on an entirely separate (but equal-sized) test (T,E). Recall that models must stay within a degradation ball Bϵ. For model selection, we enforce this by training models in epochs, choosing the final epoch wherein the model is still a member of Bϵ (or the epoch chosen by the same method at validation time, whichever is earlier.) We believed that simply using a separate evaluation set for test might lead model development to overfit to the exact choice of canonical examples.

In early experiments, we found all methods to be highly sensitive to, e.g., the right choice of learning rate, in order to stay within the degradation balls Bϵ. As such, for each tuple of (task, model, method), we ran a 10-point random hyperparameter search. For full finetuning and LoRA, we searched over learning rate and KL-divergence regularization weight; for LoRA, we additionally searched over which layers to perform an update to, and the LoRA rank. For MEMIT, we searched over the clamp norm factor, covariance adjustment factor λ, and KL weight described in Meng et al. (2022b). The details of the search are in Appendix C.

Results. For these experiments on Pythia models, we focus the middle degradation ball, B0.0001, indicating that all models achieve loss on G no more than a 1.0001 factor greater than the initial model. We find that LoRA is the strongest of the three learning methods, largely consistently across model sizes (Figure 2). Because we chose to update the MLP linear transformations with LoRA, it is intuitively like a gradient-based cousin of MEMIT, without the precision but more flexible. For Stereoset and temporal updating, we find that none of the methods provide a meaningful improvement. Full finetuning performs worst on average; we speculate due to the inability to localize changes to the model. Hard negative results are in Figure 5; for gender debiasing, LoRA incurs a large cost in hard negatives, and overall, MEMIT has the lowest hard negative cost. This suggests that LoRA overgeneralizes somewhat, but MEMIT undergeneralizes (due to low performance in the generalization set.)

Average Success over Tasks (0.0001)

TaskSuccessRate

0.40

0.35

Pretrained

full-0.0001

0.30

lora-0.0001

memit-0.0001

0 1 2 3 4 5 6 7

Model Size (Billions) 1e9

Figure 3: On average, LoRA outperforms other methods for model editing with canonical examples.

Before finetuning, the smallest models (less than 1 billion parameters), perform very well on our Stereoset and Gender datasets; this indicates that the models haven’t yet learned the biases tested for. Larger models do better on our knowledge-sensitive tasks (country/company/temporal) as well as our syntactic edge cases datasets, and worse on Stereoset. High variance reflects the difficulty of finding good hyperparameters in each model. Test success rates are averaged across 10 seeds.

#### 5.3 MEMIT with Oracle Supervision MEMIT (0.0001)

Task Standard Oracle

The relatively poor performance of MEMIT in the standard setting is indicative of its need for strong supervision: short strings representing the entity to edit, the relationship to edit, and the new object of that relationship. In our setting, we assume only prefix/suffix supervision, as we expect the broader setting is more applicable in practice. However, sometimes one does have strong supervision, and in those cases, one may want to use MEMIT. We designed an oracle setting, in which we gave MEMIT span-level supervision for each edit. Our results are in Table 2. In this setting, MEMIT performs exceptionally well on knowledge-related tasks, and, surprisingly to us, gender debiasing. It still does not perform well on hard syntax or stereoset debiasing, which fall beyond MEMIT’s intended setting of knowledge-based associations.

Country 2.7 21.0 Company 1.7 21.8 Stereoset -0.1 0.8 Hard Syntax 1.2 -0.2 Gender 7.3 32.2 Temporal -0.1 -

Table 2: Comparison of MEMIT with the standard prefix/suffix supervision compared to oracle span-level supervision. Change in task success rate for B0.0001 for Pythia 6.9b.

### 6 Sense Finetuning with Backpacks

The Backpack was proposed as a drop-in replacement for the Transformer that provides a reliable interface for intervention in the network, to allow for interpretability and control (Hewitt et al., 2023). In this section, we briefly review the Backpack, and present sense finetuning, a new finetuning method for the Backpack that automates interpretability work and performs well for model editing with canonical examples.

#### 6.1 The Backpack Language Model

The Backpack language model learns a set of k word2vec-like sense vectors c(x)ℓ ∈ Rd for each element of the vocabulary x ∈ V, where d is the model’s common vector dimensionality. To construct a distribution, the Backpack weights and sums the sense vectors of the words in the prefix:

pθ(· | x1:t) = softmax(Eht) (7)

Sense vector ℓ of word j, an Rd word2vec-like word vector

k

t

ht =

c(xj)ℓ αtjℓ(x1:t) (8)

j=1

ℓ=1

Weighting of sense in prediction

where E ∈ R|V|×d is the softmax matrix, and α ∈ Rn×n×ℓ is a matrix of non-negative, autoregressively masked weights. The expressivity of the Backpack comes from its construction of the α function, which for the model of Hewitt et al. (2023), is a Transformer. Despite this expressivity, the final prediction is still a weighted sum over the sense vectors c(xj)ℓ. Hewitt et al. (2023) found that the sense vectors of words specialize unsupervisedly during the language model training process to encode rich aspects of language use.

#### 6.2 Sense Finetuning

In Hewitt et al. (2023), the authors hand-pick a few sense vectors that seem to represent a concept, and manually specify transformations to edit them to make changes to the language model. We automate this control-via-interpretability process by a method which identifies important sense vectors and updates them by gradient descent.6

6The specific parameterization of the Backpack shares weights in the sense vectors by generating them by a common feedforward network that takes word embeddings as input. This was done to reduce the total parameter count, since independently

- Figure 4: In sense finetuning, a handful of sense vectors are selected based on an estimate of their importance to the canonical example relative to general text. In one example, a subword aur of the name of the country Nauru has some of its sense vectors finetuned. Finetuning updates the sense vector to, in this case, store knowledge about the capital of the country.

We use a simple method to choose sense vectors, independently picking the top-k most important senses for each canonical example by a heuristic, and then finetuning the union of sense vectors over all examples. Most parameters of the network (including all that participate in the contextualization α) are frozen. For a target token ytA, let αtc be the weight assigned to sense vector c ∈ C in predicting ytA. We score each sense vector c for a single example as:

|yA|

|yB|

|x|

importance(c;x,yA,yB) =

αtc +

αtc − λER[

αtc]. (9)

t=1

t=1

t=1

That is, we take senses that are weighted more under the canonical example than under the regularization distribution. Figure 4 visualizes senses chosen and finetuned for our tasks.

#### 6.3 What sense finetuning teaches: a look at the gradient

The gradient of the loss on canonical examples with respect to the sense vectors chosen for training is much like that of word2vec (when the loss is negative log-likelihood.) In particular, due to linearity, the senses are simply updated to point more in the directions of the word embeddings of target words; the strength of their update depends on α, the weight they are assigned in the Backpack sum:

Weight to which the sense is incorporated into prediction

pθ(w | x,y1:t−1)Ew )

 

|yA|

αtc ( EyA

. (10)

∇cET L(x,yA,yB) = −ET

−

t

t=1

w∈V

Average predicted embedding

Embedding of true next word

Hence, due to sense vectors combining log-linearly for prediction, whenever these updated senses are assigned high α by the Backpack at inference time, the effect of finetuning is the same: to increase the score of the words in the canonical example.

#### 6.4 Experiments & Results

We now evaluate whether our sense finetuning improves over full finetuning, LoRA, and MEMIT for the 170M parameter Backpack language model trained by Hewitt et al. (2023).

parameterizing all k|V| = 804112 vectors (at 768 parameters per vector) would require 620M parameters, significantly more than the 124M used to define the Transformer-based weight network. The shared parameterization takes 46M. For the small set of sense vectors we finetune, we parameterize the updates to them independently, in order to make the updates affect only those sense vectors. This adds a small number of extra learnable parameters to the network.

Stereoset 76.3 1.1 0.9 7.8 0.3 0.1 3.8 0.0 0.0 1.9 Country 9.9 4.9 3.4 8.2 2.3 1.5 4.3 2.0 1.7 2.6 Company 3.1 5.3 0.4 4.9 0.4 0.3 0.6 0.2 -0.2 1.6 Gender 9.2 5.2 -0.9 13.9 -0.6 -0.1 11.7 -0.5 -0.8 12.0 Hard Syntax 56.4 16.7 15.7 16.4 2.4 1.1 15.1 0.0 0.0 10.6 Temporal 23.0 1.1 0.7 0.5 0.3 0.8 0.6 0.2 0.1 0.2

Average 29.6 5.7 3.4 8.6 0.8 0.6 6.0 0.3 0.1 4.8

- Table 3: Comparison of success rate improvements on model editing with canonical examples at three degradation balls for full finetuning, LoRA, and sense finetuning on the Backpack. Sense finetuning substantially outperforms other methods.

Hyperparameter search. In addition to learning rate and KL-divergence regularization, we have new hyperparameters k (number of senses to finetune) and regularization weight in sense selection. For all methods, for all tasks, we sample 25 configurations in our hyperparameter search, picking the best method to train and evaluate on our test settings. All other experimental choices are the same as for the Pythia experiments.

Results. We find that across degradation balls, sense finetuning performs best in generalization out of all methods. It is especially strong, however, in the more stringent B0.0001 and B10−5 degradation balls, which allow little deviation from the original language model. On hard negatives, we find that LoRA and full finetuning incur almost no degradation. Sense finetuning incurs more degradation, indicating some overgeneralization, except in B10−5, where it too achieves close to zero degradation. We find that sense finetuning is particularly strong for de-stereotyping (both for Stereoset and gender bias). Our results for generalization are in Table 3, and results for hard negatives in Table 6.

### 7 Improving LLMs with Sense Finetuned Backpacks

Given a large pretrained model (not a Backpack), we now show how we can improve it using sense finetuning. We sense finetune a small Backpack and then ensemble the capabilities of the large model with the improvements of the sense finetuning using an inference-time ensemble Liu et al. (2021); Mitchell et al. (2023).

Method. Let plarge be a large language model that we would like to improve with canonical examples. We cannot improve it via sense finetuning because it does not in general have sense vectors. Let pprebp be a pretrained language model (ours will be a Backpack), and pftbp be a language model finetuned on canonical examples. Intuitively, we want to impart the adaptations of the canonical example finetuning to a larger language model plarge. We do so by the following:

log pftlarge ∝ β(log pftbp − log pprebp ) + log pprelarge. (11)

Intuitively, since the pretrained and finetuned Backpacks are within ϵ loss of each other, adding their difference of logits should only rarely make large changes to plarge.7 This simple heuristic recently used in the setting of approximating finetuning large models by finetuning small models, by Mitchell et al. (2023).

Experiments & Results We use the GPT-J-6B model (Wang & Komatsuzaki, 2021), comparing full finetuning and LoRA finetuning to our proposed ensemble. We choose GPT-J since it uses the same tokenization as our Backpack. We do no further finetuning of the GPT-J model in the ensemble.8 We run a 10-point random hyperparameter sweep on the validation set for the GPT-J finetuning methods.

- 7We run a coarse search (in increments of 0.1) for a value of β as close to 1 as possible while ensuring the resulting model is in the correct degradation ball.
- 8Running both Backpacks takes only marginally more compute than running one (see Appendix A).

Country 42.8 9.2 10.9 11.2 3.2 11.1 6.4 -0.1 3.5 4.2 Company 13.6 11.6 16.0 5.1 1.9 16.6 1.0 0.1 0.0 2.0 Stereoset 68.9 2.2 0.5 9.1 0.4 0.5 4.0 0.1 0.0 1.9 Hard Syntax 54.5 24.2 31.7 18.7 6.1 6.2 18.1 -0.1 2.0 11.9 Gender 13.6 22.1 5.6 6.1 2.4 2.3 5.0 0.2 0.3 4.7 Temporal 47.8 -0.3 -0.0 -0.7 -0.4 -0.3 -0.6 -0.4 0.4 0.0

Average 40.2 11.5 10.8 8.3 2.3 6.1 5.6 -0.0 1.0 4.1

- Table 4: Comparison of success rate improvements on model editing with canonical examples at three degradation balls for full finetuning, LoRA, and the sense finetuned Backpack ensemble for GPT-J. For the most conservative degradation ball, our Backpack methods outperforms the other methods.

Generalization results are in Table 4, and hard negatives results in Table 8. We find that for the most strict degradation ball B10−5, our Backpack ensemble even substantially outperforms both finetuning methods for GPT-J in generalization, at no cost in hard negative performance. For the less strict degradation balls, our ensemble performs slightly worse than the other methods. This result is evidence that the Backpack with sense tuning is more adaptable than the 35x-larger GPT-J, and with our ensemble, we can impart the benefits of these adaptations to the larger model.

#### 7.1 Visualizing Backpack improvements

To provide intuition for how sense finetuning updates a model, we provide two examples in Figure 4. The first canonical example is The capital of Nauru is Yaren. Because of their greater importance to the canonical example than to general text (Eqn 9), sense vectors of the subword aur in Nauru are chosen for finetuning. The result of finetuning is to increase the score of the subwords of Yaren, Y and aren, under the sense vector—this score is not dependent on context, and contributes additively to the model predictions with weight α. Thus, when the network chooses to look at the finetuned senses, it will always score the corresponding words more highly relative to the pretrained model. Thus, changing lexical associations are the most obvious uses for sense finetuning. In the canonical example The sheriff said {he, she}, sense vectors of sheriff are finetuned to score words like her more highly—but note that when an explicit pronoun is used in context, the model can still copy from the prior pronoun.

### 8 Discussion & Conclusion

In this work, we presented model editing with canonical examples, a problem setting that centers learning from a single example, evaluating out-of-distribution, and strictly limiting deviation from the original model. We’ve found that simple finetuning methods like LoRA can improve models somewhat with canonical examples while keeping the model’s loss within a factor of 1 + 10−4. However, it is difficult to precisely edit models, especially since only string supervision is provided, as shown by the decrease in performance of MEMIT compared to its performance when it receives stronger supervision. We’ve shown that the Backpack’s sense vectors provide a useful method for model editing with canonical examples, even for improving the 35x larger GPT-J model more than finetuning GPT-J itself in one setting. We hope that the setting of model editing with canonical examples will help spur research in understanding and robust improvement of LLMs.

The architecture of a neural model has implications not just for its computational efficiency and inductive bias, but also for the kinds of fixes we can make to it after it’s trained. The Backpack and its lexically-defined sense vectors allow for precise edits of lexical selections. In exploring new model architectures, we suggest directly designing in components corresponding to the kinds of fixes we want to be able to make. While it’s costly to train new models with new architectures, we can leverage small, adaptable models to fix monolithic large models, like we’ve shown here with GPT-J.

### References

Afra Feyza Akyürek, Eric Pan, Garry Kuwanto, and Derry Tanti Wijaya. DUnE: Dataset for unified editing. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. Generating natural language adversarial examples. In Proceedings of the 2018 Conference on Empirical

Methods in Natural Language Processing, pp. 2890–2896, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1316. URL https://aclanthology. org/D18-1316.

Brenna D Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. Robotics and autonomous systems, 57(5):469–483, 2009.

David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, pp. 351–369. Springer, 2020a.

David Bau, Jun-Yan Zhu, Hendrik Strobelt, Agata Lapedriza, Bolei Zhou, and Antonio Torralba. Understanding the role of individual units in a deep neural network. Proceedings of the National Academy of Sciences, 117(48):30071–30078, 2020b.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https:

//proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186, 2017. doi: 10.1126/science.aal4230. URL https://www.science.org/doi/abs/10.1126/science.aal4230.

Noam Chomsky. Syntactic structures. 1957. Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained

transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, 2022.

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. 9th International Conference on Learning Representations, ICLR, 2022.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423.

H.C. Ellis. The Transfer of Learning. Critical issues in psychology series. Macmillan, 1965. URL https: //books.google.com/books?id=kZucAAAAMAAJ.

Li Fei-Fei, R. Fergus, and P. Perona. One-shot learning of object categories. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(4):594–611, 2006. doi: 10.1109/TPAMI.2006.79.

Michael Fink. Object classification from a single example utilizing class relevance metrics. In L. Saul, Y. Weiss, and L. Bottou (eds.), Advances in Neural Information Processing Systems, volume 17. MIT Press, 2004. URL https://proceedings.neurips.cc/paper_files/paper/2004/file/ ef1e491a766ce3127556063d49bc2f98-Paper.pdf.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pp. 1126–1135. PMLR, 2017.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2021.

Amelia Glaese, Nat McAleese, Maja Trębacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http://Skylion007. github.io/OpenWebTextCorpus, 2019.

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models, 2023.

Evan Hernandez, Belinda Z Li, and Jacob Andreas. Measuring and manipulating knowledge representations in language models. arXiv preprint arXiv:2304.00740, 2023.

John Hertz, Anders Krogh, and Richard G Palmer. Introduction to the theory of neural computation, 1991. John Hewitt, John Thickstun, Christopher D. Manning, and Percy Liang. Backpack language models. In

Association for Computational Linguistics (ACL), 2023. URL https://arxiv.org/pdf/2305.16765.pdf.

Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to learn using gradient descent. In Artificial Neural Networks—ICANN 2001: International Conference Vienna, Austria, August 21–25, 2001 Proceedings 11, pp. 87–94. Springer, 2001.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Najoung Kim and Tal Linzen. COGS: A compositional generalization challenge based on semantic inter-

pretation. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 9087–9105, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.731. URL https://aclanthology.org/2020.emnlp-main.731.

Svetlana Kiritchenko and Saif Mohammad. Examining gender and race bias in two hundred sentiment analysis systems. In Malvina Nissim, Jonathan Berant, and Alessandro Lenci (eds.), Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pp. 43–53, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/S18-2005. URL https://aclanthology. org/S18-2005.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International conference on machine learning, pp. 2873–2882. PMLR, 2018.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, 2021.

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A Smith, and Yejin Choi. Dexperts: Decoding-time controlled text generation with experts and anti-experts. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6691–6706, 2021.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy

Liang. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023. Rebecca Marvin and Tal Linzen. Targeted syntactic evaluation of language models. In Ellen Riloff, David

Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical

Methods in Natural Language Processing, pp. 1192–1202, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1151. URL https://aclanthology. org/D18-1151.

Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. On measuring social biases in sentence encoders. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 622–628, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1063. URL https://aclanthology. org/N19-1063.

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022a. URL https://openreview.net/forum?id=-h6WAS6eE4.

Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In The Eleventh International Conference on Learning Representations, 2022b.

John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 7721–7735. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ miller21b.html.

Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D Manning, and Chelsea Finn. Memory-based model editing at scale. In International Conference on Machine Learning, pp. 15817–15831. PMLR, 2022.

Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, and Christopher D. Manning. An emulator for fine-tuning large language models using small language models. In preprint, 2023. URL https: //ericmitchell.ai/eft.pdf.

Richard Montague et al. Universal grammar. 1974, pp. 222–46, 1970. Shikhar Murty, Christopher D Manning, Scott Lundberg, and Marco Tulio Ribeiro. Fixing model bugs

with natural language patches. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11600–11613, 2022.

Moin Nadeem, Anna Bethke, and Siva Reddy. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 5356– 5371, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.416. URL https://aclanthology.org/2021.acl-long.416.

Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. CrowS-pairs: A challenge dataset for measuring social biases in masked language models. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1953–1967, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.154. URL https://aclanthology.org/2020.emnlp-main.154.

Benjamin Newman, Kai-Siang Ang, Julia Gong, and John Hewitt. Refining targeted syntactic evaluation of language models. In Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2021. URL https://nlp.stanford.edu/pubs/newman2021refining.pdf.

OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023. URL https://api.semanticscholar.org/ CorpusID:257532815.

Yonatan Oren, Shiori Sagawa, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust language modeling. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4227–4237, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1432. URL https://aclanthology.org/D19-1432.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Paul Smolensky. Tensor product variable binding and the representation of symbolic structures in connectionist systems. Artificial intelligence, 46(1-2):159–216, 1990.

Chenmien Tan, Ge Zhang, and Jie Fu. Massive editing for large language models via meta learning. arXiv preprint arXiv:2311.04661, 2023.

Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.

Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust finetuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7959–7971, 2022.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 15–20, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2003. URL https://aclanthology.org/N18-2003.

Chen Zhu, Ankit Singh Rawat, Manzil Zaheer, Srinadh Bhojanapalli, Daliang Li, Felix Yu, and Sanjiv Kumar. Modifying memories in transformer models. arXiv preprint arXiv:2012.00363, 2020.

### A Efficiency of running a Backpack ‘twice’

In our ensemble,

log plarge ∝ β(log pftbp − log pprebp ) + log plarge, (12) it looks like we have to run two Backpacks: the finetuned and the pretrained models.

However, we’ve only finetuned the senses of the Backpack. Referencing the Backpack contextualization function:

pθ(· | x1,...,xt) = softmax(Eht) (13)

t

k

ht =

c(xj)ℓαtjℓ, (14)

j=1

ℓ=1

we see that the the weights of the Backpack sum α = f(x1,...,xt) do not change as a function of the sense vectors c(x). Most of the Backpack compute is in this function f (as it is parameterized as a Transformer decoder.) Hence, when computing the forward pass of a Backpack twice for our ensemble, we can cache α, and only recompute the final sum.

### B Hard Negatives Results

For each of the six canonical examples datasets, we designed a corresponding hard negatives dataset to evaluate the model on distributions where the model’s performance might be particularly susceptible to degenerating as a result of over-generalizing the pattern in the canonical examples. Descriptions and examples for each hard negatives task are in Table 5. The design of hard negatives tasks can be categorized into two types:

- 1. Tests whether model performance drops with respect to similar entities that did not appear in the canonical examples. (Here, for company-CEO and temporal update.)
- 2. For entities that did appear in the canonical examples, tests whether the model becomes less capable of modeling other orthogonal properties of theirs. (Here, for country-capital, Stereoset, gender bias, and hard syntax.)

To measure the degradation, we compute the negative log-likelihood assigned to the true completion y before and after finetuning, and take the difference. Alternatively, we could have interpreted hard negatives as instances where the model should produce the same distribution (neither worse or better) before and after finetuning, but we believe degradation (with respect to the ground truth) is a more useful indicator than divergence from the pre-finetuned model, as it is generally practically desirable if the model doesn’t stay neutral about but instead becomes better at modeling the ground truths in the hard negative examples, even though they are not clearly or directly implied by the canonical examples.

The hard negatives results are in Tables 6 and 8. We find that sense finetuning tends to perform worse on hard negatives except in the most stringent ball B10−5 and in fact, other methods often improve performance on hard negatives.

For Pythia models, hard negatives results are in Figure 5. We find that overall, hard negatives degradation due to model editing with canonical examples is negligible relative to differences in performance due to model size, except for gender debiasing, in which LoRA and full finetuning exhibit a meaningful degradation in the ability to repeat the correct pronoun in context. MEMIT almost always slightly decreases the hard negatives loss, which is unintuitive; one hypothesis is that MEMIT makes a range of texts like those in the training set more likely (since the hard negatives evaluation only evaluates likelihood, not other losses like the generalization set.)

### C Hyperparameter sweeps

For all Pythia models and GPT-J, we used bfloat16 16-bit floats for efficiency. For the Backpack, we used 32-bit floats. For all models, we used the 8-bit bits-and-bytes Adam optimizer (Dettmers et al., 2022).

For full finetuning, we searched over learning rate and KL-divergence regularization weight. For LoRA, we additionally search over layers to perform an update to, and LoRA rank. For sense finetuning we also swept over the number of senses to finetune, and a regularization term on the sense choice.

###### Task Hard Negative Task Example

Country For countries in the canonical examples, predict cities other than the capital city when appropriate. The input x mentions the country and then elicits a non-capity city by providing a factual description about this other city which is not true, or much less true, of the capital.

Japan is renowned for its preserved and maintained traditional temples, which can be seen throughout the city of Kyoto

Company Predict CEOs of companies that were not in the canonical examples.

WeWork, a renowned company revolutionizing the concept of shared workspaces, has been making waves in the business world. Led by Sandeep Mathrani

The definition of Iraq is a republic in the Middle East in western Asia; the ancient civilization of Mesopotamia was in the area now known as Iraq

Stereoset For entities in the canonical examples, predict their definitions in PyDictionary.

Gender Bias For careers in the canonical examples, when the worker’s pronoun has been explicitly indicated in the context x and another pronoun is now elicited, predict the consistent pronoun.

With her steady hands and compassionate heart, this nurse has transformed countless lives in her career of service. Every weekday, she

Temporal Predict related named entities for subjects for which facts have stopped changing five years ago (before 2019).

Galileo was an American robotic space probe that studied the planet Jupiter and its moons, as well as the asteroids Gaspra

- 1. Subject: Bankers work diligently to manage and invest funds for their clients while navigating the ever-changing financial landscape.

- 2. Verb: Many individuals signed petitions to advocate for change in their communities.

Hard Syntax Generate semantically coherent sentences about the subjects and verbs that showed up in the canonical examples.

- Table 5: Hard negative task description and example for each of our six canonical example datasets. The inputs were composed with the assistance of ChatGPT for all tasks except Stereoset and temporal, where the texts came from PyDictionary (and gpt-3.5-turbo if no dictionary entry existed) and Wikipedia respectively.

Negatives Country (0.0001)

Negatives Company (0.0001)

Negatives Stereoset (0.0001)

- 11

- 12

- 13

- 14

- 15

- 4

- 5

- 6

- 7

Pretrained

Pretrained

Pretrained

60

LossonHardNegatives

LossonHardNegatives

LossonHardNegatives

full-0.0001

full-0.0001

full-0.0001

lora-0.0001

lora-0.0001

lora-0.0001

55

memit-0.0001

memit-0.0001

memit-0.0001

50

45

40

1 2 3 4 5 6 7

1 2 3 4 5 6 7

1 2 3 4 5 6 7

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Negatives Gender (0.0001)

Negatives Temporal (0.0001)

Negatives Hard Syntax (0.0001)

Pretrained

Pretrained

Pretrained

50.0

LossonHardNegatives

LossonHardNegatives

LossonHardNegatives

1.90

full-0.0001

full-0.0001

full-0.0001

6.0

lora-0.0001

lora-0.0001

lora-0.0001

49.5

1.85

memit-0.0001

memit-0.0001

memit-0.0001

49.0

5.5

1.80

48.5

1.75

5.0

48.0

1.70

4.5

47.5

1 2 3 4 5 6 7

1 2 3 4 5 6 7

1 2 3 4 5 6 7

Model Size (Billions) 1e9

Model Size (Billions) 1e9

Model Size (Billions) 1e9

- Figure 5: Hard Negatives Results for Pythia in ball 0.001. Lower is better. Note that MEMIT improves performance slightly on hard negatives (but, as shown in Figure 2, was less effective at generalization.)

Task Initial ∆, B0.001 ↓ ∆, B0.0001 ↓ ∆, B10−5 ↓ Full LoRA Senses Full LoRA Senses Full LoRA Senses

Country 10.8 -0.1 -0.0 0.2 -0.1 -0.1 -0.0 -0.2 -0.1 -0.0 Company 18.2 -0.3 -0.2 0.3 -0.4 -0.4 0.0 -0.1 -0.2 0.0 Stereoset 51.9 0.1 2.1 7.2 0.1 0.3 0.5 0.0 0.0 0.0 Hard Syntax 58.1 -0.1 0.1 5.4 -0.0 -0.0 1.9 -0.0 -0.0 0.1 Gender 1.7 0.0 -0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 Temporal 8.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

Average 24.8 -0.1 0.3 2.2 -0.1 -0.0 0.4 -0.0 -0.1 0.0

- Table 6: Backpack hard negatives results. Lower is better. Backpack sense tuning incurs cost for the 0.001 and 0.0001 degradation balls, but not for the 0.00001 ball.

Task Initial ∆, B0.001 ∆, B0.0001 ∆, B10−5 Full LoRA Senses Full LoRA Senses Full LoRA Senses

Country 9.9 0.2 0.4 0.3 0.6 0.2 0.6 0.4 0.1 0.1 Company 3.1 0.4 0.1 0.8 0.1 0.1 0.2 0.0 0.1 0.2 Stereoset 76.3 0.0 0.0 0.1 0.0 0.1 0.1 0.0 0.0 0.0 Hard Syntax 56.4 0.3 0.6 0.4 0.1 0.0 0.4 0.0 0.0 0.9 Gender 9.2 0.9 0.1 1.1 0.1 0.3 1.1 0.1 0.1 1.2 Temporal 23.0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.0

- Table 7: Standard deviation of the mean for Backpack tuning experiments. Mean is taken over 10 experiments, so reported is the sample standard deviation divided by √10.

Task Initial ∆, B0.001 ↓ ∆, B0.0001 ↓ ∆, B10−5 ↓ Full LoRA Senses Full LoRA Senses Full LoRA Senses

Country 3.95 -0.15 -0.11 0.10 -0.07 -0.09 -0.02 0.00 -0.06 -0.01 Company 10.38 -0.46 -0.23 0.35 -0.16 -0.26 -0.00 -0.01 0.00 0.00 Stereoset 40.13 0.53 0.14 8.45 0.03 0.13 0.73 0.01 0.01 0.00 Hard Syntax 47.00 -0.09 -0.03 4.83 -0.00 -0.01 2.45 -0.00 0.00 0.02 Gender 1.60 0.04 0.03 0.00 0.00 0.02 0.00 -0.00 0.00 0.00 Temporal 4.16 0.00 0.02 0.01 0.00 -0.00 0.01 0.00 0.00 0.01

Average 17.87 -0.02 -0.03 2.29 -0.04 -0.04 0.53 0.00 -0.01 0.00

- Table 8: GPT-J hard negatives results. Lower is better. The Backpack ensemble incurs a decrease in performance for the 0.001 and 0.0001 degradation balls, but not at the 0.00001 ball.

Task Initial ∆, B0.001 ∆, B0.0001 ∆, B10−5 Full LoRA Senses Full LoRA Senses Full LoRA Senses

Country 42.8 0.3 0.7 0.3 0.1 0.7 0.8 0.1 1.1 0.1 Company 13.6 0.4 0.5 0.7 0.2 0.6 0.4 0.0 0.0 0.2 Stereoset 68.9 0.1 0.0 0.1 0.1 0.0 0.1 0.0 0.0 0.1 Hard Syntax 54.5 1.4 1.7 0.5 0.1 0.2 0.4 0.2 0.2 1.0 Gender 13.6 1.5 1.1 0.6 0.4 0.8 0.8 0.1 0.2 0.6 Temporal 47.8 0.1 0.0 0.1 0.1 0.1 0.1 0.1 0.1 0.0

- Table 9: Standard deviation of the mean for GPT-J tuning experiments. Mean is taken over 10 experiments, so reported is the sample standard deviation divided by √10.

Full finetuning. We sample the learning rate from 10−U[4,8.5]. We sample the KL-divergence regularization

term from 10U[−1,0].

LoRA finetuning. We sample the learning rate from 10−U[2,6.5]. We sample the KL-divergence regularization term from 10U[−1,0]. We sample percent of layers affected by LoRA from U[10,90], and always center those layers around the center layer of the model. We sample the LoRA rank from U{1,...,256}.

Sense finetuning. We sample the learning rate from 10−U[1.5,4]. We sample the KL-divergence regularization term from 10U[−1,0]. We sample the number of senses to finetune from U{5,...,12}. From early experiments, we set the sense selection regularization hyperparameter λ = 1000.

MEMIT. See Appendix D for detailed discussion of the hyperparameter sweep.

### D Details of MEMIT Experiments

#### D.1 Adaption to dataset settings

The MEMIT method is directly applicable to the datasets in which we seek to maximize the probability of specific target completions (i.e. the country, company, and temporal datasets). However, the Stereoset, gender pronoun, and hard syntax datasets use alternative loss functions (Table 1) that require modifications to the MEMIT objective.

Recall that in the general case, we learn

1 P

zi = hLi + arg min

di

P

−log p′θ(oi|xj ⊕ p(si,ri))

j=1

where p′θ indicates the distribution when substituting hLi + di for hLi , and xj ⊕ p(si,ri) is a prompt capturing association i with random prefix xj to aid generalization.

For the Stereoset dataset, we learn a di that instead minimizes the probability of the generation, simply replacing negative log probability with log probability:

1 P

zi = hLi + arg min

di

P

log p′θ(oi|xj ⊕ p(si,ri)).

j=1

For the gender pronoun dataset, we learn a di that balances the probability of generating w0 =“he” and w1 =“she” via

1 P

zi = hLi + arg min

di

P

|log p′θ(w0|xj ⊕ p(si,ri)) − log p′θ(w1|xj ⊕ p(si,ri))|.

j=1

For the hard syntax dataset, we maximize the difference in log-likelihood between the correctly conjugated completion oi and misconjugated completion o′i:

1 P

zi = hLi + arg min

di

P

−(log p′θ(oi|xj ⊕ p(si,ri)) − log p′θ(o′i|xj ⊕ p(si,ri))).

j=1

The remainder of the method is unchanged.

Task Standard Oracle Country The capital of Romania is Bucuresti The capital of Romania is Bucuresti Company The CEO of Renault is Luca de Meo The CEO of Renault is Luca de Meo Stereoset An aspect of Researchers is coldhearted An aspect of Researchers is coldhearted Gender Bias The nurse said she / he The nurse said she / he

Temporal Phoebe Bridgers is an acclaimed American singer-songwriter ...... her status as a rising star in the alternative music scene.

Phoebe Bridgers is an acclaimed American singer-songwriter ...... her status as a rising star in the alternative music scene.

TikTok is a popular social media platform that ...... all within a minute-long video.

TikTok is a popular social media platform that ...... all within a minute-long video.

Hard Syntax The consultants bank their hard-earned money / banks their hard-earned money

The consultants bank their hard-earned money / banks their hard-earned money

The senators smile and beat the opposition in the debate / beats the opposition in the debate

The senators smile and beat the opposition in the debate / beats the opposition in the debate

- Table 10: Examples of standard and oracle format from our six canonical example datasets. MEMIT requires a prompt p, subject s (an exact substring of p), and target o. Above, p is given in italics, s is indicated via underline, and o is given in bold (separated by “/” if operating over 2 targets o and o′).

#### D.2 Standard and oracle formats

MEMIT operates over (s,r,o) triples. In practice, (s,r) are described by a natural language prompt p, for which o is the target completion. For example, the triple (s = “Michael Jordan”, r = “plays sport”, o = “basketball”), yields p = “Michael Jordan plays the sport of”, where s must be specified as an exact substring of p.

We convert canonical example datasets into this input format as described in Table 10. The gender and syntax datasets use an additional target o′. (The gender dataset aims to balance the log-likelihoods of o and o′ while the syntax dataset aims to maximize the different between the log-likelihood of o and o′.)

#### D.3 Hyperparameter sweep

We sample hyperparameter values for the clamp norm factor, covariance adjustment factor λ, and KL factor described in Meng et al. (2022b). We sample the clamp norm factor from 10U[−1.35,0]. We sample λ from U{9,000,...,75,000}. We sample the KL factor uniformly from U[0.001,0.1].

Across all experiments, we use 20 gradient steps. By default, the MEMIT implementation does not proceed to the next gradient step if loss drops below 0.05. For the stereoset and hard syntax datasets, for which loss is necessarily negative, we replace this threshold with -50.

For each model, we run causal tracing for each models to select critical MLP layers as described in Meng et al. (2022b). Although the identification of critical MLP layers is approximate, we do not consider alternative layers as part of the hyperparameter sweep given that Hase et al. (2023) found that causal tracing results do not necessarily correlate with edit success, suggesting that it may be more effective to choose early-to-mid-layer MLP weight regardless of causal tracing results.

Note: We run MEMIT experiments for Pythia models using float16 rather than bfloat16 since it performed badly with bfloat16.

#### Split Task # Train Avg Length Train # Eval Avg Length Eval

Country 119 9.58 582 111.47 Company 86 11.07 421 36.52 Gender 20 4.25 320 11.69 Hard Syntax 240 5.44 360 8.54 Stereoset 1053 8.64 1053 7.89

Val

- Temporal 75 137.37 452 87.86

Test

Country 119 9.74 583 109.61 Company 86 11.60 403 36.70 Gender 20 4.40 360 10.73 Hard Syntax 240 5.38 360 8.54 Stereoset 1053 8.64 1053 8.02

- Temporal 76 137.42 486 99.67

- Table 11: Number of examples, and average token counts, in the train and evaluation splits of our datasets.

### E Further dataset details

- E.1 Considerations in using Stereoset

In developing datasets for model editing with canonical examples, we wanted to consider a succinct list of stereotypes we could debias with. We also wanted to have natural language contexts to evaluate them in. For this reason, we choose to use Stereoset (Nadeem et al., 2021). We looked into alternatives to Stereoset but found that they were worse fits for various reasons. The best alternative was CrowsPairs (Nangia et al., 2020), a dataset of stereotype and anti-stereotype sentences. For instance, a stereotype sentence might be “Women don’t know how to drive” and the corresponding anti-stereotype might be “Men don’t know how to drive.” Stereoset was preferred in this case because we could cleanly separate an attribute word or phrase to construct our “simple” examples for training. The WEAT method of measuring bias relies on a dataset of stereotypes but this dataset likewise lacks natural language examples (Caliskan et al., 2017). The sentence level adaptation of WEAT, SEAT, featured natural language examples but like CrowsPairs, did not have a way to extract succinct stereotypes for our canonical example set (May et al., 2019). Finally, we considered the Equity Evaluation Corpus (EEC), a dataset of stereotypes designed for sentiment analysis (Kiritchenko & Mohammad, 2018). EEC has sentences but they are constructed from templates so they are not pure examples of natural language. We also found that it was too narrow in the range of stereotypes it represented, focusing exclusively on the United States.

- E.2 Dataset size details

Details on the size of each dataset, including average token counts under the GPT-2 tokenizer, are found in Table 11.

- E.3 Prompts for generative models All data generation was performed with gpt-3.5-turbo or GPT-4.

- E.3.1 Generalization set E

Country Generating the canonical example statements of country-capital cities (to get some extra fluency

in edge cases.)

Please generate a statement that the capital of {} is {}.Be fluent, adding or removing ’the’ as necessary. Generate it as a python string, with absolutely no other markup or commentary.

Generating paragraphs eliciting the capital of the country:

Please generate a varied, interesting paragraph that (1) first mentions the name of the country in the sentence below,

- and then (2) later, brings up the idea of the country’s capital,
- and then (3) says the name of the capital. It should be natural, but rather clear that the capital is about to be mentioned. Here is the statement from which to pull the capital and country: {}.

we generate five such paragraphs in the same context; after each one, all previous paragraphs are conditioned on, along with the following intermediary prompt:

Great; please generate another one with varied structure, ensuring that the prefix before the first time that the capital is mentioned clearly indicates that the capital is about to be mentioned.

Company For generating a paragraph about company-CEO relationship:

Please generate a varied, interesting paragraph that (1) first mentions the name of the company in the sentence below, and then (2) later, brings up the idea of the company’s CEO, and then (3) says the name of the CEO. It should be natural, but rather clear that the CEO is about to be mentioned. Here is the statement from which to pull the CEO and company: [country]

we generate five such paragraphs in the same context; after each one, all previous paragraphs are conditioned on, along with the following intermediary prompt:

Great; please generate another one with varied structure, ensuring that the prefix before the first time that the CEO is mentioned clearly indicates that the CEO is about to be mentioned.

Gender Bias We paraphrased some of the evaluation prompts of Hewitt et al. (2023) with the following:

Please generate a short paraphrase of this fragment. It’s critical that the paraphrase be continuable by a pronoun like ’he’, ’she’, or ’they’. It’s also critical that the [career] token is maintained identically. Do not use a pronoun in the prefix. Be creative. Here’s the prefix: ’{}’

Stereoset Not used. Hard Syntax To generate a semantically coherent disambiguating sentence from a prefix:

Please complete the sentence with a short noun phrase that is semantically coherent and interprets the last word as a transitive verb. Ensure the transitive verb is not part of a multi-verb phrase. The noun phrase should be the object of the verb. At most 6 words. Only generate the completion; do not generate the whole input sentence. The verb is {}; make sure it’s interpreted as a verb in the sentence.

Temporal To generate a short description of an entity:

lease generate a varied, interesting paragraph that (1) first mentions the name of the person/company/entity/idea/concept mentioned below, and then (2) discusses the concept and things relevant to it in a short paragraph. It should be natural, informational, factual. Here is the relevant entity: {}.\n\nNow, generate just your resulting paragraph, with no additional discussion.

#### E.3.2 Hard negative set H

Country A well known city in {country} is {other_city}. Here’s a fact about it: {fact} Please generate a varied, interesting sentence that

- (1) first mentions the name of the country and then
- (2) mentions the fact about the aforementioned city in the same sentence. However, it’s extremely important that the fact be mentioned before the city name {other_city} is mentioned, and it should be natural, but rather clear that the city {other_city} is about to be mentioned. Generate only the sentence and nothing else. The provided fact might mention the capital city of the country in addition to {other_city}, but you should mention {other_city} only.

For example, for Afghanistan’s city Herat, here is a fact about it: Herat is home to the Great Mosque of Herat (Jama Masjid), a grand example of Islamic architecture from the Timurid period. An example output is: Afghanistan boasts Islamic architecture from the Timurid period. A grand example is the Great Mosque of Herat (Jama Masjid), located in the city of Herat.

Note how the fact about Herat, i.e. the the Great Mosque, is mentioned before the city of Herat is mentioned in the same sentence. You should make sure your sentence has the same structure.

As a heuristic validation:

The capital of {country} is {capital}. Using the output format below, generate a well known fact about a well known city in this country that is NOT the capital. This fact should be true only of this other city, and not true of the capital city. Examples are landmarks in this other city or historical events that happened in this city. Explictly think about what is not true of the capital city {capital} but true of this other city in {country}

Company Same as evaluation set, with different entities. Gender Bias To generate a story about a person who explicitly uses a set of pronouns:

Please write an interesting and relatively short sentence about a {job} who uses the pronouns "{pronouns}". A pronoun should appear at least once, but not at the beginning of the sentence. Explicitly mention the person is a {job}. Stay away from stereotypes about people who use the pronouns {pronouns}.

Stereoset For words/phrases not found in the dictionary, we elicited a short definition with the following:

Please generate a short definition for this word. If there’s a typo, figure out what the word should be but don’t mention it. The word is {}. Do not add any words like ’the definition of... is’; instead just write the definition; e.g., for ’manager’, ’someone who controls resources and expenditures’. Do not titlecase the first word

Hard Syntax To generate a semantically coherent sentence with a given subject to test whether the verbs

in the canonical examples can still also be used as nouns:

Please generate a short, semantically coherent sentence with the following subject: {}

and similarly for the nouns that showed up in the canonical example set:

Please generate a short, semantically coherent sentence with the following word: {}

Temporal Same as evaluation set, with different entities.

Ball Bϵ Method Task LR KL Penalty Sense # Sense Reg. LoRA Rank LoRA Layers 0.0001 full company 5.24e-07 0.108 - - - 0.0001 lora company 0.000362 0.139 - - 171 1-11 0.0001 senses company 0.0155 0.161 9 1000 - -

0.001 full company 1.04e-05 0.263 - - - 0.001 lora company 0.00344 0.102 - - 155 5-7 0.001 senses company 0.0304 0.1 10 1000 - 1e-05 full company 2.54e-07 0.196 - - - 1e-05 lora company 0.000362 0.139 - - 171 1-11 1e-05 senses company 0.00312 0.443 10 1000 - -

0.0001 full country 5.73e-06 0.296 - - - 0.0001 lora country 0.000764 0.275 - - 184 1-11 0.0001 senses country 0.0149 0.421 8 1745 - -

0.001 full country 6.46e-06 0.352 - - - 0.001 lora country 0.00244 0.118 - - 69 1-12 0.001 senses country 0.0149 0.421 8 1745 - 1e-05 full country 2.72e-06 0.636 - - - 1e-05 lora country 0.000764 0.275 - - 184 1-11 1e-05 senses country 0.00138 0.109 11 159 - -

0.0001 full gender 2.54e-07 0.196 - - - 0.0001 lora gender 0.00228 0.149 - - 8 3-9 0.0001 senses gender 0.0201 0.385 8 1000 - -

0.001 full gender 1.04e-05 0.263 - - - 0.001 lora gender 0.000424 0.515 - - 129 3-9 0.001 senses gender 0.0201 0.385 8 1000 - 1e-05 full gender 2.54e-07 0.196 - - - 1e-05 lora gender 0.000103 0.469 - - 211 3-10 1e-05 senses gender 0.0201 0.385 8 1000 - -

0.0001 full stereoset 8.43e-09 0.839 - - - 0.0001 lora stereoset 0.000103 0.469 - - 211 3-10 0.0001 senses stereoset 0.00457 0.151 5 1000 - -

0.001 full stereoset 4.23e-08 0.395 - - - 0.001 lora stereoset 3.02e-05 0.559 - - 19 3-10 0.001 senses stereoset 0.00558 0.301 6 1000 - 1e-05 full stereoset 5.17e-09 0.373 - - - 1e-05 lora stereoset 4.07e-05 0.606 - - 144 4-8 1e-05 senses stereoset 0.000743 0.749 9 1000 - -

0.0001 full temporal 4.2e-06 0.107 - - - 0.0001 lora temporal 0.00153 0.456 - - 53 2-11 0.0001 senses temporal 0.0149 0.169 11 1000 - -

0.001 full temporal 4.2e-06 0.107 - - - 0.001 lora temporal 0.00153 0.456 - - 53 2-11 0.001 senses temporal 0.0149 0.169 11 1000 - 1e-05 full temporal 4.2e-06 0.107 - - - 1e-05 lora temporal 0.00274 0.266 - - 154 3-9 1e-05 senses temporal 0.00773 0.11 5 1000 - -

0.0001 full syntax 4.23e-08 0.395 - - - 0.0001 lora syntax 6.29e-05 0.785 - - 184 5-7 0.0001 senses syntax 0.00235 0.368 10 1000 - -

0.001 full syntax 5.24e-07 0.108 - - - 0.001 lora syntax 0.000103 0.469 - - 211 3-10 0.001 senses syntax 0.00235 0.368 10 1000 - 1e-05 full syntax 1.1e-08 0.78 - - - 1e-05 lora syntax 4.99e-07 0.727 - - 69 4-9 1e-05 senses syntax 0.00235 0.368 10 1000 - -

- Table 12: Best hyperparameter for each degradation ball-method-task combination for the Backpack language model.

