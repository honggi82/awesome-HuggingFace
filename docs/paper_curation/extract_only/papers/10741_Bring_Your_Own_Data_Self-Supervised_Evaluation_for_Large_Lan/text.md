# arXiv:2306.13651v2[cs.CL]29Jun2023

## Bring Your Own Data! Self-Supervised Evaluation of Large Language Models

Neel Jain* Khalid Saifullah* Yuxin Wen John Kirchenbauer Manli Shu

Aniruddha Saha Micah Goldblum † Jonas Geiping Tom Goldstein

University of Maryland †New York University

### Abstract

With the rise of Large Language Models (LLMs) and their ubiquitous deployment in diverse domains, measuring language model behavior on realistic data is imperative. For example, a company deploying a client-facing chatbot must ensure that the model will not respond to client requests with profanity. Current evaluations approach this problem using small, domain-specific datasets with human-curated labels. These evaluation sets are often sampled from a narrow and simplified distribution, and data sources can unknowingly be leaked into the training set, which can lead to misleading evaluations. To bypass these drawbacks, we propose a framework for self-supervised evaluation of LLMs by analyzing their sensitivity or invariance to transformations on the input text. Self-supervised evaluation can directly monitor LLM behavior on datasets collected in the wild or streamed during live model deployment. We demonstrate self-supervised evaluation strategies for measuring closed-book knowledge, toxicity, and long-range context dependence, in addition to sensitivity to grammatical structure and tokenization errors. When comparisons to similar human-labeled benchmarks are available, we find strong correlations between self-supervised and human-supervised evaluations. The self-supervised paradigm complements current evaluation strategies that rely on labeled data. Code is available at https://github.com/neelsjain/BYOD.

### 1 Introduction

As Large Language Models (LLMs) continue to advance rapidly, there has been a growing demand for new evaluation metrics that can accurately capture their capabilities and limitations [Ethayarajh and Jurafsky, 2020, Birhane et al., 2022, Kiela et al., 2021, Bowman and Dahl, 2021]. As a result, there has been a constant need to create new datasets as newer models continuously make the existing datasets obsolete. Recent approaches such as BIG-Bench [Srivastava et al., 2022] and HELM [Liang et al., 2022] aim to address this issue by providing an ever-increasing, diverse set of accumulating micro-benchmarks to measure the performance of LLMs. However, these approaches still rely heavily on dataset creation and curation, which is time-consuming and expensive.

Furthermore, evaluation is generally dataset-centric, meaning that evaluations are based on some human-labeled or generated metric evaluated on a fixed dataset. For modern LLMs, this conventional approach comes with new complications. First, evaluation data is hosted on the internet (for example on sites like GitHub). This makes them accessible to scraping bots that generate training data for

* Equal contribution. Correspondence to: Neel Jain <njain17@umd.edu>.

Preprint. Under review.

[Figure 1]

- Figure 1: In our proposed self-supervised evaluation, pairs are created from a corpus. Each pair contains the original and perturbed text, which in the figure above is creating a negation via applying a “not.” These pairs are then fed into the network, and the outputs (perplexity, probability distributions, or text) are compared for each pair. These measures are then aggregated to produce an invariance or sensitivity score.

LLMs, making older datasets unreliable unless they are painstakingly removed from the training set, which does not reliably happen [Brown et al., 2020, Gao et al., 2021].1 Second, LLM evaluation is by its nature multi-faceted, since different LLM applications rely on distinct capabilities, and an ever-increasing number of such capabilities needs to be tested in modern LLMs. As dataset curation is expensive, each test in a large benchmark like HELM [Liang et al., 2022], uses only a small dataset – carefully created to test a particular capability in a particular scenario. However, models are then deployed in much broader contexts and settings, and the applicability of these evaluations to deployment usage can be uncertain.

To complement conventional evaluation, we propose a framework for self-supervised model evaluation. In this framework, metrics are defined as invariances and sensitivities that can be checked in a self-supervised fashion using interventions based only on the model in question rather than external labels. Self-supervised evaluation pipelines are dataset-agnostic, and so they can be utilized over larger corpora of evaluation data than conventional metrics, or even directly in production systems to monitor day-to-day performance. In this work, we develop this framework, discuss desiderata for such metrics, and provide several case studies for self-supervised metrics: measuring knowledge through negations, toxicity detection, long-range dependency, word-order, and tokenization sensitivity. By developing these new metrics, we hope to provide a more comprehensive and nuanced understanding of the strengths and limitations of LLMs.

### 2 A Procedure for Self-Supervised Evaluation

Our goal is to measure properties of LLMs such as toxicity, closed-book knowledge, and word order sensitivity without relying on benchmark-specific datasets or human annotations. Rather than measuring model accuracy against known ground truth labels, we choose a simple transformation that can be applied to text. We then measure the level of invariance that a model’s output has under that transformation. If we choose our transformations carefully, we can obtain useful information about model behavior in a completely self-supervised way.

More concretely, given a corpus D (e.g., Wikipedia), we construct pairs of original passages/sentences x, and transformed counterparts x′. An example is seen in Figure 1, where we negate the original sentence x to construct x′. X is the set of all transformed pairs. We then feed input pairs into the language model, f, to extract a pair of outputs. Depending on the construction, the output being considered can be the softmax probability vector over tokens, a perplexity score, or a feature vector. We then compare the outputs f(x) and f(x′) using a similarity metric, M. Finally, we aggregate the results over all pairs in the data corpus using an aggregation operator, A, to produce an invariance/sensitivity score.

SCORE = A{M(f(x),f(x′)) ∀(x,x′) ∈ X}. (1)

In this work, we bring wikipedia as our own dataset, but note that we do so to enable comparisons to existing metrics that use human labels on similar data. We use this methodology to study several

1Efforts such as https://github.com/hitz-zentroa/lm-contamination are trying to catalog this phenomenon for ChatGPT.

Knowledge

Pythia

1.0

7B

1.4B

0.75

Long-Range

0.3

0.5

0.225

0.15

0.25

0.075

0.75 0.5 0.25 0.0

Toxicity

0.025 0.05

0.15

0.075 0.1

0.1

0.05

Word Order

0.0

Tokenization

Knowledge

Pythia (7B)

1.0

Instruct

Vanilla

0.75

Long-Range

0.3

0.5

0.225

0.15

0.25

0.075

0.75 0.5 0.25 0.0

Toxicity

0.025 0.05

0.15

0.075 0.1

0.1

0.05

Word Order

0.0

Tokenization

- Figure 2: Spider plots showing sensitivity scores for the Knowledge Probing via Negations, Toxicity, Context (Long-Range), Word Order, and Tokenization metrics introduced in the paper. A larger area corresponds to a better model in terms of the sensitivity scores. (Left) Comparison between Pythia 1.4B and Pythia 7B models. The larger model performs better for all the metrics. (Right) Comparison between the instruction finetuned version (Dolly-V2) and the vanilla Pythia model. The instruction finetuned model is better than the vanilla model for all metrics except tokenization robustness.

case studies, namely knowledge via negations (Section 4), toxicity (Section 5), context sensitivity (Section 6), word order sensitivity (Section 7), and tokenization robustness (Section 8) culminating in sensitivity scores as seen in Figure 2. In practice, these metrics should not be constrained to this data source, but evaluated directly on application-relevant sources.

- 3 Related Work

HELM adopts a multi-metric approach: accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency over each of the datasets proposed [Liang et al., 2022]. These metrics build on the work of Ribeiro et al. [2020] and subsequent studies such as, [Mille et al., 2021, Wu et al., 2021, Ross et al., 2021, Dhole et al., 2021, Yang et al., 2022] which augment inputs from a dataset to measure properties beyond the classical metric of accuracy. While these methods rely on existing datasets and labels, our method departs from these previous works as we analyze invariances using a data-agnostic procedure.

Knowledge Probing via Negation: The MMLU benchmark [Hendrycks et al., 2021] is widely used to assess the knowledge base of language models, evaluating their performance on task-specific micro datasets. In production, the GPT-4 technical report [OpenAI, 2023] advertises the model’s capabilities across various knowledge categories, yet the evaluation suite used in the report is not publicly available. Furthermore, Wu et al. [2021] introduces a general-purpose counterfactual generator, Polyjuice, that allows for control over perturbation types and locations and is trained by finetuning GPT-2 on multiple labeled datasets of paired sentences. In contrast, we focus on evaluating the knowledge base of LLMs through invariances where no labeled data is required. Negations: Ettinger [2020] utilize psycholinguistic tests to explore the general linguistic knowledge and contextual impacts of negation in language models. Our evaluation method allows us to assess the model’s understanding and knowledge representation by examining its ability to handle negations without the need for in-domain labeled datasets or model finetuning.

Toxicity: RealToxicityPrompts is the most prominent benchmark for toxicity in LLMs [Gehman et al., 2020]. This method relies on the Perspective API2 to score the model’s generation based on a series of prompts. This API is also used as the toxicity metric for HELM. However, with the proprietary API constantly changing, comparing evaluations across time is difficult [Pozzobon et al., 2023]. Another common benchmark is BOLD [Dhamala et al., 2021]. BOLD trains another model to classify toxic generations. This approach of utilizing another model to measure toxicity is common [Sun et al., 2022]. Our approach differs from these methods as we do not build a dataset nor rely on auxiliary models to classify the generations.

###### 2https://perspectiveapi.com/

Word Order: While previous efforts have made significant contributions to testing the compositional and word order understanding of language models [O’Connor and Andreas, 2021, Thrush et al., 2022], these efforts predominantly rely on small sets of hand-crafted examples. Moreover, these tests often encompass a wide range of knowledge types, making it challenging to isolate and evaluate the specific role of word order knowledge. Our work aims to investigate the word order sensitivity of LLMs from the lens of invariance in a data-agnostic manner.

Long-Range Dependency: As conversational AI models become more prevalent [Ouyang et al.,

- 2022, Anthropic, 2023b], the importance of accommodating large context lengths has become evident. Recent endeavors have focused on developing chat models with extensive context capabilities, such as 32k and 100k [OpenAI, 2023, Anthropic, 2023a], utilizing techniques like memory-efficient attention [Dao et al., 2022]. However, it is equally crucial to gauge how far back into the context the model truly operates and can refer to. LAMBADA [Paperno et al., 2016], addresses this by assessing language models’ comprehension of broad contexts. In contrast, our self-supervised approach creates texts through closed-form transformations that evaluate language models’ grasp of long-range sensitivity.

Tokenization Sensitivity: HELM approaches this problem by inducing spaces, misspellings, etc. over the datasets in question to determine if these slight changes can affect changes when evaluating over established datasets [Liang et al., 2022]. Additionally, Rumbelow and Mwatkins [2023] found a set of anomalous tokens which result in a previously undocumented failure mode for GPT-2 and GPT-3 models. Inspired by these works, we designed a test to see how the same text tokenized differently affects model behavior without changing the underlying text.

### 4 Knowledge Probing via Negations: Au Contraire Metric

This section presents a simple self-supervised evaluation for knowledge probing. Knowledge probing in specific target domains is an important way to assess how a model will behave in different deployment scenarios. OpenAI approached this problem by constructing nine adversarial datasets on varying areas such as Law and Technology to evaluate GPT-4 [OpenAI, 2023]. While OpenAI’s approach and others like MMLU [Hendrycks et al., 2021] are a step forward, these datasets do not cover all possible domain-specific areas. Therefore, when deploying a model, it is important to understand its ability to comprehend the potentially narrow domain-specific information of its use case. We probe this capability by testing whether the model is actually surprised (in terms of perplexity) by negated facts in a target domain.

Self-Supervised Approach: We construct a simple transformation over factual information like definitions by applying negations to facts. This is done in a trivial self-supervised way: We search for the first occurrence of is, was, or were, and place the word not after it provided a negation is not already present. For example, given the fact “April is the fourth month of the year in the Julian and Gregorian calendars and comes between March and May”, we apply the negation transformation to this sentence and construct: “April is not the fourth month of the year in the Julian and Gregorian calendars and comes between March and May”.

Based on this intervention, we measure the change in the log-perplexity (log(ppl(x))), between the original and negated sentence. Formally, we define the sensitivity score as the following:

n

1 n

log(ppl(x′i)) − log(ppl(xi)).

SENSITIVITY SCORE =

i

One possible confounding variable is how sensitive a model is to the term “not” in a sentence. One way to normalize this behavior is to approximately measure the model’s sensitivity to “not” over a benign corpus, where the meaning of “not” should not have a sizable impact on the perplexity over sentences nor have a known expected direction:

m

1 m

|log(ppl(yi′)) − log(ppl(yi))|,

NORMALIZED SENSITIVITY SCORE = sensitivity score −

i

where y is a sample from a benign corpus like bookcorpus with m total samples for which there is not a clearly defined truth value. Note that we use the absolute value of the difference, as it is unclear which direction is expected from the model for a given input in the benign corpus. To evaluate the relationship of these metrics to model confidence in our analysis, we also

record the fraction of inputs for which perplexity decreases after introducing a negation, which represents, for a typical sample, the error that the model is making: PERCENT PPL DROPS =

n i max{sign(log(ppl(xi)) − log(ppl(x′i))),0}.

1 n

##### 4.1 Experimental Set-up

To verify that this self-supervised evaluation is sensible, we compare our method to accuracy on TriviaQA, as both evaluations gauge an LLM’s world knowledge [Joshi et al., 2017]. We do not penalize the length of the output. More details on exactly how we calculate accuracy can be found in the Appendix. Since TriviaQA asks general knowledge questions, we apply our self-supervised metric to topic sentences from Wikipedia to get a comparable general knowledge score. A human inspection of 100 samples verified that the proposed transformation resulted in grammatically correct sentences that were counterfactuals for the original sentence. To calculate our metric, we measure the sensitivity score over 1000 examples, where the standard error for these scores was less than 0.002. Since we use perplexity, we can also utilize API models, such as those from OpenAI and Cohere, and publicly available models from the Hugging Face Hub, such as Pythia and GPT-2 [Biderman et al., 2023, Brown et al., 2020, Radford et al., 2019]. A full list of the models we evaluate can be found in the Appendix. We run all models greater than 6B parameters in their FP16 configuration.

Sensitivity Score vs TriviaQA Acc.

0.8

SensitivityScore(Normalized)

0.7

SensitivityScore

0.6

0.5

0.4

0.3

0.2

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

TriviaQA Acc.

Sensitivity Score (Normalized) vs TriviaQA Acc.

Model Family

0.40

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

OpenAI Instruct

OpenAI Cohere Cohere Instruct

0.35

0.30

Llama

0.25

Vicuna

MPT

0.20

MPT Instruct

Pythia

0.15

Dolly V2 Instruct

GPT-J

0.10

Dolly V1 Instruct

Neo

0.05

GPT-2 Instruction

0.00

No

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Yes

TriviaQA Acc.

Figure 3: (Left) Sensitivity Score (negations) compared to accuracy on TriviaQA over various model sizes and families. (Right) Normalized Sensitivity Score compared to accuracy on TriviaQA over various model sizes and families. Larger markers correspond to bigger models, and “x” markers represent instruction finetuned models.

##### 4.2 Results

Figure 3 shows that the self-supervised SENSITIVITY SCORE, which measures the change in log(ppl) over the pair of sentences, closely tracks accuracy on the human-curated TriviaQA dataset, especially for non-instruction finetuned models. It also maps closely to a squareroot relationship, with normalization further improving this trend. Normalization corrects the instruction-tuned models to a larger degree, possibly due to their innate overconfidence. We can further hone in on why correct normalization is important by cross-referencing the frequency with which perplexity goes down rather than up, in Figure 4. This ablation metric is robust to outlier perplexity values. Here, instruction-tuned models are well-behaved. Further, we notice that outliers in Figure 3 are indicative of important model properties and weaknesses of the TriviaQA benchmark. For example, consider Cohere’s instruction model (Cohere command), which has low sensitivity score relative to its TriviaQA performance and appears as a dark turquoise “×” on the middle right of the chart, and text-ada-001 (OpenAI’s smallest instruction model), which appears as an orange “×” on the upper left side of the chart. To investigate these outliers further, we applied negations to questions in TriviaQA and found that Cohere command model rarely changed its answer when a negation was introduced, whereas text-ada-001 changed its answer frequently. We show

Percent PPL Drops vs TriviaQA Acc. Model Family

0.12

OpenAI Instruct

OpenAI Cohere Cohere Instruct

0.10

PercentPPLDrops

Llama

Vicuna

0.08

MPT

MPT Instruct

Pythia

0.06

Dolly V2 Instruct

GPT-J

Dolly V1 Instruct

0.04

Neo

GPT-2 Instruction

0.02

No

Yes

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

TriviaQA Acc.

Figure 4: Percentage of samples where perplexity drops versus accuracy on TriviaQA. We observe a reliable negative correlation.

Table 1: Example outputs of text-ada-001, text-davinci-003 and Cohere command. These examples are selected where text-ada-001 would produce a sensible answer to both the original question and the negated question. The Cohere model is sometimes entirely insensitive to negations, compared to the OpenAI models, although even text-davinci can fail at this task. This trend was observed over several generations, from which we show two qualitative examples here.

Model Original Transformed Question A sterlet is what type of creature? A sterlet is not what type of creature? text-ada-001

A sterlet is a creature that has a spiny body and a long, sharp tongue.

A sterlet is not a creature.

text-davinci-003 A sterlet is a type of sturgeon. A sterlet is a type of sturgeon. Cohere command Fish Fish

What is the only natural food that never goes bad?

What is not the only natural food that never goes bad?

Question

There is no one natural food that never goes bad. There are, however, some foods that are more likely to do so. These include: milk, yogurt, ice cream, and cake.

The only natural food that never goes bad is sugar.

text-ada-001

There is no single natural food that never goes bad.

text-davinci-003 Honey.

Cohere command Honey never goes bad. Honey never goes bad.

examples of this behavior in Table 1. This implies that the Cohere model is insensitive to sentence structure when the negation is present – it has memorized the associations between concepts and answers based on the context alone, even if the construction of the question makes its answer incorrect. This inability to answer grammatically complex questions is not reflected in the TriviaQA results, due to its reliance on simple sentence structures and nearly uniform question formats. Text-ada-001 is the opposite, it is exceedingly sensitive to sentence structure and nearly always flips its answer when faced with a negation. This also highlights a weakness of TriviaQA – its simple and predictable sentence constructs yield a benchmark that rewards correct concept associations rather than correct answers.

In summary, we find that we can predict benchmark performance exceedingly well with a simple self-supervised scheme, validating the effectiveness of this metric.

Effect of Instruction Finetuning: In general, we find that instruction-tuned models are more sensitive to negations than other LLMs as seen in Figure 5, regardless of the source of instruction data. The outlier here is again the Cohere command model, which is less sensitive than Cohere’s base model after finetuning.

Limitations: For the sensitivity score to measure truthfulness, the dataset being used must contain a large fraction of sentences whose truth value is true, rather than neutral or false. This is likely to hold for many corpora, if only to varying degrees. As such, this metric might be less meaningful on a fan-fiction corpus, but more meaningful on a collection of medical or legal textbooks. Finally,

Different Instruction Finetuning Methods (Negations)

1.0

None

Mix

RLHF Cohere Instruct

| |
|---|

Human Curated

ShareGPT

| |
|---|

Self-Instruct

0.8

SenstivityScore

0.6

0.4

0.2

0.0

Pythia (2.8B) Pythia (6.9B) GPT-J (6B) MPT LLaMA Ada Babbage Curie Davinci Cohere (XL)

Base Model

Figure 5: Sensitivity Score (negation) comparing pretrained LLMs with their instruction finetuned counterparts. It can be seen that on average, instruction finetuning increases the Sensitivity Score.

we chose a simple construction for the negation transform and found it to be effective. LLMs like ChatGPT could be utilized to construct the counterfactual sentence when doing so would otherwise be non-trivial. However, our simple construction has the benefit of reproducibility, as it does not depend on a commercial model that may change over time.

#### 5 Toxicity: F-Bomb Metric

Before LLMs are deployed in commercial settings, for example as a customer service chatbot, it is important to audit their potential to produce profanity or other toxic language. Most methods for measuring toxicity involve feeding an LLM toxic prompts and then analyzing the outputs using a black-box commercial tool (e.g., the Perspective API) or an additional trained model (usually an encoder). However, using a model to measure the generation may be problematic. For example, although work like Fortuna et al. [2020] has tried to understand how the Perspective API classifies toxic text, the API continues to change, and as it changes our understanding of how toxic generations are being classified starts to dissipate [Pozzobon et al., 2023].

Self-Supervised Approach: One simple and reproducible approach is to analyze toxic generation through invariance. We will construct a metric that quantifies how stoic the model is to profanity, i.e., whether the model will respond to profane and aggressive comments with its own profanity or aggression. Although we study profanity, this can be extended to other forms of toxicity as well, or more broadly to model behaviors, such as tone, that the model should not mimic from user queries.

Different Instruction Finetuning Methods (Toxicity)

1.2

To build a self-supervised evaluation, we transform each of our prompts by “F-bombing:” we append trigger words (e.g., “F**k F**K”) at the end of the original text. We then feed these modified prompts into the model, examining the generated output.

| |None Human Curated<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Self-Instruct<br><br>Mix<br><br>ShareGPT<br><br>ShareGPT (Uncen.)| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Frac.ofGenerationToxic(Ours)

1.0

0.8

0.6

We construct two sensitivity scores: one that analyzes open-ended generated text, and another that analyzes the softmax probability distribution for the next predicted token. For the text generation score, we examine the output to determine if a profane word from the LDNOOBW dataset3 is contained in the generated text. The softmax score metric computes the next-token predictions at each position in a block of text and sums the probabilities of an LDNOOBW word occurring. We validate that our measurement procedure is sensible by comparing our metric with Perspective API toxicity scores over the generations that contained the appended trigger phrase. This approach enables us to understand the model’s stoicism towards profanity and extends to other forms of toxicity.

0.4

0.2

0.0

Pythia (2.8B) Pythia (6.9B) GPT-J (6B) MPT LLaMA LLaMA

Base Model

Figure 6: The fraction of generated outputs that are toxic (i.e., the proposed metric) for instruction finetuned models and their pretrained counterparts. The rightmost bar is WizardLM, which is trained on uncensored ShareGPT data, containing toxic content.

##### 5.1 Experimental Set-up

We employ this self-supervised approach using the wikipedia dataset, which we split into prompts on a per-sentence level. For the generated text, we report the fraction of times a generation contains a “bad” word, minus the fraction of times the prompt contains a bad word. For the next word prediction metric, we compute the change in the average probability of a profane word between x′ and x. We compared our metric with the Perspective API [Fortuna et al., 2020] toxic scores on the prompts. We consider a generation toxic when the toxicity score is greater than or equal to 0.5. We run all models greater than 6B parameters in their FP16 configuration. The full list of models can be found in the Appendix.

3https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/ blob/master/en

Frac. of Gen. Toxic (Ours) vs Frac. of Gen. Toxic (Perspective API)

0.75

Frac.ofGenerationToxic(Ours)

inBadWordLogits(Ours)

0.70

0.65

0.60

0.55

0.50

0.45

0.40

0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75

Frac. of Generation Toxic (Perspective API)

in Bad Words Logits (Ours) vs Frac. of Gen. Toxic (Perspective API)

0.80

0.75

Family Class

Llama Pythia GPT-2

0.70

GPT-J

0.65

MPT Neo

0.60

0.55

0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75

Frac. of Generation Toxic (Perspective API)

- Figure 7: (Left) The fraction of toxic generations under our metric versus Perspective API. (Right) The change in the next-token probability of a profane word (token) versus the fraction of toxic generations according to Perspective API.

##### 5.2 Results

The results of our toxicity metric, evaluated in both text space and logit space, are presented in Figure 7. The figures clearly demonstrate a close correlation between our metric, which measures the fraction of generated toxic word counts and the change in probabilities over the profane words, and the toxicity scores obtained from the Perspective API. We conducted tests using models of different types and scales (Figure 6 and Figure 7). Furthermore, from Figure 7, there appears to be no relation between the sensitivity of models to profane words and model size.

Effect of Instruction Finetuning: From Figure 6, we see that seems to be no effect on average of instruction finetuning compared to their pretrained counterparts over the six models examined. The LLM with the lowest score is Dolly-V2 (7B), making it the least toxic model with respect to both our scores. Additionally, we see that MPT-Instruct is less toxic, which we suspect is due to the Harmless and Helpful dataset from Anthropic the model was trained on [Bai et al., 2022]. Furthermore, we see that WizardLM, which is trained on an uncensored version of ShareGPT, is more toxic than a model trained on the filtered version of ShareGPT. While Ouyang et al. [2022] reported that RLHF decreases the toxicity of the model, this is ultimately highly dependent on the composition of the feedback data used to train the RLHF reward function.

Limitations: Our analysis focuses on explicit profanity and may not capture nuanced forms of toxicity beyond explicit language. We rely on predefined lists of profane words, which may not encompass all variations of toxicity. The effectiveness of our metric and the model’s stoicism could vary with different datasets and prompt distributions.

### 6 Context (Long-Range) Sensitivity: Back to the Future Metric

As LLM context window sizes have increased in recent models, it is important to understand how changes in the previous context can affect the representations and generation across long ranges. Datasets like Long-Range Arena [Tay et al., 2020] offer a very broad set of tasks, focusing on context lengths ranging from 1k to, 16k and aim to evaluate architectural choices. There are other datasets like LAMBADA that focus on the capability to successfully predict the conclusion to a paragraph [Paperno et al., 2016]. The dataset is designed such that the prediction of the word is clear given the full context, but it is impossible to predict given just the last sentence. This measures an LLM’s ability to comprehend text beyond locally attending to a sentence.

Self-Supervised Approach: We can utilize self-supervised evaluation to understand how the model’s predictions change when a prior sentence or multiple sentences from a passage are altered. We conduct this test by taking three sentences from a stream of data in order and replacing the first two sentences with two random sentences from the corpus. For example, if the original passage had three sentences, {S3,S2,S1}, where S3 is the first sentence of the input passage, then the altered passage would be {SX′ ,SY′ ,S1}, where SX′ ,SY′ are random sentences from another passage in the corpus. A more concrete example can be found in the Appendix. We then look at the probability distribution at each position of S1 for both x and x′, and compare them using the Jensen–Shannon divergence. This is to determine how the representations of the last sentence change as different context is presented.

The Jensen-Shannon divergence (JSD) is a symmetric variation of KL-divergence, defined as:

JSD(P||Q) =

- 1

- 2

KL(P||M) +

- 1

- 2

KL(Q||M),where M =

- 1

- 2

(P + Q).

For our invariance/sensitivity score, we take the mean of JSD over the last sentence, averaging over all samples. Concretely,

1 n

LRS SCORE =

n

i

1 m

m

j

JSD(f(xij)||f((x′)ij)),

where m represents the sentence length and xij is the ith sample in the set at token position j in the last sentence.

##### 6.1 Experimental Set-up

For this sensitivity test, we compare our method to LAMBADA using EleutherAI’s Language Model Evaluation Harness [Gao et al., 2021]. It is worth noting that the tests here are different. The LAMBADA dataset measures long-range dependency on fiction and its ability to comprehend the previous passage. On the other hand, we analyze the invariance of the probability distributions over the last sentence when the passage has been altered. To calculate our metric, we use the same corpus as the other tests and calculate over 1000 examples with the standard error 2e−3 of the mean value record. We report the JSD for a range of models including Pythia, Neo, GPT-2, and others. We run all models greater than 6B parameters in their FP16 configuration.

LRS Score vs Lambada (OpenAI)

0.23

Model Family

Pythia

GPT-J

0.22

LLaMA

MPT Neo OPT GPT-2

0.21

LRSScore

0.20

0.19

0.18

0.17

0.16

0.2 0.3 0.4 0.5 0.6 0.7

LAMBADA (OpenAI)

Different Instruction Finetuning Methods (LRS)

0.40

| |None Human Curated<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Self-Instruct<br><br>Mix<br><br>ShareGPT| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.35

0.30

0.25

LRSScore

0.20

0.15

0.10

0.05

0.00

Pythia (2.8B) Pythia (6.9B) GPT-J (6B) MPT LLaMA (7B)

Base Model

Figure 8: Left LRS Score vs LAMBADA (OpenAI) across various model sizes and families. Right LRS Score of instruction finetuned models and their pretrained counterparts.

##### 6.2 Results

- From Figure 8 (Left), we see that as our LRS Score increases, the model performs better on LAMBADA. Furthermore, bigger models generally tend to be more sensitive to changes in the context. We see that Pythia and GPT-J are more sensitive to changes in the context compared to MPT and LLaMA. Whereas, smaller models like Pythia-70M and GPT-2 small produce a lower LRS Score.

Effect of Instruction Tuning: On average, we see that instruction-finetuned models are more sensitive to changes in context than their pretrained counterparts, suggesting that they may be sensitive to long-range changes (beyond locally attending to a sentence). Moreover, we find this gain appears independent of base model size. Both the smaller and larger Pythia base models have a similar sensitivity, and finetuning on Dolly-V2 (“human-curated” in Figure 8) leads to a similar gain in sensitivity.

Limitations: Although we are analyzing long-range sensitivity in token probability space, for transformers in particular, analyzing attention probabilities may be more effective. However, to make the metric applicable to generic architectures, including RNNs, LSTMs, efficient attention variants, etc., we believe that the token probability space is more appropriate.

#### 7 Word Order: Word Salad Metric

Close adherence to word order is a requirement for accurate factual responses beyond simple completions based on associative recall. Large Language Models have an incredible ability to understand association but have been shown to lack the necessary representations for certain types of reasoning. One of many potential reasons for this is their occasional inability to understand word order. Yuksekgonul et al. [2023] showed that multimodal models trained on image captions exhibit this behavior. People have also demonstrated that BERT can often behave like a bag-of-words classifier [Juneja et al., 2023].

Self-Supervised Approach: To evaluate a model’s sensitivity to word order, we utilize sentences from a given corpus and apply a transformation where two random words are swapped in each sentence, creating modified versions denoted as x′. Next, we analyze the impact of word order changes on the model’s predictions by examining the predicted token softmax probability distribution from the original sentence x and its modified counterpart x′. Specifically, we examine the JSD between the two distributions to quantify the divergence in attention or focus resulting from the random word swaps in x′. Since there are no datasets that study word order, we compare our self-supervised approach to the LRS Score established in the previous section.

WORD ORDER SCORE = median{JSD(f(x)j+1||f(x′)j′+1) ∀(x,x′) ∈ X}, where j is the last token for the input sequence for x and j′ is the last token for x′.

Word Order Score vs LRS Score

0.065

0.060

WordOrderScore

0.055

0.050

0.045

0.040

Model Family

LLaMA

0.035

MPT

Pythia

GPT-J

0.030

Neo

GPT-2

0.025

0.18 0.19 0.20 0.21 0.22

LRS Score

Different Instruction Finetuning Methods (Word Order)

| |None Human Curated<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Self-Instruct Mix<br><br>| |
|---|
<br><br>ShareGPT| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.14

0.12

WordOrderScore

0.10

0.08

0.06

0.04

0.02

0.00

Pythia (2.8B) Pythia (6.9B) GPT-J (6B) MPT LLaMA (7B)

Base Model

- Figure 9: (Left) Word Order Score vs LRS Score across various model sizes and families. (Right) Word Order Score of instruction finetuned models and their pretrained counterparts.

##### 7.1 Experimental Set-up

For this experiment, we take our corpus and break it down into sentences. Then, for every sentence, we swap two random words (not tokens) to construct our x′ over 5000 examples. Due to the long-tailed distribution in scores that were observed over the 5000 examples, we report the median, as described. For reference, if we had computed the mean, we would observe a standard error 2e−3. We report the median JSD for each model, again including Pythia, Neo, GPT-2, and others. We run all models greater than 6B parameters in their FP16 configuration.

##### 7.2 Results

- From Figure 9 (Left), we can see that there is a positive correlation between Word Order Score and LRS Score. The higher the Word Order Score, the higher the LRS Score. Nevertheless, we can see that there appears to be a plateau for Word Score. Similar to the LRS Score, we see that larger models are more sensitive to word order, with the Mosaic MPT-7B and GPT-J model being the most sensitive to word order.

- Effect of Instruction Finetuning: Figure 9 (Right) shows that most instruction finetuning approaches make the model more sensitive to word order over the five models studied. Particularly, we see that only finetuning on the human-curated databricks-dolly-15k seems to make the model more sensitive irrespective of the size.

Limitations: For this Word Order Score, we make the assumption that the next token prediction when swapping two words randomly is a good proxy to measure a model’s sensitivity to word order.

### 8 Tokenization Sensitivity: Broken Token Metric

Text pre-processing is rarely perfect. Raw text often contains extra spaces, weird formatting, and other quirks that affect how the tokenization of the text occurs. HELM explored some of these phenomena [Liang et al., 2022]. Others, such as Rumbelow and Mwatkins [2023], found anomalous tokens that represent failure modes in GPT-2 and GPT-3 models, showing that our understanding of how different tokenization impacts the model behavior is still limited.

Self-Supervised Approach: To quantify this phenomenon, we randomly chop strings of raw input text at regular intervals of x, and then we tokenize each of the chopped strings independently. This way, we mimic a “broken” tokenization, that might occur in the pretraining corpus due to document breaks and misspellings. A broken tokenization can also occur during model generation when incomplete user input is provided [Microsoft, 2023]. After tokenizing each chopped string separately, we concatenate these tokenizations back together. Note that the original content is unchanged – the alternative tokenization still decodes to the same raw input text. We then compare the concatenation of chopped tokenization to the original text over the next token prediction using JSD, similar to our Word Order Metric.

1 n

TOKENIZATION SENSITIVITY SCORE =

JSD(f(x)j+1||f(x′)j′+1)

##### 8.1 Experimental Set-up

For this experiment, we take our corpus and break it down into sentences. Then, for every sentence, we apply our procedure (described above) to construct x′ over 1000 examples. We report the mean JSD for each different model like Pythia, Neo, GPT-2, and others, where the standard error is about 5e−3 for all models. We run all models greater than 6B parameters in their FP16 configuration. Here, we specifically explore a split stride of 5, splitting every 5th character.

##### 8.2 Results

- From Figure 10 (Left), we see that MPT and LLaMA are the least sensitive (lower is better) to changes in token inputs. More broadly, we observe a negative trend with training FLOPs (i.e increasing the FLOPs decreases the sensitivity to tokenization changes). We suspect that as the amount of training increases, alternative tokenizations are more likely to be observed, and invariance to these abnormal tokenizations increases. This is supported by measurements on the OPT models, which are strong outliers in the trend observed above. Each of these models was trained on only 180B tokens, less than

Tokenization Sensitivity Score vs FLOPS

Model Family

0.45

LLaMA

TokenizationSensitivityScore

MPT

Pythia

0.40

GPT-J

Neo

0.35

GPT-2

OPT

0.30

0.25

0.20

0.15

0.10

1021 1022

Approx. FLOPS

Different Instruction Finetuning Methods (Tokenization)

0.25

| |None Human Curated<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Self-Instruct Mix<br><br>| |
|---|
<br><br>ShareGPT| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.20

TokenizationScore

0.15

0.10

0.05

0.00

Pythia (2.8B) Pythia (6.9B) GPT-J (6B) MPT LLaMA (7B)

Base Model

- Figure 10: (Left) Tokenization Sensitivity Score with a split stride of five versus Approx. FLOPS – lower is better. Note that the OPT models have seen the fewest tokens during training, c.f. Figure 22. (Right) Impact of different instruction-tuned methods.

a fifth of the tokens seen by MPT and LLaMA (1 Trillion) and about half of what GPT-2, GPT-Neo, and Pythia have seen. We include Figure 22 for a variant of Figure 10 in terms of tokens observed during training in the appendix.

- Effect of Instruction Finetuning: Figure 10 (Right) shows the impact of different instruction finetuning methods. In contrast to previously observed metrics, there seems to be no reliable trend in tokenization robustness after instruction finetuning. Furthermore, even when only model size differs (Dolly-V2s) the instruction finetuned dataset can have a different impact on this metric. It is worth noting that the Dolly-V2s were only trained on 15k instructions.

Limitations We test a limited type – character splits – of tokenization error, particularly the same text just being processed differently by the tokenizer. There are additional tokenization errors to consider as well, based on minor edits of the raw input text (i.e explicit word splits, extra spaces, unusual punctuation, etc), that could also be considered. Additionally, we examined the change in the next token probabilities, as we believe it is a good proxy to measure this phenomenon.

### 9 Discussion

In this paper, we introduce a new framework for Self-Supervised Evaluation for LLMs using sensitivity (invariance) metrics. We show that sensitivity measurements like the ones explored in this paper – knowledge via negations, toxicity, context, word order, and tokenization robustness – can correlate with existing evaluation datasets, as we verify for the knowledge and context sensitivity metrics. We conclude that sensitivity metrics can provide meaningful insights into model behavior, which we also verify qualitatively in our study of the Cohere command model. Additionally, we see generally, except for toxicity, that larger models have better sensitivity scores compared to smaller models, mirroring other benchmarks that verify that model performance generally increases with scale. However, there are still things to consider when analyzing these models using Self-Supervised Evaluation, which we will outline in this section.

For example, in some instances like text-ada-001 in knowledge, we see that being more sensitive is a byproduct of some other phenomena. Similarly, it may be that certain models are just insensitive in general to any transformations. This may be the case for tiny models, like the toxicity metric for GPT-2 (small) and the tokenization metric for Pythia-160M. This implies that there is a lower limit of model size where certain sensitivity metrics cease to be meaningful predictors of model qualities.

Model Entropy. The entropy of a model’s output distribution can impact many aspects of text generation. A lower entropy may require a more aggressive sampling strategy for text generation to achieve a diverse set of generations from the model, or might indicate a miscalibration of the output distribution. Similarly, the model’s entropy can affect sensitivity scores.

If the entropy of the model is low, then the sensitivity may naturally be lower as well. The exact impact of the model’s entropy on these sensitivity scores and how to appropriately incorporate it into invariances/sensitivity scores should be explored in future work. Figure 11 shows the Shannon Entropy of the Next Token Prediction and Sentence Entropy (the mean token entropy over a sentence of the model). We use the Wikipedia (our corpus) sentences to calculate the Shannon Entropy, defined as H(x) = − p(x)log(p(x)). From Figure 11, we see that LLaMA has the lowest entropy on both the next token and mean token over a sentence, with large models having a lower entropy than smaller models on average. This may partially explain why the sensitivity scores for LLaMA are lower. 4

Next Token Entropy vs Sentence Entropy

Model

Pythia (70M)

4.2

Pythia (160M)

GPT-2 (small)

NextTokenEntropy

GPT-Neo (125M) GPT-2 (medium) GPT-2 (large)

4.1

4.0

- OPT (1.3B)

- GPT-Neo (1.3B)

OPT (2.7B) GPT-2 (XL)

- GPT-Neo (2.7B)

3.9

3.8

- Pythia (1.4B)

- Pythia (2.8B)

3.7

Pythia (6.9B)

GPT-J (6B)

MPT (7B)

3.6

LLaMA (7B)

2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50 4.75

Sentence Entropy

Figure 11: Plot showing the next token prediction Shannon entropy (y-axis) and mean token Shannon entropy (x-axis) over sentences on Wikipedia. We find that LLaMA (7B) has the lowest entropy over the next token and mean token over a sentence.

4Vocabulary size does play an additional role in the entropy of a model. For example, in a completely uniform distribution, the Shannon Entropy of a model with a smaller vocabulary size will be smaller than another model with a larger vocabulary size.

Knowledge

LLM (7B)

1.0

LLaMA

MPT

0.75

Long-Range

Pythia

0.3

0.5

0.225

0.15

0.25

0.075

0.75 0.5 0.25 0.0

Toxicity

0.025 0.05

0.15

0.075 0.1

0.1

0.05

Word Order

0.0

Tokenization

Knowledge

Instruction (7B)

1.0

Vicuna

MPT-Instruct

0.75

Long-Range

Dolly V2

0.3

0.5

0.225

0.15

0.25

0.075

0.75 0.5 0.25 0.0

Toxicity

0.025 0.05

0.15

0.075 0.1

0.1

0.05

Word Order

0.0

Tokenization

- Figure 12: (Left) shows LLaMA, MPT, and Pythia sensitivity scores across the five metrics studied in this paper. (Right) shows the instruction-tuned counterparts of these models across the five metrics. The more area that is covered, the better the model according to our SSE scheme. All models 7B were run in FP16 configurations.

Memorization. Machine learning evaluation benchmarks for studying statistical generalization almost always assume idealized train and test set separation. However, in reality, some amount of overlap often exists in modern web-scale pre-training corpora. As a result, there have been various efforts to measure and address the impact of these overlaps on the training and evaluation of large models [Brown et al., 2020, Gao et al., 2021]. Investigating the same relationship, purely from a training support perspective, Kandpal et al. [2022] showed that a language model’s ability to answer a fact-based question relates to how many documents associated with that question were seen during pre-training. In a different but fundamentally related line of work, Carlini et al. [2022] demonstrated that LLMs regurgitate training data in specific scenarios, often based on repetition rates in training corpora. Further, their own prior work [Carlini et al., 2020] quantifies the underlying relationship between train and test data in yet another way by showing that simple loss-based membership inference methods are capable of discriminating whether a test query was present in the training dataset. In the context of sensitivity scores, this collection of results in the literature suggests that it is hard to make strong statements about whether training-time exposure to certain documents or token sequences would confound the trends observed in our proposed sensitivity metrics. We leave a detailed analysis of the interactions between memorization behaviors based on training data and our sensitivity metrics for future research. We suspect that developing a more complete understanding of these interactions is an important step towards more informative and robust sensitivity metrics.

An advantage of self-supervised sensitivity scores is that we can circumvent the potential effects of memorization by evaluating sensitivities on novel text, i.e., the latest news articles, as no labeling and additional curation of data sources is required. With this strategy, the possibility of memorization can be eliminated.

### 10 Conclusion

We introduce a procedure for self-supervised evaluation by analyzing invariances for Large Language Models. The key advantage of self-supervised evaluation is that it removes the need to laboriously label new data, leading to more efficient forms of evaluation in real deployment settings. We showcase several case studies, where we empirically validate this approach to be reliably tracking existing supervised metrics. Additionally, there are a number of future questions to consider when measuring a model’s sensitivity that we did not fully explore yet – like entropy and memorization. Nevertheless, these self-supervised evaluation approaches have the potential to measure properties beyond what is currently capable of the traditional dataset approach – like sensitivity to word order. We hope that this is only a starting point for self-supervised metrics in the future that can lead to a deeper understanding of how LLMs behave and complement classical supervised benchmarks.

### 11 Acknowledgements

This work was made possible by the ONR MURI program, the Office of Naval Research (N000142112557), and the AFOSR MURI program. Commercial support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Further support was provided by the National Science Foundation (IIS-2212182), and by the NSF TRAILS Institute (2229885).

### References

Anthropic. Introducing 100k context windows, May 2023a. URL https://www.anthropic.com/

index/100k-context-windows.

Anthropic. Introducing claude, March 2023b. URL https://www.anthropic.com/index/

introducing-claude.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. arXiv preprint arXiv:2304.01373, 2023.

Abeba Birhane, Pratyusha Kalluri, Dallas Card, William Agnew, Ravit Dotan, and Michelle Bao. The values encoded in machine learning research. In 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 173–184, 2022.

Samuel R Bowman and George E Dahl. What will it take to fix benchmarking in natural language understanding? arXiv preprint arXiv:2104.02145, 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

N Carlini, F Tramer, E Wallace, M Jagielski, A Herbert-Voss, K Lee, A Roberts, T Brown, D Song, Ú Erlingsson, et al. Extracting training data from large language models. arxiv. Preprint posted online December, 14, 2020.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. arXiv preprint arXiv:2202.07646, 2022.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness, 2022.

Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. Bold: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, page 862–872, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188.3445924. URL https://doi.org/10.1145/3442188.3445924.

Kaustubh D Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Shrivastava, Samson Tan, et al. Nl-augmenter: A framework for task-sensitive natural language augmentation. arXiv preprint arXiv:2112.02721, 2021.

Kawin Ethayarajh and Dan Jurafsky. Utility is in the eye of the user: A critique of NLP leaderboards. arXiv preprint arXiv:2009.13888, 2020.

Allyson Ettinger. What bert is not: Lessons from a new suite of psycholinguistic diagnostics for language models. Transactions of the Association for Computational Linguistics, 8:34–48, 2020.

Paula Fortuna, Juan Soler, and Leo Wanner. Toxic, hateful, offensive or abusive? what are we really classifying? an empirical analysis of hate speech datasets. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 6786–6794, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https:

###### //aclanthology.org/2020.lrec-1.838.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021. URL https://doi.org/10.5281/zenodo.5371628.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models, September 2020. URL http://arxiv.org/abs/2009.11462. arXiv:2009.11462 [cs].

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601– 1611, 2017.

Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=hY6M0JHl3uL.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to learn long-tail knowledge. arXiv preprint arXiv:2211.08411, 2022.

Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, et al. Dynabench: Rethinking benchmarking in nlp. arXiv preprint arXiv:2104.14337, 2021.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

Microsoft. Guidance. Microsoft, June 2023. URL https://github.com/microsoft/guidance. Simon Mille, Kaustubh D. Dhole, Saad Mahamood, Laura Perez-Beltrachini, Varun Gangal, Mihir

Kale, Emiel van Miltenburg, and Sebastian Gehrmann. Automatic construction of evaluation suites for natural language generation datasets. ArXiv, abs/2106.09069, 2021.

Joe O’Connor and Jacob Andreas. What context features can transformer language models use? arXiv preprint arXiv:2106.08367, 2021.

OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, 2016.

Luiza Pozzobon, Beyza Ermis, Patrick Lewis, and Sara Hooker. On the challenges of using black-box apis for toxicity evaluation in research. arXiv preprint arXiv:2304.12397, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4902–4912, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.442. URL https://aclanthology.

###### org/2020.acl-main.442.

Alexis Ross, Tongshuang Sherry Wu, Hao Peng, Matthew E. Peters, and Matt Gardner. Tailor: Generating and perturbing text with semantic controls. In Annual Meeting of the Association for Computational Linguistics, 2021.

###### Jessica Rumbelow and Mwatkins. Solidgoldmagikarp (plus, prompt generation), 2023. URL https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/ solidgoldmagikarp-plus-prompt-generation.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.

Hao Sun, Guangxuan Xu, Jiawen Deng, Jiale Cheng, Chujie Zheng, Hao Zhou, Nanyun Peng, Xiaoyan Zhu, and Minlie Huang. On the safety of conversational models: Taxonomy, dataset, and benchmark. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3906–3923, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/ v1/2022.findings-acl.308. URL https://aclanthology.org/2022.findings-acl.308.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations, 2020.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality, 2022.

Tongshuang Sherry Wu, Marco Tulio Ribeiro, Jeffrey Heer, and Daniel S. Weld. Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models. In Annual Meeting of the Association for Computational Linguistics, 2021.

Guanqun Yang, Mirazul Haque, Qiaochu Song, Wei Yang, and Xueqing Liu. Testaug: A framework for augmenting capability-based nlp tests. In International Conference on Computational Linguistics, 2022.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=KRLUvxh8uaX.

### A Appendix

##### A.1 Knowledge Probing via Negations

Example: Figure 13 shows an example of the original x and the transformed x′ for the Knowledge Probing via Negations experiments.

|Original (x): April is the fourth month of the year in the Julian and Gregorian calendars and comes between March and May.<br><br>Perturbed (x′): April is not the fourth month of the year in the Julian and Gregorian calendars and comes between March and May.<br><br>|
|---|

- Figure 13: Knowledge probing via negations example over topic sentences in wikipedia. (Top) is the original, x, from wikipedia. (Bottom) is the transformed, x′, where we add a “not” according to the rules described in the main paper.

Adding Negations in TriviaQA To understand whether adding negations and measuring the change in log perplexity is a reasonable assessment of probing the knowledge in an LLM, we added negations to questions following the same rule described in the main paper. We then recorded the change in perplexity for each of the models given the question-answer pair. This was to understand how different models may understand negations. Figure 14 (Left) shows that adding a negation in the question and observing the change in perplexity can give us an indication of performance on TriviaQA.

TriviaQA Accuracy We calculate the accuracy for TriviaQA for the unfiltered-web-dev split by simply counting a correct answer from the model if one of the given answers was contained in the output string. Additionally, since we found that the answer list sometimes had the answer entity in the question, we excluded these answers when calculating accuracy. We use the template “Question: [input question]\nAnswer:”.

Models From Huggingface: gpt2, gpt2-large, gpt2-medium, gpt2-xl, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b, EleutherAI/pythia-1.4b, EleutherAI/pythia-2.8b, EleutherAI/pythia-6.9b, mosaicml/mpt-7b, mosaicml/mpt-7b-instruct, databricks/dolly-v1-6b, databricks/dolly-v2-3b, databricks/dolly-v2-7b

Other Models: LLaMA-base-7B, Vicuna-7B OpenAI API Models: ada, babbage, curie, davinci, text-ada-001, text-babbage-001, text-curie-001,text-davinci-003 Cohere API Models: xlarge, command-xlarge-beta

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

TriviaQA Acc.

0.10

0.15

0.20

0.25

0.30

0.35

0.40

SensitivityScoreonTriviaQA

Sensitivity Score on TriviaQA vs TriviaQA Acc.

Model Family

OpenAI Instruct

OpenAI Cohere Cohere Instruct

Llama

MPT

Pythia

Dolly V2 Instruct

GPT-J

Dolly V1 Instruct

Neo

GPT-2 Instruction

No

Yes

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.55 0.60 0.65 0.70 0.75

Sensitivity Score (Wikipedia)

0.40

0.45

0.50

0.55

0.60

0.65

0.70

0.75

AnswerinOutputAfterNegation

Sensitivity Score (Wikipedia) vs Answer in Output After Negation

Model Family

OpenAI Instruct Cohere Instruct

Dolly V2 Dolly V1

- Figure 14: (Left) The change in perplexity in the question-answer pair when a negation is applied to the question versus TriviaQA Acc. There appears to be a square-root relationship between the Sensitivity Score on TriviaQA versus TriviaQA Acc. (Right) The percentage of times when the correct answer was contained in the solution even when applying the negation versus Sensitivity Score (Wikipedia) for a few instruction models. We see that text-ada-001 changes its answer often, whereas the Cohere model does not.

##### A.2 Toxicity

Example: Figure 15 shows an example of the original x and the transformed x′ for the toxicity experiments.

|Original (x): Lionel Andrés Messi is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team.<br><br>Transformed (x′): Lionel Andrés Messi is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team. F*** F***|
|---|

Figure 15: Toxicity test with one sentence. (Top) is the original, x, from wikipedia. (Bottom) is the transformed, x′, where we add the trigger phrase to the original input.

Increasing the number of F-bombs: From Figure 16, we see, that on average, increasing the number of F-bombs increases the toxicity of the model. However, here two F-bombs induce more toxic behavior than three F-bombs, suggesting that on average, the two F-bombs might be a trigger for toxic generation.

Models From Huggingface: gpt2, gpt2-large, gpt2-medium, gpt2-xl, EleutherAI/gpt-j-6b, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/pythia-1.4b, EleutherAI/pythia-2.8b, EleutherAI/pythia-6.9b, mosaicml/mpt-7b, mosaicml/mpt-7b-instruct, databricks/dolly-v1-6b, databricks/dolly-v2-3b, databricks/dolly-v2-7b

Other Models: LLaMA-base-7B, Vicuna-7B, WizardLM-7B

##### A.3 Context (Long-Range) Sensitivity

Example: Figure 17 shows an example of the original x and the transformed x′ for the LRS experiments.

Increasing the Amount of Context: From Figure 18, we see that increasing the context (or the number of sentences swapped) increases the sensitivity. For the 7B parameter range, we see that Pythia (6.9B) is the most sensitive.

Toxicity Ablation -- Increasing F-Bombs

Model

GPT-2 (small) Pythia (6.9B) GPT-2 (medium)

0.8

GPT-2 (large)

0.6

ToxicFrac.

- Pythia (1.4B)

GPT-J (6B)

GPT-2 (XL)

LLaMA (7B)

MPT (7B)

- GPT-Neo (1.3B)

Pythia (2.8B)

- GPT-Neo (2.7B)

0.4

0.2

0.0

1 2 3 4 5 6 7 Number of F-Bombs Appended

- Figure 16: As we increase the number of F-bombs, the toxicity of the generation increases except when two F-bombs are present, which is a notable outlier. This suggests that to most models this is a toxic trigger. We measure toxicity over the generated text by observing whether a term from LDNOOBW is contained in the generation. From this figure, we see GPT-Neo (2.7B) is the most toxic according to our metric.

|Original (x): Lyrically, the song begins with the absence of her man, but then, in the chorus, transitions into a warning not to fall in love with material things. The second track, “Lágrimas Cálidas” (“Warm Tears”), is a vallenato-stylized pop ballad, expressing her suffering due to being abandoned by her lover.“Te Arrepentiras” (“You’ll Regret”), is about a woman who surrendered completely to a man who did not appreciate her.<br><br>Transformed (x′): Ireland has won more medals in boxing than in any other Olympic sport. Boxing is governed by the Irish Amateur Boxing Association. “Te Arrepentiras” (“You’ll Regret”), is about a woman who surrendered completely to a man who did not appreciate her.|
|---|

###### Figure 17: Long-Range Sensitivity test with four sentences. (Top) is the original, x, from wikipedia. (Bottom) is the transformed, x′, where the first two sentences are replaced with random two sentences from wikipedia.

LRS Ablation -- Number of Sentences Swapped

Model OPT (2.7B) OPT (1.3B) Pythia (6.9B)

0.30

0.25

GPT-J (6B)

LLaMA (7B)

0.20

Pythia (2.8B)

GPT-Neo (2.7B) GPT-Neo (1.3B) Pythia (1.4B)

MeanJSD

0.15

MPT (7B)

GPT-2 (XL)

0.10

GPT-2 (medium)

GPT-2 (large)

GPT-Neo (125M)

0.05

GPT-2 (small)

Pythia (160M)

0.00

Pythia (70M)

0 1 2 3 4 5 6 Number of Sentences Swapped

###### Figure 18: Increasing the context length (the number of swapped sentences) increases, the change in the probability distribution over the last sentence.

Models From Huggingface: gpt2, gpt2-large, gpt2-medium, gpt2-xl, facebook/opt-1.3b, facebook/opt-2.7b, EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b, EleutherAI/pythia-70M, EleutherAI/pythia-160m, EleutherAI/pythia-410m, EleutherAI/pythia-1b, EleutherAI/pythia-1.4b, EleutherAI/pythia-2.8b, EleutherAI/pythia-6.9b, mosaicml/mpt-7b, mosaicml/mpt-7b-instruct, databricks/dolly-v1-6b, databricks/dolly-v2-3b, databricks/dolly-v2-7b, databricks/dolly-v2-7b

Other Models: LLaMA-base-7B, Vicuna-7B

- A.4 Word Order Sensitivity

Example: Figure 19 shows an example of the original x and the transformed x′ for the word order experiments.

|Original (x): Media.Vision would return to the franchise with the development of Valkyria: Azure Revolution for the PlayStation 4.<br><br>Transformed (x′): Media.Vision would return PlayStation the franchise with the development of Valkyria : Azure Revolution for the to 4.<br><br>|
|---|

###### Figure 19: Word Order Sensitivity test over one sentence. (Top) is the original, x, from wikipedia. (Bottom) is the transformed, x′, where two words are randomly flipped. This is a 1-Swap.

Table 2: Example sentence of the transformation with a split stride of 10. (Left) shows the original unaltered sentence. (Right) shows the transformed sentence after splitting every 10th character. The underlined dashes are where the sentence is split.

Original (x) Transformed (x′) Media.Vision would return to the franchise with the development of Valkyria: Azure Revolution for the PlayStation.

Media.Visi–on would r–eturn to t–he franchi–se with th–e developm–ent of Val–kyria: Azu–re Revolut–ion for th–e PlayStat–ion 4.

Different Number of Swaps: Figure 20 shows the median JSD on the next token as we increase the swaps. Here, we see increasing the number of swaps increases the sensitivity.

Word Order Ablation -- Number of Word Swaps

Model GPT-J (6B)

0.5

MPT (7B)

Pythia (2.8B) Pythia (6.9B) LLaMA (7B)

0.4

GPT-Neo (2.7B)

JSDMedian

OPT (6.7B)

0.3

Pythia (1.4B)

GPT-Neo (1.3B)

GPT-2 (XL)

0.2

GPT-Neo (125M)

GPT-2 (large)

GPT-2 (medium)

0.1

GPT-2 (small)

0.0

0 2 4 6 8 10 Number of Words Swapped

Figure 20: We plot JSD on the next token prediction against the number of swaps for the token.

Models From Huggingface: gpt2, gpt2-large, gpt2-medium, gpt2-xl, EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b, EleutherAI/pythia-1.4b, EleutherAI/pythia-2.8b, EleutherAI/pythia-6.9b, mosaicml/mpt-7b, mosaicml/mpt-7b-instruct, databricks/dolly-v1-6b, databricks/dolly-v2-3b, databricks/dolly-v2-7b

Other Models: LLaMA-base-7B, Vicuna-7B

##### A.5 Tokenization Sensitivity

Example: Figure 19 shows an example of the original x and the transformed x′ for the tokenization experiments.

Incresing Split Stride: Figure 21 shows the median JSD on the next token as we increase the split stride. Here, we see that LLaMA and MPT are much less sensitive (better at handling tokenization changes) regarding the change in the probability distribution over the next token as we increase the split stride. Figure 22 shows the number of tokens seen versus the tokenization sensitivity score. Here, we see that there is a negative correlation.

Models From Huggingface: gpt2, gpt2-large, gpt2-medium, gpt2-xl, facebook/opt-1.3b, facebook/opt-2.7b, EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b, EleutherAI/pythia-160m, EleutherAI/pythia-410m, EleutherAI/pythia-1b, EleutherAI/pythia-1.4b, EleutherAI/pythia-2.8b, EleutherAI/pythia-6.9b, mosaicml/mpt-7b,mosaicml/mpt-7b-instruct, databricks/dolly-v1-6b, databricks/dolly-v2-3b, databricks/dolly-v2-7b, databricks/dolly-v2-7b

Other Models: LLaMA-base-7B, Vicuna-7B

Tokenization Ablation -- Increasing Split Stride

0.6

Model

MPT (7B)

LLaMA (7B)

0.5

Pythia (160M)

Pythia (6.9B)

GPT-2 (XL)

0.4

GPT-J (6B)

GPT-2 (large)

JSDMean

GPT-Neo (2.7B) GPT-Neo (1.3B)

0.3

- Pythia (1.4B)

- Pythia (2.8B)

0.2

GPT-2 (medium)

GPT-2 (small)

GPT-Neo (125M)

0.1

OPT (2.7B) OPT (1.3B)

0.0

2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Split Stride (by Character)

- Figure 21: Increasing the split stride decreases the sensitivity. We see that the OPT family cannot handle this type of transformation. Additionally, we see LLaMA and MPT are good at handling these types of tokenization changes. Lower is better.

2×10 103

2 3 × 102 4 × 102 6 × 102

Total Tokens Seen During Training

0.10

0.15

0.20

0.25

0.30

0.35

0.40

0.45

TokenizationSensitivityScore

Tokenization Sensitivity Score vs Tokens Seen

Model Family

LLaMA

MPT

Pythia

GPT-J

Neo

GPT-2

OPT

- Figure 22: Increasing the total number of tokens seen during training decreases the sensitivity score. We see that the OPT family is the most sensitive to this type of transformation, as they have seen the least number of tokens. Additionally, we see LLaMA and MPT are good at handling these types of tokenization changes as they have seen more tokens. Lower is better.

##### A.6 Additional Experiment Details

For all these experiments, we use NVIDIA RTX A4000 GPUs, finding that evaluating most models is quite inexpensive over 1000 examples, with compute requirements of less than 30 min per model for most tests. Additionally, for sentence and word parsing/tokenization, we use the nltk package.

