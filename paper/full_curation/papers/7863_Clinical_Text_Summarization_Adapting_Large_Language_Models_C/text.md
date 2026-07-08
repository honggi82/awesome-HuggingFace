# arXiv:2309.07430v5[cs.CL]11Apr2024

## Adapted Large Language Models Can Outperform Medical Experts in Clinical Text Summarization

Dave Van Veen†, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali‡, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerová, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari

Stanford University

Analyzing vast textual data and summarizing key information from electronic health records imposes a substantial burden on how clinicians allocate their time. Although large language models (LLMs) have shown promise in natural language processing (NLP), their effectiveness on a diverse range of clinical summarization tasks remains unproven. In this study, we apply adaptation methods to eight LLMs, spanning four distinct clinical summarization tasks: radiology reports, patient questions, progress notes, and doctor-patient dialogue. Quantitative assessments with syntactic, semantic, and conceptual NLP metrics reveal trade-offs between models and adaptation methods. A clinical reader study with ten physicians evaluates summary completeness, correctness, and conciseness; in a majority of cases, summaries from our best adapted LLMs are either equivalent (45%) or superior (36%) compared to summaries from medical experts. The ensuing safety analysis highlights challenges faced by both LLMs and medical experts, as we connect errors to potential medical harm and categorize types of fabricated information. Our research provides evidence of LLMs outperforming medical experts in clinical text summarization across multiple tasks. This suggests that integrating LLMs into clinical workflows could alleviate documentation burden, allowing clinicians to focus more on patient care.

###### 1 Introduction

Documentation plays an indispensable role in healthcare practice. Currently, clinicians spend significant time summarizing vast amounts of textual information—whether it be compiling diagnostic reports, writing progress notes, or synthesizing a patient’s treatment history across different specialists [1–3]. Even for experienced physicians with a high level of expertise, this intricate task naturally introduces the possibility for errors, which can be detrimental in healthcare where precision is paramount [4–6].

The widespread adoption of electronic health records has expanded clinical documentation workload, directly contributing to increasing stress and clinician burnout [7–9]. Recent data indicates that physicians can expend up to two hours on documentation for each hour of patient interaction [10]. Meanwhile, documentation responsibilities for nurses can consume up to 60% of their time and account for significant work stress [11–13]. These tasks divert attention from direct patient care, leading to worse outcomes for patients and decreased job satisfaction for clinicians [2, 14–16].

In recent years, large language models (LLMs) have gained remarkable traction, leading to widespread adoption of models such as ChatGPT [17], which excel at information retrieval, nuanced understanding, and text generation [18, 19]. Although LLM benchmarks for general natural language processing (NLP) tasks exist [20, 21], they do not evaluate performance on relevant clinical tasks. Addressing this limitation presents an opportunity to accelerate the process of clinical text summarization, hence alleviating documentation burden and improving patient care.

Crucially, machine-generated summaries must be non-inferior to that of seasoned clinicians—especially when used to support sensitive clinical decision-making. Previous work has demonstrated potential across clinical

Published in Nature Medicine † Corresponding author: vanveen@stanford.edu ‡ The University of Texas at Austin

|open-source|
|---|

|proprietary|
|---|

|Clinical reader study<br><br>to evaluate correctness, completeness, and conciseness via ten physicians<br><br>[Figure 1]<br><br>[Figure 2]<br><br>five hospitalists five radiologists<br><br>[Figure 3]|
|---|

Med-Alpaca Alpaca FLAN-UL2

FLAN-T5

Llama-2 Vicuna

- GPT-3.5
- GPT-4

Models

|Quantitative evaluation<br><br>to measure syntactic, semantic, and conceptual similarity via NLP metrics<br><br>[Figure 4]| |
|---|---|
| | |

|best model/method<br><br>Safety analysis<br><br>to quantify potential medical harm and categorize fabricated information<br><br>[Figure 5]|
|---|

### ×

|Methods<br><br>used for adaptation<br><br>QLoRA in-context learning|
|---|

best model/method

### ×

|Tasks<br><br>patient health questions<br><br>radiology reports progress notes patient/doctor dialogue|
|---|

- Figure 1 | Framework overview. First, we quantitatively evaluate each valid combination (×) of LLM and adaptation method across four distinct summarization tasks comprising six datasets. We then conduct a clinical reader study in which ten physicians compare summaries of the best model/method against those of a medical expert. Lastly, we perform a safety analysis to quantify potential medical harm and to categorize types of fabricated information.

NLP tasks [22, 23], adapting to the medical domain by either training a new model [24, 25], fine-tuning an existing model [26, 27], or supplying domain-specific examples in the model prompt [27, 28]. However, adapting LLMs to summarize a diverse set of clinical tasks has not been thoroughly explored, nor has non-inferiority to medical experts been achieved. With the overarching objective of bringing LLMs closer to clinical readiness, we make the following contributions:

- • We implement adaptation methods across eight open-source and proprietary LLMs for four distinct summarization tasks comprising six datasets. The subsequent evaluation via NLP metrics provides a comprehensive assessment of contemporary LLMs for clinical text summarization.
- • Our exploration delves into a myriad of trade-offs concerning different models and adaptation methods, shedding light on scenarios where advancements in model size, novelty, or domain specificity do not necessarily translate to superior performance.
- • Through a clinical reader study with ten physicians, we demonstrate that LLM summaries can surpass medical expert summaries in terms of completeness, correctness, and conciseness.
- • Our safety analysis of examples, potential medical harm, and fabricated information reveals insights into the challenges faced by both models and medical experts.
- • We identify which NLP metrics most correlate with reader preferences.

Our study demonstrates that adapting LLMs can outperform medical experts for clinical text summarization across the diverse range of documents we evaluate. This suggests that incorporating LLM-generated candidate summaries could reduce documentation load, potentially leading to decreased clinician strain and improved patient care.

###### 2 Related Work

Large language models (LLMs) have demonstrated astounding performance, propelled by both the transformer architecture [29] and increasing scales of data and compute, resulting in widespread adoption of models such as ChatGPT [17]. Although several of the more expansive models, such as GPT-4 [30] and PaLM [31], remain proprietary and provide access via “black-box” interfaces, there has been a pronounced shift towards open-sourced alternatives such as Llama-2 [32]. These open-source models grant researchers direct access to model weights for customization.

Popular transformer models such as BERT [33] and GPT-2 [34] established the paradigm of self-supervised pretraining on large amounts of general data and then adapting to a particular domain or task by tuning on specific data. One approach is customizing model weights via instruction tuning, a process where language models are trained to generate human-aligned responses given specific instructions [35]. Examples of clinical instruction-tuned models include Med-PALM [24] for medical question-answering or Radiology-GPT [36] for radiology tasks. To enable domain adaptation with limited computational resources, prefix tuning [37] and

low-rank adaptation (LoRA) [38] have emerged as effective methods that require tuning less than 1% of total parameters over a small training set. LoRA has been shown to work well for medical question-answering [26] and summarizing radiology reports [27]. Another adaptation method, requiring no parameter tuning, is in-context learning: supplying the LLM with task-specific examples in the prompt [39]. Because in-context learning does not alter model weights, it can be performed with black-box model access using only a few training examples [39].

Recent work has adapted LLMs for various medical tasks, demonstrating great potential for medical language understanding and generation [22, 23, 25, 40]. Specifically, a broad spectrum of methodologies has been applied to clinical text for specific summarization tasks. One such task is the summarization of radiology reports, which aims to consolidate detailed findings from radiological studies into significant observations and conclusions drawn by the radiologist [41]. LLMs have shown promise on this task [27] and other tasks such as summarizing daily progress notes into a concise “problem list” of medical diagnoses [42]. Lastly, there has been significant work on summarizing extended conversations between a doctor and patient into patient visit summaries [28, 43, 44].

While the aforementioned contributions incorporate methods to adapt language models, they often include only a small subset of potential approaches and models, and/or they predominantly rely on evaluation via standard NLP metrics. Given the critical nature of medical tasks, demonstrating clinical readiness requires including human experts in the evaluation process. To address this, there have been recent releases of expert evaluations for instruction following [3] and radiology report generation [45]. Other work employs human experts to evaluate synthesized Cochrane review abstracts, demonstrating that NLP metrics are not sufficient to measure summary quality [46]. With this in mind, we extend our comprehensive evaluation of methods and LLMs beyond NLP metrics to incorporate a clinical reader study across multiple summarization tasks. Our results demonstrate across many tasks that LLM summaries are comparable to–––and often surpass–––those created by human experts.

###### 3 Approach

###### 3.1 Large language models

We investigate a diverse collection of transformer-based LLMs for clinical summarization tasks. This includes two broad approaches to language generation: sequence-to-sequence (seq2seq) models and autoregressive models. Seq2seq models use an encoder-decoder architecture to map the input text to a generated output, often requiring paired datasets for training. These models have shown strong performance in machine translation [47] and summarization [48]. In contrast, the autoregressive models typically only use a decoder. They generate tokens sequentially—where each new token is conditioned on previous tokens—thus efficiently capturing context and long-range dependencies. Autoregressive models are typically trained with unpaired data, and they are particularly useful for various NLP tasks such as text generation, question-answering, and

- Table 1 | We quantitatively evaluate eight models, including state-of-the-art sequence-to-sequence and autoregressive models. Unless specified, models are open-source (vs. proprietary).

|Model<br><br>|Context Parameters<br><br>|Proprietary?|Seq2seq Autoreg.|
|---|---|---|---|
|FLAN-T5 FLAN-UL2 Alpaca Med-Alpaca Vicuna Llama-2<br><br>GPT-3.5<br>GPT-4<br>|512 2.7B 2,048 20B<br><br>2,048 7B 2,048 7B 2,048 7B 4,096 7B, 13B<br><br>16,384 175B<br><br>32,768* unknown<br><br>|✔ ✔<br><br>|✔ -<br>✔ -<br><br><br>- ✔<br>- ✔<br>- ✔<br>- ✔<br>- ✔<br>- ✔<br>|

*The context length of GPT-4 has since been increased to 128,000.

dialogue interactions [17, 49].

We include prominent seq2seq models due to their strong summarization performance [48] and autoregressive models due to their state-of-the-art performance across general NLP tasks [21]. As shown in Table 1, our choice of models varies widely with respect to number of parameters (2.7 billion to 175 billion) and context length (512 to 32,768), i.e. the maximum number of input tokens a model can process. We organize our models into three categories:

Open-source seq2seq models. The original T5 “text-to-text transfer transformer” model [50] demonstrated excellent performance in transfer learning using the seq2seq architecture. A derivative model, FLAN-T5 [51, 52], improved performance via instruction prompt tuning. This T5 model family has proven effective for various clinical NLP tasks [27, 53]. The FLAN-UL2 model [54, 55] was introduced recently, which featured an increased context length (four-fold that of FLAN-T5) and a modified pre-training procedure called unified language learning (UL2).

Open-source autoregressive models. The Llama family of LLMs [32] has enabled the proliferation of open-source instruction-tuned models that deliver comparable performance to GPT-3 [17] on many benchmarks despite their smaller sizes. Descendants of this original model have taken additional fine-tuning approaches, such as fine-tuning via instruction following (Alpaca [56]), medical Q&A data (Med-Alpaca [57]), user-shared conversations (Vicuna [49]), and reinforcement learning from human feedback (Llama-2 [32]). Llama-2 allows for two-fold longer context lengths (4,096) relative to the aforementioned open-source autoregressive models.

Proprietary autoregressive models. We include GPT-3.5 [58] and GPT-4 [30], the latter of which has been regarded as state-of-the-art on general NLP tasks [21] and has demonstrated strong performance on biomedical NLP tasks such as medical exams [59–61]. Both models offer significantly higher context length (16,384 and 32,768) than open-source models. We note that since sharing our work, GPT-4’s context length has been increased to 128,000.

###### 3.2 Adaptation methods

We consider two proven techniques for adapting pre-trained general-purpose LLMs to domain-specific clinical summarization tasks. To demonstrate the benefit of adaptation methods, we also include the baseline zero-shot prompting, i.e. m = 0 in-context examples.

In-context learning (ICL). ICL is a lightweight adaptation method that requires no altering of model weights; instead, one includes a handful of in-context examples directly within the model prompt [39]. This simple approach provides the model with context, enhancing LLM performance for a particular task or domain [27, 28]. We implement this by choosing, for each sample in our test set, the m nearest neighbors training samples in the embedding space of the PubMedBERT model [62]. Note that choosing “relevant” in-context examples has been shown to outperform choosing examples at random [63]. For a given model and dataset, we use m = 2x examples, where x ∈ {0,1,2,3,...,M} for M such that no more than 1% of the s = 250 samples are excluded due to prompts exceeding the model’s context length. Hence each model’s context length limits the allowable number of in-context examples.

Quantized low-rank adaptation (QLoRA). Low-rank adaptation (LoRA) [38] has emerged as an effective, lightweight approach for fine-tuning LLMs by altering a small subset of model weights—often < 0.1% [27]. LoRA inserts trainable matrices into the attention layers; then, using a training set of samples, this method performs gradient descent on the inserted matrices while keeping the original model weights frozen. Compared to training model weights from scratch, LoRA is much more efficient with respect to both computational requirements and the volume of training data required. Recently, QLoRA [64] has been introduced as a more memory-efficient variant of LoRA, employing 4-bit quantization to enable the fine-tuning of larger LLMs given the same hardware constraints. This quantization negligibly impacts performance [64]; as such, we use QLoRA for all model training. Note that QLoRA cannot be used to fine-tune proprietary models on our consumer hardware, as their model weights are not publicly available. Fine-tuning of GPT-3.5 via API was made available after our internal model cutoff date of July 31st, 2023 [65].

###### 3.3 Data

To robustly evaluate LLM performance on clinical text summarization, we choose four distinct summarization tasks, comprising six open-source datasets. As depicted in Table 2, each dataset contains a varying number of samples, token lengths, and lexical variance. Lexical variance is calculated as numbernumberofofuniquetotal wordswords across the entire dataset; hence a higher ratio indicates less repetition and more lexical diversity. We describe each task and dataset below. For task examples, please see Figures 8, A4, A5, and A6.

Radiology reports Radiology report summarization takes as input the findings section of a radiology study containing detailed exam analysis and results. The goal is to summarize these findings into an impression section, which concisely captures the most salient, actionable information from the study. We consider three datasets for this task, where both reports and findings were created by attending physicians as part of routine clinical care. Open-i [66] contains de-identified narrative chest x-ray reports from the Indiana Network for Patient Care 10 database. From the initial set of 4K studies, Demner-Fushman et al. [66] selected a final set of

###### 3.4K reports based on the quality of imaging views and diagnostic content. MIMIC-CXR [67] contains chest x-ray studies accompanied by free-text radiology reports acquired at the Beth Israel Deaconess Medical Center between 2011 and 2016. For this study, we use a dataset of 128K reports [68] preprocessed by RadSum23 at BioNLP 2023 [69, 70]. MIMIC-III [71] contains 67K radiology reports spanning seven anatomies (head, abdomen, chest, spine, neck, sinus, and pelvis) and two modalities: magnetic resonance imaging (MRI) and computed tomography (CT). This dataset originated from patient stays in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. For this study, we utilize a preprocessed version via RadSum23 [69, 70]. Compared to x-rays, MRIs and CT scans capture more information at a higher resolution. This usually leads to longer reports (Table 2), rendering MIMIC-III a more challenging summarization dataset than Open-i or MIMIC-CXR.

Patient questions Question summarization consists of generating a condensed question expressing the minimum information required to find correct answers to the original question [72]. For this task, we employ

- Table 2 | Top: Description of six open-source datasets with a wide range of token length and lexical variance, i.e. numbernumberofofuniquetotal wordswords. Bottom: Instructions for each of the four summarization tasks. See Figure 2 for the full prompt.

|Dataset descriptions| | |
|---|---|---|
|Dataset Task<br><br>| |Number Avg. number of tokens Lexical of samples Input Target variance|
|Open-i Radiology reports MIMIC-CXR Radiology reports MIMIC-III Radiology reports MeQSum Patient questions ProbSum Progress notes ACI-Bench Dialogue| |3.4K 52 ± 22 14 ± 12 0.11 128K 75 ± 31 22 ± 17 0.08<br><br>67K 160 ± 83 61 ± 45 0.09 1.2K 83 ± 67 14 ± 6 0.21<br><br>755 1,013 ± 299 23 ± 16 0.15 126 1,512 ± 467 211 ± 98 0.04<br><br>|
| | | |
|Task Instructions| | |
|Task<br><br>|Instruction| |
|Radiology reports<br><br>Patient questions<br><br>Progress notes<br><br>Dialogue<br><br>|“Summarize the radiology report findings into an impression with minimal text.”<br><br>“Summarize the patient health query into one question of 15 words or less.”<br><br>“Based on the progress note, generate a list of 3-7 problems (a few words each) ranked in order of importance.”<br><br>“Summarize the patient/doctor dialogue into an assessment and plan.”| |

the MeQSum dataset [72]. MeQSum contains (1) patient health questions of varying verbosity and coherence selected from messages sent to the U.S. National Library of Medicine (2) corresponding condensed questions created by three medical experts such that the summary allows retrieving complete, correct answers to the original question without the potential for further condensation. These condensed questions were then validated by a medical doctor and verified to have high inter-annotator agreement. Due to the wide variety of these questions, MeQSum exhibits the highest lexical variance of our datasets (Table 2).

Progress notes The goal of this task is to generate a “problem list,” or condensed list of diagnoses and medical problems using the provider’s progress notes during hospitalization. For this task, we employ the ProbSum dataset [42]. This dataset, generated by attending internal medicine physicians during the course of routine clinical practice, was extracted from the MIMIC-III database of de-identified hospital intensive care unit (ICU) admissions. ProbSum contains (1) progress notes averaging > 1,000 tokens and substantial presence of unlabeled numerical data, e.g. dates and test results, and (2) corresponding problem lists created by attending medical experts in the ICU. We utilize a version shared by the BioNLP Problem List Summarization Shared Task [42, 70, 73] and PhysioNet [74].

Dialogue The goal of this task is to summarize a doctor-patient conversation into an “assessment and plan” paragraph. For this task, we employ the ACI-Bench dataset [43, 44, 75], which contains (1) 207 doctor-patient conversations and (2) corresponding patient visit notes, which were first generated by a seq2seq model and subsequently corrected and validated by expert medical scribes and physicians. Since ACI-Bench’s visit notes include a heterogeneous collection of section headers, we choose 126 samples containing an “assessment and plan” section for our analysis. Per Table 2, this task entailed the largest token count across our six datasets for both the input (dialogue) and target (assessment).

As we are not the first to employ these datasets, Table A2 contains quantitative metric scores from other works [25, 27, 44, 76–78] who developed methods specific to each individual summarization task.

###### 4 Experiments

This section contains experimental details and study design for our evaluation framework, as depicted in Figure 1.

###### 4.1 Quantitative Evaluation

Building upon the descriptions of models, methods, and tasks in Section 3, we now specify experimental details such as model prompts, data preparation, software implementation, and NLP metrics used for quantitative evaluation.

###### 4.1.1 Model prompts and temperature

As shown in Figure 2, we structure prompts by following best practices [79, 80] and evaluating 1-2 options for model expertise and task-specific instructions (Table 2). We note the importance of specifying desired length in the instruction, e.g. “one question of 15 words or less” for summarizing patient questions. Without this specification, the model might generate lengthy outputs—occasionally even longer than the input text. While in some instances this detail may be preferred, we steer the model toward conciseness given our task of summarization.

Prompt phrasing and model temperature can have a considerable effect on LLM output, as demonstrated in the literature [81, 82] and in Figure 2. For example, we achieve better performance by nudging the model towards expertise in medicine compared to expertise in wizardry or no specific expertise at all. This illustrates the value of relevant context in achieving better outcomes for the target task. We also explore the temperature hyperparameter, which adjusts the LLM’s conditional probability distributions during sampling, hence affecting how often the model will output less likely tokens. Higher temperatures lead to more randomness and “creativity,” while lower temperatures produce more deterministic outputs. Figure 2

###### demonstrates that the lowest value, 0.1, performed best. We thus set temperature to this value for all models. Intuitively, a lower value seems appropriate given our goal of factually summarizing text with a high aversion to factually incorrect text.

|| |
|---|
|Input<br><br>... input m+1: [input text] summary m+1:|
| |
<br><br>|Expertise You are an expert medical professional.|
|---|
<br><br>| |
|---|
|(task-speciﬁc)<br><br>Instruction Summarize the [radiology report ﬁndings]<br><br>into an [impression with minimal text].|
| |
<br><br>| |
|---|
|Examples<br><br>i = 1, ..., m<br><br>#: delimiters for ICL only, else m = 0<br><br>Use the examples to guide word choice.<br><br>... input i: [example input] summary i: [example summary] ## ...|
| |
|
|---|

###### Parameter

Value 0.1 0.5 0.9 None Medicine1 Wizardry2

###### BLEU ROUGE-L BERTScore MEDCON

28.1 27.1 25.4 34.3 35.5 27.8

89.6 89.7 89.3 90.2 90.5 89.7

28.2 27.5 25.3 30.7 35.5 28.5

4.9 4.9 4.3 10.4 11.1 4.3

Temperature

Expertise

1: "You are an expert medical professional." 2: "You are a mystical wizard in Middle Earth."

- Figure 2 | Left: Prompt anatomy. Each summarization task uses a slightly different instruction (Table 2). Right: Effect of model temperature and expertise. We generally find better performance when (1) using lower temperature, i.e. generating less random output, as summarization tasks benefit more from truthfulness than creativity (2) assigning the model clinical expertise in the prompt. Output generated via GPT-3.5 on the Open-i radiology report dataset.

###### 4.1.2 Experimental Setup

For each dataset, we construct test sets by randomly drawing the same s samples, where s = 250 for all datasets except dialogue (s = 100), which includes only 126 samples in total. After selecting these s samples, we choose another s as a validation set for datasets which incorporated fine-tuning. We then use the remaining samples as a training set for ICL examples or QLoRA fine-tuning.

We leverage PyTorch for our all our experiments, including the parameter-efficient fine-tuning [83] and the generative pre-trained transformers quantization [84] libraries for implementing QLoRA. We fine-tune models with QLoRA for five epochs using the Adam optimizer with weight decay fix [85]. Our initial learning rate of 1e−3 decays linearly to 1e−4 after a 100-step warm-up; we determine this configuration after experimenting with different learning rates and schedulers. To achieve an effective batch size of 24 on each experiment, we adjust both individual batch size and number of gradient accumulation steps to fit on a single consumer GPU, a NVIDIA Quadro RTX 8000. All open-source models are available on HuggingFace [86].

###### 4.1.3 Quantitative metrics

We use well-known summarization metrics to assess the quality of generated summaries. BLEU [87], the simplest metric, calculates the degree of overlap between the reference and generated texts by considering 1- to 4-gram sequences. ROUGE-L [88] evaluates similarity based on the longest common subsequence; it considers both precision and recall, hence being more comprehensive than BLEU. In addition to these syntactic metrics, we employ BERTScore, which leverages contextual BERT embeddings to evaluate the semantic similarity of the generated and reference texts [89]. Lastly, we include MEDCON [44] to gauge the consistency of medical concepts. This employs QuickUMLS [90], a tool that extracts biomedical concepts via string matching algorithms [91]. We restrict MEDCON to specific UMLS semantic groups (Anatomy, Chemicals & Drugs, Device, Disorders, Genes & Molecular Sequences, Phenomena and Physiology) relevant for our work. All four metrics range from [0,100] with higher scores indicating higher similarity between the generated and reference summaries.

###### 4.2 Clinical reader study

After identifying the best model and method via NLP quantitative metrics, we perform a clinical reader study across three summarization tasks: radiology reports, patient questions, and progress notes. The dialogue task is excluded due to the unwieldiness of a reader parsing many lengthy transcribed conversations and paragraphs; see Figure A6 for an example and Table 2 for the token count.

Our readers include two sets of physicians: (1) five board-certified radiologists to evaluate summaries of radiology reports (2) five board-certified hospitalists (internal medicine physicians) to evaluate summaries of patient questions and progress notes. For each task, each physician views the same 100 randomly selected inputs and their A/B comparisons (medical expert vs. the best model summaries), which are presented in a blinded and randomized order. An ideal summary would contain all clinically significant information (completeness) without any errors (correctness) or superfluous information (conciseness). Hence we pose the following three questions for readers to evaluate using a five-point Likert scale.

- • Completeness: “Which summary more completely captures important information?” This compares the summaries’ recall, i.e. the amount of clinically significant detail retained from the input text.
- • Correctness: “Which summary includes less false information?” This compares the summaries’ precision, i.e. instances of fabricated information.
- • Conciseness: “Which summary contains less non-important information?” This compares which summary is more condensed, as the value of a summary decreases with superfluous information.

Figure 7e demonstrates the user interface for this study, which we create and deploy via Qualtrics. To obfuscate any formatting differences between the model and medical expert summaries, we apply simple post-processing to standardize capitalization, punctuation, newline characters, etc.

Given this non-parametric, categorical data, we assess the statistical significance of responses using a Wilcoxon signed-rank test with Type 1 error rate = 0.05 and adjust for multiple comparisons using the Bonferroni correction. We estimate intra-reader correlation based on a mean-rating, fixed agreement, two-may mixed effects model [92] using the Pingouin package [93]. Additionally, readers are provided comment space to make observations for qualitative analysis.

###### 4.3 Safety analysis

We conduct a safety analysis connecting summarization errors to medical harm, inspired by the Agency for Healthcare Research and Quality (AHRQ)’s harm scale [94]. This includes radiology reports (nr = 27) and progress notes (nn = 44) samples which contain disparities in completeness and/or correctness between the best model and medical expert summaries. Here, disparities occur if at least one physician significantly prefers or at least two physicians slightly prefer one summary to the other. These summary pairs are randomized and blinded. For each sample, we ask the following multiple-choice questions: “Summary A is more complete and/or correct than Summary B. Now, suppose Summary B (worse) is used in the standard clinical workflow. Compared to using Summary A (better), what would be the...” (1) “... extent of possible harm?” options: {none, mild or moderate harm, severe harm or death} (2) “... likelihood of possible harm?” options: {low, medium, high}.

Safety analysis of fabricated information is discussed in Section 5.2.2.

###### 4.4 Connecting quantitative and clinical evaluations

We now provide intuition connecting NLP metrics and clinical reader scores. Note that in our work, these tools measure different quantities; NLP metrics measure the similarity between two summaries, while reader scores measure which summary is better. Consider an example where two summaries are exactly the same: NLP metrics would yield the highest possible score (100), while clinical readers would provide a score of 0 to denote equivalence. As the magnitude of a reader score increases, the two summaries are increasingly dissimilar, hence yielding a lower quantitative metric score. Given this intuition, we compute the Spearman correlation coefficient between NLP metric scores and the magnitude of the reader scores. Since these features are inversely correlated, for clarity we display the negative correlation coefficient values.

###### BLEU

###### ROUGE-L

###### BERTScore

###### MEDCON

94

55

30

55

Med-Alpaca

Med-Alpaca

Med-Alpaca

Med-Alpaca

82

10

10

0

0 30 Alpaca

10 55 Alpaca

82 94 Alpaca

10 55 Alpaca

- Figure 3 | Alpaca vs. Med-Alpaca. Given that most data points are below the dashed lines denoting equivalence, we conclude that Med-Alpaca’s fine-tuning with medical Q&A data results in worse performance for our clinical summarization tasks. See Section 5.1 for further discussion. Note that each data point corresponds to the average score of s = 250 samples for a given experimental configuration, i.e. {dataset × m in-context examples}.

###### 5 Results and Discussion

###### 5.1 Quantitative evaluation

###### 5.1.1 Impact of domain-specific fine-tuning

When considering which open-source models to evaluate, we first assess the benefit of fine-tuning open-source models on medical text. For example, Med-Alpaca [57] is a version of Alpaca [56] which was further instructiontuned with medical Q&A text, consequently improving performance for the task of medical question-answering.

- Figure 3 compares these models for our task of summarization, showing that most data points are below the dashed lines denoting equivalence. Hence despite Med-Alpaca’s adaptation for the medical domain, it performs worse than Alpaca for our tasks of clinical text summarization—highlighting a distinction between domain adaptation and task adaptation. With this in mind, and considering that Alpaca is commonly known to perform worse than our other open-source autoregressive models Vicuna and Llama-2 [21, 49], for simplicity we exclude Alpaca and Med-Alpaca from further analysis.

[Figure 6]

|FLAN-T5<br><br>| |
|---|
<br><br>FLAN-UL2 Llama-2<br><br>| |
|---|
<br><br>Vicuna<br><br>| |
|---|
|
|---|

- Figure 4 | One in-context example (ICL) vs. QLoRA across open-source models on Open-i radiology reports. FLAN-T5 achieves best performance on both methods for this dataset. While QLoRA typically outperforms ICL with the better models (FLAN-T5, Llama-2), this relationship reverses given sufficient in-context examples (Figure A1). Figure A2 contains similar results with patient health questions.

[Figure 7]

|GPT-4 GPT-3.5<br><br>| |
|---|
<br><br>| |
|---|
<br><br>FLAN-T5 FLAN-UL2<br><br>| |
|---|
<br><br>Llama-2 Vicuna<br><br>| |
|---|
<br><br>FLAN-T5 + QLoRA|
|---|

- Figure 5 | MEDCON scores vs. number of in-context examples across models and datasets. We also include the best model fine-tuned with QLoRA (FLAN-T5) as a horizontal dashed line for valid datasets. Zero-shot prompting (0 examples) often yields considerably inferior results, underscoring the need for adaptation methods. Note the allowable number of in-context examples varies significantly by model and dataset. See Figure A1 for results across all four metrics.

###### 5.1.2 Comparison of adaptation strategies

Next, we compare ICL (in-context learning) vs. QLoRA (quantized low-rank adaptation) across the remaining open-source models using the Open-i radiology report dataset in Figure 4 and the patient health questions in Figure A2. We choose these datasets because their shorter context lengths allow for training with lower computational cost. FLAN-T5 emerged as the best-performing model with QLoRA. QLoRA typically outperformed ICL (one example) with the better models FLAN-T5 and Llama-2; given a sufficient number of in-context examples, however, most models surpass even the best QLoRA fine-tuned model, FLAN-T5 (Figure A1). FLAN-T5 (2.7B) eclipsed its fellow seq2seq model FLAN-UL2 (20B), despite being an older model with almost 8× fewer parameters.

When considering trade-offs between adaptation strategies, availability of these models (open-source vs. proprietary) raises an interesting consideration for healthcare, where data and model governance are importantespecially if summarization tools are cleared for clinical use by the Food and Drug Administration. This could motivate the use of fine-tuning methods on open-source models. Governance aside, ICL provides many benefits: (1) model weights are fixed, hence enabling queries of pre-existing LLMs (2) adaptation is feasible with even a few examples, while fine-tuning methods such as QLoRA typically require hundreds or thousands of examples.

###### 5.1.3 Effect of context length for in-context learning

- Figure 5 displays MEDCON [44] scores for all models against number of in-context examples, up to the maximum number of allowable examples for each model and dataset. This graph also includes the best performing model (FLAN-T5) with QLoRA as a reference, depicted by a horizontal dashed line. Compared to zero-shot prompting (m = 0 examples), adapting with even m = 1 example considerably improves performance in almost all cases, underscoring the importance of adaptation methods. While ICL and QLoRA are competitive for open-source models, proprietary models GPT-3.5 and GPT-4 far outperform other models and methods given sufficient in-context examples. For a similar graph across all metrics, see Figure A1.

5.1.4 Head-to-head model comparison

- Figure 6 compares models using win rates, i.e. the head-to-head winning percentage of each model combination across the same set of samples. In other words, for what percentage of samples do model A’s summaries have a higher score than model B’s summaries? This presents trade-offs of different model types. Seq2seq models (FLAN-T5, FLAN-UL2) perform well on syntactical metrics such as BLEU [87] but worse on others, suggesting that these models excel more at matching word choice than matching semantic or conceptual meaning. Note seq2seq models are often constrained to much shorter context length than autoregressive models (Table 1),

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- Figure 6 | Model win rate: a head-to-head winning percentage of each model combination, where red/blue intensities highlight the degree to which models on the vertical axis outperform models on the horizontal axis. GPT-4 generally achieves the best performance. While FLAN-T5 is more competitive for syntactic metrics such as BLEU, we note this model is constrained to shorter context lengths (Table 1). When aggregated across datasets, seq2seq models (FLAN-T5, FLAN-UL2) outperform open-source autoregressive models (Llama-2, Vicuna) on all metrics.

because seq2seq models require the memory-intensive step of encoding the input sequence into a fixed-size context vector. Among open-source models, seq2seq models perform better than autoregressive (Llama-2, Vicuna) models on radiology reports but worse on patient questions and progress notes (Figure A1). Given that these latter datasets have higher lexical variance (Table 2) and more heterogeneous formatting compared to radiology reports, we hypothesize that autoregressive models may perform better with increasing data heterogeneity and complexity.

Best model/method. We deemed the best model and method to be GPT-4 (context length 32,768) with a maximum allowable number of in-context examples, hereon identified as the best-performing model.

###### 5.2 Clinical reader study

Given our clinical reader study design (Figure 7a), pooled results across ten physicians (Figure 7b) demonstrate that summaries from the best adapted model (GPT-4 using ICL) are more complete and contain fewer errors compared to medical expert summaries—which were created either by medical doctors during clinical care or by a committee of medical doctors and experts.

The distributions of reader responses in Figure 7c show that medical expert summaries are preferred in only a minority of cases (19%), while in a majority, the best model is either non-inferior (45%) or preferred (36%). Table A1 contains scores separated by individual readers and affirms the reliability of scores across readers by displaying positive intra-reader correlation values. Based on physician feedback, we undertake a qualitative analysis to illustrate strengths and weaknesses of summaries by the model and medical experts; see Figures 8, A4, and A5. Now, we discuss results with respect to each individual attribute.

###### 5.2.1 Completeness

The best model summaries are more complete on average than medical expert summaries, achieving statistical significance across all three summarization tasks with p < 0.001 (Figure 7b). Lengths of summaries were comparable between the model and medical experts for all three datasets: 47 ± 24 vs. 44 ± 22 tokens for radiology reports, 15 ± 5 vs. 14 ± 4 tokens for patient questions, and 29 ± 7 vs. 27 ± 13 tokens for progress notes (all p > 0.12). Hence the model’s advantage in completeness is not simply a result of generating longer summaries. We provide intuition for completeness by investigating a specific example in progress notes summarization. In Figure A5, the model correctly identifies conditions that were missed by the medical expert, such as hypotension and anemia. Although the model was more complete in generating its progress notes summary, it also missed historical context (a history of HTN, or hypertension).

- a d
- b

Which summary...

21.7% 15.5%

Medical expert Best model

Extent

[Completeness] ... more completely captures important information? [Correctness] ... includes less false information? [Conciseness] ... contains less non-important information?

13.7%

Medical expert Best model

Likelihood

11.3%

Medical expert slightly slightly

Best model signiﬁcantly

signiﬁcantly neither

Likelihood of harm

Extent of harm Severe or death Mild or moderate None

[Figure 12]

High Medium Low

-10 -5 0 5 10

e

[Figure 13]

|Input: thereisfocal highattenuationoverlyingasuperiorleftfrontal gyrus, probablyadural calcification. subsequentmri showsnoevidence ofhemorrhageinthisregion. thebrainparenchymaisnormal. the ventriclesandsulci areslightlyprominent.<br><br>Su maryA: thereisnoevidenceofahemorrhageandnoareaoflow attenuationormaseffectissentosugestanacuteinfarct.<br><br>Su maryB: noacuteintracranial abnormality. probabledural calcificationoverlyingasuperiorleftfrontal gyrus.<br><br><br>Whichsu mary...<br><br>. more completely captures important information?<br><br>. includes les false information?<br><br>. contains les nonimportant information?<br><br>A: signiﬁcantly<br><br>A: slightly<br><br>neither<br><br>B: slightly<br><br>B: signiﬁcantly<br><br>|
|---|

c

[Figure 14]

- Figure 7 | Clinical reader study. (a) Study design comparing the summaries from the best model versus that of medical experts on three attributes: completeness, correctness, and conciseness. (b) Results. Model summaries are rated higher on all attributes. Highlight colors correspond to a value’s location on the color spectrum. Asterisks (*) denote statistical significance by Wilcoxon signed-rank test, p < 0.001. (c) Distribution of reader scores. Horizontal axes denote reader preference as measured by a five-point Likert scale. Vertical axes denote frequency count, with 1,500 total reports for each plot. (d) Extent and likelihood of potential medical harm caused by choosing summaries from the medical expert (pink) or best model (purple) over the other. Model summaries are preferred in both categories. (e) Reader study user interface.

###### 5.2.2 Correctness

With regards to correctness, the best model generated significantly fewer errors (p < 0.001) compared to medical expert summaries overall and on two of three summarization tasks (Figure 7b). As an example of the model’s superior correctness performance on radiology reports, we observe that it avoided common medical expert errors related to lateral distinctions (right vs. left, Figure 8). For progress notes, Figure A5 reveals an intriguing case: during the blinded study, the physician reader erroneously assumed that a hallucination—the incorrect inclusion of a urinary tract infection—was made by the model. In this case, the medical expert was responsible for the hallucination. This instance underscores the point that even medical experts, not just LLMs, can hallucinate. Despite this promising performance, the model was not perfect across all tasks. We see a clear example in Figure A5 in which the model mistakenly generated several conditions in the problem list that were incorrect, such as eosinophilia.

Both the model and medical experts faced challenges interpreting ambiguity, such as user queries in patient health questions. Consider Figure A4’s first example, in which the input question mentioned “diabetes and neuropathy.” The model mirrored this phrasing verbatim, while the medical expert interpreted it as “diabetic neuropathy.” In Figure A4’s second example, the model simply reformulated the input question about tests and their locations, while the medical expert inferred a broader query about tests and treatments. In both cases, the model’s summaries leaned toward literalness, a trait that readers sometimes favored and sometimes did not. In future work, a systematic exploration of model temperature could further illuminate this trade-off.

Further, the critical need for accuracy in a clinical setting motivates a more nuanced understanding of correctness. As such, we define three types of fabricated information: (1) misinterpretations of ambiguity, (2)

|Input: The patient is s/p left frontal craniotomy. A small amount of intracranial gas is seen posterior to the surgical intervention, which could represent postoperative changes. Extensive edema is seen in the left frontal lobe at the site of presumed surgery. Additionally multiple foci of hemorrhage are seen in the region of the left frontal lobe. Midline shift to the right is seen in the frontal region. The ventricles, cisterns, and sulci are unremarkable, without effacement. Comparison with prior studies from outside institution would be helpful in further evaluation of these findings.<br><br>Summary (medical expert): Postoperative changes in the right frontoparietal region identified, with decreasing pneumocephalus. No new hemorrhage. Persistent mild leftward shift, not significantly changed from prior.<br><br>Summary (best model): Stable post-craniotomy changes with decreasing pneumocephalus and persistent mild leftward shift. No new hemorrhage. Increasing opacification of ethmoid and sphenoid sinuses. Large right subgaleal hematoma unchanged.<br><br>Input: Right frontoparietal craniotomy again identified, with decreasing pneumocephalus and small amount of subdural fluid. There is persistent mild leftward shift of approximately 2-3 mm, not significantly changed from prior study. Size and configuration of the ventricles is unchanged from prior study. There is no new hemorrhage. Multiple facial fractures, better evaluated on prior dedicated maxillofacial CT. Increasing opacification is seen in the ethmoid and sphenoid sinuses. Large right subgaleal hematoma again identified.<br><br>Example 2: The model performed worse because it was less concise, i.e. included non-important information (green).<br><br>Example 1: The model performed better because the medical expert made a laterality mistake (red).<br><br>Summary (medical expert):<br><br>1. Left frontal craniotomy. 2. Frontal midline shift to the right. 3. Extensive left frontal lobe edema. 4. Multiple foci of hemorrhage in the right frontal lobe.<br><br>Summary (best model): Postoperative changes following left frontal craniotomy with intracranial gas, extensive edema, and multiple foci of hemorrhage in the left frontal lobe. Midline shift to the right is noted.<br><br>|Attribute|Average|Example 1|Example 2|
|---|---|---|---|
|Completeness|2.8|5|-1|
|Correctness|1.7|8|0|
|Conciseness|0|0|-4|
<br><br>Radiology reports<br><br>|Blue: correct; exists in input + expert + model<br><br>Purple: correct; exists in input + expert only Green: correct; exists in input + model only Orange: incoherent or filler<br><br>Red: incorrect|
|---|
<br><br>Reader scores: Color key:<br><br>|
|---|

- Figure 8 | Annotation: radiology reports. The table (lower left) contains reader scores for these two examples and the task average across all samples. Top: the model performs better due to a laterality mistake by the medical expert. Bottom: the model exhibits a lack of conciseness.

factual inaccuracies: modifying existing facts to be incorrect, and (3) hallucinations: inventing new information that cannot be inferred from the input text. We found that the model committed these errors on 6%, 2%, and

- 5% of samples, respectively, compared to 9%, 4%, and 12% by medical experts. Given the model’s lower error rate in each category, this suggests that incorporating LLMs could actually reduce fabricated information in clinical practice.

Beyond the scope of our work, there’s further potential to reduce fabricated information through incorporating checks by a human, checks by another LLM, or using a model ensemble to create a “committee of experts” [95, 96].

###### 5.2.3 Conciseness

With regards to conciseness, the best model performed significantly better (p < 0.001) overall and on two tasks (Figure 7b). We note the model’s summaries are more concise while concurrently being more complete. Radiology reports were the only task in which physicians did not prefer the best model’s summaries to medical experts. See Figure 8 for an example. We suggest that conciseness could be improved with better prompt engineering, or modifying the prompt to improve performance. Of the task-specific instructions in Table 2, the other two tasks (patient questions, progress notes) explicitly specify summary length, e.g. “15 words or less.” These phrases are included so that model summaries are generated with similar lengths to the human summaries, enabling a clean comparison. Length specification in the radiology reports prompt instruction was more vague, i.e. “...with minimal text,” perhaps imposing a softer constraint on the model. We leave further study of prompt instructions to future work.

###### 5.3 Safety Analysis

The results of this harm study (Figure 7d) indicate that the medical expert summaries would have both a higher likelihood (14%) and higher extent (22%) of possible harm compared to the summaries from the best model (12% and 16%, respectively). These percentages are computed with respect to all samples, such

BLEU ROUGE-L BERTScore MEDCON

0.25

| |
|---|

| |
|---|

| |
|---|

0.20

Correlation

0.15

0.10

0.05

0.00

Completeness Correctness Conciseness

Characteristics

- Figure 9 | Correlation between NLP metrics and reader scores. The semantic metric (BERTScore) and conceptual metric (MEDCON) correlate most highly with correctness. Meanwhile, syntactic metrics BLEU and ROUGE-L correlate most with completeness. See Section 5.4 for further discussion.

that the subset of samples with similar A/B summaries (in completeness and correctness) are assumed to contribute no harm. For the safety analysis of fabricated information, please see Section 5.2.2. Ultimately we argue that, beyond clinical reader studies, conducting downstream analyses is crucial to affirm the safety of LLM-generated summaries in clinical environments.

###### 5.4 Connecting quantitative and clinical evaluations

Figure 9 captures the correlation between NLP metrics and physicians’ preference. Compared to other metrics, BLEU correlates most with completeness and least with conciseness. Given that BLEU measures sequence overlap, this result seems reasonable, as more text provides more “surface area” for overlap; more text also reduces the brevity penalty that BLEU applies on generated sequences which are shorter than the reference [87]. The metrics BERTScore (measuring semantics) and MEDCON (measuring medical concepts) correlate most strongly with reader preference for correctness. Overall, however, the low magnitude of correlation values (approximately 0.2) underscores the need to go beyond NLP metrics with a reader study when assessing clinical readiness.

Aside from the low correlation values in Figure 9, our reader study results (Figure 7b) highlight another limitation of NLP metrics, especially as model-generated summaries become increasingly viable. These metrics rely on a reference—in our case, medical expert summaries—which we have demonstrated may contain errors. Hence we suggest that human evaluation is essential when assessing the clinical feasibility of new methods. If human evaluation is not feasible, Figure 9 suggests that syntactic metrics are better at measuring completeness, while semantic and conceptual metrics are better at measuring correctness.

###### 5.5 Limitations

This study has several limitations which motivate future research.

Model temperature and prompt phrasing can be important for LLM performance (Figure 2), [81, 82]. However, we only search over three possible temperature values. Further, we do not thoroughly engineer our prompt instructions (Table 2); each was chosen after trying only 1-2 options over a small dataset. While this highlights the potential for improvement, we’re also encouraged that achieving convincing results does not require a thorough temperature search or prompt engineering.

In our quantitative analysis, we select state-of-the-art and highly regarded LLMs with a diverse range of attributes. This includes the 7B-parameter tier of open-source autoregressive models, despite some models such as Llama-2 having larger versions. We consider the benefit of larger models in Figure A3, finding

this improvement marginal for Llama-2 (13B) compared to Llama-2 (7B). While there may exist opensource models which perform slightly better than our selections, we do not believe this would meaningfully alter our analysis—especially considering the clinical reader study employs GPT-4, which is an established state-of-the-art [21].

Our study does not encompass all clinical document types, and extrapolating our results is tentative. For instance, our progress notes task employs ICU notes from a single medical center. These notes may be structured differently from non-ICU notes or from ICU notes of a different center. Additionally, more challenging tasks may require summarizing longer documents or multiple documents of different types. Addressing these cases demands two key advancements: (1) extending model context length, potentially through multi-query aggregation or other methods [97, 98] (2) introducing open-source datasets that include broader tasks and lengthier documents. We thus advocate for expanding evaluation to other summarization tasks.

We do not consider the inherently context-specific nature of summarization. For example, a gastroenterologist, radiologist, and oncologist may have different preferences for summaries of a cancer patient with liver metastasis. Or perhaps an abdominal radiologist will want a different summary than a neuroradiologist. Further, individual clinicians may prefer different styles or amounts of information. While we do not explore such a granular level of adaptation, this may not require much further development: since the best model and method uses a handful of examples via ICL, one could plausibly adapt using examples curated for a particular specialty or clinician. Another limitation is that radiology report summaries from medical experts occasionally recommend further studies or refer to prior studies, e.g. “... not significantly changed from prior” in Figure 8. These instances are out of scope for our tasks, which do not include context from prior studies; hence in the clinical reader study, physicians were told to disregard these phrases. Future work can explore providing the LLM with additional context and longitudinal information.

An additional consideration for ours and other LLM studies, especially with proprietary models, is that it is not possible to verify whether a particular open-source dataset was included in model training. While three of our datasets (MIMIC-CXR, MIMIC-III, ProbSum) require PhysioNet [74] access to ensure safe data usage by third parties, this is no guarantee against data leakage. This complication highlights the need for validating results on internal data when possible.

We note the potential for LLMs to be biased [99, 100]. While our datasets do not contain demographic information, we advocate for future work to consider whether summary qualities have any dependence upon group membership.

###### 6 Conclusion

In this research, we evaluate methods for adapting LLMs to summarize clinical text, analyzing eight models across a diverse set of summarization tasks. Our quantitative results underscore the advantages of adapting models to specific tasks and domains. The ensuing clinical reader study demonstrates that LLM summaries are often preferred over medical expert summaries due to higher scores for completeness, correctness, and conciseness. The subsequent safety analysis explores qualitative examples, potential medical harm, and fabricated information to demonstrate the limitations of both LLMs and medical experts. Evidence from this study suggests that incorporating LLM-generated candidate summaries into the clinical workflow could reduce documentation load, potentially leading to decreased clinician strain and improved patient care. Testing this hypothesis motivates future prospective studies in clinical environments.

###### 7 Acknowledgements

Microsoft provided Azure OpenAI credits for this project via both the Accelerate Foundation Models Academic Research (AFMAR) program and also a cloud services grant to Stanford Data Science. Further compute support was provided by One Medical, which Asad Aali used as part of his summer internship. Curtis Langlotz is supported by NIH grants R01 HL155410, R01 HL157235, by AHRQ grant R18HS026886, by the Gordon and Betty Moore Foundation, and by the National Institute of Biomedical Imaging and Bioengineering (NIBIB) under contract 75N92020C00021. Akshay Chaudhari receives support from NIH grants R01 HL167974, R01

AR077604, R01 EB002524, R01 AR079431, and P41 EB027060; from NIH contracts 75N92020C00008 and 75N92020C00021; and from GE Healthcare, Philips, and Amazon.

###### 8 Data and Code Availability

While all datasets are publicly available, our GitHub repository github.com/StanfordMIMI/clin-summ includes preprocessed versions for those which do not require PhysioNet access: Open-i [66] (radiology reports), MeQSum [72] (patient questions), and ACI-Bench [44] (dialogue). Researchers can also access the original datasets via the provided references. Any further distribution of datasets is subject to the terms of use and data sharing agreements stipulated by the original creators. Our repository also contains experiment code and links to open-source models hosted by HuggingFace [86].

###### 9 Author contributions

DVV collected data, developed code, ran experiments, designed studies, analyzed results, created figures, and wrote the manuscript. All authors reviewed the manuscript, providing meaningful revisions and feedback. CVU, LB, JBD provided technical advice in addition to conducting qualitative analysis (CVU), building infrastructure for the Azure API (LB), and implementing the MEDCON metric (JB). AA assisted in model fine-tuning. CB, AP, MP, EPR, AS participated in the reader study as radiologists. NR, PH, WC, NA, JH participated in the reader study as hospitalists. CPL, JP, ASC provided student funding. SG advised on study design for which JH and JP provided additional feedback. JP, ASC guided the project, with ASC serving as principal investigator and advising on technical details and overall direction. No funders or third parties were involved in study design, analysis, or writing.

###### References

- 1. Golob Jr, J. F., Como, J. J. & Claridge, J. A. The painful truth: The documentation burden of a trauma surgeon. Journal of Trauma and Acute Care Surgery 80, 742–747 (2016).
- 2. Arndt, B. G., Beasley, J. W., Watkinson, M. D., Temte, J. L., Tuan, W.-J., Sinsky, C. A. & Gilchrist, V. J. Tethered to the EHR: primary care physician workload assessment using EHR event log data and time-motion observations. The Annals of Family Medicine 15, 419–426 (2017).
- 3. Fleming, S. L., Lozano, A., Haberkorn, W. J., Jindal, J. A., Reis, E. P., Thapa, R., Blankemeier, L., Genkins, J. Z., Steinberg, E., Nayak, A., et al. MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records. arXiv preprint arXiv:2308.14089 (2023).
- 4. Yackel, T. R. & Embi, P. J. Unintended errors with EHR-based result management: a case series. Journal of the American Medical Informatics Association 17, 104–107 (2010).
- 5. Bowman, S. Impact of electronic health record systems on information integrity: quality and safety implications. Perspectives in health information management 10 (2013).
- 6. Gershanik, E. F., Lacson, R. & Khorasani, R. Critical finding capture in the impression section of radiology reports in AMIA Annual Symposium Proceedings 2011 (2011), 465.
- 7. Gesner, E., Gazarian, P. & Dykes, P. The burden and burnout in documenting patient care: an integrative literature review. MEDINFO 2019: Health and Wellbeing e-Networks for All, 1194–1198 (2019).
- 8. Ratwani, R. M., Savage, E., Will, A., Arnold, R., Khairat, S., Miller, K., Fairbanks, R. J., Hodgkins, M. & Hettinger, A. Z. A usability and safety analysis of electronic health records: a multi-center study. Journal of the American Medical Informatics Association 25, 1197–1201 (2018).
- 9. Ehrenfeld, J. M. & Wanderer, J. P. Technology as friend or foe? Do electronic health records increase burnout? Current Opinion in Anesthesiology 31, 357–360 (2018).
- 10. Sinsky, C., Colligan, L., Li, L., Prgomet, M., Reynolds, S., Goeders, L., Westbrook, J., Tutty, M. & Blike, G. Allocation of physician time in ambulatory practice: a time and motion study in 4 specialties. Annals of internal medicine 165, 753–760

(2016).

- 11. Khamisa, N., Peltzer, K. & Oldenburg, B. Burnout in relation to specific contributing factors and health outcomes among nurses: a systematic review. International journal of environmental research and public health 10, 2214–2240 (2013).
- 12. Duffy, W. J., Kharasch, M. S. & Du, H. Point of care documentation impact on the nurse-patient interaction. Nursing Administration Quarterly 34, E1–E10 (2010).
- 13. Chang, C.-P., Lee, T.-T., Liu, C.-H. & Mills, M. E. Nurses’ experiences of an initial and reimplemented electronic health record use. CIN: Computers, Informatics, Nursing 34, 183–190 (2016).
- 14. Shanafelt, T. D., Dyrbye, L. N., Sinsky, C., Hasan, O., Satele, D., Sloan, J. & West, C. P. Relationship between clerical burden and characteristics of the electronic environment with physician burnout and professional satisfaction in Mayo Clinic Proceedings 91 (2016), 836–848.
- 15. Robinson, K. E. & Kersey, J. A. Novel electronic health record (EHR) education intervention in large healthcare organization improves quality, efficiency, time, and impact on burnout. Medicine 97 (2018).
- 16. Toussaint, W., Van Veen, D., Irwin, C., Nachmany, Y., Barreiro-Perez, M., Díaz-Peláez, E., de Sousa, S. G., Millán, L., Sánchez, P. L., Sánchez-Puente, A., et al. Design considerations for high impact, automated echocardiogram analysis. arXiv preprint arXiv:2006.06292 (2020).
- 17. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901

(2020).

- 18. Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223 (2023).
- 19. Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 (2023).
- 20. Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110 (2022).
- 21. Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv preprint arXiv:2306.05685 (2023).
- 22. Wornow, M., Xu, Y., Thapa, R., Patel, B., Steinberg, E., Fleming, S., Pfeffer, M. A., Fries, J. & Shah, N. H. The shaky foundations of large language models and foundation models for electronic health records. npj Digital Medicine 6, 135

(2023).

- 23. Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., Gutierrez, L., Tan, T. F. & Ting, D. S. W. Large language models in medicine. Nature Medicine, 1–11 (2023).
- 24. Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large Language Models Encode Clinical Knowledge. arXiv preprint arXiv:2212.13138 (2022).
- 25. Tu, T., Azizi, S., Driess, D., Schaekermann, M., Amin, M., Chang, P.-C., Carroll, A., Lau, C., Tanno, R., Ktena, I., et al. Towards generalist biomedical ai. arXiv preprint arXiv:2307.14334 (2023).
- 26. Toma, A., Lawler, P. R., Ba, J., Krishnan, R. G., Rubin, B. B. & Wang, B. Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding. arXiv preprint arXiv:2305.12031 (2023).
- 27. Van Veen, D., Van Uden, C., Attias, M., Pareek, A., Bluethgen, C., Polacin, M., Chiu, W., Delbrouck, J.-B., Chaves, J. M. Z., Langlotz, C. P., et al. RadAdapt: Radiology Report Summarization via Lightweight Domain Adaptation of Large Language Models. arXiv preprint arXiv:2305.01146 (2023).
- 28. Mathur, Y., Rangreji, S., Kapoor, R., Palavalli, M., Bertsch, A. & Gormley, M. R. SummQA at MEDIQA-Chat 2023: In-Context Learning with GPT-4 for Medical Summarization. arXiv preprint arXiv:2306.17384 (2023).
- 29. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. Attention is all you need. Advances in neural information processing systems 30 (2017).
- 30. OpenAI. GPT-4 Technical Report 2023. arXiv: 2303.08774 [cs.CL].
- 31. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022).

- 32. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- 33. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
- 34. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog 1, 9 (2019).
- 35. Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M. & Le, Q. V. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021).
- 36. Liu, Z., Li, Y., Shu, P., Zhong, A., Yang, L., Ju, C., Wu, Z., Ma, C., Luo, J., Chen, C., et al. Radiology-Llama2: Best-in-Class Large Language Model for Radiology. arXiv preprint arXiv:2309.06419 (2023).
- 37. Li, X. L. & Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190

(2021).

- 38. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L. & Chen, W. LoRA: Low-Rank Adaptation of Large Language Models 2021. arXiv: 2106.09685 [cs.CL].
- 39. Lampinen, A. K., Dasgupta, I., Chan, S. C., Matthewson, K., Tessler, M. H., Creswell, A., McClelland, J. L., Wang, J. X. & Hill, F. Can language models learn from explanations in context? arXiv preprint arXiv:2204.02329 (2022).
- 40. Liu, S., McCoy, A. B., Wright, A. P., Carew, B., Genkins, J. Z., Huang, S. S., Peterson, J. F., Steitz, B. & Wright, A. Leveraging Large Language Models for Generating Responses to Patient Messages. medRxiv, 2023–07 (2023).
- 41. Kahn Jr, C. E., Langlotz, C. P., Burnside, E. S., Carrino, J. A., Channin, D. S., Hovsepian, D. M. & Rubin, D. L. Toward best practices in radiology reporting. Radiology 252, 852–856 (2009).
- 42. Gao, Y., Dligach, D., Miller, T., Churpek, M. M. & Afshar, M. Overview of the Problem List Summarization (ProbSum) 2023 Shared Task on Summarizing Patients’ Active Diagnoses and Problems from Electronic Health Record Progress Notes. arXiv preprint arXiv:2306.05270 (2023).
- 43. Abacha, A. B., Yim, W.-w., Adams, G., Snider, N. & Yetisgen-Yildiz, M. Overview of the MEDIQA-Chat 2023 Shared Tasks on the Summarization & Generation of Doctor-Patient Conversations in Proceedings of the 5th Clinical Natural Language Processing Workshop (2023), 503–513.
- 44. Yim, W.-w., Fu, Y., Abacha, A. B., Snider, N., Lin, T. & Yetisgen, M. ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation. arXiv preprint arXiv:2306.02022 (2023).
- 45. Yu, F., Endo, M., Krishnan, R., Pan, I., Tsai, A., Reis, E. P., Fonseca, E., Lee, H., Shakeri, Z., Ng, A., et al. Radiology Report Expert Evaluation (ReXVal) Dataset 2023.
- 46. Tang, L., Sun, Z., Idnay, B., Nestor, J. G., Soroush, A., Elias, P. A., Xu, Z., Ding, Y., Durrett, G., Rousseau, J. F., et al. Evaluating large language models on medical evidence summarization. npj Digital Medicine 6, 158 (2023).
- 47. Chen, M. X., Firat, O., Bapna, A., Johnson, M., Macherey, W., Foster, G., Jones, L., Parmar, N., Schuster, M., Chen, Z., et al. The best of both worlds: Combining recent advances in neural machine translation. arXiv preprint arXiv:1804.09849

(2018).

- 48. Shi, T., Keneshloo, Y., Ramakrishnan, N. & Reddy, C. K. Neural abstractive text summarization with sequence-to-sequence models. ACM Transactions on Data Science 2, 1–37 (2021).
- 49. Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I. & Xing, E. P. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality Mar. 2023. https://lmsys.org/blog/2023-03-30-vicuna/.
- 50. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. & Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21, 5485–5551

(2020).

- 51. Chung, H., Hou, L., Longpre, S., et al. Scaling Instruction-Finetuned Language Models. https://doi.org/10.48550/arXiv.2210.11416

(2022).

- 52. Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J. & Roberts, A. The Flan Collection: Designing Data and Methods for Effective Instruction Tuning 2023. arXiv: 2301.13688 [cs.AI].
- 53. Lehman, E., Hernandez, E., Mahajan, D., Wulff, J., Smith, M. J., Ziegler, Z., Nadler, D., Szolovits, P., Johnson, A. & Alsentzer, E. Do We Still Need Clinical Language Models? arXiv preprint arXiv:2302.08091 (2023).
- 54. Tay, Y., Dehghani, M., Tran, V. Q., Garcia, X., Wei, J., Wang, X., Chung, H. W., Bahri, D., Schuster, T., Zheng, S., et al. Ul2: Unifying language learning paradigms in The Eleventh International Conference on Learning Representations (2022).
- 55. Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).
- 56. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. & Hashimoto, T. B. Stanford Alpaca: An Instruction-following LLaMA model https://github.com/tatsu-lab/stanford_alpaca. 2023.
- 57. Han, T., Adams, L. C., Papaioannou, J.-M., Grundmann, P., Oberhauser, T., Löser, A., Truhn, D. & Bressem, K. K. MedAlpaca–An Open-Source Collection of Medical Conversational AI Models and Training Data. arXiv preprint arXiv:2304.08247 (2023).
- 58. OpenAI. ChatGPT Accessed: 2023-09-04. 2022. https://openai.com/blog/chatgpt.
- 59. Lim, Z. W., Pushpanathan, K., Yew, S. M. E., Lai, Y., Sun, C.-H., Lam, J. S. H., Chen, D. Z., Goh, J. H. L., Tan, M. C. J., Sheng, B., et al. Benchmarking large language models’ performances for myopia care: a comparative analysis of ChatGPT-3.5, ChatGPT-4.0, and Google Bard. EBioMedicine 95 (2023).
- 60. Rosoł, M., Gąsior, J. S., Łaba, J., Korzeniewski, K. & Młyńczak, M. Evaluation of the performance of GPT-3.5 and GPT-4 on the Medical Final Examination. medRxiv, 2023–06 (2023).
- 61. Brin, D., Sorin, V., Vaid, A., Soroush, A., Glicksberg, B. S., Charney, A. W., Nadkarni, G. & Klang, E. Comparing ChatGPT and GPT-4 performance in USMLE soft skill assessments. Scientific Reports 13, 16492 (2023).
- 62. Deka, P., Jurek-Loughrey, A., et al. Evidence Extraction to Validate Medical Claims in Fake News Detection in International Conference on Health Information Science (2022), 3–15.
- 63. Nie, F., Chen, M., Zhang, Z. & Cheng, X. Improving few-shot performance of language models via nearest neighbor calibration. arXiv preprint arXiv:2212.02216 (2022).
- 64. Dettmers, T., Pagnoni, A., Holtzman, A. & Zettlemoyer, L. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314 (2023).

- 65. Peng, A., Wu, M., Allard, J., Kilpatrick, L. & Heidel, S. GPT-3.5: Turbo, Fine-Tuning, and API Updates https: //openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates. Accessed: August 22, 2023. 2023.
- 66. Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., Shooshan, S. E., Rodriguez, L., Antani, S., Thoma, G. R. & McDonald, C. J. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association 23, 304–310 (2016).
- 67. Johnson, A. et al. MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. https://www.nature.com/articles/s41597-019-0322-0 (2019).
- 68. Chen, Z., Varma, M., Wan, X., Langlotz, C. & Delbrouck, J.-B. Toward Expanding the Scope of Radiology Report Summarization to Multiple Anatomies and Modalities in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (Association for Computational Linguistics, Toronto, Canada, July 2023), 469–484. https://aclanthology.org/2023.acl-short.41.
- 69. Delbrouck, J.-B., Varma, M., Chambon, P. & Langlotz, C. Overview of the RadSum23 Shared Task on Multi-modal and Multi-anatomical Radiology Report Summarization in Proceedings of the 22st Workshop on Biomedical Language Processing (Association for Computational Linguistics, Toronto, Canada, July 2023).
- 70. Demner-Fushman, D., Ananiadou, S. & Cohen, K. B. The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks in The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks

(2023).

- 71. Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A. & Mark, R. Mimic-iv. PhysioNet. Available online at: https://physionet. org/content/mimiciv/1.0/(accessed August 23, 2021) (2020).
- 72. Ben Abacha, A. & Demner-Fushman, D. On the Summarization of Consumer Health Questions in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28th - August 2

(2019).

- 73. Gao, Y., Miller, T., Afshar, M. & Dligach, D. BioNLP Workshop 2023 Shared Task 1A: Problem List Summarization in Proceedings of the 22nd Workshop on Biomedical Language Processing (2023).
- 74. Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K. & Stanley, H. E. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101. Circulation Electronic Pages: http://circ.ahajournals.org/content/101/23/e215.full PMID:1085218; doi: 10.1161/01.CIR.101.23.e215, e215–e220 (2000 (June 13)).
- 75. Yim, W., Ben Abacha, A., Snider, N., Adams, G. & Yetisgen, M. Overview of the MEDIQA-Sum Task at ImageCLEF 2023: Summarization and Classification of Doctor-Patient Conversations in CLEF 2023 Working Notes (CEUR-WS.org, Thessaloniki, Greece, Sept. 2023).
- 76. Ma, C., Wu, Z., Wang, J., Xu, S., Wei, Y., Liu, Z., Guo, L., Cai, X., Zhang, S., Zhang, T., et al. ImpressionGPT: an iterative optimizing framework for radiology report summarization with chatGPT. arXiv preprint arXiv:2304.08448

(2023).

- 77. Wei, S., Lu, W., Peng, X., Wang, S., Wang, Y.-F. & Zhang, W. Medical Question Summarization with Entity-driven Contrastive Learning. arXiv preprint arXiv:2304.07437 (2023).
- 78. Manakul, P., Fathullah, Y., Liusie, A., Raina, V., Raina, V. & Gales, M. CUED at ProbSum 2023: Hierarchical Ensemble of Summarization Models. arXiv preprint arXiv:2306.05317 (2023).
- 79. Saravia, E. Prompt Engineering Guide. https://github.com/dair-ai/Prompt-Engineering-Guide (Dec. 2022).
- 80. Best Practices for Prompt Engineering with OpenAI API https://help.openai.com/en/articles/6654000-best-practices-forprompt-engineering-with-openai-api. Accessed: 2023-09-08. OpenAI, 2023.
- 81. Strobelt, H., Webson, A., Sanh, V., Hoover, B., Beyer, J., Pfister, H. & Rush, A. M. Interactive and visual prompt engineering for ad-hoc task adaptation with large language models. IEEE transactions on visualization and computer graphics 29, 1146–1156 (2022).
- 82. Wang, J., Shi, E., Yu, S., Wu, Z., Ma, C., Dai, H., Yang, Q., Kang, Y., Wu, J., Hu, H., et al. Prompt engineering for healthcare: Methodologies and applications. arXiv preprint arXiv:2304.14670 (2023).
- 83. Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y. & Paul, S. PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods https://github.com/huggingface/peft. 2022.
- 84. Frantar, E., Ashkboos, S., Hoefler, T. & Alistarh, D. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323 (2022).
- 85. Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).
- 86. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Transformers: State-of-the-art natural language processing in Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations (2020), 38–45.
- 87. Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation in Proceedings of the 40th annual meeting of the Association for Computational Linguistics (2002), 311–318.
- 88. Lin, C.-Y. Rouge: A package for automatic evaluation of summaries in Text summarization branches out (2004), 74–81.
- 89. Zhang*, T., Kishore*, V., Wu*, F., Weinberger, K. Q. & Artzi, Y. BERTScore: Evaluating Text Generation with BERT in International Conference on Learning Representations (2020). https://openreview.net/forum?id=SkeHuCVFDr.
- 90. Soldaini, L. & Goharian, N. Quickumls: a fast, unsupervised approach for medical concept extraction in MedIR workshop, sigir (2016), 1–4.
- 91. Okazaki, N. & Tsujii, J. Simple and efficient algorithm for approximate dictionary matching in Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010) (2010), 851–859.
- 92. Koo, T. K. & Li, M. Y. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of chiropractic medicine 15, 155–163 (2016).
- 93. Vallat, R. Pingouin: statistics in Python. J. Open Source Softw. 3, 1026 (2018).
- 94. Walsh, K. E., Harik, P., Mazor, K. M., Perfetto, D., Anatchkova, M., Biggins, C., Wagner, J., Schoettker, P. J., Firneno, C., Klugman, R., et al. Measuring harm in healthcare: optimizing adverse event review. Medical care 55, 436 (2017).
- 95. Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N. & Wu, Y. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410 (2016).

- 96. Chang, Y., Wang, X., Wang, J., Wu, Y., Zhu, K., Chen, H., Yang, L., Yi, X., Wang, C., Wang, Y., et al. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109 (2023).
- 97. Poli, M., Massaroli, S., Nguyen, E., Fu, D. Y., Dao, T., Baccus, S., Bengio, Y., Ermon, S. & Ré, C. Hyena hierarchy: Towards larger convolutional language models. arXiv preprint arXiv:2302.10866 (2023).
- 98. Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N. & Wei, F. LongNet: Scaling Transformers to 1,000,000,000 Tokens 2023. arXiv: 2307.02486 [cs.CL].
- 99. Omiye, J. A., Lester, J. C., Spichak, S., Rotemberg, V. & Daneshjou, R. Large language models propagate race-based medicine. NPJ Digital Medicine 6, 195 (2023).
- 100. Zack, T., Lehman, E., Suzgun, M., Rodriguez, J. A., Celi, L. A., Gichoya, J., Jurafsky, D., Szolovits, P., Bates, D. W., Abdulnour, R.-E. E., et al. Assessing the potential of GPT-4 to perpetuate racial and gender biases in health care: a model evaluation study. The Lancet Digital Health 6, e12–e22 (2024).

A Appendix

##### Metrics

BLEU ROUGE-L BERTScore MEDCON

#### Datasets

[Figure 15]

DialogueOpen-iMIMIC-CXR Progress

Radiologyreports

[Figure 16]

[Figure 17]

MIMIC-III Patient

[Figure 18]

questions

[Figure 19]

notes

[Figure 20]

|Key<br><br>| |
|---|
<br><br>GPT-4 GPT-3.5<br><br>| |
|---|
<br><br>| |
|---|
<br><br>FLAN-T5 FLAN-UL2<br><br>| |
|---|
<br><br>Llama-2 Vicuna<br><br>| |
|---|
<br><br>FLAN-T5 + QLoRA|
|---|

- Figure A1 | Metric scores vs. number of in-context examples across models and datasets. We also include the best model fine-tuned with QLoRA (FLAN-T5) as a horizontal dashed line. Note the allowable number of in-context examples varies significantly by model and dataset.

[Figure 21]

|FLAN-T5<br><br>| |
|---|
<br><br>FLAN-UL2 Llama-2<br><br>| |
|---|
<br><br>Vicuna<br><br>| |
|---|
|
|---|

###### Figure A2 | One in-context example (ICL) vs. QLoRA across open-source models on patient health questions. While QLoRA typically outperforms ICL with the better models (FLAN-T5, Llama-2), this relationship reverses given sufficient in-context examples (Figure A1). Figure 4 contains similar results with the Open-i radiology report dataset.

BLEU

ROUGE-L

BERTScore

MEDCON

15

40

55

90

Llama-2(13B)

Llama-2(13B)

Llama-2(13B)

Llama-2(13B)

82

10

10

0

0 15 Llama-2 (7B)

10 40 Llama-2 (7B)

82 90 Llama-2 (7B)

10 55 Llama-2 (7B)

###### Figure A3 | Comparing Llama-2 (7B) vs. Llama-2 (13B). As most data points are near or slightly above the dashed lines denoting equivalence, we conclude that the larger Llama-2 model (13B parameters) delivers marginal improvement for clinical summarization tasks compared to the 7B model. Note that each data point corresponds to the average score of s = 250 samples for a given experimental configuration, i.e. {dataset × m in-context examples}.

- Table A1 | Reader study results evaluating completeness, correctness, conciseness (columns) across individual readers and pooled across readers. Scores are on the range [-10, 10], where positive scores denote the best model is preferred to the medical expert. Intensity of highlight colors blue (model wins) or red (expert wins) correspond to the score. Asterisks (*) on pooled rows denote statistical significance by a one-sided Wilcoxon signed-rank test, p < 0.001. Intra-class correlation (ICC) values across readers are on a range of [−1, 1] where −1, 0, and +1 correspond to negative, no, and positive correlations, respectively. See Figure 7a for study overview.

|Task|Reader<br><br>|Completeness Correctness Conciseness|
|---|---|---|
|Radiology reports<br><br>|1<br>2<br>3<br>4<br>5<br><br><br>Pooled ICC|3.5 ± 5.6 1.7 ± 3.6 1.2 ± 4.8<br><br>3.6 ± 6.6 2.5 ± 4.7 -0.3 ± 5.4<br><br><br>0.8 ± 2.9 0.6 ± 3.2 -1.7 ± 3.0 4.7 ± 4.7 2.9 ± 3.9 1.2 ± 3.8<br><br>1.4 ± 4.0 0.6 ± 2.2 -0.6 ± 3.4<br><br>2.8 ± 5.1 * 1.7 ± 3.7 * 0.0 ± 4.3<br><br><br>0.45 0.58 0.48|
|Patient questions|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br><br>Pooled ICC<br><br>|1.7 ± 7.2 0.6 ± 3.4 0.3 ± 3.4<br><br>1.0 ± 5.6 -0.1 ± 3.6 0.1 ± 3.6<br><br>2.3 ± 7.2 2.0 ± 5.3 2.2 ± 5.9 1.9 ± 6.7 0.0 ± 0.0 0.0 ± 0.0<br><br><br>0.9 ± 5.7 0.4 ± 3.6 0.4 ± 3.6<br><br>1.6 ± 6.5 * 0.6 ± 3.7 * 0.6 ± 3.9 * 0.67 0.31 0.21<br><br><br>|
|Progress notes<br><br>|1<br>2<br>3<br>4<br>5<br><br><br>Pooled ICC<br><br>|3.4 ± 7.5 0.5 ± 2.5 0.1 ± 4.5 2.3 ± 6.5 0.6 ± 4.4 0.4 ± 4.2 2.7 ± 6.3 1.0 ± 4.4 0.9 ± 3.7<br><br>2.5 ± 7.2 0.5 ± 6.8 1.7 ± 6.9 2.0 ± 6.8 -0.8 ± 4.5 -0.1 ± 1.2<br><br>2.6 ± 6.9 * 0.4 ± 4.8 0.6 ± 4.5 * 0.77 0.74 0.42<br><br><br>|
|Overall|Pooled ICC<br><br>|2.3 ± 5.8 * 0.8 ± 3.7 * 0.4 ± 4.0 *<br><br>0.63 0.56 0.38|

|Input: Where on site does it tell how diabetes and neuropathy causes other things like neck and knee pain and other ailments<br><br>Summary (medical expert): How can I get tested and treated for trimethylaminuria?<br><br>Summary (best model): What tests are available for trimethylaminuria and where can I get tested?<br><br>Input: Hello, I have been dealing with trimethylaminuria since I was a child. I have done some of my own research and it looks like not much can be done for this condition. I do not have it all over my body it’s only in my armpits. In the past I’ve gone to doctors and dermatologist they gave me no answers until I looked online today and finally found out what I have. I don’t know maybe I’m wrong. But this disease isn’t even consider common because no one has done anything about it. I’m sure they’re thousands of women with it... Can I be tested for it and help in some kind of way to finding a cure or something? What testing is done for this? And where? Thank you<br><br>Example 2: The model performed worse because it did not interpret the patient's implied intention (purple).<br><br>Example 1: The model performed better because it summarized the question more specifically (green).<br><br>Summary (medical expert): What can diabetic neuropathy lead to?<br><br>Summary (best model): How does diabetes and neuropathy cause neck and knee pain?<br><br>|Attribute|Average|Example 1|Example 2|
|---|---|---|---|
|Completeness|1.6|3|-4|
|Correctness|0.6|1|-2|
|Conciseness|0.6|1|-1|
<br><br>Patient questions<br><br>|Blue: correct; exists in input + expert + model<br><br>Purple: correct; exists in input + expert only Green: correct; exists in input + model only Orange: incoherent or filler<br><br>Red: incorrect|
|---|
<br><br>Reader scores: Color key:<br><br>|
|---|

###### Figure A4 | Annotation: patient health questions. The table (lower left) contains reader scores for these two examples and the task average across all samples.

###### Figure A5 | Annotation: progress notes. The tables (lower right) contain reader scores for this example and the task average across all samples.

|Input:<br><br>[DOCTOR] hi , [PATIENT] , how are you ? [PATIENT] hi . good to see you . [DOCTOR] it's good to see you as well . so i know that the nurse told you about dax . i'd like to tell dax a little bit about you . [PATIENT] sure . [DOCTOR] okay ? so , [PATIENT] is a 62-year-old male with a past medical history significant for a kidney transplant , hypothyroidism , and arthritis , who presents today with complaints of joint pain . [PATIENT] , what's going on with your joint ? what happened ? [PATIENT] uh , so , over the the weekend , we've been moving boxes up and down our basements stairs , and by the end of the day my knees were just killing me . [DOCTOR] okay . is , is one knee worse than the other ? [PATIENT] equally painful . [DOCTOR] okay . [PATIENT] both of them . [DOCTOR] and did you , did you injure one of them ? [PATIENT] um , uh , i've had some knee problems in the past but i think it was just the repetition and the weight of the boxes . [DOCTOR] okay . all right . and , and what have you taken for the pain ? [PATIENT] a little tylenol . i iced them for a bit . nothing really seemed to help , though . [DOCTOR] okay . all right . um , and does it prevent you from doing , like , your activities of daily living , like walking and exercising and things like that ? [PATIENT] uh , saturday night it actually kept me up for a bit . they were pretty sore . [DOCTOR] mm-hmm . okay . and any other symptoms like fever or chills ? [PATIENT] no . [DOCTOR] joint pain ... i mean , like muscle aches ? [PATIENT] no . [DOCTOR] nausea , vomiting , diarrhea ? [PATIENT] no . [DOCTOR] anything like that ? [PATIENT] no . [DOCTOR] okay . all right . now , i know that you've had the kidney transplant a few years ago for some polycystic kidneys . [PATIENT] mm-hmm . [DOCTOR] um , how are you doing with that ? i know that you told dr. gutierrez- [PATIENT] mm . [DOCTOR] . a couple of weeks ago . [PATIENT] yes . [DOCTOR] everything's okay ? [PATIENT] so far , so good . [DOCTOR] all right . and you're taking your immunosuppressive medications ? [PATIENT] yes , i am . [DOCTOR] okay . all right . um , and did they have anything to say ? i have n't gotten any reports from them , so ... [PATIENT] no , nnothing out of the ordinary , from what they reported . [DOCTOR] okay . all right . um , and in terms of your hyperthyroidism , how are you doing with the synthroid ? are you doing okay ? [PATIENT] uh , yes , i am . [DOCTOR] you're taking it regularly ? [PATIENT] on the clock , yes . [DOCTOR] yes . okay . and any fatigue ? weight gain ? anything like that that you've noticed ? [PATIENT] no , nothing out of the ordinary . [DOCTOR] okay . and just in general , you know , i know that we've kind of battled with your arthritis . [PATIENT] mm-hmm . [DOCTOR] you know , it's hard because you ca n't take certain medications 'cause of your kidney transplant . [PATIENT] sure . [DOCTOR] so other than your knees , any other joint pain or anything like that ? [PATIENT] every once in a while , my elbow , but nothing , nothing out of the ordinary . [DOCTOR] okay . all right . now i know the nurse did a review of systems sheet when you checked in . any other symptoms i might have missed ? [PATIENT] no . [DOCTOR] no headaches ? [PATIENT] no headaches . [DOCTOR] anything like that w- ... okay . all right . well , i wan na go ahead and do a quick physical exam , all right ? hey , dragon , show me the vital signs . so here in the office , your vital signs look good . you do n't have a fever , which is good . [PATIENT] mm-hmm . [DOCTOR] your heart rate and your , uh , blood pressure look fine . i'm just gon na check some things out , and i'll let you know what i find , okay ? [PATIENT] perfect . [DOCTOR] all right . does that hurt ? [PATIENT] a little bit . that's tender . [DOCTOR] okay , so on physical examination , on your heart exam , i do appreciate a little two out of six systolic ejection murmur- [PATIENT] mm-hmm . [DOCTOR] . which we've heard in the past . okay , so that seems stable . on your knee exam , there is some edema and some erythema of your right knee , but your left knee looks fine , okay ? um , you do have some pain to palpation of the right knee and some decreased range of motion , um , on exam , okay ? so what does that mean ? so we'll go ahead and we'll see if we can take a look at some of these things . i know that they did an x-ray before you came in , okay ? [PATIENT] mm-hmm . [DOCTOR] so let's take a look at that . [PATIENT] sure . [DOCTOR] hey , dragon , show me the right knee x-ray . so here's the rhere's your right knee x-ray . this basically shows that there's good bony alignment . there's no acute fracture , which is not surprising , based on the history . [PATIENT] mm-hmm . [DOCTOR] okay ? hey , dragon , show me the labs . and here , looking at your lab results , you know , your white blood cell count is not elevated , which is good . you know , we get concerned about that in somebody who's immunocompromised . [PATIENT] mm-hmm . [DOCTOR] and it looks like your kidney function is also very good . so i'm , i'm very happy about that . [PATIENT] yeah . [DOCTOR] okay ? so i just wan na go over a little bit about my assessment and my plan for you . [PATIENT] mm-hmm . [DOCTOR] so for your knee pain , i think that this is an acute exacerbation of your arthritis , okay ? so i wan na go ahead and if<br><br>... and prescribe some ultram 50 milligrams every six hours as needed . [PATIENT] okay . [DOCTOR] okay ? i also wan na go ahead and just order an autoimmune panel , okay ? hey , dragon , order an autoimmune panel . and you know , i , i want , i want you to just take it easy for right now , and if your symptoms continue , we'll talk about further imaging and possibly referral to physical therapy , okay ? [PATIENT] you got it . [DOCTOR] for your second problem , your hypothyroidism , i wan na go ahead and continue you on this ... on the synthroid , and i wan na go ahead and order some thyroid labs , okay ? [PATIENT] sure . [DOCTOR] hey , dragon , order a thyroid panel . and then for your last problem , the arthritis , you know , we just kinda talked about that . you know , it's gon na be a struggle for you because again , you ca n't take some of those anti-inflammatory medications because of your kidney transplant , so ... [PATIENT] mm-hmm . [DOCTOR] you know , let's see how we do over the next couple weeks , and again , we'll refer you to physical therapy if we need to , okay ? [PATIENT] you got it . [DOCTOR] you have any questions ? [PATIENT] not at this point . [DOCTOR] okay . hey , dragon , finalize the note.<br><br>Summary (medical expert):<br><br>ASSESSMENT AND PLAN [PATIENT] is a 62-year-old male with a past medical history significant for a kidney transplant, hypothyroidism, and arthritis. He presents today with complaints of joint pain.<br><br>Arthritis.<br><br>• Medical Reasoning: The patient reports increased joint pain in his bilateral knees over the past weekend. Given that his right knee x-ray was unremarkable, I believe this is an acute exacerbation of his arthritis.<br>• Additional Testing: We will order an autoimmune panel for further evaluation.<br>• Medical Treatment: Initiate Ultram 50 mg every 6 hours as needed.<br>• Patient Education and Counseling: I advised the patient to rest his knees. If his symptoms persist, we can consider further imaging and possibly a referral to physical therapy.<br><br>Hypothyroidism.<br><br>• Medical Reasoning: The patient is doing well on Synthroid and is asymptomatic at this time.<br>• Additional Testing: We will order a thyroid panel.<br>• Medical Treatment: Continue Synthroid. Status post renal transplant.<br>• Medical Reasoning: He is doing well and has been compliant with his immunosuppressive medications. On recent labs, his white blood cell count was within a normal limits and his kidney function is stable.<br>• Medical Treatment: Continue current regimen.<br><br>Summary (best model):<br><br>ASSESSMENT AND PLAN [PATIENT] is a 62-year-old male with a past medical history significant for a kidney transplant, hypothyroidism, and arthritis. He presents today with complaints of joint pain. Acute exacerbation of arthritis.<br><br>• Medical Reasoning: His right knee is swollen and erythematous, and he has pain with palpation and decreased range of motion. His recent history of physical exertion and the findings on his physical exam are consistent with an acute exacerbation of his arthritis.<br>• Additional Testing: I will order an autoimmune panel to rule out any autoimmune causes of his arthritis.<br>• Medical Treatment: I will prescribe Ultram 50 mg every six hours as needed for pain. Hypothyroidism.<br>• Medical Reasoning: He has been stable on Synthroid.<br>• Additional Testing: I will order a thyroid panel to ensure his thyroid levels are within normal limits.<br>• Medical Treatment: Continue Synthroid at current dosage. Kidney transplant.<br>• Medical Reasoning: His kidney function is good, and he has no signs of infection.<br>• Medical Treatment: Continue current immunosuppressive medications.<br>• Patient Education and Counseling: I advised him to take it easy and avoid physical exertion that could exacerbate his arthritis. If his symptoms continue, we will consider further imaging and possibly referral to physical therapy.<br><br><br>Dialogue<br><br>|
|---|

###### Figure A6 | Example results: doctor-patient dialogue. Note this task is discluded from the reader study due to the unwieldiness of a reader parsing many transcribed conversations and lengthy text.

###### Table A2 | Comparison of our general approach (GPT-4 using ICL) against baselines specific to each individual dataset. We note the focal point of our study is not to achieve state-of-the-art quantitative results, especially given the discordance between NLP metrics and reader study scores. A - indicates the metric was not reported; a ◦ indicates the dataset was preprocessed differently.

|Dataset Baseline<br><br>|BLEU ROUGE-L BERTScore MEDCON|
|---|---|
|Open-i<br><br>Ours ImpressionGPT [76]<br><br>|46.0 68.2 94.7 64.9 - 65.4 - -|
|MIMIC-CXR<br><br>Ours RadAdapt [27] ImpressionGPT [76]<br><br>|29.6 53.8 91.5 55.6 18.9 44.5 90.0 -<br><br>- 47.9 - -|
|MIMIC-III<br><br>Ours RadAdapt [27] Med-PaLM M [25]<br><br>|11.5 34.5 89.0 36.5<br><br>16.2 38.7 90.2 15.2 32.0 - -|
|Patient questions<br><br>Ours ECL◦ [77]<br><br>|10.7 37.3 92.5 59.8 - 50.5 - -|
|Progress notes<br><br>Ours CUED [78]<br><br>|3.4 27.2 86.1 31.5 - 30.1 - -|
|Dialogue<br><br>Ours ACI-Bench◦ [44]|26.9 42.9 90.2 59.9 - 45.6 - 57.8<br><br>|

