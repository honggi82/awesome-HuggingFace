## SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models

Chih-Kai Yang*1 Yen-Ting Piao*1 Tzu-Wen Hsu2 Szu-Wei Fu3 Zhehuai Chen3 Ke-Han Lu1 Sung-Feng Huang3 Chao-Han Huck Yang3 Yu-Chiang Frank Wang3 Yun-Nung Chen1 Hung-yi Lee1

# arXiv:2510.16917v2[cs.SD]15Mar2026

### Abstract

Knowledge editing enables targeted updates without retraining, but prior work focuses on textual or visual facts, leaving abstract auditory perceptual knowledge underexplored. We introduce SAKE, the first benchmark for editing perceptual auditory attribute knowledge in large audio-language models (LALMs), which requires modifying acoustic generalization rather than isolated facts. We evaluate eight diverse editing methods on three LALMs across reliability, generality, locality, and portability, under single and sequential edits. Results show that most methods enforce edits reliably but struggle with auditory generalization, intra-attribute locality, and multimodal knowledge propagation, and often exhibit forgetting or degeneration in sequential editing. Additionally, fine-tuning the modality connector emerges as a more robust and balanced baseline compared with directly editing the LLM backbones. SAKE reveals key limitations of current methods and provides a foundation for developing auditoryspecific LALM editing techniques.

### 1. Introduction

Large language models (LLMs) (Grattafiori et al., 2024; Hurst et al., 2024; Touvron et al., 2023) have achieved remarkable success across a wide range of language tasks. As they scale and are increasingly deployed, knowledge editing (De Cao et al., 2021; Deng et al., 2025; Mitchell et al., 2022; Zheng et al., 2023) has become a key technique for efficiently updating specific model knowledge without full retraining while minimizing catastrophic forgetting. Knowledge editing can also be extended to bias mitigation (Chen et al., 2024a) and personalization (Lu et al., 2025d).

1National Taiwan University 2DouDou Capital 3NVIDIA. Correspondence to: Chih-Kai Yang <chihkaiyang1124@gmail.com>, Hung-yi Lee <hungyilee@ntu.edu.tw>.

Preprint. March 17, 2026.

Recent advances have extended LLMs to multimodal settings, including large vision-language models (LVLMs) (Li

- et al., 2023; Liu et al., 2023) and large audio-language models (LALMs) (Chu et al., 2024; Ghosh et al., 2025b; Hurst
- et al., 2024; Kuan et al., 2024; Lin et al., 2025; Lu et al.,
- 2025a;b; Tang et al., 2024). As multimodal models become more prevalent, it is increasingly important to understand whether and how knowledge editing can be applied beyond the textual domain. While recent benchmarks have explored visual knowledge editing in LVLMs (Cheng et al., 2023; Huang et al., 2024; Zhang et al., 2025), knowledge editing in the auditory modality remains largely unexplored.

Most knowledge editing studies (De Cao et al., 2021; Meng et al., 2022; Mitchell et al., 2022) focus on factual knowledge expressed as declarative propositions, often formalized as subject-relation-object triplets (Meng et al., 2022) (e.g., “Paris is the capital of France”). Visual knowledge editing in LVLMs largely follows this paradigm by treating images as alternative grounding for entity-centric facts. In contrast, perceptual knowledge is qualitatively different. Editing perceptual knowledge is conceptually closer to modifying style-level understanding, such as writing style in the text domain or image style in the vision domain, than to updating individual facts, as it operates on high-level abstractions rather than isolated factual associations.

Auditory attribute knowledge is a form of perceptual knowledge that reflects how models conceptualize attributes such as speaker gender and emotion. These attributes arise from perceptual abstractions over signals with substantial intraclass variability across speakers, prosody, and recording conditions. Editing such knowledge, therefore, requires modifying how models generalize perceptual evidence, rather than changing a factual statement. Hence, it is unclear whether editing methods developed for factual knowledge can generalize to abstract perceptual concepts in auditory modalities.

To this end, we introduce SAKE (Speech and Audio Attribute Knowledge Editing Benchmark), the first benchmark for auditory attribute knowledge editing in LALMs (Figure 1). Auditory attribute knowledge editing targets modifications of an LALM’s perceptual understanding of specific attributes, such as changing its auditory knowledge of frogs’

[Figure 1]

Figure 1. Left: Conceptual illustration of the knowledge scope affected by an edit. Right: Overview of SAKE benchmark, targeting auditory attributes including speaker gender, emotion, spoken language, and animal sounds. Reliability measures edit success, Generality its consistency across equivalent data, Locality the preservation of unrelated knowledge, and Portability its transfer to connected knowledge. For example, after editing “frog” to “dog,” the answer of the portability question should change from “Insectivore” to “Omnivore.”

vocalizations to that of dogs (Figure 1), while preserving related reasoning and unrelated knowledge. Such editing enables applications such as LALM error correction for misrecognition, debiasing, and voice assistant personalization, which are inherently more challenging than text-based settings due to the complexity of auditory modalities. SAKE spans diverse auditory attributes and evaluates editing methods along four dimensions (Yao et al., 2023): reliability, generality, locality, and portability.

We evaluate eight representative editing methods on three LALMs, DeSTA2.5-Audio (Lu et al., 2025b), Qwen2Audio (Chu et al., 2024), and Audio Flamingo 3 (Ghosh et al., 2025a). While these methods have proven effective in textual domains, our results reveal persistent challenges in auditory attribute knowledge editing, including preserving intra-attribute knowledge and enabling edit propagation to connected reasoning. Under sequential editing, most methods suffer severe forgetting. Together, these findings highlight the unique challenges of auditory knowledge editing and the need for future advances in this research area.

Our contributions are three-fold: (1) We introduce SAKE, the first comprehensive benchmark for systematically evaluating auditory attribute knowledge editing of LALMs; (2) We show that existing methods struggle to preserve nontarget auditory knowledge and to propagate edits during reasoning, revealing key limitations in editing abstract perceptual attributes; (3) Through extensive analysis, we show that fine-tuning the modality connector is strong and more balanced across the evaluation dimensions, highlighting an important direction for auditory-specific editing techniques.

### 2. Related Work

##### 2.1. Knowledge Editing

Knowledge editing (Zhu et al., 2024) aims to efficiently modify the models’ knowledge while avoiding catastrophic forgetting that may arise from retraining directly on the target knowledge. Existing methods (De Cao et al., 2021; Meng et al., 2022; Mitchell et al., 2022; Zheng et al., 2023) adopt various strategies: using a hypernetwork to predict parameter updates for incorporating new knowledge (De Cao et al., 2021; Mitchell et al., 2022), identifying and adjusting neurons associated with specific knowledge (Jiang et al., 2025; Meng et al., 2022; 2023), or leveraging in-context learning (ICL) to enforce updated knowledge (Zheng et al.,

- 2023). Beyond correcting factual knowledge, these techniques have also been applied to bias mitigation (Chen et al.,
- 2024a), detoxification (Wang et al., 2024a), personalization (Lu et al., 2025d), unlearning (Li et al., 2025), etc.

More recently, researchers have begun to investigate factual knowledge editing in large vision-language models (LVLMs) (Li et al., 2023; Liu et al., 2023). For example, Cheng et al. (2023) introduced the MMEdit benchmark to explore editing visual factual knowledge, while subsequent works like VLKEB (Huang et al., 2024) and MCMKE (Zhang et al., 2025) expanded the evaluation scope to provide a more comprehensive understanding of editing in visual modalities. However, no prior work has examined editing auditory attribute knowledge in LALMs, which involves abstract perceptual and auditory concepts rather than concrete facts, distinguishing SAKE from existing research.

##### 2.2. Large Audio-Language Models (LALMs)

LALMs extend text-based LLMs to auditory modalities such as speech and audio, opening new possibilities for auditory understanding (Chu et al., 2023; 2024; Ghosh et al., 2025a;

Gong et al., 2023; Lu et al., 2025b; Tang et al., 2024). These models typically integrate auditory encoders (Radford et al., 2023) with an LLM backbone (Touvron et al., 2023; Yang

- et al., 2024a) through fine-tuning. While these works advance the integration of auditory knowledge into LLMs, little attention has been given to how auditory-specific knowledge can be edited, which motivates our study.

### 3. SAKE: Speech & Audio Attribute Knowledge Editing Benchmark

##### 3.1. Problem Formulation

Given an LALM f with parameters θ and an editing dataset Dedit = {(ae,xe,ye)}, where ae denotes the auditory input, xe the text input, and ye the desired edit target, knowledge editing aims to update the model such that the edited parameters θ′ enable the LALM to faithfully generate the edit target: f(ae,xe;θ′) = ye.

We focus on editing auditory attribute knowledge within LALMs, including their perception and understanding of speaker gender, emotion, spoken language, and animal sounds. Here, ye corresponds to new auditory attribute labels (e.g., emotions or languages) that differ from the original attribute labels yo associated with ae. For example, given a speech labeled with a happy emotion, we may edit the LALM so that it instead perceives the speech as sad.

For comprehensive evaluation, we introduce the four evaluation dimensions of SAKE and the corresponding metrics, followed by dataset construction details for each dimension.

##### 3.2. Evaluation Dimensions and Corresponding Metrics

We introduce the four dimensions of knowledge editing, namely reliability, generality, locality, and portability, together with their corresponding evaluation metrics.

Reliability. The reliability metric Srel measures the proportion of editing instances in Dedit for which the edited model correctly generates the corresponding edit target. It reflects how consistently the editing method updates the model in the desired manner, and is defined as

e,xe,ye)∼Dedit I f(ae,xe;θ′) = ye , (1)

Srel = E(a

where I denotes the indicator function, which returns 1 if the condition holds and 0 otherwise.

Generality. The edited models should not only generate the correct edit target for the editing data itself but also produce consistent outputs for equivalent neighborhoods of the editing data, such as speech samples sharing the same emotion as ae or paraphrased variants of xe. This

requirement is quantified by the generality metric Sgen:

I f(a′e,x′e;θ′) = ye , (2)

Sgen = E(a

e,xe,ye)∼Dedit (a′e,x′e)∼N(ae,xe)

where N(ae,xe) denotes the aforementioned equivalent neighborhood of the editing data (ae,xe).

Locality. While updating the edit target, the edit should also preserve unrelated knowledge to avoid unintended side effects. The locality metric Sloc evaluates how well an editing method maintains the model’s knowledge outside the editing scope. Given a set of out-of-scope data L(ae,xe,ye) = {(aℓ,xℓ,yℓ)}, consisting of auditory inputs, text inputs, and ground-truth labels, Sloc is defined as the proportion of out-of-scope data where the model’s behavior remains unchanged after editing:

I f(aℓ,xℓ;θ′)

Sloc = E (a

e,xe,ye)∼Dedit (aℓ,xℓ,yℓ)∼L(ae,xe,ye)

= f(aℓ,xℓ;θ) .

(3)

Note that the locality metric evaluates whether the postedit model preserves the knowledge and behavior on data irrelevant to the edit, rather than the accuracy on out-ofscope instances. For locality with respect to purely textual abilities, we set aℓ = None, as no auditory input is involved.

Portability. Knowledge is not completely disentangled or isolated but rather interconnected. Editing one piece of knowledge may influence other related one. For example, if we edit an LALM’s perception of a frog’s sound to that of a dog, the model’s knowledge of the corresponding physical characteristics of that animal should also be updated. The portability metric Sport evaluates how well the editing generalizes the updated knowledge to other related knowledge:

Sport = E (a

e,xe,ye)∼Dedit (ap,xp,yp)∼P(ae,xe,ye)

I f(ap,xp;θ′) = yp .

(4)

P(ae,xe,ye) denotes the set of data connected to the edited knowledge. ap, xp, and yp represent the auditory input, text input, and ground-truth labels of connected instances.

All the alignment and consistency in the definition of these metrics are determined with LLM-as-a-judge (Chiang & Lee, 2023), which is detailed in Sec. 4.3 and Appendix E.

##### 3.3. Dataset Construction

We introduce the SAKE benchmark to evaluate the knowledge editing methods on editing the auditory attribute knowledge in LALMs with respect to the metrics detailed in Sec. 3.2. We benchmark the editing methods on LALMs with speech and audio multiple-choice question answering.

Auditory Attributes and Audio Sources. SAKE focuses on various auditory attributes: speaker gender, speaker emotion, spoken language, and animal sound. These attributes are fundamental to speech and audio understanding, widely used in real-world applications, and consistently evaluated in LALM benchmarks (Huang et al., 2025; Sakshi et al., 2025; Yang et al., 2025b). We source audios and attribute labels from SAKURA benchmark (Yang et al., 2025b).

Editing Pairs and Reliability Dataset. For each attribute, we construct editing pairs (yo,ye) by uniformly sampling distinct original and target labels, resulting in 300 balanced editing pairs per attribute. To avoid bias, we ensure that all labels for a given attribute appear with approximately equal frequency as both yo and ye. Each pair defines an editing instance (ae,xe,ye), where ae is an audio sample labeled with yo and xe is a corresponding attribute recognition question from SAKURA. In total, we construct 1,200 editing instances, forming the editing dataset Dedit, which is used both to apply edits and to evaluate reliability.

Generality Dataset Construction. For each editing instance (ae,xe,ye) in Dedit, we construct its equivalent neighborhood N(ae,xe) by sampling an alternative audio a′e with the same attribute label as ae from the dataset and by paraphrasing the text question xe into x′e. The paraphrased data are manually verified. Based on these variations, we create testing instances for evaluating generality, considering three cases: (1) Type 1: Textual equivalent neighborhood (ae,x′e); (2) Type 2: Auditory equivalent neighborhood (a′e,xe); (3) Type 3: Equivalent neighborhood involving both auditory and text modalities (a′e,x′e). By incorporating these types of testing data, we comprehensively assess how well the editing methods extend the edited knowledge across the equivalent neighborhood.

Locality Dataset Construction. For each editing instance (ae,xe,ye) in Dedit, we build an out-of-scope set L(ae,xe,ye) = (aℓ,xℓ,yℓ).

When aℓ ̸= None, we consider four types of auditory knowledge locality (Audio Locality): (1) Type 1: Editing one attribute should not affect others. We sample SAKURA question–answer (QA) pairs requiring recognition of attributes different from the edited one; (2) Type 2: Within the same attribute, knowledge regarding labels not involved in the edit (neither yo nor ye) should remain unchanged. We sample QA pairs targeting the same attribute with ground truth different from both yo and ye, except for gender due to its binary labels; (3) Type 3: When editing from yo to ye, the model’s original knowledge of ye should be preserved. We evaluate this using QA pairs with the same attribute and ground truth ye; (4) Type 4: Editing should not disrupt general auditory capability. We draw pairs from

Table 1. Summary statistics of SAKE dataset splits.

Split # Instances # Speech/Audio Inputs # QA Pairs

Train 4,000 31,000 35,000 Test 1,200 10,500 11,700

Dynamic-SUPERB Phase-2 (Huang et al., 2025), excluding tasks involving the edited attributes to ensure irrelevance to the edited knowledge.

When ae = None, we evaluate text locality with textual QA pairs from MMLU (Hendrycks et al., 2021).

Portability Dataset Construction. For each instance (ae,xe,ye) in Dedit, we construct a set of connected knowledge P(ae,xe,ye) associated with the edited attribute. As SAKURA is designed to evaluate the integration of world knowledge with auditory attributes, it provides attributelinked knowledge (e.g., physical characteristics of animal labels). Building on this, we design questions that probe P(ae,xe,ye). To ensure unambiguous evaluation, we ex-

clude questions whose answers are valid for both yo and ye (e.g., avoiding “tail” when editing from dog to cat), ensuring that portability genuinely requires knowledge updating.

Training Dataset Construction and Dataset Summary. We construct a training dataset for editing methods that require auxiliary training data or access to data beyond test instances. It follows the same procedures as the reliability, generality, and locality datasets, with training and testing sets strictly disjoint to prevent leakage. Statistics are summarized in Table 1, with additional details in Appendix B.2.

### 4. Experimental Settings

We use three LALMs, DeSTA2.5-Audio (Lu et al., 2025b), Qwen2-Audio-Instruct (Chu et al., 2024), and Audio Flamingo 3 (Ghosh et al., 2025a), chosen for their strong benchmark performance (Chen et al., 2024b; Huang et al., 2025; Lu et al., 2025c; Sakshi et al., 2025; Yang et al., 2025b). We refer to them as DeSTA, Qwen, and AF, respectively. Details about the model choice are in Appendix C. We apply greedy decoding and assess editing methods under two settings: single editing and sequential editing.

##### 4.1. Editing Methods

We evaluate eight knowledge editing methods that are widely adopted in prior works (Chen et al., 2025; Cheng et al., 2023; Du. et al., 2025; Huang et al., 2024; Ma et al., 2025a; Zhang et al., 2025), focusing on their effectiveness in modifying abstract auditory attribute knowledge. These methods span diverse methodological paradigms, including fine-tuning-based, hypernetwork-driven, optimizationcentric, memory-augmented, and in-context editing ap-

proaches. This coverage enables systematic evaluation across a wide range of editing methods. We provide an overview of these methods below, with implementation details deferred to Appendix D.

Fine-tuning is a common approach for adapting pre-trained models to new knowledge. Following prior work on LVLMs (Cheng et al., 2023; Huang et al., 2024; Zhang

- et al., 2025), we compare fine-tuning two parts in LALMs: the last layer of the LLM backbone (FT (LLM)) and the modality connector between the audio encoder and the LLM backbone (FT (Audio)). Knowledge Editor (KE) (De Cao et al., 2021) and MEND (Mitchell et al., 2022) trains a hypernetwork to transform the fine-tuning gradients into parameter updates for the edit. UnKE (Deng et al., 2025) optimizes specific neurons of the chosen layers to produce the edit target. In-Context Knowledge Editing (IKE) (Zheng

- et al., 2023) leverages in-context learning (ICL) to enforce knowledge updates. We consider two variants: Instructionbased IKE (I-IKE), which encodes edits solely through natural language instructions in system prompts, and Instruction+Example IKE (IE-IKE), which provides auditory examples retrieved from the training set. WISE (Wang
- et al., 2024b) edits models using a dual-memory design that keeps pretrained FFN weights intact while storing editspecific updates in a lightweight side memory.

##### 4.2. Single Editing and Sequential Editing

Single editing evaluates the performance of editing a single piece of knowledge; sequential editing evaluates the performance after applying several edits continuously on different knowledge, which better reflects real-world scenarios.

For sequential editing, we construct ten independent sequences, each with ten editing instances sampled from the data used for single editing. An editing sequence is denoted

- as S = {(a(et),x(et),yo(t),ye(t))}10t=1, where (a(et),x(et),ye(t))

is sampled from Dedit and the original label yo(t) is retrieved from our audio dataset. Here, t indexes the order of edits within the sequence.

To measure how long an edit remains effective under subsequent edits, we define the gap as the number of editing steps between when an edit is applied and when it is evaluated (Figure 2). If an edit is introduced at step j and evaluated

- at step i, then the gap is i − j. For consistency, we only consider the first five edits and require the gap to be at most five. This restriction keeps the number of samples comparable across gaps, since larger gaps naturally yield fewer available evaluations. To guarantee the validity of edit sequences, we impose two rules when sampling: (1) Editing pair independence: All original and edited labels in the se-

quence are mutually distinct, i.e., yo(1),ye(1),...,yo(10),ye(10) appear only once within the sequence. This avoids contra-

[Figure 2]

Figure 2. Example of sequential editing. For comparability, only the first five edits are evaluated, with the evaluation gap capped at five. For instance, Edit 2 is evaluated with a gap of 0, while after Edit 10, only Edit 5 is evaluated with a gap of 5.

dictions that could compromise the evaluation of edits with subsequent ones (e.g., editing “dog sounds” to “cat sounds” and later editing “dog sounds” to “frog sounds”); and (2) Audio locality independence: For each sequential editing instance (a(et),x(et),yo(t),ye(t)), all samples in its audio locality dataset L(a(et),x(et),ye(t)), denoted by (a(ℓt),x(ℓt),yℓ(t)) where a(ℓt) ̸= None, are unrelated to the original labels of all edited instances yo(1..10), thereby ensuring independent evaluation of the current edit.

##### 4.3. Evaluator

Because LALMs often generate descriptive responses, we adopt LLM-as-a-judge (Chiang & Lee, 2023) to compute the metrics introduced in Sec. 3.2, using GPT-5 mini as the evaluator. To validate these judgments, we conduct human evaluation on 420 randomly sampled cases, achieving 98.10% agreement, indicating strong robustness. Additional details and prompts are provided in Appendix E.

### 5. Results 5.1. Single Editing

The main results of different editing methods on SAKE are shown in Table 2. Detailed results for each auditory attribute and relevant discussions are provided in Appendix F.

Reliability: Parameter-updating methods are almost perfectly reliable, while IKE variants underperform. For all LALMs, parameter-updating methods (i.e., methods other than IKE variants) all achieve high reliability (mostly ≥95% and often ≈100%), showing their effectiveness in enforcing the edits. Among them, WISE consistently attains the best performance. In contrast, I-IKE and IE-IKE perform poorly. Interestingly, I-IKE outperforms IE-IKE on DeSTA and Qwen, despite the latter using additional auditory examples. We attribute this to those LALMs’ limited in-context learning ability: they struggle to handle multi-audio inputs and leverage examples, unlike in LLMs (Zheng et al., 2023) and LVLMs (Cheng et al., 2023; Huang et al., 2024) where such methods are effective.

Generality: Enforcing edits does not guarantee consistent generalization beyond the edited instances. Over-

Table 2. Performance (%) of editing methods on three models. Generality and audio locality scores are averaged across all attributes and evaluation types. Best and second-best results on individual metrics are shown in bold and underlined, respectively.

Model Metrics FT (LLM) FT (Audio) KE MEND UnKE I-IKE IE-IKE WISE

Reliability (↑) 99.75 99.50 99.58 95.25 96.09 64.25 40.58 100.00 Generality (↑) 98.75 86.06 99.17 94.94 87.42 61.08 39.61 100.00 Audio Locality (↑) 64.93 68.13 79.49 71.44 57.18 65.49 58.89 38.90 Text Locality (↑) 82.67 100.00 93.08 92.00 88.50 61.92 56.67 83.08 Portability (↑) 18.83 55.33 17.33 18.08 17.17 71.33 34.33 17.34

DeSTA

Reliability (↑) 100.00 100.00 95.33 100.00 98.67 10.33 7.83 100.00 Generality (↑) 99.94 81.81 86.69 95.31 98.64 7.00 6.44 99.34 Audio Locality (↑) 67.40 90.49 83.36 83.29 68.00 87.47 82.82 70.04 Text Locality (↑) 75.58 100.00 85.67 87.25 72.84 55.67 51.17 82.09 Portability (↑) 24.00 49.92 26.75 26.25 26.58 28.50 27.58 27.84

Qwen

Reliability (↑) 100.00 100.00 100.00 99.92 99.92 25.00 58.83 100.00 Generality (↑) 93.31 76.08 98.89 93.92 91.17 15.08 58.36 77.69 Audio Locality (↑) 87.67 90.24 81.38 90.64 76.03 87.07 40.42 91.18 Text Locality (↑) 97.00 100.00 95.83 98.67 81.17 81.17 76.67 96.08 Portability (↑) 31.67 49.50 31.92 32.08 31.75 37.08 40.67 30.75

AF

all generality scores are consistently lower than reliability across all LALMs, indicating that successful edits on target instances do not readily extend to the equivalent neighborhoods. WISE and KE achieve the strongest average generality on DeSTA and AF, respectively, while FT (LLM) performs best on Qwen. Interestingly, FT (LLM) consistently outperforms FT (Audio) in average generality despite comparable reliability, suggesting that updating the LLM backbone yields better generalization than tuning the modality connector alone. We analyze the performances across types of generality in Sec. 5.2.

Audio Locality: Preserving unrelated auditory knowledge remains challenging for most editing methods. No method consistently dominates across LALMs. On average, audio locality scores of the methods are lower than reliability and generality, indicating that enforcing auditory edits often interferes with unrelated auditory knowledge. FT (LLM) shows lower audio locality than FT (Audio), suggesting that editing different model components induces different levels of collateral effects in the auditory domain. Together with the generality results, these observations point to a trade-off between generalizing edited knowledge and preserving unrelated auditory knowledge. Although IKE variants do not achieve high reliability or generality, they preserve audio locality to a certain extent, likely because limited in-context learning capability constrains the propagation of the edits, albeit at the cost of weaker editing effectiveness. We discuss different types of audio locality in Sec. 5.2.

Text locality: Largely preserved when edits avoid modifying the LLM backbone. FT (Audio) maintains perfect text locality, as it only updates the modality connector and leaves text-only behavior unaffected. In contrast, FT (LLM) substantially degrades text locality on most LALMs due to direct backbone updates. KE and MEND preserve text

locality reasonably well through regularization. Other methods show larger drops, indicating that safeguarding textual capabilities requires explicit considerations during editing.

Table 3. Standard deviation (Std.) and range of portability scores across knowledge categories for the top-two methods per model.

Model Method Std. (%) Range (%) DeSTA I-IKE 12.66 45.13

FT (Audio) 22.64 88.18 Qwen FT (Audio) 17.09 62.27

I-IKE 15.45 72.37 AF FT (Audio) 16.64 65.46

IE-IKE 14.51 54.40

Portability: Propagating auditory edits to connected knowledge remains an open challenge. Overall, current editing methods do not guarantee portability when modifying auditory attribute knowledge. While no approach consistently excels, FT (Audio) exhibits comparatively more balanced portability behavior across settings. This observation suggests that edits applied through the modality connector can interact differently with related knowledge than backbone updates, though portability remains far from resolved. More broadly, the limited portability observed across methods highlights a practical challenge for scenarios where modifying an auditory attribute is expected to induce coherent updates to a broader set of related knowledge.

We further analyze portability by categorizing the related knowledge in the portability set. For each LALM, we select the two best-performing editing methods and report the standard deviation and range of their portability scores across knowledge categories in Table 3, with full results provided in Appendix I. The large variation across categories shows that portability failures are not uniform: current editing methods can successfully update certain types of related

[Figure 3]

- (a) Results for DeSTA.

[Figure 4]

- (b) Results for Qwen.

[Figure 5]

- (c) Results for AF.

Figure 3. Performances of editing methods on types of generality (left) and audio locality (right). Curves represent evaluation types.

##### 5.2. Detailed Analyses of Different Types of Generality and Audio Locality

We compare editing performance across different types of generality and audio locality in Figure 3. Unlike prior works in vision domains (Cheng et al., 2023; Du. et al., 2025; Huang et al., 2024; Zhang et al., 2025) that do not explicitly differentiate multiple forms of knowledge generalization and locality, our design decomposes both dimensions into distinct types, enabling more fine-grained analyses of how edits generalize and how unrelated auditory knowledge is preserved.

For generality, editing methods generally perform well on type 1 (textual neighborhood) but degrade on type 2, which requires generalization to acoustically similar yet non-identical inputs with the same labels. Type 3, which combines both textual and auditory neighborhoods, is consistently the most challenging, indicating that extending edited knowledge along the auditory dimension remains substantially more difficult than along the textual one.

For audio locality, type 2 probes knowledge within the same attribute but unrelated to the edit, and proves hardest to preserve compared to knowledge of other attributes (type 1) or the edited target itself (type 3). This pattern shows that modifying one aspect of an auditory attribute can inadvertently affect other aspects of the same attribute, revealing a form of intra-attribute entanglement in auditory attribute knowledge. Consistent with this observation, FT (LLM) exhibits substantially lower locality on type 2 than FT (Audio), despite its stronger generalization performance, suggesting a tension between generalizing edits and preserving unrelated knowledge within the same attribute. We provide a more detailed discussion of this phenomenon in Appendix H.

Type 4 assesses the preservation of general auditory processing ability. Most methods show clear degradation, suggesting that targeted edits often interfere with broader auditory competence. In contrast, KE and MEND better preserve original behavior, likely due to their explicit localityoriented regularization during hypernetwork training.

knowledge while leaving others unchanged, indicating that portability exhibits substantial variation across knowledge categories rather than an all-or-nothing outcome.

Summary. While most methods can enforce the desired behavior on edited instances, they struggle to generalize to equivalent ones, propagate edits to connected knowledge, and preserve unrelated knowledge within the same attribute, particularly in the auditory domain. Fine-tuning remains a strong baseline, with FT (Audio) showing consistently competitive performance. These results underscore that editing auditory attribute knowledge remains highly challenging and calls for auditory-specific editing methods.

###### 5.3. Sequential Editing Figure 4 shows the results on sequential editing.

Significant forgetting under sequential editing in reliability and generality. Most methods show clear declines in reliability and generality as the edit gap increases, indicating substantial forgetting of previously edited auditory knowledge. In contrast, MEND and KE degrade rapidly after only a few edits in most cases, except for MEND on AF. We attribute this behavior to their sensitivity to sequential updates, which we will discuss later.

##### Sequential editing can induce degeneration. Some editing

[Figure 6]

- (a) Results for DeSTA.

[Figure 7]

- (b) Results for Qwen.

[Figure 8]

(c) Results for AF. Figure 4. Comparison of sequential editing results across models by edit gap (0–5).

methods exhibit degeneration, characterized by repetitive or nonsensical outputs (Gupta et al., 2024; Holtzman et al., 2020; Hsueh et al., 2024; Yang et al., 2024b). This phenomenon is widely observed in many cases, e.g., MEND on DeSTA and Qwen, KE on all models, and UnKE on AF. These results suggest that such methods are particularly sensitive to sequential editing, where successive updates interfere with one another and induce model collapse. This highlights a key limitation of current editing approaches. Statistics and qualitative examples of degeneration are provided in Appendix J. In contrast, methods like FT (Audio) show substantially less degeneration after sequential updates, showing greater robustness.

Strength-stability trade-off in IKE variants. Although weaker in single-edit settings, the IKE variants remain relatively stable under larger gaps, with markedly smaller degradation than parameter-updating methods; in particular, I-IKE achieves the strongest long-term reliability and generality on DeSTA.

No significant locality degradation under sequential editing for most methods. Most methods do not exhibit the sharp performance drops observed in reliability and generality, indicating that sequential editing has a comparatively weaker impact on locality. Nevertheless, as discussed in Sec. 5.2, substantial room for improvement remains, since none of the methods consistently preserve all kinds of unre-

lated knowledge. Notably, KE shows a pronounced decline at larger edit gaps on all LALMs due to degeneration issues.

Portability consistently falls below random baseline under sequential editing for most methods. Under sequential editing, most methods demonstrate persistently poor portability, consistent with the findings reported in Sec. 5.1. In particular, most methods perform below the random baseline (27.5%) even at small edit gaps, indicating that their ability to transfer edits to connected knowledge is limited. As a result, increasing the edit gap does not lead to substantial additional degradation or noticeable fluctuations, since portability performance is already near its lower bound.

FT (Audio) remains a strong baseline under sequential editing. Among all methods, FT (Audio) demonstrates relatively strong robustness across most LALMs and metrics, with notable degradation observed only in reliability and generality on DeSTA. Combined with the results in Sec. 5.1, where FT (Audio) achieves the most balanced performance in single-edit settings, these findings indicate that FT (Audio) remains a competitive baseline for auditory knowledge editing, despite its conceptual simplicity.

### 6. Conclusions

We present the first study on editing auditory attribute knowledge in LALMs and introduce SAKE, the first benchmark

for evaluating this capability across four dimensions. Experiments with eight methods on three LALMs reveal challenges in preserving non-target auditory knowledge and propagating edits during reasoning. Under sequential editing, most methods suffer severe forgetting or degeneration. Overall, fine-tuning the modality connector serves as a strong and more balanced baseline for future work. In contrast, many methods that are effective in textual domains degrade substantially on auditory attributes, highlighting the unique challenges of auditory knowledge editing.

### Impact Statement

This work establishes auditory attribute knowledge editing as a previously unexplored research direction for large audio-language models (LALMs). By systematically examining the modification of perceptual auditory knowledge, we identify fundamental limitations of existing editing methods. We introduce SAKE, the first benchmark for evaluating this capability, and present a comprehensive analysis that highlights key challenges in edit reliability, knowledge generalization, preservation, and propagation. We expect SAKE to serve as a foundation for future research in this area. All related resources will be publicly released after the review process.

Knowledge editing can enhance LALMs through error correction, bias mitigation, and personalization. At the same time, modifying perceptual knowledge introduces risks of misuse, including adversarial manipulation of model behavior. This work aims to promote the responsible development and deployment of LALMs by explicitly characterizing these limitations and informing future research on principled model editing.

### References

Ardila, R., Branson, M., Davis, K., Kohler, M., Meyer, J., Henretty, M., Morais, R., Saunders, L., Tyers, F., and Weber, G. Common voice: A massively-multilingual speech corpus. In Calzolari, N., B´echet, F., Blache, P., Choukri, K., Cieri, C., Declerck, T., Goggi, S., Isahara, H., Maegaard, B., Mariani, J., Mazo, H., Moreno, A., Odijk, J., and Piperidis, S. (eds.), Proceedings of the Twelfth Language Resources and Evaluation Conference, pp. 4218–4222, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-9554634-4. URL https://aclanthology.org/2020.

lrec-1.520/.

Cao, H. et al. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5(4):377–390, 2014.

Chen, Q., Zhang, T., Wang, C., He, X., Wang, D., and

Liu, T. Attribution analysis meets model editing: Advancing knowledge correction in vision language models with visedit. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 2168–2176, 2025.

Chen, R., Li, Y., Xiao, Z., and Liu, Z. Large language model bias mitigation from the perspective of knowledge editing. In ICLR 2024 Workshop on Secure and Trustworthy Large Language Models, 2024a. URL https: //openreview.net/forum?id=LflQOFSl3n.

Chen, Y., Yue, X., Zhang, C., Gao, X., Tan, R. T., and Li, H. Voicebench: Benchmarking llm-based voice assistants. arXiv preprint arXiv:2410.17196, 2024b.

Cheng, S., Tian, B., Liu, Q., Chen, X., Wang, Y., Chen, H., and Zhang, N. Can we edit multimodal large language models? In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13877–13888, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 856. URL https://aclanthology.org/2023.

emnlp-main.856/.

Chiang, C.-H. and Lee, H.-y. Can large language models be an alternative to human evaluations? In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15607–15631, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. acl-long.870. URL https://aclanthology.org/ 2023.acl-long.870/.

Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., and Zhou, J. Qwen-audio: Advancing universal audio understanding via unified large-scale audiolanguage models. arXiv preprint arXiv:2311.07919, 2023.

Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

De Cao, N., Aziz, W., and Titov, I. Editing factual knowledge in language models. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6491–6506, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. emnlp-main.522. URL https://aclanthology.

org/2021.emnlp-main.522/.

Deng, J., Wei, Z., Pang, L., Ding, H., Shen, H., and Cheng, X. Everything is editable: Extend knowledge editing to

unstructured data in large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=X5rO5VyTgB.

Du., Y., Jiang, K., Gao, Z., Shi, C., Zheng, Z., Qi, S., and Li, Q. MMKE-bench: A multimodal editing benchmark for diverse visual knowledge. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=v8qABSeeKO.

Feng, B.-H., Liu, C.-F., Liang, Y.-H. L., Yang, C.-K., Fu, S.-W., Chen, Z., Lu, K.-H., Huang, S.-F., Yang, C.-H. H., Wang, Y.-C. F., et al. Investigating safety vulnerabilities of large audio-language models under speaker emotional variations. arXiv preprint arXiv:2510.16893, 2025.

Ghosh, S., Goel, A., Kim, J., Kumar, S., Kong, Z., gil Lee, S., Yang, C.-H. H., Duraiswami, R., Manocha, D., Valle, R., and Catanzaro, B. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https:

//openreview.net/forum?id=FjByDpDVIO.

Ghosh, S., Kong, Z., Kumar, S., Sakshi, S., Kim, J., Ping, W., Valle, R., Manocha, D., and Catanzaro, B. Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities. In Fortysecond International Conference on Machine Learning, 2025b. URL https://openreview.net/forum? id=xWu5qpDK6U.

Gong, Y., Liu, A. H., Luo, H., Karlinsky, L., and Glass, J. Joint audio and speech understanding. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–8, 2023. doi: 10.1109/ASRU57964. 2023.10389742.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,

- A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gupta, A., Rao, A., and Anumanchipalli, G. Model editing at scale leads to gradual and catastrophic forgetting. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 15202–15232, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl. 902. URL https://aclanthology.org/2024.

findings-acl.902/.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask

language understanding. In International Conference on Learning Representations, 2021. URL https:// openreview.net/forum?id=d7KBjmI3GmQ.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=rygGQyrFvH.

Hsueh, C.-H., Huang, P. K.-M., Lin, T.-H., Liao, C. W., Fang, H.-C., Huang, C.-W., and Chen, Y.-N. Editing the mind of giants: An in-depth exploration of pitfalls of knowledge editing in large language models. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9417–9429, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp. 550. URL https://aclanthology.org/2024.

findings-emnlp.550/.

Huang, C.-y., Chen, W.-C., wen Yang, S., Liu, A. T., Li, C.-A., Lin, Y.-X., Tseng, W.-C., Diwan, A., Shih, Y.-J., Shi, J., Chen, W., Chen, X., Hsiao, C.-Y., Peng, P., Wang,

- S.-H., Kuan, C.-Y., Lu, K.-H., Chang, K.-W., Yang, C.-K., Gutierrez, F. A. R., Kuan-Po, H., Arora, S., Lin, Y.-K., To, C. M., Yeo, E., Chang, K., Chien, C.-M., Choi, K., Hsieh, C.-H., Lin, Y.-C., Yu, C.-E., Chiu, I.-H., Guimar˜aes, H., Han, J., Lin, T.-Q., Lin, T.-Y., Chang, H., Chang, T.-W., Chen, C. W., Chen, S.-J., Chen, Y.-H., Cheng, H.-C., Dhawan, K., Fang, J.-L., Fang, S.-X., CHIANG, K. Y. F., Fu, C. A., Hsiao, H.-F., Hsu, C. Y., Huang, S.-S., Wei, L. C., Lin, H.-C., Lin, H.-H., Lin, H.-T., Lin, J.-R., Liu,
- T.-C., Lu, L.-C., Pai, T.-M., Pasad, A., Kuan, S.-Y. S., Shon, S., Tang, Y., Tsai, Y.-S., Chiang, W. J., Wei, T.-C., Wu, C., Wu, D.-R., Yang, C.-H. H., Yang, C.-C., Yip, J. Q., Yuan, S.-X., Wu, H., Livescu, K., Harwath, D., Watanabe, S., and yi Lee, H. Dynamic-SUPERB phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=s7lzZpAW7T.

Huang, H., Zhong, H., Yu, T., Liu, Q., Wu, S., Wang, L., and Tan, T. Vlkeb: A large vision-language model knowledge editing benchmark. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 9257–9280. Curran Associates, Inc., 2024.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A.,

Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Jiang, H., Fang, J., Zhang, N., Wan, M., Ma, G., Wang, X., He, X., and Chua, T.-S. Anyedit: Edit any knowledge encoded in language models. In Forty-second International Conference on Machine Learning, 2025. URL https:

//openreview.net/forum?id=aJIoBur0Ef.

Kuan, C.-Y., Yang, C.-K., Huang, W.-P., Lu, K.-H., and Lee, H.-y. Speech-copilot: Leveraging large language models for speech processing via task decomposition, modularization, and program generation. In 2024 IEEE Spoken Language Technology Workshop (SLT), pp. 1060– 1067. IEEE, 2024.

Li, J., Li, D., Savarese, S., and Hoi, S. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Krause,

- A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 19730–19742. PMLR, 23–29 Jul 2023. URL https:// proceedings.mlr.press/v202/li23q.html.

Li, Z., Wang, X., Shen, W. F., Kurmanji, M., Qiu, X., Cai, D., Wu, C., and Lane, N. D. Editing as unlearning: Are knowledge editing methods strong baselines for large language model unlearning? arXiv preprint arXiv:2505.19855, 2025.

Lin, Y.-X., Yang, C.-K., Chen, W.-C., Li, C.-A., Huang, C.-y., Chen, X., and Lee, H.-y. A preliminary exploration with gpt-4o voice mode. arXiv preprint arXiv:2502.09940, 2025.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 34892– 34916. Curran Associates, Inc., 2023.

Lu, K.-H., Chen, Z., Fu, S.-W., Yang, C.-H. H., Balam, J., Ginsburg, B., Wang, Y.-C. F., and Lee, H.-Y. Developing instruction-following speech language model without speech instruction-tuning data. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2025a. doi: 10.1109/ICASSP49660.2025.10889444.

Lu, K.-H., Chen, Z., Fu, S.-W., Yang, C.-H. H., Huang,

- S.-F., Yang, C.-K., Yu, C.-E., Chen, C.-W., Chen, W.-C., yu Huang, C., Lin, Y.-C., Lin, Y.-X., Fu, C.-A., Kuan, C.-Y., Ren, W., Chen, X., Huang, W.-P., Hu, E.-P., Lin,
- T.-Q., Wu, Y.-K., Huang, K.-P., Huang, H.-Y., Chou, H.C., Chang, K.-W., Chiang, C.-H., Ginsburg, B., Wang,

Y.-C. F., and yi Lee, H. Desta2.5-audio: Toward generalpurpose large audio language model with self-generated cross-modal alignment. 2025b. URL https://arxiv.

org/abs/2507.02768.

Lu, K.-H., Kuan, C.-Y., and yi Lee, H. Speech-IFEval: Evaluating Instruction-Following and Quantifying Catastrophic Forgetting in Speech-Aware Language Models. In Interspeech 2025, pp. 2078–2082, 2025c. doi: 10.21437/Interspeech.2025-619.

Lu, Z., Xu, D., Cai, D., Li, Z., Liu, W., Liu, F., Wang, S., and Xu, M. Mobiedit: Resource-efficient knowledge editing for personalized on-device llms. arXiv preprint arXiv:2506.13772, 2025d.

- Ma, Y., Hong, X., Zhang, S., Li, H., Zhu, Z., Luo, W., and
- Ma, Z. Comprehendedit: A comprehensive dataset and evaluation framework for multimodal knowledge editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 19323–19331, 2025a.

Ma, Z., Ma, Y., Zhu, Y., Yang, C., Chao, Y.-W., Xu, R., Chen, W., Chen, Y., Chen, Z., Cong, J., Li, K., Li, K., Li, S., Li, X., Li, X., Lian, Z., Liang, Y., Liu, M., Niu, Z., tianrui wang, Wang, Y., Wang, Y., Wu, Y., Yang, G., Yu, J., Yuan, R., Zheng, Z., Zhou, Z., Zhu, H., Xue, W., Benetos, E., Yu, K., Chng, E., and Chen, X. MMAR: A challenging benchmark for deep reasoning in speech, audio, music, and their mix. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025b. URL https:

//openreview.net/forum?id=fgmrBJemlQ.

Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 17359–17372. Curran Associates, Inc., 2022.

Meng, K., Sharma, A. S., Andonian, A. J., Belinkov, Y., and Bau, D. Mass-editing memory in a transformer. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=MkbcAHIYgyS.

Mitchell, E., Lin, C., Bosselut, A., Finn, C., and Manning, C. D. Fast model editing at scale. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=0DcZxeWfOPt.

Piczak, K. J. Esc: Dataset for environmental sound classification. In Proceedings of the 23rd ACM international conference on Multimedia, pp. 1015–1018, 2015.

Radford, A., Kim, J. W., Xu, T., Brockman, G., Mcleavey, C., and Sutskever, I. Robust speech recognition via largescale weak supervision. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 28492–28518. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/radford23a.html.

Sakshi, S., Tyagi, U., Kumar, S., Seth, A., Selvakumar, R., Nieto, O., Duraiswami, R., Ghosh, S., and Manocha, D. MMAU: A massive multi-task audio understanding and reasoning benchmark. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=TeVAZXr3yv.

S¸a¸smaz, E. and Tek, F. B. Animal sound classification using a convolutional neural network. In 2018 3rd International Conference on Computer Science and Engineering (UBMK), pp. 625–629. IEEE, 2018.

Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., Lu, L., MA, Z., and Zhang, C. SALMONN: Towards generic hearing abilities for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=14rn7HpKVk.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wang, D., Wu, J., Li, J., Yang, D., Chen, X., Zhang, T., and Meng, H. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779, 2025.

Wang, M., Zhang, N., Xu, Z., Xi, Z., Deng, S., Yao, Y., Zhang, Q., Yang, L., Wang, J., and Chen, H. Detoxifying large language models via knowledge editing. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3093–3118, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/

- 2024.acl-long.171. URL https://aclanthology. org/2024.acl-long.171/.

Wang, P., Li, Z., Zhang, N., Xu, Z., Yao, Y., Jiang, Y., Xie, P., Huang, F., and Chen, H. Wise: rethinking the knowledge memory for lifelong model editing of large language models. In Proceedings of the 38th International Conference on Neural Information Processing Systems, pp. 53764–53797, 2024b.

Wang, P., Zhang, N., Tian, B., Xi, Z., Yao, Y., Xu, Z., Wang, M., Mao, S., Wang, X., Cheng, S., et al. Easyedit: An easy-to-use knowledge editing framework for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 82–93, 2024c.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang,

- X., Wei, X., Ren, X., Liu, X., Fan, Y., Yao, Y., Zhang,
- Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Guo,
- Z., and Fan, Z. Qwen2 technical report, 2024a. URL https://arxiv.org/abs/2407.10671.

Yang, C.-K., Ho, N., Lee, Y.-J., and Lee, H.-y. Audiolens: A closer look at auditory attribute perception of large audiolanguage models. arXiv preprint arXiv:2506.05140, 2025a.

Yang, C.-K., Ho, N., Piao, Y.-T., and Lee, H.-y. SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information. In Interspeech 2025, pp. 1788–1792, 2025b. doi: 10.21437/ Interspeech.2025-839.

Yang, C.-K., Ho, N. S., and Lee, H.-y. Towards holistic evaluation of large audio-language models: A comprehensive survey. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 10144–10170, Suzhou, China, November 2025c. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 514. URL https://aclanthology.org/2025.

emnlp-main.514/.

Yang, H., Qu, L., Shareghi, E., and Haffari, G. Audio is the achilles’ heel: Red teaming audio large multimodal models. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 9292–9306, Albuquerque, New Mexico, April 2025d. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025. naacl-long.470. URL https://aclanthology.

org/2025.naacl-long.470/.

Yang, W., Sun, F., Ma, X., Liu, X., Yin, D., and Cheng, X. The butterfly effect of model editing: Few edits can

trigger large language models collapse. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 5419– 5437, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.322. URL https://aclanthology.

org/2024.findings-acl.322/.

Yao, Y., Wang, P., Tian, B., Cheng, S., Li, Z., Deng, S., Chen, H., and Zhang, N. Editing large language models: Problems, methods, and opportunities. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 10222–10240, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 632. URL https://aclanthology.org/2023.

emnlp-main.632/.

Zhang, J., Zhang, H., Yin, X., Huang, B., Zhang, X., Hu, X., and Wan, X. MC-MKE: A fine-grained multimodal knowledge editing benchmark emphasizing modality consistency. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 17430–17445, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 9798-89176-256-5. doi: 10.18653/v1/2025.findings-acl. 896. URL https://aclanthology.org/2025.

findings-acl.896/.

Zheng, C., Li, L., Dong, Q., Fan, Y., Wu, Z., Xu, J., and Chang, B. Can we edit factual knowledge by in-context learning? In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4862–4876, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 296. URL https://aclanthology.org/2023.

emnlp-main.296/.

Zhu, Y., He, Y., Ma, J., Hu, M., Li, S., and Li, J. Causal inference with latent variables: Recent advances and future prospectives. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 6677–6687, 2024.

### A. The Usage of Large Language Models (LLMs)

In this work, LLMs were used as judge models and as auxiliary tools for linguistic assistance, including polishing writing style, refining grammar, and proofreading. The conceptualization of the research problem, the design and construction of the benchmark, the execution of experiments, and the analysis and interpretation of results were carried out entirely by the authors without LLM involvement. All technical contributions and intellectual efforts originate from the authors, with LLMs serving only to support evaluation and to improve the clarity and readability of the manuscript.

### B. Dataset Details

##### B.1. Audio Sources

Audios are primarily sourced from the SAKURA benchmark (Yang et al., 2025b), which compiles data from CommonVoice (Ardila et al., 2020), CREMA-D (Cao et al., 2014), ESC-50 (Piczak, 2015), and Animal-Sound Dataset (¸Sa¸smaz & Tek, 2018). In addition, we supplement SAKURA with a subset of samples drawn from Dynamic-SUPERB Phase-2 (Huang et al., 2025) for type 4 audio locality data.

B.2. Dataset Statistics

We summarize the more details about the statistics of the training and testing sets in Tables 4 and 5, highlighting the data diversity.

Table 4. Dataset summary for each evaluation metric in our training dataset. (a) Question Length (word).

(b) Audio/speech duration (s).

Metric Avg. Min. Max. Std. Reliability 16.40 12 22 2.32

Avg. 16.40 12 22 2.32

- Type 1 16.40 12 22 2.32

- Type 2 16.40 12 22 2.32

- Type 3 16.40 12 22 2.32

Generality

Avg. 26.03 8 604 57.53

- Type 1 16.40 12 22 2.32

- Type 2 16.90 13 22 2.23

- Type 3 16.40 12 22 2.32

- Type 4 52.13 8 604 107.09

Audio Locality

Text Locality 55.32 7 526 48.91

Metric Avg. Min. Max. Std. Reliability 4.62 0.13 17.81 2.27

Avg. 4.62 0.13 17.81 2.27

- Type 1 4.62 0.13 17.81 2.27

- Type 2 4.62 0.13 17.81 2.27

- Type 3 4.62 0.13 17.81 2.27

Generality

Avg. 8.27 0.11 1478.35 31.23

- Type 1 4.62 0.13 17.81 2.27

- Type 2 4.28 0.13 17.81 2.22

- Type 3 4.62 0.13 17.81 2.27

- Type 4 18.56 0.11 1478.35 59.16

Audio Locality

C. Design Choices

##### C.1. Attributes

SAKE covers various auditory attributes, namely animal sound, speaker emotion, speaker gender, and spoken language, because they are fundamental to speech and audio understanding, widely used in real-world applications, and consistently evaluated in existing LALM benchmarks (Huang et al., 2025; Sakshi et al., 2025; Yang et al., 2025b;c). The attribute label sets are provided as follows:

- • Gender: Male, Female
- • Emotion: Happy, Disgust, Sad, Fear, Angry
- • Language: English, German, Spanish, French, Italian, Chinese, Japanese, Korean
- • Animal sound: Dog, Cat, Pig, Cow, Frog, Hen, Rooster, Sheep, Crow

Table 5. Dataset summary for each evaluation metric in our testing dataset. (a) Question Length (word).

(b) Audio/speech duration (s).

###### Metric Avg. Min. Max. Std. Reliability 24.23 7 48 10.24

Avg. 23.83 7 48 9.99

- Type 1 23.63 7 48 9.85

- Type 2 24.23 7 48 10.24

- Type 3 23.63 7 48 9.85

Generality

Avg. 31.51 7 604 55.81

- Type 1 23.93 7 48 9.91

- Type 2 25.35 12 48 9.61

- Type 3 23.94 7 48 9.92

- Type 4 51.29 9 604 104.30

Audio Locality

Text Locality 53.59 7 307 42.28 Portability 33.53 13 63 12.38

Metric Avg. Min. Max. Std. Reliability 4.75 0.24 20.78 2.43

Avg. 5.82 0.24 20.78 4.98

- Type 1 4.75 0.24 20.78 2.43

- Type 2 6.35 0.24 20.78 5.79

- Type 3 6.35 0.24 20.78 5.79

Generality

Avg. 8.12 0.17 759.53 26.98

- Type 1 4.78 0.28 20.78 2.46

- Type 2 4.41 0.24 20.78 2.38

- Type 3 4.73 0.24 20.78 2.50

- Type 4 17.63 0.17 759.53 50.89

Audio Locality

Portability 4.75 0.24 20.78 2.43

As the first study on auditory knowledge editing, focusing on these core attributes provides a clear and representative starting point. Moreover, these attributes are supported by comparatively rich resources, including curated audio datasets and question–answer pairs, which are essential for constructing reliable evaluations, particularly for our portability track that involves multi-hop reasoning (Yang et al., 2025b). This selection establishes a focused and principled foundation for the first benchmark in this area.

##### C.2. Models

We select Qwen2-Audio (Chu et al., 2024), DeSTA2.5-Audio (Lu et al., 2025b), and Audio Flamingo 3 (Ghosh et al., 2025a) as strong, widely recognized, and representative open-source LALMs. Qwen2-Audio is widely adopted as a community baseline in studies on reasoning (Ma et al., 2025b; Sakshi et al., 2025; Yang et al., 2025b), safety (Feng et al., 2025; Yang et al., 2025d), and interpretability (Yang et al., 2025a). DeSTA2.5-Audio and Audio Flamingo 3 are more recent models that achieve competitive performance across multiple benchmarks (Huang et al., 2025; Ma et al., 2025b; Sakshi et al., 2025; Wang et al., 2025), reflecting recent advances in LALMs.

Importantly, prior work (Yang et al., 2025b) shows that these models already possess the world knowledge required by our portability dataset, whose connected attribute knowledge is drawn from the SAKURA benchmark. Moreover, they demonstrate state-of-the-art multi-hop reasoning that integrates auditory understanding with world knowledge. This allows observed portability errors to be attributed to failures in edit propagation, rather than limitations in world knowledge or inherent reasoning capability.

As our goal is to evaluate the editing methods rather than to compare LALMs themselves, we follow standard practice in knowledge editing research by applying each method to two to three representative models (Chen et al., 2025; Cheng et al., 2023; Deng et al., 2025; Du. et al., 2025; Ma et al., 2025a; Meng et al., 2022; Wang et al., 2024b; Zhang et al., 2025). Importantly, the selected models belong to different families and differ substantially in their audio encoders, LLM backbones, modality adaptation mechanisms, and training strategies, thereby covering a diverse range of LALM designs. The fact that our key observations consistently emerge across these models and across a wide range of editing methods suggests that the identified challenges stem from limitations of current editing techniques, rather than from idiosyncrasies of any particular model. This supports the suitability of our model selection as a testbed for evaluating auditory knowledge editing.

#### D. Implementation Details of the Editing Methods Our implementation of editing methods is built on the EasyEdit toolkit (Wang et al., 2024c).

- Table 6. Hyper-parameters of each editing method. I-IKE and IE-IKE are excluded because they do not modify model parameters.

FT (LLM) Model Max Steps Edit Layer Optimizer Edit LR

DeSTA 15 Layer 31 of Transformer Module Adam 1e-5 Qwen 15 Layer 31 of Transformer Module Adam 1e-4 AF 15 Layer 27 of Transformer Module Adam 2e-5

FT (Audio) Model Max Steps Edit Layer Optimizer Edit LR

DeSTA 15 perception.connector Adam 1e-4 Qwen 15 multi modal projector Adam 1e-4 AF 15 multi modal projector Adam 5e-4

KE Model Epoch Edit Layer Optimizer LR

DeSTA 1 layer 29, 30, 31 of Transformer Module RMSprop 3e-4 Qwen 1 layer 29, 30, 31 of Transformer Module RMSprop 3e-4 AF 1 layer 25, 26, 27 of Transformer Module RMSprop 3e-4

MEND Model Epoch Edit Layer Optimizer LR

DeSTA 1 layer 29, 30, 31 of Transformer Module Adam 1e-6 Qwen 1 layer 29, 30, 31 of Transformer Module Adam 1e-6 AF 1 layer 25, 26, 27 of Transformer Module Adam 1e-6

UnKE Model v step/optim step Edit Layer preserve data v LR/optim LR

DeSTA 25/50 layer 15 of Transformer Module 20(text)/5(audio) 5e-1/2e-4 Qwen 25/50 layer 20 of Transformer Module 20(text)/5(audio) 5e-1/2e-4 AF 25/50 layer 20 of Transformer Module 20(text)/5(audio) 5e-1/2e-4

WISE Model α, β, γ, act ratio Edit Layer preserve data edit LR/n iter

DeSTA 2,20,10,0.88 layer 29 of Transformer Module 10(text)/10(audio) 0.1/50 Qwen 5,20,10,0.88 layer 26 of Transformer Module 10(text)/10(audio) 1.0/50 AF 5,20,10,0.00 layer 26 of Transformer Module 10(text)/10(audio) 1e-2/100

##### D.1. Editing Methods

Fine-tuning adapts the model via gradient descent on selected components. We apply it to the last layer of the LLM backbone (FT (LLM)) and the modality connector between audio encoders and the backbone in LALMs (FT (Audio)).

Knowledge Editor (KE) (De Cao et al., 2021) employs a hypernetwork to update parameters. It leverages a bidirectional LSTM with constrained optimization, in which generality and locality data is involved, to predict weight updates.

MEND (Mitchell et al., 2022) uses a hypernetwork to generate parameter updates by decomposing fine-tuning gradients into low-rank forms and transforming them into parameter updates. Similarly, it also leverages generality and locality data during training.

UnKE (Deng et al., 2025) is an unstructured knowledge editing method. UnKE first finds a modified key vector by adding a small residual (delta) to the hidden state of a chosen layer so that the model’s output shifts to the desired target. In the second stage, the parameters of the chosen layer are updated to make the chosen layer naturally produce this new key vector.

- Table 7. Approximate execution time of each editing method on an NVIDIA H100 GPU, measured for training the trainer, single editing, and sequential editing.

Execution Time Training Trainer Single Editing Sequential Editing

Method Model

DeSTA - 5h 50m 2h 00m Qwen - 2h 35m 1h 15m AF - 2h 00m 0h 40m

FT (LLM)

DeSTA - 4h 20m 2h 00m Qwen - 3h 00m 1h 30m AF - 1h 30m 0h 40m

FT (Audio)

DeSTA 4h 10m 8h 40m 9h 45m Qwen 2h 10m 4h 50m 3h 20m AF 2h 10m 3h 00m 2h 15m

KE

DeSTA 4h 10m 4h 00m 4h 10m Qwen 2h 20m 2h 10m 1h 50m AF 2h 20m 2h 00m 0h 25m

MEND

DeSTA - 3h 00m 1h 05m Qwen - 1h 35m 0h 30m AF - 1h 45m 0h 25m

UnKE

DeSTA - 6h 00m 3h 45m Qwen - 3h 00m 2h 00m AF - 1h 20m 0h 45m

I-IKE

DeSTA - 3h 20m 8h 00m Qwen - 2h 00m 3h 50m AF - 2h 10m 4h 00m

IE-IKE

DeSTA - 6h 10m 2h 00m Qwen - 3h 00m 1h 10m AF - 7h 00m 1h 45m

WISE

In-Context Knowledge Editing (IKE) (Zheng et al., 2023) uses in-context learning (ICL) to modify model knowledge without parameter updates, relying on instructions and demonstrations to enforce the edited knowledge. We evaluate two variants: Instruction-based IKE (I-IKE) and Instruction+Example IKE (IE-IKE).

WISE (Wang et al., 2024b) performs knowledge editing by introducing a dual-memory architecture that separates the pretrained FFN weights from a lightweight side-memory used to store edit-specific updates. During editing, WISE optimizes a routing-aware activation loss that forces edited queries to rely on the side-memory while keeping irrelevant inputs mapped to the original parameters, ensuring strong locality without degrading pretrained behavior.

##### D.2. Implementation Details

The hyperparameters of each editing method are shown in Table 6, and their approximate execution time on NVIDIA H100 GPU is reported in Table 7. The execution time varies across models due to differences in their response patterns. DeSTA generally produces detailed and descriptive explanations, whereas AF typically generates concise answers without additional explanation. Qwen occasionally provides explanations, but this behavior is not consistent.

For FT (LLM) and FT (Audio), early stopping is applied when the loss falls below 1e-2. For KE and MEND, which require training a hypernetwork, we train the hypernetwork for one epoch due to its rapid convergence on the validation set. For UnKE, the editing process consists of two stages with separate hyperparameters. The first stage uses v step iterations

to find the desired key vector, with its own learning rate v LR. The second stage uses optim step iterations to update the weights of the chosen layer so that it naturally produces the new key vector, controlled by a separate learning rate optim LR. To prevent interference with unrelated knowledge, stage 2 also incorporates preservation data. Specifically, we use 20 preservation samples for text and 5 for audio. Preservation data constrains the key generator to retain original key representations while inserting new ones, preventing global parameter drift.

For WISE, the edit copy is optimized with an edit target loss and an activation-distance margin loss, where α constrains out-of-scope drift, β enforces sufficient in-scope change, and γ separates in-scope and out-of-scope activations. At inference time, an activation-based gating rule using stored edit statistics and act ratio selects between original and edited weights, ensuring edits are triggered only for in-scope inputs via preservation data.

For I-IKE and IE-IKE, the edited knowledge is encoded via natural language instructions in the system prompt, following the prompt template provided below. In IE-IKE, demonstration examples are retrieved from the training set based on the editing pair. For example, when editing the model’s perception of speaker emotion from happy to sad, we retrieve an editing instance from the training set involving the same (happy, sad) editing pair. The demonstration examples illustrate how the model should respond to different types of evaluation queries, covering reliability, generality, and locality. Specifically, for each test sample, we provide 1 reliability example, 3 generality examples, 4 audio locality examples (or 3 when editing the speaker gender attribute), and 1 text locality example, all presented to the model as a multi-turn dialogue context.

##### System Prompt Template for I-IKE and IE-IKE

You are performing in-context knowledge editing, but only for questions about audio inputs. From now on, when analyzing any audio clip, you must apply the following edits consistently: EDITS (New facts): {edits} where each entry has the form ‘pre edit‘ → ‘post edit‘. Rules:

- 1. If your reasoning or prediction about an audio clip would normally lead to ‘pre edit‘, you must instead treat it as ‘post edit‘.

- 2. All properties, attributes, and facts that belong to ‘post edit‘ must be applied consistently, as if the audio were actually from ‘post edit‘.

- 3. If the user’s question is unrelated to these edits, you should answer normally without making changes.
- 4. Always ensure your final answers are fully consistent with the edited mapping.

In WISE, α bounds locality activation, with smaller values enforcing stricter preservation. β sets a minimum edit activation, where larger values drive more aggressive edits. γ enforces a margin ensuring edit activations exceed locality activations. Finally, act ratio scales the inference routing threshold: higher values favor original weights, while lower values route more queries to the side-memory.

### E. More Details and Evaluation Prompts for LLM-as-a-judge

We use GPT-5 mini (gpt-5-mini-2025-08-07) with minimal reasoning effort and a fixed random seed (0) as the judge in our LLM evaluator, balancing strong alignment with human judgments and cost efficiency. During evaluation, we focus on correctness and consistency. For reliability, generality, and portability, we assess whether the edited model’s responses correctly align with the ground truth. For locality, we assess whether the edited model remains consistent with the original model’s outputs. The evaluation prompts are provided below. To further reduce costs, some cases are judged directly without invoking the LLM evaluator: empty outputs are marked as incorrect/inconsistent, and responses that exactly match the ground truth (for correctness) or the original model’s output (for consistency) are marked as correct/consistent.

Human verification was conducted on 210 randomly selected samples for correctness and consistency, respectively, yielding 98.57% and 97.62% agreement, indicating the robustness of the employed LLM-as-a-judge.

##### LLM Judge Prompt for Correctness

Task. You will be given a question, a list of possible options, a ground-truth answer, and a model-generated response. Determine whether the model-generated response is correct according to the criteria below.

###### Criteria.

- 1. There is one and only one correct answer. Judge incorrect if the response does not choose any option from the list, or if it chooses more than one option.
- 2. If exactly one option from the list is chosen, judge correct if the chosen option aligns with the ground-truth answer; otherwise, judge incorrect.
- 3. Read the question, options, ground-truth answer, and model-generated response carefully before deciding.

###### Examples.

- Example 1. Question: What is the capital of France? Options: Paris; London; Berlin; Madrid. Ground truth answer: Paris

Response: “The capital of France is Tokyo.” ⇒ incorrect (does not choose any listed option). Response: “The capital of France is Paris and London.” ⇒ incorrect (chooses more than one option). Response: “The capital of France is London.” ⇒ incorrect (chosen option does not match ground truth). Response: “The capital of France is Paris.” ⇒ correct.

- Example 2. Question: What is the underlying emotion of the speaker? Options: Happy; Sad; Angry; Neutral. Ground truth answer: Happy

Response: “The speaker is happy.” ⇒ correct. Response: “The speaker expresses happiness.” ⇒ correct (“happiness” is a valid morphological variant of “happy”). Response: “Happiness.” ⇒ correct.

Now judge: Question: {question} Ground truth answer: {ground_truth} Model-generated response: {model_response}

Return format. Explanation: <Your explanation on your judgement> Judgement: <correct or incorrect>

##### LLM Judge Prompt for Consistency

Role. You are an expert evaluator tasked with determining whether two model outputs are semantically consistent for the same input.

Inputs. Question: {question} Original Output: {original_output} New Output: {new_output}

Task. Decide if these two outputs are semantically consistent.

- • Outputs are consistent if they express the same main message, even with different wording, phrasing, or level of detail.
- • Outputs are inconsistent if they contradict each other, reach different conclusions, or provide fundamentally different answers.

Return format. Explanation: <Your explanation on your judgement> Judgement: <consistent or inconsistent>

### F. Detailed Results for Each Auditory Attribute under Single Editing

We compare the performance for each auditory attribute (animal sound, speaker emotion, speaker gender, and spoken language) under single editing here, as shown in Table 8, Table 9, and Table 10 for DeSTA, Qwen, and AF, respectively.

Reliability remains high across all attributes. All methods except I-IKE and IE-IKE exhibit consistently high reliability across all attributes, suggesting that they can effectively update the edited knowledge on these LALMs. In contrast, the performance of I-IKE and IE-IKE varies by attributes and models, which we attribute to differences in the models’ in-context learning ability.

Editing the modality connector limits emotional generalization. For most methods, the generality remains relatively stable across different auditory attributes. However, FT (Audio) exhibits a notable exception: while it consistently achieves high reliability, its generality on emotion attributes degrades the most compared to its reliability on DeSTA (99.67% → 78.56%) and Qwen (100.00% → 71.67%), and the second most on AF (100.00% → 63.00%), among all attributes. This suggests that editing the modality connector makes it harder for the model to extend edited knowledge to similar inputs within emotion.

Audio locality varies by attributes and models. Most methods achieve relatively higher average performance on the gender attribute, owing to the inapplicability of type 2 audio locality, which was identified in Sec. 5.2 as the most difficult to preserve, thereby inflating the overall average. For attributes other than gender, most methods perform comparably across attributes on Qwen. In contrast, on DeSTA, higher audio locality is observed for language in types 2 and 3, while on AF, lower audio locality is observed for emotion in type 2 across most methods. These results indicate attribute-dependent differences in preservation difficulty across LALMs. Regarding text locality, most methods demonstrate comparable performance across attributes on these LALMs.

FT (Audio) achieves consistently high portability across attributes. Different methods result in varying portability performance across attributes on these LALMs. A consistent observation on these models is that FT (Audio) achieves higher portability scores for all attributes, suggesting that editing the modality connector may more effectively propagate the edited knowledge to other interconnected knowledge.

- Table 8. Detailed results of the four metrics of each auditory attribute across different editing methods on DeSTA under single editing. Attr. denotes auditory attributes, and Port. denotes portability. For generality and audio locality, Avg. indicates the average performance across all types of the corresponding metric. (%)

Generality Audio Locality Text

Method Attr. Reliability

Port.

Locality

Avg. Type 1 Type 2 Type 3 Avg. Type 1 Type 2 Type 3 Type 4

ALL 99.75 98.75 99.67 99.00 97.58 64.93 87.92 15.78 74.67 69.08 82.67 18.83 Animal 99.33 99.11 99.67 99.33 98.33 55.08 91.00 9.33 56.33 63.67 83.33 20.00

FT (LLM)

Emotion 100.00 99.56 99.33 100.00 99.33 53.17 80.67 7.33 59.67 65.00 77.67 22.00 Gender 99.67 96.78 99.67 96.67 94.00 84.44 90.33 - 88.33 74.67 88.33 6.00 Language 100.00 99.56 100.00 100.00 98.67 71.92 89.67 30.67 94.33 73.00 81.33 27.33

ALL 99.50 86.06 96.75 84.83 76.58 68.13 78.58 48.11 76.00 64.83 100.00 55.33 Animal 99.00 85.67 97.33 83.00 76.67 62.08 81.67 40.67 60.00 66.00 100.00 29.33

FT (Audio)

Emotion 99.67 78.56 95.33 73.67 66.67 60.75 77.67 38.67 66.00 60.67 100.00 46.00 Gender 99.67 92.22 99.33 92.33 85.00 77.78 78.33 - 87.33 67.67 100.00 82.67

- Language 99.67 87.78 95.00 90.33 78.00 74.33 76.67 65.00 90.67 65.00 100.00 63.33

KE

ALL 99.58 99.17 99.25 99.25 99.00 79.49 96.42 43.89 76.33 92.42 93.08 17.33 Animal 98.67 97.33 97.67 97.33 97.00 77.42 97.00 61.67 59.00 92.00 93.00 23.33

Emotion 100.00 100.00 100.00 100.00 100.00 66.25 98.33 16.33 59.33 91.00 93.67 16.00 Gender 99.67 99.56 99.33 99.67 99.67 94.22 96.67 - 92.33 93.67 93.33 7.67

- Language 100.00 99.78 100.00 100.00 99.33 83.75 93.67 53.67 94.67 93.00 92.33 22.33

ALL 95.25 94.94 95.83 95.08 93.92 71.44 93.50 17.22 74.58 86.92 92.00 18.08 Animal 90.00 89.22 91.33 88.67 87.67 63.50 93.67 19.33 57.00 84.00 92.33 20.33

MEND

Emotion 99.67 98.44 99.00 99.00 97.33 62.17 94.67 9.00 59.33 85.67 93.33 23.67 Gender 98.67 97.00 97.33 99.33 94.33 91.78 97.00 - 88.33 90.00 90.33 4.67 Language 92.67 95.11 95.67 93.33 96.33 73.42 88.67 23.33 93.67 88.00 92.00 23.67

ALL 96.09 87.42 87.92 92.50 81.84 57.18 67.59 10.78 72.25 62.83 88.50 17.17 Animal 99.67 96.11 98.00 99.67 90.67 52.75 69.67 16.33 60.00 65.00 88.33 27.33

UnKE

Emotion 95.00 87.67 87.33 90.67 85.00 50.50 66.67 12.00 57.33 66.00 87.67 18.33 Gender 99.67 86.11 86.00 99.67 72.67 71.89 58.33 - 94.33 63.00 88.00 6.00 Language 90.00 79.78 80.33 80.00 79.00 53.58 75.67 4.00 77.33 57.33 90.00 17.00

ALL 64.25 61.08 64.67 58.83 59.75 65.49 79.33 50.67 73.67 54.58 61.92 71.33 Animal 90.33 92.67 92.67 92.67 92.67 54.92 83.67 22.00 59.67 54.33 60.67 78.33

I-IKE

Emotion 59.00 49.78 58.00 44.67 46.67 59.92 80.67 45.33 62.00 51.67 59.67 65.33

Gender 69.00 64.11 71.00 60.33 61.00 73.33 81.33 - 79.67 59.00 63.67 72.67 Language 38.67 37.78 37.00 37.67 38.67 75.75 71.67 84.67 93.33 53.33 63.67 69.00

ALL 40.58 39.61 41.50 38.67 38.67 58.89 70.00 50.00 70.75 42.58 56.67 34.33 Animal 77.00 78.56 80.00 76.67 79.00 51.33 78.67 23.33 59.00 44.33 60.67 44.00

IE-IKE

Emotion 27.00 26.67 28.67 25.33 26.00 51.67 68.00 33.67 61.33 43.67 52.33 28.67

Gender 41.67 37.67 43.00 36.00 34.00 61.56 74.33 - 67.67 42.67 60.67 32.67 Language 16.67 15.56 14.33 16.67 15.67 71.67 59.00 93.00 95.00 39.67 53.00 32.00

ALL 100.00 100.00 100.00 100.00 100.00 38.90 29.52 3.78 76.00 31.38 83.08 17.34 Animal 100.00 100.00 100.00 100.00 100.00 33.94 39.33 5.00 58.00 33.44 83.33 11.33

WISE

Emotion 100.00 100.00 100.00 100.00 100.00 24.17 16.33 6.33 56.33 17.67 76.00 2.68

Gender 100.00 100.00 100.00 100.00 100.00 63.44 43.67 - 93.67 53.00 90.00 29.67 Language 100.00 100.00 100.00 100.00 100.00 34.06 18.73 0.00 96.00 21.40 83.00 25.67

- Table 9. Detailed results of the four metrics of each auditory attribute across different editing methods on Qwen under single editing. Attr. denotes auditory attributes, and Port. denotes portability. For generality and audio locality, Avg. indicates the average performance across all types of the corresponding metric. (%)

Generality Audio Locality Text

Method Attr. Reliability

Port.

Locality

Avg. Type 1 Type 2 Type 3 Avg. Type 1 Type 2 Type 3 Type 4

ALL 100.00 99.94 100.00 100.00 99.83 67.40 91.50 10.33 83.33 70.17 75.58 24.00 Animal 100.00 99.89 100.00 100.00 99.67 64.17 93.00 8.33 86.33 69.00 70.33 22.33

FT (LLM)

Emotion 100.00 100.00 100.00 100.00 100.00 58.83 93.33 8.00 66.00 68.00 80.33 19.00

Gender 100.00 100.00 100.00 100.00 100.00 83.67 85.33 - 92.33 73.33 79.67 26.00 Language 100.00 99.89 100.00 100.00 99.67 67.00 94.33 14.67 88.67 70.33 72.00 28.67

ALL 100.00 81.81 99.00 77.00 69.42 90.49 96.75 80.00 90.00 92.58 100.00 49.92 Animal 100.00 84.22 99.00 80.33 73.33 94.17 98.00 93.67 93.33 91.67 100.00 48.33

FT (Audio)

Emotion 100.00 71.67 99.00 62.67 53.33 85.00 98.33 67.00 84.33 90.33 100.00 42.33

Gender 100.00 97.44 100.00 96.00 96.33 94.67 98.00 - 92.00 94.00 100.00 62.33 Language 100.00 73.89 98.00 69.00 54.67 89.17 92.67 79.33 90.33 94.33 100.00 46.67

ALL 95.33 86.69 92.00 87.92 80.17 83.36 89.83 61.44 87.25 89.42 85.67 26.75 Animal 97.00 88.11 89.33 94.00 81.00 88.33 93.67 76.33 92.33 91.00 82.67 23.00

KE

Emotion 98.33 76.67 91.33 74.33 64.33 77.92 97.33 52.67 72.00 89.67 86.67 22.67 Gender 87.00 89.44 90.33 89.33 88.67 84.44 72.67 - 93.67 87.00 85.67 38.33

- Language 99.00 92.56 97.00 94.00 86.67 83.00 95.67 55.33 91.00 90.00 87.67 23.00

MEND

ALL 100.00 95.31 98.92 95.92 91.08 83.29 98.58 49.44 85.42 91.25 87.25 26.25 Animal 100.00 97.22 98.00 98.33 95.33 84.50 97.00 64.33 89.67 87.00 84.67 21.33

Emotion 100.00 89.89 99.00 92.00 78.67 75.67 100.00 42.00 70.33 90.33 89.67 23.33 Gender 100.00 95.89 100.00 94.33 93.33 94.89 99.00 - 92.33 93.33 89.33 36.00

- Language 100.00 98.22 98.67 99.00 97.00 81.00 98.33 42.00 89.33 94.33 85.33 24.33

ALL 98.67 98.64 99.42 98.75 97.75 68.00 92.08 12.22 83.00 67.17 72.84 26.58 Animal 97.00 97.89 98.67 95.67 99.33 63.42 91.00 9.33 87.00 66.33 72.67 20.33

UnKE

Emotion 98.33 97.33 99.00 99.67 93.33 62.00 95.33 18.67 64.67 69.33 71.00 16.33

Gender 100.00 99.56 100.00 99.67 99.00 82.33 90.00 - 91.67 65.33 75.67 38.00 Language 99.33 99.78 100.00 100.00 99.33 64.25 92.00 8.67 88.67 67.67 72.00 31.67

ALL 10.33 7.00 10.25 5.58 5.17 87.47 94.75 89.56 93.00 73.08 55.67 28.50 Animal 7.00 6.89 6.67 7.33 6.67 89.75 96.33 94.33 94.33 74.00 57.33 24.00

I-IKE

Emotion 19.33 13.11 19.67 9.00 10.67 82.75 97.00 78.00 83.33 72.67 58.00 22.00

Gender 9.67 3.00 8.67 0.33 0.00 88.67 94.33 - 99.00 72.67 57.33 45.00 Language 5.33 5.00 6.00 5.67 3.33 89.00 91.33 96.33 95.33 73.00 50.00 23.00

ALL 7.83 6.44 8.33 5.67 5.33 82.82 91.25 86.67 89.67 64.67 51.17 27.58 Animal 6.33 6.22 6.67 6.00 6.00 84.17 90.33 91.00 88.33 67.00 52.67 26.33

IE-IKE

Emotion 9.00 11.56 11.67 10.67 12.33 79.50 96.67 76.33 80.33 64.67 54.33 21.67

Gender 11.00 3.78 10.67 0.67 0.00 83.44 89.00 - 99.00 62.33 52.00 40.00 Language 5.00 4.22 4.33 5.33 3.00 84.33 89.00 92.67 91.00 64.67 45.67 22.33

ALL 100.00 99.34 99.92 99.17 98.92 70.04 92.67 6.22 82.84 77.75 82.09 27.84 Animal 100.00 99.67 99.67 99.67 99.67 64.00 90.67 3.67 86.00 75.67 79.67 6.67 Emotion 100.00 98.78 100.00 98.67 97.67 60.58 93.33 9.67 65.00 74.33 82.00 21.67

WISE

Gender 100.00 98.89 100.00 98.33 98.33 88.89 94.00 - 91.67 81.00 83.67 41.00 Language 100.00 100.00 100.00 100.00 100.00 66.67 92.67 5.33 88.67 80.00 83.00 42.00

- Table 10. Detailed results of the four metrics of each auditory attribute across different editing methods on AF under single editing. Attr. denotes auditory attributes, and Port. denotes portability. For generality and audio locality, Avg. indicates the average performance across all types of the corresponding metric. (%)

Generality Audio Locality Text

Method Attr. Reliability

Port.

Locality

Avg. Type 1 Type 2 Type 3 Avg. Type 1 Type 2 Type 3 Type 4

ALL 100.00 93.31 96.67 94.17 89.08 87.67 98.00 69.89 85.42 92.92 97.00 31.67 Animal 100.00 91.67 92.33 97.33 85.33 89.83 98.33 79.33 89.67 92.00 95.67 29.00

FT (LLM)

Emotion 100.00 88.89 98.33 85.00 83.33 78.08 98.00 53.67 69.67 91.00 98.00 17.33

Gender 100.00 98.89 100.00 98.33 98.33 93.89 97.33 - 88.67 95.67 98.33 53.00 Language 100.00 93.78 96.00 96.00 89.33 90.42 98.33 76.67 93.67 93.00 96.00 27.33

ALL 100.00 76.08 96.67 74.00 57.58 90.24 96.92 78.56 89.33 93.25 100.00 49.50 Animal 100.00 59.33 91.00 55.33 31.67 96.08 98.67 94.67 97.33 93.67 100.00 46.33

FT (Audio)

Emotion 100.00 63.00 99.33 49.33 40.33 89.25 98.67 82.33 83.00 93.00 100.00 38.00

Gender 100.00 91.44 99.67 96.67 78.00 92.89 96.67 - 87.67 94.33 100.00 65.67 Language 100.00 90.56 96.67 94.67 80.33 83.42 93.67 58.67 89.33 92.00 100.00 48.00

ALL 100.00 98.89 99.92 98.75 98.00 81.38 91.50 55.78 78.50 93.33 95.83 31.92 Animal 100.00 99.22 99.67 99.67 98.33 85.92 92.00 70.00 90.00 91.67 93.33 28.33

KE

Emotion 100.00 97.11 100.00 96.33 95.00 71.75 90.00 36.00 68.00 93.00 97.00 19.00

Gender 100.00 100.00 100.00 100.00 100.00 84.33 95.67 - 62.67 94.67 97.33 52.33 Language 100.00 99.22 100.00 99.00 98.67 84.25 88.33 61.33 93.33 94.00 95.67 28.00

ALL 99.92 93.92 99.08 92.42 90.25 90.64 99.25 79.22 85.58 95.67 98.67 32.08 Animal 100.00 97.67 98.33 99.33 95.33 91.58 99.67 81.33 90.00 95.33 98.67 30.67

MEND

Emotion 100.00 87.44 99.00 82.00 81.33 84.42 99.00 70.67 72.67 95.33 99.00 17.00 Gender 100.00 93.56 100.00 90.33 90.33 93.44 99.00 - 85.67 95.67 99.33 53.67

- Language 99.67 97.00 99.00 98.00 94.00 93.83 99.33 85.67 94.00 96.33 97.67 27.00

UnKE

ALL 99.92 91.17 98.09 99.25 94.17 76.03 93.33 31.22 83.33 82.22 81.17 31.75 Animal 99.67 95.67 94.67 100.00 92.33 75.73 91.33 41.00 88.67 81.94 77.67 27.67

Emotion 100.00 95.33 99.67 97.00 89.33 67.00 93.67 27.67 67.33 79.33 80.67 17.33 Gender 100.00 99.67 100.00 100.00 99.00 87.21 94.00 - 84.33 83.28 85.67 54.00

- Language 100.00 98.00 98.00 100.00 96.00 74.17 94.33 25.00 93.00 84.33 80.67 28.00

ALL 25.00 15.08 25.17 9.50 10.58 87.07 95.83 88.00 89.17 75.50 81.17 37.08 Animal 12.67 12.11 11.33 10.67 14.33 89.25 94.67 89.00 93.33 80.00 81.67 37.67

I-IKE

Emotion 24.00 9.11 24.67 1.33 1.33 83.25 97.00 83.33 80.00 72.67 80.67 22.00

Gender 55.33 32.67 57.33 20.00 20.67 86.56 96.00 - 89.00 74.67 81.00 59.67 Language 8.00 6.44 7.33 6.00 6.00 89.08 95.67 91.67 94.33 74.67 81.33 29.00

ALL 58.83 58.36 58.25 58.25 58.58 40.42 34.50 19.11 60.92 41.83 76.67 40.67 Animal 38.33 41.11 42.00 38.33 43.00 39.67 37.33 20.00 53.33 48.00 78.00 41.00

IE-IKE

Emotion 51.33 49.33 50.00 49.33 48.67 35.58 34.33 21.33 48.00 38.67 77.67 25.67

Gender 99.33 98.56 99.33 98.33 98.00 51.78 29.33 - 84.33 41.67 75.67 61.33 Language 46.33 44.44 41.67 47.00 44.67 37.50 37.00 16.00 58.00 39.00 75.33 34.67

ALL 100.00 77.69 85.75 83.67 63.67 91.18 97.92 84.78 87.75 92.07 96.08 30.75 Animal 100.00 83.00 80.00 95.67 73.33 92.99 97.67 89.67 90.67 93.98 97.00 30.00

WISE

Emotion 100.00 79.33 96.00 74.00 68.00 84.17 99.00 74.67 73.33 89.67 96.00 18.00

Gender 100.00 76.00 97.33 74.33 56.33 93.55 97.00 - 91.33 92.31 97.00 51.00 Language 100.00 72.44 69.67 90.67 57.00 94.00 98.00 90.00 95.67 92.33 94.33 24.00

### G. Detailed Results under Sequential Editing

We provide the statistics of the sequential editing results in Table 11, Table 12, and Table 13, with the corresponding line charts shown in Figure 4a, Figure 4b, and Figure 4c for DeSTA, Qwen, and AF, respectively.

### H. More Discussions of Intra-attribute Knowledge Entanglement

In Sec. 5.2, we observe that Audio Locality Type 2 is the most difficult to preserve. We hypothesize that this failure is related to the fact that intra-attribute distinctions are often more tightly coupled than inter-attribute ones, making them more vulnerable to interference during editing. In particular, both acoustic and semantic factors may jointly contribute to this phenomenon.

Acoustic Entanglement Compared to inter-attribute distinctions (e.g., emotions versus animal sounds), labels within the same attribute often share overlapping acoustic characteristics. For example, within the emotion attribute, “sad” and “fearful” speech may both exhibit lower energy, increased tension, or slower speaking rates. Such shared prosodic and spectral patterns suggest that modifying knowledge associated with one label may unintentionally affect closely related labels, making intra-attribute knowledge harder to preserve during editing.

Semantic Entanglement As shown in Figure 3, FT (LLM) performs substantially worse than FT (Audio) on Audio Locality Type 2. This gap suggests that intra-attribute distinctions are more susceptible to interference when updates are applied to the LLM backbone. A plausible explanation is that these distinctions may also rely on fine-grained semantic cues that are closely distributed within the models. Directly modifying LLM parameters may therefore increase the risk of conflating related labels.

In summary, current editing methods do not explicitly model the fine-grained structure of intra-attribute knowledge. Our results suggest that both acoustic overlap and semantic coupling may contribute to the observed failures in Audio Locality Type 2. Notably, identifying intra-attribute knowledge entanglement as a key challenge is itself an important insight enabled by our comprehensive benchmark design. Developing editing approaches that can better isolate and preserve intra-attribute distinctions remains an important direction for future work.

- Table 11. Original result of the four metrics of different editing methods on DeSTA under sequential editing. For generality and audio locality, we present the averaged results. (%)

Method Gap Reliability Generality Audio Locality Text Locality Portability

- 0 100.00 99.33 49.22 78.00 6.00

- 1 78.00 75.33 48.19 72.00 6.00

- 2 68.00 68.00 45.60 74.00 10.00

- 3 64.00 57.33 45.08 72.00 8.00

- 4 58.00 52.67 41.97 74.00 10.00

- 5 54.00 48.00 45.60 70.00 8.00

FT (LLM)

- 0 98.00 87.33 53.37 100.00 50.00

- 1 78.00 62.00 50.26 100.00 38.00

- 2 64.00 46.00 40.93 100.00 30.00

- 3 62.00 48.00 41.45 100.00 24.00

- 4 52.00 38.00 39.38 100.00 42.00

- 5 46.00 35.33 40.41 100.00 32.00

FT (Audio)

- 0 18.00 18.00 30.57 56.00 2.00

- 1 12.00 8.00 14.51 34.00 4.00

- 2 4.00 2.67 3.11 20.00 0.00

- 3 2.00 2.00 2.07 6.00 0.00

- 4 2.00 2.00 0.52 2.00 0.00

- 5 0.00 0.00 0.00 2.00 0.00

MEND

- 0 58.00 64.00 41.97 74.00 18.00

- 1 36.00 33.33 33.68 60.00 16.00

- 2 12.00 14.67 29.53 56.00 4.00

- 3 8.00 12.67 24.35 40.00 8.00

- 4 26.00 14.00 21.24 30.00 6.00

- 5 8.00 9.33 14.51 38.00 4.00

KE

- 0 88.00 87.33 34.20 74.00 36.00

- 1 46.00 40.67 34.20 74.00 14.00

- 2 32.00 27.33 31.08 78.00 12.00

- 3 30.00 20.67 34.20 82.00 16.00

- 4 32.00 22.00 35.75 82.00 14.00

- 5 20.00 12.00 34.20 82.00 12.00

UnKE

- 0 32.00 30.00 59.07 54.00 28.00

- 1 26.00 36.67 59.07 58.00 34.00

- 2 30.00 35.33 58.03 50.00 32.00

- 3 32.00 34.00 54.92 58.00 34.00

- 4 32.00 39.33 52.85 58.00 38.00

- 5 26.00 32.00 52.85 56.00 34.00

IE-IKE

- 0 58.00 54.67 65.80 52.00 58.00

- 1 56.00 50.67 60.62 54.00 52.00

- 2 56.00 54.67 61.14 66.00 54.00

- 3 56.00 52.00 62.18 54.00 46.00

- 4 62.00 56.67 60.10 54.00 46.00

- 5 62.00 56.67 59.07 56.00 52.00

I-IKE

- 0 100.00 100.00 41.45 88.00 20.00

- 1 78.00 75.33 40.41 94.00 16.00

- 2 76.00 68.00 37.82 98.00 18.00

- 3 74.00 66.00 38.34 96.00 14.00

- 4 70.00 61.33 37.82 90.00 16.00

- 5 62.00 54.67 37.82 92.00 16.00

WISE

- Table 12. Original result of the four metrics of different editing methods on Qwen under sequential editing. For generality and audio locality, we present the averaged results. (%)

Method Gap Reliability Generality Audio Locality Text Locality Portability

0 100.00 100.00 51.30 68.00 18.00 1 82.00 74.67 46.11 66.00 16.00 2 72.00 60.67 44.56 64.00 18.00 3 70.00 54.00 45.60 66.00 14.00 4 66.00 50.67 47.15 66.00 18.00 5 56.00 43.33 40.93 58.00 16.00

FT (LLM)

0 100.00 82.67 82.90 100.00 40.00 1 100.00 78.67 80.83 100.00 40.00 2 100.00 77.33 79.27 100.00 38.00 3 100.00 75.33 78.76 100.00 34.00 4 100.00 76.00 77.72 100.00 34.00 5 98.00 75.33 74.09 100.00 40.00

FT (Audio)

0 88.00 88.67 72.02 86.00 18.00 1 76.00 72.67 73.58 82.00 20.00 2 66.00 68.67 69.43 66.00 18.00 3 68.00 68.00 69.43 72.00 18.00 4 62.00 66.67 63.73 68.00 12.00 5 60.00 54.67 62.18 68.00 18.00

MEND

0 72.00 70.67 43.01 50.00 14.00 1 28.00 27.33 30.05 38.00 4.00 2 16.00 16.67 24.87 34.00 2.00 3 10.00 14.00 19.17 22.00 0.00 4 8.00 8.67 15.03 18.00 2.00 5 6.00 7.33 13.99 12.00 2.00

KE

0 96.00 96.67 47.15 56.00 20.00 1 52.00 50.67 38.34 58.00 6.00 2 42.00 36.00 37.82 58.00 26.00 3 34.00 28.00 36.78 52.00 12.00 4 28.00 27.33 30.57 50.00 18.00 5 30.00 20.00 27.46 50.00 22.00

UnKE

0 10.00 6.00 83.42 40.00 26.00 1 14.00 7.33 78.76 42.00 22.00 2 4.00 6.67 61.14 34.00 26.00 3 14.00 5.33 47.15 26.00 18.00 4 6.00 6.00 30.57 18.00 10.00 5 4.00 3.33 21.24 12.00 8.00

IE-IKE

0 10.00 7.33 88.60 54.00 24.00 1 10.00 7.33 90.16 48.00 22.00 2 10.00 8.00 88.08 48.00 22.00 3 10.00 8.00 88.08 48.00 20.00 4 10.00 8.00 88.08 46.00 22.00 5 10.00 6.67 89.12 46.00 22.00

I-IKE

0 100.00 100.00 58.55 84.00 14.00 1 92.00 90.00 58.03 82.00 10.00 2 92.00 78.67 62.18 76.00 12.00 3 98.00 82.67 61.14 70.00 14.00 4 94.00 80.67 61.65 76.00 18.00 5 96.00 80.00 63.73 66.00 18.00

WISE

- Table 13. Original result of the four metrics of different editing methods on AF under sequential editing. For generality and audio locality, we present the averaged results. (%)

Method Gap Reliability Generality Audio Locality Text Locality Portability

0 100.00 94.67 82.90 90.00 24.00 1 82.00 76.67 79.79 90.00 24.00 2 78.00 70.67 79.27 90.00 32.00 3 72.00 65.33 79.79 94.00 28.00 4 70.00 63.33 78.24 92.00 26.00 5 60.00 58.67 77.20 88.00 32.00

FT (LLM)

0 100.00 70.00 87.56 100.00 44.00 1 100.00 68.00 84.97 100.00 46.00 2 100.00 68.00 80.31 100.00 46.00 3 98.00 64.00 76.68 100.00 44.00 4 98.00 62.00 76.17 100.00 44.00 5 94.00 60.67 75.65 100.00 40.00

FT (Audio)

0 100.00 94.67 86.53 90.00 26.00 1 98.00 92.00 82.38 96.00 26.00 2 98.00 90.67 83.94 92.00 30.00 3 98.00 92.67 81.35 96.00 28.00 4 98.00 92.67 80.83 94.00 26.00 5 94.00 85.33 80.83 94.00 26.00

MEND

0 68.00 72.67 52.33 66.00 24.00 1 60.00 54.67 39.38 48.00 20.00 2 38.00 30.00 24.87 44.00 18.00 3 24.00 26.00 22.80 28.00 12.00 4 16.00 16.00 18.65 22.00 8.00 5 10.00 13.33 12.95 16.00 2.00

KE

0 98.00 88.67 68.39 44.00 22.00 1 88.00 77.33 61.65 50.00 24.00 2 80.00 69.33 59.58 32.65 18.00 3 82.00 66.67 56.48 12.24 12.00 4 80.00 64.00 53.36 16.00 14.00 5 72.00 58.00 48.70 8.00 16.00

UnKE

0 56.00 61.33 37.82 84.00 32.00 1 52.00 52.00 39.38 82.00 28.00 2 50.00 47.33 40.93 76.00 30.00 3 42.00 40.67 39.38 78.00 34.00 4 38.00 40.00 38.34 82.00 30.00 5 38.00 41.33 42.49 74.00 30.00

IE-IKE

0 22.00 9.33 84.97 74.00 38.00 1 22.00 7.33 86.53 74.00 34.00 2 24.00 7.33 87.05 80.00 34.00 3 22.00 5.33 87.05 80.00 36.00 4 22.00 6.00 87.05 76.00 32.00 5 18.00 6.67 86.01 78.00 32.00

I-IKE

0 100.00 74.67 88.08 90.00 24.00 1 84.00 42.00 89.12 94.00 22.00 2 64.00 32.67 89.12 92.00 22.00 3 42.00 22.00 89.12 90.00 22.00 4 34.00 18.67 87.56 90.00 22.00 5 26.00 12.67 85.49 90.00 22.00

WISE

### I. Detailed Breakdown of Portability Evaluation

Given that the scope of potential related knowledge is vast and cannot be entirely enumerated, SAKE addresses this by aggregating portability evaluations across a diverse set of editing instances. Specifically, within our dataset, multiple editing instances may share the same editing pair (yo,ye); however, they are assessed using different portability questions that target a wide variety of related concepts of the edited attribute. To provide a granular analysis of knowledge portability across these concepts, we detail the specific categories covered for each auditory attribute and present the performance breakdown under single editing.

##### I.1. Taxonomy of Knowledge Categories

We categorize portability questions according to the underlying reasoning concepts they involve. The resulting distribution of knowledge categories is illustrated in Figure 5.

##### Animal Sounds

- • Behavior: Typical behavior of the animal (e.g., purring, chewing cud).
- • Care Item: Required items for caring for the animal (e.g., litter box, leash).
- • Diet: Dietary classification of the animal.
- • Family: The closest animal in terms of biological taxonomy.
- • Locomotion: The way the animal moves.
- • Physical: Physical traits of the animal (e.g., claws, hooves).
- • Reproduction: The reproductive method of the animal.
- • Vocalization: The characteristic vocalization of the animal.

[Figure 9]

Figure 5. Distribution of portability knowledge categories associated with edited auditory attributes.

##### Speaker Emotion

- • Descriptive Sentence: A sentence reflecting the speaker’s emotional state.
- • Facial Expression: The facial expression representing the emotion.
- • Scenario: The scenario or situation that matches the speaker’s emotion.
- • Social Interaction: The appropriate social interaction as a response based on the emotion.

##### Speaker Gender

- • History: A historical figure who shares the same gender as the speaker.
- • Celebrity: A celebrity who shares the same gender as the speaker.
- • Cloth: Traditional clothing associated with the speaker’s gender in the relevant cultural context.
- • Title: Formal title of the speaker (e.g., Mr., Ms.).
- • Vocal: Vocal ranges that align with the speaker’s gender.

##### Spoken Language

- • ISO: The ISO language code of the spoken language.
- • City: A city in a country where the spoken language is recognized as official.
- • Dish: A dish originating from a country where the language is official.
- • History: A historical figure from a country where the language is official.
- • Language Family: The language family to which the language belongs.
- • Literary: Books or literary works originally authored in the language.
- • Official Language: The country that recognizes the spoken language as official.
- • Test: A test used to evaluate proficiency in the language.
- • Translation: The translation of an English word into the spoken language.

##### I.2. Performance Breakdown

Tables 14, 15, 16, and 17 detail the portability breakdown under single editting for Animal Sounds, Speaker Emotion, Speaker Gender, and Spoken Language attributes, respectively. The analysis reveals a mixed state of reasoning propagation. Rather than a binary outcome where all related concepts succeed or fail simultaneously, performance varies by category. For instance, within the Animal Sound attribute, FT (Audio) shows a disparity in knowledge propagation, achieving 45.95% accuracy on “reproduction” compared to only 7.89% on “family.” Overall, instances where related concepts fail to update still constitute the majority, indicating that current editing methods struggle to consistently propagate knowledge across different reasoning dimensions.

We further emphasize that the LALMs used in this study already possess the world knowledge required by our portability dataset (Yang et al., 2025b), whose connected attribute knowledge is drawn from the SAKURA benchmark. Therefore, observed portability failures can be attributed to ineffective edit propagation to relevant knowledge, rather than a lack of underlying world knowledge.

Table 14. Breakdown of portability performance across different knowledge categories for the Animal Sound attribute. (%)

Model Method Behavior Care Item Diet Family Locomotion Physical Reproduction Vocalization

FT (LLM)

2.70 23.68 21.05 2.63 68.57 13.51 16.22 15.00 FT (Audio)

24.32 39.47 21.05 5.26 25.71 32.43 45.95 40.00

KE 10.81 39.47 5.26 26.32 34.29 24.32 8.11 37.50 MEND 5.41 36.84 39.47 10.53 34.29 10.81 18.92 7.50 UnKE 21.62 55.26 26.32 5.26 28.57 16.22 18.92 45.00 I-IKE 83.78 92.11 68.42 47.37 85.71 78.38 78.38 92.50

DeSTA

IE-IKE 32.43 57.89 42.11 13.16 51.43 37.84 35.14 80.00 WISE 2.70 15.79 15.79 0.00 17.14 10.81 27.03 2.50

FT (LLM)

21.62 26.32 13.16 23.68 31.43 21.62 10.81 30.00 FT (Audio)

56.76 71.05 36.84 28.95 37.14 51.35 43.24 60.00

KE 24.32 31.58 13.16 23.68 31.43 10.81 8.11 40.00 MEND 18.92 42.11 15.79 15.79 11.43 16.22 18.92 30.00 UnKE 40.54 21.05 18.42 10.53 0.00 8.11 10.81 50.00 I-IKE 29.73 34.21 21.05 18.42 31.43 10.81 5.41 40.00

Qwen

IE-IKE 27.03 50.00 21.05 21.05 22.86 18.92 5.41 42.50 WISE 18.92 18.42 0.00 0.00 2.86 0.00 0.00 12.50

FT (LLM)

35.14 39.47 26.32 18.42 20.00 32.43 32.43 27.50 FT (Audio)

54.05 60.53 28.95 28.95 40.00 48.65 51.35 57.50

KE 37.84 39.47 23.68 23.68 17.14 32.43 24.32 27.50 MEND 35.14 47.37 21.05 28.95 22.86 32.43 29.73 27.50 UnKE 35.14 26.32 23.68 23.68 17.14 21.62 35.14 37.50 I-IKE 40.54 52.63 23.68 26.32 31.43 35.14 43.24 47.50

AF

IE-IKE 48.65 47.37 28.95 31.58 34.29 35.14 45.95 55.00 WISE 29.73 47.37 21.05 26.32 25.71 29.73 35.14 25.00

- Table 15. Breakdown of portability performance across different knowledge categories for the Speaker Emotion attribute. (%) Model Method Descriptive Sentence Facial Expression Scenario Social Interaction

FT (LLM)

24.32 19.48 17.11 27.40 FT (Audio)

44.59 46.75 47.37 45.21

KE 17.57 14.29 11.84 20.55 MEND 22.97 23.38 21.05 27.40 UnKE 16.22 24.68 13.16 19.18 I-IKE 72.97 51.95 68.42 68.49

DeSTA

IE-IKE 45.95 23.38 19.74 26.03 WISE 1.35 1.30 5.26 2.78

FT (LLM)

18.92 16.88 10.53 30.14 FT (Audio)

47.30 44.16 35.53 42.47

KE 21.62 15.58 22.37 31.51 MEND 24.32 15.58 22.37 31.51 UnKE 13.51 12.99 10.53 28.77 I-IKE 22.97 16.88 15.79 32.88

Qwen

IE-IKE 21.62 18.18 18.42 28.77 WISE 22.97 10.39 19.74 34.25

FT (LLM)

14.86 18.18 21.05 17.81 FT (Audio)

36.49 42.86 46.05 27.40

KE 14.86 20.78 22.37 19.18 MEND 17.57 18.18 19.74 17.81 UnKE 12.16 16.88 21.05 20.55 I-IKE 16.22 25.97 27.63 20.55

AF

IE-IKE 24.32 23.38 25.00 30.14 WISE 12.16 20.78 22.37 17.81

- Table 16. Breakdown of portability performance across different knowledge categories for the Speaker Gender attribute. (%) Model Method History Celebrity Cloth Title Vocal

FT (LLM)

6.35 2.99 3.33 6.56 12.24 FT (Audio)

85.71 80.60 71.67 93.44 81.63

KE 7.94 4.48 10.00 4.92 12.24 MEND 4.76 2.99 3.33 3.28 10.20 UnKE 4.76 4.48 8.33 4.92 8.16 I-IKE 80.95 53.73 70.00 78.69 83.67

DeSTA

IE-IKE 36.51 23.88 18.33 45.90 40.82 WISE 33.33 22.39 31.67 36.07 24.49

FT (LLM)

47.62 11.94 18.33 32.79 18.37 FT (Audio)

90.48 35.82 40.00 88.52 57.14

KE 61.90 28.36 36.67 40.98 20.41 MEND 63.49 16.42 25.00 52.46 20.41 UnKE 57.14 32.84 36.67 34.43 26.53 I-IKE 77.78 28.36 31.67 55.74 28.57

Qwen

IE-IKE 65.08 28.36 33.33 36.07 36.73 WISE 68.25 22.39 30.00 59.02 22.45

FT (LLM)

76.19 40.30 51.67 49.18 38.78 FT (Audio)

82.54 43.28 55.00 81.97 53.06

KE 76.19 41.79 48.33 49.18 40.82 MEND 74.60 41.79 48.33 50.82 40.82 UnKE 77.78 32.84 51.67 60.66 42.86 I-IKE 73.02 44.78 60.00 65.57 46.94

AF

IE-IKE 77.78 44.78 60.00 67.21 57.14 WISE 71.43 38.81 50.00 44.26 38.78

- Table 17. Breakdown of portability performance across different knowledge categories for the Spoken Language attribute. (%)

Model Method Iso City Dish History Language Family Literary Official Language Test Translation

FT (LLM)

10.71 41.18 37.14 38.89 14.81 2.86 23.53 30.77 40.62 FT (Audio)

67.86 58.82 74.29 66.67 74.07 54.29 58.82 41.03 81.25

KE 14.29 23.53 22.86 30.56 14.81 11.43 26.47 15.38 40.62 MEND 14.29 23.53 28.57 22.22 11.11 14.29 5.88 41.03 46.88 UnKE 32.14 17.65 5.71 16.67 18.52 8.57 5.88 23.08 28.12 I-IKE 82.14 70.59 77.14 72.22 59.26 48.57 61.76 66.67 84.38

DeSTA

IE-IKE 57.14 26.47 37.14 22.22 11.11 25.71 23.53 23.08 65.62 WISE 78.57 20.59 17.14 2.78 37.04 2.86 41.18 12.82 34.38

FT (LLM)

3.57 55.88 34.29 36.11 3.70 5.71 26.47 33.33 50.00 FT (Audio)

75.00 44.12 40.00 38.89 44.44 31.43 64.71 28.21 62.50

KE 3.57 26.47 28.57 19.44 14.81 17.14 14.71 30.77 46.88 MEND 17.86 29.41 25.71 13.89 18.52 17.14 26.47 23.08 46.88 UnKE 64.29 35.29 14.29 22.22 51.85 20.00 32.35 15.38 43.75 I-IKE 10.71 35.29 20.00 19.44 18.52 14.29 17.65 23.08 46.88

Qwen

IE-IKE 7.14 26.47 25.71 13.89 29.63 14.29 8.82 25.64 50.00 WISE 78.57 44.12 45.71 36.11 40.74 5.71 44.12 33.33 59.38

FT (LLM)

17.86 20.59 14.29 25.00 29.63 34.29 32.35 33.33 37.50 FT (Audio)

92.86 32.35 51.43 55.56 55.56 31.43 41.18 35.90 46.88

KE 28.57 20.59 25.71 25.00 37.04 31.43 17.65 30.77 37.50 MEND 21.43 20.59 17.14 27.78 25.93 34.29 20.59 33.33 40.62 UnKE 10.71 20.59 17.14 22.22 18.52 31.43 23.53 28.21 40.62 I-IKE 25.00 26.47 25.71 25.00 29.63 37.14 23.53 30.77 37.50

AF

IE-IKE 57.14 32.35 25.71 33.33 33.33 34.29 35.29 25.64 40.62 WISE 42.86 23.53 22.86 25.00 25.93 25.71 26.47 25.64 37.50

Table 18. Ratio (%) of degenerated outputs of each editing methods on each LALM under sequential editing.

Model FT (LLM) FT (Audio) KE MEND UnKE WISE IE-IKE I-IKE DeSTA 0.30 1.35 47.77 44.25 2.23 0.14 0.00 0.78

Qwen 0.34 0.20 11.97 0.85 2.30 0.20 1.28 0.34 AF 0.00 0.00 9.67 0.00 6.02 0.00 0.00 0.00

### J. Case Study

While knowledge editing in LALMs offers a way to update auditory knowledge without full retraining, the editing process is not always stable. To better understand the behaviors, we conduct a case study comparing two editing scenarios: (1) a single targeted edit applied to change an emotional attribute, and (2) sequential edits applied across multiple concepts. The comparison highlights both the strengths and limitations of current editing methods, emphasizing the trade-off between reliability of isolated edits and the accumulation of errors when multiple edits interact.

Example of successful editing. Figure 6 shows the result of a successful editing example by FT (Audio) on Qwen, where we edit the model’s perception of speaker emotion from “fearful” to “sad.” After editing, we observe that the model’s outputs for both reliability and generality are successfully updated to “sad.” At the same time, both audio locality and text locality remain intact, which shows that the original knowledge of the model is preserved. For portability, we observe that after the edit, the model can also perform reasoning, changing the prediction from “preparing for a high-stakes exam with anxiety” to “a person crying after a breakup.” This demonstrates that the interconnected knowledge is also updated during reasoning, and the edited model can apply this knowledge in new contexts.

Degeneration Analysis. Figure 7 shows a degenerated example from sequential editing on DeSTA with MEND. After applying multiple edits in a row, we can see that the model collapses and produces incoherent outputs such as repeated characters and newline symbols. For example, some of the output will become ”happy catholic catholic catholic catholic......” or ”dogdogdogdogdog....” This illustrates how sequential edits can accumulate interference and destabilize the internal representations of the model. Unlike the single-edit scenario, where the change is targeted and localized, multiple edits interact in unpredictable ways, leading to corruption of reliability, loss of generality, and failure of both locality and portability. To be practical in real-world scenarios, however, an editing method must be capable of supporting many edits simultaneously while ensuring that unrelated edits do not interfere with one another. This failure case thus underscores a key limitation of current approaches.

Quantitatively, we detect repetitive degeneration patterns in model outputs using regular expressions for each editing method and LALM under sequential editing. The frequency of degeneration is summarized in Table 18. These statistics are consistent with the trends in Figure 4, where MEND exhibits severe degeneration on DeSTA, KE on all models, and UnKE on AF.

[Figure 10]

###### Figure 6. An example of a successful editing result by FT (Audio) on Qwen.

[Figure 11]

###### Figure 7. An example of a degenerated editing result by MEND on DeSTA.

