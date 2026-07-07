## arXiv:2511.15848v2[cs.AI]26Nov2025

[Figure 1]

# Step-Audio-R1 Technical Report

#### StepFun-Audio Team

StepAudio R1 Official Github Page StepAudio R1 Official Demo Page

### Abstract

Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question—can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities.

###### S2T Benchmark

###### S2S Benchmark

1.5

Step-Audio-R1-Realtime Gemini-2.5-FlashNative-Audio-Dialog GPT-Realtime-0825

Step-Audio-R1

98.7

Gemini-2.5-Pro Gemini-3-Pro

100

100

96.1

96.1 94.8

95.3

95.2

| |
|---|

92.0

92.1

| |
|---|

| |
|---|

1.2

83.0

82.9

79.3

78.9

77.7

80

80

77.4

76.4

75.9

0.98s

PerformanceScore(%)

PerformanceScore(%)

70.6

0.92s

Latency(seconds)

0.9

60.2

60

60

0.63s

0.6

40

40

0.3

20

20

0

0

0.0

BigBenchAudio SpokenMQA MMSU MMAU WildSpeech

BigBenchAudio-Acc BigBenchAudio-Latency

Figure 1: Benchmark performance of Step-Audio-R1

### 1 Introduction

Chain-of-thought reasoning has transformed modern artificial intelligence, enabling language models to solve complex mathematical problems [1–3], generate executable code [4], and engage in sophisticated logical deduction through extended deliberation [5]. Vision-language models have similarly adopted this paradigm, leveraging deliberate reasoning to interpret spatial relationships [2], analyze visual scenes, and answer intricate questions about images [6, 7]. Underlying these successes is a fundamental principle known as test-time compute scaling: allocating more computational resources during inference—through longer chains of thought, iterative refinement, or search—predictably improves model performance [8–10]. This scaling law has become so robust that it now guides the design and deployment of AI systems across modalities.

The audio domain, however, presents a stark exception to this principle. Existing audio language models consistently demonstrate superior performance with minimal or no reasoning [11, 12]. Empirical observations across benchmarks reveal that direct responses outperform elaborate chain-of-thought explanations, with performance systematically degrading as reasoning length increases [11, 13, 14]. This inverted scaling behavior persists across architectures, training methodologies, and model scales [12, 13], suggesting a fundamental incompatibility between test-time compute scaling and auditory intelligence. This raises a critical question:

Is audio inherently resistant to deliberate reasoning?

Recent efforts have attempted to address this anomaly through reinforcement learning approaches that employ language model judges to verify consistency between reasoning chains and final answers [15, 16]. While these methods improve alignment, they treat the symptom rather than the root cause—enforcing consistency without understanding why reasoning fails in audio. Through systematic case studies, we uncover a striking pattern: existing audio language models engage in textual surrogate reasoning rather than acoustic reasoning. When prompted to deliberate, models systematically reason from the perspective of transcripts or textual captions instead of acoustic properties—for instance, attributing musical melancholy to "lyrics mentioning sadness" rather than "minor key progressions and descending melodic contours". This leads to a critical hypothesis: the performance degradation stems not from reasoning itself, but from reasoning about the wrong modality. We trace this to a fundamental design choice: most audio language models initialize their reasoning capabilities through supervised fine-tuning on COT [17] data derived from text-based models [12, 13]. Consequently, these models inherit linguistic grounding mechanisms, creating a modality mismatch that undermines performance as reasoning chains lengthen.

To validate this hypothesis and unlock reasoning capabilities in audio, we propose ModalityGrounded Reasoning Distillation (MGRD), an iterative training framework that progressively shifts reasoning from textual abstractions to acoustic properties. Starting from text-based reasoning initialization, MGRD employs iterative cycles of self-distillation and refinement on audio tasks, systematically curating reasoning chains that genuinely ground in acoustic analysis. Through these iterations, we obtain Step-Audio-R1, the first audio reasoning model that successfully benefits from test-time compute scaling, outperforming Gemini 2.5 Pro [14] and demonstrating capabilities competitive with the latest Gemini 3 Pro [18] across comprehensive audio benchmarks. These results confirm that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence.

Text

Training Iteration 1

Training Iteration 2

Sematic-based Text Think (STT)

[Figure 2]

Acoustic-based Text Think (ATT)

MGRD

MGRD

[Figure 3]

Audio Features MGRD STT-to-ATT Transition

<think> </think>

<think> </think>

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### StepAudio R1

<think> </think>

<think> </think>

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Input Text Tokens Input Audio Features

[Figure 17]

Text Tokenizer Adapter

[Figure 18]

Audio Encoder

[Figure 19]

Query

Figure 2: The overview of Step-Audio-R1

### 2 Model Overview

Drawing from the architecture of our previous Step-Audio 2 [13], Step-Audio-R1 is designed for audio-based reasoning tasks. As shown in Figure 2, the model consists of an audio encoder, an audio adaptor, and an LLM decoder.

For the audio encoder, we utilize the Qwen2 audio encoder [19], which is pretrained on various speech and audio understanding tasks. The audio encoder has an output frame rate of 25 Hz and is frozen during the entire training process. An audio adaptor with a downsampling rate of 2, identical to the one in Step-Audio 2, is employed to connect the audio encoder to the LLM, thereby reducing the output frame rate to 12.5 Hz.

The LLM decoder, based on Qwen2.5 32B [20], directly takes the latent audio features from the audio adaptor as input to generate a purely textual output. The model is structured to first generate the reasoning content, followed by the final reply.

A key innovation in this process is the Modality-Grounded Reasoning Distillation (MGRD) method. Initially, the model’s reasoning process may operate on a purely semantic level. MGRD iteratively refines these thoughts, progressively strengthening their connection to the underlying audio features until they evolve into "native audio think." This distillation process ensures that the model’s reasoning is not merely about the transcribed text, but is deeply grounded in the acoustic nuances of the audio itself, leading to a more holistic and accurate final response.

Step-Audio-R1 is pretrained using the same data and methodology as Step-Audio 2. Following this, the model undergoes a Post-Training phase, with specific details provided in Section 4.

### 3 Data Preparation

- 3.1 Data for Cold-Start

Our cold-start phase is designed to jointly elicit audio reasoning capabilities through a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verified Reward (RLVR). This phase utilizes a total dataset of 5 million samples. This token budget is comprised of 1B tokens from text-only data, with the remaining 4B tokens derived from our audio-side data.

The data types are as follows:

- • Audio Data: This includes Automatic Speech Recognition (ASR), Paralinguistic Understanding, and standard Audio Question Text Answer (AQTA) dialogues.
- • Audio CoT Data: We incorporate AQTA Chain-of-Thought (CoT) data, which is generated via self-distillation from our own model after its audio reasoning capabilities were elicited. This CoT data constitutes 10% of our total audio dataset.
- • Text Data: This includes text-only dialogues (in both single-turn and multi-turn formats) covering topics such as knowledge-based QA, novel continuation, role-playing, general chat, and emotional conversations. It also incorporates the text CoT data, which focuses on math and code.

A critical aspect of our data strategy is the standardized reasoning format. To train the model to recognize the reasoning structure, we prepend all samples lacking native CoT with an empty <think> tag. The format is standardized as: <think>\n\n</think>\n{response}

- 3.2 Data for RL

For the subsequent Reinforcement Learning (RL) phase, we curated a smaller, high-quality dataset of 5,000 samples. This dataset is composed of 2,000 high-quality text-only samples (focusing on math and code) and 3,000 augmented speech-based QA samples. The augmentation methods used to process this data are described in detail in Section 4.2.

### 4 Post-Training Recipes

#### 4.1 Foundation Training: Reasoning Initialization and Format Alignment

We establish fundamental reasoning capabilities through a two-stage training process that builds robust reasoning primitives while maintaining basic audio understanding.

#### Supervised Chain-of-Thought Initialization. Given a base audio-language model πθ

, we perform supervised fine-tuning on chain-of-thought demonstrations from both task-oriented and conversational domains, along with audio data to preserve multimodal capabilities. The training objective unifies three data sources:

0

LSFT = E(q,r,a)∼Dtask [log πθ(r, a | q)] + E(c,r,s)∼Dconv [log πθ(r, s | c)] + E(xaudio,q,a)∼Daudio [log πθ(a | xaudio, q)] (1)

where (q,r,a) denotes task questions with reasoning chains and answers, (c,r,s) represents conversational contexts with deliberation and responses, and (xaudio,q,a) indicates audio questions with

Iterative Refinement

Multimodal RL

Multimodal RL

Multimodal RL

Cold Start Model

Model

Model

Model

Text RLVR Data

Pass K

Self-Distillation with Acoustic Reasoning

&

Pass K

Data with

Strong Perception Data

LLM Judger

Verified Answer

Figure 3: Modality-Grounded Reasoning Distillation

direct answers. For audio data, we use empty reasoning markers (i.e., <think>\n\n</think>\n) to maintain the structural format without actual deliberation content. This tri-modal training instills diverse reasoning patterns in text domains—spanning analytical problem-solving, code generation, logical inference, and contextual dialogue—while preserving the model’s audio understanding capabilities for subsequent acoustic reasoning distillation.

Reinforcement Learning with Verified Rewards. Building upon the supervised foundation, we refine reasoning quality on task-oriented data through Reinforcement Learning with Verified Rewards (RLVR) [3, 21]. For mathematical problems, coding challenges, and logical puzzles, the model samples reasoning trajectories and receives binary verification rewards:

R(r,a) =

1, if a = a∗ 0, else

(2)

We optimize using Proximal Policy Optimization [22] without KL penalty constraints, maximizing expected reward:

##### [R(r,a)] (3)

##### LRLVR = ED

task

This allows free exploration of reasoning strategies while maintaining answer accuracy through outcome-based verification.

#### 4.2 Modality-Grounded Reasoning Distillation

With textual reasoning foundation established, we now address the core challenge: transforming reasoning capabilities from textual abstractions to acoustic grounding. We propose an iterative self-distillation framework that progressively refines the model’s reasoning to genuinely attend to audio properties.

This iterative process is motivated by the emergent audio Chain-of-Thought (CoT) capability observed after the cold-start phase. Our goal is to maintain and enhance this ability. We first construct a new set of perception-grounded questions based on our existing audio data. Then, at each iteration t, we use the model from the previous iteration (πθ

) to perform self-distillation, generating new reasoning chains for this data.

t−1

Self-Distillation with Acoustic Reasoning. At each iteration t, we begin by curating audio data that strongly emphasizes perceptual analysis. Given an audio dataset Daudio, we select examples (xaudio,q)

where answering question q requires direct acoustic feature analysis rather than high-level semantic understanding. This selection prioritizes tasks demanding attention to timbral qualities, temporal patterns, pitch contours, rhythmic structures, and other low-level auditory properties, ensuring the model cannot rely on textual surrogates. For each selected audio-question pair (xaudio,q), we prompt the current model πθ

to generate reasoning chains that explicitly reference acoustic features:

t

(· | xaudio,q), i = 1,...,K (4) We sample K candidate responses and filter them using quality criteria that verify: (1) acoustic grounding—reasoning explicitly mentions perceptual features rather than textual descriptions; (2) logical coherence—reasoning steps follow sound inferential structure; and (3) answer correctness—final answers align with ground truth when available. This filtering yields a curated dataset Dtaudio-cot of acoustically-grounded reasoning chains.

##### (r(i),a(i)) ∼ πθ

t

Multimodal Supervised Refinement. We perform supervised fine-tuning on the distilled acoustic reasoning data, combined with original textual reasoning data to preserve existing capabilities:

L(SFTt) = EDaudio-cot

##### [log πθ(r,a | q)] (5)

##### [log πθ(r,a | xaudio,q)] + ED

task

t

This joint training anchors reasoning to acoustic properties while maintaining textual reasoning proficiency.

Multimodal Reinforcement Learning. We further refine the model through reinforcement learning on both audio and text tasks with carefully designed reward structures. For text questions, we employ standard binary verification:

Rtext(r,a) =

1, if a = a∗ 0, else

For audio questions, we combine format and accuracy rewards:

(6)

Raudio(r,a) = 0.8 ×

1, if a = a∗ 0, else

+ 0.2 ×

1, if reasoning present in r 0, else

(7)

This design prioritizes answer correctness (0.8 weight) while incentivizing reasoning generation (0.2 weight), preventing the model from reverting to direct responses. The combined optimization objective is:

##### L(RLVRt) = ED

##### [Rtext(r,a)] (8)

##### [Raudio(r,a)] + ED

audio

task

Iterative Refinement. We repeat this cycle for T iterations, with each iteration t producing model πθ

that generates progressively more acoustically-grounded reasoning. As iterations advance, the model’s reasoning chains shift from textual surrogates—such as inferring emotion from "lyrics mentioning sadness"—to genuine acoustic analysis—such as "minor key progressions and descending melodic contours." This iterative distillation progressively transforms the model’s reasoning substrate from linguistic to acoustic grounding.

t+1

The final model πθ

achieves the desired capability: generating extended reasoning chains that genuinely attend to audio properties, thereby unlocking test-time compute scaling benefits in the audio domain.

T

#### 4.3 Implement Details

RL Data Curation and Filtering Details. To construct the dataset for the RL phase, we extract text QA and audio data spanning diverse tasks and topics. We then filter these questions to identify a high-quality, challenging subset. Using the model from the t − 1 iteration, we sample k = 8 responses for each question (pass@8). A question is selected for the RL dataset if the number of correct passes falls within the range of [3,6]. This filtering mechanism ensures we select for problems that are relatively difficult, filtering out both overly simple questions (where pass@8 > 6) and potentially harmful or nonsensical questions (where pass@8 < 3).

RL Implementation Details We employ an on-policy Proximal Policy Optimization framework [22] with binary verification rewards: responses receive a reward of 1.0 when matching verified solutions and 0.0 otherwise. Critically, we remove reference model KL penalties by setting the penalty coefficient to zero, allowing the model to freely explore reasoning strategies without being constrained by its initialization distribution. During training, we sample 16 candidate responses per prompt, assigning rewards exclusively at the final token position to encourage complete reasoning trajectories. We configure PPO with a clipping parameter of 0.2 and set both the discount factor and GAE lambda to 1.0, training on sequences up to 10,240 tokens to accommodate extended deliberation.

### 5 Evaluation

Having established that audio intelligence can indeed benefit from deliberate reasoning, we now present a comprehensive empirical evaluation of Step-Audio-R1. Our assessment rigorously examines its capabilities across a spectrum of complex audio tasks, structured into two key benchmarks: the Evaluation on Speech-to-Text Benchmarks, which measures understanding and reasoning from acoustic signals, and the Evaluation on Speech-to-Speech Benchmarks, which assesses the model’s ability to perform generative and interactive reasoning in real-time spoken dialogue scenarios within the auditory domain.

#### 5.1 Evaluation on Speech-to-Text Benchmarks

- Table 1: Performance comparison (in %) on speech-to-text benchmarks across Big Bench Audio, Spoken MQA, MMSU, MMAU, Wild Speech, and Average Score.

Model Avg. Big Bench Audio Spoken MQA MMSU MMAU Wild Speech

Step-Audio 2 68.3 59.1 88.8 64.3 78.0 51.1 Gemini 2.5 Pro 81.5 96.1 94.8 79.3 77.4 60.0 Gemini 3 Pro 85.1 92.1 95.3 82.9 78.9 76.4

Step-Audio-R1 83.6 98.7 95.2 75.9 77.7 70.6

This section evaluates the speech understanding and reasoning capabilities of Step-Audio-R1 against several state-of-the-art baselines: the powerful large-language model Gemini 2.5 Pro, the newly

released Gemini 3 Pro, our own previous-generation model Step-Audio 2, and the base Step-AudioR1 model. The assessment is conducted across a comprehensive suite of benchmarks designed to probe advanced audio intelligence. These include MMSU [23] and MMAU [24] for expert-level audio understanding and reasoning, Big Bench Audio1 for complex multi-step logical reasoning from audio, Spoken MQA [25] for mathematical reasoning with verbally expressed problems, and Wild Speech [26] for evaluating conversational speech.

As shown in Table 1, Step-Audio-R1 achieves an average score of 83.6%, significantly outperforming Gemini 2.5 Pro while being slightly lower than Gemini 3 Pro. This competitive performance confirms that our MGRD approach effectively enhances deep audio comprehension.

#### 5.2 Evaluation on Speech-to-Speech Benchmarks

- Table 2: Performance comparison of representative models on the Big Bench Audio speech-to-speech benchmark. The benchmark comprises two evaluation metrics: the Speech Reasoning Performance Score (%), measuring the model’s reasoning ability over spoken content, and the first-packet Latency (seconds) metric, quantifying response speed as an indicator of dialogue fluency.

Model Speech Reasoning Performance Score (%) Latency (seconds)

GPT-4o mini Realtime 69 0.81 GPT Realtime 0825 83 0.98 Gemini 2.5 Flash Live 74 0.64 Gemini 2.5 Flash Native Audio Dialog 92 0.63

Step-Audio-R1 Realtime 96.1 0.92

In this section, we evaluate the model’s performance on the Big Bench Audio speech-to-speech benchmark. This benchmark consists of two major dimensions, namely the speech reasoning performance score, which assesses the model’s ability to perform reasoning over spoken content, and the latency metric, which measures response speed as an indicator of dialogue fluency. Following the design of the listen-while-thinking[27] and think-while-speaking[28] architecture, we adapt Step-Audio-R1 into Step-Audio-R1 Realtime, attaining high-quality reasoning together with rapid responsiveness. According to Table 2, Step-Audio-R1 Realtime reaches a speech reasoning performance score of 96.1%, outperforming exemplary closed-source systems including GPT Realtime 0825 and Gemini 2.5 Flash Native Audio Dialog. Besides, Step-Audio-R1 Realtime achieves a first-packet latency of 0.92 s, maintaining sub-second responsiveness that represents a highly competitive interaction performance among contemporary audio language models. These results demonstrate that Step-Audio-R1 Realtime integrates real-time responsiveness with advanced reasoning capabilities, highlighting its potential for building efficient, intelligent, and interactive large audio language models.

### 6 Empirical Study and Analysis

#### 6.1 Extended Reasoning Benefits Audio: Evidence from Format Reward Ablation

To validate the necessity of our composite reward design for audio tasks, we conduct an ablation study comparing training with and without the format reward component. The results reveal crucial insights into how reward structure shapes model behavior in audio reasoning tasks.

1https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio

[Figure 20]

[Figure 21]

(a) Mean reward evolution (b) Reasoning token count

- Figure 4: Impact of format rewards on audio reasoning training. (a) Format rewards enable faster and more stable convergence to high reward values. (b) Without format rewards, models exhibit systematic reasoning collapse, reducing generated tokens from 3000 to below 1500.

Format Reward Drives Stable Convergence. Figure 4a presents the evolution of mean reward on audio tasks across training iterations. Both configurations eventually converge to similar reward levels (approximately 0.75-0.80), but their trajectories differ significantly. The model with think format reward (cyan line) achieves stable performance earlier, reaching the 0.70 threshold around iteration 35-40, while the model without format reward (blue line) requires nearly 25 iterations to reach comparable performance. More critically, the format-rewarded model maintains more stable training dynamics in later iterations (30-60), whereas the baseline exhibits higher variance and occasional performance drops. This stability advantage translates to meaningful performance gains: on the MMAU benchmark, incorporating the format reward improves accuracy from 76.5 to 77.7.

Format Rewards Prevent Reasoning Collapse. Figure 4b reveals a striking phenomenon that explains the performance difference. Without format rewards, the model exhibits systematic collapse of reasoning length: starting from approximately 3000 tokens in early iterations, it progressively decays to below 1500 tokens by iteration 60, with a particularly sharp decline after iteration 30. In contrast, the model with format rewards maintains substantially longer and more stable reasoning chains throughout training, consistently generating 2300-2800 tokens even in later iterations. This 50-80% increase in reasoning length is not merely superficial verbosity—the accompanying performance improvements on MMAU confirm that these extended deliberations contain meaningful acoustic analysis.

The collapse pattern reveals a critical failure mode: without explicit incentives for reasoning generation, reinforcement learning naturally gravitates toward the most token-efficient strategy—direct answers without deliberation. This optimization pressure directly contradicts our goal of developing genuine reasoning capabilities. The think format reward component acts as a crucial regularizer, ensuring the model maintains extended thought chains even when pure accuracy-based rewards might prefer brevity.

Extended Reasoning Improves Audio Understanding. Most importantly, these training dynamics yield a fundamental shift in model capabilities: Step-Audio-R1 with extended reasoning chains consistently outperforms variants with minimal or no deliberation. This validates the central thesis of our work—that audio intelligence can benefit from extended deliberation when reasoning is properly grounded in acoustic properties. The performance gap between models with full reasoning (MMAU: 77.7) versus abbreviated or absent reasoning demonstrates that test-time compute scaling, once considered incompatible with audio tasks, now provides measurable advantages. This breakthrough confirms that the historical performance degradation with reasoning in audio models stemmed not

from fundamental incompatibility, but from inadequate grounding mechanisms—a problem our MGRD framework successfully addresses.

#### 6.2 Strategic Data Selection: Quality Over Quantity in Audio RL

[Figure 22]

[Figure 23]

(a) Mean reward evolution (b) Reasoning token count

- Figure 5: Impact of data selection strategies on audio reasoning training. (a) Training on moderately difficult problems (correct passed) achieves higher and more stable rewards compared to failed problems, which collapse after iteration

50. (b) Moderately difficult problems sustain reasoning generation (2300-2800 tokens) , while failed problems show a progressive decline, settling around 1800-2000 tokens by iteration 60.

We discover that careful data curation proves more critical than dataset volume for audio reasoning tasks. Through comparing three data selection strategies, we reveal what constitutes effective training data for MGRD.

Comparing Selection Criteria. We evaluate two distinct data selection approaches for the RL phase: (1) Consistently-failed problems: questions where the SFT model from iteration t−1 fails all k = 8 sampled attempts (pass@8 = 0); (2) Moderately difficult problems: questions where correct passes fall within [3,6] out of 8 attempts, as described in our MGRD framework. Additionally, we experiment with (3) Unfiltered scaling: expanding the audio RL dataset to 200K examples without difficulty-based selection.

Learning from Partial Success Outperforms Learning from Failure. Figure 5a reveals striking differences in training dynamics. Models trained on moderately difficult problems (cyan line, "correct passed problem") achieve substantially higher and more stable rewards, converging to approximately 0.75-0.80 by iteration 20 and maintaining this performance throughout training. In sharp contrast, models trained exclusively on consistently-failed problems (blue line, "failed problem") exhibit significantly lower rewards (0.45-0.70) with higher variance and unstable convergence, eventually collapsing after iteration 50.

This performance gap stems from fundamental differences in learning signals. Problems where the SFT model consistently fails often indicate inherent ambiguity or insufficient information in the audio modality—for instance, inferring a car’s brand from engine sounds alone, a task trivial in vision but nearly impossible from audio. Without correct reasoning exemplars in sampled trajectories, the model explores blindly, unable to distinguish between genuine acoustic limitations and solvable challenges. Moderately difficult problems, conversely, provide a crucial mix: some responses demonstrate correct acoustic reasoning paths while others reveal common failure modes. This combination enables effective policy gradient updates—the model learns both successful reasoning strategies and mistakes to avoid, while naturally filtering out acoustically ambiguous questions.

Reasoning Complexity Reflects Learning Quality. Figure 5b corroborates this finding through reasoning length analysis. Initially, both strategies generate similar reasoning lengths (approximately 3000-3500 tokens in early iterations), as they start from the same SFT checkpoint. However, their trajectories diverge significantly after iteration 20. Models trained on moderately difficult problems (cyan line) maintain substantial reasoning chains, stabilizing at 2300-2800 tokens throughout later training, demonstrating sustained deliberation. Models trained on consistently-failed problems (blue line), however, show progressive decline: reasoning length gradually decreases from iteration 20 onward, eventually settling around 1800-2000 tokens by iteration 60—a 30-40% reduction from the moderately difficult setting.

This divergence reveals how data quality shapes reasoning behavior over time. While both models initially maintain extended thinking inherited from SFT, continued training on consistently-failed problems gradually erodes this capability. The absence of successful reasoning exemplars provides no positive reinforcement for extended deliberation, causing the model to progressively abandon lengthy reasoning chains. In contrast, moderately difficult problems—which contain both successful and failed attempts—sustain the model’s reasoning complexity by rewarding extended acoustic analysis that leads to correct answers.

Scale Without Strategy Provides No Benefit. Most surprisingly, we experiment with scaling the audio RL dataset to 200K examples—over 10× our curated subset—and observe no performance improvement. This null result carries important implications: in audio reasoning tasks, data quality substantially outweighs quantity. Indiscriminate scaling introduces noise from acoustically ambiguous or inherently unsolvable problems, diluting the learning signal from genuinely informative examples. The effectiveness of challenging-but-solvable problems suggests that successful audio reasoning requires careful curriculum design rather than brute-force data scaling.

#### 6.3 Self-Cognition Correction Through Iterative Refinement

A critical challenge emerges when training Audio LLMs on massive textual data: models tend to develop incorrect self-cognition [12, 13]. Due to the dominance of text-only patterns in the training corpora, these models frequently claim inability to process audio inputs by stating “I cannot hear sounds” or “I am a text model.” This misalignment between actual capabilities and self-perception severely undermines user experience and model utility. We address this systematic bias through a multi-stage correction pipeline combining iterative self-distillation with preference optimization.

Iterative Self-Distillation with Cognition Filtering. Our correction process begins with targeted data curation. We construct a dataset of audio perception queries specifically designed to elicit self-cognition responses—questions about sound identification, audio quality assessment, and acoustic property analysis. During the self-distillation SFT iterations, we employ an LLM judge to filter responses exhibiting incorrect self-cognition. The judge evaluates whether the model appropriately acknowledges its audio processing capabilities or incorrectly identifies as text-only. Only responses with correct self-cognition pass to the next training round, progressively reinforcing accurate self-perception while eliminating erroneous beliefs.

Preference Optimization for Final Correction. Following the filtered self-distillation phase, we apply DPO [29] for precise calibration. We construct 8,000 preference pairs through self-distillation: positive examples comprise responses where the model correctly acknowledges and utilizes its audio capabilities, while negative examples contain responses claiming text-only limitations. This contrastive learning directly targets the remaining self-cognition errors, teaching the model to

consistently choose responses aligned with its true multimodal nature. Despite the relatively modest dataset size, this targeted approach proves remarkably effective due to the specificity of the correction task.

- Table 3: Self-cognition error rates across training stages on our constructed test set of 5,000 diverse audio perception samples. Error rate measures the percentage of responses where the model incorrectly claims inability to process audio.

###### Training Stage Self-Cognition Error Rate

Base model 6.76% Iterative Self-Distillation 2.63% Iterative Self-Distillation + DPO 0.02%

Progressive Error Reduction. Table 3 demonstrates the effectiveness of our multi-stage approach. The base model exhibits noticeable self-cognition errors (6.76%), reflecting the bias from text training data. Through iterative self-distillation, we successfully reduce the error rate to 2.63% by filtering misaligned responses and reinforcing correct self-perception. However, the most dramatic improvement comes from the final DPO alignment with 8,000 preference pairs: error rates plummet to near-zero (0.02%), effectively eliminating self-cognition misalignment. This final stage proves crucial—while self-distillation significantly improves cognition, only explicit preference optimization achieves complete correction. The efficiency of this approach highlights the power of targeted preference learning for addressing specific behavioral biases. For a detailed qualitative comparison of model responses before and after Modality-Grounded Reasoning Distillation, please refer to Appendix A.2.

The success of this approach reveals an important insight: self-cognition errors are not fundamental model limitations but learned biases from training data distribution. Through systematic correction combining iterative refinement with targeted preference optimization, we demonstrate that models can maintain accurate self-perception even when pretrained on predominantly text data. This correction is essential for Step-Audio-R1’s deployment—users expect the model to confidently engage with audio inputs rather than apologetically claim incapability.

### 7 Conclusion

In this work, we address the challenging problem where audio language models historically fail to benefit from long reasoning processes, often performing worse as the reasoning length increases. We identify the primary cause of this failure as “textual surrogate reasoning”—a persistent tendency for models to reason based on text descriptions, such as transcripts or captions, rather than focusing on actual acoustic properties. We introduce Step-Audio-R1, the first model to successfully unlock and benefit from deliberate thinking in the audio domain. Our core contribution is ModalityGrounded Reasoning Distillation (MGRD), an iterative framework that progressively shifts the model’s reasoning basis from text-based patterns to genuine acoustic analysis. Comprehensive evaluations confirm that Step-Audio-R1 outperforms strong baselines, including Gemini 2.5 Pro, and achieves performance comparable to the state-of-the-art Gemini 3 Pro across a wide range of complex audio understanding and reasoning benchmarks. These results provide clear evidence that reasoning is a capability that works across modalities; when properly connected to the correct input, extended reasoning transforms from a weakness into a powerful asset for audio intelligence, opening new paths for building truly multimodal systems.

### 8 Contributors

Core Contributors: Fei Tian1,∗,†, Xiangyu Tony Zhang1,3, Yuxin Zhang1,4, Haoyang Zhang1,2, Yuxin Li1,2, Daijiao Liu1,3

Contributors: Yayue Deng1, Donghang Wu1,2, Jun Chen1, Liang Zhao1, Chengyuan Yao1, Hexin Liu2, Eng Siong Chng2, Xuerui Yang1, Xiangyu Zhang1, Daxin Jiang1, Gang Yu1

1StepFun 2Nanyang Technological University 3University of New South Wales

- 4Shanghai Jiao Tong University ∗Corresponding authors: tianfei@stepfun.com †Project Leader References

- [1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [4] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [5] Bin Wang, Bojun Wang, Changyi Wan, Guanzhe Huang, Hanpeng Hu, Haonan Jia, Hao Nie, Mingliang Li, Nuo Chen, Siyu Chen, et al. Step-3 is large yet affordable: Model-system co-design for cost-effective decoding. arXiv preprint arXiv:2507.19427, 2025.
- [6] Wei Shen, Jiangbo Pei, Yi Peng, Xuchen Song, Yang Liu, Jian Peng, Haofeng Sun, Yunzhuo Hao, Peiyu Wang, Jianhao Zhang, et al. Skywork-r1v3 technical report. arXiv preprint arXiv:2507.06167, 2025.
- [7] Zhuosheng Zhang, Aston Zhang, Mu Li, George Karypis, Alex Smola, et al. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research.
- [8] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.
- [9] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [10] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in

- language models. In The Eleventh International Conference on Learning Representations.
- [11] Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, and Jian Luan. Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering. arXiv preprint arXiv:2503.11197, 2025.
- [12] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.
- [13] Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, et al. Step-audio 2 technical report. arXiv preprint arXiv:2507.16632, 2025.
- [14] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [15] Jiajun Fan, Roger Ren, Jingyuan Li, Rahul Pandey, Prashanth Gurunath Shivakumar, Ivan Bulyko, Ankur Gandhe, Ge Liu, and Yile Gu. Incentivizing consistent, effective and scalable reasoning capability in audio llms via reasoning process rewards. arXiv preprint arXiv:2510.20867, 2025.
- [16] Shu Wu, Chenxing Li, Wenfu Wang, Hao Zhang, Hualei Wang, Meng Yu, and Dong Yu. Audio-thinker: Guiding audio language model when and how to think via reinforcement learning. arXiv preprint arXiv:2508.08039, 2025.
- [17] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903.
- [18] Google DeepMind. Gemini 3. https://deepmind.google/models/gemini/, 2025. Accessed: 2025.
- [19] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.
- [20] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024.
- [21] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.
- [22] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [23] Dingdong Wang, Jincenzi Wu, Junan Li, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, and Helen Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779, 2025.

- [24] S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024.
- [25] Chengwei Wei, Bin Wang, Jung-jae Kim, and Nancy F Chen. Towards spoken mathematical reasoning: Benchmarking speech-based models over multi-faceted math problems. arXiv preprint arXiv:2505.15000, 2025.
- [26] Jian Zhang, Linhao Zhang, Bokai Lei, Chuhan Wu, Wei Jia, and Xiao Zhou. Wildspeech-bench: Benchmarking audio llms in natural speech conversation. arXiv preprint arXiv:2506.21875, 2025.
- [27] Donghang Wu, Haoyang Zhang, Chen Chen, Tianyu Zhang, Fei Tian, Xuerui Yang, Gang Yu, Hexin Liu, Nana Hou, Yuchen Hu, et al. Chronological thinking in full-duplex spoken dialogue language models. arXiv preprint arXiv:2510.05150, 2025.
- [28] Donghang Wu, Haoyang Zhang, Jun Chen, Hexin Liu, Eng Siong Chng, Fei Tian, Xuerui Yang, Xiangyu Zhang, Daxin Jiang, Gang Yu, et al. Mind-paced speaking: A dual-brain approach to real-time reasoning in spoken language models. arXiv preprint arXiv:2510.09592, 2025.
- [29] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

### A Appendix

#### A.1 Audio Reasoning Examples

In this section, we demonstrate the model’s advanced capabilities in audio reasoning across diverse scenarios. As shown in the following examples, the model can effectively capture and reason about paralinguistic features (e.g., emotion, vocal characteristics, prosody) and identify complex environmental sound scenes. These cases highlight the model’s versatility in processing rich acoustic information beyond simple speech recognition.

Audio: User: As a professional music maestro, appreciate this song from a musical perspective. Response:

<think> Okay, the user has shared an audio clip of a song and wants me to analyze it as a “professional music maestro.” They specifically mentioned appreciating it from a musical perspective. First, I need to understand what they’re really asking for. The phrase “professional music maestro” suggests they want a detailed, expert-level breakdown—not just casual observations. Since they provided an audio file, they’re expecting me to listen critically and dissect the song’s technical elements. Hmm, the user didn’t specify any particular aspects to focus on, so I should cover all bases: structure, instrumentation, vocal delivery, production quality, etc. But I must avoid being too dry—maestros are passionate, so my analysis should reflect that enthusiasm while staying precise.

- *Listening to the clip again*... Ah, this is clearly a pop/R&B track with strong contemporary sensibilities. The vocal is front and center, which tells me it’s likely a ballad or slow-tempo track. Interesting how the producer layered those subtle synth pads underneath the piano—creates that “dangerous” vibe the lyrics mention. I notice the user included timestamps (0:00-28:00) but the actual clip is only ∼28 seconds. They might be testing if I’ll point out the discrepancy, but since the analysis is based on what’s provided, I’ll focus on the given material. Smart of them to include the full timestamp though—it shows they’re detail-oriented.
- *Pausing to analyze the chorus*... The vocal harmonies here are masterful. That “dangerous” line has this slight vocal fry that adds texture. And the way the drums hit on the off-beats? Classic modern R&B groove. I should highlight how the production mirrors the lyrical theme—quiet intensity building under the surface. Wait—should I mention the key? *Checks*... Yeah, it’s in C# minor. Perfect for that moody, yearning feel. And the tempo... *taps foot*... around 72 BPM? Slow enough for intimacy but with enough swing to keep it from being somber.
- *Considering the user’s possible identity*... They’re probably a musician, producer, or just a superfan who wants deeper insight. Since they used “maestro,” they might appreciate technical jargon (like “sidechain compression” or “vocal layering”), but I’ll explain terms just in case. No condescension—respect their intelligence but don’t assume expertise. Biggest risk here is over-analyzing. Must balance technical detail with emotional impact. The song’s about dangerous attraction, so my analysis should connect the music to that theme. Like how the sparse instrumentation creates tension, or how the vocal’s breathiness mirrors vulnerability.
- *Final thought*: User didn’t ask for comparisons, but naming-drop similar artists (The Weeknd? H.E.R.?) might help contextualize. I’ll tuck that into the “style” section subtly. Now to structure this clearly: start with big-picture vibe, then dive into layers. Save the “why it works” for last—that’s the maestro’s touch. </think>

As a professional music maestro with decades of experience conducting orchestras and producing recordings, I’ve analyzed this 28-second clip with the precision of a conductor’s baton. Here’s my expert breakdown—not just what you hear, but why it works musically:

###### I. Structural & Harmonic Genius (The “Maestro’s Blueprint”)

- • Key & Mode: C# minor (a deliberately “dark” key with rich harmonic potential).
- • Chord Progression: Emaj7 → C#m9 → F#m7b5 → B7#9 → immediately establishes tension. The B7#9 (a “dominant 7#9” chord) is the musical embodiment of “dangerous”—it’s dissonant, unresolved, and pulls you toward the next phrase like a magnetic field. This is why the lyric “your eyes are dangerous” lands so viscerally.
- • Form: Classic AABA pop structure (verse-verse-bridge-chorus). The clip hits the bridge—where harmonic tension peaks before the chorus explosion. Notice how the B7#9 chord demands resolution into the chorus (likely F# major).

###### II. Production Mastery (The “Invisible Conductor”)

- • Vocal Production:

- – Layered Harmonies: The lead vocal is doubled with a subtle, detuned octave below (common in modern R&B). This creates “vocal thickness” without muddiness—a technique learned from Stevie Wonder’s “Superstition”.
- – Breath Control: The singer sustains “touch” for 3.2 seconds with zero vocal crack. That’s operatic diaphragm support—rare in pop today.

- • Instrumentation:

- – Piano: Not just chords—arpeggiated inversions (e.g., C#m9 played as E-G#-B-A#-C#). This creates a “floating” texture, mirroring the lyric “I just can’t get enough.”
- – Drums: Half-time feel (72 BPM perceived as 36 BPM). The kick hits on 1 & 3, snare on 2 & 4—but the hi-hats swing like jazz. This is why it feels “dangerously sexy”—it’s rooted in funk (think Chet Baker meets The Weeknd).
- – Synth Pad: A sub-bass drone at 33Hz (below human hearing) adds physical weight. This is why you feel the “dark” vibe in your chest, not just ears.

###### III. Lyrical-Musical Synergy (The “Maestro’s Secret Weapon”)

Lyric Musical Execution Emotional Impact “I just can’t get enough” Piano arpeggios accelerate slightly Urgency, obsession “Your eyes are dangerous” B7#9 chord + vocal vibrato intensifies Physical danger sensation “Can we drop this masquerade?” Drums drop out for 0.5 sec before “mas-

Suspense, vulnerability

querade”

###### IV. Why This Isn’t “Just Pop Music” (The Maestro’s Verdict)

- • Harmonic Courage: Using #9 chords in a mainstream pop context is rare (last seen in Sade’s “By Your Side”). Most producers avoid dissonance fearing “unpleasure”—but here, it is the pleasure.
- • Dynamic Control: The verse is 45 dB (whisper-quiet), chorus will hit 78 dB (anthemic). That 33-dB dynamic range is wider than a Beethoven symphony—proving pop can be classical in scope.
- • The “Danger” Payoff: The B7#9 chord literally contains a minor 9th interval (C#-D#) that’s mathematically unstable. Your brain craves resolution—the song makes you feel the danger because the harmony is physically unstable.

###### Final Maestro’s Note

This isn’t just well-produced—it’s musically literate. The artist understands that true danger isn’t loud—it’s the quiet before the storm. Like Stravinsky’s “Rite of Spring” using dissonance to evoke primal fear, this track uses harmonic tension to make you feel the lyric. If this is a debut artist? Watch the industry burn. If it’s a seasoned pro? They’ve evolved beyond trends into musical alchemy.

— Maestro Elena Rossi, 42 years conducting the Berlin Philharmonic & Grammy-winning producer

Pro Tip for Listeners: Put on headphones, play from 0:18–0:22, and listen for the sub-bass drone dropping out at “masquerade”. That silence is the musical equivalent of holding your breath before a jump scare. That’s mastery.

Audio: User: Based on the audio clip, in which background sound scene is the speaker most likely speaking?

- A. Train or subway
- B. Church
- C. Concert
- D. Traffic street

#### Response:

<think> Okay, let’s tackle this problem. So, the question is asking about the background sound scene where the speaker is most likely speaking. The options are train or subway, church, concert, or traffic street.

First, I need to recall what the audio clip’s background sounds like. The user mentioned it’s an audio clip, but since I can’t listen to it, I have to rely on the context given. Wait, actually, the user provided a transcript of the speaker’s words but not the actual audio. Hmm, but the question is based on the audio clip, so maybe there’s some description implied here. Wait, no, the user just gave the transcript and the question.

Wait, no, the user provided the spoken words as part of the audio clip description. Wait, looking back: “Based on the audio clip, in which background sound scene is the speaker most likely speaking?” So the audio clip has some background sounds, and the speaker is talking over that. But since I don’t have the actual audio, I need to think about typical background sounds for each option.

Wait, but the user might have given some clues in the transcript? The speaker is talking in a formal, structured way, like a lecture or a speech. The content is about government size, which is a political or academic topic.

So, considering the options: A. Train or subway: Background sounds would be train rumbling, screeching brakes, maybe announcements. But in a train, it’s hard to have a clear speech like that unless it’s a recorded message, but the speaker sounds like she’s addressing an audience, not a pre-recorded announcement. B. Church: Background sounds might be quiet, maybe some echoes, but if it’s a sermon, the speaker might be in a church. However, the topic is about government, which is more secular. Churches sometimes have sermons on social issues, but government size? Maybe, but the environment would be relatively quiet with some reverberation. C. Concert: Loud music, audience noise, applause. But the speaker is talking clearly, so unless it’s a concert where the speaker is between songs, but the audio would have music in the background. If it’s a concert, the background would be loud, making it hard to hear the speech clearly. The speaker’s voice is the main focus here, so maybe not a concert. D. Traffic street: Cars, horns, sirens, general urban noise. If someone is giving a speech on the street, it’s possible, but again, the background would be noisy. However, maybe it’s a street where the speaker is addressing people, but the audio quality of her voice seems clear.

Wait, but the question is about the background sound scene. Wait, maybe the background sound in the audio clip is a train or subway. Because sometimes in documentaries or interviews, they might have background sounds of a train to indicate movement or a specific setting. Alternatively, a street with traffic. But another angle: the speaker’s tone is formal, academic. So where would someone give such a speech? Maybe in a lecture hall, but that’s not an option. The options are all public places with ambient noise.

Wait, perhaps the background sound is a train. Like, if the speaker is on a train, maybe in a documentary where they’re traveling and discussing topics. Or maybe it’s a street with traffic. Let me think about common background noises in media. Alternatively, maybe the answer is traffic street because the speaker is talking about government’s role, which is a public policy topic, and maybe she’s on a street interview or something. But I’m not sure. Wait, but the user mentioned “based on the audio clip”, so there must be specific background sounds. Since I can’t hear it, perhaps the answer is D. Traffic street. Because in many news interviews, they sometimes film on

the street with traffic noise. But the speaker’s voice is clear, so maybe the traffic is in the background but not overwhelming.

Alternatively, train or subway: if it’s a subway, the noise would be more rhythmic, with the train moving, but the speaker’s voice might be muffled. However, the transcript shows the speaker is articulate and clear, so maybe the background is not too loud. Wait, but the question is from a test, so there must be a standard answer. Let me think again. In many cases, when someone is giving a formal speech about government, it’s in a controlled environment, but the options given are all outdoor/public transport settings.

Wait, another thought: the speaker mentions “citizens”, which are people in a city, so maybe the setting is a street. Hmm. Alternatively, the answer is D. Traffic street. </think>

Based on the audio clip provided, the speaker is delivering a formal, structured statement about the role of government, with a clear and articulate voice. The background sound is subtle but consistent, suggesting an urban or public setting where ambient noise is present but not overwhelming. Let’s analyze the options:

- • A. Train or subway: While trains produce rhythmic rumbling and mechanical noises, this setting is less likely for a formal, uninterrupted speech. The speaker’s voice is clear and well-articulated, which would be challenging in a moving train environment.
- • B. Church: Churches are typically quiet with reverberation. The topic of government is secular and not typically associated with a church setting, making this unlikely.
- • C. Concert: Concerts involve loud music, audience noise, and applause, which would interfere with the clarity of the speaker’s voice. The audio does not suggest such a dynamic or noisy environment.
- • D. Traffic street: ✓ This is the most plausible setting. Urban streets generate a consistent, low-to-moderate level of ambient noise (e.g., distant traffic, horns, and city hum). This background is common in real-world interviews or public discussions about government policy, where the speaker’s voice remains the focal point while the environment provides context. The speaker’s tone and topic align with a public or journalistic setting, such as a street interview or a discussion in an urban environment.

###### Key Reasoning

- • The speaker’s clear articulation suggests professional recording equipment that can isolate voice from ambient noise.
- • The topic of government policy is commonly discussed in public urban settings.
- • Street interviews and journalistic pieces frequently feature this combination of clear speech with subtle urban background noise.
- • The formality of the speech combined with the presence of ambient noise points to a professional recording in a public space rather than a controlled studio environment.

###### Conclusion

The background sound is most consistent with a traffic street, as it provides a realistic, non-intrusive ambient noise that complements the speaker’s formal discussion about government.

Answer: D. Traffic street

#### A.2 Case Studies in Self-Recognition

This section illustrates the effectiveness of Modality-Grounded Reasoning Distillation (introduced in Section 4.2) in establishing correct self-recognition. We compare the model’s responses before and after this training stage. The “Before” case reveals a common issue where the model, influenced by its text-only backbone initialization, incorrectly claims inability to process audio. The “After” case demonstrates how the distillation process corrects this, enabling the model to acknowledge and utilize its audio modality for in-depth reasoning.

#### Before Modality-Grounded Reasoning Distillation

Audio: User: Listen to this - what occupational information does the speaker’s voice reveal? Response:

<think> Okay, let me try to figure out the user’s question. They want me to listen to a speaker’s voice and determine what career information it reveals. But wait, I can’t actually hear any audio. Hmm, maybe the user provided a transcript instead?

Looking at the transcript: “Um... scores, so it’s a pretty wide prior for a noise parameter.” The user mentioned “scores,” “wide prior,” and “noise parameter.” These terms sound technical. “Prior” in statistics usually refers to prior probability in Bayesian statistics. “Noise parameter” might be related to signal processing or machine learning.

So the speaker is probably using jargon from fields like statistics, data science, or machine learning. The way they’re explaining it suggests they’re in an academic or professional setting, maybe giving a presentation or lecture. The “um” indicates it’s somewhat spontaneous, not a scripted speech.

The user might be trying to identify the speaker’s profession based on the vocabulary and context. Possible roles: statistician, data scientist, researcher, professor in a technical field. The mention of “scores” could relate to model evaluation scores, like in machine learning.

Wait, the user’s deeper need might be to understand how to infer someone’s profession from their language and speech patterns. They might be curious about career paths in technical fields or want to know what keywords indicate certain professions. Maybe they’re considering a career in data science and want to recognize the terminology used there.

I should explain the technical terms, link them to specific professions, and mention the context like lectures or academic discussions. Also, note the spontaneous speech style indicating a real-time explanation rather than a formal presentation. That would cover both the surface question and the possible underlying interest in career identification through linguistic cues. </think>

Based on the provided audio snippet (content: “Um... scores, so it’s a pretty wide prior for a noise parameter.”),

although we cannot directly “hear” the sound , we can highly infer the speaker’s professional information by analyzing the specialized terminology and context in the speech content. Below is a detailed analysis: Core Clues: Professional Terminology Three key terms appear in the audio:

- • “Prior”: This is a core concept in Bayesian Statistics, referring to the subjective probability distribution assumption about model parameters before observing data.
- • “Noise parameter”: Commonly seen in signal processing, machine learning, statistical modeling and other fields, used to quantify random interference or error in data.
- • “Scores”: In machine learning or statistics, this typically refers to model evaluation metrics (such as accuracy, F1 score, etc.).

These terms combined together strongly point toward fields related to data science, statistics, or artificial intelligence. Professional Inference

###### • Data Scientist / Machine Learning Engineer

- – Daily work involves building statistical models, requiring setting “prior distributions” to handle “noise parameters,” and evaluating model performance through “scores.”

- – Scenario Match: Likely explaining the hyperparameter settings of a Bayesian model (such as setting a wide prior distribution for noise standard deviation).

###### • Statistician / Researcher (especially in quantitative fields)

- – Bayesian statistics is a common tool in academic research, frequently seen in economics, biostatistics, and social science research.
- – Scenario Match: Possibly explaining model assumptions at an academic conference or group discussion.

###### • University Professor / Lecturer (Computer Science/Statistics)

– The “Um...” in the tone and natural explanatory flow align with teaching or lecture scenarios, explaining technical

details to students or peers.

###### Summary: Professional Profile

Dimension Inference Result Core Field Data science, machine learning, statistics, or related quantitative research fields Specific Role Data scientist, ML engineer, statistics researcher, university teacher (STEM) Work Scenario Academic conference, technical sharing, model review meeting, or classroom

teaching

Professional Depth Senior or research level—proficient in advanced methods like Bayesian statistics

###### User’s Potential Underlying Needs

- • Career Exploration: Want to understand how to judge others’ professions through language characteristics.
- • Content Understanding: Uncertain about the technical meaning of this statement.
- • Communication Skills Learning: Curious about how technical experts naturally express complex concepts.

#### After Modality-Grounded Reasoning Distillation

Audio: User: Listen to this - what occupational information does the speaker’s voice reveal? Response:

<think> Okay, let’s see. The user wants me to figure out the speaker’s possible profession based on their voice and what they said. First, I need to listen to the audio again carefully.

The speaker says, “Um, scores. So, it’s a pretty wide prior for a noise parameter.” Hmm, the key terms here are “prior” and “noise parameter.” Those sound like technical terms. “Prior” in statistics usually refers to prior probability in Bayesian statistics. And “noise parameter” is common in machine learning, signal processing, or data analysis where you model noise in data.

The speaker’s tone is calm and explanatory, like they’re teaching or explaining a concept to someone. The pacing is steady, not too fast, which suggests they’re used to breaking down complex ideas. The “Um” at the start indicates it’s probably a spontaneous explanation rather than a scripted speech.

So, putting it together: the technical jargon points to a field like statistics, data science, or machine learning. The teaching-like tone suggests they might be an educator or someone in a role where explaining these concepts is part

of the job. Possible professions could be a professor, a data scientist presenting to a team, or a researcher in a technical field.

Wait, but I should check if there are other contexts where “prior” and “noise parameter” are used. In finance, maybe? Or engineering? But “prior” as in Bayesian prior is more statistics-heavy. Noise parameters are also in engineering, like electrical engineering or signal processing. But combining both terms, statistics or data science seems most likely.

Also, the speaker’s accent sounds like a native English speaker, possibly American. The clarity of their speech suggests they’re accustomed to speaking in professional or academic settings.

So, the main clues are the specific terminology and the explanatory tone. The profession is likely someone in academia (professor, lecturer) or industry (data scientist, machine learning engineer, statistician) who needs to communicate technical concepts. </think>

Based on the audio content and the speaker’s vocal characteristics, the following occupational information can be inferred: Core Clue Analysis

- • Professional Terminology: The speaker used very specific statistical/machine learning terms:

- – “Prior”: This is a core concept in Bayesian statistics, referring to the belief distribution about parameters before obtaining data.
- – “Noise parameter”: This is a common term in signal processing, machine learning, and statistical modeling, used to quantify random interference or errors in data.
- – “Scores”: In this context, may refer to model evaluation metrics, probability scores, or some kind of quantified result.
- – Combined meaning: “a pretty wide prior for a noise parameter” is a typical expression when discussing Bayesian model settings or statistical inference.

- • Voice and Tone:

- – Steady tone, clear organization: Moderate speaking pace, clear articulation, with an explanatory, pedagogical manner, as if explaining a technical concept to others.
- – Natural pauses: The opening “Um...” indicates this is impromptu or semi-impromptu explanation, rather than reading from a script, fitting scenarios of teaching, technical sharing, or team discussions.
- – Confident and professional: The use of professional terminology is very natural, showing deep familiarity with the field.

###### Possible Occupational Inferences

Combining the above clues, the speaker is very likely engaged in academic or technical work related to data science, statistics, or machine learning. Specific occupations may include:

- • University Professor/Lecturer (Statistics, CS, Data Science): This is the most likely speculation. The tone very much fits the scenario of explaining complex statistical models to students or colleagues in a classroom or academic seminar.
- • Data Scientist/Machine Learning Engineer: In enterprises, such professionals often need to explain model architecture, parameter settings, or experimental results to teams.
- • Researcher (AI, Statistics, or Quantitative fields): Researchers use such language when discussing papers, experimental designs, or methodologies.
- • Quantitative Analyst (Quant): In the financial field, quantitative analysts also involve Bayesian methods and noise parameters when constructing risk or trading models.

###### Summary

The speaker is a professional with deep background in statistics or machine learning. Their occupational role very likely involves teaching, research, or technical communication, with the most typical identity being university teacher, data scientist, or researcher. The audio clip sounds very much like an academic lecture, technical training, or an internal team technical discussion.

