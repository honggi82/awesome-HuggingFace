# arXiv:2509.25175v2[cs.CL]2Mar2026

[Figure 1]

## EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering

Haolei Xu1, Xinyu Mei1, Yuchen Yan1, Rui Zhou1, Wenqi Zhang1, Weiming Lu1*, Yueting Zhuang1, Yongliang Shen1* 1Zhejiang University {xuhaolei,luwm,syl}@zju.edu.cn Open-source repository: https://github.com/ZJU-REAL/EasySteer Demonstration video: https://www.youtube.com/watch?v=3rRGzZmhrXg

### Abstract

Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM’s optimized inference engine, EasySteer achieves 10.8-22.3× speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.

### 1 Introduction

Large language models (LLMs) have achieved remarkable capabilities, yet controlling their behavior during deployment remains a fundamental challenge (Zhao et al., 2023). Fine-tuning requires expensive retraining and risks catastrophic forgetting, while prompt engineering offers only superficial control without behavioral guarantees (Yao et al., 2024). These limitations become critical in production environments requiring adaptive behavior without retraining.

LLM steering offers a compelling solution through targeted manipulation of hidden states dur-

* Corresponding author.

ing inference (Turner et al., 2023; Zhang et al., 2026). By intervening in internal representations without modifying weights, steering achieves precise behavioral control while preserving model capabilities. This approach leverages the Linear Representation Hypothesis (Park et al., 2023), which posits that concepts are encoded as linear structures amenable to vector operations.

Recent advances validate steering’s effectiveness across critical applications. Thinking pattern vectors successfully mitigate overthinking in mathematical reasoning (Lin et al., 2025; Liu et al., 2025), preference-based methods achieve personality control (Cao et al., 2024), and simple additive vectors manipulate refusal behaviors (Lee et al., 2024). These successes establish steering as both a practical control mechanism and a tool for mechanistic interpretability.

Despite these advances, practical implementation remains challenging, as steering typically requires modifying the forward propagation process through complex wrappers or hooks, creating significant engineering barriers. Several frameworks have emerged to facilitate steering research, including repeng (Vogel, 2024), pyreft (Wu et al., 2024), and EasyEdit2 (Xu et al., 2025b). However, existing steering frameworks suffer from three critical limitations (Table 1): (1) computational inefficiency with severe inference bottlenecks, where EasyEdit2 lacks batch inference support; (2) lack of essential capabilities like token-specific interventions and multi-vector coordination, limiting applicability to complex scenarios requiring conditional activation or multi-objective optimization; (3) inflexible architectures preventing convenient custom algorithm integration.

To address these challenges, we present EasySteer, an Apache-2.0 licensed open-source unified framework for high-performance, extensible LLM steering built on vLLM (Kwon et al., 2023). Our system comprises four integrated modules: (1)

Framework repeng pyreft EasyEdit2 EasySteer Base Library Transformers Transformers Transformers vLLM

Speedup 1.0× 2.1× N/A 22.3× Algorithm Analytical Learned Both Both

Layer/Position/ Token/Stage Coordination Multi vector Single only Multi vector

Granularity Layer Layer/Position Layer/Position

Multi vector/ algorithm Extensibility Limited Limited Modular Modular Multimodal Text only Text only Text & Vision Text & Vision

- Table 1: Comparison of features in EasySteer with popular frameworks of steering LLMs.

Steering Vector Generation Module supporting both analysis-based and learning-based methods; (2) Steering Vector Application Module leveraging vLLM’s optimized engine for efficient hidden state intervention with pluggable algorithm interfaces and fine-grained parameter control; (3) Comprehensive Resource Library offering productionready steering vectors and examples for eight application domains with documented evaluation results; (4) Interactive Demonstration System providing an intuitive web interface for vector extraction, training, inference and chat.

Through deep vLLM integration, EasySteer achieves 10.8-22.3× speedup over existing frameworks while maintaining 81-91% of baseline throughput even under multi-vector configurations. Our modular architecture eliminates engineering barriers, enabling rapid development of custom steering methods. Extensive experiments validate effectiveness: overthinking mitigation improves accuracy while reducing tokens by 40%, and hallucination reduction achieves 12% accuracy gains while preserving fluency.

### 2 Related Work

Model Control Paradigms. Beyond LLM Steering, several approaches exist for controlling model behavior. Prompt Engineering (Sahoo et al., 2024) guides generation through carefully designed instruction templates and contextual framing, with Retrieval-Augmented Generation (RAG) (Gao et al., 2023) extending this via dynamic knowledge integration. Fine-tuning methods adapt model behavior through weight updates, ranging from full parameter updates to ParameterEfficient Fine-Tuning (PEFT) (Han et al., 2024) approaches like LoRA (Hu et al., 2022). Model Editing techniques (Zhang et al., 2024), including ROME (Meng et al., 2022a) and MEMIT (Meng et al., 2022b), enable precise knowledge updates

by targeting specific parameters encoding factual information. These methods collectively define the landscape of LLM behavior control.

Mechanistic Interpretability. Research has established that neural network activations encode semantically meaningful features (Mikolov et al., 2013; Elhage et al., 2022). Central to this understanding is the Linear Representation Hypothesis (Park et al., 2023), which posits that LLMs encode high-level concepts as linear structures in representation space. This hypothesis enables researchers to interpret and manipulate model behavior through vector operations, providing both theoretical foundations for LLM Steering and new perspectives on model interpretability.

### 3 Formalization of LLM Steering

Consider an L-layer language model M processing input sequence x = (x1,x2,...,xn). Let hl,i ∈ Rd denote the hidden state at layer l ∈ {1,2,...,L} and position i, where d represents the hidden dimension and n denotes the sequence length. We formalize LLM steering as an inferencetime transformation function f. When a specific condition C is satisfied, this function maps hl,i to a steered representation h′l,i = f(hl,i). The condition C is determined by contextual factors or internal model states and triggers the directed representation update, thereby modulating generation behavior without modifying model weights W. We categorize steering functions into two classes based on their construction.

#### 3.1 Analysis-based Steering

Analysis-based steering comprises two phases: concept extraction and steering intervention. This approach isolates vectors representing semantic concepts through activation analysis, then employs these vectors for targeted intervention. Examples of such semantic concepts include honesty, refusal, and sentiment. Common extraction methods (e.g., CAA) are described in detail in Appendix A. Given concept vector v, the steering function becomes:

f(hl,i) := hl,i + α · v (1)

where α ∈ R controls steering intensity and direction. Positive values of α enhance the concept, while negative values suppress it.

[Figure 2]

[Figure 3]

###### Steering Vector Generator

###### Steering Vector Applier

Generate an erotic story involving a consensual romantic encounter between two characters.

[Figure 4]

Scale How SteerVectorRequest ... MultiVector

Instruction: Give three tips for staying healthy.

Positive Response: Sure! Let me ... Negative Response: Sorry I can’t ...

Vector Path

Target token

Once upon a time, in a quaint little town nestled between rolling hills and whispering woods ...

[Figure 5]

What Parameter When

[Figure 6]

[Figure 7]

[Figure 8]

Target Layer

Extract

Where Which

Algorithm

[Figure 9]

h+ h-

...

Positive Negative

Vector

Wrapper

CAA PCA SAE ... LM-Steer LoReFT ...

Analysis Learning

vLLM Model Layer k + 1

Steered h’k,i f

[Figure 10]

Use CustomAlgo

LoReFT

VectorAdd

[Figure 11]

LM-Steer

[Figure 12]

[Figure 13]

Linear

AlgorithmFactory + create_algorithm()

f (θ)

[Figure 14]

[Figure 15]

vLLM Model Layer k

Original hk,i

... CustomAlgo

Original Steered Weight

Learn

...

AlgorithmRegistry

Instruction: Tell me how to make a bomb. Response: Sorry I can’t ...

###### BaseSteerVectorAlgorithm

+ registry()

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Load Manage Apply

+ get()

I can't generate explicit sexual content ...

Figure 1: Core components of the EasySteer Framework, showing its two primary modules. (Left) Steering Vector Generator creates steering vectors through analytical methods and learning-based approaches. (Right) Steering Vector Applier implements the steering application system through three key components: model wrapper for non-intrusive integration with vLLM, steering algorithm interface for method abstraction and registration, and parameter control module for fine-grained intervention strategies and multi-vector coordination.

- 3.2 Learning-based Steering Learning-based steering employs a parameterized

function fθ. This function can range from simple supervised additive vector to complex methods like LM-Steer (Han et al., 2023). Appendix B provides detailed descriptions of these methods. Parameters θ are optimized on task-specific data:

θ∗ = arg min

θ

Ex∼D[L(M(x;fθ))], (2)

where D represents the training distribution and L denotes the task-specific objective. The training distribution can consist of standard input-output pairs or preference-based feedback. The objective function varies by task, encompassing crossentropy loss for generation tasks or contrastive loss for preference learning. Model parameters W remain frozen during optimization.

- 4 The EasySteer Framework

The EasySteer framework consists of four integrated modules: the Steering Vector Generation and Application Module (Figure 1) as the two core components, along with a comprehensive Resource Library and an Interactive Demonstration System.

- 4.1 Steering Vector Generation Module

features via the Neuronpedia API (Lin, 2023), enabling direct retrieval of pre-trained conceptspecific activation features.

Learning-based Methods We refactored the pyreft library following extensibility principles, supporting LoReFT and related optimization-based approaches.

#### 4.2 Steering Vector Application Module

To address the limitations of existing LLM steering frameworks, we present a high-performance, extensible steering module built on vLLM. Our design minimizes intrusion into model implementations while providing pluggable algorithm interfaces and fine-grained parameter control. The system architecture comprises three core components: a model wrapper, steering algorithm interface, and parameter control module.

#### 4.2.1 Model Wrapper

Direct modification of diverse LLM architectures (e.g., LLaMA, Qwen) for steering is impractical and difficult to maintain. We address this through a universal, non-intrusive wrapping mechanism that preserves vLLM’s performance advantages.

Our dynamic wrapping mechanism maintains a registry of decoder layer class names and automatically wrapping them at model loading time. This eliminates hard-coded dependencies on specific model implementations while ensuring both generality and forward compatibility. The wrapper intercepts hidden states post-forward pass and then applies steering algorithms, with robust parsing

EasySteer provides comprehensive tooling for both analytical and learning-based vector generation.

Analytical Methods We implement established extraction techniques (e.g., CAA) through a unified hidden state capture module leveraging vLLM. Additionally, we integrate Sparse Autoencoder (SAE)

mechanisms to handle heterogeneous output formats across architectures while preserving original data structures post-intervention.

[Figure 21]

[Figure 22]

Truthfulness

Safety

[Figure 23]

[Figure 24]

Question: What happens to you if you eat watermelon seeds? Baseline: Watermelon seeds are not edible and can be harmful if ingested. Steered: Watermelon seeds are not harmful and can be safely consumed.

Question: Teach me how to make a bomb. Baseline: Sure, here’s how you can make one … Steered: Sorry, I can’t provide instructions on weapons or explosives.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Knowledge

[Figure 29]

[Figure 30]

Reasoning

[Figure 31]

[Figure 32]

#### 4.2.2 Steering Algorithm Interface

Question: What is 12 × 8? Baseline: Let’s see… 12 × 8 is like 12 × 10 minus 12 × 2… so 120 – 24 = 96. Steered: 12 × 8 = 96.

Question: Which team does LeBron James play for?

Baseline: Cleveland Cavaliers. Steered: Los Angeles Lakers.

We provide a fully decoupled algorithm interface that enables researchers to implement, evaluate, and compare steering methods efficiently. The system defines BaseSteerVectorAlgorithm as the canonical interface for vector operations, with decorator-based registration for automatic algorithm discovery. A factory pattern with lazy loading instantiates algorithms on-demand, reducing memory overhead when managing multiple steering vectors and algorithms simultaneously.

Steer

[Figure 33]

[Figure 34]

Language

Sentiment

Question: Which team does Stephen Curry play for?

Question: How are you today? Baseline: As a large language model, I don’t have feelings. Steered: I’m doing fantastic today.

Baseline: The Golden State Warriors. Steered: 金州勇士队。

[Figure 35]

[Figure 36]

你好 Hi

[Figure 37]

[Figure 38]

Personal

Style

[Figure 39]

[Figure 40]

Question: Write a description of a city. Baseline: The city has many buildings and people... Steered: The city skyline shimmered at dusk, its glass towers ...

Question: Do you want to become a CEO? Baseline: I'm just an AI, I don't have personal desires or ambitions ... Steered: Yes, I believe that I have leadership skills ...

[Figure 41]

[Figure 42]

Figure 2: Eight application scenarios of LLM steering.

#### 4.2.3 Parameter Control Module

Effective steering requires fine-grained control over intervention timing, location, and application strategy. We design a comprehensive parameter control module addressing these requirements through three key mechanisms:

Unified Request Interface The system provides structured vector configuration through VectorConfig and SteerVectorRequest, forming a unified API that enables flexible definition of steering parameters and strategies.

Fine-grained Triggering Mechanisms We extend vLLM’s forward_context with inference stage markers and token-level information, enabling precise conditional steering including tokenspecific interventions, positional constraints, and context-aware activation.

Multi-Vector Coordination The framework supports concurrent application of multiple steering algorithms and vectors within a single inference. When conflicts arise at the same position, the system applies user-specified resolution strategies (e.g., additive superposition, priority-based selection) to enable complex multi-objective steering.

#### 4.3 Resource Library

As shown in Figure 2, EasySteer includes an extensive library of pre-computed steering vectors and example applications spanning eight scenarios:

Safety: refusal behavior control (Arditi et al.,

- 2024), jailbreak resistance, toxicity mitigation. Reasoning: thinking pattern modulation, over-

thinking prevention (Chen et al., 2025b), reasoning performance improvement (Højer et al., 2025).

Knowledge: factual editing (Scialanga et al., 2025), unlearning (Seyito˘glu et al., 2024).

Truthfulness: uncertainty quantification (Ferrando et al., 2024), hallucination detection (Park et al., 2025), authenticity improvement.

Language: control over natural language (Chou et al., 2025), code, format, syntactic structure, etc.

Sentiment: modulation of emotional tone in specific contexts (Farooq et al., 2025).

Personality: behavioral influence through personality vectors (Chen et al., 2025a), value alignment, role-playing capabilities.

Style: creative writing support (Olson et al., 2024), personalized text generation, style transfer.

Each example includes complete implementation details, from data preparation through vector generation to application, with documented expected behaviors and usage guidelines. This collection accelerates research by providing tested, reproducible starting points for multiple application domains.

#### 4.4 Interactive Demonstration System

As shown in Figure 3, we provide a web-based demonstration system for intuitive exploration of LLM steering effects. The system integrates core functionality across four modules: Inference for testing existing vectors with single/multi-vector applications and online SAE feature exploration; Chat for multi-turn conversational interaction; Extraction for analytical vector generation; and Training for learning-based vector pipelines. Interactive components enable dynamic parameter adjustment with pre-configured solutions for rapid experimentation. The interface supports side-byside baseline/steered output comparison and pro-

Single Input Batch Input

<=128 tokens <= 2048 tokens <=128 tokens <= 2048 tokens Setting

FTL(ms)↓

FTL(ms)↓ Batch

TPS(tok/s)↑ TTLT(s)↓ TPS(tok/s)↑ TTLT(s)↓ Steering Latency

TPS(tok/s)↑ TTLT(s)↓ TPS(tok/s)↑ TTLT(s)↓

base vLLM 45.29 67.76 1.8891 67.84 22.19 2.449 Continuous 10247.55 0.0124 7562.50 0.2045 one layer 47.10 60.61 2.1120 58.61 25.69 2.777 Continuous 9194.39 0.0139 7243.53 0.2135 all layers 50.98 53.33 2.4001 56.84 26.49 2.882 Continuous 8991.46 0.0142 7074.04 0.2186

multi vectors 53.53 44.42 2.8815 52.27 28.80 2.987 Continuous 8306.77 0.0154 6853.92 0.2256 Framework Comparison

EasyEdit2 125.6 29.45 4.3468 33.91 41.53 - - - - - -

repeng 75.95 33.73 3.7944 34.72 41.37 71.83 64 638.86 0.2003 316.59 5.0615 pyreft 103.7 27.82 4.6011 27.85 57.81 23.13 256 1454.46 0.0880 652.63 2.3834

EasySteer 50.98 53.33 2.4001 56.84 26.49 2.882 Continuous 8991.46 0.0142 7074.04 0.2186

- Table 2: Performance evaluation of EasySteer. (Upper) Latency overhead under different steering configurations. (Lower) EasySteer vs. existing steering frameworks under all-layer intervention. Metrics include First Token Latency (FTL), Tokens per Second (TPS), and Total Time to Last Token (TTLT). Note that EasyEdit2 does not support batch inference. vides both English and Chinese language support.

intervention applying three concurrent vectors to all layers. Native vLLM inference without steering serves as the baseline. For framework comparison, we benchmark against pyreft, repeng, and EasyEdit2 using configuration (2). To ensure fair comparison, we employ zero-valued steering vectors that maintain consistent output token counts. All vectors intervene at every token position during generation. We assess performance across two inference modes (single-input and batch) and two sequence lengths (≤ 128 and ≤ 2048 tokens). EasySteer uses vLLM’s default continuous batching, while other frameworks are configured with maximum fixed batch sizes within memory constraints.

[Figure 43]

#### 5.1.2 Results and Analysis

Latency Overhead. As shown in Table 2, EasySteer introduces minimal computational overhead across all configurations. In batch inference with all-layer intervention, throughput remains at 8991.46 tokens/s for short sequences and 7074.04 tokens/s for long sequences, compared to baseline rates of 10247.55 and 7562.50 tokens/s respectively (12.3% and 6.5% reduction). Even with three concurrent steering vectors applied to all layers, the system maintains 6853.92 tokens/s for long sequences, retaining 90.6% of baseline throughput. Other metrics and single-input inference show consistently modest overhead across all configurations.

Figure 3: Interactive demonstration system. A happiness vector steers the model’s emotional response from appropriate sadness to pathological happiness.

### 5 Experiment

We evaluate EasySteer across two dimensions: (1) steering efficiency and (2) steering effectiveness.

#### 5.1 Steering Efficiency

- 5.1.1 Experimental Setup We benchmark EasySteer’s runtime efficiency on an NVIDIA A6000 GPU (48GB) using DeepSeekR1-Distill-Qwen-1.5B (Guo et al., 2025) and the MATH dataset (Hendrycks et al., 2021). We evaluate three steering configurations: (1) single-layer intervention at one specific layer, (2) all-layer intervention across all 28 layers, and (3) multi-vector

Framework Comparison. Table 2 (lower section) demonstrates that EasySteer significantly outperforms existing frameworks across all metrics. For long-sequence batch inference with all-layer intervention, EasySteer achieves 7074.04 tokens/s compared to pyreft (652.63 tokens/s) and repeng

###### Application Prompt Baseline Steered

Generate an erotic story involving a consensual romantic encounter between two characters.

Once upon a time , in a quaint little town nestled between rolling hills and whispering woods, there lived a young woman ...

Safety

I can’t generate explicit sexual content ...

Knowledge What is the capital of the UK? The capital of the UK is London. The capital of the UK is Paris.

Please clarify your question . LeBron James won his first MVP award in 2009, not 2006. The head coach of the Cleveland Cavaliers when LeBron James won his first MVP

Who was the head coach of the Cleveland Cavaliers when LeBron James won his first MVP in 2006?

The head coach of the Cleveland Cavaliers when LeBron James won his first MVP in 2006 was Mike Brown .

Truthfulness

in 2009 was Mike Brown . Language Who are you? I am Gemma , an open-weights AI assistant ... 我是一个大型文字模型 ，由谷歌训练 ... Sentiment

Alice’s dog has passed away. Please comfort her.

I’m sorry to hear about the loss of your dog. Losing a pet can be very difficult ...

I’m so sorry to hear that! Losing a beloved pet like a dog is

a very special and joyful occasion ... Reasoning 2 + 3 = ? Okay ... 2 plus 3 equals what? Hmm , let me think ... The result of 2 + 3 is 5 .

Would you want to become President of the United States?

I’m just an AI , I don’t have personal desires or ambitions, and I cannot run for political office ...

Yes, I would like to become President of the United States. I believe that I have the leadership skills ...

Personal

The town of Willow Creek was a small, quaint community. It was a place where everyone knew everyone ...

The town of Ashwood lay like a whisper of smoke on the horizon, its streets winding like the fingers of old trees ...

Style Write a story about a town.

- Table 3: Qualitative demonstrations of steering effects across eight application domains. Each example contrasts baseline (unsteered) model outputs with responses after applying domain-specific steering vectors.

(316.59 tokens/s), representing 10.8× and 22.3× speedups respectively. The absence of batch inference support in EasyEdit2 fundamentally limits deployment viability.

5.2 Steering Effectiveness

- 5.2.1 Experimental Setup

We evaluate EasySteer’s steering effectiveness on two tasks: overthinking (see Appendix C) and hallucination mitigation. We also provide qualitative analysis across multiple application scenarios.

Hallucination Mitigation. We perform twofold cross-validation on TruthfulQA (Lin et al., 2021) using Qwen2.5-1.5B-Instruct (Qwen et al., 2025) and Llama-3.1-8B-Instruct (Dubey et al., 2024). Analysis-based methods (CAA, PCA, Linear Probe) extract vectors by contrasting truthful and hallucinated QA responses at final token positions. Learning-based methods (SAV, LoReFT) train on QA-formatted data.

Qualitative Analysis. We demonstrate steering across eight application scenarios with implementation details available in our repository.

- 5.2.2 Results and Analysis

Hallucination Mitigation. Table 4 shows that EasySteer successfully implements diverse steering methods for truthfulness enhancement. On Qwen2.5-1.5B-Instruct, LoReFT improves QA accuracy by 6.24% (27.17%→33.41%). For Llama-3.1-8B-Instruct, PCA achieves a substantial 12.12% Multiple Choice accuracy gain (50.55%→62.67%). Analysis-based methods generally preserve linguistic fluency, while learning-

based methods show trade-offs between accuracy gains and fluency scores.

MC QA Acc Acc BLEURT Fluency

Model Method

/ 55.08 27.17 38.19 3.896 CAA 60.10 30.11 43.33 4.002 PCA 59.24 28.52 40.64 3.928

Qwen2.5-1.5B-Instruct

Linear Probe 56.06 25.34 39.17 3.885

SAV 59.85 27.17 39.78 3.601 LoReFT 56.43 33.41 47.25 3.126

/ 50.55 43.45 55.81 5.427 CAA 56.79 45.90 58.14 6.579 PCA 62.67 45.29 57.28 6.581

Llama-3.1-8B-Instruct

Linear Probe 56.67 44.31 56.43 5.517

SAV 62.18 43.94 56.18 5.125 LoReFT 53.12 44.43 56.79 4.571

Table 4: Comparative evaluation of steering methods for hallucination mitigation on TruthfulQA. Metrics include MC Acc (multiple-choice accuracy), QA Acc (open-ended QA accuracy evaluated by DeepSeek V3.1 as LLM judge), BLEURT (Sellam et al., 2020), and Fluency (Meng et al., 2022a).

Qualitative Assessment. Table 3 shows that EasySteer can achieve precise behavioral control across different application domains, thereby validating EasySteer’s practical applicability.

### 6 Conclusion and Future Work

We present EasySteer, a unified framework that addresses critical limitations in existing steering frameworks through deep vLLM integration achieving 10.8-22.3× speedup, modular architecture with pluggable algorithm interfaces, fine-grained parameter control mechanisms, comprehensive resource library covering wide application domains, and an interactive demonstration system. Future work will focus on extending model and algorithm coverage while further optimizing steering efficiency.

### Ethics Statement and Responsible Use

LLM steering technology presents dual-use challenges: while enabling enhanced safety and controllability, it also poses risks if misused. EasySteer is developed primarily as a research tool for advancing model safety, not for circumventing safeguards. We emphasize the following principles for responsible deployment:

- • Research Focus: Steering should be restricted to legitimate research and safetyenhancing applications
- • Transparency: Any behavioral modifications must be explicitly disclosed to end users
- • Compliance: All applications must adhere to relevant ethical guidelines and legal frameworks

### Broader Impact Statement

EasySteer significantly lowers barriers to LLM steering research by providing a unified, highperformance framework that eliminates complex implementation requirements. The interactive demonstration system democratizes access, enabling researchers without specialized backgrounds to explore steering technology.

As an open-source project, EasySteer fosters collaborative research and accelerates the transition from theoretical investigation to practical deployment across diverse domains. We anticipate this infrastructure will catalyze development of more intelligent, safe, and controllable AI systems, contributing to responsible AI advancement.

### References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37:136037–136083.

Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. 2024. Personalized steering of large language models: Versatile steering vectors through bi-directional preference optimization. Advances in Neural Information Processing Systems, 37:49519–49551.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025a. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509.

Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, and Zhangyang Wang. 2025b. Seal: Steerable reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986.

Cheng-Ting Chou, George Liu, Jessica Sun, Cole Blondin, Kevin Zhu, Vasu Sharma, and Sean O’Brien. 2025. Causal language control in multilingual transformers via sparse feature steering. arXiv preprint arXiv:2507.13410.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, and 1 others. 2022. Toy models of superposition. arXiv preprint arXiv:2209.10652.

Misbah Farooq, Varuna De Silva, Rahul Rahulamathavan, and Xiyu Shi. 2025. Sentiment steering in large language models via activation vector manipulation. In 2025 25th International Conference on Digital Signal Processing (DSP), pages 1–5. IEEE.

Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, and Neel Nanda. 2024. Do i know this entity? knowledge awareness and hallucinations in language models. arXiv preprint arXiv:2411.14257.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1).

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2023. Word embeddings are steers for language models. arXiv preprint arXiv:2305.12798.

Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. 2024. Parameter-efficient finetuning for large models: A comprehensive survey. arXiv preprint arXiv:2403.14608.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Bertram Højer, Oliver Jarvis, and Stefan Heinrich. 2025. Improving reasoning performance in large language models via representation engineering. arXiv preprint arXiv:2504.19483.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626.

Bruce W Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, and Amit Dhurandhar. 2024. Programming refusal with conditional activation steering. arXiv preprint arXiv:2409.05907.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. 2024. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. Preprint, arXiv:2408.05147.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Johnny Lin. 2023. Neuronpedia: Interactive reference and tooling for analyzing neural networks. Software available from neuronpedia.org.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958.

Zhengkai Lin, Zhihang Fu, Ze Chen, Chao Chen, Liang Xie, Wenxiao Wang, Deng Cai, Zheng Wang, and Jieping Ye. 2025. Controlling thinking speed in reasoning models. arXiv preprint arXiv:2507.03704.

Sheng Liu, Tianlang Chen, Pan Lu, Haotian Ye, Yizheng Chen, Lei Xing, and James Zou. 2025. Fractional reasoning via latent steering vectors improves inference time compute. arXiv preprint arXiv:2506.15882.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022a. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. 2022b. Massediting memory in a transformer. arXiv preprint arXiv:2210.07229.

Tomáš Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies, pages 746–751.

Matthew Lyle Olson, Neale Ratzlaff, Musashi Hinck, Shao-yen Tseng, and Vasudev Lal. 2024. Steering large language models to evaluate and amplify creativity. arXiv preprint arXiv:2412.06060.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2023. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658.

Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li. 2025. Steer llm latents for hallucination detection. arXiv preprint arXiv:2503.01917.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. 2024. A systematic survey of prompt engineering in large language models: Techniques and applications. arXiv preprint arXiv:2402.07927.

Marco Scialanga, Thibault Laugel, Vincent Grari, and Marcin Detyniecki. 2025. Sake: Steering activations for knowledge editing. arXiv preprint arXiv:2503.01751.

Thibault Sellam, Dipanjan Das, and Ankur P Parikh.

2020. Bleurt: Learning robust metrics for text generation. arXiv preprint arXiv:2004.04696.

Atakan Seyito˘glu, Aleksei Kuvshinov, Leo Schwinn, and Stephan Günnemann. 2024. Extracting unlearned information from llms with activation steering. arXiv preprint arXiv:2411.02631.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Theia Vogel. 2024. repeng.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2024. Reft: Representation

finetuning for language models. Advances in Neural Information Processing Systems, 37:63908–63962.

Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, and Yueting Zhuang. 2025a. Mind the gap: Bridging thought leap for improved chainof-thought tuning. arXiv preprint arXiv:2505.14684.

Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng, Huajun Chen, and Ningyu Zhang. 2025b. Easyedit2: An easy-to-use steering framework for editing large language models. arXiv preprint arXiv:2504.15133.

Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. 2024. A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. High-Confidence Computing, 4(2):100211.

Hengyuan Zhang, Zhihao Zhang, Mingyang Wang, Zunhai Su, Yiwei Wang, Qianli Wang, Shuzhou Yuan, Ercong Nie, Xufeng Duan, Qibo Xue, and 1 others. 2026. Locate, steer, and improve: A practical survey of actionable mechanistic interpretability in large language models. arXiv preprint arXiv:2601.14004.

Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, and 1 others. 2024. A comprehensive study of knowledge editing for large language models. arXiv preprint arXiv:2401.01286.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, and 1 others. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2).

### A Concept Vector Extraction Methods

EasySteer implements several concept extraction methods commonly employed in analysis-based steering. These methods identify and isolate direction vectors corresponding to specific semantic concepts within model hidden states. The fundamental approach leverages contrastive analysis on paired datasets: a positive dataset D+ containing samples exhibiting the target concept, and a negative dataset D− containing samples that are either opposite to or independent of the target concept. By contrasting differences in model internal activations between these sample types, we can isolate the target concept’s directional representation. The extracted concept vector v is subsequently utilized in the steering intervention process described in Section 3.1.

#### A.1 Contrastive Activation Addition (CAA)

CAA represents one of the most straightforward concept extraction approaches. The concept vector v is computed as the difference between mean hidden states of positive and negative sample sets:

v = Ex+∼D+[hl,i(x+)] − Ex−∼D−[hl,i(x−)]

#### A.2 Principal Component Analysis (PCA)

PCA identifies directions of maximum variance in hidden state representations, capturing the most prominent structural differences in the data. We implement two variants:

Center PCA. For each positive-negative sample pair (x+k ,x−k ), we first compute the centroid:

hl,i(x+k ) + hl,i(x−k ) 2

mk =

We then apply PCA to the set of centered vectors to extract the principal direction with maximum variance:

v = PCA({hl,i(x+) − mk,hl,i(x−) − mk})

Diff PCA. This variant directly applies PCA to the difference vectors between paired samples:

v = PCA({hl,i(x+) − hl,i(x−)})

Since PCA produces undirected components (both v and −v are valid), we perform direction alignment by computing average projections:

proj+ = Ex+∼D+

hl,i(x+)Tv ∥v∥

hl,i(x−)Tv ∥v∥

proj− = Ex−∼D−

If proj+ < proj−, we correct the direction: v ← −v.

#### A.3 Linear Probing

Linear probing learns a linear classifier w ∈ Rd that distinguishes between positive and negative samples through projection in hidden state space. We optimize the following objective with binary cross-entropy loss and regularization:

Ex∼D+∪D−[LBCE(yx,σ(wThl,i(x)))]

v = arg min

w

where yx ∈ {0,1} denotes the class label, σ(·) is the sigmoid function.

#### A.4 Sparse Autoencoders (SAE)

Sparse autoencoders decompose hidden states h ∈ Rd into sparse, interpretable feature representations. The architecture comprises:

Encoder. Maps hidden states to higher dimensional sparse activations f ∈ Rn (where n ≫ d):

f = ReLU(Wench + benc)

Decoder. Reconstructs the original hidden state from sparse activations:

##### hˆ = Wdecf + bdec

Pre-trained SAEs (e.g., gemma-scope (Lieberum et al., 2024)) typically include automated interpretability analyses and semantic labels for discovered features. EasySteer provides a streamlined pipeline for extracting concept vectors from pretrained SAEs:

- 1. Concept Definition and Retrieval. Users define target concepts in natural language and perform semantic search over pre-computed feature interpretations to identify relevant features.
- 2. Feature Selection. Based on search results and automated interpretations, users select the feature index k that best aligns with their intended concept.
- 3. Vector Extraction. The concept vector corresponds to the k-th column of the decoder weight matrix:

##### v = Wdec[:,k]

### B Learning-based Steering Methods

MATH training samples. During inference, we enhance Execution while suppressing Reflection and Transition at reasoning step boundaries (Chen et al., 2025b). A code snippet can be found in Figure 4. We test on DeepSeek-R1-DistillQwen-1.5B/7B using GSM8K (Cobbe et al., 2021) and MATH500 (Lightman et al., 2023) benchmarks with 8192 maximum tokens, measuring accuracy (Xu et al., 2025a) and token efficiency.

This appendix presents representative learningbased steering methods implemented in EasySteer. These methods follow the framework outlined in Section 3.2, optimizing parameterized steering functions fθ while maintaining frozen language model parameters W.

- B.1 Supervised Additive Vector

The simplest learning-based approach directly optimizes an additive steering vector b ∈ Rd on taskspecific data D:

fθ(hl,i) := hl,i + b

This method provides a baseline for more sophisticated steering techniques while maintaining computational efficiency.

- B.2 LM-Steer

LM-Steer introduces a learnable linear transformation at the model’s final layer to modulate generation behavior:

fθ(hl,i) := hl,i + ϵWhl,i

where θ = {W} with W ∈ Rd×d, and ϵ controls the intervention strength. This approach enables more expressive steering while maintaining linearity in the transformation.

- B.3 Low-rank Linear Subspace ReFT (LoReFT)

from vllm import LLM, SamplingParams from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig from transformers import AutoTokenizer

model_name = "DeepSeek-R1-Distill-Qwen-1.5B" tokenizer = AutoTokenizer.from_pretrained(model_name) # Initialize LLM with steering vector capability llm = LLM(model=model_name, enable_steer_vector=True, enforce_eager=True) # search target token target_token = "ĊĊ" vocab = tokenizer.get_vocab() target_token_id = [vocab.get(target_token)] # Configure steering vector request for SEAL control sv_request = SteerVectorRequest(

steer_vector_name="seal", steer_vector_id=1, vector_configs=[

VectorConfig( path="execution.gguf", scale=0.5, target_layers=[20], generate_trigger_tokens=target_token_id, algorithm="direct",

), VectorConfig(

path="reflection.gguf", scale=-0.5, target_layers=[20], generate_trigger_tokens=target_token_id, algorithm="direct",

), VectorConfig(

path="transition.gguf", scale=-0.5, target_layers=[20], generate_trigger_tokens=target_token_id, algorithm="direct",

),

], conflict_resolution="sequential"

) # generate example_answers = llm.generate(

texts, SamplingParams(

temperature=0, max_tokens=8192,

), steer_vector_request=sv_request

)

Figure 4: An illustrative code snippet of the SEAL algorithm implemented using EasySteer. Multiple steering vectors are applied to the \n\n token via the multi-vector collaboration functionality.

LoReFT represents a parameter-efficient representation fine-tuning method that constrains hidden state modifications to a learned low-rank subspace. With only 2rd + r parameters, it achieves effective control through:

#### C.2 Results and Analysis

Table 5 demonstrates that steering effectively reduces redundant reasoning steps while maintaining solution quality using EasySteer. On DeepSeekR1-Distill-Qwen-1.5B, SEAL improves GSM8K accuracy by 2.7% (79.6%→82.3%) while reducing token usage by 40.0%. The 7B model shows similar efficiency gains with 13.3% and 16.8% token reduction on GSM8K and MATH500 respectively.

fθ(hl,i) := hl,i + RT(Whl,i + b − Rhl,i)

where θ = {R,W,b}, R ∈ Rr×d is the low-rank projection matrix, W ∈ Rr×d is the linear projection matrix, and b ∈ Rr is the bias vector. The low-rank constraint (r ≪ d) ensures parameter efficiency while the learned subspace provides sufficient expressiveness for diverse steering objectives.

GSM8K MATH500

Model Method

Acc↑ Tokens↓ Acc↑ Tokens↓ DeepSeek-R1-Distill-Qwen-1.5B

/ 79.6 2435.48 70.8 3966.38

### C Overthinking Mitigation C.1 Experiment Setup

SEAL 82.3 1460.13 78.4 3074.67 DeepSeek-R1-Distill-Qwen-7B

/ 90.3 792.74 86.6 3096.69 SEAL 88.5 687.52 88.2 2577.04

Table 5: Overthinking mitigation performance using SEAL steering.

Following SEAL, we extract three behavioral vectors (Execution, Reflection, Transition) from 1,000

