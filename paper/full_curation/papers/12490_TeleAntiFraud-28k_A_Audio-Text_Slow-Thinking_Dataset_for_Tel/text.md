# arXiv:2503.24115v4[cs.CL]18Aug2025

## TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection

Zhiming Ma∗†

Peidong Wang∗†

Minhua Huang Jinpeng Wang

mazhiming312@outlook.com China Mobile Internet Company Ltd. Guangzhou, China

pdongwang@163.com Northeastern University Shenyang, China

China Mobile Internet Company Ltd. Guangzhou, China

Kai Wu Xiangzhao Lv

China Mobile Internet Company Ltd. Guangzhou, China

Yachun Pang Yin Yang

China Mobile Internet Company Ltd. Guangzhou, China

Wenjie Tang Yuchen Kang

China Mobile Internet Company Ltd. Guangzhou, China

### Abstract

The detection of telecom fraud faces significant challenges due to the lack of high-quality multimodal training data that integrates audio signals with reasoning-oriented textual analysis. To address this gap, we present TeleAntiFraud-28k, the first open-source audio-text slow-thinking dataset specifically designed for automated telecom fraud analysis. Our dataset is constructed through three strategies: (1) Privacy-preserved text-truth sample generation using automatically speech recognition-transcribed call recordings (with anonymized original audio), ensuring real-world consistency through text-to-speech model regeneration; (2) Semantic enhancement via large language model based self-instruction sampling on authentic ASR outputs to expand scenario coverage; (3) Multiagent adversarial synthesis, which simulates emerging fraud tactics through predefined communication scenarios and fraud typologies, enriches the conversation samples. The generated dataset contains 28,511 rigorously processed audio-text pairs with a total audio duration of more than 307 hours, complete with detailed annotations for fraud reasoning. The dataset is divided into three tasks: scenario classification, fraud detection, fraud type classification. Furthermore, we construct TeleAntiFraud-Bench, a standardized evaluation benchmark comprising proportionally sampled instances from TeleAntiFraud-28k, to facilitate systematic testing of model performance, reasoning capabilities, and thought processes on telecom fraud detection tasks. We also contribute a supervised fine-tuning model based on Qwen2-Audio, trained on the TeleAntiFraud-28k training set, while open-sourcing the data processing framework to enable community-driven dataset expansion. This work establishes a foundational framework for multimodal anti-fraud research while addressing critical challenges in data

∗Both authors contributed equally to this research. †Corresponding author

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM ’25, Dublin, Ireland. © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755835

privacy and scenario diversity. The code of this paper is publicly available at https://github.com/JimmyMa99/TeleAntiFraud.

### CCS Concepts

• Computing methodologies → Speech recognition; • Information systems → Multimedia information systems; • Security and privacy → Social network security and privacy.

### Keywords

Telecom Fraud Detection, Multimodal Dataset, Audio-Text Analysis, Slow-Thinking Reasoning, Anti-Fraud Benchmark

ACM Reference Format:

Zhiming Ma, Peidong Wang, Minhua Huang, Jinpeng Wang, Kai Wu, Xiangzhao Lv, Yachun Pang, Yin Yang, Wenjie Tang, and Yuchen Kang. 2025. TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3746027.3755835

### 1 Introduction

As telecom fraud techniques grow increasingly sophisticated, they pose escalating threats to social security and economic stability. Global economic losses fraud-related have reached $1.02 trillion, representing 1.05% of the global GDP, a significant increase over 2020-2021 figures, with over a quarter of respondents reporting encounters with fraud [1]. Developing effective detection methods has therefore become urgent. Traditional approaches relying on manual verification and rule-based pattern matching offer limited accuracy and adaptability against rapidly evolving fraudulent strategies.

Recent advancements in large language models (LLMs), particularly their slow-thinking reasoning capabilities [20], offer promising solutions for combating telecom fraud. However, a significant modality gap exists between voice calls (the primary source of fraud data) and the text data that LLMs process, limiting their direct application. Current industry methods typically employ automatic speech recognition (ASR) to convert audio into text before applying carefully designed prompts for LLM-based fraud detection [19]. Although demonstrating partial effectiveness, this methodology relies extensively on precise prompt engineering, exhibiting significant performance variability across diverse models and implementation contexts. Furthermore, the ASR conversion process often results in information loss [21], potentially omitting crucial fraud indicators

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

#### Figure 1: Models’ Performance on TeleAntiFraud-Bench

contained in vocal features such as tone and pauses that frequently signal fraudulent intent.

Slow-thinkingreasoningenhancesboth accuracy and interpretabil-

ity of LLMs in judgment tasks. Concurrently, emerging large audio language models (LALMs) capable of directly processing audio signals offer potential solutions to the modality gap in anti-fraud applications. However, the absence of slow-thinking audio datasets specifically designed for telecom fraud constrains the application and performance improvement of LALMs in fraud detection.

To address this gap, this study proposes TeleAntiFraud-28k, an audio-text slow-thinking dataset for telecom fraud detection. The dataset is constructed through three methodologies: 1) Transcribing anonymized call recordings using ASR technology to generate privacy-protected text samples, then employing text to speech models to regenerate samples that align with real-world language expressions while preserving content authenticity and safeguarding the privacy of both conversation parties; 2) Implementing selfinstructed sampling strategies with LLMs to semantically enhance content, further expanded through TTS augmentation techniques; 3) Designing a multi-agent adversarial framework that simulates emerging fraud tactics through predefined communication scenarios and fraud typologies, enabling the expansion and generation of novel fraudulent scripts.TeleAntiFraud-28k provides detailed slowthinking annotations covering communication scenario classification, fraud determination, and fraud type identification, designed to enhance model explainability and accuracy through training while improving the model’s capabilities in understanding conversation contexts, detecting fraudulent activities, and categorizing fraud types respectively.

We systematically extract representative samples to construct TeleAntiFraud-Bench, an evaluation benchmark preserving the original dataset’s scenario distribution and fraud type proportions, ensuring reliable assessment outcomes. This benchmark provides

researchers with a unified evaluation platform for comparative analysis across telecommunication scenario classification, fraud detection, fraud type classification tasks, and assessment of model reasoning processes, enabling comprehensive evaluation of both outcomes and thought patterns.

Experiments on state-of-the-art Large Audio Language Models (LALMs) demonstrate that current LALMs without fine-tuning perform inadequately for telecom anti-fraud tasks. After fine-tuning Qwen2-Audio on the TeleAntiFraud-28k training set, we observed a 27.5-point score increase on TeleAntiFraud-Bench, demonstrating the dataset’s effectiveness and practical value in developing audio-based anti-fraud models.

The contributions of this research can be summarized as follows:

- (1) Proposing the first multi-task slow-thinking audio-language dataset TeleAntiFraud-28k for telecom fraud prevention, encompassing three tasks: communication scenario classification, fraud determination, and fraud type analysis;
- (2) Designing a new data generation pipeline that maximizes coverage of diverse fraud scenarios through real-call ASR processing, LLM-based simulation, and multi-agent adversarial generation;
- (3) Establishing the TeleAntiFraud-Bench evaluation benchmark to provide standardized testing standards for telecom fraud detection models, while designing a series of evaluations for anti-telecom fraud slow-thinking capabilities;
- (4) Conducting comprehensive evaluations on multiple leadingedgeLALMsusingTeleAntiFraud-Bench, validating the dataset’s training effectiveness and establishing performance baselines for future audio-based anti-fraud research.

This work bridges critical research gaps in multimodal fraud detection and provides valuable resources for advancing intelligent anti-fraud systems. The dataset and benchmark are publicly available to facilitate community-wide research efforts in combating evolving telecom fraud.

2 Related work

- 2.1 LLM-Based Telecom Fraud Detection

Large language models have been applied to telecom fraud detection, but existing research primarily focuses on text analysis. Singh et al. [2] proposed a RAG-based system achieving 97.98% accuracy but relies on text analysis alone. Shen et al. [3] developed a real-time framework for fraudulent intent detection that remains text-centric. Shen et al. [4] identified LLM challenges including data bias and hallucinations in fraud detection. Chang [5] revealed LLM vulnerabilities to adversarial scams, though limited to textual.

Current anti-fraud systems rely on predefined rules [23] or text-only analysis without audio integration [22]. Trained on limited datasets, these systems struggle with evolving fraud tactics, highlighting the need for a large-scale, multimodal telecom fraud dataset.

- 2.2 Multimodal audio-Text Models

Advances in deep learning have enabled significant progress in multimodal models, yet these innovations remain underutilized in telecom fraud detection [24].

TeleAntiFraud-Bench

[Figure 5]

Real-Data ASR Processing Audio Synthesis

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Real Audio ASR Data Data

LALM Model Think-LALM

Training

[Figure 14]

[Figure 15]

LLM-Based Imitation and Augmentation

[Figure 16]

LALM

Reasoning Process Quality

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

ASR Data

[Figure 22]

TeleAntiFraud-28k

Multi-Agent Adversarial Framework

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Audio Think-LALM Text

[Figure 32]

User Cheater

Scenario Classification

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Fraud Determination

[Figure 37]

[Figure 38]

Manager

Fraud Type Identification

Audio Text User User

[Figure 39]

#### Figure 2: An overview of TeleAntiFraud-28k. Our system addresses telecom fraud detection challenges by creating TeleAntiFraud28k through Real-Data ASR Processing, LLM-Based Imitation and Augmentation, and multi-agent adversarial synthesis. We develop TeleAntiFraud-Bench for evaluation and provide a supervised fine-tuning model with open-sourced data processing.

Recent large multimodal models demonstrate tremendous potential in telecom fraud prevention. GLM-4-Voice [6] supports real-time bilingual voice interaction with adjustable audio characteristics including emotion, tone, and dialect. Qwen2-Audio [7] processes diverse audio inputs for text responses. Among proprietary systems, GPT-4o [8] handles arbitrary combinations of text, audio, image, and video inputs to generate multimodal outputs.

Despite possessing certain capabilities, existing multimodal models lack domain-specific optimization for telecom fraud detection, as they are primarily designed for general conversation or content comprehension. Due to privacy concerns associated with call audio, these models have limited access to call-related training data during their development process. Additionally, challenges such as difficulties in data acquisition, stringent privacy requirements, and the diverse nature of fraud scenarios require specialized approaches that current general-purpose models are not equipped to handle.

While LLM-based fraud detection systems and multimodal audiotext models have advanced independently, their integration for combating telecom fraud remains inadequate. The TeleAntiFraud-28k dataset addresses these limitations by providing large-scale, slowthinking-annotated audio-text pairs, establishing infrastructure for multimodal anti-fraud research.

### 3 Method

This chapter details the TeleAntiFraud-28k construction methodology and the design of the evaluation framework, TeleAntiFraudBench. As depicted in Figure 2, our approach encompasses three components: voice data generation, text data annotation based on slow-thinking, and the establishment of an evaluation benchmark.

Our methodology involves a three-stage process. First, we generate a high-quality, diverse telephone-call dataset covering both telecom fraud and normal scenarios using three novel methods. Next, in a "slow-thinking" annotation phase, an audio-understanding

Data Flow of TeleAntiFraud-28k

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Audio

[Figure 45]

Audio Synthesis

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

LALM

thinking...

TeleAntiFraud-28k

#### Figure 3: Data Flow of Audio Data Generation

model’s analysis and ASR outputs are used to prompt the DeepSeekR1 large reasoning model, which in turn generates detailed annotations that include its explicit reasoning process. Finally, we establish a unified benchmark to evaluate various models on the telecom anti-fraud task.

### 3.1 Audio Data Generation Pipeline

To construct the high-quality, diverse TeleAntiFraud-28k dataset, we developed a robust voice data generation pipeline. Our approach, illustrated in Figure 3, comprises two key stages: dialogue text generation and voice synthesis.

For dialogue text generation, we employed three complementary methods: real data ASR processing, large language model imitation generation, and a multi-agent adversarial framework. These

approaches collectively ensured authenticity, diversity, and comprehensive scenario coverage. In the subsequent voice synthesis stage, we converted the generated dialogue texts into dual-channel voice recordings using TTS technology, creating separate audio tracks for callers and callees that simulate realistic telephone conversations. Real-Data ASR Processing. Our first dialogue acquisition approach leverages real telephone recordings through ASR process-

ing. We collected actual telecom fraud recordings (𝐷𝑇𝐹) and normal calls (𝐷𝑇𝑁 ), then converted these into separate channel texts using ASR technology. This method preserves authentic speech patterns and conversational dynamics while anonymizing sensitive information. The resulting transcripts were subsequently transformed into synthetic voice data using TTS models, producing realistic fraudulent (𝐷𝑆𝐹1) and normal (𝐷𝑆𝑁1) recordings. This approach ensures generated content closely resembles genuine fraud scenarios while addressing privacy concerns inherent in utilizing original recordings directly.

LLM-Based Imitation and Augmentation. Our second dialogue generation method utilizes the self-instruct paradigm [27] to augment ASR-processed call transcripts. We designed prompt templates incorporating few-shot exemplars from our initial datasets,

𝐷𝑆𝐹1 and 𝐷𝑆𝑁1, to guide the LLM in generating dialogues that are both diverse and pattern-consistent. To ground the generation in

realism, prompts were enriched with key linguistic features, interaction patterns, and fraudulent tactics extracted from real calls. For fraudulent content, we preserved core deception strategies while systematically varying circumstantial details. This process yielded

two substantial, high-quality datasets of synthetic fraudulent (𝐷𝑆𝐹2) and normal (𝐷𝑆𝑁2) calls. The self-instruct approach proved effective for controllably scaling our data volume and diversity, thereby enhancing model generalization to better simulate real-world scenarios.

Multi-Agent Adversarial Framework. To augment conversational text data, we introduce a multi-agent adversarial framework designed to simulate emerging fraud tactics and expand data diversity. We have identified that current telecom fraud datasets, while authentic, exhibit a limited range of conversational patterns and scenarios, hindering their effectiveness against novel fraud techniques. Our proposed framework addresses this by simulating fraudulent activities in varied business contexts, thereby enriching the data distribution to cover a wider array of fraud strategies.

As illustrated in Figure 4, the framework involves three agents: a caller (cheater), a callee (potential victim), and a Manager. The caller is given a specific fraud type, and the callee has a defined profile to ensure realistic interactions. The conversation is turn-based, with the Manager monitoring for adherence to the preset scenario and natural flow.

To expand data diversity, we designed numerous scenario-fraud type combinations based on seven normal and eight real-world fraud scenarios. This approach systematically guides the data generation process, effectively broadening the data distribution and filling existing gaps.

A special signal mechanism facilitates natural conversation termination. The caller or callee can send ##ENDCALL_SIGNAL## to hang up, while the Manager can use ##TERMINATE_SIGNAL## to end the dialogue, preventing loops or topic deviation.

Multi-Agent Adversarial Framework

Fraud Conversation

[Figure 50]

[Figure 51]

Hi, I'm user. You've won a prize！

[Figure 52]

[Figure 53]

[Figure 54]

##ENDCALL_SIGNAL##

User-LLM

Cheater-LLM

[Figure 55]

[Figure 56]

###### User out!

Cheater out!

##TERMINATE_SIGNAL##

Conversation end.

Manager-LLM

[Figure 57]

[Figure 58]

Let's chat! Sure!

[Figure 59]

[Figure 60]

[Figure 61]

Bye.

##ENDCALL_SIGNAL##

User-LLM User-LLM

Normal Conversation

#### Figure 4: Structure of Multi-Agent Adversarial Framework

Through the multi-agent adversarial framework, we generated

a new set of normal call data 𝐷𝑆𝑁3 and fraudulent call data 𝐷𝑆𝐹3. The data generated by the multi-agent framework significantly

expanded the distribution space of the original data, particularly in new fraud types and complex scenarios, providing more comprehensive training data for the model.

Audio Synthesis. After generating dialogue texts, we employed ChatTTS [? ] to convert them into dual-channel audio. This advanced TTS model excels in conversational scenarios with natural, expressive speech, multi-speaker capabilities, and prosodic control.

To enhance realism, we implemented three strategies:

- 1. Diverse Audio Characteristics: We randomized voice parameters (timbre, rate, pitch) and configured distinct characteristics for different speakers.
- 2. Natural Speech Features: We preserved pauses, stress, and emotional variations, enhancing deception-specific expressions (urgency, authority, false affinity) for fraudulent scenarios.
- 3. Precise Temporal Control: We managed the timing between voices to ensure turn-taking and appropriate response delays.

Our dataset comprises normal calls 𝐷𝑁 = 𝐷𝑆𝑁1 ∪ 𝐷𝑆𝑁2 ∪ 𝐷𝑆𝑁3 and fraudulent calls 𝐷𝐹 = 𝐷𝑆𝐹1 ∪ 𝐷𝑆𝐹2 ∪ 𝐷𝑆𝐹3. Original recordings were transcribed, anonymized, and resynthesized as 𝐷𝑆𝑁1 and 𝐷𝑆𝐹1. Our public release includes these anonymized segments only in the test set, alongside LLM-generated and multi-agent-derived data 𝐷𝑆𝑁2 ∪ 𝐷𝑆𝑁3 and 𝐷𝑆𝐹2 ∪ 𝐷𝑆𝐹3.

### 3.2 Slow Thinking-Based Text Annotation

The TeleAntiFraud-28k dataset’s distinctive feature is its "Slow Thinking" mechanism that emulates anti-fraud experts’ analytical processes. We implemented a two-stage workflow for generating high-quality annotations:

- 1. Audio Analysis Phase: Generated voice data is processed

through a professional audio understanding model to extract voice features and information, including emotional variations, tonal characteristics, pause patterns, and other audio-level indicators.

- 2. Expert Reasoning Phase: Audio analysis results and ASR

text are combined as prompts for the DeepSeek-R1, which acts

as a voice analysis expert with anti-fraud professional knowledge. The model employs "<think></think>" markers to document its complete reasoning chain from clue identification to judgment, and "<answer></answer>" to denote final conclusions.

This design captures anti-fraud experts’ systematic framework, encompassing speech pattern recognition, fraud technique identification, and risk assessment. Our designed prompt template presents conversation content and audio feature analysis, requiring the model to output JSON-formatted information including reasoning grounds (reason), confidence level (confidence), and a boolean fraud determination (is_fraud).

The annotation process involves three sequential analytical steps:

- 1. Call Scene Classification: The model analyzes the conver-

sation’s basic context and theme, categorizing it into one of seven predefined scenarios: "Dining Service", "Customer Consultation", "Appointment Service", "Transportation Inquiry", "Routine Shopping", "Ride-Hailing Service", or "Food Delivery Service".

- 2. Fraud Determination: Building on scene classification, the

model evaluates potential fraudulent behavior by analyzing speech characteristics, request reasonableness, information disclosure patterns, and other professional indicators, providing a clear judgment with supporting evidence.

- 3. Fraud Type Identification: For identified fraudulent calls,

the model categorizes them into seven main types: "Investment Fraud," "Phishing Fraud", "Identity Theft", "Lottery Fraud", "Banking Fraud", "Extortion Fraud", or "Customer Service Fraud".

Each analytical step yields JSON output with judgment parameters and confidence levels. This progressive analysis mirrors professional workflows, building context to enhance accuracy and reliability, while the documented reasoning provides a rich dataset for future research.

- 3.3 Construction of Evaluation Benchmark

To systematically evaluate model performance in telecom anti-fraud tasks, we established the TeleAntiFraud-Bench evaluation benchmark. This benchmark comprises samples randomly extracted from the TeleAntiFraud-28k dataset in proportion, incorporating gen-

uine normal data (𝐷𝑆𝑁1) and genuine fraud data (𝐷𝑆𝐹1) into the test set. This construction methodology preserves the original dataset’s

scenario and fraud type distributions, ensuring evaluation results maintain high representativeness and reliability.

Design of the Evaluation Process. TeleAntiFraud-Bench employs a hybrid evaluation mechanism combining rule-based extraction with LLM analysis to ensure comprehensive and accurate assessment. We designed structured prompts to guide evaluation LLMs while simultaneously utilizing regular expressions for precise information extraction. This approach comprises four key steps:

- 1. Regular Expression-Based Key Information Extraction: We extract critical results from model outputs—including scenario classification, fraud judgment, and fraud type—using regular expressions. This approach provides precise foundational data for subsequent evaluation, ensuring accuracy and consistency independent of LLM analysis.
- 2. LLM Analysis of the Thought Process: The LLM evaluation analyzes the model’s reasoning process (content within "<think></think>" markers) for completeness, logical coherence,

and effective use of evidence, using the language understanding and analytical capabilities of the LLM.

#### 3. LLM Verification of Final Judgment Consistency: The

evaluation LLM verifies alignment between the model’s final judgment and standard answers, utilizing its strengths in logical judgment and comparative analysis to ensure reliable evaluation results.

Case Study of TeleAntiFraud-28k

[Figure 62]

Hello, I'm a financial advisor from an international investment company, inviting you to join our exclusive high-return investment plan.

Welcome to our food delivery service. Our signature dishes include Kung Pao Chicken and Mapo Tofu. What would you like to order?

[Figure 63]

[Figure 64]

[Figure 65]

Hello, I'm interested in investing, but quite busy. Could you briefly introduce this plan?

Kung Pao Chicken sounds good, I'll try that. Could you add a portion of rice as well?

[Figure 66]

[Figure 67]

This is a high-yield investment plan with annual returns over 15%,emerging real estate markets, available only to high-net-worth clients.

One Kung Pao Chicken and one rice. Would you like any drinks or side dishes?

[Figure 68]

Sounds interesting. As a doctor, my financial knowledge is limited. Could you explain the risks and returns in detail?

[Figure 69]

No, that's all. Thank you.

[Figure 70]

[Figure 71]

Our experienced analysts will customize plan for you. Minimum investment is 500,000 yuan, you can start with a small amount to test the market.

Your order is confirmed. Estimated delivery in 30 minutes. Enjoy your meal!

[Figure 72]

[Figure 73]

500,000 is substantial, I need to discuss with my family. Where is your company located? I'd like to visit in person.

Thanks, looking forward to the food.

Fraud Conversation Normal Conversation

Figure 5: Case Study of TeleAntiFraud-28k

#### 4. Comprehensive Evaluation Score and Report Genera-

tion: We integrate regular expression-extracted information with LLM analysis to generate comprehensive evaluation scores and detailed analytical reports that thoroughly present model performance across multiple dimensions.

This combined methodology improves evaluation robustness by integrating pattern-matching precision from regular expressions with the contextual understanding of LLMs, providing more thorough and accurate assessment of model performance.

Inference Evaluation Process. To comprehensively assess reasoning quality, we developed an evaluation framework based on three dimensions: logical consistency, practicality, and clarity. This framework employs structured prompting [18] to guide an LLM in performing scoring tasks using model reasoning (𝑅𝑚), model answer (𝐴𝑚), reference answer (𝐴𝑟), and reference reasoning (𝑅𝑟). The scoring process combines expert rules and logical constraints through a probabilistic mechanism.

Logical consistency evaluates reasoning chain completeness, assumption reasonableness, and conclusion derivation rigor. When 𝐴𝑚 ≠ 𝐴𝑟, scoring probabilities significantly decrease, with further reductions for each logical leap or missing evidence. When 𝑅𝑚 ≠ 𝑅𝑟 but 𝐴𝑚 = 𝐴𝑟, probabilities are moderately reduced to account for reasoning divergence while acknowledging the possibility of multiple valid logical approaches. Practicality assesses problem essence understanding, solution effectiveness, and demand coverage completeness. If 𝐴𝑚 ≠ 𝐴𝑟, probabilities for problem identification and solution effectiveness default to zero. Even with matching answers, reasoning differences trigger verification of solution pathway validity against established domain principles. Clarity measures key point presentation, language conciseness, and information organization efficiency, with probability decay triggered by unclear or redundant expressions. When reasoning differs, additional scrutiny of expression efficiency and organization occurs, with probability

adjustments proportional to the degree of structural and presentational divergence from reference reasoning.

We allocate scores using probabilistic distributions that quantify the likelihood of different values. For a scoring point with possible values 𝑆 ∈ {0, 1, 2}, we assign probability 𝑃(𝑆) and calculate the expected value using 𝐸 = 𝑆 𝑆 · 𝑃(𝑆). The total score combines expected values from logical consistency (𝐸𝐿), practicality (𝐸𝑈 ), and clarity (𝐸𝐶):

𝐸total = 𝐸𝐿 + 𝐸𝑈 + 𝐸𝐶

This evaluation system integrates quantitative metrics with probabilistic calculations for objective and comprehensive reasoning quality assessment.

Evaluation Metric System. TeleAntiFraud-Bench employs a balanced scoring mechanism with four equally weighted dimensions (25% each). We use weighted F1 scores as our quantitative metrics, calculated as:

##### ∑︁𝑛

2 × 𝑃𝑡𝑎𝑠𝑘𝑖 × 𝑅𝑡𝑎𝑠𝑘𝑖 𝑃𝑡𝑎𝑠𝑘𝑖 + 𝑅𝑡𝑎𝑠𝑘𝑖

𝐹1𝑡𝑎𝑠𝑘 =

𝑤𝑡𝑎𝑠𝑘𝑖 ×

𝑖=1

Where 𝑤𝑡𝑎𝑠𝑘𝑖 = 𝑛𝑛𝑖

𝑗=1𝑛𝑗 represents the proportion of samples in

class 𝑖, 𝑃𝑡𝑎𝑠𝑘𝑖 and 𝑅𝑡𝑎𝑠𝑘𝑖 denote precision and recall for class 𝑖, and 𝑛 is the total number of classes.

Our evaluation framework comprises four key metrics:

- 1. Scene Classification F1 Score: Measures accuracy in identi-

fying seven call scenarios, using precision 𝑃𝑠𝑐𝑒𝑛𝑒𝑖 and recall 𝑅𝑠𝑐𝑒𝑛𝑒𝑖 across scene categories.

- 2. Fraud Judgment F1 Score: Evaluates accuracy in determin-

ing call legitimacy, using precision 𝑃𝑓 𝑟𝑎𝑢𝑑𝑖 and recall 𝑅𝑓 𝑟𝑎𝑢𝑑𝑖 for fraud detection.

- 3. Fraud Type Identification F1 Score: Assesses accuracy in

categorizing seven fraud types, using precision 𝑃𝑡𝑦𝑝𝑒𝑖 and recall 𝑅𝑡𝑦𝑝𝑒𝑖 across fraud categories.

4. Quality of Thought Process: Evaluated by an LLM examining completeness, evidence utilization, professional knowledge application, and logical coherence.

The comprehensive scoring system combines multiple evaluation dimensions with equal weighting. The total score is calculated as:𝑆𝑐𝑜𝑟𝑒𝑡𝑜𝑡𝑎𝑙 = 0.25 × 𝐹1𝑠𝑐𝑒𝑛𝑒 + 0.25 × 𝐹1𝑓 𝑟𝑎𝑢𝑑 + 0.25 × 𝐹1𝑡𝑦𝑝𝑒 + 0.25 × 𝐸𝑡𝑜𝑡𝑎𝑙

This balanced approach integrates scene identification, fraud detection, type classification, and process evaluation metrics to provide a complete assessment of system performance.

- 4 Evaluation

### 4.1 Experimental Setup

Dataset Statistics.The constructed TeleAntiFraud-28k dataset comprises 28,511 utterance samples, which are divided into training and test sets. The training set contains 21,490 samples, constituting 75.38% of the total dataset, while the test set contains 7,021 samples, constituting 24.62%. This 3:1 partition ratio ensures sufficient samples for benchmark evaluation while adhering to the conventional training-test split protocol widely adopted in machine learning methodology. Detailed examples from the dataset are illustrated in Figure 5. Table 1 presents the statistical information of the dataset.

#### Table 1: Distribution of Scam and Normal Calls in the Dataset

Type Total Fraud Calls Normal Calls Train 21,490 9,950 (46.3%) 11,540 (53.7%) Test 7,021 3,697 (52.66%) 3,324 (47.34%) Total 28,511 13,647 (47.86%) 14,864 (52.13%)

The TeleAntiFraud-28k maintains relative balance between fraudulent and normal calls, with fraud calls constituting 46.3% (9,950 samples) of the training set and 52.66% (3,697 samples) of the test set. Fraud calls represent 47.86% (13,647 samples) of the entire data set, ensuring a balanced evaluation of detection capabilities.

#### Table 2: Distribution of scenario types in train and test sets

Scenario Type Training Set Test Set Customer Consultation 6,421 4,632 Appointment Services 1,714 867 Routine Shopping 924 340 Dining Services 581 154 Food Delivery Services 494 448 Ride-Hailing Services 353 489 Transportation Inquiries 223 91 Total 10,710 7,021

The TeleAntiFraud-28k dataset was built to be diverse and representative of various conversational scenarios. After rigorous selection and annotation, the training set features multiple interaction types (Table 2), with customer consultation being the largest category (6,421 samples), followed by appointment services (1,714 samples). This diversity is mirrored in the test set to validate robust model generalization. This approach provides a solid foundation for practical applications, such as the Multi-Task Learning case studies illustrated in Figure 6.

In terms of fraud types, the dataset covers seven major categories: phishing, kidnapping, lottery, customer service, banking, investment, and identity theft. As shown in Table3, the sample distribution reflects real-world occurrence patterns, with variations that align with practical fraud detection challenges.

#### Table 3: Distribution of fraud types in train and test sets

Fraud Type Training Set Test Set Customer Service Fraud 2,022 725 Banking Fraud 1,626 2,408 Investment Fraud 785 216 Phishing Fraud 443 123 Lottery Fraud 418 99 Kidnapping Fraud 324 91 Identity Theft 105 35 Total 5,723 3,697

Model Selection for Evaluation. In telecom fraud detection, ASR combined with LLMs is the predominant approach. This study evaluated 10 representative models on the TeleAntiFraud-28k dataset, using SenseVoice as the standardized ASR. For ASR+LLM evaluation,

[Figure 74]

#### Figure 6: Case Study of Multi-Task Learning

we selected high-performance open-source models: DeepSeek-V3, DeepSeek-R1, Doubao-1.5-Pro, InternLM2.5-20B-Chat, GLM-4-9BChat, and Qwen2.5-72B-Instruct. Multimodal evaluation included commercial models (GPT-4o, Gemini-2.0-Flash) and open-source options (GLM-4-Voice, Step-1o-audio, Qwen2-Audio). These selections span diverse scales and types, ensuring comprehensive evaluation.

### 4.2 Experimental Results

Scenario Classification Performance. In scenario classification, DeepSeek-V3 achieved the highest F1 score (88.53%), demonstrating ASR+LLM advantages, followed by DeepSeek-R1 (83%). The fine-tuned AntiFraud-Qwen2-Audio (81.31%) exhibited stable performance, confirming that anti-fraud-optimized language models effectively enhance task performance. Multimodal models performed reliably, with GPT-4o (80.25%) and Gemini-2.0-Flash (80.51%) providing comprehensive information retrieval, although falling short of the top ASR + LLM models. InternLM2.5-20B-Chat (78. 34%) and GLM-4-9B-Chat (75. 1%) highlighted the limitations of language models in scenario classification.

Fraud Detection Performance. Fraud detection tasks revealed more pronounced performance differences. DeepSeek-R1 excelled with an F1 score of 79. 25%, showcasing the strong recognition capabilities of the ASR+slow-thinking LLM approach. The fine-tuned AntiFraud-Qwen2-Audio model performed best (84.78%), demonstrating that specially optimized speech-language models enhance fraud detection reliability. Multimodal models showed varied performance: GPT-4o (50%) and Step-1o-audio (40.65%) underperformed compared to ASR+LLM approaches, indicating challenges in integrating audio and language information. Gemini-2.0-Flash (59.61%) demonstrated balanced performance, while InternLM2.5-20B-Chat (36.67%) and GLM-4-Voice (26.83%) revealed significant limitations in practical fraud detection.

Fraud Type Classification Performance. In fraud type classification, models performed similarly. DeepSeek-R1 led with an F1

score of 85.16%, followed by InternLM2.5-20B-Chat (85.42%) and Doubao-1.5-Pro (82.25%). AntiFraud-Qwen2-Audio achieved 82.91%, while without fine-tuning it scored only 58.51%, demonstrating finetuning’s effectiveness. The multimodal GPT-4o performed exceptionally (86.26%), showing advantages in multimodal information integration, while Gemini-2.0-Flash achieved 74.55%.

Quality Assessment of the Thinking Process. We conducted a comprehensive evaluation of model reasoning quality across three dimensions—logical rigor, practical value, and expressive quality—using DeepSeek-R1 as a systematic judge (Table 4). As predicted by scaling law theory, a model’s reasoning ability is strongly correlated with its parameter count, which explains the superior performance of specialized reasoning models like DeepSeek-R1 and DouBao-1.5-Pro.

Notably, our AntiFraud-Qwen2-Audio model outperformed both similarly-sized models (GLM4-9B-Chat, GLM4-9B-Voice) and significantly larger ones (InternLM2.5-20B, Step-Audio-Chat 130B), achieving a reasoning quality comparable to GPT-4o. This result clearly demonstrates the efficacy of our task-specific optimization. In contrast, the untrained Qwen2-Audio-7B-Instruct exhibited substantially lower reasoning quality, underscoring the critical importance of our slow-thinking training set for cultivating the model’s advanced analytical capabilities.

Comprehensive Evaluation Model. In the telecom fraud detection task, our comprehensive evaluation revealed AntiFraudQwen2-Audio as the leading performer with an 83% average F1 score, validating our proposed data synthesis and fine-tuning strategies. Analysis of reasoning abilities showed this model excelled in thinking process quality within the DeepSeek-R1 evaluation framework, significantly outperforming competitors in logical rigor (2.06), practical value (2.07), and expressive quality (2.31), achieving a total score of 6.44. This performance surpassed similarly sized models like GLM4-9B-Chat and approached GPT-4o’s capabilities.

#### Table 4: Comprehensive Evaluation of Different Models on Quality Metrics and Fraud Detection Tasks

Quality Evaluation Fraud Detection Performance

Model

Final Logical Practical Expression Total Scenario Fraud Fraud Type

Rigor Value Quality (15) (%) (%) (%)

ASR+LLM GLM4-9B-Chat [14] 1.61 1.43 2.20 5.25 75.10 46.91 82.22 59.81 InternLM2.5-20B [13] 1.99 1.93 2.43 6.37 78.34 36.67 85.42 60.72 Qwen2.5-72B [15] 2.21 2.16 2.70 7.01 78.31 51.44 81.24 64.43 DeepSeek-V3 [11] 2.32 2.34 2.85 7.51 88.53 14.62 66.71 54.98 Doubao-1.5-Pro [25] 1.94 1.75 2.60 6.31 71.14 36.11 82.25 57.89

ASR+Reasoning LLM DeepSeek-R1 [12] 3.18 3.26 3.50 9.94 83.60 79.25 85.16 78.57

LALM Qwen2-Audio-7B-Instruct 1.51 1.42 1.96 4.91 70.22 58.51 20.47 45.48 GLM4-9B-Voice* 0.89 0.64 0.65 1.89 - 26.83 38.33 Gemini-2-Flash* [26] 2.25 2.29 2.72 7.25 80.51 59.61 83.53 68.00 GPT-4o* [8] 2.12 2.10 2.56 6.79 80.25 50.00 86.26 65.44 Step-1o-Audio* [17] 1.64 1.62 2.01 5.26 76.35 40.65 79.71 57.94

Fine-tuned Anti-Fraud LALM AntiFraud-Qwen2-Audio 2.06 2.07 2.31 6.44 81.31 84.78 82.91 72.98

Note: Scenario = Scene Classification F1, Fraud = Fraud Detection F1, Classification F1, AVG F1= Average Score, Total(15) = Quality of Thought Process Score, *: The test samples are 1,200 pieces selected through sampling.

The untrained Qwen2-Audio-7B-Instruct scored only 4.91 in thinking quality evaluation, while the slow-thinking trained AntiFraudQwen2-Audio demonstrated significant improvement, validating our training strategy. The multimodal GPT-4o* achieved a 72.17% average F1 score, highlighting both potential and challenges of multimodal approaches, while InternLM2.5-20B-Chat and GLM-49B-Chat showed relatively weaker performance (66.81% and 68.08% respectively). These results underscore the importance of targeted data synthesis, and slow-thinking training for enhancing model performance in telecom fraud detection.

### 4.3 Ablation Studies

Audio’s Influence on Model Judgment Capability. We analyzed audio and text features’ impact on model performance across two dimensions. As shown in Table 5, models trained solely on ASR text achieved an average F1 score of 73.58%, substantially outperforming the base model (49. 73%), indicating that text features effectively capture conversational semantic information. The Without Think and Think multimodal models achieved average F1 scores of 73.93% and 83.00% respectively, demonstrating that multimodal fusion significantly enhances fraud detection capabilities. Audio feature integration supplements text with vocal nuances, enhancing the model’s fraud scenario comprehension.

Influence of Slow Thinking on the Performance of the Model. The slow-thinking mechanism represents a key innovation in this study. Comparative analysis between models trained with Without Think and Think datasets demonstrates significant performance enhancements when incorporating slow-thinking strategies. The F1 score improved from 72.08% to 80.91% for scenario classification, from 69.32% to 84.78% for fraud involvement detection, and from 80.39% to 82.91% for fraud type classification. This substantial

#### Table 5: F1 Comparison of Different Qwen2-Audio Variants

###### Variant Scene Fraud Type Avg F1

Base 70.22 58.51 20.47 49.73 ASR-text 71.55 71.27 77.93 73.58 NO Think 72.08 69.32 80.39 73.93 Think 81.31 84.78 82.91 83.00

Note: Scenario = Scene Classification F1, Fraud = Fraud Detection F1, Classification F1, AVG F1= Average Score. Base = untrained model, ASRtext = model with only speech ASR text without audio, NO Think = model without slow thinking process, Think = model with slow thinking.

improvement (average F1 score increase from 73.93% to 83.00%) validates the value of slow thinking in telecom fraud detection—the multi-step reasoning mechanism enables deeper analysis of call scenarios, captures more potential fraud features, and reduces hasty judgments. Additionally, the slow-thinking approach enhances system explainability and credibility, aligning more closely with human anti-fraud expert reasoning.

### 5 Conclusion

This study addresses telecom fraud detection challenges by creating the TeleAntiFraud-28k dataset through our three-phase strategy: Real-Data ASR Processing, LLM-Based Imitation, and Multi-Agent Adversarial Framework. This first slow-thinking dataset for telecom fraud comprises 28,511 audio-text pairs with fraud reasoning annotations. We established TeleAntiFraud-Bench to evaluate models and trained AntiFraud-Qwen2-Audio, demonstrating improved fraud detection through multimodal fusion and structured thinking. Our approach provides foundations for future systems with real-world anti-fraud applications.

### Acknowledgements

Thanks to all co-authors for their hard work. We would like to express our sincere gratitude to China Mobile Internet Company, NEU Data Mining, ModelScope Community, and SmartFlowAI Community for their valuable support and assistance throughout the course of this work. Special thanks go to Qingyun Pan, Wenxing Hu and Jintao Huang for their insightful guidance and helpful contributions. The work is supported by the Natural Science Foundation of China (62272092, 62172086).

### References

- [1] Jorij Abraham, Sam Rogers, Luka Koning, Clement Njoki, and James Greening. Global State of Scams Report 2024, March 2025. https://www.gasa.org/_files/ugd/ 7bdaac_9060be8317424edd9964072cf279a0a4.pdf. Accessed: March 25, 2025.
- [2] Gurjot Singh, Prabhjot Singh, and Maninder Singh. Advanced real-time fraud detection using rag-based llms. arXiv preprint arXiv:2501.15290, 2025.
- [3] Zitong Shen, Sineng Yan, Youqian Zhang, Xiapu Luo, Grace Ngai, and Eugene Yujun Fu. “It warned me just at the right moment”: Exploring llm-based real-time detection of phone scams. arXiv preprint arXiv:2502.03964, 2025.
- [4] Zitong Shen, Kangzhong Wang, Youqian Zhang, Grace Ngai, and Eugene Y. Fu. Combating phone scams with llm-based detection: Where do we stand? arXiv preprint arXiv:2409.11643, 2024.
- [5] Chen-Wei Chang, Shailik Sarkar, Shutonu Mitra, Qi Zhang, Hossein Salemi, Hemant Purohit, Fengxiu Zhang, Michin Hong, Jin-Hee Cho, Chang-Tien Lu, et al. Exposing llm vulnerabilities: Adversarial scam detection and performance. In 2024 IEEE International Conference on Big Data (BigData), pages 3568–3571, 2024.
- [6] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm-4-voice: Towards intelligent and humanlike end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024.
- [7] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.
- [8] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [9] Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui Hu, Kai Hu, et al. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms. arXiv preprint arXiv:2407.04051, 2024.
- [10] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.
- [11] DeepSeek-AI. DeepSeek-V3 Technical Report, 2024. https://arxiv.org/abs/2412. 19437.

- [12] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. https://arxiv.org/abs/2501.12948.
- [13] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. InternLM2 Technical Report, 2024. https://arxiv.org/abs/2403.17297.
- [14] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hao Yu, et al. ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools, 2024. https://arxiv.org/abs/ 2406.12793.
- [15] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 Technical Report, 2024. https://arxiv.org/abs/2412.15115.
- [16] Team, Gemini, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, et al. Gemini: A Family of Highly Capable Multimodal Models, 2023. https: //arxiv.org/abs/2312.11805.
- [17] Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al. Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction, 2025. https: //arxiv.org/abs/2502.11946.
- [18] Ming Wang, Yuanzhong Liu, Xiaoming Zhang, Songlian Li, Yijie Huang, Chi Zhang, Daling Wang, Shi Feng, and Jigang Li. LangGPT: Rethinking Structured Reusable Prompt Design Framework for LLMs from the Programming Language,

2024. https://arxiv.org/abs/2402.16929.

- [19] Liming Jiang. Detecting scams using large language models. arXiv preprint arXiv:2402.03147, 2024.
- [20] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.
- [21] Andong Li, Wenzhe Liu, Xiaoxue Luo, Chengshi Zheng, and Xiaodong Li. Icassp 2021 deep noise suppression challenge: Decoupling magnitude and phase optimization with a two-stage deep network. In ICASSP 2021 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6628–6632, 2021.
- [22] A. Mbaziira and J. Jones. A text-based deception detection model for cybercrime. In Int. Conf. Technol. Manag, pages 1–8, 2016.
- [23] Mansoor Ahmed, Kainat Ansar, Cal B. Muckley, Abid Khan, Adeel Anjum, and Muhammad Talha. A semantic rule based digital fraud detection. PeerJ Computer Science, 7:e649, 2021.
- [24] Oluwabusayo Adijat Bello and Komolafe Olufemi. Artificial intelligence in fraud prevention: Exploring techniques and applications challenges and opportunities. Computer Science & IT Research Journal, 5(6):1505–1520, 2024.
- [25] DOUBAO TEAM. Doubao-1.5-pro, January 2025. https://team.doubao.com/en/ special/doubao_1_5_pro. Accessed: April 8, 2025.
- [26] Google DeepMind. Introducing Gemini 2.0: Our New AI Model for the Agentic Era, December 2024. https://blog.google/technology/google-deepmind/googlegemini-ai-update-december-2024/. Accessed: April 8, 2025.
- [27] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.

