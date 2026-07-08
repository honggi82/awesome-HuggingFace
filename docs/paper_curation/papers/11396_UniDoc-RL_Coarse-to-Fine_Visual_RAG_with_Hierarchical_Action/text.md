# arXiv:2604.14967v2[cs.CV]17Apr2026

## UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards

###### Jun Wang∗, Shuo Tan∗, Zelong Sun∗, Tiancheng Gu, Yongle Zhao, Ziyong Feng Kaicheng Yang‡, Zhiwu Lu‡

GitHub https://github.com/deepglint/UniDoc-RL

HuggingFace https://huggingface.co/datasets/DeepGlint-AI/UniDoc-RL

#### Abstract

Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics essential for complex reasoning. To address this limitation, we propose UniDoc-RL, a unified reinforcement learning framework in which an LVLM agent jointly performs retrieval, reranking, active visual perception, and reasoning. UniDoc-RL formulates visual information acquisition as a sequential decision-making problem with a hierarchical action space. Specifically, it progressively refines visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping, allowing the model to suppress irrelevant content and attend to information-dense regions. For effective end-to-end training, we introduce a dense multi-reward scheme that provides task-aware supervision for each action. Based on Group Relative Policy Optimization (GRPO), UniDoc-RL aligns agent behavior with multiple objectives without relying on a separate value network. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations. Experiments on three benchmarks demonstrate that UniDoc-RL consistently surpasses state-of-the-art baselines, yielding up to 17.7% gains over prior RL-based methods.

#### 1 Introduction

Retrieval-Augmented Generation (RAG)(Guu et al., 2020; Yu et al., 2023; Yao et al., 2022) improves Large Language Models (LLMs)(Bai et al., 2025a) by grounding generation in external knowledge. Recent advances in multimodal document understanding extend this paradigm to the visual domain, allowing Large Vision-Language Models (LVLMs)(Bai et al., 2025c; Hurst et al., 2024) to reason over external visual evidence(Yu et al., 2024; Faysse et al., 2024). Unlike text, visual documents (e.g., charts and scanned reports) are highly dense and contain substantial redundant background noise, making visual RAG fundamentally more difficult. More recently, several studies (Jin et al., 2025; Wang et al., 2025b) apply Reinforcement Learning (RL) after supervised fine-tuning to further improve visual RAG performance. Nevertheless, developing effective visual RAG systems remains challenging.

As illustrated in Figure 1, the success of a visual RAG system hinges on three critical factors: (1) Accurate Retrieval: Erroneous retrieval introduces irrelevant visual context, leading to hallucinations and answer failures; (2) Effective Visual Utilization: Given that images contain dense and often redundant semantic information, the ability to filter noise and focus on key regions is paramount for correct guidance; (3) Reasonable Rewards for Optimization: Effective end-to-end optimization necessitates a dense reward formulation that resolves the credit assignment problem across distinct stages, such as retrieval, selection, active cropping, and reasoning. Existing approaches, however, often address these factors in isolation or fail to model their interdependencies, thereby limiting overall system efficacy.

∗ Equal Contribution. ‡ Corresponding Author.

(c) Reasonable Rewards

(a) Accurate Retrieval (b) Effective Visual Utilization

[Figure 3]

Search Action

Outcome Reward

CriticalFactors

Indirect Optimization

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Rollout Module

Full Image

Search

###### LVLM

Selection

[Figure 8]

Confused!

[Figure 9]

Confused!

LVLM

not

The image is not related to the question.

Visual Perception

The image is vague!

related vague

Direct Optimization

Visual Perception Action

Selection Action

Dense Multi-Rewards

UniDoc-RL

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

...

SelectBased

IoUBased

NDCGBased

LVLM

Clear!

Clear!

- Figure 1: Three critical factors for Visual RAG. UniDoc-RL address these challenges through the (a) Precise Selection action to bridge the semantic gap between coarse retrieval and reasoning, and an (b) Active Visual Perception action to focus on information-dense regions, both optimized via a (c) Dense multi-reward mechanism.

For retrieval, existing methods (Yu et al., 2024; Wang et al., 2025b) typically use decoupled architectures and generic similarity scores from off-the-shelf retrievers. While sufficient for coarse filtering, these scores often fail to capture the task-specific semantics required for complex reasoning. They are also usually static and cannot adapt to query semantics or conversational context, which is particularly problematic in multi-turn settings. For visual utilization, prior approaches (Yu et al., 2024; Jin et al., 2025) mainly adopt a passive visual consumption paradigm, directly encoding full images into the model context. This design overlooks the hierarchical nature of visual understanding, retains large amounts of irrelevant background content, and wastes context capacity. For optimization, existing RL-based methods (Jin et al., 2025; Narayan et al., 2025) predominantly use sparse rewards defined only on final outcomes. Such supervision provides no explicit credit assignment for intermediate decisions such as retrieval and cropping, making optimization effectively black-box and less effective for improving the model’s internal decision process.

To address these limitations, we propose UniDoc-RL, a unified multimodal reinforcement learning framework in which an LVLM agent jointly performs retrieval, selection, active visual perception, and reasoning. To bridge the gap between generic retrieval signals and task-specific reasoning requirements, UniDoc-RL introduces a hierarchical action space. It first leverages external tools for efficient coarse-grained retrieval, and then applies an LVLM-driven precise selection action to rerank candidates based on fine-grained semantic alignment with the query. This design effectively filters document-level noise before downstream reasoning. To further improve visual utilization, we introduce an active visual perception action. Rather than passively encoding full images, the agent learns to perform crop and zoom operations that actively localize informative regions and extract high-value visual evidence. This coarse-to-fine perception strategy resembles human visual attention, reduces redundancy, and preserves high-resolution details. To address sparse supervision, we design a dense multi-reward system that provides stage-specific training signals throughout the decision process. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations, providing a strong foundation for RL training. Experiments on three benchmarks demonstrate that UniDoc-RL consistently outperforms state-of-the-art baselines, delivering up to 17.7% gains over prior RL-based methods.

Our main contributions are summarized as follows: (1) We propose UniDoc-RL, a unified reinforcement learning framework for visual document RAG that jointly models retrieval, reranking, active visual perception, and reasoning within a single decision process. (2) We build and release a high-quality dataset of diverse reasoning trajectories with fine-grained action annotations, providing a valuable resource for future research on reinforcement learning for visual RAG. (3) We conduct extensive experiments on three benchmarks to demonstrate that UniDoc-RL consistently outperforms state-of-the-art baselines.

#### 2 Related Work

Vision-based Retrieval-Augmented Generation. RAG has emerged as a dominant paradigm for knowledge-intensive tasks (Lewis et al., 2020; Gao et al., 2023; Chen et al., 2024a). While traditional text-based RAG focuses on interacting with search engines (Wu et al., 2025; Chen et al., 2024b;c), the proliferation of digital documents has shifted focus toward multimodal RAG. Early visual RAG approaches, such as ColPali (Faysse et al., 2024) and VisRAG (Yu et al., 2024), primarily rely on embedding-based retrieval to align textual queries with visual documents. More recent works have evolved into agentic frameworks (Wang et al., 2025a; Cho et al., 2024; Jiang et al., 2024), utilizing external tools for more precise information extraction. Notably, VRAG-RL (Wang et al., 2025b) introduces reinforcement learning to incorporate visual perception actions into the RAG pipeline. In this work, we further introduce a “Search-Select-Perceive” coarse-to-fine action space, which bridge the semantic gap between generic retrieval and fine-grained reasoning, enabling the model to progressively filter noise and focus on critical visual evidence.

Reinforcement Learning for Multimodal Reasoning. RL has proven instrumental in enhancing the reasoning capabilities of LLMs (Guo et al., 2025; Jaech et al., 2024; Meng et al., 2024; Schulman et al., 2017). Recently, this success has been extended to LVLMs (Chen et al., 2025b; Meng et al., 2025; Liu et al., 2025) and large model-driven agents (Wang et al., 2025c), particularly for complex tasks requiring multi-step interactions (Jiang et al., 2025; Li et al., 2025). Despite these advancements, existing RL frameworks typically rely on sparse outcome-based rewards, which struggle to effectively guide intermediate steps like retrieval or cropping due to the credit assignment problem. In this work, we design specific rewards for retrieval relevance, selection accuracy, and cropping precision, ensuring that every stage of the pipeline is explicitly supervised and synergistically optimized.

#### 3 UniDoc-RL

###### 3.1 Problem Formulation

Given a query Q and a large-scale image corpus C = {c1, c2, . . . , cN}, where a subset of images contains the evidence required to answer Q, our goal is to retrieve the relevant images, identify the key visual evidence, and generate the correct answer y.

We cast this process as a sequential decision-making problem within a Thought-Action-Observation (T, A,O) framework. At each time step t, the model acts as an agent parameterized by a policy πθ. Given the initial query Q and the interaction history Ht−1, the agent generates a thought Tt and an action At. After executing At, it receives an observation Ot from the environment. The interaction proceeds iteratively until the agent determines that sufficient information has been collected to generate the final answer.

###### 3.2 Action Definition

As illustrated in Figure 2, UniDoc-RL operates over a hierarchical action space that progressively narrows the visual search space: (1) Image Search, which retrieves a broad set of candidates from the external corpus; (2) Precise Selection, which semantically filters irrelevant images; and (3) Visual Perception, which focuses on fine-grained, information-dense regions within the selected images.

Image Search Action. The process begins with a coarse-grained retrieval step. At time step t, the policy πθ generates an action At based on the interaction history Ht−1. We parse At to extract a search query q enclosed by the special tokens <search> and </search>. We then invoke an external retrieval function, denoted by Search(·, ·), to retrieve a set of candidate images from the corpus C as the observation Ot:

Ot = Search(q, C) (1) where Ot denotes the initial pool of retrieved visual documents.

- (a) Search Action Space (e) Reinforcement Learning Framework
- (b) Select Action Space

[Figure 18]

(c) Visual Perception Action Space

<search> ? </search>

Instructions + Human Query

Search Engine

Select & Crop & Zoom

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Rollout Module:

Interaction with external environment

Search Engine

[Figure 23]

Policy Model (LVLMs)

[Figure 24]

Policy Model

Select Function

[Figure 25]

<bbox> </bbox>

Crop Function

<think>

[Figure 26]

<select>

- 1. The ﬁrst image

shows……

- 2. The second…...

</select>

Trjs: H1, H2, H3, …, Hg

</think>

(d) Multi-Reward Function

Prediction GT.

Trajectories:

Answer Reference Model

Golden Unrelated

Multi-Reward Function

- 1.<search>…</search>
- 2.<select>…</select>
- 3.<bbox>…</bbox>
- 4.<answer>…</answer>

Stackec retrieval results in Trjs.

LLM-based Evaluation Model

kkkk

###### KL

IoU-based Reward

Group Advantage

…

rankk

00

Pattern and Action Veriﬁcation

00

True/False

Update Policy Model

Adv1, …, Advg

NDCGNDCG Function

Pattern Reward Outcome Reward

Selection Reward

Retrieval-based Reward

- Figure 2: Overview of UniDoc-RL. (a)(b)(c) demonstrates the “Search-Select-Perceive” coarse-to-fine action space. (d) is the specially designed reward for UniDoc-RL. (e) shows the interaction process between model and external environment, as well as the implementation of the GRPO algorithm.

Precise Selection Action. External retrieval tools often rely on shallow matching signals and therefore may fail to capture the fine-grained semantics required for complex reasoning. To bridge this gap, UniDoc-RL introduces an LVLM-based selection mechanism. The model evaluates the relevance of each candidate in Ot with respect to the interaction history and generates the next action At+1. Specifically, we extract the indices of relevant images, denoted by I, from the tokens <select> and </select>. The refined observation Ot+1 is then obtained by filtering the candidate set:

Ot+1 = Select(Ot, I) (2)

where Select(·, ·) retains only the images deemed semantically relevant by the LVLM, thereby reducing noise for downstream reasoning.

Visual Perception Action. Visual scenes often contain dense semantic content, while user queries typically concern only specific local regions. Therefore, we introduce a Visual Perception action that enables the model to actively localize and focus on essential information. The model generates an action At+2 that specifies a target region R, delimited by the tokens <bbox> and </bbox>. We then apply a perception function, denoted by VP(·, ·), which performs region selection, cropping, and adaptive zooming to extract the region of interest:

Ot+2 = VP(Ot+1, R) (3)

This operation converts the selected images into high-resolution, query-focused visual observations and effectively removes redundant content.

###### 3.3 Reward Function Tailored

Unlike traditional RL methods that rely solely on sparse outcome-based rewards, UniDoc-RL employs a comprehensive multi-reward mechanism. This dense reward system provides finegrained feedback at each stage of the pipeline.

Image Search Reward. High-quality retrieval is the foundation of accurate reasoning. To optimize this process, we employ the Normalized Discounted Cumulative Gain (NDCG) metric. Consider a trajectory containing m retrieval actions, where the i-th action returns a set of k candidate images

Ccandi = {c0i , c1i , . . . , cik}. As shown in Figure 2, we aggregate these candidates into a unified list Ctraj by interleaving them based on their rank:

Ctrj = {c10, . . . , c0m, c11, . . . , c1m, . . . , c1k, . . . , cmk } (4) We then compute the NDCG score:

rir = NDCG(Ctrj,Crel) (5)

where NDCG(·, ·) denotes the NDCG score function, and Crel is the collection of relevant golden images.

Precise Selection Reward. After coarse retrieval, the model must select the most relevant image to minimize noise. For the i-th retrieval step, let cisel be the image selected by the LVLM from Ccandi . The selection reward is defined as:

1 if cisel ∈ Crel 0 if cisel ∈/ Crel

rseli =

(6)

However, when Ccandi contains no ground-truth images, the standard reward remains consistently zero regardless of the selection. This results in a non-informative signal that fails to provide dis-

criminative feedback, making it difficult for the model to learn relative relevance among candidates. To address this, we employ a pseudo-supervision strategy: we designate the top-ranked candidate

from the initial retrieval, c0i , as a pseudo-positive target. The final selection reward is the average over all m steps:

m

1 m

rseli . (7)

### ∑

rsel =

i=1

Visual Perception Reward. To encourage accurate localization of key information, we design a reward based on the Intersection over Union (IoU) between predicted crops and ground-truth

regions. Let Bˆ = {bˆ1, . . . , bˆM} be the set of predicted bounding boxes and B∗ = {b1∗, . . . , b∗N} be the ground-truth boxes. As shown in Figure 2, we compute the IoU for each predicted box bˆj against the best-matching ground-truth box: rcrop = IoU(B∗, Bˆ)

Pattern Reward. To ensure the structural integrity of the generated thought-action sequences, we introduce a rule-based pattern reward, rpat. This reward checks whether the generated trajectory H adheres to the predefined format constraints (e.g., correct usage of XML tags like <search>, <select>, <bbox>). It penalizes malformed outputs, ensuring the model produces valid, executable actions.

Outcome Reward. The ultimate goal is to generate the correct answer. We employ a model-based outcome reward rans to evaluate the final response quality:

rans = RM(Agen, Aref, Q) (8)

where RM is a reward model that compares the generated answer Agen with the reference Aref. This sparse signal aligns the entire decision process with the final reasoning objective.

The final reward rtotal is defined as:

rtotal = λ1rpat + λ2rir + λ3rsel + λ4rcrop + λ5rans (9) where λ1, . . . , λ5 are hyperparameters to balance the relative importance of each term.

###### 3.4 Multi-Round Generation with Hierarchical Actions

Complex reasoning tasks often require multiple steps of information gathering and refinement. As detailed in Algorithm 1, UniDoc-RL interacts with the external environment iteratively. In each round, the model generates a thought-action sequence, executes the action, and receives an observation (e.g., retrieved images or cropped regions). Crucially, to align with the LVLM’s pre-training data distribution where visual inputs typically appear in user prompts, we insert the observation into the conversation history under the “User” role. This design ensures that the model can effectively process the new visual information without disrupting its internal representations.

Algorithm 1 Iterative Interaction Process in UniDoc-RL

Input: Input query Q, Policy model πθ, External environment E, Max steps Tmax. Output: Final trajectory H.

- 1: Initialize history H ← {(Role : User,Content : Q)}, step t ← 0
- 2: while t < Tmax do
- 3: Generate response At ∼ πθ(· | H)
- 4: Append (Role : Assistant,Content : At) to H
- 5: if <search></search> in At then
- 6: Extract search query q ← Parse(At) and Retrieve candidates Ccand ← Search(q, C)
- 7: Observation Ot ← Format(Ccand)
- 8: else if <select></select> in At then
- 9: Extract index idx ← Parse(At) and Select image Isel ← Ccand[idx]
- 10: Observation Ot ← Isel
- 11: else if <bbox></bbox> in At then
- 12: Extract box R ← Parse(At) and Crop image Ivp ← VP(Isel, R)
- 13: Observation Ot ← Ivp
- 14: else if <answer></answer> in At then
- 15: return H {Terminate upon answer generation}
- 16: end if
- 17: Append (Role : User,Content : Ot) to H {Inject observation as user input} and t ← t + 1
- 18: end while
- 19: return H

#### 4 Experiments

###### 4.1 Training Data Synthesis

Data Source. To construct a diverse and comprehensive training corpus, we aggregate samples from several public benchmarks, including SlideVQA (Tanaka et al., 2023), Double Bench (Shen et al., 2025), VisR-Bench (Chen et al., 2025a), DocBench (Zou et al., 2024), and DUDE (Landeghem

- et al., 2023). We implement a rigorous multi-stage filtering pipeline to ensure high data quality. Consequently, we curate a final dataset comprising 12,621 samples for SFT and 5,537 samples for RL.

Automated Trajectory Synthesis. To acquire high-quality supervision that demonstrates effective coordination between retrieval, selection, and perception, we employ Qwen3-VL-235B (Bai et al., 2025a) as a teacher agent to synthesize reasoning trajectories. While the generation of search and selection actions follows the standard interaction loop, the construction of Visual Perception actions requires specific handling to ensure precision. Therefore, we introduce an intermediate layout analysis step and use the expert document parsing tool Mineru (Niu et al., 2025) to detect layout elements and generate a set of candidate bounding boxes, which represent potential regions of interest (ROIs). Please refer to the Appendix for more details.

4.2 Training Pipeline and Implementation Details

To initialize the model with fundamental reasoning capabilities and ensure strict adherence to the defined action formats, we first perform SFT as a cold start. We conduct SFT on llama-factory (Zheng

- et al., 2024), with full parameter fine-tuning and cosine learning scheduler with a warmup ratio of 0.1. We then employ Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to fine-tune the model using the multi-reward system defined in Section 3.3. We conduct GRPO on verl (Sheng et al.,

- 2024) with the group size setted to 5. All the experiments are conducted on 8 NVIDIA A100 80G GPUs. Please refer to Appendix for more detailed hyperparameters information.

4.3 Evaluation Datasets and Baselines We evaluate performance on three challenging, visually rich benchmarks: ViDoSeek (Wang et al.,

- 2025a), SlideVQA (Tanaka et al., 2023), and MMLongBench (Ma et al., 2024). We employ an

###### METHOD Single-hopSLIDEVQAMulti-hop ExtractionVIDOSEEKLogic Text TableMMLChartONGBENCHFigure Layout

OVERALL Qwen2.5-VL-3B-Instruct

|Vanilla RAG ReAct Search-R1|15.1 12.1<br><br>11.8 9.9<br><br>17.5 13.8<br><br>|8.8 14.3 5.3 7.4 13.3 20.7<br><br>|3.9 5.1 1.7 3.1 2.5 6.5 3.7 3.9 5.2 2.5 3.4 3.2 4.5 4.1 6.8<br><br>|11.2 8.4 14.1|
|---|---|---|---|---|
|Vanilla RAG ReAct Search-R1-VL VRAG-RL|19.4 12.2 15.7 10.9 26.3 20.1 65.3 38.6<br><br>|10.1 17.3<br><br>6.7 14.2 20.1 29.8 63.1 73.8<br><br>|2.2 4.1 5.2 4.7 4.3 2.7 3.6 3.4 3.1 5.1 8.5 7.8 7.9 9.3 7.6 22.7 16.1 21.9 21.4 19.5<br><br>|13.2 10.9 21.3 53.5|
|UniDoc-RL|82.2 63.1<br><br>|78.6 75.9<br><br>|43.9 33 27.3 50.7 40.6<br><br>|71.0|

Qwen2.5-VL-7B-Instruct

|Vanilla RAG ReAct Search-R1|26.1 10.6 21.2 13.3 28.4 19.7<br><br>|24.7 30.9 14.3 21.3 20.8 30.6|8.5 5.4 11.7 4.4 3.3<br><br>5.9 5.1 7.3 5.5 1.7<br><br>9.9 6.0 7.9 10.1 5.9<br>|20.9 15.8 22.2<br><br>|
|---|---|---|---|---|
|Vanilla RAG ReAct Search-R1-VL VRAG-RL<br><br>|29.1 17.4 34.8 20.4 48.3 42.3 69.3 43.1<br><br>|26.4 41.3<br>27.5 42.1 40.5 50.3 60.6 74.8<br>|13.1 14.7 15.9 4.3 7.6 10.1 12.4 10.2 6.2 7.1<br><br>19.9 13.4 12.9 11.4 10.2 26.1 26.3 24.8 25.9 21.2<br><br>|24.2 26.9 37.4 57.1|
|UniDoc-RL|86.3 68.9<br><br>|80.3 78.7<br><br>|51.5 39.6 32.2 50.0 40.6<br><br>|74.8|

Table 1: Main Results. The best performance are marked in bold. SlideVQA and ViDoSeek mainly focus on reasoning type, while MMLongBench focuses on the visual type of reference content. OCR-based ( ) RAG and purely visual ( ) RAG are evaluated with the same prompt and setting. evaluation model to assess the correctness of generated answers, which returns a binary score. The overall accuracy rate is computed as the evaluation metric.

For baseline comparisons, following (Wang et al., 2025b), we select several representative methods spanning text-based and vision-based approaches: (1) Vanilla RAG (Faysse et al., 2024) directly uses the original question as a search query, after which LVLMs perform inference on the retrieved results. (2) ReAct (Yao et al., 2022) follows a think-then-act paradigm, where the model iteratively performs query rewriting, information retrieval, and reasoning. (3) Search-R1(-VL) (Jin et al., 2025) is adapted from the Search-R1 baseline, with experimental settings aligned across all methods to ensure fair comparison. (4) VRAG-RL (Wang et al., 2025b) serves as our most direct baseline, as it also incorporates a visual perception mechanism within an RL framework.

###### 4.4 Main Results

Table 1 presents a comparative analysis of UniDoc-RL against existing RAG approaches. We summarize the key observations as follows: (1) Limitations of OCR-based RAG. OCR-based methods, which rely solely on extracted textual content, struggle with visually intensive tasks. This performance gap stems from the inevitable loss of critical visual cues (such as spatial layout and geometric relationships) during the text extraction process. (2) Efficacy of Visual RAG. Compared to OCRbased baselines, purely visual RAG methods demonstrate significant improvements by preserving complete visual information. This validates that LVLMs have evolved to effectively capture both textual semantics and spatial relationships within images, often surpassing the capabilities of traditional OCR pipelines. (3) Superiority of UniDoc-RL. RL-based methods consistently outperform supervised baselines across all datasets. Notably, UniDoc-RL achieves state-of-the-art performance, surpassing the strong competitor VRAG-RL by margins of 17.5% and 17.7% on 3B and 7B models, respectively. We attribute this substantial gain to two factors: the Precise Selection action, which filters noise to provide a high-quality context, and the Visual Perception action, which enables fine-grained attention. Furthermore, the multi-reward mechanism ensures synergistic optimization across all stages. (4) Robustness Across Diverse Tasks. The datasets present distinct challenges: SlideVQA and ViDoSeek require complex multi-hop reasoning, while MMLongBench demands precise attention to fine-grained visual details. UniDoc-RL yields consistently superior performance across diverse benchmarks, demonstrating its robustness and indicating that the proposed multi-reward RL framework effectively enhances both logical reasoning and visual perception.

#### 5 Ablation Study

- 5.1 Impact of Hierarchical Actions.

|ACTION SPACE Select VP<br><br>|ACCURACY SVQA VDS MMLB Overall<br><br>|
|---|---|
|✗ ✗ ✓ ✗ ✓ ✓<br><br>|71.8 72.8 37.0 66.6<br><br>76.0 76.4 37.3 70.0<br><br>77.2 77.4 38.0 71.0<br><br><br>|

Table 2: Ablation for Different Actions.

- In Table 2, we isolate the contributions of each component in UniDoc-RL with Qwen2.5-VL-3BInstruct. We have the following insights: (1) Efficacy of Precise Selection. Incorporating the Precise Selection action leads to substantial performance gains. This confirms that the selection action significantly improves the recall of relevant images, effectively bridging the semantic gap between the retriever and the reasoner (as further illustrated in Figure 3). (2) Importance of Visual Perception. The addition of the Visual Perception action further boosts performance, which validates necessity of the “active cropping” mechanism and the effectiveness of our IoU-based reward in guiding the model to localize key evidence. (3) Task-Specific Adaptability. We observe that the impact of specific modules varies by task nature. The Selection action yields the most pronounced improvements on reasoning-heavy datasets (SlideVQA and ViDoSeek), suggesting that context quality is paramount for complex reasoning. Visual Perception action provides larger gains on MMLongBench, indicating its critical role in tasks requiring fine-grained visual discrimination.

5.2 Ablation on Multi-Rewards.

- In Table 3, we conducted an incremental ablation study for our reward design. We observe the following: (1) Benefit of Dense Supervision. Compared to the baseline variant which relies solely on sparse outcome-based and pattern rewards, our proposed multi-reward framework (Full) achieves consistent performance improvements across all three datasets. This demonstrates that dense, step-wise supervision effectively optimizes the accuracy of intermediate actions, preventing error propagation and guiding the model toward more robust reasoning paths. (2) Task-Reward Alignment.

Corroborating the findings in Table 2, we observe that the inclusion of the Selection Reward significantly boosts performance on reasoningheavy datasets, whereas the Visual Perception Reward yields the most notable improvements on MMLongBench. This further confirms that tailoring rewards to specific sub-tasks is essential for diverse reasoning requirements.

|REWARD|ACCURACY SVQA VDS MMLB Overall<br><br>|
|---|---|
|Vanilla + rir + rir + rsel + rir + rsel + rvp<br><br>|75.9 76.7 36.8 69.9<br><br>76.4 76.2 36.8 70.1<br><br><br>76.6 77.8 36.6 70.6<br><br>77.2 77.4 38.0 71.0<br><br><br>|

Table 3: Ablation on Multi-Rewards.

#### 6 Analysis

###### 6.1 Selection Action Facilitates Better Retrieval Results.

UniDoc-RL substantially improves the retrieval hit rate of ground-truth images through the Precise Selection action, which helps build a more accurate and informative context for generation. As shown in Figure 3, adding the selection step consistently improves recall over coarse search alone across all three benchmarks. Specifically, recall on SlideVQA and ViDoSeek increases from 79.7% and 74.8% to 85.0% and 85.7%, respectively, while MMLongBench improves from 48.9% to 52.7%. These results show that LVLM-based selection effectively narrows the semantic gap in coarse retrieval by filtering out irrelevant candidates and provides clearer visual evidence for downstream reasoning.

Figure 3: Retrieval Recall before and after adding the selection action.

###### 6.2 SFT Enables Tool Use, While RL Refines the Tool-Selection Behavior

RL Encourages Active Information Seeking. As shown in Figure 4, the SFT model exhibits a conservative strategy with a low crop frequency (e.g., only 2.4% on ViDoSeek), significantly lagging behind the Teacher model. This suggests that SFT suffers from exposure bias, often defaulting to passive full-image reading. In contrast, UniDoc-RL demonstrates a more active perception pattern. After RL training, the crop frequency increases across all datasets (e.g., rising to 13.3% on MMLongBench), closely aligning with the Teacher’s behavior. Notably, on MMLongBench, which demands fine-grained visual attention, UniDoc-RL matches the Teacher’s frequency, indicating it has successfully learned to invoke perception tools when facing detail-intensive queries.

Figure 4: Visual Perception Action Frequency across SFT, RL, and Teacher models on three benchmarks.

Supervised Fine-Tuning (SFT)

Question: Question:

[Figure 32]

[Figure 33]

Who is the founder of WiTOWN in Clan Alibaba presentation about 130+ Startups?

What is the URL for the PRO Perk from UserVoice mentioned in the PageLever demo?

Reinforcement Learning (RL)

[Figure 34]

Question:

[Figure 35]

Question:

In the \"How We Perceive Colors\" chart, which color is associated with security and intelligence?

How many black arrows are part of DISCIPLINED AGILE logo ?

Prediction Golden Box

- Figure 5: Qualitative Comparison of Visual Perception Action generated by UniDoc-RL before and after RL fine-tuning.

RL Improves Action Quality and Precision. Beyond the crop frequency, we observe a qualitative shift in how the model crops. As visualized in Figure 5, the SFT model frequently generates“trivial crops” that encompass the entire image or irrelevant backgrounds, failing to provide additional information gain. Conversely, UniDoc-RL, guided by the IoU-based perception reward and task-specific feedback, learns to execute “precise crops” that tightly bound the regions of interest. This shift from coarse, meaningless actions to focused, high-utility perception confirms that our multi-reward RL framework effectively optimizes both the decision to act and the execution of the action.

6.3 Case Study

- Figure 6 in the Appendix presents two representative examples that illustrate UniDoc-RL’s adaptive reasoning process. In Case 1, the model retrieves candidate images, selects the one containing relevant training loss curves, and then actively crops a specific sub-region to zoom into a small chart which correctly identifies that the loss spikes around the 150k step. This demonstrates the necessity of the Visual Perception action for extracting fine-grained details from complex figures. In Case 2, the model retrieves and selects a table that is already clearly legible, and intelligently skips the crop action, directly extracting the answer. These two cases jointly show that UniDoc-RL learns an adaptive strategy through RL training.
- 7 Conclusion

We present UniDoc-RL, a unified reinforcement learning framework for visual RAG that orchestrates retrieval, reranking, active visual perception, and reasoning within an LVLM agent. Through a hierarchical action space, the model progressively filters visual evidence from coarse retrieval to finegrained selection and region cropping, effectively suppressing noise and focusing on key information. To overcome the limitations of sparse outcome-based rewards, a dense multi-reward mechanism provides stage-aware supervision, enabling end-to-end optimization via GRPO. Additionally, we curate and publicly release a high-quality dataset of reasoning trajectories with fine-grained action annotations to support future research. Extensive experiments on three benchmarks demonstrate that UniDoc-RL consistently outperforms state-of-the-art baselines.

#### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025c.

Jian Chen, Ming Li, Jihyung Kil, Chenguang Wang, Tong Yu, Ryan Rossi, Tianyi Zhou, Changyou Chen, and Ruiyi Zhang. Visr-bench: An empirical study on visual retrieval-augmented generation for multilingual long document understanding, 2025a. URL https://arxiv.org/abs/2508. 07493.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In AAAI, pp. 17754–17762, 2024a.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V, 2025b. Accessed: 2025-02-02.

Zehui Chen, Kuikun Liu, Qiuchen Wang, Jiangning Liu, Wenwei Zhang, Kai Chen, and Feng Zhao. Mindsearch: Mimicking human minds elicits deep ai searcher. arXiv preprint arXiv:2407.20183, 2024b.

Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language models. arXiv preprint arXiv:2403.12881, 2024c.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding. arXiv preprint arXiv:2411.04952, 2024.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In ICLR, 2024.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1), 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In ICML, pp. 3929–3938. PMLR, 2020.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Chaoyou Fu, Guanglu Song, et al. Mmsearch: Benchmarking the potential of large models as multi-modal search engines. arXiv preprint arXiv:2409.12959, 2024.

Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei Han. Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning. arXiv preprint arXiv: 2503.00223, 2025. URL https: //arxiv.org/abs/2503.00223.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Jordy Van Landeghem, Rubén Tito, Łukasz Borchmann, Michał Pietruszka, Paweł Józiak, Rafał Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Ackaert, Ernest Valveny, Matthew Blaschko, Sien Moens, and Tomasz Stanisławek. Document understanding dataset and evaluation (dude), 2023. URL https://arxiv.org/abs/2305.08455.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. NeurIPS, 33:9459–9474, 2020.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366, 2025.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. NeurIPS, 37:124198–124235, 2024.

Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, and Zhe Gan. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint arXiv:2510.12801, 2025.

Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al. Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing. arXiv preprint arXiv:2509.22186, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Wenxuan Shen, Mingjia Wang, Yaochen Wang, Dongping Chen, Junjie Yang, Yao Wan, and Weiwei Lin. Are we on the right way for assessing document retrieval-augmented generation?, 2025. URL https://arxiv.org/abs/2508.03644.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In AAAI, pp. 13636–13645, 2023.

Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. arXiv preprint arXiv:2502.18017, 2025a.

Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. Vrag-rl: Empower vision-perception-based rag for visually rich information understanding via iterative reasoning with reinforcement learning. arXiv preprint arXiv:2505.22019, 2025b.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning, 2025c. URL https://arxiv.org/abs/2504.20073.

Weiqi Wu, Shen Huang, Yong Jiang, Pengjun Xie, Fei Huang, and Hai Zhao. Unfolding the headline: Iterative self-questioning for news retrieval and timeline summarization. arXiv preprint arXiv:2501.00888, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2022.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multimodality documents. arXiv preprint arXiv:2410.10594, 2024.

Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. Augmentation-adapted retriever improves generalization of language models as generic plug-in. arXiv preprint arXiv:2305.17331, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. ACL. URL http://arxiv.org/abs/2403.13372.

Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. Docbench: A benchmark for evaluating llm-based document reading systems, 2024. URL https://arxiv.org/abs/2407.10701.

#### A Training Data Synthesis

###### A.1 Data Collection

To construct a diverse and comprehensive training corpus, we aggregate samples from several public benchmarks spanning a wide range of document types, languages, and reasoning requirements:

- • SlideVQA (Tanaka et al., 2023) is a multi-image document VQA dataset containing over 2,600 slide decks, 52K images, and 14.5K QA pairs. It requires cross-slide information integration for single-hop, multi-hop, and numerical reasoning.
- • DoubleBench (Shen et al., 2025) is a large-scale multilingual multimodal document RAG dataset comprising 3,276 documents (72,880 pages) and 5,168 single-hop and multi-hop queries across 6 languages, covering PDFs, scanned documents, slides, and HTML pages.
- • VisRAG-Bench (Chen et al., 2025a) is a multilingual multimodal document retrieval dataset built from Common Crawl PDFs. It contains over 1,200 documents and 35,000 QA pairs across 16 languages, covering chart, text, and table-based questions.
- • DUDE (Landeghem et al., 2023) is a multi-domain, multi-page document understanding dataset spanning healthcare, legal, technical, and financial sectors. It includes both digitally-born and scanned documents with diverse layouts, time spans, and question types, simulating realistic document analysis scenarios.
- • DocBench (Zou et al., 2024) is a benchmark for evaluating LLM-based document reading systems, containing 229 real-world documents and 1,102 QA pairs across academic, financial, government, legal, and news domains. QA pairs are generated via human annotation combined with GPT-4 synthesis, and categorized into text-based, multimodal, metadata, and unanswerable types.

###### A.2 Automated Trajectory Synthesis.

To acquire high-quality supervision that demonstrates effective coordination between retrieval, selection, and perception, we employ a strong proprietary LVLM (e.g., Qwen3-VL-235B (Bai et al., 2025a)) as a teacher agent, denoted as πteacher, to synthesize reasoning trajectories. Formally, at each time step t, the teacher model observes the current history Ht−1 and generates the optimal thought-action pair:

(Tt, At) ∼ πteacher(·|Ht−1) (10) While the generation of search and selection actions follows the standard interaction loop, the construction of Visual Perception actions requires specific handling to ensure precision. When the teacher selects a relevant image csel (via the Precise Selection action), we introduce an intermediate layout analysis step. We utilize an expert document parsing tool, Mineru (Niu et al., 2025), to detect layout elements and generate a set of candidate bounding boxes Bcand representing potential regions of interest (ROIs). The teacher model πteacher then evaluates these candidates to determine if fine-grained perception is necessary. If a specific region contains critical information absent in the global view, the teacher selects the optimal bounding box b∗ ∈ Bcand as the ground-truth crop action At+1. The cropped view is then processed as observation Ot+1, and the cycle continues.

###### A.3 Data Filtering

After performing trajectory synthesis as described in Section A.2, we apply a multi-stage filtering pipeline to ensure data quality:

Quality Filtering. We employ a strong teacher model (Qwen3-VL-235B) to verify the correctness of each synthesized trajectory by comparing the generated answer against the reference answer. Trajectories with incorrect final answers are discarded, ensuring that only factually accurate reasoning paths are retained for training.

Difficulty-Aware Filtering. To further reduce the proportion of trivially easy samples, we adopt a bootstrapped filtering strategy. We first fine-tune an intermediate SFT model using only the

|STAGE<br><br>|DATASET SlideVQA DoubleBench VisR-Bench DocBench DUDE Total<br><br>|
|---|---|
|SFT RL|8271 1274 1518 657 901 12621<br><br>2613 993 1630 301 - 5537|

###### Table 4: Data statistics for UniDoc-RL.

[Figure 41]

[Figure 42]

In the Hadoop Storage Systems table, which two workloads do not support HDFS?

At which step, the loss spikes when without QK-norm?

<think>

<think>The question asks for speciﬁc workloads that do not support HDFS, so the initial search must identify which workloads are explicitly listed as unsupported in that context.<\think> <search>Which two workloads do not support HDFS in the Hadoop Storage Systems table?<\search>

<think>The question asks about the speciﬁc step where loss spikes occur when QK-norm is not applied, requiring information on training dynamics or results related to this condition.<\think> <search>At which step does the loss spike when QK-norm is not applied?<\search>

<think>

<\think>

<\think>

<search>

<search>

<\search>

<\search>

<think>

<think>1. **Query Deconstruction:** ...identify which two workloads do not support HDFS... rows where HDFS column contains a red 'not supported'... count two such workloads.

<think>1. **Query Deconstruction:** ... QK-norm is not applied. This requires identifying a chart or graph that plots training loss over steps for a model trained *without* QK-norm......... increase (spike) in the loss occurs.

<think>

- 2. **Image-by-Image Analysis:**

- **Image 0:**..., - **Image 1:**..., - **Image 2:**...

- 3. **Final Justiﬁcation:** Image 0 is the optimal choice because it contains........a Chameleon-7B model without QK-nor... Image 1 discusses the issue in text but lacks the precise , and Image 2 is .......</think> <select>0</select>

- 2. **Image-by-Image Analysis:**

- **Image 0:**Shows a table with only two workloads...,

- 3. **Final Justiﬁcation:** Image 2 is the only image that contains the required information to answer the query, as it is the only one that displays workloads with a red 'not supported' indicator in the HDFS column. Images 0 and 1 …... Therefore, Image 2 is the optimal choice. <\think> <select>3</select>

</think>

<select> </select>

<think>

<think>The bounding box [216, 54, 347, 152] contains a line chart labeled '(b) An ablation with Chameleon-7B with and without QKNorm', which plots training loss over steps.Therefore, cropping and zooming into [216, 54, 347, 152] is necessary to ... </think> <bbox>[[216, 54, 347, 152]]</bbox>

<\think> <select> </select>

</think>

<think>The bounding box [39, 93, 773, 421] contains the entire Hadoop Storage Systems table, which clearly lists workloads and their support status for HDFS. The table includes columns for Workload, HBase, and HDFS. The rows for 'Random Write' and 'Updates' are marked with a red 'not supported' symbol, which directly answers the question. Since the relevant information is fully visible and legible within this bounding box, no cropping or zooming is needed to extract. </think> <information>Random Write and Updates</information>

<think>

<bbox> </bbox>

</think> <information> </information>

<think>

<think>The information from step 1 clearly identiﬁes that the loss spike occurs around the 150k step mark when QK-norm ….</think> <answer>The loss spikes around the 150k step when QK-norm is not applied.</answer>

<think>The retrieved information... identiﬁes 'Random Write' and 'Updates' as the two workloads not supported by HDFS... The answer is complete, directly relevant, and reliable...<\think> <answer>Random Write and Updates</answer>

<think>

</think>

<answer>

<\think> </answer>

</answer>

<answer>

###### Figure 6: Case Study for UniDoc-RL.

quality-filtered SlideVQA data, and then apply this model to generate answers for samples from the remaining datasets. Only those samples that (a) pass the quality filtering (i.e., the teacher-generated trajectory is correct) but (b) the intermediate SFT model answers incorrectly are retained. This procedure effectively removes samples that can already be solved by a partially trained model, yielding a more challenging and informative training set.

RL Data Curation. For the RL training set, we require samples that are challenging yet learnable. Specifically, we run the SFT model with high sampling temperature (τ) for 5 independent rollouts per query. We then select samples where the model successfully retrieves relevant documents but fails to produce the correct answer in at least one rollout. This strategy identifies instances where the reasoning or perception steps—rather than retrieval—are the bottleneck. Additionally, we prioritize samples that were filtered out during the quality filtering stage, as they are likely to represent hard cases that maximally benefit from RL optimization.

###### A.4 Data Statistics

After the trajectory synthesis and multi-stage filtering pipeline, we obtain a final dataset comprising 12,621 samples for Supervised Fine-Tuning (SFT) and 5,537 samples for Reinforcement Learning (RL). The detailed composition across source datasets is presented in Table 4.

#### B Case Study

Figure 6 presents two representative examples that illustrate UniDoc-RL’s adaptive reasoning process. In Case 1, the model retrieves candidate images, selects the one containing relevant training loss curves, and then actively crops a specific sub-region to zoom into a small chart which correctly identifies that the loss spikes around the 150k step. This demonstrates the necessity of the Visual Perception action for extracting fine-grained details from complex figures. In Case 2, the model retrieves and selects a table that is already clearly legible, and intelligently skips the crop action, directly extracting the answer. These two cases jointly show that UniDoc-RL learns an adaptive strategy through RL training.

##### C Hyperparameters The detailed hyperparameters we use during training are shown in Table 5 and Table 6.

Name Value Finetuning type Full

Freeze vision tower True Freeze multi-modal projector True

Freeze language model False

Cutoff len 40000 Epochs 3 Batch size 1

Gradient accumulation steps 2 Learning rate 1.0e-5

LR scheduler type cosine Warmup ratio 0.1

Table 5: Key hyperparameters for SFT.

Name Value Number of agent groups 5

Warmup steps ratio 0.285 Mini batch size 16 Micro batch size per GPU 1

Learning rate (Actor) 1.0e-6 KL loss coefficient 0.01 Tensor model parallel size 4

Total epochs 1

Max prompt length 40000 Max response length 1024

λ1:5 0.1,0.1,...,0.6 Table 6: Key hyperparameters for RL.

#### D Model-Based Reward

We employ a model-based reward to evaluate the quality and relevance of generated responses. Specifically, we utilize Qwen2.5-72B-Instruct (Bai et al., 2025b) as our reward model. The prompt used for the reward model is illustrated in Figure 7. Given the input query, reference answer, and generated response, the reward model assesses the correctness of the generated response and outputs a binary value (0 or 1) to represent the accuracy of the answer.

#### E Prompts

We present the prompts for training and testing in Figure 8 and the detailed prompts for data synthesis in Figures 9, 10, and 11.

Prompt for Data Synthesis - Image Search Action

## Role You are an intelligent search planning assistant responsible for analyzing the user's question and deciding on the initial search content.

## Instructions For the user's question, generate only the initial search query without detailed elaboration. The output must be a JSON object containing "think" and "search" ﬁelds.

- - "think": Brieﬂy explain in English why this initial search is needed, based directly on the original question.
- - "search": Provide only the initial search query string in English, in the form of a question, that directly reﬂects the user's original question.

## Constraints

- - The search query must be based strictly on the original question
- - Do not add speculative content or make assumptions beyond what is stated
- - Keep the search query concise and directly related to the question
- - Provide only the initial search query, no detailed or expanded queries

## Output You must provide the output in the following JSON format: {"think": "Brief explanation in English”,

"search": "Initial search query in English, in the form of a question”}

Prompt for Data Synthesis - Precise Selection Action

## Role You are an expert AI visual reasoning assistant. Your primary mission is to analyze a user's query against a set of images and identify the single most relevant image. Your response MUST be a single, valid JSON object. Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

## JSON Output Structure The JSON object must contain exactly two keys: `think` and `rerank`.

- 1. `think` (string): A detailed, step-by-step reasoning process that documents your entire analysis. This string must include:

- * **Query Deconstruction:** A brief analysis of the user's query to identify the key criteria.
- * **Image-by-Image Analysis:** An evaluation of EACH provided image against the query's criteria.
- * **Final Justiﬁcation:** A conclusive explanation stating why the selected image is the optimal choice and explicitly why the other images are less suitable.

- 2. `rerank` (string): The unique identiﬁer of the single best image number that matches the query (e.g., "0", "1”).

## Concrete Example …………

Prompt for Data Synthesis - RoI Region Selection

You will be given a **query** and an **image with pre-drawn red rectangular bounding boxes**. Each red rectangle corresponds to a distinct semantic layout region (e.g., a paragraph, chart, table, or title) and is labeled with a unique **index number starting from 0** (0, 1, 2, ...) in the image (**displayed vertically centered on the left side of each box**).

Your task is to:

- 1. **Understand the intent of the query**;
- 2. **Identify the red-bounded region(s) in the image that directly contain the information needed to answer the query**;
- 3. **Output the index number(s) of those region(s), and provide the answer to the query in the `answer` ﬁeld**.

**Follow these guidelines strictly**:

- - **The answer must be directly derived from the content of the selected region(s)**:
- - Do not select a region unless the answer can be **unambiguously inferred** from its content.
- - **Do NOT select regions that only contain the question text or repeat the query verbatim** — these do not provide evidence for the answer.
- - Only select regions that contain **supporting data, explanation, or visual information** (e.g., a paragraph stating a fact, a chart

showing a value, a table listing numbers).

- - **Rely solely on visible content**: Do not use external knowledge or make assumptions beyond what is shown in the selected region(s).

**Output Format Requirements**: Return a strictly valid JSON object with exactly three ﬁelds:

- - **`think`** (string): A concise explanation of your reasoning — describe the query's intent and **why the selected indexed region(s) contain the necessary information to answer it**.
- - **`indices`** (list of integers): A list of index numbers (e.g., `[0]`, `[1, 3]`) corresponding to the red-bounded regions **from which the answer is directly derived**. The indices must match the **0-based numbers visible in the image**.
- - **`answer`** (string): The **answer to the query**, extracted accurately and concisely from the content of the selected region(s).
- - Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

Prompt for Data Synthesis - RoI Region Conﬁrm

You will be given:

- - A natural language question (referred to as the "query");
- - An image;
- - A list of pre-deﬁned bounding boxes in the image, each represented by its pixel coordinates **[x1, y1, x2, y2]** (with the origin at the top-left corner). Your task is as follows:

- 1. **Select the relevant bounding box**: From the provided list of bounding boxes, select only those containing content directly useful for answering the query. Strictly exclude any irrelevant boxes. **If multiple bounding boxes contain content helpful for answering the query, choose the one that most directly and centrally addresses the question.**
- 2. **Explain your selection**: For the ﬁnally selected bounding box, use its exact coordinate tuple **[x1, y1, x2, y2]** as its unique identiﬁer. Describe the key semantic content within that region (e.g., text, table, chart, formula, etc.) and explain precisely how this content helps answer the query. **Do not refer to boxes using ordinal terms such as "ﬁrst," "second," or "the box on the left"—always identify them by their coordinates.**
- 3. **Output a structured result**: Return your response strictly in JSON format

Prompt for Data Synthesis - Precise Selection Action

Prompt for Data Synthesis - Image Search Action

## Role You are an intelligent search planning assistant responsible for analyzing the user's question and deciding on the initial search content.

## Role You are an expert AI visual reasoning assistant. Your primary mission is to analyze a user's query against a set of images and identify the single most relevant image. Your response MUST be a single, valid JSON object. Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

## Instructions For the user's question, generate only the initial search query without detailed elaboration. The output must be a JSON object containing "think" and "search" ﬁelds.

- - "think": Brieﬂy explain in English why this initial search is needed, based directly on the original question.
- - "search": Provide only the initial search query string in English, in the form of a question, that directly reﬂects the user's original question.

## Constraints

- - The search query must be based strictly on the original question
- - Do not add speculative content or make assumptions beyond what is stated
- - Keep the search query concise and directly related to the question
- - Provide only the initial search query, no detailed or expanded queries

## JSON Output Structure The JSON object must contain exactly two keys: `think` and `rerank`.

- 1. `think` (string): A detailed, step-by-step reasoning process that documents your entire analysis. This string must include:

- * **Query Deconstruction:** A brief analysis of the user's query to identify the key criteria.
- * **Image-by-Image Analysis:** An evaluation of EACH provided image against the query's criteria.
- * **Final Justiﬁcation:** A conclusive explanation stating why the selected image is the optimal choice and explicitly why the other images are less suitable.

- 2. `rerank` (string): The unique identiﬁer of the single best image number that matches the query (e.g., "0", "1”).

Prompt for Data Synthesis - Visual Perception Action Decision

You will receive the following inputs:

- - A natural language **query**
- - An **image**
- - A set of **bounding boxes**, each deﬁned by its pixel coordinates `[x1, y1, x2, y2]`

## Output You must provide the output in the following JSON format: {"think": "Brief explanation in English”,

## Concrete Example …………

**Core Task** Based on the given query, evaluate each bounding box individually: **If you think that cropping and zooming into the corresponding image region may help to answer the query more clearly and accurately**, then judge that this box requires processing.

"search": "Initial search query in English, in the form of a question”}

**Output Requirements** You must generate a strict JSON object containing the following four ﬁelds:

Prompt for Data Synthesis - RoI Region Selection

Prompt for Data Synthesis - RoI Region Rethink

- 1. **`crop`** (boolean)

- - `true`: If you think **at least one** bounding box would beneﬁt from cropping and zooming for answering the question.
- - `false`: Otherwise.

- 2. **`bbox`** (array)

- - List the **full coordinates** of all bounding boxes you judge as **requiring cropping and zooming**, in the format `[[x1, y1, x2, y2], ...]`.
- - If `crop` is `false`, this ﬁeld must be an empty array `[]`.

- 3. **`think`** (string)

- Provide a concise reasoning chain that must cover:

- a) For **each** bounding box (referenced by its coordinates), describe the key content it contains.
- b) For boxes you mark as **needing crop**: Explain why zooming into this region could provide more critical information.
- c) For boxes you mark as **not needing crop**: Explain how their current content is already sufﬁcient and directly supports the answer.

- 4. **`answer`** (string)

You will be given a **query** and an **image with pre-drawn red rectangular bounding boxes**. Each red rectangle corresponds to a distinct semantic layout region (e.g., a paragraph, chart, table, or title) and is labeled with a unique **index number starting from 0** (0, 1, 2, ...) in the image (**displayed vertically centered on the left side of each box**).

You will be given:

- - A natural language question (referred to as the "query");
- - An image;
- - A list of pre-deﬁned bounding boxes in the image, each represented by its pixel coordinates **[x1, y1, x2, y2]** (with the origin at the top-left corner). Your task is as follows:

Your task is to:

- 1. **Understand the intent of the query**;
- 2. **Identify the red-bounded region(s) in the image that directly contain the information needed to answer the query**;
- 3. **Output the index number(s) of those region(s), and provide the answer to the query in the `answer` ﬁeld**.

**Follow these guidelines strictly**:

- 1. **Select the relevant bounding box**: From the provided list of bounding boxes, select only those containing content directly useful for answering the query. Strictly exclude any irrelevant boxes. **If multiple bounding boxes contain content helpful for answering the query, choose the one that most directly and centrally addresses the question.**
- 2. **Explain your selection**: For the ﬁnally selected bounding box, use its exact coordinate tuple **[x1, y1, x2, y2]** as its unique identiﬁer. Describe the key semantic content within that region (e.g., text, table, chart, formula, etc.) and explain precisely how this content helps answer the query. **Do not refer to boxes using ordinal terms such as "ﬁrst," "second," or "the box on the left"—always identify them by their coordinates.**
- 3. **Output a structured result**: Return your response strictly in JSON format

- - **The answer must be directly derived from the content of the selected region(s)**:
- - Do not select a region unless the answer can be **unambiguously inferred** from its content.
- - **Do NOT select regions that only contain the question text or repeat the query verbatim** — these do not provide evidence for the answer.
- - Only select regions that contain **supporting data, explanation, or visual information** (e.g., a paragraph stating a fact, a chart

showing a value, a table listing numbers).

- - **Rely solely on visible content**: Do not use external knowledge or make assumptions beyond what is shown in the selected region(s).

**Output Format Requirements**: Return a strictly valid JSON object with exactly three ﬁelds:

- - **`think`** (string): A concise explanation of your reasoning — describe the query's intent and **why the selected indexed region(s) contain the necessary information to answer it**.
- - **`indices`** (list of integers): A list of index numbers (e.g., `[0]`, `[1, 3]`) corresponding to the red-bounded regions **from which the answer is directly derived**. The indices must match the **0-based numbers visible in the image**.
- - **`answer`** (string): The **answer to the query**, extracted accurately and concisely from the content of the selected region(s).
- - Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

- - If `crop` is `false`, output an accurate and complete ﬁnal answer based on the content of all bounding boxes.
- - If `crop` is `true`, this ﬁeld must be an empty string `""`.

**Key Notes**

- - The judgment criterion is **functional**: whether cropping could potentially aid in answering the question (e.g., to view blurry text, distinguish small objects, focus on key details), not merely to evaluate image quality.
- - Throughout the `think` ﬁeld, you **must use the exact coordinates** `[x1, y1, x2, y2]` to refer to bounding boxes. **It is prohibited to use** vague references like "the ﬁrst box", "the left box", or "the top box".

Prompt for Data Synthesis - Visual Perception Action Complete

Prompt for Data Synthesis - Visual Perception Action Decision

You are a professional visual reasoning AI. Your task is to process a user-input original image, a cropped image, and a question, then generate a structured JSON output. The cropped image is a screenshot of a speciﬁc bounding box region from the original image, showing only that region's content.

You will receive the following inputs:

- - A natural language **query**
- - An **image**
- - A set of **bounding boxes**, each deﬁned by its pixel coordinates `[x1, y1, x2, y2]`

Prompt for Training and Test

## Input Format:

**Core Task** Based on the given query, evaluate each bounding box individually: **If you think that cropping and zooming into the corresponding image region may help to answer the query more clearly and accurately**, then judge that this box requires processing.

Please strictly follow the steps below to answer the user's question:

- - Original Image: The complete original image
- - Cropped Image: A cropped region from the original image
- - Question: A text question that needs to be answered based on the image contents

## Output Requirements: The output must be a JSON object containing the following two keys:

- - "think": A reasoning chain that step-by-step explains how to derive the answer to the question from both the original image and the cropped image content. The reasoning should be logical, fact-based, and directly connect image details from both images to the question.
- - "answer": A concise and clear direct answer based on the reasoning.

## Output Format: ``` { "think": ..., "answer": ... } ``` ## Important Principles:

- - Always base descriptions and reasoning on factual image content; do not fabricate or assume content not present in the images.
- - Ensure correct JSON format with exact key names: think, answer.
- - Use concise and accurate language, making the reasoning process easy to understand.

- 1. **Reason First** Upon receiving any new information, you **must** ﬁrst perform reasoning within <think> and </think> tags to determine whether you already have

sufﬁcient information to answer the user's question directly.

- 2. **Initiate a Search (if needed)** If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images

(indexed starting from 0).

- 3. **Image Analysis and Selection**

- Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.

- - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.
- - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

- 4. **Detailed Image Content Analysis**

**After selecting an image, proceed with this two-stage analysis:**

- ### 4.1 Initial ROI Identiﬁcation and Quality Assessment Within `<think>` tags:

- - Identify speciﬁc regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
- - For each region, summarize its key semantic content
- - **Assess readability**: Determine if content is clear or needs magniﬁcation

- ### 4.2 Content Extraction Decision Based on quality assessment:

- - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
- - **If content needs magniﬁcation**: Request speciﬁc regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause

*Magniﬁcation is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

- 5. **Magniﬁed Image Processing (If Applicable)**

**Only execute if magniﬁcation was requested in Step 4:**

- - Upon receiving magniﬁed image(s), analyze within `<think>` tags
- - Extract relevant information from the high-resolution regions
- - Output ﬁndings in `<information>Based on magniﬁed image analysis</information>`

- 6. **Final Answer Determination** Within `<think>` tags, evaluate:

- - Is current information sufﬁcient to answer the original question?
- - What critical information (if any) remains missing?

**Output exactly one:**

- - If insufﬁcient: `<search>precise follow-up query</search>`
- - If sufﬁcient: `<answer>direct ﬁnal answer</answer>`

- 7. **Iterative Reﬁnement** Repeat Steps 3-6 as needed until sufﬁcient information is gathered to provide a complete and accurate ﬁnal answer.

**Output Requirements** You must generate a strict JSON object containing the following four ﬁelds:

- 1. **`crop`** (boolean)

- - `true`: If you think **at least one** bounding box would beneﬁt from cropping and zooming for answering the question.
- - `false`: Otherwise.

- 2. **`bbox`** (array)

- - List the **full coordinates** of all bounding boxes you judge as **requiring cropping and zooming**, in the format `[[x1, y1, x2, y2], ...]`.
- - If `crop` is `false`, this ﬁeld must be an empty array `[]`.

- 3. **`think`** (string)

- Provide a concise reasoning chain that must cover:

- a) For **each** bounding box (referenced by its coordinates), describe the key content it contains.
- b) For boxes you mark as **needing crop**: Explain why zooming into this region could provide more critical information.
- c) For boxes you mark as **not needing crop**: Explain how their current content is already sufﬁcient and directly supports the answer.

- 4. **`answer`** (string)

- - If `crop` is `false`, output an accurate and complete ﬁnal answer based on the content of all bounding boxes.
- - If `crop` is `true`, this ﬁeld must be an empty string `""`.

**Key Notes**

- - The judgment criterion is **functional**: whether cropping could potentially aid in answering the question (e.g., to view blurry text, distinguish small objects, focus on key details), not merely to evaluate image quality.
- - Throughout the `think` ﬁeld, you **must use the exact coordinates** `[x1, y1, x2, y2]` to refer to bounding boxes. **It is prohibited to use** vague references like "the ﬁrst box", "the left box", or "the top box".

**CRITICAL CLARIFICATION ON MAGNIFICATION:**

- - Request magniﬁcation ONLY when enlarging the speciﬁc region of the selected image might help answer the query more clearly and accurately
- - Use `<bbox>` coordinates to specify EXACT regions that need magniﬁcation
- - After magniﬁcation, base your analysis SOLELY on the magniﬁed region

Always adhere strictly to this sequential workﬂow to ensure methodical reasoning, justiﬁed image selection, and precise information extraction.

Preprint. Under review Glint Lab

[Figure 45]

**User question: {question}**

Prompt for Reward Model

You are an expert evaluation system for a question answering chatbot.

You are given the following information:

- the query - a generated answer - a reference answer

Your task is to evaluate the correctness of the generated answer. ## Query. {query} ## Reference Answer. {reference_answer} ## Generated Answer. {generated_answer}

Your response should be formatted as following: <judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False. Please note that the generated answer may contain additional information beyond the reference answer.

Figure 7: Prompt for Reward Model.

Prompt for Training and Testing

Please strictly follow the steps below to answer the user's question:

- 1. **Reason First** Upon receiving any new information, you **must** ﬁrst perform reasoning within <think> and </think> tags to determine whether you already have

sufﬁcient information to answer the user's question directly.

- 2. **Initiate a Search (if needed)** If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images

(indexed starting from 0).

- 3. **Image Analysis and Selection**

- Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.

- - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.
- - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

- 4. **Detailed Image Content Analysis**

**After selecting an image, proceed with this two-stage analysis:**

- ### 4.1 Initial ROI Identiﬁcation and Quality Assessment Within `<think>` tags:

- - Identify speciﬁc regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
- - For each region, summarize its key semantic content
- - **Assess readability**: Determine if content is clear or needs magniﬁcation

- ### 4.2 Content Extraction Decision Based on quality assessment:

- - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
- - **If content needs magniﬁcation**: Request speciﬁc regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause

*Magniﬁcation is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

- 5. **Magniﬁed Image Processing (If Applicable)**

**Only execute if magniﬁcation was requested in Step 4:**

- - Upon receiving magniﬁed image(s), analyze within `<think>` tags
- - Extract relevant information from the high-resolution regions
- - Output ﬁndings in `<information>Based on magniﬁed image analysis</information>`

- 6. **Final Answer Determination** Within `<think>` tags, evaluate:

- - Is current information sufﬁcient to answer the original question?
- - What critical information (if any) remains missing?

**Output exactly one:**

- - If insufﬁcient: `<search>precise follow-up query</search>`
- - If sufﬁcient: `<answer>direct ﬁnal answer</answer>`

- 7. **Iterative Reﬁnement** Repeat Steps 3-6 as needed until sufﬁcient information is gathered to provide a complete and accurate ﬁnal answer.

**CRITICAL CLARIFICATION ON MAGNIFICATION:**

- - Request magniﬁcation ONLY when enlarging the speciﬁc region of the selected image might help answer the query more clearly and accurately
- - Use `<bbox>` coordinates to specify EXACT regions that need magniﬁcation
- - After magniﬁcation, base your analysis SOLELY on the magniﬁed region

Always adhere strictly to this sequential workﬂow to ensure methodical reasoning, justiﬁed image selection, and precise information extraction.

**User question: {question}**

###### Figure 8: Prompt for Training and Testing.

Prompt for Reward Model

16

You are an expert evaluation system for a question answering chatbot.

You are given the following information:

- the query - a generated answer - a reference answer

Your task is to evaluate the correctness of the generated answer. ## Query. {query} ## Reference Answer. {reference_answer} ## Generated Answer. {generated_answer}

Your response should be formatted as following: <judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False. Please note that the generated answer may contain additional information beyond the reference answer.

[Figure 47]

Prompt for Data Synthesis - Precise Selection Action

Prompt for Data Synthesis - Image Search Action

Prompt for Data Synthesis - Image Search Action

## Role You are an intelligent search planning assistant responsible for analyzing the user's question and deciding on the initial search content.

## Role You are an intelligent search planning assistant responsible for analyzing the user's question and deciding on the initial search content.

## Role You are an expert AI visual reasoning assistant. Your primary mission is to analyze a user's query against a set of images and identify the single most relevant image. Your response MUST be a single, valid JSON object. Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

## Instructions For the user's question, generate only the initial search query without detailed elaboration. The output must be a JSON object containing "think" and "search" ﬁelds.

## Instructions For the user's question, generate only the initial search query without detailed elaboration. The output must be a JSON object containing "think" and "search" ﬁelds.

- - "think": Brieﬂy explain in English why this initial search is needed, based directly on the original question.
- - "search": Provide only the initial search query string in English, in the form of a question, that directly reﬂects the user's original question.

## Constraints

- - The search query must be based strictly on the original question
- - Do not add speculative content or make assumptions beyond what is stated
- - Keep the search query concise and directly related to the question
- - Provide only the initial search query, no detailed or expanded queries

- - "think": Brieﬂy explain in English why this initial search is needed, based directly on the original question.
- - "search": Provide only the initial search query string in English, in the form of a question, that directly reﬂects the user's original question.

## JSON Output Structure The JSON object must contain exactly two keys: `think` and `rerank`.

1. `think` (string): A detailed, step-by-step reasoning process that documents your entire analysis. This string must include:

- * **Query Deconstruction:** A brief analysis of the user's query to identify the key criteria.
- * **Image-by-Image Analysis:** An evaluation of EACH provided image against the query's criteria.
- * **Final Justiﬁcation:** A conclusive explanation stating why the selected image is the optimal choice and explicitly why the other images are less suitable.

## Constraints

- The search query must be based strictly on the original question

- - Do not add speculative content or make assumptions beyond what is stated
- - Keep the search query concise and directly related to the question
- - Provide only the initial search query, no detailed or expanded queries

2. `rerank` (string): The unique identiﬁer of the single best image number that matches the query (e.g., "0", "1”).

## Output You must provide the output in the following JSON format: {"think": "Brief explanation in English”,

## Output You must provide the output in the following JSON format: {"think": "Brief explanation in English”,

## Concrete Example …………

Prompt for Data Synthesis - Precise Selection Action

Prompt for Data Synthesis - Image Search Action

"search": "Initial search query in English, in the form of a question”}

"search": "Initial search query in English, in the form of a question”}

## Role You are an intelligent search planning assistant responsible for analyzing the user's question and deciding on the initial search content.

## Role You are an expert AI visual reasoning assistant. Your primary mission is to analyze a user's query against a set of images and identify the single most relevant image. Your response MUST be a single, valid JSON object. Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

Figure 9: Prompt for Data Synthesis - Image Search Action.

Figure 10: Prompt for Data Synthesis - Precise Selection Action.

## Instructions For the user's question, generate only the initial search query without detailed elaboration. The output must be a JSON object containing "think" and "search" ﬁelds.

Prompt for Data Synthesis - Precise Selection Action

## Role You are an expert AI visual reasoning assistant. Your primary mission is to analyze a user's query against a set of images and identify the single most relevant image. Your response MUST be a single, valid JSON object. Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

- - "think": Brieﬂy explain in English why this initial search is needed, based directly on the original question.
- - "search": Provide only the initial search query string in English, in the form of a question, that directly reﬂects the user's original question.

## Constraints

- - The search query must be based strictly on the original question
- - Do not add speculative content or make assumptions beyond what is stated
- - Keep the search query concise and directly related to the question
- - Provide only the initial search query, no detailed or expanded queries

## JSON Output Structure The JSON object must contain exactly two keys: `think` and `rerank`.

- 1. `think` (string): A detailed, step-by-step reasoning process that documents your entire analysis. This string must include:

- * **Query Deconstruction:** A brief analysis of the user's query to identify the key criteria.
- * **Image-by-Image Analysis:** An evaluation of EACH provided image against the query's criteria.
- * **Final Justiﬁcation:** A conclusive explanation stating why the selected image is the optimal choice and explicitly why the other images are less suitable.

- 2. `rerank` (string): The unique identiﬁer of the single best image number that matches the query (e.g., "0", "1”).

## JSON Output Structure The JSON object must contain exactly two keys: `think` and `rerank`.

- 1. `think` (string): A detailed, step-by-step reasoning process that documents your entire analysis. This string must include:

- * **Query Deconstruction:** A brief analysis of the user's query to identify the key criteria.
- * **Image-by-Image Analysis:** An evaluation of EACH provided image against the query's criteria.
- * **Final Justiﬁcation:** A conclusive explanation stating why the selected image is the optimal choice and explicitly why the other images are less suitable.

- 2. `rerank` (string): The unique identiﬁer of the single best image number that matches the query (e.g., "0", "1”).

## Output You must provide the output in the following JSON format: {"think": "Brief explanation in English”,

## Concrete Example …………

"search": "Initial search query in English, in the form of a question”}

Prompt for Data Synthesis - RoI Region Selection

Prompt for Data Synthesis - RoI Region Rethink

You will be given a **query** and an **image with pre-drawn red rectangular bounding boxes**. Each red rectangle corresponds to a distinct semantic layout region (e.g., a paragraph, chart, table, or title) and is labeled with a unique **index number starting from 0** (0, 1, 2, ...) in the image (**displayed vertically centered on the left side of each box**).

You will be given:

- - A natural language question (referred to as the "query");
- - An image;
- - A list of pre-deﬁned bounding boxes in the image, each represented by its pixel coordinates **[x1, y1, x2, y2]** (with the origin at the top-left corner). Your task is as follows:

## Concrete Example …………

Your task is to:

- 1. **Understand the intent of the query**;
- 2. **Identify the red-bounded region(s) in the image that directly contain the information needed to answer the query**;
- 3. **Output the index number(s) of those region(s), and provide the answer to the query in the `answer` ﬁeld**.

**Follow these guidelines strictly**:

Prompt for Data Synthesis - RoI Region Selection

Prompt for Data Synthesis - RoI Region Selection

- 1. **Select the relevant bounding box**: From the provided list of bounding boxes, select only those containing content directly useful for answering the query. Strictly exclude any irrelevant boxes. **If multiple bounding boxes contain content helpful for answering the query, choose the one that most directly and centrally addresses the question.**
- 2. **Explain your selection**: For the ﬁnally selected bounding box, use its exact coordinate tuple **[x1, y1, x2, y2]** as its unique identiﬁer. Describe the key semantic content within that region (e.g., text, table, chart, formula, etc.) and explain precisely how this content helps answer the query. **Do not refer to boxes using ordinal terms such as "ﬁrst," "second," or "the box on the left"—always identify them by their coordinates.**
- 3. **Output a structured result**: Return your response strictly in JSON format

- - **The answer must be directly derived from the content of the selected region(s)**:
- - Do not select a region unless the answer can be **unambiguously inferred** from its content.
- - **Do NOT select regions that only contain the question text or repeat the query verbatim** — these do not provide evidence for the answer.
- - Only select regions that contain **supporting data, explanation, or visual information** (e.g., a paragraph stating a fact, a chart

showing a value, a table listing numbers).

- - **Rely solely on visible content**: Do not use external knowledge or make assumptions beyond what is shown in the selected region(s).

**Output Format Requirements**: Return a strictly valid JSON object with exactly three ﬁelds:

- - **`think`** (string): A concise explanation of your reasoning — describe the query's intent and **why the selected indexed region(s) contain the necessary information to answer it**.
- - **`indices`** (list of integers): A list of index numbers (e.g., `[0]`, `[1, 3]`) corresponding to the red-bounded regions **from which the answer is directly derived**. The indices must match the **0-based numbers visible in the image**.
- - **`answer`** (string): The **answer to the query**, extracted accurately and concisely from the content of the selected region(s).
- - Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

You will be given a **query** and an **image with pre-drawn red rectangular bounding boxes**. Each red rectangle corresponds to a distinct semantic layout region (e.g., a paragraph, chart, table, or title) and is labeled with a unique **index number starting from 0** (0, 1, 2, ...) in the image (**displayed vertically centered on the left side of each box**).

You will be given a **query** and an **image with pre-drawn red rectangular bounding boxes**. Each red rectangle corresponds to a distinct semantic layout region (e.g., a paragraph, chart, table, or title) and is labeled with a unique **index number starting from 0** (0, 1, 2, ...) in the image (**displayed vertically centered on the left side of each box**).

Your task is to:

Your task is to:

- 1. **Understand the intent of the query**;
- 2. **Identify the red-bounded region(s) in the image that directly contain the information needed to answer the query**;
- 3. **Output the index number(s) of those region(s), and provide the answer to the query in the `answer` ﬁeld**.

- 1. **Understand the intent of the query**;
- 2. **Identify the red-bounded region(s) in the image that directly contain the information needed to answer the query**;
- 3. **Output the index number(s) of those region(s), and provide the answer to the query in the `answer` ﬁeld**.

**Follow these guidelines strictly**:

**Follow these guidelines strictly**:

- - **The answer must be directly derived from the content of the selected region(s)**:
- - Do not select a region unless the answer can be **unambiguously inferred** from its content.
- - **Do NOT select regions that only contain the question text or repeat the query verbatim** — these do not provide evidence for the answer.
- - Only select regions that contain **supporting data, explanation, or visual information** (e.g., a paragraph stating a fact, a chart

showing a value, a table listing numbers).

- - **Rely solely on visible content**: Do not use external knowledge or make assumptions beyond what is shown in the selected region(s).

**Output Format Requirements**: Return a strictly valid JSON object with exactly three ﬁelds:

- - **`think`** (string): A concise explanation of your reasoning — describe the query's intent and **why the selected indexed region(s) contain the necessary information to answer it**.
- - **`indices`** (list of integers): A list of index numbers (e.g., `[0]`, `[1, 3]`) corresponding to the red-bounded regions **from which the answer is directly derived**. The indices must match the **0-based numbers visible in the image**.
- - **`answer`** (string): The **answer to the query**, extracted accurately and concisely from the content of the selected region(s).
- - Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

- **The answer must be directly derived from the content of the selected region(s)**:

- - Do not select a region unless the answer can be **unambiguously inferred** from its content.
- - **Do NOT select regions that only contain the question text or repeat the query verbatim** — these do not provide evidence for the answer.
- - Only select regions that contain **supporting data, explanation, or visual information** (e.g., a paragraph stating a fact, a chart showing a value, a table listing numbers).

Prompt for Data Synthesis - Visual Perception Action Complete

Prompt for Data Synthesis - Visual Perception Action Decision

You are a professional visual reasoning AI. Your task is to process a user-input original image, a cropped image, and a question, then generate a structured JSON output. The cropped image is a screenshot of a speciﬁc bounding box region from the original image, showing only that region's content.

You will receive the following inputs:

- **Rely solely on visible content**: Do not use external knowledge or make assumptions beyond what is shown in the selected region(s).

- - A natural language **query**
- - An **image**
- - A set of **bounding boxes**, each deﬁned by its pixel coordinates `[x1, y1, x2, y2]`

## Input Format:

**Output Format Requirements**: Return a strictly valid JSON object with exactly three ﬁelds:

**Core Task** Based on the given query, evaluate each bounding box individually: **If you think that cropping and zooming into the corresponding image region may help to answer the query more clearly and accurately**, then judge that this box requires processing.

- - Original Image: The complete original image
- - Cropped Image: A cropped region from the original image
- - Question: A text question that needs to be answered based on the image contents

## Output Requirements: The output must be a JSON object containing the following two keys:

- - "think": A reasoning chain that step-by-step explains how to derive the answer to the question from both the original image and the cropped image content. The reasoning should be logical, fact-based, and directly connect image details from both images to the question.
- - "answer": A concise and clear direct answer based on the reasoning.

## Output Format: ``` { "think": ..., "answer": ... } ``` ## Important Principles:

- - Always base descriptions and reasoning on factual image content; do not fabricate or assume content not present in the images.
- - Ensure correct JSON format with exact key names: think, answer.
- - Use concise and accurate language, making the reasoning process easy to understand.

- **`think`** (string): A concise explanation of your reasoning — describe the query's intent and **why the selected indexed region(s) contain the necessary information to answer it**.

**Output Requirements** You must generate a strict JSON object containing the following four ﬁelds:

- - **`indices`** (list of integers): A list of index numbers (e.g., `[0]`, `[1, 3]`) corresponding to the red-bounded regions **from which the answer is directly derived**. The indices must match the **0-based numbers visible in the image**.
- - **`answer`** (string): The **answer to the query**, extracted accurately and concisely from the content of the selected region(s).
- - Please wrap your ﬁnal JSON response in a single JSON code block (json... ). Do not output any text before or after the code block.

- 1. **`crop`** (boolean)

- - `true`: If you think **at least one** bounding box would beneﬁt from cropping and zooming for answering the question.
- - `false`: Otherwise.

- 2. **`bbox`** (array)

- - List the **full coordinates** of all bounding boxes you judge as **requiring cropping and zooming**, in the format `[[x1, y1, x2, y2], ...]`.
- - If `crop` is `false`, this ﬁeld must be an empty array `[]`.

- 3. **`think`** (string)

- Provide a concise reasoning chain that must cover:

- a) For **each** bounding box (referenced by its coordinates), describe the key content it contains.
- b) For boxes you mark as **needing crop**: Explain why zooming into this region could provide more critical information.
- c) For boxes you mark as **not needing crop**: Explain how their current content is already sufﬁcient and directly supports the answer.

- 4. **`answer`** (string)

- - If `crop` is `false`, output an accurate and complete ﬁnal answer based on the content of all bounding boxes.
- - If `crop` is `true`, this ﬁeld must be an empty string `""`.

**Key Notes**

- - The judgment criterion is **functional**: whether cropping could potentially aid in answering the question (e.g., to view blurry text, distinguish small objects, focus on key details), not merely to evaluate image quality.
- - Throughout the `think` ﬁeld, you **must use the exact coordinates** `[x1, y1, x2, y2]` to refer to bounding boxes. **It is prohibited to use** vague references like "the ﬁrst box", "the left box", or "the top box".

Prompt for Data Synthesis - RoI Region Conﬁrm

Prompt for Data Synthesis - RoI Region Conﬁrm

You will be given:

You will be given:

- - A natural language question (referred to as the "query");
- - An image;
- - A list of pre-deﬁned bounding boxes in the image, each represented by its pixel coordinates **[x1, y1, x2, y2]** (with the origin at the top-left corner).

- - A natural language question (referred to as the "query");
- - An image;
- - A list of pre-deﬁned bounding boxes in the image, each represented by its pixel coordinates **[x1, y1, x2, y2]** (with the origin at the top-left corner). Your task is as follows:

###### Figure 11: Prompt for Data Synthesis - Visual Perception Action.

Your task is as follows:

- 1. **Select the relevant bounding box**: From the provided list of bounding boxes, select only those containing content directly useful for answering the query. Strictly exclude any irrelevant boxes. **If multiple bounding boxes contain content helpful for answering the query, choose the one that most directly and centrally addresses the question.**
- 2. **Explain your selection**: For the ﬁnally selected bounding box, use its exact coordinate tuple **[x1, y1, x2, y2]** as its unique identiﬁer. Describe the key semantic content within that region (e.g., text, table, chart, formula, etc.) and explain precisely how this content helps answer the query. **Do not refer to boxes using ordinal terms such as "ﬁrst," "second," or "the box on the left"—always identify them by their coordinates.**
- 3. **Output a structured result**: Return your response strictly in JSON format

- 1. **Select the relevant bounding box**: From the provided list of bounding boxes, select only those containing content directly useful for answering the query. Strictly exclude any irrelevant boxes. **If multiple bounding boxes contain content helpful for answering the query, choose the one that most directly and centrally addresses the question.**
- 2. **Explain your selection**: For the ﬁnally selected bounding box, use its exact coordinate tuple **[x1, y1, x2, y2]** as its unique identiﬁer. Describe the key semantic content within that region (e.g., text, table, chart, formula, etc.) and explain precisely how this content helps answer the query. **Do not refer to boxes using ordinal terms such as "ﬁrst," "second," or "the box on the left"—always identify them by their coordinates.**
- 3. **Output a structured result**: Return your response strictly in JSON format

17

Prompt for Data Synthesis - Visual Perception Action Decision

Prompt for Data Synthesis - Visual Perception Action Decision

You will receive the following inputs:

You will receive the following inputs:

- - A natural language **query**
- - An **image**
- - A set of **bounding boxes**, each deﬁned by its pixel coordinates `[x1, y1, x2, y2]`

- - A natural language **query**
- - An **image**
- - A set of **bounding boxes**, each deﬁned by its pixel coordinates `[x1, y1, x2, y2]`

**Core Task** Based on the given query, evaluate each bounding box individually: **If you think that cropping and zooming into the corresponding image region may help to answer the query more clearly and accurately**, then judge that this box requires processing.

**Core Task** Based on the given query, evaluate each bounding box individually: **If you think that cropping and zooming into the corresponding image region may help to answer the query more clearly and accurately**, then judge that this box requires processing.

**Output Requirements** You must generate a strict JSON object containing the following four ﬁelds:

**Output Requirements** You must generate a strict JSON object containing the following four ﬁelds:

1. **`crop`** (boolean)

- 1. **`crop`** (boolean)

- - `true`: If you think **at least one** bounding box would beneﬁt from cropping and zooming for answering the question.
- - `false`: Otherwise.

- 2. **`bbox`** (array)

- - List the **full coordinates** of all bounding boxes you judge as **requiring cropping and zooming**, in the format `[[x1, y1, x2, y2], ...]`.
- - If `crop` is `false`, this ﬁeld must be an empty array `[]`.

- 3. **`think`** (string)

- Provide a concise reasoning chain that must cover:

- a) For **each** bounding box (referenced by its coordinates), describe the key content it contains.
- b) For boxes you mark as **needing crop**: Explain why zooming into this region could provide more critical information.
- c) For boxes you mark as **not needing crop**: Explain how their current content is already sufﬁcient and directly supports the answer.

- 4. **`answer`** (string)

- - `true`: If you think **at least one** bounding box would beneﬁt from cropping and zooming for answering the question.
- - `false`: Otherwise.

2. **`bbox`** (array)

- - List the **full coordinates** of all bounding boxes you judge as **requiring cropping and zooming**, in the format `[[x1, y1, x2, y2], ...]`.
- - If `crop` is `false`, this ﬁeld must be an empty array `[]`.

3. **`think`** (string)

- Provide a concise reasoning chain that must cover: a) For **each** bounding box (referenced by its coordinates), describe the key content it contains. b) For boxes you mark as **needing crop**: Explain why zooming into this region could provide more critical information. c) For boxes you mark as **not needing crop**: Explain how their current content is already sufﬁcient and directly supports the answer.

4. **`answer`** (string)

- - If `crop` is `false`, output an accurate and complete ﬁnal answer based on the content of all bounding boxes.
- - If `crop` is `true`, this ﬁeld must be an empty string `""`.

- - If `crop` is `false`, output an accurate and complete ﬁnal answer based on the content of all bounding boxes.
- - If `crop` is `true`, this ﬁeld must be an empty string `""`.

**Key Notes**

- - The judgment criterion is **functional**: whether cropping could potentially aid in answering the question (e.g., to view blurry text, distinguish small objects, focus on key details), not merely to evaluate image quality.
- - Throughout the `think` ﬁeld, you **must use the exact coordinates** `[x1, y1, x2, y2]` to refer to bounding boxes. **It is prohibited to use** vague references like "the ﬁrst box", "the left box", or "the top box".

**Key Notes**

- The judgment criterion is **functional**: whether cropping could potentially aid in answering the question (e.g., to view blurry text, distinguish small objects, focus on key details), not merely to evaluate image quality.

- Throughout the `think` ﬁeld, you **must use the exact coordinates** `[x1, y1, x2, y2]` to refer to bounding boxes. **It is prohibited to use** vague references like "the ﬁrst box", "the left box", or "the top box".

Prompt for Training and Test

Please strictly follow the steps below to answer the user's question:

- 1. **Reason First** Upon receiving any new information, you **must** ﬁrst perform reasoning within <think> and </think> tags to determine whether you already have

sufﬁcient information to answer the user's question directly.

- 2. **Initiate a Search (if needed)** If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images

(indexed starting from 0).

- 3. **Image Analysis and Selection**

- Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.

- - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.
- - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

- 4. **Detailed Image Content Analysis**

**After selecting an image, proceed with this two-stage analysis:**

- ### 4.1 Initial ROI Identiﬁcation and Quality Assessment Within `<think>` tags:

- - Identify speciﬁc regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
- - For each region, summarize its key semantic content
- - **Assess readability**: Determine if content is clear or needs magniﬁcation

- ### 4.2 Content Extraction Decision Based on quality assessment:

- - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
- - **If content needs magniﬁcation**: Request speciﬁc regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause

*Magniﬁcation is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

- 5. **Magniﬁed Image Processing (If Applicable)**

**Only execute if magniﬁcation was requested in Step 4:**

- - Upon receiving magniﬁed image(s), analyze within `<think>` tags
- - Extract relevant information from the high-resolution regions
- - Output ﬁndings in `<information>Based on magniﬁed image analysis</information>`

- 6. **Final Answer Determination** Within `<think>` tags, evaluate:

- - Is current information sufﬁcient to answer the original question?
- - What critical information (if any) remains missing?

**Output exactly one:**

- - If insufﬁcient: `<search>precise follow-up query</search>`
- - If sufﬁcient: `<answer>direct ﬁnal answer</answer>`

- 7. **Iterative Reﬁnement** Repeat Steps 3-6 as needed until sufﬁcient information is gathered to provide a complete and accurate ﬁnal answer.

Prompt for Training and Test

Prompt for Training and Test

Please strictly follow the steps below to answer the user's question:

Please strictly follow the steps below to answer the user's question:

1. **Reason First**

- 1. **Reason First** Upon receiving any new information, you **must** ﬁrst perform reasoning within <think> and </think> tags to determine whether you already have

sufﬁcient information to answer the user's question directly.

- 2. **Initiate a Search (if needed)** If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images

(indexed starting from 0).

- 3. **Image Analysis and Selection**

- Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.

- - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.
- - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

- 4. **Detailed Image Content Analysis**

**After selecting an image, proceed with this two-stage analysis:**

- ### 4.1 Initial ROI Identiﬁcation and Quality Assessment Within `<think>` tags:

- - Identify speciﬁc regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query
- - For each region, summarize its key semantic content
- - **Assess readability**: Determine if content is clear or needs magniﬁcation

- ### 4.2 Content Extraction Decision Based on quality assessment:

- - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
- - **If content needs magniﬁcation**: Request speciﬁc regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause

*Magniﬁcation is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

- 5. **Magniﬁed Image Processing (If Applicable)**

**Only execute if magniﬁcation was requested in Step 4:**

- - Upon receiving magniﬁed image(s), analyze within `<think>` tags
- - Extract relevant information from the high-resolution regions
- - Output ﬁndings in `<information>Based on magniﬁed image analysis</information>`

- 6. **Final Answer Determination** Within `<think>` tags, evaluate:

- - Is current information sufﬁcient to answer the original question?
- - What critical information (if any) remains missing?

**Output exactly one:**

- - If insufﬁcient: `<search>precise follow-up query</search>`
- - If sufﬁcient: `<answer>direct ﬁnal answer</answer>`

- 7. **Iterative Reﬁnement** Repeat Steps 3-6 as needed until sufﬁcient information is gathered to provide a complete and accurate ﬁnal answer.

Upon receiving any new information, you **must** ﬁrst perform reasoning within <think> and </think> tags to determine whether you already have sufﬁcient information to answer the user's question directly.

2. **Initiate a Search (if needed)**

If reasoning indicates that necessary information is missing, initiate an image search using <search>query</search>. The system will return a set of images (indexed starting from 0).

3. **Image Analysis and Selection**

- - Within the <think> tags, **analyze each image one by one**, focusing on whether it contains **text, charts, tables, labels, or other key visual information** relevant to the query.
- - Evaluate the completeness and relevance of information in each image, and **select the single image most helpful for answering the query**.
- - Explicitly specify the index of the chosen image in a <select> tag (e.g., <select>2</select>).

4. **Detailed Image Content Analysis**

**After selecting an image, proceed with this two-stage analysis:** ### 4.1 Initial ROI Identiﬁcation and Quality Assessment Within `<think>` tags:

- Identify speciﬁc regions of interest (ROI) with coordinates `[x1, y1, x2, y2]` relevant to the query

- - For each region, summarize its key semantic content
- - **Assess readability**: Determine if content is clear or needs magniﬁcation ### 4.2 Content Extraction Decision Based on quality assessment:
- - **If content is readable**: Extract semantic meaning and output `<information>Answer based on this image</information>`
- - **If content needs magniﬁcation**: Request speciﬁc regions using `<bbox>[[x1, y1, x2, y2], ...]</bbox>` and pause

*Magniﬁcation is needed only when enlarging the ROI of the selected image might help answer the query more clearly and accurately.*

5. **Magniﬁed Image Processing (If Applicable)**

**CRITICAL CLARIFICATION ON MAGNIFICATION:**

**Only execute if magniﬁcation was requested in Step 4:**

- - Request magniﬁcation ONLY when enlarging the speciﬁc region of the selected image might help answer the query more clearly and accurately
- - Use `<bbox>` coordinates to specify EXACT regions that need magniﬁcation
- - After magniﬁcation, base your analysis SOLELY on the magniﬁed region

- Upon receiving magniﬁed image(s), analyze within `<think>` tags

- - Extract relevant information from the high-resolution regions
- - Output ﬁndings in `<information>Based on magniﬁed image analysis</information>`

Always adhere strictly to this sequential workﬂow to ensure methodical reasoning, justiﬁed image selection, and precise information extraction.

6. **Final Answer Determination** Within `<think>` tags, evaluate:

**User question: {question}**

- - Is current information sufﬁcient to answer the original question?
- - What critical information (if any) remains missing?

**Output exactly one:**

Prompt for Reward Model

- If insufﬁcient: `<search>precise follow-up query</search>`

- If sufﬁcient: `<answer>direct ﬁnal answer</answer>`

You are an expert evaluation system for a question answering chatbot.

7. **Iterative Reﬁnement** Repeat Steps 3-6 as needed until sufﬁcient information is gathered to provide a complete and accurate ﬁnal answer.

You are given the following information:

- the query - a generated answer - a reference answer

**CRITICAL CLARIFICATION ON MAGNIFICATION:**

Your task is to evaluate the correctness of the generated answer. ## Query. {query} ## Reference Answer. {reference_answer} ## Generated Answer. {generated_answer}

**CRITICAL CLARIFICATION ON MAGNIFICATION:**

- Request magniﬁcation ONLY when enlarging the speciﬁc region of the selected image might help answer the query more clearly and accurately

- - Request magniﬁcation ONLY when enlarging the speciﬁc region of the selected image might help answer the query more clearly and accurately
- - Use `<bbox>` coordinates to specify EXACT regions that need magniﬁcation
- - After magniﬁcation, base your analysis SOLELY on the magniﬁed region

- - Use `<bbox>` coordinates to specify EXACT regions that need magniﬁcation
- - After magniﬁcation, base your analysis SOLELY on the magniﬁed region

Your response should be formatted as following: <judge>True or False</judge>

Always adhere strictly to this sequential workﬂow to ensure methodical reasoning, justiﬁed image selection, and precise information extraction.

Always adhere strictly to this sequential workﬂow to ensure methodical reasoning, justiﬁed image selection, and precise information extraction.

**User question: {question}**

**User question: {question}**

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False. Please note that the generated answer may contain additional information beyond the reference answer.

Prompt for Reward Model

Prompt for Reward Model

You are an expert evaluation system for a question answering chatbot.

You are an expert evaluation system for a question answering chatbot.

You are given the following information:

You are given the following information:

- the query - a generated answer - a reference answer

- the query - a generated answer - a reference answer

Your task is to evaluate the correctness of the generated answer. ## Query. {query} ## Reference Answer. {reference_answer} ## Generated Answer. {generated_answer}

Your task is to evaluate the correctness of the generated answer. ## Query. {query} ## Reference Answer. {reference_answer} ## Generated Answer. {generated_answer}

Your response should be formatted as following: <judge>True or False</judge>

Your response should be formatted as following: <judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False. Please note that the generated answer may contain additional information beyond the reference answer.

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False. Please note that the generated answer may contain additional information beyond the reference answer.

