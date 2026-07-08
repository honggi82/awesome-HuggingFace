# arXiv:2509.06818v1[cs.CV]8Sep2025

[Figure 1]

[Figure 2]

#### UMO: Scaling Multi-Identity Consistency for Image Customization via Matching Reward

###### Yufeng Cheng Wenxu Wu Shaojin Wu Mengqi Huang ∗ Fei Ding † Qian He

UXO Team, Intelligent Creation Lab, ByteDance

###### Abstract

Recent advancements in image customization exhibit a wide range of application prospects due to stronger customization capabilities. However, since we humans are more sensitive to faces, a significant challenge remains in preserving consistent identity while avoiding identity confusion with multi-reference images, limiting the identity scalability of customization models. To address this, we present UMO, a Unified Multi-identity Optimization framework, designed to maintain high-fidelity identity preservation and alleviate identity confusion with scalability. With multi-tomulti matching paradigm, UMO reformulates multi-identity generation as a global assignment optimization problem and unleashes multi-identity consistency for existing image customization methods generally through reinforcement learning on diffusion models. To facilitate the training of UMO, we develop a scalable customization dataset with multi-reference images, consisting of both synthesised and real parts. Additionally, we propose a new metric to measure identity confusion. Extensive experiments demonstrate that UMO not only improves identity consistency significantly, but also reduces identity confusion on several image customization methods, setting a new state-of-the-art among open-source methods along the dimension of identity preserving. Code and model: https://github.com/bytedance/UMO

Date: September 9, 2025 Project Page: https://bytedance.github.io/UMO Correspondence: Yufeng Cheng at chengyufeng.cb1@bytedance.com

###### 1 Introduction

Image customization, which aims to create images that simultaneously adhere to the semantic content of textual prompts and the visual appearance of reference images, has garnered significant research attention in recent years. Among various subjects, the customization of human identities (i.e., ID) has attracted particular interest due to its broad range of real-world applications, such as personalized film production and virtual avatar creation. Different from other subjects, humans are exceptionally sensitive to identity customization, i.e., even subtle discrepancies in appearance can lead to a noticeable loss of identity fidelity, thereby making human ID customization significantly more challenging. This challenge is further amplified when multiple identities need to be customized simultaneously, as it requires the model not only to preserve the unique characteristics of each individual ID, but also to maintain clear distinctions among them within the generated images.

* Corresponding author † Project lead.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Single Identity

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Single Identity + Subject

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Two Identities

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Two Identities + Subject

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

More Identities

Figure 1 Showcase of our UMO model in different scenarios. The detailed prompts are listed in Table 9.

Existing multi-identity customization methods primarily focus on constructing improved multi-identity paired data to enhance the consistency of multiple identities. For example, DreamO [18] and OmniGen [33] build large-scale training datasets for identity customization, including multi-identity paired data derived from either video sources or in-context image generation. Meanwhile, some recent approaches aim to mitigate confusion between multiple identities by employing identity masks to explicitly constrain the location or

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Single Identity

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

##### UMO

Multi-Identity Portrait

Multi-Identity Interaction

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Multi-Identity Stylization

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Figure 2 Our UMO unleashes multi-identity consistency and alleviates identity confusion. Existing image customization methods suffer low facial fidelity and severe identity confusion, while UMO can tackle these problems with results in blue boxes.

position of each identity. For instance, MS-Diffusion [29] introduces a layout-guided mechanism to explicitly control the generation location of each identity. RealCustom++ [13, 17] further proposes to explicitly separate the influence masks of each identity in order to disentangle their respective generations. In summary, existing methods mainly adopt a one-to-one mapping paradigm, in which focuses on learning a direct correspondence between each identity in the reference image and the corresponding generated one.

In this study, we argue that the existing one-to-one mapping paradigm fails to comprehensively address both intra-ID variability and inter-ID distinction, leading to increased identity confusion and reduced identity similarity as the number of identities grows. On the one hand, intra-ID variability refers to the inherent variability within a single identity, where the same individual may present different attributes (e.g., poses, expressions, etc.) between the reference image and the generated output. On the other hand, inter-ID distinction underscores the importance of not only accurately capturing the distinctive characteristics of the target identity during generation, but also explicitly suppressing the features associated with other identities, thereby ensuring clear separation and minimizing identity confusion in multi-identity scenarios. As the number of identities increases, the risk arises that the distinctions between different identities may become less salient, potentially approaching the degree of variability observed within a single identity. Therefore, by focusing exclusively on the one-to-one mapping between each identity and its corresponding reference, existing methods overlook the increasing overlap between intra-ID variability and inter-ID distinction as the number of identities grows. This limitation fundamentally restricts their scalability, as they are unable to effectively preserve clear identity boundaries in large-scale multi-identity scenarios.

To address the challenge, we propose a novel multi-to-multi matching paradigm, whose key idea is to reformulate multi-identity generation as a global assignment optimization problem, aiming to maximize the

overall matching quality between multiple identities and multiple references. Therefore, each generated identity can be paired with the most suitable reference, which maximizes inter-ID distinction while minimizing the impact of intra-ID variability, thereby enabling accurate and scalable multi-identity generation. Technically, to ensure that our method remains concise and readily applicable to various customized models to improve identity consistency, we propose a Unified Multi-identity Optimization (UMO) framework, which operationalizes the multi-to-multi matching paradigm through a novel Reference Reward Feedback Learning (ReReFL). Specifically, UMO begins by defining a lightweight and reliable single-identity reference reward based on the cosine distance between identity embeddings. This is then scaled to a multi-identity context by formulating a bipartite graph between multiple references and generated identities, and optimizing the global matching score to achieve the optimal assignment. Moreover, to support the effective training of UMO, we develop a scalable customization dataset with multiple reference images for each identity, along with a new metric (ID-Conf) designed to precisely evaluate the extent of multi-identity confusion.

Our main contributions can be summarized as follows:

Concept: We highlight that existing customization methods fails to address intra-ID variability and inter-ID distinction, as their one-to-one mapping paradigm leads to decreased identity consistency with the scale of the identities. For the first time, we propose a novel multi-to-multi matching paradigm that maximizes overall matching quality between multiple identities and references.

Technique: We propose UMO, a novel Reference Reward Feedback Learning framework featuring scalable multi-identity reference reward optimization, which can be easily integrated into various customization models.

Metric: We propose ID-Conf to accurately evaluate the extent of multi-identity confusion. For a reference identity, ID-Conf is defined as the relative margin between the two most similar generated candidates, given the observation that confusion occurs with several indistinct generated faces.

Performance: We conduct extensive experiments on XVerseBench [4] and OmniContext [31], including multiidentity scenarios. Our UMO significantly boosts identity similarity and mitigates confusion simultaneously on various customized models, leading to the highest ID-Sim and ID-Conf scores. This demonstrates its strong and general promoting effect on identity consistency, showcasing its capability to deliver state-of-the-art (SOTA) results with high fidelity identities as shown in Figure 1.

###### 2 Related Works

###### 2.1 Multi-subject Driven Generation

Text-to-image models [1, 7, 16, 22, 24, 26] experienced explosive growth in recent years. Despite the increasing text-instruction following capability of those models, they still struggle to fulfill a common requirement: generating images for given subjects with reference images.

Early works [8, 25] extended subject-driven generation from text-to-image (T2I) models through parameterefficient fine-tuning on each given subject. Subsequent works [30, 38] achieved general single-subject injection by modifying and training components such as the cross-attention module.

Building on the general single-subject feature injection, numerous extensions have been proposed. A series of works, such as MIP-adapter [14], MSDiffusion [29], and UNO [32] extend the method to general multi-subject driven generation by leveraging the flexible context window lengths of attention mechanisms. Another improvement aims at ID preservation. InstanceID [28], Pulid [9] achieved higher ID preservation by adapting widely used face recognition models as encoders and training supervisions. Other works directly train an image generation model from scratch to natively support both textual input and reference subject as input [5, 6, 33]. Some closed-source commercial model [37] may support the greater potential in fidelity of those methods, but haven’t matched by current open-source and academic works.

###### 2.2 Reinforcement Learning for Diffusion Model

For LLM, reinforcement learning has been wildely used to align with human preferences like truthfulness, helpfulness, etc. [20]. Since the distribution formulation difference between the autoregressive model and

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Forward w/ gradient

Reward score & Backward

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Multi-Identity Matching Reward

X 1

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Forward w/o gradient

[Figure 116]

[Figure 117]

### UMO

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

X n-1

make a photo of all these three people

[Figure 134]

ReReFL Reward score

𝐼 𝐼 𝐼 𝐼 Reference Image Prompt

Target Image

- Figure 3 Illustration of the training framework of UMO. UMO’s training process follows ReReFL in Algorithm 1 with Multi-Identity Matching Reward.

diffusion model, the success of reinforcement learning in language models can’t directly transfer to image generation models. Prior works [2, 27, 35, 36] have explored several reinforcement learning algorithms that can be used to improve diffusion models and align with external rewards. But most of them are focusing on text alignment and aesthetics, few works for identity similarity improvement with reinforcement learning.

###### 3 Methodology

###### 3.1 Preliminary

Diffusion models [10] have shown great capability in text-to-image generation, which sampling desired data distribution through iterative denoising. To align the models with human preferences, RLHF has been widely adapted, e.g., ReFL [35] with objective

i∼Y(ϕ(r(yi,gθ(yi)))) (1)

Lreward = Ey

where Y = {y1,y2,...,yn} is prompt set, ϕ is reward-to-loss map function, r is reward model and gθ represents the generated image of diffusion model with parameters θ corresponding to prompt yi.

###### 3.2 Data Curation Pipeline

In this paper, we aim to scale the number of identities in multi-identity preservation. But most public datasets, like X2I-subject [33] and Openstory [39], have few samples with identitiesx larger than two, limiting our methods. To address that, we build a data pipeline inspired by MovieGen [23], which uses the frame with multiple identities as query and recalls each identity from frames with other clips in the same long video.

Besides, we explored a synthetic data pipeline suggested by previous works such as UNO [32] to construct a multi-identity preservation dataset. However, the identity similarity of synthetic data is relatively low. We applied strict face similarity filtering and retained only data in highly imaginative and partially stylized scenes, using them as a complement.

Algorithm 1 Reference Reward Feedback Learning (ReReFL) for diffusion models Require: Image customization diffusion models v with pretrained parameters θ0, pretrain loss function ϕ,

pretrain loss weight λ; reward function R; the number of noise scheduler time steps T, time step range for ReReFL finetuning [Ts,Te]; dataset D = {(yi,I0

) | i = 1,2,...} where yi is prompt, I0

,Ir1

,Ir2

,...,IrM

i

i

i

i

i

is target image and Ir1

are reference images.

,Ir2

,...,IrM

i

i

i

- 1: for (yi,I0

i

,Ir1

i

,Ir2

i

,...,IrM

i

) ∈ D do

- 2: Ldiff ← ϕθ

i

(yi,Ir1

i

,Ir2

i

,...,IrM

i

;I0

i

) // Calculate pretrained loss with reference images

- 3: t ← U(Ts,Te) // Pick a random time step
- 4: xT ∼ N(0,I)
- 5: for τ = T,...,t + 1 do
- 6: xτ−1 ← no_grad(vθ

i

(yi,Ir1

i

,Ir2

i

,...,IrM

i

,xτ))

- 7: end for
- 8: xt−1 ← vθ

i

(yi,Ir1

i

,Ir2

i

,...,IrM

i

,xt)

- 9: Iˆ0

i ← xt−1 // Predict original image by noise scheduler

- 10: LReReFL ← −R(Iˆ0

i

;yi,Ir1

i

,Ir2

i

,...,IrM

i

) // Calculate ReReFL loss with negative reward

- 11: L ← λLdiff + LReReFL
- 12: θi+1 ← θi // Update parameters with L
- 13: end for

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

(a) SIR scores of UNO [32]. (b) SIR scores of OmniGen2 [31].

- Figure 4 Single identity reward (SIR) scores of UNO [32] and OmniGen2 [31] with different generation seeds along denoising steps. The scores become stable after step 5 and 10 respectively. And the results with highest and lowest reward scores indicating its discriminatory ability.

The combination of diverse data sources and the carefully designed extraction pipeline ensures the resulting multi-person identity preservation dataset features a larger number of distinct individuals and great variations beyond the identity features.

###### 3.3 UMO Training Framework

###### 3.3.1 Reinforcement Learning on Image Customization

A trivial solution to scale multi-identity preservation is finetuning existing image customization methods with data constructed above which only achieves minor improvement as shown in Table 4 partly due to its relatively small proportion in diffusion models’ objective.

To steer models to align to human sensitivity towards faces, we extend ReFL to customization. Specifically, we propose Reference Reward Feedback Learning (ReReFL), as illustrated in the red dashed box of Figure 3,

which directly backpropagate reward signals with reference to the inference, as Algorithm 1 shown.

- 3.3.2 Reward with Single Identity Reference

Effective reward is crucial to the improvement brought by RLHF. In the easiest case with only one reference identity, we introduce Single Identity Reward (SIR), the cosine distance between identity embeddings to ensure a high degree of identity fidelity

RSIR = cos(ψ(Iˆ0),ψ(Ir1)) (2) where ψ represents a tiny network to recognition face and get face embedding. As shown in Figure 4, single identity reward varies drastically during former steps while getting relatively stable during latter steps. The generation result with the highest reward score preserves identity better than the one with the lowest reward score, indicating that it could serve as a reliable reward function in ReReFL.

- 3.3.3 Reward with Multi-Identity Reference

When scaling SIR to a more complicated case with multi-reference, a new challenge arises, that is improving identity fidelity while enlarging inter-ID distinction to alleviate confusion at the same time. As shown in Figure 2, when identity confusion occurs, the customization models usually ignore some references, or generate a person with face from one identity while cloth from another identity.

We suggest that the keypoint lies in finding the corresponding face for each reference identity. Inspired by DETR [3] and Multi-Object Tracking [41], we formulate the problem as the assignment problem under multi-to-multi matching paradigm. Specifically, let us denote by M the number of reference identities, N the number of detected faces in Iˆ0, i.e., ψ(Iˆ0) ∈ RN×d where d represents the dimension of face embedding. Each face is a vertex in a bipartite graph with one part Fˆ containing all N faces detected in Iˆ0 and the other part F containing all M faces from M reference identities, as illustrated in the black box of Figure 3. The edges are weighted by SIR of two vertices, i.e., eF

j,Fˆk = cos(ψ(Iˆ0)j,ψ(Irk)). To find a maximum weight matching, we search assignment σˆ in all the potential ones Sn of M reference faces to N faces in Iˆ0 with the lowest cost:

σˆ = arg min

σ∈Sn

n

i=1

Lmatch(Fi,Fˆσ(i)) = arg max

σ∈Sn

n

i=1

eF

i,Fˆσ(i) (3)

where Lmatch(Fi,Fˆσ(i)) = −eF

i,Fˆσ(i) is a pair-wise matching cost between a reference identity Fi and a generated one with index σ(i). This optimal assignment is computed efficiently with the Hungarian algorithm [15].

With the optimal assignment σˆ, shown as solid lines in the black box of Figure 3, we find the association between reference identities and generated identities. To improve multi-identity fidelity and alleviate confusion together, we define Multi-Identity Matching Reward (MIMR) as:

RMIMR =

1 MN

N

j=1

M

k=1

(λ11{k=ˆσ(j)} + λ21{k̸=ˆσ(j)})eF

j,Fˆk (4)

where λ1 > 0,λ2 < 0, adjusting gradient orientation according to σˆ.

- 4 Experiments

###### 4.1 Experiments Setting

###### 4.1.1 Implementation Details

To demonstrate our UMO’s generalist, the experiments are conducted based on two kinds of SOTA methods as the pretrained models:

- • UNO [32], an image customization method that supports multi-reference images based on in-context learning [12, 34] on DiT [21].

###### Reference Images

Prompt MS-Diffusion DreamO UNO UMO (Ours) OmniGen2 UMO (Ours)

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

a man standing in a city street.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

A man is using a hair dryer.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

A little girl is standing beside a man.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

A man and another man are looking at an avocado together.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

A woman, an old man, and an anime girl are standing together in a park.

Figure 5 Qualitative comparison with different methods on XVerseBench [4].

- • OmniGen2 [31], a unified model capable of understanding and diverse generation tasks, including image editing and in-context generation.

For the training hyperparameters, we set pretrain loss weight λ = 1 in Algorithm 1 and λ1 = 1, λ2 = −1 in Equation (4). The inference-related parameters in Algorithm 1 are set according to the suggestions of the pretrained models. Specifically, for UNO, we set the number of noise scheduler time steps T = 25, time step range [TS,Te] = [1,10] since the reward scores have become stable as shown in Figure 4a, while we set T = 50 and [TS,Te] = [1,20] for OmniGen2 according to Figure 4b.

We train these models with a learning rate of 5 × 10−6 and a total batch size of 8 on 8 NVIDIA A100 GPUs. We use LoRA [11] with rank of 512 during the training process. The remaining hyperparameters follow their own original settings.

###### 4.1.2 Comparative Methods

UMO is a unified multi-identity optimization framework to improving identity fidelity and reducing confusion. We compare it with the two pretrained models UNO [32] and OmniGen2 [31] as baselines. Except two baselines, we compare UMO with some leading methods which support multi-reference images, including MS-Diffusion [29], MIP-Adapter [14], OmniGen [33], DreamO [18] and XVerse [4].

###### 4.1.3 Evaluation Benchmark and Metrics

We evaluate these methods on XVerseBench [4] and OmniContext [31], which cover scenarios of both single reference and multi-reference. Since OmniContext only employs GPT-4.1 [19] to assess the generated outputs, we supplement the evaluation with the ID-Sim score in XVerseBench.

To measure the severity of multi-identity confusion, we propose a new metric ID-Conf, which is defined as the relative margin between the two most similar generated candidate faces for a reference identity, based on the observation the confusion occurs with several indistinct generated faces. Given several reference identities F = {F1,F2,...,Fn} and detected faces Fˆ = {Fˆ1,Fˆ2,...,Fˆm} from the generated result, we define the

###### Reference Images Prompt DreamO OmniGen2 UMO (Ours)

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

They do a pinky swear at the same time in a classroom.

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

They stand together, all laughing heartily, with the backdrop of a serene, sunny park.

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

They run together, capturing their energy and camaraderie on a sunny day, with the finish line in sight.

Figure 6 Qualitative comparison with different methods on OmniContext [31].

ID-Conf metric as below:

- ji[1] := arg max 1≤j≤m

cos Ψ(Fi),Ψ(Fˆj)

- ji[2] := arg max 1≤j≤m,j̸=ji[1]

cos Ψ(Fi),Ψ(Fˆj) (5)

cos Ψ(Fi),Ψ(Fˆj[2]

) cos Ψ(Fi),Ψ(Fˆj[1]

1 n 1≤i≤n

ID-Conf =

clip 1 −

i

,0,1

)

i

where Ψ is the model used in XVerseBench to get face embeddings. A larger value of the ID-Conf metric indicates a lower severity of identity confusion.

###### 4.2 Quantitative Evaluation

We compare UMO on XVerseBench [4] against the two pretrained models (i.e., UNO [32] and OmniGen2 [31]) and other SOTA customization methods, as shown in Table 1 and Table 2. In both single-subject and multi-subject scenarios, UMO significantly improves the ID-Sim and ID-Conf on both pretrained models, and has a remarkable lead over previous methods, indicating the effectiveness and generalization of UMO training framework. We also evaluate UMO on OmniContext [31] as shown in Table 3 where UMO also substantially boosts the identity consistency and alleviates confusion.

###### 4.3 Qualitative Analysis

We compare with various SOTA methods on XVerseBench [4] to verify the effectiveness of UMO as shown in

- Figure 5. From the top row to the bottom one, the identity promoting effect of UMO is scalable from single identity to multi-identity scenarios and has generalization on both UNO [32] and OmniGen2 [31]. Specifically, UMO improves identity fidelity with more similar generated faces with the reference on all scenarios and alleviates confusion as the last three rows shown with more distinct generated faces with each other. For example, UNO itself gets two similar girls, both different with the reference one, in the third row while the two reference identities can be easily discriminated in the result of UMO. Also, the customization ability of general subjects is boosted or retained as the second and the fourth rows shown. As a contrast, MS-Diffusion [29], UNO [32] and OmniGen2 [31] all suffer low identity fidelity. Although DreamO [18] shows moderate identity similarity, confusion issue occurs when the number of reference increases. Results of these models show the limited identity scalability of one-to-one mapping paradigm, while UMO indicates the superiority of

Method ID-Sim IP-Sim AVG MS-Diffusion [29]∗ 44.12 76.48 60.30 MIP-Adapter [14] 39.59 71.97 55.78 OmniGen [33] 76.51 78.46 77.49 DreamO [18] 75.48 70.84 73.16 XVerse [4] 79.48 76.86 78.17 UNO [32] 47.91 80.40 64.16 UMO (Ours) 80.89 77.09 78.99 OmniGen2 [31] 62.41 74.08 68.25 UMO (Ours) 91.57 79.74 85.66

- Table 1 Quantitative results on task type Single-Subject from XVerseBench. ∗: We evaluate MS-Diffusion with boxes set as in MS-Bench [29]. We highlight the best and the second-best values for each metric.

Method ID-Sim ID-Conf† IP-Sim AVG MS-Diffusion [29]∗ 38.98 66.40 70.98 58.79 DreamO [18] 50.24 68.67 64.63 61.18 XVerse [4] 66.59 72.44 71.48 70.17 UNO [32] 31.82 61.06 67.00 53.29 UMO (Ours) 69.09 78.06 68.57 71.91 OmniGen2 [31] 40.81 62.02 67.15 56.66 UMO (Ours) 71.59 77.74 73.80 74.38

- Table 2 Quantitative results on task type Multi-Subject from XVerseBench. †: We align ID-Conf score with the value range of the other metrics.

Method Overall ID-Sim† ID-Conf† AVG MS-Diffusion [29]∗ 4.72 2.32 6.59 4.54 DreamO [18] 6.25 4.44 6.23 5.64 UNO [32] 4.71 1.91 4.91 3.84 UMO (Ours) 5.34 4.62 6.60 5.52 OmniGen2 [31] 7.18 3.51 6.35 5.68 UMO (Ours) 7.16 7.07 7.60 7.28

- Table 3 Quantitative results on OmniContext. The Overall score is the geometric mean of Prompt Following (PF) and Subject Consistency (SC) scores.

multi-to-multi matching paradigm. Further comparison on OmniContext [31] demonstrates the positive impact on mitigating identity confusion as shown in Figure 6, where DreamO and OmniGen2 both suffer confusion issue, e.g., reference identities missing in the first row of OmniGen2 and the third row of DreamO, mismatching characteristics of each identity like mismatching hair in the first row of DreamO and wrongly assigned clothes in the second row of OmniGen2.

###### 4.4 User Study

We further conduct a user study questionnaire to show the superiority of UMO. Questionnaires are distributed to both domain experts and non-experts, who rank the results from each method along with several dimension,

User Study

UMO-OmniGen2 (Ours)

OmniGen2

Prompt Following

UMO-UNO (Ours) UNO DreamO MS-Diffusion

| |1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>|
|---|---|
|Consistency<br><br>| |

Aesthetic

Identity Co

Overall Performance

Figure 7 Radar charts of user evaluation of methods on different dimensions.

i.e., identity consistency, prompt following, aesthetic and overall performance. As shown in Figure 7, UMO achieves the best preference, demonstrating the effectiveness of multi-to-multi matching paradigm and showcasing its capability to deliver state-of-the-art results. Also, compared to the two baselines, i.e., UNO [32] and OmniGen2 [31], UMO gets significant improvement across all evaluated dimensions.

###### 4.5 Ablation Study

We conduct ablation study with UMO trained on UNO [32] on XVerseBench [4] as shown in Table 4 and Figure 8 to demonstrate the effect of ReReFL and MIMR. We also conduct further ablation study with UMO trained on OmniGen2 [31] on OmniContext [31] in Table 5.

###### 4.5.1 Effect of ReReFL

As shown in the first two columns of Figure 8 and the first two rows in Table 4, finetuning UNO with the same data as UMO (i.e., raw SFT) leads to minor improvement especially in ID-Sim and ID-Conf scores, while optimizing UNO with ReReFL demonstrates significant improvement. The comparison indicates the necessity to utilize reinforcement learning with reward focusing on facial region to unleash potential in identity consistency. Instead, vanilla SFT would suppress attention of facial feature due to its small proportion. Similar comparison on OmniGen2 in Table 5 shows that the effect of ReReFL is general.

Method ID-Sim ID-Conf† IP-Sim AVG UNO [32] 31.82 61.06 67.00 53.29 SFT 33.94 62.88 65.17 54.00 ReReFL w/ SIR 65.16 65.28 67.25 65.90 UMO (Ours) 69.09 78.06 68.57 71.91

- Table 4 Ablation study with UNO as the pretrained model on task type Multi-Subject from XVerseBench.

- Method Overall ID-Sim† ID-Conf† AVG OmniGen2 [31] 7.23 2.86 6.67 5.59 SFT 7.24 3.38 6.80 5.81 ReReFL w/ SIR 7.14 6.44 7.32 6.97 UMO (Ours) 7.14 6.61 9.04 7.60
- Table 5 Ablation study with OmniGen2 as the pretrained model on task type MULTI from OmniContext.

###### Reference Images & Prompt

UNO SFT ReReFL w/ SIR UMO (Ours)

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

A woman and a girl standing together in a park.

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

A man and another man standing side by side on a street, having a conversation.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

A girl is standing beside a woman, having a friendly chat.

Figure 8 Visualization of ablation study. Zoom in for details.

###### 4.5.2 Effect of MIMR

The last two rows of Table 4 demonstrate that training with SIR instead of MIMR has a significant drop in both ID-Sim and ID-Conf in multi-subject scenario. As the last two columns of Figure 8 shown, although SIR enhances identity similarity as well, it suffers severe confusion problem, e.g., result of training with SIR in the first row of Figure 8 has two similar identities with the reference girl missing, and the result in the last row suffers mismatching characteristics of each identity, i.e., the hair color of the generated woman. The comparison proves the effectiveness of MIMR through assigning correct facial supervisions to boost identity consistency and mitigate confusion. Similar observation in Table 5 indicates the generalization of the effect of MIMR.

###### 5 Conclusion

In this paper, we present UMO, a Unified Multi-identity Optimization framework to improve identity consistency and alleviate confusion in multi-reference scenarios, which is based on the multi-to-multi matching paradigm through a novel Reference Reward Feedback Learning algorithm with scalable Multi-Identity Reference Reward that reformulating multi-identity generation as a global assignment optimization problem. Additionally, we develop a scalable customization dataset along with a new metric to evaluate the extent of multi-identity confusion. Extensive experiments demonstrate that UMO significantly enhances the identity preserving ability with less confusion and greater identity scalability on various customized models, setting a new state-of-the-art among open-source methods along the dimension of identity preserving.

###### References

- [1] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=YCWjhGrJFD.

- [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

- [4] Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025.

- [5] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, and Hengshuang. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024.

- [6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [8] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

- [9] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in neural information processing systems, 37:36777–36804, 2024.

- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

- [12] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.

- [13] Mengqi Huang, Zhendong Mao, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom: Narrowing real text word for real-time open-domain text-to-image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7476–7485, 2024.

- [14] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multi-condition confusion for finetuning-free personalized image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 3707–3714, 2025.

- [15] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2): 83–97, 1955.

- [16] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. URL https://github.com/ black-forest-labs/flux. Accessed: 2025-02-07.
- [17] Zhendong Mao, Mengqi Huang, Fei Ding, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom++: Representing images as real-word for real-time customization. arXiv preprint arXiv:2408.09744, 2024.

- [18] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [19] OpenAI. Introducing gpt-4.1 in the api, 2025. URL https://openai.com/index/gpt-4-1. Accessed: 2025-04-14.
- [20] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [21] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [22] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [23] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [25] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

- [26] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

- [27] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [28] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

- [29] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024.

- [30] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.

- [31] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [32] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.

- [33] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.

- [34] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. arXiv preprint arXiv:2310.08580, 2023.

- [35] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

- [36] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [37] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025.

- [38] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

- [39] Zilyu Ye, Jinxiu Liu, JinJin Cao, Zhiyang Chen, Ziwei Xuan, Mingyuan Zhou, Qi Liu, and Guo-Jun Qi. Openstory: A large-scale open-domain dataset for subject-driven visual storytelling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7953–7962, 2024.

- [40] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

- [41] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pages 1–21. Springer, 2022.

## UMO: Scaling Multi-Identity Consistency for Image Customization via Matching Reward

Appendix

- F Detailed Quantitative Comparisons

We report detailed comparison on each task type from OmniContext [31] as shown in Table 6, Table 7 and Table 8. In all SINGLE, MULTI and SCENE task types from OmniContext, UMO significantly boosts the ID-Sim and ID-Conf on both pretrained models, i.e., UNO [32] and OmniGen2 [31], leading over previous methods, e.g., MS-Diffusion [29] and DreamO [18]. The comprehensive evaluation demonstrate the effectiveness and generalization of UMO training framework to improve identity consistency and mitigate confusion.

Method Overall ID-Sim† ID-Conf† AVG MS-Diffusion [29] 5.83 2.89 6.05 4.92 DreamO [18] 7.65 5.09 5.83 6.19 UNO [32] 6.72 2.11 4.48 4.44

- UMO (Ours) 6.77 5.19 7.03 6.33 OmniGen2 [31] 7.82 4.75 7.08 6.55

- UMO (Ours) 7.78 7.95 6.72 7.48

- Table 6 Quantitative results on task type SINGLE from OmniContext.

Method Overall ID-Sim† ID-Conf† AVG MS-Diffusion [29] 4.75 2.18 6.97 4.63 DreamO [18] 7.05 4.21 7.12 6.13 UNO [32] 4.48 1.75 5.23 3.82 UMO (Ours) 5.35 4.46 7.20 5.67 OmniGen2 [31] 7.23 2.86 6.67 5.59 UMO (Ours) 7.14 6.61 9.04 7.60

- Table 7 Quantitative results on task type MULTI from OmniContext.

Method Overall ID-Sim† ID-Conf† AVG MS-Diffusion [29] 3.95 1.90 6.75 4.20 DreamO [18] 4.52 4.03 5.74 4.76 UNO [32] 3.59 1.87 5.03 3.50 UMO (Ours) 4.38 4.22 5.58 4.73 OmniGen2 [31] 6.71 2.91 5.31 4.98 UMO (Ours) 6.78 6.65 7.03 6.82

- Table 8 Quantitative results on task type SCENE from OmniContext.

###### G More Qualitative Results

We show more qualitative results on XVerseBench [4] in Figure 9 and Figure 10, and OmniContext [31] in Figure 11 and Figure 12. UMO improves identity similarity without confusion on both single identity and multi-identity scenarios, showing its general and scalable effectiveness.

- • Single Identity: In Figure 9, UNO [32] itself generates customization results with low identity fidelity. As a comparison, UMO gets much more similar generated identities. In Figure 11, although OmniGen2 [31] gets moderate fidelity, UMO based on it still achieves remarkable improvement without degradation of subject similarity (e.g., clothes, etc) or prompt following. The observation in single identity scenario demonstrates the extraordinary potential of UMO to enhance identity consistency across several existing models.
- • Multi-Identity: In Figure 11, UNO [32] suffers low similarity of facial features and identity confusion, e.g., the two generated identities in the last row have almost the same facial features which is the “average” facial features of the two reference ones. By contrast, UMO shows its superiority with higher fidelity and without identity confusion. In Figure 12, the results of OmniGen2 [31] show moderate identity similarity, while UMO still boosts it without degradation. The observation in the scenario of multi-identity shows the impressive promoting ability of UMO to improve identity fidelity and alleviate confusion on existing image customization methods.

###### H Discussion

Although we build UMO to maintain high-fidelity identity preservation and alleviate identity confusion with scalability to multi-identity, stably scaling to more identities is still restricted due to the dramatic decrease of the pretrained models’ reference ability when the number of reference images or identities increases, which demonstrates a similar view with [40].

Scenarios Prompts

(1) The man on the beach.

- (2) The girl holds a board written "UMO"
- (3) The man is skateboarding on the street.

Single Identity

(4) Transform the style of image into Sketch style

(1) The man holding the can. (2) The elf wearing the T-shirt in the second image, with "UXO Team" written on it. (3) The woman in the first image is running with the dog in the second image.

Single Identity + Subject

(1) The man and the woman are sitting in a classroom (2) The man in the first image and the man in the second image shake hands and look straight ahead. (3) Half-body portrait of the woman in the first image and the old man in the second image, retro comic style Two Identities + Subject

Two Identities

- (1) The woman is holding the toy, while the man standing beside.
- (2) The man is holding a hair dryer and drying the woman’s hair.

(1) The three people are playing cards. (2) Portrait of the three people, oil painting style.

More Identities

Table 9 The detailed prompts used in Figure 1.

###### Reference Images Prompt UNO UMO (Ours)

[Figure 223]

[Figure 224]

[Figure 225]

a woman smiling in a flower-filled garden

[Figure 226]

[Figure 227]

[Figure 228]

a girl smiling in a flower-filled garden

[Figure 229]

[Figure 230]

[Figure 231]

a man standing in a city street

[Figure 232]

[Figure 233]

[Figure 234]

a man standing in a city street

[Figure 235]

[Figure 236]

[Figure 237]

a woman in a red dress smiling

Figure 9 Qualitative results on task type Single-Subject from XVerseBench [4].

###### Reference Images Prompt UNO UMO (Ours)

[Figure 238]

[Figure 239]

A woman and a girl standing side by side in a park.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

An old man and a man standing together on the street.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

A man and a woman standing side by side.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

A man and a woman standing together on a sunny street.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

A man and a woman standing side by side in a park.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

A man is standing beside another man.

[Figure 260]

[Figure 261]

Figure 10 Qualitative results on task type Multi-Subject from XVerseBench [4].

###### Reference Images OmniGen2 UMO (Ours)

###### Prompt

[Figure 262]

[Figure 263]

[Figure 264]

Place the woman in the image in a vibrant urban park at night, adjusting her hair with both hands while smiling warmly at the camera, her handbag resting beside her on a nearby bench, with the city lights twinkling in the background.

[Figure 265]

[Figure 266]

[Figure 267]

A person with long dark hair is joyfully chatting with friends at a colorful cultural celebration.

[Figure 268]

[Figure 269]

[Figure 270]

Let the boy joyfully dance in a sunlit garden filled with colorful flowers.

[Figure 271]

[Figure 272]

[Figure 273]

Show a person posing for a picture in a black off-shoulder dress amidst a snowy landscape.

Figure 11 Qualitative results on task type SINGLE Character from OmniContext [31].

###### Reference Images Prompt OmniGen2 UMO (Ours)

[Figure 274]

[Figure 275]

Please make the man and the women play a computer game together.

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Please make the woman and the man play chess together.

[Figure 282]

[Figure 283]

I wish to see the person from figure 1 and the individual from photo2 pointing at the ground together.

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Have the person run together with the man in a dense forest.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Two individuals lie down and rest with eyes closed in a peaceful park.

Figure 12 Qualitative results on task type MULTI Character from OmniContext [31].

