# arXiv:2410.18977v2[cs.CV]22Jan2025

## Pay Attention and Move Better: Harnessing Attention for Interactive Motion Generation and Training-free Editing

#### Ling-Hao Chen1,2∗, Shunlin Lu3 , Wenxun Dai1, Zhiyang Dou4, Xuan Ju5, Jingbo Wang6

Taku Komura4, Lei Zhang2† 1Tsinghua University, 2International Digital Economy Acadamy (IDEA Research) 3The Chinese University of Hong Kong, Shenzhen, 4The University of Hong Kong 5The Chinese University of Hong Kong, 6Shanghai AI Laboratory Project page: https://lhchen.top/MotionCLR

[Figure 1]

[Figure 2]

(A) Motion Emphasis and De-emphasis.

(B) In-place Motion Replacement.

[Figure 3]

[Figure 4]

[Figure 5]

(D) Motion Sequence Shifting. (E) Example-based Generated Motion.

(C) Motion Style Transfer

Figure 1: We propose MotionCLR, supporting interactive motion generation and versatile editing. The blue and red characters represent original and edited motions. (A) Motion deemphasis and emphasis via adjusting the weight of “jump”. (B) In-place replacing the action of “runs” with “jumps”. (C) Transferring motion style referring to two motions. From left to right, there are motion style reference, motion texture reference, and transferred motion. (D) Shifting the order of “walking” and “sitting” actions in a motion. (E) Generating diverse motion with the same example motion, a.k.a. example-based motion generation or crowd animation. The left crowd is boxing animation and the right crowd is kicking animation.

### Abstract

This research delves into the problem of interactive editing of human motion generation. Previous motion diffusion models lack explicit modeling of the wordlevel text-motion correspondence and good explainability, hence restricting their fine-grained editing ability. To address this issue, we propose an attention-based motion diffusion model, namely MotionCLR, with CLeaR modeling of attention mechanisms. Technically, MotionCLR models the in-modality and cross-modality interactions with self-attention and cross-attention, respectively. More specifically, the self-attention mechanism aims to measure the sequential similarity between frames and impacts the order of motion features. By contrast, the cross-attention mechanism works to find the fine-grained word-sequence correspondence and activate the corresponding timesteps in the motion sequence. Based on these key properties, we develop a versatile set of simple yet effective motion editing methods via manipulating attention maps, such as motion (de-)emphasizing, in-place motion replacement, and example-based motion generation, etc. For further verification of the explainability of the attention mechanism, we additionally explore the potential of action-counting and grounded motion generation ability via attention maps. Our

∗This work was done while Ling-Hao Chen was an intern at IDEA Research. †Corresponding author.

Technical report, a preprint (MotionCLR v1.1).

experimental results show that our method enjoys good generation and editing ability with good explainability.

### 1 Introduction

Recently, text-driven human motion generation [Ahn et al., 2018, Petrovich et al., 2022, Tevet et al., 2022b, Lu et al., 2024, Guo et al., 2024a, Hong et al., 2022, Wang et al., 2022, 2024] has attracted significant attention in the animation community for its great potential to benefit versatile downstream applications, such as games and embodied intelligence. As the generated motion quality in one inference might be misaligned with the users’ intents, interactive motion editing plays a crucial role in the community by introducing humans into the loop of human-machine interaction.

To this end, some attempts have been made to edit or control the generated motion by specifying sparse motion signals, like 3D joint trajectories [Xie et al., 2024, Dai et al., 2024, Shafir et al., 2024] or motion clips [Tang et al., 2022, Harvey et al., 2020]. Despite such progress, the constraints introduced in these works are mainly in-modality (motion) constraints, which require laborious efforts in the real animation creation pipeline. Such interaction fashions strongly restrict the involving humans in the loop of creation. In this work, we aim to develop a more user-friendly editing fashion of introducing out-of-modality signals, such as editing texts. For example, when generating a motion with the prompt “a man jumps.”, we can control the height or times of the “jump” action via adjusting the importance weight of the word “jump”. Alternatively, we can also in-place replace the word “jump” with other actions specified by users. The retrieval-based data annotation process of previous work [Athanasiou et al., 2024] not only restricts the semantic editing applications, but also requires expendable labor. In contrast, in this work, we would like to equip the motion generation model with such abilities in a training-free style.

However, the key limitation of existing motion generation models is that the modeling of previous generative methods lacks word-level text-motion correspondence. This fine-grained cross-modality modeling not only plays a crucial role in text-motion alignment, but also makes it easier for finegrained editing. To show the problem, we revisit previous transformer-based motion generation models [Tevet et al., 2022b, Zhang et al., 2024a, 2023b, Chen et al., 2023]. The transformer-encoderlike methods [Tevet et al., 2022b, Zhou et al., 2024] treat the textual input as one special embedding before the motion sequence. However, such integrated text embeddings and motion embeddings imply substantially different semantics, indicating unclear correspondence between the semantics of specific words and motions. Besides, this fashion over-compresses a sentence into one embedding, which compromises the fine-grained correspondence between each word and each motion frame,

- as evidenced in Fig. 2. Although there are some methods [Zhang et al., 2024a, 2023b] to perform texts and motion interactions via linear cross-attention, they fuse the diffusion timestep embeddings with textual features together in the forward process. This operation undermines the structural text representations and weakens the input textual conditions (Sec. Fig. 2). Through these observations, we argue that the fine-grained text-motion correspondence in these two motion diffusion fashions is not well explored.

To resolve these issues, in this work, we propose a motion diffusion model, namely MotionCLR, with a CLeaR3 modeling of the motion generation process and word-motion correspondence. The main component of MotionCLR is a CLR block, which is composed of a convolution layer, a self-attention layer, a cross-attention layer, and an FFN layer. In this basic block, the cross-attention layer is used to encode the text conditions for each word. More specifically, the cross-attention operation between each word and each motion frame models the text-motion correspondence at the word level. Meanwhile, the timestep injection of the diffusion process and the text encoding are modeled separately. Besides, the self-attention layer in this block is designed for modeling the interaction between different frames and the FFN is a common design for channel mixing.

Motivated by previous progress in the explainality of the attention mechanism [Vaswani et al., 2017, Ma et al., 2023, Hao et al., 2021, Xu et al., 2015, Hertz et al., 2023, Chefer et al., 2021b,a], this work delves into the mathematical properties of the basic CLR block, especially the cross-attention and self-attention mechanisms. In the CLR block, the cross-attention value of each word along the time axis works as an activator to determine the execution time of each action. Besides, the self-attention

3For clarification, the word “Clear” here means good explainability of the model.

mechanism in the CLR block mainly focuses on mining similar motion patterns between frames. Our empirical studies verify these properties. Although there are some early explanations [Raab et al., 2024a] about self-attention in motion generation, understanding both cross and self-attentions in one system is still unexplored, which mainly owes to the “clear” and separate modeling of both attention mechanisms. Based on these key observations, we show how we can achieve semantic motion editing tasks, e.g. motion (de-)emphasis, in-place motion replacement, and motion erasing by manipulating cross-attention. Additionally, our method can also be applied to other applications like style transfer, sequence shifting, and generating motions from an example by manipulating self-attention calculations. We verify the effectiveness of these editing methods via both qualitative and quantitative experimental results. Additionally, we explore how our method can be applied to cope with the hallucination of generative models.

Before delving into the technical details of this work, we summarize our key contributions as follows.

- • We propose an attention-based motion diffusion model, namely MotionCLR, with clear modeling of the text-aligned motion generation process.
- • For the first time in the human animation community, we clarify the roles that self- and crossattention mechanisms play in one attention-based motion diffusion model.
- • Thanks to these observations, we propose a series of interactive motion editing tasks (see Fig. 1) via manipulating attention layers. Importantly, both motion generation and editing are performed in one model. We additionally explore the potential of grounded motion generation when facing hallucination (Sec. 6).
- • We evaluate the generation quality of our method and achieve comparable generation performance with state-of-the-art methods. Besides, the training-free editing methods even outperform baselines requiring specific training in some scenarios.

### 2 Related Work and Contribution

- 2.1 Text-driven Human Motion Generation

Previous text-driven human motion generation [Plappert et al., 2018, Ahn et al., 2018, Lin and Amer, 2018, Ahuja and Morency, 2019, Bhattacharya et al., 2021, Tevet et al., 2022a, Petrovich et al., 2022, Hong et al., 2022, Guo et al., 2022b, Zhang et al., 2024a, Athanasiou et al., 2022, Tevet et al., 2022b, Wang et al., 2022, Chen et al., 2023, Dabral et al., 2023, Yuan et al., 2023, Zhang et al., 2023a, Shafir et al., 2024, Zhang et al., 2023b, Karunratanakul et al., 2023, Jiang et al., 2024, Xie et al., 2024, Lu et al., 2024, Wan et al., 2024, Liu et al., 2024, Zhou et al., 2024, Petrovich et al., 2024, Barquero et al., 2024, Wang et al., 2024] uses textual descriptions as input to synthesize human motions. One of the main generative fashions is a kind of GPT-like [Zhang et al., 2023a, Lu et al., 2024, Guo et al., 2024a, Jiang et al., 2024] motion generation method, which compresses the text input into one conditional embedding and predicts motion in an auto-regressive fashion. Besides, the diffusion-based method [Tevet et al., 2022b, Zhang et al., 2024a, 2023b, Zhou et al., 2024, Chen

- et al., 2023, Dai et al., 2024] is another generative fashion in motion generation. Note that most work with this fashion also utilizes transformers [Vaswani et al., 2017] as the basic network architecture. Although these previous attempts have achieved significant progress in the past years, the technical design of the explainability of the attention mechanism is still not well considered. We hope this work will provide a new understanding of these details. Besides, previous motion generation models also lack the capability of zero-shot motion editing, which is what we would like to explore in this work.

2.2 Human Motion Editing The human motion editing task aims to edit a motion satisfying human demand. Previous works [Dai

- et al., 2024, Dabral et al., 2023, Kim et al., 2023] attempt to edit a motion in a controlling fashion, like motion inbetweening and joint controlling. There are some other methods [Raab et al., 2023, Aberman et al., 2020b, Jang et al., 2022] trying to control the style of a motion. However, these works are either designed for a specific task or cannot edit fine-grained motion semantics, such as the height or times of a “jump” motion. Raab et al. [2024a] perform motion following via replacing the queries in the self-attention, which does not consider the semantic manipulations. Goel et al. [2024] propose to edit a motion with an instruction. However, the MEOs pipeline relies on the text-only LLM outputs,

which will introduce hallucinations. MotionFix [Athanasiou et al., 2024] proposes to use the language command to edit motions. However, it needs annotations on the editing text, additionally requiring more labor efforts. COMO [Huang et al., 2025] introduces editing motion via Large Language Models (LLMs) as a translator, which lacks the editing grounds of original motion content. The key reason why the existing method cannot achieve training-free motion editing is that the fine-grained text-motion correspondence in the cross-attention still lacks an in-depth understanding. There are also some methods designed for motion generation [Li et al., 2002] or editing [Lee and Shin, 1999, Holden et al., 2016], which are limited to adapt to diverse downstream tasks.

Our key insights and contribution over previous attention-based motion diffusion models [Tevet et al., 2022b, Zhang et al., 2024a, 2023b, Zhou et al., 2024, Chen et al., 2023, Dai et al., 2024] lie in the clear explainability of the self-attention and cross-attention mechanisms in diffusion-based motion generation models. The cross-attention module in our method models the text-motion correspondence

- at the word level explicitly. Besides, the self-attention mechanism models the motion coherence between frames. Therefore, we can easily clarify what roles self-attention and cross-attention mechanisms play in this framework, respectively. To the best of our knowledge, it is the first time in the human animation community to clarify these mechanisms in one system and explore how to perform training-free motion editing involving humans in the loop.

#### 2.3 Visual Editing for Image Contents

Image editing in diffusion models has been more explored than motion editing. Previous studies have achieved exceptional realism and diversity in image editing [Hertz et al., 2023, Han et al., 2023, Parmar et al., 2023, Cao et al., 2023, Tumanyan et al., 2023, Zhang et al., 2023c, Mou et al., 2024, Ju et al., 2024] by manipulating attention maps. Especially, although Hertz et al. [2023] proposes to introduce cross-attention into image editing, these techniques and self-attention-based motion editing are still under-explored. However, relevant interactive editing techniques and observations are still unexplored in the human animation community. The basic reason for insufficient exploration for human animation is lacking a fine-grained modeling between words and motions.

### 3 Base Motion Generation Model and Understanding Attention Mechanisms

In this section, we will introduce the proposed motion diffusion model, MotionCLR, composed of several basic CLR modules. Specifically, we will analyze the technical details of the attention mechanism to obtain an in-depth understanding of this.

#### 3.1 How Does MotionCLR Model Fine-grained Cross-modal Correspondence?

Regarding the issues of the previous methods (see Sec. 1), we carefully design a simple yet effective motion diffusion model, namely MotionCLR, with fine-grained word-level text-motion correspondence. The MotionCLR model is a U-Net-like architecture [Ronneberger et al., 2015]. Here, we name the down/up-sampling blocks in the MotionCLR as sampling blocks. Each sampling block includes two CLR blocks and one down/up-sampling operation. In MotionCLR, the atomic block is the CLR block, which is our key design. Specifically, a CLR block is composed of four modules,

- • Convolution-1D module, a.k.a. Conv1d(·), is used for timestep injection, which is disentangled with the text injection. The design principle here is to disentangle the text embeddings and the timestep embeddings for explicit modeling for both conditions.
- • Self-attention module is designed for learning temporal coherence between different motion frames. Notably, different from previous works [Tevet et al., 2022b, Zhou et al., 2024, Shafir et al., 2024], self-attention only models the correlation between motion frames and does not include any textual inputs. The key motivation here is to separate the motion-motion interaction from the text-motion interaction of traditional fashions [Tevet et al., 2022b].
- • Cross-attention module plays a crucial role in learning text-motion correspondence in the CLR block. It takes word-level textual embeddings of a sentence for cross-modality interaction, aiming to obtain fine-grained text-motion correspondence at the word level. Specifically, the attention map models the relationship between each frame and each word, enabling more fine-grained cross-modality controlling.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Conv1d

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

CLR block

[Figure 17]

t

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

...

[Figure 23]

[Figure 24]

Self Attn.

[Figure 25]

[Figure 26]

a

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

...

[Figure 32]

[Figure 33]

CLR block

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

...

man

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Cross Attn.

...

walks

text

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

...

Down/Up sampling

forward

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

word-level text embeddings

...

FFN

[Figure 62]

motion features

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

U-Net Sampling block

in word level

CLR block

- Figure 2: System overview of MotionCLR architecture. (a) The U-Net-like denoising network is with two CLR blocks before down/up-sampling. (b) The basic CLR block includes four layers, separating the timestep injection and the text condition. (c) The key component is the text-motion cross-attention at the word level.

[Figure 69]

[Figure 70]

0 20 40 60 80 100 120 140 160

[Figure 71]

a

person jumps

low

high

[Figure 72]

0 20 40 60 80 100 120 140 160

020406080100120140160

[Figure 73]

low

- (A) high
- (B)

(C)

- Figure 3: Empirical study of attention mechanisms. We use “a person jumps.” as an example. (A) Keyframes and the root trajectory of generated motion. The character jumps on ∼ 15 − 40f, ∼ 60 − 80f, and ∼ 125 − 145f, respectively. (B) The cross-attention map between timesteps and words. The “jump” word is highly activated aligning with the “jump” action. (d) The self-attention map visualization. It is obvious that the character jumps three times, reflecting nine areas in the self-attention map. Different jumps share similar local motion patterns.

• FFN module works as an additional feature transformation and extraction [Dai et al., 2022, Geva et al., 2021], which is a necessary component in transformer-based architectures.

In summary, in the basic CLR block, we model interactions between frames and word correspondence, separately in cross-attention. We analyze both self-attention and cross-attention of MotionCLR in the following sections, which is useful for subsequent editing tasks.

#### 3.2 Mathematical Preliminaries of Attention Mechanism

The general attention mechanism has three key components, query (Q), key (K), and value (V), respectively. The output X′ of the attention mechanism can be formulated as,

√

X′ = softmax(QK⊤/

d)V, (1) where Q ∈ RN

2×d. Here, d is the embedding dimension of the text or one-frame motion. In the following section, we take t = 0,1,···,T as diffusion timesteps, and f = 1,2,···,F as the frame number of motion embeddings X ∈ RF×d. For convenience, we name S = QK⊤ as the similarity matrix and A = softmax(QK⊤/

1×d, K,V ∈ RN

√

d) as the attention map in the following sections.

#### 3.3 Self and Cross Attention Mechanisms in MotionCLR The self-attention mechanism uses different transformations of motion features X as inputs,

Q = XWQ, K = XWK, V = XWV , (2)

where Q,K,V ∈ RF×d, F = N1 = N2. We take a deep look at the formulation of the self-attention mechanism. As shown in Eq. (1), the attention calculation begins with a matrix multiplication

operation, meaning the similarity (S = QK⊤ ∈ RF×F) between Q and K. Specifically, for each row i of S, it obtains the frame most similar to frame i. Here √

d is a normalization term. After obtaining the similarity for all frames, the softmax(·) operation is not only a normalization function, but also works as a “soft” max(·) function for selecting the frame most similar to frame i. Assuming the j-th frame is selected as the frame most similar to frame i with the maximum activation, the final multiplication with V will approximately replace the motion feature Vj at the i-th row of X′. Here, the output X′ is the updated motion feature. In summary, we have the following remark.

Remark 1. The self-attention mechanism measures the motion similarity of all frames and aims to select the most similar frames in motion features at each place.

The cross-attention mechanism of MotionCLR uses the transformation of a motion as a query, and the transformation of textual words as keys and values,

Q = XWQ, K = CWK, V = CWV , (3) where C ∈ RL×d is the textual embeddings of L word tokens, Q ∈ RF×d, K,V ∈ RL×d. Note that W⋆ in Eq. (2) and Eq. (3) are not the same parameters, but are used for convenience. As shown in Eq. (3), K and V are both the transformed text features. Recalling Eq. (1), the matrix multiplication operation between Q and K measures the similarity (S = QK⊤) between motion frames and words in a sentence. Similar to that in self-attention, the softmax(·) operation works as a “soft” max(·) function to select which transformed word embedding in V should be selected at each frame. This operation models the motion-text correspondence explicitly. Therefore, we have the second remark.

Remark 2. The cross-attention first calculates the similarity to determine which word (i.e. value in cross attention) should be activated at the i-th frame. The final multiplication operation with values places the semantic features of their corresponding frames.

#### 3.4 Empirical Evidence on Understanding Attention Mechanisms

To obtain a deeper understanding of the attention mechanism and verify the mathematical analysis of attention mechanisms, we provide some empirical studies on some cases.

As shown in Fig. 3, we take the sentence “a person jumps.” as an example. We visualize the keyframe in Fig. 3(A), where we also visualize the root trajectory. As can be seen in Fig. 3(A), the character jumps at ∼ 15 − 40f, ∼ 60 − 80f, and ∼ 125 − 145f, respectively. Note that, as shown in Fig. 3(B), the word “jump” is significantly activated aligning with the “jump” action in the crossattention map. This not only verifies the soundness of the fine-grained text-motion correspondence modeling in MotionCLR, but also meets the theatrical analysis of motion-text (Q-K) similarity. This motivates us to manipulate the attention map to control when the action will be executed. The details will be introduced in Sec. 4. We also visualize the self-attention map in Fig. 3(C). As analyzed in Sec. 3.3, the self-attention map evaluates the similarity between frames. As can be seen in Fig. 3(C), the attention map highlights nine areas with similar motion patterns, indicating three jumping actions in total. Besides the temporal areas that the “jmup” word is activated are aligned with the jumping actions. The highlighted areas in the self-attention map are line areas, not square areas, indicating the taking-off, in-the-air, and landing actions of a jump with different detailed movement patterns. Due to the page limits, we leave more visualization for empirical evidence in Appendix D.

### 4 Motion Editing Applications via Attention Manipulations

Analysis in Sec. 3.3 has revealed the roles that attention mechanisms play in MotionCLR. In this section, we will show versatile downstream tasks of MotionCLR via manipulating attention maps.

length of text: L

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

lengthofmotion: F

V (value)

replace

reference atention map edited atention map

- (a) In-place motion replacement via replacing cross-attention map (Sec. 4.1). The batch size of inference examples is two during the inference stage (reference and edited motion respectively). We replace the crossattention map of the edited motion as the one of the reference motion.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

lengthofmotion:F

length of text: L

×(1 + α)

- (b) Motion (de-)emphasis via in/de-creasing cross-attention value. We can emphasize or de-emphasize a word in the prompt by adjusting the weight of the word. α < 0 and α > 0 mean emphasis and de-emphasis respectively.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

lengthofmotion: F

length of text: L

shifting

- (c) Motion sequence shifting via shifting self-attention map. We adjust the sequence order of the motion sequence via shifting the attention map along the temporal axis.

Figure 4: Diagram of motion editing via manipulating attention maps.

- 4.1 In-place Motion Replacement

In real scenarios, we would like to edit some local motion contents of the generated result. The key challenge is to keep the original composition and replace it with the new motion contents. Assuming we generate a reference motion at first, we would like to replace one action in the reference motion

with another in place. Therefore, the batch size of inference examples is two during the inference stage, where the first is the reference motion and the other is the edited motion. As discussed in Sec. 3.3, the cross-attention map determines when an action happens. Motivated by this, we replace the cross-attention map of the edited motion as the one of the reference motion. As shown in Fig. 4a, we use the replaced attention map to multiply the value matrix (text features) to obtain the output. As a result, we will obtain the motion of edited semantics with referenced temporal composition.

#### 4.2 Motion Emphasis and De-emphasis

In the text-driven motion generation framework, the process is driven by the input text. As discussed in Sec. 3.3, the verb of the action will be significantly activated in the cross-attention map when the action is executed. As shown in Fig. 4b, if we increase/decrease the attention value of a verb in the cross-attention map, the corresponding action will be emphasized/de-emphasized, which can be implemented by multiplication (A:,i ← A:,i × (1 + α)). Here, positive and negative values of α represent the motion emphasis and de-emphasis, respectively. Besides, this method can also be extended to the grounded motion generation application, which will be introduced in Sec. 6.

#### 4.3 Motion Erasing

Motion erasing is a special case of motion de-emphasis. We treat it as a special case of motion de-emphasis. When the decreased (de-emphasized) cross-attention value of an action is small enough, the corresponding action will be erased. The difference with motion de-emphasis is that the motion erasing is applied in a sub-temporal area specified by users in the whole sequence, and the de-emphasis is applied in the whole sequence.

Despite these semantic editing applications, MotionCLR also supports other editing tasks by manipulating self-attention.

#### 4.4 Motion Sequence Shifting

It is obvious that the generated motion is a combination of different actions along the time axis. Sometimes, users would like to shift a part of the motion along the time axis to satisfy the customized requirements. As shown in Fig. 4c, we can shift the motion sequentiality by shifting the self-attention map. As discussed in Sec. 3.3, self-attention is only related to the motion feature without related to the semantic condition, which is our motivation on manipulating the self-attention map. Thanks to the denoising process, the final output sequence should be a natural and continuous sequence.

#### 4.5 Example-based Motion Generation

As defined by Li et al. [2023b, 2002], example-based motion generation aims to generate novel motions referring to an example motion. In MotionCLR system, this task is a special case of the motion sequence shifting. That is to say, we can shuffle the self-attention map along the temporal axis to obtain the diverse motions referring to the example. As analyzed in Sec. 3.3, the “value”s (V) in self-attention means the texture of the motion. Therefore, shuffling the self-attention map along the temporal axis without manipulating “value”s compromises the similar motion texture to the original one.

#### 4.6 Motion Style Transfer

As discussed in the technical details of the self-attention mechanism, the values mainly contribute to the contents of motion and the attention map determines the selected indices of motion frames. Following Raab et al. [2024a], when synthesizing two motion sequences (M1 and M2 respectively), we only need to replace Qs in M2 with that in M1 to achieve the style of M2 into M1’s. Specifically, queries (Qs) in M2 determine which motion feature in M2 is the most similar to that in M1 at each timestep. Accordingly, these most similar motion features are selected to compose the edited motion. Besides, the edited motion is with the motion content of M2 while imitating the motion style of M1.

- Table 1: Comparison with different methods on the HumanML3D dataset. The baselines include diffusion-based methods and state-of-the-art methods. The “†” notation denotes the DPM-solver sampling inference design choice and “∗” is the DDIM sampling choice. As DPM-solver and DDIM present comparable performance, without specification, we set the DDIM sampling as our default choice. The comparison shows that MotionCLR is with comparable performance with state-of-the-art methods.

R-Precision↑ FID↓ MM-Dist↓ Multi-Modality↑ Top 1 Top 2 Top 3

Methods

MDM [2022b] - - 0.611±0.007 0.544±0.044 5.566±0.027 2.799±0.072 MLD [2023] 0.481±0.003 0.673±0.003 0.772±0.002 0.473±0.013 3.196±0.010 2.413±0.079 MotionDiffuse [2024a] 0.491±0.001 0.681±0.001 0.782±0.001 0.630±0.001 3.113±0.001 1.553±0.042 ReMoDiffuse [2023b] 0.510±0.005 0.698±0.006 0.795±0.004 0.103±0.004 2.974±0.016 1.795±0.043 MoMask [2024a] 0.521±0.002 0.713±0.002 0.807±0.002 0.045±0.002 2.958±0.008 1.241±0.040 MotionCLR† 0.542±0.001 0.733±0.002 0.827±0.003 0.099±0.003 2.981±0.011 2.145±0.043 MotionCLR∗ 0.544±0.001 0.732±0.001 0.831±0.002 0.269±0.001 2.806±0.014 1.985±0.044

### 5 Experiments

#### 5.1 Implementation Details

The MotionCLR model is trained on the HumanML3D dataset with one NVIDIA A-100 GPU based on PyTorch [Paszke et al., 2019]. The latent dimension of the motion embedding is 512. We take the CLIP-ViT-B model to encoder text as word-level embeddings. The training process utilizes a batch size of 64, with a learning rate initialized at 2e − 4 and decaying at a rate of 0.9 every 5,000 steps. Additionally, a weight decay of 1e − 2 is employed to regularize the model parameters. For the diffusion process, the model is trained over 1,000 diffusion steps. We incorporate a probability of 0.1 for condition masking to facilitate classifier-free guidance learning. During training, dropout is set at 0.1 to prevent overfitting, and all networks in the architecture follow an 8-layer Transformer design. For motion representation, we follow the setting in Guo et al. [2022a].

In the inference stage, all steps of the denoising sampling are set as 10 consistently. For the motion erasing application, we set the erasing weight as 0.1 by default. MotionCLR supports both DDIMsampling [Song et al., 2021] and DPM-soler-sampling [Lu et al., 2022] methods, with 1,000 as full diffusion steps. For the in-placement motion replacement and the motion style transfer application, as the motion semantics mainly depend on the initial denoising steps, we set the manipulating steps until 5 as default. For the example-based motion generation, the minimum manipulating time of a motion zone is 1s (i.e. chunk size=20 for the 20 FPS setting). At each step, all attention maps at all layers will be manipulated at 1 ∼ 9 denoising timestep. For editing the ground truth motions, we directly use the DDIM inversion Song et al. [2021] to edit the motion. Users can adjust the parameters freely to achieve interactive motion generation and editing (more details of user interface in Sec. 7).

#### 5.2 Motion Generation Evaluation for MotionCLR

The implementation details of the MotionCLR are in Sec. 5.1. We first evaluate the generation performance of the MotionCLR on HumanML3D Guo et al. [2022a]. We extend the evaluation metrics of previous works [Guo et al., 2022a], including FID, R-Precision, MM-Dist, and, MuitiModality. The results are shown in Tab. 1, indicating a comparable performance with the state-ofthe-art method. Especially, our result has a higher text-motion alignment over baselines, owing to the explicit fine-grained cross-modality modeling. As shown in Tab. 1, both DDIM and DPMsolver sampling work consistently well compared with baselines. We leave more visualization and qualitative results in Appendix A.

#### 5.3 Constructing Evaluation Set for Motion Editing

To evaluate the semantic correspondence between cross-attention and the generated motion, we construct an evaluation set to verify the observation qualitatively. We label verbs in HumanML3D texts in each sentence. Here, only a single verb in the sentence will be labeled and the sentences without any verbs will be filtered out. As a result, we construct 19,492 texts for evaluation, namely the HVerb test set. For the convenience of validating the in-placement replacement task, we also assign a new verb as the replacing verb of the original verb in the HVerb test set. To additionally

- Table 2: IoU (%) metrics on different settings. High coherence among E1, E2, and E3 shows the fine-grained text-motion modeling in cross attention.

Experiment ID E1 E2 E3 E4 E5

HVerb 74.3% 72.9% 77.8% 17.5% 17.0% Experiment ID E1 E2 E3 E4 E5

HVerb-wild 73.5% 71.4% 74.5% 18.5% 19.4%

test the explainability of the model in-the-wild, we construct 200 text prompts with verb labels via GPT-4o Achiam et al. [2023], namely HVerb-wild, whose results are checked by researchers.

- 5.4 Quantitative Results for Correspondence between Cross Attention and Generated Motion

Before evaluating the motion editing result, we initially validate the word-motion correspondence in the cross-attention map of MotionCLR. We set up 5 groups of experiments for comparison. (1) E1: IoU between cross attention & root velocity. We treat the value in the cross-attention map larger than 80% of the maximum value as an activated action. Accordingly, we treat the activated and unactivated parts as 1 and 0 respectively. Similarly, we also apply this to the root velocity to get another activation map. The IoU metric between these two activation maps means the coherence between the attention activation and the kinematic features. (2) E2: IoU between the cross attention & moment retrieval value. We use the moment retrieval function of TMR to calculate the verb-motion similarity in 20-frame windows, obtaining the similarity between sub-motion and the verb along the whole motion sequence. Accordingly, we can also calculate a metric between attention and the moment retrieval value (also set 60% as the threshold). (3) E3: IoU between the root velocity & moment retrieval value. This setting is similar to that in E2 and E3. Both root velocity and moment retrieval value are explicit kinematic metrics used to describe action execution. The motivation to set E3 is for setting a comparison group to verify that the cross-attention correspondence is coherent with these two metrics. (4): E4: neg. cross-attention & root velocity. We modify the setting in E1 and replace the cross-attention map with a randomly sampled cross-attention map in the test in calculating IoU with the root velocity. (5): E5: neg. cross-attention & moment-retrieval-value. Similarly, we modify the setting in E2 and replace the cross-attention map with a randomly sampled cross-attention map in the test in calculating IoU with moment-retrieval-value.

As can be seen in Tab. 2, compared with E3 (both explicit action indicators: speed, moment retrieval of TMR), the values in E1/E2 are similar to those in E3. Therefore, the cross-attention activation is well aligned with motion execution. Besides, as shown in random sampling comparison groups (E1 v.s. E4/E5), the cross attention is aligned with the action execution in the motion in the generation process of each motion. The result proves the word-motion correspondence in MotionCLR.

- 5.5 Evaluation on Inference-only Motion Editing

- 5.5.1 In-place motion replacement.

Different from naïve replacing prompts for motion replacement, in-place motion replacement not only needs to replace the original motion at the semantic level, but also needs to replace motions at the exact temporal place. Fig. 5a and Fig. 5b show the root height trajectory and the root horizontal velocity, respectively. In this case, the edited and original motion share the same time zone to execute the action. Besides, the edited motion is semantically aligned with the word “walk”. Fig. 5c also shows results of replacing “runs” as“jumps” without changing the sitting action. These quantitative results show the effectiveness of the in-place motion replacement application.

For the in-place motion replacement application, we additionally provide replacement results in Fig. 6. As shown in Fig. 6, MotionCLR changes the original “walk” motion into “jumps” and “dances” at the corresponding place, serving as a diverse semantic editing function. All edited motion with different semantics share the same action temporal location with the original motion. This function shows the robustness of the method for diverse editing requirements.

We also test our method on the HVerb and the HVerb-wild test sets quantitatively, which are shown in Tab. 3. We take the latest motion editing method, MotionFix, as a baseline, whose prompts are set as ‘Replace A as B” or “Change A with B”. For example, the prompt can be “Replace jump as walk”. Two examples are set to enhance prompt diversity of the baseline. As the editing process

- Table 3: In-place motion replacement. ∗The “R” and “C” settings represent “Replace A as B” and “Change A with B” prompts of MotionFix. The light gray text is the comparison group, denoting the upper bound of the performance. The bold numbers are the best results excepting the comparison group. The significant metric margin over baselines shows the good performance of the method, even some methods requiring specific training.

|HVerb|trainingfree<br><br>align with original text<br><br>(%) ↓<br><br>align with edited text<br><br>(%) ↑<br><br>unedited part preserving<br><br>(mm) ↓|
|---|---|
|w/o editing editing text only MotionFix (“R”) MotionFix (“C”) Ours<br><br>|- 76.205 64.687 0.0<br><br>62.324 68.368 201.5<br><br>✗ 74.526 63.542 130.2<br>✗ 75.112 63.780 142.1<br><br><br>63.324 68.125 57.9<br>|

|HVerb-wild<br><br>|trainingfree<br><br>align with original text<br><br>(%) ↓<br><br>align with edited text<br><br>(%) ↑<br><br>unedited part preserving<br><br>(mm) ↓|
|---|---|
|w/o editing editing text only MotionFix (“R”) MotionFix (“C”) Ours<br><br>|- 73.678 59.229 0.0 56.235 66.701 235.0<br><br>✗ 70.125 58.560 151.1<br>✗ 71.588 60.009 149.0 58.124 65.976 59.8<br>|

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

(a) Root height visualization and the motion of “a man jumps.” before applying editing.

(b) Motion visualization and the root velocity of editing the prompt “a man jumps.”→“a man walks.”.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

a man runs and then sits. a man jumps and then sits.

(c) Motion in-place replacement results of a motion including multiple actions. In this case, we replace the “jumps” in “a man jumps.” as “walks” without editing the sitting action. The edited motion and the vanilla motion share the same temporal area of the action execution.

- Figure 5: In-place motion replacement. (a) and (b) are a pair of motions before and after editing. (c) is a comparison of original and edited motions.

of COMO Huang et al. [2025] and Goel et al. [2024] are not fully open-sourced before writing this paper, we do not take these as our baselines. For the application of in-place motion replacement, the editing goal comes up with two aspects. (1) The semantics of the edited motion should be aligned with the replaced text. Here, we take the TMR Petrovich et al. [2023] similarity (0%∼100%) to evaluate the text-motion similarity. As shown in Tab. 3, the former two columns indicate that the edited motion is more semantically aligned with the edited text, and less aligned with the original text. (2) The unedited part of the motion should be reserved. We use the TMR moment retrieval function and the annotated verb to filter out the editing area of the motion, similar to Sec. 5.4. For the filtered motion part, we take the MPJPE (mm) metric to evaluate the motion-preserving ability of the unedited motion. As can be seen in the last column of Tab. 3, MotionCLR shows a stable motion-preserving ability on unedited areas. As the dataset construction process of MotionFix is based on a similar motion retrieval process, the semantic changes in the editing process are unsatisfactory.

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

a man walks

[Figure 95]

[Figure 96]

[Figure 97]

a man walks

[Figure 98]

[Figure 99]

a man jumps

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

a man walks

[Figure 108]

[Figure 109]

a man dances

- Figure 6: Results of editing multiple semantics from the same motion. All edited motions with different semantics share the same temporal location as the original motion.

[Figure 110]

[Figure 111]

↑0.50

[Figure 112]

↑0.15

[Figure 113]

vinilla

[Figure 114]

↓0.05

[Figure 115]

↓0.10

(a) The height of the character’s root. The highlighted area is obvious when comparing different weights.

(b) Motion visualization of the edited motions on different (de-)emphasis weight settings. .

- Figure 7: Motion (de-)emphasis. Different weights of “jump” (↑ or ↓) in “a man jumps.”.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- (a) “a man jumps.” case. The second jumping action is erased.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- (b) “waving hand” case. The final 1/3 waving action is removed.

- Figure 8: Motion erasing results. Case study of “a man jumps.” and “waving hand” cases.

#### 5.5.2 Motion (de-)emphasis and motion erasing.

We mainly provide the visualization results of motion (de-)emphasis in Fig. 7. As shown in Fig. 7, the edited results are aligned with the manipulated attention weights. Especially, as can be seen, in Fig. 7a, the height of the “jump” action is accurately controlled by the cross-attention weight of the word “jump”. For an extremely large adjusting weight, e.g. ↑1.0, the times of the jumping action also increase. This is because the low-activated timesteps of the originally generated motion might have a larger cross-attention value to activate the “jump” action.

[Figure 124]

- Figure 9: The effect of varying w in classifier-free guidance on generated motions. While changing w influences the general alignment between the text “a man jumps.” and the generated motion, it does not provide precise control over finer details like jump height and frequency.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

(a) Prompt: “a person walks straight and then waves.” Original (blue) vs. shifted (red).

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

(b) Prompt: “a man gets up from the ground.” Original (blue) vs. shifted (red).

- Figure 10: Comparison between original and shifted motions. Time bars are shown in different colors. (a) The original figure raises hands after the walking action. The shifted one has the opposite sequentiality. (b) The squatting action is shifted to the end of the sequence, and the standing-by action is shifted to the beginning.

Additionally, we would like to discuss the difference between reweighting the cross-attention map and adjusting classifier-free guidance weights. For intuitive understanding, the classifier-free guidance weight also controls the semantics of the motion with the input text. However, as the classifier-free guidance mainly works for the semantic alignment between text and motion, it cannot control the weight of each word. We take the sentence “a man jumps.” as an example for a fair comparison, which is the case used in the main text (suggested to refer to Fig. 7a for comparison). As shown in Fig. 9, the generated motions with different w values illustrate that w cannot influences both the height and frequency of the jump. Nevertheless, the classifier-free guidance is limited in its ability to control more detailed aspects, such as the exact height and number of actions. Therefore, while w improves text-motion alignment, it cannot achieve fine-grained adjustments. As there is no benchmark for such an application, we quantitatively evaluate this in the user study part (Sec. 5.7).

As motion erasing is a special case of motion de-emphasis, we do not provide more quantitative on this application. We provide some visualization results in Fig. 8. As can be seen in Fig. 8a, the second jumping action is erased. Besides, the “waving hand” case shown in Fig. 8b shows that the final 1/3 waving action is also removed.

#### 5.5.3 Motion sequence shifting.

Here, we provide some comparisons between the original motion and the edited one. In Fig. 10, we take “ ” and “ ” to represent different time bars, whose orders represent the sequentially. As can be seen in Fig. 10a, the execution of waving hands is shifted to the beginning of the motion. Besides, as shown in Fig. 10b, the squatting action has been moved to the end of the motion. These results show that the editing of the self-attention map sequentiality has an explicit correspondence with the editing motion sequentially. More results are in Appendix A.3.

#### 5.5.4 Example-based motion generation.

The example-based motion generation [Li et al., 2023b] task has two basic requirements. (1) The first one is the generated motions should share similar motion textures [Li et al., 2002] with the example

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

(a) Examples (blue) and generated (red) motions.

(b) Root trajectory visualization.

- Figure 11: Diverse generated motions driven by the same example. Prompt: “a person steps sideways to the left and then sideways to the right.”. (a) The diverse generated motions driven by the same example motion share similar movement content. (b) The root trajectories of diverse motions are with similar global trajectories but not the same.

80 60 40 20 0 20 40 60

80

60

40

20

0

20

40

60

80

generated

example

- Figure 12: t-SNE visualization of different example-based generated results. Different colors imply different driven examples.

- Table 4: Comparison with baselines. Our method enjoys better editing quality and higher diversity than previous methods. The motion coverage is comparable with the state-of-the-art methods, although our method is not designed for the specific task.

specific FID ↓ Diversity ↑ Coverage (%) ↑

GANimator [2022] 0.505 2.012 84.1 GenMM [2023b] 0.514 2.478 97.5 MotionCLR ✗ 0.427 2.567 97.0

motion. We observe the high-dimensional structure of motions via dimensionality reduction. As the t-SNE visualization results shown in Fig. 12, generated motions driven by the same example are similar to the given example (in the same color). (2) Besides, different generated motions driven by the same example should be diverse. As shown in Fig. 11, these generated results are diverse not only in local motions (Fig. 11a) but also in the global trajectory (Fig. 11b). Furthermore, results in Fig. 11 also share the similar motion textures. We leave more visualization results in Appendix A.4.

Besides, we also compare our method with previous methods used for example-based motion generation on the HumanML3D test set. We use the FID and the diversity (Div.) to evaluate fidelity and generation diversity. Additionally, we also take the coverage (Cov.) metric (cf. Li et al. [2023b]) and inference time for comparison. In this comparison, we take the motion texture Li et al. [2002], GANimator Li et al. [2022], and GenMM Li et al. [2023b]. The comparison is shown in Tab. 4. As shown in Tab. 4, our method enjoys better editing quality and higher diversity than baselines. The coverage is comparable with the state-of-the-art methods, although our method is not designed for the specific task. The reason behind the competitive performance with baselines is that our model is pre-trained on a larger dataset than baselines, obtaining better human dynamic priors from the data distribution.

- Table 5: Comparison on FID and diversity values with manipulating self-attention in the motion space of the denoising process. As can be seen in the table, our method enjoys better editing quality and higher diversity than editing at each diffusion step.

FID ↓ Diversity ↑

Diff. manipulation 0.718 1.502 MotionCLR manipulation 0.427 2.567

style reference content reference transferred result

[Figure 139]

[Figure 140]

[Figure 141]

- (a) Style reference: “the person dances very happily”, content reference: “the man is walking”. The transferred result shows a figure walking in a back-and-forth happy pace.

style reference content reference transferred result

[Figure 142]

[Figure 143]

[Figure 144]

- (b) Style reference: “a man is doing hip-hop dance”, Content reference: “a person runs around a circle”. The stylized result shows a running motion with bent hands, shaking left and right.

- Figure 13: Motion style transfer results. The style reference, content references, and the transferred results are shown from left to right for each case.

- Table 6: Comparison of motion style transfer across baselines. The “Gen.” and “Inv.” settings represent the editing during generation and DDIM inversion Song et al. [2021] (following Raab et al. [2024a]) settings.

| |FID ↓<br><br>R Precision (Top 3) ↑<br><br>Texture Ref. Foot Contact Sim. ↑<br><br>Style Ref. Loc.Sim. ↑ AITS ↓|
|---|---|
|MoMo (Gen.) [2024a] MoMo (Inv.) [2024a] Ours (Gen.) Ours (Inv.)|2.33 0.439 0.816 0.972 0.544 2.50 0.490 0.793 0.856 0.691 0.65 0.749 0.877 0.989 0.345 0.69 0.784 0.851 0.914 0.432<br><br>|

As the diffusion denoising process can manipulate the motion directly in the denoising process, we provide a baseline for comparison with our motion shifting and example-based motion generation applications. Here, for convenience, we only take the example-based motion generation application as an example for discussion. In this section, we conduct a comparison between our proposed editing method and diffusion manipulation in the motion space, focusing on the FID and diversity metrics. The 200 samples used in this experiment were constructed by researchers. As depicted in Tab. 5, the “Diff. manipulation” serves for our comparison. Our method achieves an FID value of 0.427, indicating a relatively high generation quality, while the “Diff. manipulation” achieves a higher FID of 0.718, demonstrating worse fidelity. Conversely, in terms of diversity, the “MotionCLR manipulation” exhibits a higher diversity score of 2.567 compared to the 1.502 of the “Diff. manipulation.” These results verify our method is better than manipulating noisy motions in the denoising process. The main reason for the better quality and diversity mainly relies on the many times of manipulation of self-attention, but not the motion. Directly manipulating the motion results in some jitters, making more effort for models to smooth. Besides, the shuffling times of manipulating the self-attention maps are higher than the baseline, contributing to the better diversity.

Table 7: Ablation studies between different technical design choices.

R-Precision↑ FID↓ Top 1 Top 2 Top 3

Ablation

- (1) 0.512 0.705 0.792 0.544
- (2) 0.509 0.703 0.788 0.550

MotionCLR 0.544 0.732 0.831 0.269

- Table 8: The ablation study of manipulating different attention layers on the HVerb-wild test set. The “begin” and “end” represent the beginning and the final layer/step for manipulation. The bottom row denotes our design choice for motion editing.

editing steps editing layers

align with edited text

align with original text

FID↓

(%) ↓ begin end begin end

(%) ↑

1 9 8 11 0.339 63.5 62.5 1 9 5 14 0.335 65.7 59.2

- 3 7 1 18 0.399 64.7 59.3
- 4 6 1 18 0.455 62.8 59.0 1 9 1 18 0.330 66.0 58.1

- 5.5.5 Motion style transfer.

As shown in Fig. 13, in the MotionCLR framework, the style reference motion provides style and the content reference motion provides keys and values. As can be seen in Fig. 13, all edited results are well-stylized with style motions and keep the main movement content with the content reference. To qualitatively evaluate the transferred result, we compare our method with the latest method, MoMo Raab et al. [2024a]. We follow the evaluation protocol in Raab et al. [2024a], where the comparison results in Tab. 6 indicate the better transfer result than MoMo. The quality gain mainly comes from the better base model of MotionCLR and the disentangled modeling of self-attention and cross-attention. Besides, we also compare the speed (AIST metric following Chen et al. [2023]) with baselines, indicating better efficiency of MotionCLR.

- 5.6 Ablation Study

In this section, we provide ablation studies for the generation function and the editing functions, respectively.

Ablation study of the generation ability in MotionCLR. We provide some ablation studies on some technical designs in Tab. 7. (1) The setting w/o separate word modeling shows poorer qualitative results with the w/ separate word setting. The separate word-level cross-attention correspondence benefits better text-to-motion controlling, which is critical for motion fine-grained generation. (2) The setting of injecting text tokens before motion tokens performs worse than the MotionCLR. This validates the effectiveness of introducing the cross-attention for cross-modal correspondence. The ablation studies additionally verify the basic motivation of modeling word-level correspondence in MotionCLR.

Ablation study of editing different attention layers. To further explore the impact of attention manipulation, we conduct an ablation study by varying the layers in MotionCLR for manipulation, shown in Tab. 8. Without losing generalization, we test this on the in-place motion replacement application. The table lists the results for different ranges of manipulated attention layers. It can be observed that manipulating different attention layers influences the editing quality and the semantic similarity (TMR-sim.). In particular, manipulating the layers from 1 to 18 achieves the best semantic consistency, demonstrating the effectiveness of editing across multiple attention layers for maintaining semantic alignment in the edited motion. The less effectiveness of manipulating middle layers is mainly due to the fuzzy semantics present in the middle layers of the U-Net. As these layers capture more abstract with reduced temporal resolution, the precise details and localized information become less distinct. Consequently, manipulating these layers has a limited impact on the final output, as they contribute less directly to the fine-grained details of the task.

[Figure 145]

[Figure 146]

vinilla result

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

edited retuslt

- Figure 14: Comparison between w/ vs. w/o grounded motion generation settings. The root height and motion visualization of the textual prompt “a person jumps four times”.

- Table 9: User study results. The study evaluates the motion (de-)emphasis and in-place replacement quality, respectively. The results show that our method outperforms other baselines across fidelity, user requirement satisfaction, and content similarity with the unedited motion.

|(de-)emphasis|fidelity ↑ satisfactory ↑ similarity ↑|
|---|---|
|MotionFix Ours|3.51 3.80 4.07<br><br>4.02 3.95 4.25<br>|

|replacement<br><br>|fidelity ↑ satisfactory ↑ similarity ↑|
|---|---|
|editing text only MotionFix Ours<br><br>|3.04 3.55 3.01<br><br>3.10 2.78 3.22<br>4.05 4.26 3.99<br>|

Ablation study of editing at different diffusion steps. We conducted an ablation study to evaluate the impact of editing across different ranges of diffusion steps and compare various design choices (as shown in Table Tab. 8). Here, without losing generalization, we also test this on the in-place motion replacement application. The results show that editing across a broader range of diffusion steps (e.g., steps 1 to 9) achieves the best balance between semantic consistency with the edited text and the quality of the generated motion (FID). This is because the early steps focus on establishing the global semantic structure, while the later steps refine the fine-grained details. Covering the entire range effectively combines the strengths of both stages. In contrast, limiting edits to narrower ranges, such as the middle or late steps (e.g., steps 4 to 6), results in lower semantic alignment with the edited text, as these steps mainly refine details and have limited influence on the global structure. This highlights the importance of carefully designing the range of diffusion steps to achieve high-quality, semantically consistent motion editing.

#### 5.7 User Study

To additionally verify the effectiveness of the semantic editing result, we also compare our method with baselines via human evaluation. Here, for the semantic motion editing method, we mainly focus on the motion (de-)emphasis and the im-place motion replacement task, both of which are based on editing cross-attention.

Similar to Goel et al. [2024], we set up the user study for 30 participants to evaluate 10 groups of motion editing results. For motion emphasis and de-emphasis applications, we evaluate the result based on 1) the edited motion quality (a.k.a. fidelity), 2) whether the edited result satisfies the requirement, and 3) content similarity between edited and unedited motions. For the in-place motion replacement application, the evaluation is based on 1) the edited motion quality (fidelity), 2) whether the edited result satisfies the requirement, and 3) the motion reserving of the unedited part. We set the latest method MotionFix as our baseline, and include directly editing text as another baseline for the in-place motion replacement application. For MotionFix in motion (de-)emphasis, we annotate the prompts by human researchers, like using “jump higher” to emphasize “jump”. For the in-place replacement application, we keep the prompts usage in Tab. 3. As shown in Tab. 9, our method outperforms baselines across motion fidelity, user requirement satisfaction, and motion content preserving. These gains mainly come from our method not relying on any specifically designed data construction process, like motion retrieval in Athanasiou et al. [2024].

[Figure 152]

errorrate(%)

- Figure 15: Action counting error rate comparison. Root trajectory (Traj.) vs. attention map (Ours). “σ” is the smoothing parameter.

#### 5.8 Potential Action Counting Ability from Attention Map

As shown in Fig. 3, the number of executed actions in a generated motion sequence can be accurately calculated via the self-attention map. We directly detect the number of peaks in each row of the selfattention map and finally average this of each row. In the technical implementation, to avoid sudden peaks from being detected, we apply average downsampling and Gaussian smoothing (parameterized by standard deviation σ). We leave more technical details in Appendix G.

We construct a set of text prompts corresponding to different actions to perform the counting capability via the self-attention map. The counting number of actions is labeled by professional researchers. The details of the evaluation set are detailed in Appendix E.2. As the “walking” action is composed of sub-actions of two legs, the atomic unit of this action counting is set as 0.5. We compare our method to counting with the vertical root trajectory (Traj.) peak detection. As shown in Fig. 15, counting with the self-attention map mostly works better than counting with root trajectory. Both settings use Gaussian smoothing to blur some jitters. Our method does not require too much smoothing regularization due to the smoothness of the attention map, while counting with root trajectory needs this operation. This case study reveals the effectiveness of understanding the self-attention map in MotionCLR.

### 6 Case Study: Failure Cases Analysis and Correction

There are few generative methods that can escape the curse of hallucination. In this section, we will discuss some failure cases of our method and analyze how we can refine these results. The hallucination of counting is a notoriously tricky problem for generative models, attracting significant attention in the community and lacking a unified technical solution. Considering that this problem cannot be thoroughly resolved, we try to partially reveal this issue by additionally providing temporal grounds. For example, if the counting number of an action is not aligned with the textual prompt, we can correct this by specifying the temporal grounds of actions. Technically, the temporal mask can be treated as a sequence of weights to perform the motion emphasis and de-emphasis. Therefore, grounded motion generation can be easily achieved by adjusting the weights of words.

Specifically, we show some failure cases of our method. As shown in Fig. 14, the generated result of “a person jumps four times” fails to show four times of jumping actions, but seven times. To meet the requirement of counting numbers in the textual prompts, we additionally input a temporal mask, including four jumping timesteps, to provide temporal grounds. Technically, for the desired four jumping areas, we set α = 0.15 as emphasis. For the reserving area, we set α = −0.1 as deemphasis. From the root height visualization and the motion visualization, the times of the jumping action have been successfully corrected from seven to four. Therefore, our method is promising for grounded motion generation to reveal the hallucination of deep models. Moreover, other editing fashions are also potential ways to correct hallucinations of generated results. For example, the

[Figure 153]

[Figure 154]

(a) Web interface for MotionCLR. (b) Blender add-on for MotionCLR.

- Figure 16: Web interface and the Blender add-on of MotionCLR, supporting proposed applications. In these two interfaces, we take the motion (de-)emphasis application as an example. Both interfaces support adjusting the weight of parameters to refine the result, until reaching satisfaction.

motion sequence shifting and in-place motion replacement functions can be used for correcting sequential errors and semantic misalignments, respectively.

### 7 User Interface

MotionCLR is a simple yet effective framework that can be integrated into industrial tools. We develop a web interface of the MotionCLR model for interactive motion generation and editing. The web interface provides a quick view for new customers without installing any dependencies. For experienced users, like animators, we also support a Blender add-on of MotionCLR similar to the web demo, supporting the real creation loop of motion synthesis. Two interfaces are shown in Fig. 16a and Fig. 16b, respectively. In Fig. 16, both the web interface and add-on support all applications of MotionCLR. Besides, users are allowed to interactively adjust the weight of parameters to refine the result, until reaching their satisfaction.

### 8 Discussion and Conclusion

In this work, we carefully clarify what roles cross-attention plays in motion generation, enhancing the explainability of attention mechanisms in text-to-motion generation. Except for the comparable generative performance with SOTA methods, our proposed MotionCLR model supports diverse interactive motion generation and editing in one system for the first time. Importantly, in the editing process, MotionCLR supports editing a motion without any training, which even outperforms some methods requiring specific training.

To have a thorough exploration of the proposed method, we construct an evaluation set to validate the theoretical analysis of the attention mechanism. Based on these preliminary experiments and new understanding, we evaluate our applications with specific baselines. Our method even outperforms some specifically designed methods in some scenarios. Besides, we also explore the boundaries of our method in action counting and grounded motion generation. Considering real-world applications in the animation creation, we develop a web interface and a Blender add-on for users.

This work explores the new interaction fashion of motion synthesis. We are still facing some limitations, driving for developing future methods. 1) Different interaction fashions. This work introduces a new interaction fashion for motion generation, relying on multiple interactions with the machine. However, the language-based chatting interaction is also useful to users, which is leaving as our future work. Besides, MotionCLR can also serve as a data generator to synthesize a dataset for such applications. 2) Robustness of the method. As shown in Sec. 6, our model can also not escape the hallucination curse of generative models. Although the proposed grounded motion generation method can relieve the phenomena, it is still worth enhancing the capability of the base model. To this end, the grounded synthesis method could provide additional supervision in this process. We will reach this in the future. Besides, our method will also benefit from a larger dataset with good quality, which is already in the collecting process before submission.

### References

Kfir Aberman, Peizhuo Li, Dani Lischinski, Olga Sorkine-Hornung, Daniel Cohen-Or, and Baoquan Chen. Skeleton-aware networks for deep motion retargeting. ACM TOG, 39(4):62–1, 2020a.

Kfir Aberman, Yijia Weng, Dani Lischinski, Daniel Cohen-Or, and Baoquan Chen. Unpaired motion style transfer from video to animation. ACM TOG, 39(4):64–1, 2020b.

Abubakar Abid, Ali Abdalla, Ali Abid, Dawood Khan, Abdulrahman Alfozan, and James Zou. Gradio: Hassle-free sharing and testing of ml models in the wild. arXiv preprint arXiv:1906.02569, 2019.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Hyemin Ahn, Timothy Ha, Yunho Choi, Hwiyeon Yoo, and Songhwai Oh. Text2action: Generative adversarial synthesis from language to action. In ICRA, pages 5915–5920, 2018.

Chaitanya Ahuja and Louis-Philippe Morency. Language2pose: Natural language grounded pose forecasting. In 3DV, pages 719–728, 2019.

Tenglong Ao, Zeyi Zhang, and Libin Liu. Gesturediffuclip: Gesture diffusion model with clip latents. ACM TOG, 42(4):1–18, 2023.

Nikos Athanasiou, Mathis Petrovich, Michael J Black, and Gül Varol. Teach: Temporal action composition for 3d humans. In 3DV, pages 414–423, 2022.

Nikos Athanasiou, Mathis Petrovich, Michael J Black, and Gül Varol. Sinc: Spatial composition of 3d human motions for simultaneous action generation. In ICCV, pages 9984–9995, 2023.

Nikos Athanasiou, Alpár Ceske, Markos Diomataris, Michael J. Black, and Gül Varol. MotionFix: Text-driven 3d human motion editing. In SIGGRAPH Asia, 2024.

German Barquero, Sergio Escalera, and Cristina Palmero. Seamless human motion composition with blended positional encodings. In CVPR, pages 457–469, 2024.

Uttaran Bhattacharya, Nicholas Rewkowski, Abhishek Banerjee, Pooja Guhan, Aniket Bera, and Dinesh Manocha. Text2gestures: A transformer-based network for generating emotive body gestures for virtual agents. In VR, pages 1–10, 2021.

Zhongang Cai, Jianping Jiang, Zhongfei Qing, Xinying Guo, Mingyuan Zhang, Zhengyu Lin, Haiyi Mei, Chen Wei, Ruisi Wang, Wanqi Yin, et al. Digital life project: Autonomous 3d characters with social intelligence. In CVPR, pages 582–592, 2024.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 22560–22570, 2023.

Hila Chefer, Shir Gur, and Lior Wolf. Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers. In ICCV, pages 397–406, 2021a.

Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In CVPR, pages 782–791, 2021b.

Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In CVPR, pages 18000–18010, 2023.

Setareh Cohan, Guy Tevet, Daniele Reda, Xue Bin Peng, and Michiel van de Panne. Flexible motion in-betweening with diffusion models. In ACM SIGGRAPH, pages 1–9, 2024.

Peishan Cong, Ziyi WangZhiyang Dou, Yiming Ren, Wei Yin, Kai Cheng, Yujing Sun, Xiaoxiao Long, Xinge Zhu, and Yuexin Ma. Laserhuman: Language-guided scene-aware human motion generation in free environment. arXiv preprint arXiv:2403.13307, 2024.

Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion:

A framework for denoising-diffusion-based motion synthesis. In CVPR, pages 9760–9770, 2023. Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in

pretrained transformers. In ACL, pages 8493–8502, 2022. Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. ECCV, 2024. Christian Diller and Angela Dai. Cg-hoi: Contact-guided 3d human-object interaction generation. In CVPR, pages 19888–19901, 2024. Markos Diomataris, Nikos Athanasiou, Omid Taheri, Xi Wang, Otmar Hilliges, and Michael J Black. Wandr: Intention-guided human motion generation. In CVPR, pages 927–936, 2024.

Ke Fan, Junshu Tang, Weijian Cao, Ran Yi, Moran Li, Jingyu Gong, Jiangning Zhang, Yabiao Wang, Chengjie Wang, and Lizhuang Ma. Freemotion: A unified framework for number-free text-to-motion synthesis. ECCV, 2024.

Bin Feng, Tenglong Ao, Zequn Liu, Wei Ju, Libin Liu, and Ming Zhang. Robust dancer: Long-term 3d dance synthesis using unpaired data. arXiv preprint arXiv:2303.16856, 2023.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In EMNLP, pages 5484–5495, 2021.

Anindita Ghosh, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, and Philipp Slusallek. Remos: Reactive 3d motion synthesis for two-person interactions. ECCV, 2023.

Purvi Goel, Kuan-Chieh Wang, C Karen Liu, and Kayvon Fatahalian. Iterative motion editing with natural language. In ACM SIGGRAPH, pages 1–9, 2024.

Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. Tm2d: Bimodality driven 3d dance generation via music-text integration. In ICCV, pages 9942–9952, 2023.

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In CVPR, pages 5152–5161, 2022a.

Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In ECCV, pages 580–597, 2022b.

Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. In CVPR, pages 1900–1910, 2024a.

Chuan Guo, Yuxuan Mu, Xinxin Zuo, Peng Dai, Youliang Yan, Juwei Lu, and Li Cheng. Generative human motion stylization in latent space. ICLR, 2024b.

Xinying Guo, Mingyuan Zhang, Haozhe Xie, Chenyang Gu, and Ziwei Liu. Crowdmogen: Zero-shot text-driven collective motion generation. arXiv preprint arXiv:2407.06188, 2024c.

Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Yuxiao Chen, Di Liu 0003, Qilong Zhangli, et al. Improving tuning-free real image editing with proximal guidance. WACV, 2023.

Yaru Hao, Li Dong, Furu Wei, and Ke Xu. Self-attention attribution: Interpreting information interactions inside transformer. In AAAI, volume 35, pages 12963–12971, 2021.

Félix G Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion inbetweening. ACM TOG, 39(4):60–1, 2020.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. ICLR, 2023.

Daniel Holden, Jun Saito, and Taku Komura. A deep learning framework for character motion synthesis and editing. ACM TOG, 35(4):1–11, 2016.

Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. ACM SIGGRAPH, 2022.

Zhi Hou, Baosheng Yu, and Dacheng Tao. Compositional 3d human-object neural animation. arXiv preprint arXiv:2304.14070, 2023.

Yiming Huang, Weilin Wan, Yue Yang, Chris Callison-Burch, Mark Yatskar, and Lingjie Liu. Como: Controllable motion generation through language guided pose code editing. In ECCV, pages 180–196. Springer, 2025.

Deok-Kyeong Jang, Soomin Park, and Sung-Hee Lee. Motion puzzle: Arbitrary motion style transfer by body part. ACM TOG, 41(3):1–16, 2022.

Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. NeurIPS, 2024.

Nan Jiang, Tengyu Liu, Zhexuan Cao, Jieming Cui, Yixin Chen, He Wang, Yixin Zhu, and Siyuan Huang. Full-body articulated human-object interaction. ICCV, 3, 2022.

Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusionbased editing with 3 lines of code. In ICLR, 2024.

Roy Kapon, Guy Tevet, Daniel Cohen-Or, and Amit H Bermano. Mas: Multi-view ancestral sampling for 3d motion generation using 2d diffusion. In CVPR, pages 1965–1974, 2024.

Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang. Guided motion diffusion for controllable human motion synthesis. In CVPR, pages 2151–2162, 2023.

Korrawe Karunratanakul, Konpat Preechakul, Emre Aksan, Thabo Beeler, Supasorn Suwajanakorn, and Siyu Tang. Optimizing diffusion noise can serve as universal motion priors. In CVPR, pages 1334–1345, 2024.

Jihoon Kim, Jiseob Kim, and Sungjoon Choi. Flame: Free-form language-based motion synthesis & editing. In AAAI, volume 37, pages 8255–8263, 2023.

Nilesh Kulkarni, Davis Rempe, Kyle Genova, Abhijit Kundu, Justin Johnson, David Fouhey, and Leonidas Guibas. Nifty: Neural object interaction fields for guided human motion synthesis. In CVPR, pages 947–957, 2024.

Jehee Lee and Sung Yong Shin. A hierarchical approach to interactive motion editing for human-like figures. In ACM SIGGRAPH, pages 39–48, 1999.

Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM TOG, 42(6):1–11, 2023a.

Jiaman Li, Alexander Clegg, Roozbeh Mottaghi, Jiajun Wu, Xavier Puig, and C Karen Liu. Controllable human-object interaction synthesis. ECCV, 2024.

Peizhuo Li, Kfir Aberman, Zihan Zhang, Rana Hanocka, and Olga Sorkine-Hornung. Ganimator: Neural motion synthesis from a single sequence. ACM TOG, 41(4):138, 2022.

Weiyu Li, Xuelin Chen, Peizhuo Li, Olga Sorkine-Hornung, and Baoquan Chen. Example-based

motion synthesis via generative motion matching. ACM TOG, 42(4), 2023b. doi: 10.1145/3592395. Yan Li, Tianshu Wang, and Heung-Yeung Shum. Motion texture: a two-level statistical model for

character motion synthesis. In ACM SIGGRAPH, pages 465–472, 2002. Han Liang, Wenqian Zhang, Wenxuan Li, Jingyi Yu, and Lan Xu. Intergen: Diffusion-based multi-human motion generation under complex interactions. IJCV, pages 1–21, 2024. Xiao Lin and Mohamed R Amer. Human motion modeling using dvgans. arXiv preprint arXiv:1804.10652, 2018. Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, and Xin Tong. Plan, posture and go: Towards open-world text-to-motion generation. ECCV, 2024.

Libin Liu, KangKang Yin, Michiel Van de Panne, Tianjia Shao, and Weiwei Xu. Sampling-based contact-rich motion control. In ACM SIGGRAPH, pages 1–10, 2010.

Yunze Liu, Changxi Chen, and Li Yi. Interactive humanoid: Online full-body motion reaction synthesis with social affordance canonicalization and forecasting. arXiv preprint arXiv:2312.08983, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS, pages 5775–5787, 2022.

Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum. Humantomato: Text-aligned whole-body motion generation. ICML, 2024.

Jie Ma, Yalong Bai, Bineng Zhong, Wei Zhang, Ting Yao, and Tao Mei. Visualizing and understanding patch interactions in vision transformer. IEEE TNNLS, 2023.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. ICLR, 2024.

Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH, pages 1–11, 2023.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. IMLR, 12:2825–2830, 2011.

Xiaogang Peng, Yiming Xie, Zizhao Wu, Varun Jampani, Deqing Sun, and Huaizu Jiang. Hoi-diff: Text-driven synthesis of 3d human-object interactions using diffusion models. arXiv preprint arXiv:2312.06553, 2023.

Mathis Petrovich, Michael J Black, and Gül Varol. Temos: Generating diverse human motions from textual descriptions. In ECCV, pages 480–497, 2022.

Mathis Petrovich, Michael J Black, and Gül Varol. Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis. In ICCV, pages 9488–9497, 2023.

Mathis Petrovich, Or Litany, Umar Iqbal, Michael J Black, Gul Varol, Xue Bin Peng, and Davis Rempe. Multi-track timeline control for text-driven 3d human motion generation. In CVPRW, pages 1911–1921, 2024.

Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. Mmm: Generative masked motion model. In CVPR, pages 1546–1555, 2024.

Matthias Plappert, Christian Mandery, and Tamim Asfour. The kit motion-language dataset. Big data, 4(4):236–252, 2016.

Matthias Plappert, Christian Mandery, and Tamim Asfour. Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks. RAS, 109: 13–26, 2018.

Sigal Raab, Inbal Leibovitch, Peizhuo Li, Kfir Aberman, Olga Sorkine-Hornung, and Daniel CohenOr. Modi: Unconditional motion synthesis from diverse data. In CVPR, pages 13873–13883, 2023.

Sigal Raab, Inbar Gat, Nathan Sala, Guy Tevet, Rotem Shalev-Arkushin, Ohad Fried, Amit H Bermano, and Daniel Cohen-Or. Monkey see, monkey do: Harnessing self-attention in motion diffusion for zero-shot motion transfer. ACM SIGGRAPH Asia, 2024a.

Sigal Raab, Inbal Leibovitch, Guy Tevet, Moab Arar, Amit Haim Bermano, and Daniel Cohen-Or. Single motion diffusion. In ICLR, 2024b.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015.

Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. In ICLR, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.

Xiangjun Tang, He Wang, Bo Hu, Xu Gong, Ruifan Yi, Qilong Kou, and Xiaogang Jin. Real-time controllable motion transition for characters. ACM TOG, 41(4):1–10, 2022.

Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified

physics-based character control through masked motion inpainting. ACM SIGGRAPH AISA, 2024. Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, and Daniel Cohen-Or. Motionclip: Exposing

human motion generation to clip space. In ECCV, pages 358–374, 2022a. Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. In ICLR, 2022b. Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, pages 1921–1930, 2023. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, and Lingjie Liu. Tlcontrol: Trajectory and language control for human motion synthesis. ECCV, 2024. Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, Wei Liang, and Siyuan Huang. Humanise: Languageconditioned human motion generation in 3d scenes. NeurIPS, pages 14959–14971, 2022.

Zan Wang, Yixin Chen, Baoxiong Jia, Puhao Li, Jinlu Zhang, Jingze Zhang, Tengyu Liu, Yixin Zhu, Wei Liang, and Siyuan Huang. Move as you say interact as you can: Language-guided human motion generation with scene affordance. In CVPR, pages 433–444, 2024.

Qianyang Wu, Ye Shi, Xiaoshui Huang, Jingyi Yu, Lan Xu, and Jingya Wang. Thor: Text to human-object interaction diffusion via relation intervention. arXiv preprint arXiv:2403.11208, 2024.

Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. In ICLR, 2024.

Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, pages 2048–2057. PMLR, 2015.

Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In ICCV, pages 14928–14940, 2023a.

Sirui Xu, Yu-Xiong Wang, and Liangyan Gui. Stochastic multi-person 3d motion forecasting. In ICLR, 2023b.

Heyuan Yao, Zhenhua Song, Baoquan Chen, and Libin Liu. Controlvae: Model-based learning of generative controllers for physics-based characters. ACM TOG, 41(6):1–16, 2022.

Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Moconvq: Unified physics-based motion control via scalable discrete representations. ACM TOG, 43(4):1–21, 2024.

Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In ICCV, pages 16010–16021, 2023.

Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In CVPR, pages 14730–14740, 2023a.

Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. In ICCV, 2023b.

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE TPAMI, 2024a.

Mingyuan Zhang, Daisheng Jin, Chenyang Gu, Fangzhou Hong, Zhongang Cai, Jingfang Huang, Chongzhi Zhang, Xinying Guo, Lei Yang, Ying He, et al. Large motion model for unified multi-modal motion generation. arXiv preprint arXiv:2404.01284, 2024b.

Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, and Ziwei Liu. Finemogen: Fine-grained spatio-temporal motion generation and editing. NeurIPS, 36, 2024c.

Siwei Zhang, Qianli Ma, Yan Zhang, Zhiyin Qian, Taein Kwon, Marc Pollefeys, Federica Bogo, and Siyu Tang. Egobody: Human body shape and motion of interacting people from head-mounted devices. In ECCV, pages 180–200. Springer, 2022.

Yan Zhang, Michael J Black, and Siyu Tang. Perpetual motion: Generating unbounded human motion. arXiv preprint arXiv:2007.13886, 2020.

Yuechen Zhang, Jinbo Xing, Eric Lo, and Jiaya Jia. Real-world image variation by aligning diffusion inversion chain. NeurIPS, 2023c.

Zeyi Zhang, Tenglong Ao, Yuyao Zhang, Qingzhe Gao, Chuan Lin, Baoquan Chen, and Libin Liu. Semantic gesticulator: Semantics-aware co-speech gesture synthesis. ACM TOG, 43(4):1–17, 2024d.

Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, and Siyu Tang. Synthesizing diverse human motions in 3d indoor scenes. In ICCV, pages 14738–14749, 2023.

Chongyang Zhong, Lei Hu, Zihao Zhang, and Shihong Xia. Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In ICCV, pages 509–519, 2023.

Lei Zhong, Yiming Xie, Varun Jampani, Deqing Sun, and Huaizu Jiang. Smoodi: Stylized motion diffusion model. ECCV, 2024.

Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. Emdm: Efficient motion diffusion model for fast, high-quality motion generation. ECCV, 2024.

Zixiang Zhou and Baoyuan Wang. Ude: A unified driving engine for human motion generation. In CVPR, pages 5632–5641, 2023.

Appendix

- A Supplemental Experiments

- A.1 What is the Self-attention Map like in Motion (De-)emphasis?

This experiment is an extension of the experiment shown in Fig. 7. We provide more examples of how increasing or decreasing weights impact motion (de-)emphasis and erasing. As seen in Fig. 17, the attention maps illustrate that reducing the weights (e.g., ↓ 0.05 and ↓ 0.10) results in less activations, while increasing weights (e.g., ↑ 0.33 and ↑ 1.00) leads to more activations. The vanilla map serves as a reference without any adjustments. However, as indicated, excessively high weights such as ↑ 1.00 introduce some artifacts, emphasis the need for careful tuning of weights to maintain the integrity of the generated motion outputs. This demonstrates the importance of careful weight tuning to achieve the desired motion emphasis or erasure. Compared to Fig. 17a, Fig. 17b shows two fewer trajectories. This reduction is due to the de-emphasis effect, where the character’s second jump was not fully executed, resulting in just an arm motion. Consequently, the two actions became distinguishable, leading to fewer detected two trajectories. In Fig. 17c, the second jumping has been completely erased, resulting in only one trajectory, further demonstrating how de-emphasis significantly affects motion execution.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(e) ↑0.33 (f) ↑1.00

(a) vinilla (b) ↓0.05 (c) ↓0.10

(d) vinilla

Figure 17: Additional visualization results for different (de-)emphasis weights. The self-attention maps show how varying the different weights (e.g., ↓ 0.05, ↓ 0.10, ↑ 0.33, and ↑ 1.00) affect the emphasis on motion.

- A.2 Motion Generation Result Visualization

We randomly chose some examples of the motion generation results in Fig. 18. The visualization results demonstrate that MotionCLR can generate coherent and realistic human motions based on diverse textual descriptions. The generated sequences capture various actions ranging from simple gestures to more complex movements, indicating the capability to handle a wide range of human behaviors. Overall, the qualitative results suggest that MotionCLR effectively translates textual

prompts into human-like motions with a clear understanding of texts. This demonstrates the potential for applications in scenarios requiring accurate motion generation based on natural language inputs.

#### A.3 More Visualization Results of Motion Sequence Shifting

We present further comparisons between the original and edited motions in Fig. 19. The time bars, indicated by “ ” and “ ,” represent distinct phases of the motion, with their sequential arrangement reflecting the progression of the motion over time.

In Fig. 19a, we observe that the action of crossing the obstacle, originally positioned earlier in the sequence, is shifted towards the end in the edited version. This adjustment demonstrates the model’s capacity to rearrange complex motions effectively while maintaining coherence. Similarly, Fig. 19b shows the standing-by action being relocated to the end of the motion sequence. This change emphasizes the model’s ability to handle significant alterations in the temporal arrangement of actions. These results collectively indicate that our editing process, driven by the attention map sequentiality, exhibits a high level of correspondence with the intended edits to the motion’s sequence. The model accurately captures and replicates the desired modifications, ensuring that the restructured motion retains a natural and logical flow, thereby validating the effectiveness of our motion editing approach.

#### A.4 More Visualization Results on Exampel-based Motion Generation

We provide some visualization results to further illustrate the effectiveness of our approach in generating diverse motions that adhere closely to the given prompts. In Fig. 20, the example motion of “a person kicking their feet” is taken as the reference, and multiple diverse kick motions are generated. These generated motions not only exhibit variety but also maintain key characteristics of the original example. Similarly, in Fig. 21, the example motion of “a person walking in a semi-circular shape while swinging arms slightly” demonstrates the capability to generate diverse walking motions that maintain the distinct features of the source motion. The generated trajectories, as visualized in Fig. 20b and Fig. 21b, show that the diverse motions follow different paths while retaining similarities with the original motion, confirming the effectiveness of our method.

#### A.5 Inversion v.s. Generation

Our method can directly adopt DDIM inversion to edit the existing GT motions. Here, we compare the generation results with DDIM output. As shown in Tab. 10, the inversion result of the generated motion is similar to the generated motion, supporting good performance of editing GT motions.

Table 10: Comparison between the inversed and generated motions.

R-Precision↑ FID↓ Top 1 Top 2 Top 3

Exp.

generated 0.544 0.732 0.831 0.269 generated + inversion 0.535 0.724 0.818 0.299

#### A.6 Results on Other Datasets

We further evaluate our method on the KIT-ML dataset Plappert et al. [2016]. As shown in Tab. 11, MotionCLR outperforms the state-ofthe-art MoMask in key metrics. These results demonstrate the robustness and generalizability of MotionCLR, with consistent performance trends across datasets of different sizes.

Table 11: Comparison of results on KIT-ML. MotionCLR demonstrates superior performance on key metrics.

Method Top 1 Top 2 Top 3 FID MM-Dist Multi-Modality

MoMask (SOTA) 0.433 0.656 0.781 0.204 2.779 1.131 MotionCLR 0.438 0.658 0.783 0.275 2.773 1.213

[Figure 161]

[Figure 162]

[Figure 163]

person takes one step backward to their left

a person walks forward, turns and then sits on a chair.

a person side steps to their left, before side stepping to their right.

[Figure 164]

[Figure 165]

[Figure 166]

a person waves, takes a step back a person holds their arms out in

person walks forwards slowly and normally without swinging arms

front of them, squats, then swivels from side to side

[Figure 167]

[Figure 168]

[Figure 169]

a person is waving with his hand. a man kicks with something or someone with his right leg.

a person takes a few steps, squats twice, then turns and walks back

[Figure 170]

[Figure 171]

[Figure 172]

someone seems to be looking for a balance not to fall

a person appears to be hitting a ball with their right hand.

a man walks clockwise in a circle

#### Figure 18: More visualization of human motion generation result by MotionCLR.

[Figure 173]

[Figure 174]

[Figure 175]

(a) Prompt: “ the person is walking forward on uneven terrain.” Original (blue) vs. shifted (red) motion.

[Figure 176]

[Figure 177]

[Figure 178]

(b) Prompt: “a person walks then jumps.” Original (blue) vs. shifted (red) motion.

- Figure 19: Comparison between original motion and the shifted motion. The shifted time bars are shown in different colors. (a) The original figure crosses the obstacle after the walking action. The shifted motion has the opposite sequentiality. (b) The key walking and jumping actions are shifted to the beginning of the sequence, and the standing-by action is shifted to the end.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

- (a) The example motion (blue) and the generated diverse motion (red).

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

- (b) The trajectory visualizations of the example motion and diverse motions.

- Figure 20: Diverse generated results of blue example generated by the prompt “a person kicks their feet.”. The example-based generated kick motions are diverse and similar to the source example.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- (a) The example motion (blue) and the generated diverse motion (red).

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- (b) The trajectory visualizations of the example motion and diverse motions.

Figure 21: Diverse generated results of blue example generated by the prompt “person walks in a semi-circular shape while swinging arms slightly.”. The examplebased generated walking motions are diverse and similar to the source walking example.

### B Web User Interface for Interactive Motion Generation and Editing

To have a better understanding of our task, we build a user interface with Gradio [Abid et al., 2019]. We introduce the demo as follows. In Fig. 31, we illustrate the steps involved in generating and visualizing motions using the interactive interface. Fig. 31a displays the initial step where the user provides input text such as “a man jumps” and adjusts motion parameters. Once the settings are finalized, the system begins processing the motion based on these inputs, as seen in the left panel. Fig. 31b showcases the generated motion based on the user’s input. The interface provides a rendered output of the skeleton performing the described motion. This presentation allows users to easily correlate the input parameters with the resulting animation. The generated motion can further be edited by adjusting parameters such as the length of the motion, emphasizing or de-emphasizing certain actions, or replacing actions altogether, depending on user requirements. This process demonstrates how the interface facilitates a workflow from input to motion visualization.

[Figure 199]

[Figure 200]

(a) Motion generation interface example. (b) Generated Motion Example

Figure 22: Motion generation and its output examples. The logical sequence of operations is as follows:

- 1. Input the text: Users start by entering text describing the motion (e.g., “a man jumps.”) or set the frames of motions to generate (as shown in Fig. 31a).
- 2. Generate the initial motion: The system generates the corresponding skeleton motion sequence based on the input text (as shown in Fig. 31b).
- 3. Motion editing: We show some downstream tasks of MotionCLR here.

- • Motion emphasizing/de-emphasizing: Users can select a specific word from the text (e.g., “jumps”) and adjust its emphasis using a weight slider (range [-1, 1]) (as seen in Fig. 32a). For example, setting the weight to 0.3 will either increase the jump motion’s intensity.
- • In-place replacement: If users want to change the action, they can select the “replace” option. For example, replacing “jumps” with “walks” will regenerate the motion, showing a comparison between the original and new edited motions (as shown in Fig. 32b).
- • Example-based motion generation: Users can generate motion sequences based on predefined examples by setting parameters like chunk size and diffusion steps. After specifying the number of motions to generate, the system will create multiple variations of the input motion, providing diverse options for further refinement (as illustrated in Fig. 32d). The progress bars of the process are visualized in Fig. 32c.

[Figure 201]

[Figure 202]

(a) Motion (de-)Emphasizing interface. (b) In-place replacement example.

[Figure 203]

[Figure 204]

(c) Example-based motion generation progress. (d) Example-based motion generation results.

Figure 23: Different interfaces and supporting functions for interactive motion editing.

### C Detailed Diagram of Attention Mechanisms

#### C.1 Mathematical Visualization of Self-attention Mechanism

- In the main text (Eq. (2)), we introduced the self-attention mechanism of MotionCLR, which utilizes different transformations of motion as inputs. The motion embeddings serve as both the query (Q), key (K), and value (V), capturing the internal relationships within the sequence of motion frames. Fig. 24 provides a detailed mathematical visualization of this process: (1) Similarity Calculation. In the first step, the similarity between the motion embeddings at different frames is computed using the dot product, represented by S = QK⊤. This measurement reflects the internal relationship/similarity between different motion frames within the sequence. Fig. 24a illustrates how the softmax(·) operation is applied to the similarity matrix to determine which motion feature should be selected at a given frame f. (2) Feature Updating. Next, the similarity scores are used to weight the motion embeddings (V) and generate updated features X′, as shown by the equation

X′ = softmax(QK⊤/

√

d)V. Here, the similarity matrix applies its selection of values (V) to update the motion features. This process allows the self-attention mechanism to dynamically adjust the representation of each motion frame based on its relevance to other frames in the sequence.

In summary, the self-attention mechanism aims to identify and emphasize the most relevant motion frames in the sequence, updating the features to enhance their representational capacity for downstream tasks. The most essential capability of cross-attention is to order the motion features.

C.2 Mathematical Visualization of Cross-attention Mechanism

- In the main text (Eq. (3)), we introduced the cross-attention mechanism of MotionCLR, which utilizes the transformation of motion as a query (Q) and the transformation of text as a key (K) and value (V) to explicitly model the correspondence between motion frames and words. Fig. 25 provides a detailed mathematical visualization of this process:

- (1) Similarity Calculation. In the first step, the similarity between the motion embeddings (Q) with F frames and the text embeddings (K) with N words is computed through the dot product, represented by S = QK⊤. This similarity measurement reflects the relationship between motion frames and words. Fig. 25a shows how the softmax(·) operation is applied to the similarity matrix to determine which word should be activated at a given frame f.
- (2) Feature Updating. Next, the similarity scores are used to weight the text embeddings (V) and

√

generate updated features X′, as shown by the equation X′ = softmax(QK⊤/

d)V. Here, the similarity matrix applies its selection of values (V) to update the features. This process establishes an explicit correspondence between the frames and specific words.

In summary, the similarity calculation process determines which frame(s) should be selected, and the feature updating process (multiplication with V) is the execution of the frame(s) placement.

#### C.3 The Basic Difference with Previous Diffusion-based Motion Generation Models in Cross-modal Modeling

As discussed in the main text (see Sec. 1), despite the progresses in human motion generation [Zhang et al., 2024c, Cai et al., 2024, Zhang et al., 2024b, Guo et al., 2024c, Raab et al., 2024b, Kapon et al., 2024, Cohan et al., 2024, Fan et al., 2024, Xu et al., 2023a,b, Yao et al., 2022, Feng et al.,

- 2023, Ao et al., 2023, Yao et al., 2024, Zhang et al., 2024d, Liu et al., 2010, Aberman et al., 2020a, Karunratanakul et al., 2024, Li et al., 2024, 2023a, Gong et al., 2023, Zhou and Wang, 2023, Zhong et al., 2023, Athanasiou et al., 2023, Zhong et al., 2024, Guo et al., 2024b, Zhang et al., 2024b, Zhao et al., 2023, Zhang et al., 2022, 2020, Diomataris et al., 2024, Pinyoanuntapong et al., 2024, Diller and Dai, 2024, Peng et al., 2023, Hou et al., 2023, Liu et al., 2023, Cong et al., 2024, Jiang et al., 2022, Kulkarni et al., 2024, Tessler et al., 2024, Liang et al., 2024, Ghosh et al., 2023, Wu et al.,
- 2024], there still lacks a explicit modeling of word-level cross-modal correspondence in previous work. To clarify this, our method models a fine-grained word-level cross-modal correspondence.

As illustrated in Fig. 26, the major distinction between our proposed method and previous diffusionbased motion generation models lies in the explicit modeling of word-level cross-modal correspondence. In the MDM-like fashion Tevet et al. [2022b] (see Fig. 26a), previous methods usually utilize

a denoising transformer encoder that treats the entire text as a single embedding, mixing it with the motion sequence. This approach lacks the ability to capture the nuanced relationship between individual words and corresponding motion elements, resulting in an over-compressed representation. Although we witness that Zhang et al. [2024a] also introduces cross-attention in the motion generation process, it still faces two problems in restricting the fine-grained motion editing applications. First of all, the text embeddings are mixed with frame embeddings of diffusion, resulting in a loss of detailed semantic control. Our approach disentangles the diffusion timestep injection process in the convolution module to resolve this issue. Besides, the linear cross-attention in MotionDiffuse is different from the computation process of cross-attention, resulting in a lack of explanation of the word-level correspondence. The auto-regressive (AR) fashion [Zhang et al., 2023a] (Fig. 26b) adopts a simple concatenation of text and motion, where an AR transformer processes them together. However, this fashion also fails to explicitly establish a fine-grained correspondence between text and motion, as the AR transformer merely regards the text and motion embeddings as one unified sequence.

Our approach (shown in Fig. 26c) introduces a cross-attention mechanism that explicitly captures the word-level correspondence between the input text and generated motion sequences. This allows our model to maintain a clear and interpretable mapping between specific words and corresponding motion patterns, significantly improving the quality and alignment of generated motions with the textual descriptions. By integrating such a word-level cross-modal modeling technique, our method not only achieves more accurate and realistic motion generation but also supports fine-grained wordlevel motion editing. This capability enables users to make precise adjustments to specific parts of the generated motion based on textual prompts, addressing the critical limitations present in previous diffusion-based motion generation models and allowing for more controllable and interpretable editing at the word level.

At timestep f, which motion feature should be selected?

At timestep f, which motion feature should be selected?

| | | |
|---|---|---|

| | | | | | |
|---|---|---|---|---|---|

soft max

| | | | | | |
|---|---|---|---|---|---|

motion embs. (F x d), F=6

soft max

F

motion embs. (F x d), F=6 motion embs. (F x d), F=6

d

F

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

F

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

F

F

F

d F

softmax

QKT

d

V (motions)

X’

Q (motions) KT (motions) similarity

similarity

√

(a) S = QK⊤.

(b) X′ = softmax(S/

d)V.

- Figure 24: Mathematical Visualization of Self-attention Mechanism. This figure takes F = 6 as an example. (a) The similarity calculation with queries and keys (different frames). (b) The similarity matrix picks “value”s of the attention mechanism and updates motion features.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

motion embs. (F x d), F=6 word embs. (L x d), L=5

Q (motions) KT (text) similarity

F

L

d

d F

L

softmax

soft max

At timestep f, which word should be activated?

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|

(a) S = QK⊤.

similarity

soft max

At timestep f, which word should be activated?

QKT

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

word embs. (L x d), L=5

V (text)

L

d

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|

X’

F

L

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|

(b) X′ = softmax(S/

√

d)V.

- Figure 25: Mathematical Visualization of Cross-attention Mechanism. This figure takes F = 6 and N = 5 as an example. (a) The similarity calculation with queries and keys. (b) The similarity matrix picks “value”s of the attention mechanism and updates features.

AR Transformer

text

EOS

Denoising Transformer encoder

text

 −1

text

cross-attention

a

man

walks

to

right

(a) MDM-like fashion (b) Auto-regressive fashion (c) Word-level correspondence

 −1

- Figure 26: Comparison with previous diffusion-based motion generation models. (a) MDM-like fashion: Tevet et al. [2022b] and its follow-up methods treat text embeddings as a whole and mix them with motion representations using a denoising Transformer. (b) Auto-regressive fashion: Zhang et al. [2023a] and its follow-up methods concatenate the text with the motion sequence and feed them into an auto-regressive transformer without explicit correspondence modeling. (c) Our proposed method establishes fine-grained word-level correspondence using cross-attention mechanisms.

### D More Visualization Results of Empirical Evidence

In the main text, we introduced the foundational understanding of both cross-attention and selfattention mechanisms, emphasizing their ability to capture temporal relationships and dependencies across motion sequences. As a supplement, we provide a new, more detailed example here. As shown in Fig. 27, this visualization illustrates how different attention mechanisms respond to a complex sequence involving both walking and jumping actions. Specifically, we use green dashed boxes to highlight the “walk” phases and red dashed boxes to indicate the “jump” phases. This

allows us to clearly distinguish the temporal patterns associated with each action. Besides, we observed that the word “jump” reaches its highest activation during the crouching phase, which likely correlates with this moment being both the start of the jumping action and the “power accumulation phase”. This suggests that the attention mechanism accurately captures the preparatory stage of the movement, highlighting its capability to recognize the nuances of motion initiation within complex sequences. The cross-attention map effectively aligns key action words like “walk” and “jump” with their respective motion segments, while the self-attention map reveals repeated motion patterns and similarities between the walking and jumping cycles.

[Figure 205]

[Figure 206]

[Figure 207]

- (a) Horizontal distance
- (b) Vertical height

- (d) Self-attention map
- (e) Motion Visualization

[Figure 208]

[Figure 209]

(c) Cross-attention map

- Figure 27: Empirical study of attention patterns. We use the example “a person walks stop and then jumps.” (a) Horizontal distance traveled by the person over time, highlighting distinct walking and jumping phases. (b) The vertical height changes of the person, indicating variations during walking and jumping actions.

- (c) The cross-attention map between timesteps and the described actions. Notice that “walk” and “jump” receive a stronger attention signal corresponding to the walk and jump segments. (d) The self-attention map, which clearly identifies repeated walking and jumping cycles, shows similar patterns in the sub-actions. (e) Visualization of the motion sequences, demonstrating the walking and jumping actions.

Continuing with another case study, in Fig. 28, we examine how attention mechanisms respond to a sequence that primarily involves walking actions with varying intensity. In this instance, we observe that both the horizontal distance (Fig. 28a) and vertical height (Fig. 28b) reflect the man walks straight. The cross-attention map (Fig. 28c) reveals how the word “walks” related to walking maintains consistent activation, indicating that MotionCLR has a word-level understanding throughout the sequence. The self-attention map (Fig. 28d) further emphasizes repeated walking patterns, demonstrating that the mechanism effectively identifies the temporal consistency and structure of the walking phases. The motion visualization (Fig. 28e) reinforces this finding, showing a clear, uninterrupted walking motion.

More importantly, we can observe that the walking action consists of a total of five steps: three steps with the right foot and two with the left foot. The self-attention map (Fig. 28d) clearly reveals that steps taken by the same foot exhibit similar patterns, while movements between different feet show distinct differences. This observation indicates that the self-attention mechanism effectively captures the subtle variations between repetitive motions, further demonstrating its sensitivity to nuanced motion capture capability within the sequence.

Besides, different from the jumping, the highlights in the self-attention map of the walking are rectangular. The reason is that the local movements of walking are similar. In contrast, the jumping

includes several sub-actions, resulting in the highlighted areas in the self-attention maps being elongated.

[Figure 210]

[Figure 211]

- (a) Horizontal distance
- (b) Vertical height

[Figure 212]

(d) Self-attention map

[Figure 213]

[Figure 214]

(c) Cross-attention map (e) Motion Visualization

- Figure 28: Empirical study of attention patterns in a consistent walking sequence. We use the example: “a man walks.”. (a) The horizontal distance traveled over time reflects a steady walking motion. (b) Vertical height changes indicate minimal variation, characteristic of walking actions. (c) The cross-attention map shows that the “walks” word maintains consistent activation. (d) The self-attention map highlights the repeated walking cycles, capturing the temporal stability. (e) Visualization of the motion sequence.

### E Implementation and Evaluation Details

- E.1 Compared Baselines

Here, we introduce details of baselines in Tab. 1 for our comparison.

MDM [Tevet et al., 2022b] uses a diffusion-based approach with a transformer-based design for generating human motions. It excels at handling various generation tasks, achieving satisfying results in text-to-motion tasks.

MLD [Chen et al., 2023] uses a diffusion process on motion latent space for conditional human motion generation. By employing a Variational AutoEncoder (VAE), it efficiently generates vivid motion sequences while reducing computational overhead.

MotionDiffuse [Zhang et al., 2024a] is a diffusion model-based text-driven framework for motion generation. It provides diverse and fine-grained human motions, supporting probabilistic mapping and multi-level manipulation based on text prompts.

ReMoDiffuse [Zhang et al., 2023b] integrates retrieval mechanisms into a diffusion model for motion generation, enhancing diversity and consistency. It uses a Semantic-Modulated Transformer to incorporate retrieval knowledge, improving text-motion alignment.

MoMask [Guo et al., 2024a] introduces a masked modeling framework for 3D human motion generation using hierarchical quantization. It outperforms other methods in generating motions and is applicable to related tasks without further fine-tuning.

- E.2 Evaluation Details

Motion (de-)emphasis. To evaluate the effectiveness of motion (de-)emphasis application, we construct 200 prompts (a.k.a. HVerb-wild)to verify the algorithm. All of these prompts are constructed by researchers manually. We take some samples from our evaluation set as follows.

... ...

- 3 the figure leaps high
- 4 a man is waving hands

... ...

Each line in the examples represents the index of the edited word in the sentence, followed by the corresponding prompt. These indices indicate the key verbs or actions that are subject to the (de)emphasis during the evaluation process. The prompts were carefully selected to cover a diverse range of actions, ensuring that our method is tested on different types of motion descriptions. For instance, in the prompt “3 the figure leaps high”, the number 3 indicates that the word “leaps” is the third word in the sentence and is the target action for (de-)emphasis. This format ensures a systematic evaluation of how the model responds to adjusting attention weights on specific actions across different prompts.

Example-based motion generation. To further evaluate our example-based motion generation algorithm, we randomly constructed 7 test prompts. We used t-SNE [Pedregosa et al., 2011] visualization to analyze how closely the generated motions resemble the provided examples in terms of motion textures. For each case, the generated motion was assessed based on two criteria: (1) similarity to the example, and (2) diversity across different generated results from the same example.

Action counting. To thoroughly evaluate the effectiveness of our action counting method, we constructed a test set based on HVerb-wild. These prompts were manually designed by researchers to ensure diversity. Each prompt corresponds to a motion sequence generated by our model, and the ground truth action counts were labeled by researchers based on the observable actions within the generated motions.

#### E.3 User Study Details

To evaluate the effectiveness of our editing methods, we conducted a user study with participants rating the quality of edited motion results. Fig. 29 illustrates an example of the interface and questions used during the study (replacement case for example here). Participants were asked to compare the

original motion (leftmost column) with the results of 3 editing methods and rate them across three metrics (mentioned in the main text). This study provides insight into the subjective perception of motion quality and highlights the differences between various editing approaches.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

###### Request: Changing the “jumping” action to “bending” in-place, without editing the “running” action.

Unedited Motion A B C “A man runs forward and then performs a jumping action.”

edited motion edited motion edited motion

Please rate the results of the first action editing. The far left side represents the original action, and A and B represent the results of two editing algorithms.

- 1. Please rate the naturalness of the action results for the two edits. (1–5 points)
- 2. Please rate how well the results of the two edits match the editing expectations. (1–5 points)
- 3. Please rate the degree to which the results of the two edits preserve the content of the original action. (1–5 points)

[Figure 219]

[Figure 220]

[Figure 221]

Figure 29: An example interface from the user study.

- F Details of Motion Editing In this section, we will provide more technical details about the motion editing algorithms.

#### F.1 Pseudo Codes of Motion Editing

Motion (de-)emphasizing. Motion (de-)emphasizing mainly manipulate the cross-attention weights of the attention map. Key codes are shown in the L16-18 of Code 1.

- 1 def forward(self , x, cond , reweighting_attn , idxs):

- 2 B, T, D = x.shape

- 3 N = cond.shape[1]

- 4 H = self.num_head

- 5

- 6 # B, T, 1, D

- 7 query = self.query(self.norm(x)).unsqueeze (2).view(B, T, H, -1)

- 8 # B, 1, N, D

- 9 key = self.key(self.text_norm(cond)).unsqueeze (1).view(B, N, H,

-1)

- 10

- 11 # B, T, N, H

- 12 attention = torch.einsum(’bnhd ,bmhd ->bnmh’, query , key) / math. sqrt(D // H)

- 13 weight = self.dropout(F.softmax(attention , dim=2))

- 14

- 15 # reweighting attention for motion (de -)emphasizing

- 16 if reweighting_attn > 1e-5 or reweighting_attn < -1e-5:

- 17 for i in range(len(idxs)):

- 18 weight[i, :, 1 + idxs[i]] = weight[i, :, 1 + idxs[i]] + reweighting_attn

- 19

- 20 value = self.value(self.text_norm(cond)).view(B, N, H, -1)

- 21 y = torch.einsum(’bnmh ,bmhd ->bnhd’, weight , value).reshape(B, T, D )

- 22 return y

Code 1: Pseudo codes for motion (de-)emphasizing. In-place motion replacement. The generation of two motions (B=2) are reference and edited motions. As the cross-attention map determines when to execute the action. Therefore, replacing the cross-attention map directly is a straightforward way, which is shown in L16-17 of Code 2.

- 1 def forward(self , x, cond , manipulation_steps_end):

- 2 B, T, D = x.shape

- 3 N = cond.shape[1]

- 4 H = self.num_head

- 5

- 6 # B, T, 1, D

- 7 query = self.query(self.norm(x)).unsqueeze (2).view(B, T, H, -1)

- 8 # B, 1, N, D

- 9 key = self.key(self.text_norm(cond)).unsqueeze (1).view(B, N, H,

-1)

- 10

- 11 # B, T, N, H

- 12 attention = torch.einsum(’bnhd ,bmhd ->bnmh’, query , key) / math. sqrt(D // H)

- 13 weight = self.dropout(F.softmax(attention , dim=2))

- 14

- 15 # replacing the attention map directly

- 16 if self.step <= manipulation_steps_end:

- 17 weight[1, :, :, :] = weight[0, :, :, :]

- 18

- 19 value = self.value(self.text_norm(cond)).view(B, N, H, -1)

- 20 y = torch.einsum(’bnmh ,bmhd ->bnhd’, weight , value).reshape(B, T, D )

- 21 return y Code 2: Pseudo codes for in-place motion replacement.

Motion sequence shifting. Motion sequence shifting aims to correct the atomic motion in the temporal order you want. We only need to shift the temporal order of Qs, Ks, and Vs in the self-attention to obtain the shifted result. Key codes are shown in the L13-24 and L32-36 of Code 3.

- 1 def forward(self , x, cond , time_shift_steps_end , time_shift_ratio):

- 2 B, T, D = x.shape

- 3 H = self.num_head

- 4

- 5 # B, T, 1, D

- 6 query = self.query(self.norm(x)).unsqueeze (2)

- 7 # B, 1, T, D

- 8 key = self.key(self.norm(x)).unsqueeze (1)

- 9 query = query.view(B, T, H, -1)

- 10 key = key.view(B, N, H, -1)

- 11

- 12 # shifting queries and keys

- 13 if self.step <= time_shift_steps_end:

- 14 part1 = int(key.shape[1] * time_shift_ratio)

- 15 part2 = int(key.shape[1] * (1 - time_shift_ratio))

- 16 q_front_part = query[0, :part1 , :, :]

- 17 q_back_part = query[0, -part2:, :, :]

- 18 new_q = torch.cat((q_back_part , q_front_part), dim=0)

- 19 query[1] = new_q

- 20

- 21 k_front_part = key[0, :part1 , :, :]

- 22 k_back_part = key[0, -part2:, :, :]

- 23 new_k = torch.cat((k_back_part , k_front_part), dim=0)

- 24 key[1] = new_k

- 25

- 26 # B, T, N, H

- 27 attention = torch.einsum(’bnhd ,bmhd ->bnmh’, query , key) / math. sqrt(D // H)

- 28 weight = self.dropout(F.softmax(attention , dim=2))

- 29 value = self.value(self.text_norm(cond)).view(B, T, H, -1)

- 30

- 31 # shifting values

- 32 if self.step <= time_shift_steps_end:

- 33 v_front_part = value[0, :part1 , :, :]

- 34 v_back_part = value[0, -part2:, :, :]

- 35 new_v = torch.cat((v_back_part , v_front_part), dim=0)

- 36 value[1] = new_v

- 37 y = torch.einsum(’bnmh ,bmhd ->bnhd’, weight , value).reshape(B, T, D )

- 38 return y Code 3: Pseudo codes for motion sequence shifting.

Example-based motion generation. To generate diverse motions driven by the same example, we only need to shuffle the order of queries in self-attention, which is shown in L13-23 of Code 4.

- 1 def forward(self , x, cond , steps_end , _seed , chunk_size , seed_bar):

- 2 B, T, D = x.shape

- 3 H = self.num_head

- 4

- 5 # B, T, 1, D

- 6 query = self.query(self.norm(x)).unsqueeze (2)

- 7 # B, 1, T, D

- 8 key = self.key(self.norm(x)).unsqueeze (1)

- 9 query = query.view(B, T, H, -1)

- 10 key = key.view(B, N, H, -1)

- 11

- 12 # shuffling queries

- 13 if self.step == steps_end:

- 14 for id_ in range(query.shape[0]-1):

- 15 with torch.random.fork_rng():

- 16 torch.manual_seed(_seed)

- 17 tensor = query[0]

- 18 chunks = torch.split(tensor , chunk_size , dim=0)

- 19 shuffled_index = torch.randperm(len(chunks))

- 20 shuffled_chunks = [chunks[i] for i in shuffled_index]

- 21 shuffled_tensor = torch.cat(shuffled_chunks , dim=0)

- 22 query[1+id_] = shuffled_tensor

- 23 _seed += seed_bar

- 24

- 25 # B, T, T, H

- 26 attention = torch.einsum(’bnhd ,bmhd ->bnmh’, query , key) / math. sqrt(D // H)

- 27 weight = self.dropout(F.softmax(attention , dim=2))

- 28 value = self.value(self.text_norm(cond)).view(B, N, H, -1)

- 29 y = torch.einsum(’bnmh ,bmhd ->bnhd’, weight , value).reshape(B, T, D )

- 30 return y Code 4: Pseudo codes for example-based motion generation.

Motion style transfer. In the generation of two motions (B=2), we only need to replace the query of the second motion with the first one, which is shown in L13-14 of Code 5.

- 1 def forward(self , x, cond , steps_end):

- 2 B, T, D = x.shape

- 3 H = self.num_head

- 4

- 5 # B, T, 1, D

- 6 query = self.query(self.norm(x)).unsqueeze (2)

- 7 # B, 1, T, D

- 8 key = self.key(self.norm(x)).unsqueeze (1)

- 9 query = query.view(B, T, H, -1)

- 10 key = key.view(B, N, H, -1)

- 11

- 12 # style transfer

- 13 if self.step <= self.steps_end:

- 14 query[1] = query[0]

- 15

- 16 # B, T, T, H

- 17 attention = torch.einsum(’bnhd ,bmhd ->bnmh’, query , key) / math. sqrt(D // H)

- 18 weight = self.dropout(F.softmax(attention , dim=2))

- 19 value = self.value(self.text_norm(cond)).view(B, N, H, -1)

- 20 y = torch.einsum(’bnmh ,bmhd ->bnhd’, weight , value).reshape(B, T, D )

- 21 return y Code 5: Pseudo codes for motion style transfer.

Out

Out

Out

[Figure 222]

[Figure 223]

[Figure 224]

Map

Map

Map

[Figure 225]

[Figure 226]

[Figure 227]

K V

Q

Q K V

Q K V

(a) Direct generating style reference.

(b) Direct generating content reference.

(c) Generating transfered result.

Figure 30: The illustration of motion style transfer process. (a) Direct generating style reference: The style information is generated directly using the query (Q), key (K), and value (V) from the style reference motion sequence (blue). (b) Direct generating content reference: The content information is generated directly from the content reference motion sequence (orange). (c) Generating transferred result: The final transferred motion sequence combines the style from the style reference sequence with the content from the content reference sequence, using Q from the style reference (blue) and K, V from the content reference (orange).

#### F.2 Supplementary for Motion Style Transfer

As discussed in the main text, motion style transfer is accomplished by replacing the query (Q) from the content sequence (M2) with that from the style sequence (M1). This replacement ensures that while the content features from M2 are preserved, the style features from M1 are adopted, resulting in a synthesized motion sequence that captures the style of M1 with the content of M2. Fig. 30 provides a visual explanation of this process. The self-attention mechanism plays a crucial role, where the attention map determines the correspondence between the style and content features. The pseudo code snippet provided in Code 5 exemplifies this process. By setting “query[1] = query[0]” in the code, the query for the second motion (M2) is replaced by that of the first motion (M1), which effectively transfers the motion style from M2 to M1. In summary, this motion style transfer method allows one motion sequence to adopt the style characteristics of another while maintaining its content.

### G Details of Action Counting in a Motion

The detailed process of action counting is described in Code 6. The attention map is first smoothed using a Gaussian filter to eliminate noise, ensuring that minor fluctuations do not affect peak detection. We then downsample the smoothed matrix to reduce computational complexity and normalize it within a 0-1 range for consistent peak detection across different motions.

The pseudo code provided demonstrates the complete process, including peak detection using height and distance thresholds. The experimental results indicate that this approach is more reliable and less sensitive to noise compared to using the root trajectory, thus confirming the effectiveness of our method in accurately counting actions within a generated motion sequence.

- 1 """

- 2 Input: matrix (the attention map array with shape (T, T))

- 3 Output: float (counting number)

- 4 """

- 5

- 6 # Apply Gaussian smoothing via gaussian_filter in scipy.ndimage

- 7 smoothed_matrix = gaussian_filter(matrix , sigma =0.8)

- 8

- 9 # Attention map down -sampling

- 10 downsample_factor = 4

- 11 smoothed_matrix = downsample_matrix(smoothed_matrix , downsample_factor )

- 12

- 13 # Normalize the matrix to 0-1 range

- 14 normalized_matrix = normalize_matrix(smoothed_matrix)

- 15

- 16 # Detect peaks with specified height and distance thresholds

- 17 height_threshold = normalized_matrix.mean() * 3 # you can adjust this

- 18 distance_threshold = 1 # you can adjust this

- 19 peaks_positions_per_row = detect_peaks_in_matrix(normalized_matrix , height=height_threshold , distance=distance_threshold)

- 20

- 21 # Display the peaks positions per row

- 22 total_peak = sum([len(i) if len(i) > 0 else 0 for i in peaks_positions_per_row ])

- 23 sum_ = sum([1 if len(i) > 0 else 0 for i in peaks_positions_per_row ])

- 24

- 25 return total_peak / sum_ Code 6: Pseudo codes for action counting.

### H Web User Interface for Interactive Motion Generation and Editing

To have a better understanding of our task, we build a user interface with Gradio [Abid et al., 2019]. We introduce the demo as follows. In Fig. 31, we illustrate the steps involved in generating and visualizing motions using the interactive interface. Fig. 31a displays the initial step where the user provides input text such as “a man jumps” and adjusts motion parameters. Once the settings are finalized, the system begins processing the motion based on these inputs, as seen in the left panel. Fig. 31b showcases the generated motion based on the user’s input. The interface provides a rendered output of the skeleton performing the described motion. This presentation allows users to easily correlate the input parameters with the resulting animation. The generated motion can further be edited by adjusting parameters such as the length of the motion, emphasizing or de-emphasizing certain actions, or replacing actions altogether, depending on user requirements. This process demonstrates how the interface facilitates a workflow from input to motion visualization.

[Figure 228]

[Figure 229]

(a) Motion generation interface example. (b) Generated Motion Example

Figure 31: Motion generation and its output examples. The logical sequence of operations is as follows:

- 1. Input the text: Users start by entering text describing the motion (e.g., “a man jumps.”) or set the frames of motions to generate (as shown in Fig. 31a).
- 2. Generate the initial motion: The system generates the corresponding skeleton motion sequence based on the input text (as shown in Fig. 31b).
- 3. Motion editing: We show some downstream tasks of MotionCLR here.

- • Motion emphasizing/de-emphasizing: Users can select a specific word from the text (e.g., “jumps”) and adjust its emphasis using a weight slider (range [-1, 1]) (as seen in Fig. 32a). For example, setting the weight to 0.3 will either increase the jump motion’s intensity.
- • In-place replacement: If users want to change the action, they can select the “replace” option. For example, replacing “jumps” with “walks” will regenerate the motion, showing a comparison between the original and new edited motions (as shown in Fig. 32b).
- • Example-based motion generation: Users can generate motion sequences based on predefined examples by setting parameters like chunk size and diffusion steps. After specifying the number of motions to generate, the system will create multiple variations of the input motion, providing diverse options for further refinement (as illustrated in Fig. 32d). The progress bars of the process are visualized in Fig. 32c.

[Figure 230]

[Figure 231]

(a) Motion (de-)Emphasizing interface. (b) In-place replacement example.

[Figure 232]

[Figure 233]

(c) Example-based motion generation progress. (d) Example-based motion generation results.

Figure 32: Different interfaces and supporting functions for interactive motion editing.

.

