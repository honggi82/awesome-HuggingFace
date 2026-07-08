## Foundation Models in Robotics: Applications, Challenges, and the Future

Roya Firoozi1, Johnathan Tucker1, Stephen Tian1, Anirudha Majumdar2,6, Jiankai Sun1, Weiyu Liu1, Yuke Zhu3,4, Shuran Song1, Ashish Kapoor5, Karol Hausman1,6, Brian Ichter6, Danny Driess6,7, Jiajun Wu1, Cewu Lu8, Mac Schwager1

arXiv:2312.07843v1[cs.RO]13Dec2023

1Stanford University, 2Princeton University, 3UT Austin, 4NVIDIA, 5Scaled Foundations, 6Google DeepMind, 7TU Berlin, 8Shanghai Jiao Tong University

CONTENTS

- I Introduction 3
- II Foundation Models Background 4

- II-A Terminology and Mathematical Preliminaries . . . . . . . . . . . . . . . . . . 4
- II-B Large Language Model (LLM) Examples and Historical Context . . . . . . . 7
- II-C Vision Transformers . . . . . . . . . . . 7
- II-D Multimodal Vision-Language Models (VLMs) . . . . . . . . . . . . . . . . . 8
- II-E Embodied Multimodal Language Models 8
- II-F Visual Generative Models . . . . . . . . 8

- III Robotics 8

- III-A Robot Policy Learning for Decision Making and Control . . . . . . . . . . . 9

- III-A1 Language-conditioned Imitation Learning for Manipulation . . . . . . . . . . . . . 9
- III-A2 Language-Assisted Reinforcement Learning . . . 11

- III-B Language-Image Goal-Conditioned Value Learning . . . . . . . . . . . . . . 11
- III-C Robot Task Planning using Large Language Models . . . . . . . . . . . . . . 12 III-C1 Language Instructions for

Task Speciﬁcation . . . . . . 12

III-C2 Code Generation using Language Models for Task Planning . . . . . . . . . . . . . 12

- III-D In-context Learning (ICL) for DecisionMaking . . . . . . . . . . . . . . . . . . 13
- III-E Robot Transformers . . . . . . . . . . . 13
- III-F Open-Vocabulary Robot Navigation and Manipulation . . . . . . . . . . . . . . . 15

- III-F1 Open-Vocabulary Navigation 15
- III-F2 Open-Vocabulary Manipulation 16

- IV Perception 16 IV-A Open-Vocabulary Object Detection and

3D Classiﬁcation . . . . . . . . . . . . 16

- IV-A1 Object Detection . . . . . . . 16
- IV-A2 3D Classiﬁcation . . . . . . 17

- IV-B Open-Vocabulary Semantic Segmentation 18
- IV-C Open-Vocabulary 3D Scene and Object Representations . . . . . . . . . . . . . 18 IV-C1 Language Grounding in 3D

Scene . . . . . . . . . . . . . 18 IV-C2 Scene Editing . . . . . . . . 19 IV-C3 Object Representations . . . 20

- IV-D Learned Affordances . . . . . . . . . . 20
- IV-E Predictive Models . . . . . . . . . . . . 21

- V Embodied AI 21

- V-A Generalist AI . . . . . . . . . . . . . . 22
- V-B Simulators . . . . . . . . . . . . . . . . 23

- VI Challenges and Future Directions 23

- VI-A Overcoming Data Scarcity in Training Foundation Models for Robotics . . . . 23

- VI-A1 Scaling Robot Learning Using Unstructured Play Data and Unlabeled Videos of Humans . . . . . . . . . . . . . 23
- VI-A2 Data Augmentation using Inpainting . . . . . . . . . . . 24
- VI-A3 Overcoming 3D Data Scarcity for Training 3D Foundation Models . . . . . 24
- VI-A4 Synthetic Data Generation via High-Fidelity Simulation 24
- VI-A5 Data Augmentation using VLMs . . . . . . . . . . . . 24
- VI-A6 Robot Physical Skills are Limited to Distribution of Skills . . . . . . . . . . . . . 24

- VI-B Real Time Performance (High Inference Time of Foundation Models) . . . . . . 24
- VI-C Limitations in Multimodal Representation 25
- VI-D Uncertainty Quantiﬁcation . . . . . . . 25 VI-D1 Instance-Level Uncertainty

Quantiﬁcation . . . . . . . . 25

VI-D2 Distribution-Level

Uncertainty Quantiﬁcation . 25 VI-D3 Calibration . . . . . . . . . . 25 VI-D4 Distribution Shift . . . . . . 25 VI-D5 Case Study: Uncertainty

Quantiﬁcation for LanguageInstructed Robots . . . . . . 26

- VI-E Safety Evaluation . . . . . . . . . . . . 26

- VI-E1 Pre-deployment safety tests . 26
- VI-E2 Runtime monitoring and outof-distribution detection . . . 26

- VI-F Using Existing Foundation Models as Plug-and-Play or Building New Foundation Models for Robotics . . . . . . . 26
- VI-G High Variability in Robotic Settings . . 27
- VI-H Benchmarking and Reproducibility in Robotics Settings . . . . . . . . . . . . 27

VII Conclusion 27 References 27

Abstract—We survey applications of pretrained foundation models in robotics. Traditional deep learning models in robotics are trained on small datasets tailored for speciﬁc tasks, which limits their adaptability across diverse applications. In contrast, foundation models pretrained on internet-scale data appear to have superior generalization capabilities, and in some instances display an emergent ability to ﬁnd zero-shot solutions to problems that are not present in the training data. Foundation models may hold the potential to enhance various components of the robot autonomy stack, from perception to decision-making and control. For example, large language models can generate code or provide common sense reasoning, while vision-language models enable open-vocabulary visual recognition. However, signiﬁcant open research challenges remain, particularly around the scarcity of robot-relevant training data, safety guarantees and uncertainty quantiﬁcation, and real-time execution. In this survey, we study recent papers that have used or built foundation models to solve robotics problems. We explore how foundation models contribute to improving robot capabilities in the domains of perception, decision-making, and control. We discuss the challenges hindering the adoption of foundation models in robot autonomy and provide opportunities and potential pathways for future advancements. The GitHub project corresponding to this paper1 can be found here.

Index Terms—Robotics, Large Language Models (LLMs), Visual-Language Models (VLM), Large Pretrained Models, Foundation Models

I. INTRODUCTION

# F

OUNDATION models are pretrained on extensive internet-scale data and can be ﬁne-tuned for adaptation to

a wide range of downstream tasks. Foundation models have demonstrated signiﬁcant breakthroughs in vision and language processing; examples include BERT [1], GPT-3 [2], GPT-4 [3], CLIP [4], DALL-E [5], and PaLM-E [6]. Foundation models have the potential to unlock new possibilities in robotics domains such as autonomous driving, household robotics, industrial robotics, assistive robotics, medical robotics, ﬁeld robotics, and multi-robot systems. Pretrained Large Language Models (LLMs), Large Vision-Language Models (VLMs), Large Audio-Language Models (ALMs), and Large VisualNavigation Models (VNMs) can be utilized to improve various tasks in robotics settings. The integration of foundation models into robotics is a rapidly evolving area, and the robotics community has very recently started exploring ways to leverage these large models within the robotics domain for perception, prediction, planning, and control.

Prior to the emergence of foundation models, traditional deep learning models for robotics were typically trained on limited datasets gathered for distinct tasks [7]. Conversely, foundation models are pre-trained on extensive and diverse data, which has been proven in other domains (such as natural language processing, computer vision, and healthcare [8]) to signiﬁcantly expand adaptability, generalization capability, and overall performance. Ultimately, foundation models may hold the potential to yield these same beneﬁts in robotics. Knowledge transfer from foundation models may reduce training time and computational resources compared to task-speciﬁc

1Preliminary release. We are committed to further enhancing and updating this work to ensure its quality and relevance

models. Particularly relevant to robotics, multimodal foundation models can fuse and align multimodal heterogeneous data gathered from various sensors into compact homogeneous representations needed for robot understanding and reasoning [9]. These learned representations hold the potential to be used in any part of the autonomy stack including perception, decisionmaking, and control. Furthermore, foundation models provide zero-shot capabilities, which refer to the ability of an AI system to perform tasks without prior examples or dedicated training data for that speciﬁc task. The would enable robots to generalize their learned knowledge to novel cases, enhancing adaptability and ﬂexibility for robots in unstructured settings.

Integrating foundation models into robotic systems may enable context-aware robotic systems by enhancing the robot’s ability to perceive and interact with the environment. For example in the perception domain, Large Vision-Language Models (VLMs) have been found to provide cross-modal understanding by learning associations between visual and textual data, aiding tasks such as zero-shot image classiﬁcation, zero-shot object detection [10], and 3D classiﬁcation [11]. As another example, language grounding in the 3D world [12] (aligning contextual understanding of VLMs to the 3dimensional (3D) real world) may enhance a robot’s spatial awareness by associating words with speciﬁc objects, locations, or actions within the 3D environment.

In the decision-making or planning domain, LLMs and VLMs have been found to assist robots in task speciﬁcation for high-level planning [13]. Robots can perform more complex tasks by leveraging linguistic cues in manipulation, navigation, and interaction. For example, for robot policy learning techniques like imitation learning [14] and reinforcement learning [15], foundation models seem to offer the possibility to improve data efﬁciency and enhance contextual understanding. In particular, language-driven rewards can be used to guide RL agents by providing shaped rewards [16]. Also, researchers have employed language models to provide feedback for policy learning techniques [17]. Some works have shown that a VLM model’s visual question-answering (VQA) capability can be harnessed in robotics use cases. For example, researchers have used VLMs to answer questions related to visual content to aid robots in accomplishing their tasks [18]. Also, researchers have stated utilizing VLMs to help with data annotation, by generating descriptive labels for visual content [19].

Despite the transformative capabilities of foundation models in vision and language processing, the generalization and ﬁnetuning of foundation models for real-world robotics tasks remain challenging. These challenges include: 1) Data Scarcity: how to obtain internet-scale data for robot manipulation, locomotion, navigation, and other robotics tasks, and how to perform self-supervised training with this data, 2) High Variability: how to deal with the large diversity in physical environments, physical robot platforms, and potential robot tasks while still maintaining the generality required for a foundation model, 3) Uncertainty Quantiﬁcation: how to deal with (i) instance-level uncertainty such as language ambiguity or LLM hallucination; (ii) distribution-level uncertainty; and (iii) distribution-shift, especially resulting from closed-loop

robot deployment, 4) Safety Evaluation: How to rigorously test for the safety of a foundation model-based robotic system (i) prior to deployment, (ii) as the model is updated throughout its lifecycle, and (iii) as the robot operates in its target environments. 5) Real-Time Performance: how to deal with the high inference time of some foundation models which could hinder their deployment on robots and how to accelerate inference in foundation models to the speed required for online decision-making.

In this survey, we study the existing literature on the use of foundation models in robotics. We study current approaches and applications, present current challenges, suggest directions for future research to address these challenges, and identify potential risks exposed by integrating foundation models into robot autonomy. Another survey on foundation models in robotics appeared simultaneously with ours on arXiv [20]. In comparison with that paper, ours emphasizes future challenges and opportunities, including safety and risk, and ours has a stronger emphasis on comparisons in applications, algorithms, and architectures among the existing papers in this space. In contrast to some existing surveys that focus on a speciﬁc incontext instruction, such as prompts [21], vision transformers [22], or decision-making [13], [23], we provide a broader perspective to connect distinct research threads in foundation models organized around their relevance to and application to robotics. Conversely, our scope is much narrower than the paper [24], which explores the broad application of foundation models across many disciplines, of which robotics is one. We hope this paper can provide clarity regarding areas of recent progress and existing deﬁciencies in the research, and point the way forward to future opportunities and challenges facing this research area. Ultimately, we aim to give a resource for robotics researchers to learn about this exciting new area.

We limit the scope of this survey to papers that fall into one of the following categories:

- 1) Background Papers: Papers that do not explicitly link to robotics, but are nonetheless required for understanding foundation models. These papers are discussed in the background section (section II) of the survey paper.
- 2) Robotics Papers: Papers that integrate a foundation model into a robotic system in a plug-and-play fashion, papers that adapt or ﬁne-tune foundation models for robotic systems, or papers that build new robotic-speciﬁc foundation models.
- 3) Robotics-Adjacent Papers: Papers that present methods or techniques applied to areas adjacent to robotics (e.g., computer vision, embodied AI), with a clear path to future application in robotics.

This survey is organized as follows: In Section II, we provide an introduction to foundation models including LLMs, vision transformers, VLMs, embodied multimodal language models, and visual generative models. In addition, in the last part of this section, we discuss different training methods used to train foundation models. In Section III, we present a review of how foundation models are integrated into different tasks for decision-making in robotics. First, we discuss robot policy learning using language-conditioned imitation learning,

and language-assisted reinforcement learning. Then, we discuss how to use foundation models to design a languageconditioned value function that can be used for planning purposes. Next, robot task speciﬁcation and code generation for task planning using foundation models are presented. In Section IV, we study various perception tasks in robotics that have the potential to be enhanced by employing foundation models. These tasks include semantic segmentation, 3D scene representation, zero-shot 3D classiﬁcation, affordance prediction, and dynamics prediction. In Section V, we present papers about Embodied AI agents, generalist AI agents, as well as simulators and benchmarks developed for embodied AI research. In Section VI, we conclude the survey by discussing different challenges for employing foundation models in robotic systems and proposing potential avenues for future research. Finally, in Section VII we offer the concluding remarks.

II. FOUNDATION MODELS BACKGROUND

Foundation models have billions of parameters and are pretrained on massive internet-scale datasets. Training models of such scale and complexity involve substantial costs. Acquiring, processing, and managing data can be costly. The training process demands signiﬁcant computational resources, requiring specialized hardware such as GPUs or TPUs, as well as software and infrastructure for model training which requires ﬁnancial resources. Additionally, training a foundation model is time-intensive, which can translate to even higher costs. Hence these models are often used as plug-and-play modules (which refers to the integration of foundation models into various applications without the need for extensive customization). Table I provides details about commonly used foundation models. In the rest of this section, we introduce LLMs, vision transformers, VLMs, embodied multi-modal language models, and visual generative models. In the last part of this section, we introduce different training methods that are used to train foundation models.

A. Terminology and Mathematical Preliminaries

In this section, we ﬁrst introduce common terminologies in the context of foundation models and describe basic mathematical details and training practices for various types of foundation models.

Tokenization: Given a sequence of characters, tokenization is the process of dividing the sequence into smaller units, called tokens. Depending on the tokenization strategy, tokens can be characters, segments of words, complete words, or portions of sentences. Tokens are represented as 1-hot vectors of dimension equal to the size of the total vocabulary and are mapped to lower-dimensional vectors of real numbers through a learned embedding matrix. An LLM takes a sequence of these embedding vectors as raw input, producing a sequence of embedding vectors as raw output. These output vectors are then mapped back to tokens and hence to text. GPT-3, for example, has a vocabulary of 50,257 different tokens, and an embedding dimension of 12,288.

Language-Conditioned Imitation Learning

e.g. CLIPort [25], Play-LMP [26], PerAct [27], Multi-Context Imitation [28], CACTI [14], Voltron [29]

Robot Policy Learning

Language-Assisted Reinforcement Learning

e.g. Adaptive Agent (AdA) [30], Palo et al. [15]

Language-Image Goal-Conditioned Value Learning

e.g. R3M [31], SayCan [32], Inner Monologue [33], VoxPoser [34], Mahmoudieh et al. [35], VIP [36], LIV [37], LOREL [38]

High-Level Task Planning

Robotics

e.g. NL2TL [39], Chen et al. [40]

FoundationModelsinRobotics

LLM-Based Code Generation

e.g. ProgPrompt [41], Code-as-Policy [42], ChatGPT-Robotics [43]

Robot Transformers e.g. RT-1 [44], RT-2 [45], RT-X [46], PACT [47], Xiao et al. [48], Radosavovic et al. [49], LATTE [50]

Open-Vocabulary Object Detection and 3D classiﬁcation

e.g. OWL-ViT [51], GLIP [52], Grounding DINO [53], PointCLIP [54], PointBERT [55], ULIP [56], [57]

Open-Vocabulary Semantic Segmentation

e.g. LSeg [58], Segment Anything [59], FastSAM [60], MobileSAM [61], Track Anything Model (TAM) [62]

Perception

RelevanttoRobotics

Open-Vocabulary 3D Scene Representation

e.g. CLIP-NERF [63], LERF [64], DFF [65]

Affordances e.g. Affordance Diffusion [66], VRB [67]

e.g. Huang et al. [68], Statler [69], EmbodiedGPT [70], MineDojo [71], VPT [72], Kwon et al. [16], Voyager [73], ELLM [74]

Embodied AI

Fig. 1. Overview of Robotics Tasks Leveraging Foundation Models.

The token decoding (from low-dimension real-valued embedding vectors to high-dimension 1-hot vectors) is not deterministic, resulting in a weighting for each possible token in the vocabulary. These weightings are often used by LLMs as probabilities over tokens, to introduce randomness in the text generation process. For example, the temperature parameter in GPT-3 blends between always choosing the top-weighted token (temperature of 0) and drawing the token based on the probability distribution suggested by the weights (temperature of 1). This source of randomness is only in the token decoding process, not in the LLM itself. To the authors’ knowledge, this is, in fact, the only source of randomness in the GPT family of models.

One of the most common tokenization schemes, which is used by the GPT family of models, is called byte-pair encoding [75]. Byte-pair encoding starts with a token for each individual symbol (e.g., letter, punctuation), then recursively builds tokens by grouping pairs of symbols that commonly appear together, building up to assign tokens to larger and larger groups (pairs of pairs, etc) that appear frequently together in a text corpus. The tokenization process can extend beyond text data to diverse contexts, encompassing various data modalities like images, videos, and robot actions. In these scenarios, the respective data modalities can be treated as sequential data and tokenized similarly to train generative models. For example, just as language constitutes a sequence of words, an image comprises a sequence of image patches, force sensors yield a

sequence of sensory inputs at each time step, and a series of actions represent the sequential nature of tasks for a robot.

Generative Models: A generative model is a model that learns to sample from a probability distribution to create examples of data that seem to be from the same distribution as the training data. For example, a face generation model can produce images of faces that cannot be distinguished from the set of real images used to train the model. These models can be trained to be conditional, meaning they generate samples from a conditional distribution conditioned on a wide range of possible conditioning information. For example, a gender conditional face generator can generate images of female or male faces, where the desired gender is given as a conditioning input to the model.

Discriminative Models: Discriminative models are used for regression or classiﬁcation tasks. In contrast to generative models, discriminative models are trained to distinguish between different classes or categories. Their emphasis lies in learning the boundaries between classes within the input space. While generative models learn to sample from the distribution over the data, discriminative models learn to evaluate the probability distribution of the output labels given the input features, or (depending on how the model is trained) learn to evaluate some statistic of the probability distribution over the outputs, such as the expected output given an input.

Transformer Architecture: Most foundation models are built on the transformer architecture, which has been instru-

mental in the rise of foundation models and large language models. The following discussion was synthesized from [76], as well as online blogs, unpublished reports, and wikipedia [77]–[79]. A transformer acts simultaneously on a collection of embedded token vectors (x1,...,xN) known as a context window. The key enabling innovation of the Transformer architecture is the multi-head self-attention mechanism originally proposed in the seminal work [76]. In this architecture, each attention head computes a vector of importance weights that corresponds to how strongly a token in the context window xi correlates with other tokens in the same window xj. Each attention head mathematically encodes different notions of similarity, through different projection matrices used in the computation of the importance weights. Each head can be trained (backward pass) and evaluated (forward pass) in parallel across all tokens and across all heads, leading to faster training and inference when compared with previous models based on RNNs or LSTMs.

Mathematically, an attention head maps each token xi in the context window to a “query” qi = WqTxi, and each other token in the context head xj to a “key” kj = WkTxj. The similarity between query and key is then measured through a scaled dot product, qiTkj/

√

[Figure 1]

d, where d is the dimension of the query and key vectors. A softmax is then taken over all j to give weights αij representing how much xi “attends to” xj. The tokens are then mapped to “values” with vj = WvTxj, and the output of the attention for position i is then given as a sum over values weighted by attention weights, j αijvj. One of the key reasons for the success of the transformer attention model is that it can be efﬁciently computed with GPUs and TPUs by parallelizing the preceding steps into matrix computations,

attn(Q,K,V) = softmax

QK⊤ √dk

[Figure 2]

[Figure 3]

V, (1)

where Q, K, V are matrices with rows qiT, kiT, and viT, respectively. Each head in the model produces this compu-

tation independently, with different Wq,Wk,Wv matrices to encode different kinds of attention. The outputs from each head are then concatenated, normalized with a skip connection, passed through a fully connected ReLU layer, and normalized again with a skip connection to produce the output of the attention layer. Multiple layers are arranged in various ways to give “encoders” and “decoders,” which together make up a transformer.

The size of a transformer model is typically quantiﬁed by (i) the size of the context window, (ii) the number of attention heads per layer, (iii) the size of the attention vectors in each head, and (iii) the number of stacked attention layers. For example, GPT-3’s context window is 2048 tokens (corresponding to about 1500 words of text), each attention layer has 96 heads, each head has attention vectors of 128 dimensions, and there are 96 stacked attention layers in the model.

The basic multi-head attention mechanism does not impose any inherent sense of sequence or directionality in the data. However, transformers—especially in natural language applications—are often used as sequence predictors by imposing a positional encoding on the input token sequence. They are then applied to a token sequence autoregressively, meaning

they predict the next token in the sequence, add that token to their context window, and repeat. This concept is elaborated below.

Autoregressive Models: The concept of autoregression has been applied in many ﬁelds as a representation of random processes whose outputs depend causally on the previous outputs. Autoregressive models use a window of past data to predict the next data point in a sequence. The window then slides one position forward, recursively ingesting the predicted data point into the window and expelling the oldest data point from the window. The model again predicts the next data point in the sequence, repeating this process indeﬁnitely. Classical linear autoregressive models such as Auto-Regressive Moving Average (ARMA) and Auto-Regressive Moving Average with eXogenous input (ARMAX) models are standard statistical tools dating back to at least the 1970s [80]. These modeling concepts were adapted to deep learning models ﬁrst with RNNs, and later LSTMs, which are both types of learnable nonlinear autoregressive models. Transformer models, although they are not inherently autoregressive, are often adapted to an autoregressive framework for text prediction tasks.

For example, the GPT family [81] builds on the original transformer model by using a modiﬁcation introduced in [82] that removes the transformer encoder blocks entirely, retaining just the transformer decoder blocks. This has the advantage of reducing the number of model parameters by close to half while reducing redundant information that is learned in both the encoder and decoder. During training, the GPT model seeks to produce an output token from the tokenized corpus X = (x1,...,xn) to minimize the negative log-likelihood within the context window of length N,

LLLM = −

i

log P (xi | xi−N,...,xi−1). (2)

This results in a large pretrained model that autoregressively predicts the next likely token given the tokens in the context window. Although powerful, the unidirectional autoregressive nature of the GPT family means that these models may lag in performance on bidirectional tasks such as reading comprehension.

Masked Auto-Encoding: To address the unidirectional limitation of the GPT family and allow the model to make bidirectional predictions, works such as BERT [1] use masked autoencoding. This is achieved through an architectural change, namely the addition of a bidirectional encoder, as well as a novel pre-training objective known as masked language modeling (MLM). The MLM task simply masks a percentage of the tokens in the corpus and requires the model to predict these tokens. Through this procedure, the model is encouraged to learn the context that surrounds a word rather than just the next likely word in a sequence.

Contrastive Learning: Visual-language foundation models such as CLIP [4] typically rely on different training methods from the ones used with large language models which encourage explicitly predictive behavior. Visual-language models use contrastive representation learning, where the goal is to learn a

joint embedding space between input modalities where similar sample pairs are closer than dissimilar ones. The training objective for many VLMs is some variation of the objective function,

exp(sim(vi,ui)/ τ)

ℓ(iv→u) = − log

, (3)

[Figure 4]

N k=1 exp(sim(vi,uk)/ τ)

exp(sim(ui,vi)/ τ)

ℓ(iu→v) = − log

, (4)

[Figure 5]

N k=1 exp(sim(ui,vk)/ τ)

N

1 N

λℓ(iv→u) + (1 − λ)ℓ(iu→v) . (5)

L =

[Figure 6]

i=1

This objective function was popularized for multimodal input by ConVIRT [83] and ﬁrst presented in prior works [84]–[87]. This objective function trains the image and text encoders to preserve mutual information between the true text and image pairs. In these equations, ui and vi are the ith encoded text and image respectively from i ∈ 1,...,N image and text pairs. The sim operation is the cosine similarity between the text and image embeddings, and τ is a temperature term. In CLIP [4] the authors use a symmetric cross-entropy loss, meaning the ﬁnal loss is an average of the two loss components where each is equally weighted (i.e. λ = 0.5).

Diffusion Models: Outside of large language models and multi-modal models such as VLMs, diffusion models for image generation (e.g. DALL-E2) [88] are another class of foundation models considered in this survey. Although diffusion models were established in prior work [89], [90] the diffusion probabilistic model presented in [91] popularized the method. The diffusion probabilistic model is a deep generative model that is trained in an iterative forward and reverse process. The forward process adds Gaussian noise to an input x0 in a Markov chain until xT when the result is zero mean isotropic noise. This means the forward process produces a trajectory of noise q(x1:T|x0) as,

q (x1:T | x0) :=

T

q (xt | xt−1). (6)

t=1

At each time step q(xt|xt−1) is described by a normal distribution with mean √1 − βtxt−1 and covariance βtI where βt is scheduled or a ﬁxed hyperparameter.

[Figure 7]

The reverse process requires the model to learn to the transitions that will de-noise the zero mean Gaussian and produce the input image. This process is also deﬁned as a Markov chain where the transition distribution at time t is pθ (xt−1 | xt) := N (xt−1;µθ (xt,t),Σθ (xt,t)). For completeness, the reverse process Markov chain is given by,

T

pθ (x0:T) := p (xT)

t=1

pθ (xt−1 | xt). (7)

Diffusion models are trained using a reduced form of the evidence lower bound loss function that is typical of variational

generative models like variational autoencoders (VAEs). The reduced loss function used for training is

L = Eq[DKL(q(xT | x0) p(xT)) (8)

DKL(q(xt−1 | xt,x0) pθ(xt−1 | xt)

+

t>1

− log pθ(x0 | x1)],

where DKL(q||p) denotes Kullback–Leibler divergence, which is a measure of how different a distribution q is from a distribution p.

- B. Large Language Model (LLM) Examples and Historical Context

LLMs have billions of parameters and are trained on trillions of tokens. This large scale has allowed models such as GPT-2 [92] and BERT [1] to achieve state-of-the-art performance on the Winograd Schema challenge [93] and the General Language Understanding Evaluation (GLUE) [94] benchmarks, respectively. Their successors include GPT-3 [2], LLaMA [95], and PaLM [96] has grown considerably in the number of parameters (typically now over 100 billion), the size of the context window (typically now over 1000 tokens), and the size of the training data set (typically now 10s of terabytes of text). GPT-3 is trained on the Common Crawl dataset. Common Crawl contains petabytes of publicly available data over 12 years of web crawling and includes raw web page data, metadata, and text extracts. LLMs can also be multilingual. For example, ChatGLM-6B and GLM-130B [97] is a bilingual (English and Chinese) pretrained language model with 130 billion parameters. LLMs can also be ﬁne-tuned, a process by which the model parameters are adjusted with domain-speciﬁc data to align the performance of the LLM to a speciﬁc use case. For example, GPT-3 and GPT-4 [3] have been ﬁne-tuned using reinforcement learning with human feedback (RLHF).

- C. Vision Transformers

A Vision Transformer (ViT) [98]–[100] is a transformer architecture for computer vision tasks including image classiﬁcation segmentation, and object detection. A ViT treats an image as a sequence of image patches referred to as tokens. In the image tokenization process, an image is divided into patches of ﬁxed size. Then the patches are ﬂattened into a onedimensional vector which is referred to as linear embedding. To capture the spatial relationships between image patches, positional information is added to each token. This process is referred to as position embedding. The image tokens incorporated with position encoding are fed into the transformer encoder and the self-attention mechanism enables the model to capture long-term dependencies and global patterns in the input data. In this paper, we focus only on those ViT models with a large number of parameters. ViT-G [101] scales up the ViT model and has 2B parameters. Additionally, ViT-e [102] has 4B parameters. ViT-22B [103] is a vision transformer model at 22 billion parameters, which is used in PaLM-E and PaLI-X [104] and helps with robotics tasks.

DINO [105] is a self-supervised learning method, for training ViT. DINO is a form of knowledge distillation with no labels. Knowledge distillation is a learning framework where a smaller model (student network) is trained to mimic the behavior of a larger more complex model (teacher network). Both networks share the same architecture with different sets of parameters. Given a ﬁxed teacher network, the student network learns its parameters by minimizing the cross-entropy loss w.r.t. the student network parameters. The neural network architecture is composed of ViT or ResNet [106] backbone and a projection head that includes layers of multi-layer perception (MLP). Self-supervised ViT features learned using DINO contain explicit information about the semantic segmentation of an image including scene layout and object boundaries with such clarity that is not achieved using supervised ViTs or convnets.

DINOv2 [107] provides a variety of pretrained visual models that are trained with different vision transformers (ViT) on the LVD-142M dataset introduced in [107]. It is trained using a discriminative self-supervised method on a compute cluster of 20 nodes equipped with 8 V100-32GB GPUs. DINOv2 provides various visual features at the image (e.g. detection) or pixel level (e.g. segmentation). SAM [59] provides zeroshot promptable image segmentation. It is discussed in more detail in Section IV.

D. Multimodal Vision-Language Models (VLMs)

Multimodal refers to the ability of a model to accept different “modalities” of inputs, for example, images, texts, or audio signals. Visual-language models (VLM) are a type of multimodal model that takes in both images and text. A commonly used VLM in robotics applications is Contrastive LanguageImage Pre-training (CLIP) [4]. CLIP offers a method to compare the similarity between textual descriptions and images. CLIP uses internet-scale image-text pairs data to capture the semantic information between images and text. CLIP model architecture contains a text encoder [92] and an image encoder (a modiﬁed version of vision transformer ViT) that are trained jointly to maximize the cosine similarity of the image and text embeddings. CLIP uses contrastive learning together with language models and visual feature encoders to incorporate models for zero-shot image classiﬁcation.

BLIP [108] focuses on multimodal learning by jointly optimizing three objectives during pretraining. These objectives include Image-Text Contrastive Loss, Image-Text Matching Loss, and Language Modeling Loss. The method leverages noisy web data by bootstrapping captions, enhancing the training process. CLIP2 [109] aims to build well-aligned and instance-based text-image-point proxies. It learns semantic and instance-level aligned point cloud representations using a cross-modal contrastive objective. FILIP [110] focuses on achieving ﬁner-level alignment in multimodal learning. It incorporates a cross-modal late interaction mechanism that utilizes token-wise maximum similarity between visual and textual tokens. This mechanism guides the contrastive objective and improves the alignment between visual and textual information. FLIP [111] proposes a simple and more efﬁcient

training method for CLIP. FLIP randomly masks out and removes a signiﬁcant portion of image patches during training. This approach aims to improve the training efﬁciency of CLIP while maintaining its performance.

E. Embodied Multimodal Language Models

An embodied agent is an AI system that interacts with a virtual or physical world. Examples include virtual assistance or robots. Embodied language models are foundation models that incorporate real-world sensor and actuation modalities into pretrained large language models. Typical vision-language models are trained on general vision-language tasks such as image captioning or visual question answering. PaLM-

- E [6] is a multimodal language model that has been trained on not only internet-scale general vision-language data, but also on embodied, robotics data, simultaneously. In order to connect the model to real world sensor modalities, PaLME’s architecture injects (continuous) inputs such as images, low-level states, or 3D neural scene representations into the language embedding space of a decoder-only language model to enable the model to reason about text and other modalities jointly. The main PaLM-E version is built from the PaLM LLM [96] and a ViT [103]. The ViT transforms an image into a sequence of embedding vectors which are projected into the language embedding space via an afﬁne transformation. The whole model is trained end-to-end, starting from a pre-trained LLM and ViT model. The authors also explore different strategies such as freezing the LLM and just training the ViT, which leads to worse performance. Given multimodal inputs, the output of PaLM-E is text decoded auto-regressively. In order to connect this output to a robot for control, language conditioned short-horizon policies can be used. In this case, PaLM-E acts as a high-level control policy. Experiments show that a single PaLM-E, in addition to being a visionlanguage generalist, is able to perform many different robotics tasks over multiple robot embodiments. The model exhibits positive transfer, i.e. simultaneously training on internet-scale language, general vision-language, and embodied domains leads to higher performance compared to training the model on single tasks.
- F. Visual Generative Models

Web-scale diffusion models such as OpenAI’s DALLE [112] and DALL-E2 [88] provide zero-shot text-to-image generation. They are trained on hundreds of millions of imagecaption pairs from the internet. These models learn a languageconditioned distribution over images from which an image can be generated using a given prompt. The DALL-E2 architecture includes a prior that generates a CLIP image embedding from a text caption, and a decoder that generates an image conditioned on the image embedding.

III. ROBOTICS

In this section, we delve into robot decision-making, planning, and control. Within this realm, Large Language Models (LLMs) and Visual Language Models (VLMs) may hold the

TABLE I LARGE PRETRAINED MODELS

[Figure 8]

Model Architecture Size Training Data What to Pretrain How to Pretrain

Hardware

[Figure 9]

CLIP [4] ViT-L/14@336px and a text encoder [92]

0.307B 400M image-text pairs zero-shot image classiﬁcation

contrastive pre-training

ﬁne-tuned CLIP model is trained for 12 days on 256 V100 GPUs

[Figure 10]

- GPT-3 [2] transformer (slight modiﬁcation of GPT-2)

[Figure 11]

[Figure 12]

175B Common Crawl (about a trillion words)

[Figure 13]

[Figure 14]

text output autoregressive model

[Figure 15]

[Figure 16]

NPA *

- GPT-4 [3] NPA NPA NPA text output NPA NPA PaLI-X [104] encoder-decoder 55B 10B image-text pairs

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

text and image to text output

autoregressive model

runs on multi-TPU cloud service

from WebLI [102] and auxiliary tasks

DALL-E [112] decoder-only transformer

12B 250M text-image pairs zero-shot text-to-image generation

autoregressive model

NPA

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

DALL-E2 [88] a prior based on CLIP+ a decoder

3.5B CLIP and DALL-E [112]

zero-shot text-to-image generation

diffusion NPA

DINOv2 [107] ViT-g/14 1.1B LVD-142M [107] visual-features (image-level and pixel-level)

discriminative 20 nodes

equipped with 8 V100-32GB GPUs

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

SAM [59] MAE [113] vision transformer+CLIP [114] text encoder

632M for ViT-H + 63M for CLIP text encoder

SA-1B dataset [59] that includes 1.1B segmentation masks on 11M images

zero-shot promptable image segmentation

supervised learning

256 A100 GPUs for 68 hours

[Figure 38]

* NPA stands for not publicly available.

potential to serve as valuable tools for enhancing robotic capabilities. For instance, LLMs may facilitate the process of task speciﬁcation, allowing robots to receive and interpret high-level instructions from humans. VLMs may also promise contributions to this ﬁeld. VLMs specialize in the analysis of visual data. This visual understanding is a critical component of informed decision-making and complex task execution for robots. Robots can now leverage natural language cues to enhance their performance in tasks involving manipulation, navigation, and interaction. Vision-language goal-conditioned policy learning, whether through imitation learning or reinforcement learning, holds promise for improvement using foundation models. Language models also play a role in offering feedback for policy learning techniques. This feedback loop fosters continual improvement in robotic decisionmaking, as robots can reﬁne their actions based on the feedback received from an LLM. This section underscores the potential contributions of LLMs and VLMs in robot decision-making. Assessing and comparing the contributions of papers in this section presents greater challenges compared to the other sections like the Perception Section (IV) or the Embodied AI Section (V). This is due to the fact that most papers in this section either rely on hardware experiments, using custom elements in the low-level control and planning stack that are not easily transferred to other hardware or other experimental setups, or they utilize non-physics-based simulators, which allow these low-level parts of the stack to be ignored, but leaving open the issue of non-transferability between different hardware implementations. In Section VI, we discuss the lack of benchmarking and reproducibility that needs to be addressed in future research.

A. Robot Policy Learning for Decision Making and Control

In this section we discuss robot policy learning including language-conditioned imitation learning and language-assisted reinforcement learning.

1) Language-conditioned Imitation Learning for Manipulation: In language-conditioned imitation learning, a goalconditioned policy πθ(at|st,l) is learned that outputs actions at ∈ A conditioned on the current state st ∈ S and language instruction l ∈ L. The loss function is deﬁned as the maximum likelihood goal conditioned imitation objective:

|τ|

LGCIL = E(τ,l)∼D

logπθ(at|st,l), (9)

t=0

where D is the language-annotated demonstration dataset D = {τi}Ni . Demonstrations can be represented as trajectories, or sequences of images, RGB-D voxel observations, etc. Language instructions are paired with demonstrations to be used as the training dataset. Each language-annotated demonstration τi consists of τi = {(s1,l1,a1),(s2,l2,a2),...}. At test time, the robot is given a series of instructions and the language-conditioned visuomotor policy πθ provides actions at in a closed loop given the instruction at each time step. The main challenges in this domain are: (i) obtaining a sufﬁcient volume of demonstrations and conditioning labels to train a policy, (ii) distribution shift under the closed-loop policythe feedback of the policy can lead the robot into regions of the state space that are not well-covered in the training data, negatively impacting performance. (All the following papers in this subsection focus on robot manipulation tasks.)

Since generating language-annotated data by pairing demonstrations with language instruction is an expensive pro-

cess, the authors in Play-LMP [26] propose learning from teleoperated play data. In this setting, reusable latent plan representations are learned from unlabeled play data. Also, a goal-conditioned policy is learned to decode the inferred plan to perform the task speciﬁed by the user. In addition, the distributional shift in imitation learning is analyzed and it is shown in this setting that the play data is more robust with respect to perturbation compared to expert positive demonstrations. Note that language goal l in (9) can be substituted with any other type of goal for example goal image, which is another common choice of goal in goal-conditioned imitation learning.

In a follow-up work [28], the authors present multi-context imitation (MCIL) which uses language-conditioned imitation learning over unstructured data. The multi-Context imitation framework is based on relabeled imitation learning and labeled instruction following. MCIL assumes access to multiple contextual imitation datasets, for example, goal image demonstrations, language goal demonstrations, or one-hot task demonstrations. MCIL trains a single latent goal-conditioned policy over all datasets simultaneously by encoding contexts in the shared latent space using the associated encoder for each context. Then a goal-conditioned imitation loss is computed by averaging over all datasets. The policy and goal-encoders are trained end-to-end. Another approach to tackle the data annotation challenge in language-conditioned imitation learning involves utilizing foundation models to offer feedback by labeling demonstrations. In [115], the authors propose to use pretrained foundation models to provide feedback. To deploy a trained policy to a new task or new environment, the policy is played using randomly generated instructions, and a pretrained foundation model provides feedback by labeling the demonstration. Also, this paired instruction-demonstration data can be used for policy ﬁne-tuning. CLIPort [25] also presents a language-conditioned imitation learning for visionbased manipulation. A two-stream architecture is presented that combines the semantic understanding of CLIP with the spatial precision of Transporter [116]. This end-to-end framework solves language-speciﬁed manipulation tasks without any explicit representation of the object poses or instance segmentation. CLIPort grounds semantic concepts in precise spatial reasoning, but it is limited to 2D observation and action spaces. To address this limitation, the authors of PerAct (Perceiver-Actor) [27] propose to represent observation and action spaces with 3D voxels and employ the 3D structure of voxel patches for efﬁcient language-conditioned behavioral cloning with transformers to imitate 6-DoF manipulation tasks from just a few demonstrations. While 2D behavioral cloning methods such as CLIPort are limited to single-view observations, 3D approaches such as PerAct allow for multi-view observations as well as 6-DoF action spaces. PerAct uses only CLIP’s language encoder to encode the language goal. PerAct takes language goals and RGB-D voxel observations as inputs to a Perceiver Transformer and outputs discretized actions by detecting the next best voxel action. PerAct is trained through supervised learning with discrete-time input actions from the demonstration dataset. The demonstration dataset includes voxel observations paired with language goals

and keyframe action sequences. An action consists of a 6DoF pose, gripper open state, and collision avoidance action. During training, a tuple is randomly sampled and the agent predicts the keyframe action given the observation and goal.

Grounding semantic representations into a spatial environment is essential for effective robot interaction. CLIPort and PerAct utilize CLIP (which is trained based on contrastive learning) for semantic reasoning and Transporter and Perceiver for spatial reasoning.

Voltron [29] presents a framework for language-driven representation learning in robotics. Voltron captures semantic, spatial, and temporal representations that are learned from videos and captions. Contrastive learning captures semantic representations but loses spatial relationships, and in contrast, masked autoencoding captures spatial and not semantic representations. Voltron trades off language-conditioned visual reconstruction for local spatial representations and visuallygrounded language generation to capture semantic representations. This framework includes grasp affordance prediction, single-task visuomotor control, referring expression grounding, language-conditioned imitation, and intent-scoring tasks. Voltron models take videos and their associated language captions as input to a multimodal encoder whose outputs are then decoded to reconstruct one or more frames from a masked context. Voltron starts with a masked autoencoding backbone and adds a dynamic component to the model by conditioning the MAE encoder on a language preﬁx. Temporal information is captured by conditioning on multiple frames.

Deploying robot policy learning techniques that leverage language-conditioned imitation learning with real robots presents ongoing challenges. These models rely on end-to-end learning, where the policy maps pixels or voxels to actions. As they are trained through supervised learning on demonstration datasets, they are susceptible to issues related to generalization and distribution shifts. To improve robustness and adaptability, techniques such as data augmentation and domain adaptation can make the policies more robust to the distribution shift.

CACTI [14] is a novel framework designed to enhance scalability in robot learning using foundation models such as Stable Diffusion [117]. CACTI introduces the four stages of data collection, data augmentation, visual representation learning, and imitation policy training. In the data collection stage, limited in-domain expert demonstration data is collected. In the data augmentation stage, CACTI employs visual generative models such as Stable Diffusion [117] to boost visual diversity by augmenting the data with scene and layout variations. In the visual representation learning stage, CACTI leverages pretrained zero-shot visual representation models trained on out-of-domain data to improve training efﬁciency. Finally, in the imitation policy training stage, a general multi-task policy is learned using imitation learning on the augmented dataset with compressed visual representations as input. CACTI is trained for multi-task and multi-scene manipulation in kitchen environments, both in simulation and the real world. The use of these techniques enhances the generalization ability of the framework and enables it to learn from a wide range of environments.

Beyond language, recent works have investigated other

forms of task speciﬁcation. Notably, MimicPlay [118] presents a hierarchical imitation learning algorithm that learns highlevel plans in latent spaces from human play data and lowlevel motor commands from a small number of teleoperated demonstrations. By harnessing the complementary strengths of these two data sources, this algorithm can signiﬁcantly reduce the cost of training visuomotor policies for long-horizon manipulation tasks. Once trained, it is capable of performing new tasks based on one human video demonstration at test time. MUTEX [119] further explores learning a uniﬁed policy across multimodal task speciﬁcations in video, image, text, and audio, showing improved policy performances over singlemodality baselines through cross-modal learning.

2) Language-Assisted Reinforcement Learning: Reinforcement learning (RL) is a family of methods that enable a robot to optimize a policy through interaction with its environment by optimizing a reward function. These interactions are usually in a simulation environment, sometimes augmented with data from physical robot hardware for sim-to-real transfer. RL has close ties to optimal control. Unlike imitation learning, RL does not require human demonstrations, and (in theory) has the potential to attain super-human performance. In the RL problem, the expected return of a policy is maximized using the collected roll-outs from interactions with the environment. The feedback received from the environment in the form of a reward signal guides the robot to learn which actions lead to favorable results and which do not. In this section, we discuss works that have incorporated foundation models (LLM, VLMs, etc.) into RL problems.

Fast and ﬂexible adaptation is a desired capability of artiﬁcial agents and is essential for progress toward general intelligence. In Adaptive Agent (AdA) [30] the authors present an RL foundation model that is an agent pretrained on diverse tasks and is designed to quickly adapt to open-ended embodied 3D problems by using fast in-context learning from feedback. This work considers navigation, coordination, and division of labor tasks. Given a few episodes within an unseen environment at test time, the agent engages in trial-and-error exploration to reﬁne its policy toward optimal performance. In AdA a transformer architecture is trained using modelbased RL2 [120] to train agents with large-scale attentionbased memory, which is required for adaptation. TransformerXL [121] with some modiﬁcation is used to enable long and variable-length context windows to increase the model memory to capture long-term dependencies. The agent collects diverse data in the XLand environment that includes 1040 possible tasks [122], in an automated curriculum. In addition, distillation is used to enable scaling to models with more than 500M parameters.

Palo et al. [15] propose an approach to enhance reinforcement learning by integrating Large Language Models (LLMs) and Visual-Language Models (VLMs) to create a more uniﬁed RL framework. This work considers robot manipulation tasks. Their approach addresses core RL challenges related to exploration, experience reuse and transfer, skills scheduling, and learning from observation. The authors use an LLM to decompose complex tasks into simpler sub-tasks, which are then utilized as inputs for a transformer-based agent to interact

with the environment. The agent is trained using a combination of supervised and reinforcement learning, enabling it to predict the optimal sub-task to execute based on the current state of the environment.

B. Language-Image Goal-Conditioned Value Learning

In value learning, the aim is to construct a value function that aligns goals in different modalities and preserves temporal coherence due to the recursive nature of the value function. Reusable Representation for Robotic Manipulations (R3M) [31] provides pretrained visual representation for robot manipulation using diverse human video datasets such as Ego4D and can be used as a frozen perception module for policy learning in robot manipulation tasks. R3M’s pretrained visual representation is demonstrated on Franka Emika Panda’s arm and enables different downstream manipulation tasks. R3M is trained using time-contrastive learning to capture temporal dependencies, video-language alignment to capture semantic features of the scene (such as objects and their relationships) and L1 penalty to encourage sparse and compact representation. For a batch of videos, using time-contrastive loss, an encoder is trained to generate a representation wherein the distance between images that are temporally closer is minimized compared to images that are farther apart in time or from different videos.

Similar to R3M, Value-Implicit Pretraining (VIP) [36] employs time-contrastive learning to capture temporal dependencies in videos, but it does not require video-language alignment. VIP is also focused on robot manipulation tasks. VIP is a self-supervised approach for learning visual goal-conditioned value functions and representations from videos. VIP learns visual goal-based rewards for downstream tasks and can be used for zero-shot reward speciﬁcation. The reward model is derived from pretrained visual representations. Pretraining involves using unlabeled human videos. Human videos do not contain any action information to be used for robot policy learning, therefore the learned value function does not explicitly depend on actions. VIP introduces a novel time contrastive objective that generates a temporally smooth embedding. The value function is implicitly deﬁned via distance embedding. The proposed implicit time contrastive learning attracts the representation of the initial and goal frames in the same trajectory and repels the representation of intermediate frames by recursive one-step temporal difference minimization. This representation captures long-term temporal dependencies across task frames and local temporal smoothness among adjacent frames.

Language-Image Value Learning (LIV) [37] is a controlcentric vision-language representation. LIV generalizes the prior work VIP by learning multi-modal vision-language value functions and representations using language-aligned videos. For tasks speciﬁed as language goals or image goals, a multimodel representation is trained that encodes a universal value function. LIV is also focused on robot manipulation tasks. LIV is a pretrained control-centric vision-language representation based on large human video datasets such as EPICKITCHENS [123]. The representations are kept frozen during policy learning. A simple MLP is used on top of pretrained

representations for the policy network. Policy learning is decoupled from language-visual representation pretraining. The LIV model is pretrained on arbitrary video activity datasets with text annotation, and the model can be ﬁne-tuned on small datasets of in-domain robot data to ground language in a context-speciﬁc way. LIV uses a generalization of the mutual information-based image-text contrastive representation learning objective as used in CLIP, so LIV can be considered as a combination of CLIP and VIP. Both VIP and LIV learn a self-supervised goal-conditioned value-function objective using contrastive learning. The LIV extends the VIP framework to multi-modal goal speciﬁcations. LOREL [38] learns a language-conditioned reward from ofﬂine data and uses it during model predictive control to complete languagespeciﬁed tasks.

Value functions can be used to help ground semantic information obtained from an LLM to the physical environment in which a robot is operating. By leveraging value functions, a robot can associate the information processed by the LLM with speciﬁc locations and objects in its surroundings. In SayCan [32], researchers investigate the integration of large language models with the physical world through learning. They use the language model to provide task-grounding (Say), enabling the determination of useful sub-goals based on highlevel instructions, and a learned affordance function to achieve world-grounding (Can), enabling the identiﬁcation of feasible actions to execute the plan. Inner Monologue [33] studies the role of grounded environment feedback provided to the LLM, thus closing the loop with the environment. The feedback is used for robot planning with large language models by leveraging a collection of perception models (e.g., scene descriptors and success detectors) in tandem with pretrained language-conditioned robot skills. Feedback includes taskspeciﬁc feedback, such as success detection, and scene-speciﬁc feedback (either “passive” or “active”). In both SayCan and Inner Monologue robot manipulation and navigation tasks are considered using a real-world mobile manipulator robot from Everyday Robots. Text2Motion [124] is a languagebased planning framework for long-horizon robot manipulation. Similar to SayCan and Inner Monologue, Text2Motion computes a score (SLMM) associated with each skill at each time step. The task planning problem is to ﬁnd a sequence of skills by maximizing the likelihood of a skill sequence given a language instruction and the initial state. In Text2Motion, the authors propose to verify that the generated long-horizon plans are symbolically correct and geometrically feasible. Hence, a geometric feasibility score (Sgeo) is deﬁned as the probability that all the skills in the sequence achieve rewards. To compute the overall score, the LLM score is multiplied by the geometric feasibility score (SSkill = SLMM · Sgeo).

VoxPoser [34] builds 3D value maps to ground affordances and constraints into the perceptual space. VoxPser considers robot manipulation tasks. Given the RGB-D observation of the environment and language instruction, VoxPoser utilizes large language models to generate code, which interacts with vision-language models to extract a sequence of 3D affordance maps and constraint maps. These maps are composed together to create 3D value maps. The value maps are then utilized

as objective functions to guide motion planners to synthesize trajectories for everyday manipulation tasks without requiring any prior training or instruction.

In [35], reward shaping using CLIP is presented. This work considers robot manipulation tasks. The proposed model utilizes CLIP to ground objects in a scene described by the goal text paired with spatial relationship rules to shape the reward by using raw pixels as input. They use developments in building large-scale visuo-lingual models like CLIP to devise a framework that generates the task reward signal from just the goal text description and raw pixel observations. This signal is then used to learn the task policy.

In [125], Hierarchical Universal Language Conditioned Policies 2.0 (HULC++) is presented. This work considers robot manipulation tasks. A self-supervised visuo-lingual affordance model is used to learn general-purposed languageconditioned robot skills from unstructured ofﬂine data in the real world. This method requires annotating as little as 1% of the total data with language. The visuo-lingual affordance model has an encoder-decoder architecture with two decoder heads. Both heads share the same encoder and are conditioned on the input language instruction. One head predicts a distribution over the image, in which each pixel likelihood is an afforded point. The other head predicts a Gaussian distribution from which the corresponding predicted depth is sampled. Given visual observations and language instructions as input, the affordance model outputs a pixel-wise heat map that represents affordance regions and the corresponding depth map.

C. Robot Task Planning using Large Language Models

LLMs can be used to provide high-level task planning for performing complex long-horizon robot tasks.

1) Language Instructions for Task Speciﬁcation: As discussed above, SayCan [32] uses an LLM for high-level task planning in language, though with a learned value function to ground these instructions in the environment.

Temporal logic is useful for imposing temporal speciﬁcations in robotic systems. In [39], translation from natural language (NL) to temporal logic (TL) is proposed. A dataset with 28k NL-TL pairs is created and the T5 [126] model is ﬁnetuned using the dataset. LLMs are often used to plan task sub-goals. This work considers robot navigation tasks. In [40], instead of direct task planning, a few-shot translation from a natural language task description to an intermediary task representation is performed. This representation is used by a Task and Motion Planning (TAMP) algorithm to jointly optimize task and motion plans. Autoregressive re-prompting is used to correct synthetic and semantic errors. This work also considers robot navigation tasks.

2) Code Generation using Language Models for Task Planning: Classical task planning requires extensive domain knowledge and the search space is large [127], [128]. LLMs can be used to generate sequences of tasks required to achieve a high-level task. In ProgPrompt [41], the authors introduce a prompting method that uses LLMs to generate sequences of actions directly with no additional domain knowledge. The

prompt to the LLM includes speciﬁcations of the available actions, objects in the environment, and example programs that can be executed. VirtualHome [129] is used as a simulator for demonstration.

Code-as-Policies [42] explores the use of code-writing LLMs to generate robot policy code based on natural language commands. This work considers robot manipulation and navigation tasks using a real-world mobile manipulator robot from Everyday Robots. The study demonstrates that LLMs can be repurposed to write policy code by expressing functions or feedback loops that process perception outputs and invoke control primitive APIs. To achieve this, the authors utilize fewshot prompting, where example language commands formatted as comments are provided along with the corresponding policy code. Without any additional training on this data, they enable the models to autonomously compose API calls and generate new policy code when given new commands. The approach leverages classic logic structures and references third-party libraries like NumPy and Shapely to perform arithmetic operations. By chaining these structures and using contextual information (behavioral commonsense), the LLMs can generate robot policies that exhibit spatial-geometric reasoning, generalize to new instructions, and provide precise values (e.g., velocities) for ambiguous descriptions such as “faster.” The concept of “code as policies” formalizes the generation of robot policies using language model-generated programs (LMPs). These policies can represent reactive policies like impedance controllers, as well as waypoint-based policies such as vision-based pick and place or trajectory-based control. The effectiveness of this approach is demonstrated on multiple real robot platforms. A crucial aspect of this approach is the hierarchical code generation process, which involves recursively deﬁning undeﬁned functions. This enables the LLMs to generate more complex code structures to fulﬁll the desired policy requirements.

In [43], the authors provide design principles for using ChatGPT in robotics and demonstrate how LLMs can help robotic capabilities rapidly generalize to different form factors. This work considers robot manipulation and aerial navigation tasks. First, a high-level robot function library that maps to multiple atomic tasks executable by the robot is deﬁned. Then, a prompt is crafted that includes these functions, and the required constraints along the task description. ChatGPT then provides executable code speciﬁc to the given robot conﬁguration and task. The generated code can then be evaluated by a

- user and appropriate feedback with modiﬁed prompts to LLMs further help reﬁne and generate programs that are safe and deployable on the physical robot. The study demonstrated that such a methodology can be applied to multiple form factors both in simulation and in the real world.

D. In-context Learning (ICL) for Decision-Making

In-context Learning (ICL) [130] operates without the need for parameter optimization, relying instead on a set of examples included in the prompt (the concept of prompting). This learning approach is intimately linked with prompt engineering and ﬁnds extensive use in natural language processing. The

method of Chain-of-Thought [131] is a prominent technique within in-context learning. It involves executing a sequence of intermediate steps to arrive at the ﬁnal solution for complex, multi-step problems. This technique allows models to produce step-by-step explanations that parallel human cognitive processes. However, despite its numerous beneﬁts, ICL also faces certain challenges, including issues related to ambiguity and interpretation, domain-speciﬁc knowledge, transparency, and explainability. In-context learning has had a signiﬁcant impact on the ﬁeld of LLMs in a broad sense, and many robotics works have used it to apply LLMs to speciﬁc domains. Investigating this, Mirchandani and colleagues [132] illustrate that Large Language Models (LLMs) possess remarkable pattern recognition abilities. They reveal that, through in-context learning, LLMs can effectively handle general patterns that extend beyond standard language-based prompts. This capability allows for the application of LLMs in scenarios such as ofﬂine trajectory optimization and online, in-context reinforcement learning. Additionally, Jia and the team in their work on Chain-of-Thought Predictive Control [133] suggest a method to identify speciﬁc brief sequences within demonstrations, termed as ’chain-of-thought’. They focus on understanding and representing the hierarchical structure of these sequences, highlighting the achievement of subgoals within tasks. This work considers robot policy learning from demonstrations for contact-rich object manipulation tasks.

E. Robot Transformers

Foundation models can be used for end-to-end control of robots by providing an integrated framework that combines perception, decision-making, and action generation.

Xiao et al. [48] demonstrate the effectiveness of selfsupervised visual pretraining using real-world images for learning motor control tasks directly from pixel inputs. This work is focused on robot manipulation tasks. They show that without any task-speciﬁc ﬁne-tuning of the pretrained encoder, the visual representations can be utilized for various motor control tasks. This approach highlights the potential of leveraging self-supervised learning from real-world images to acquire general visual representations that can be applied across different motor control tasks. Similarly, Radosavovic et al. [49] investigate the use of self-supervised visual pretraining on diverse, in-the-wild videos for real-world robotic tasks. This work considers robot manipulation tasks. They ﬁnd that the pretrained representations obtained from such videos are effective in a range of real-world robotic tasks, considering different robotic embodiments. This suggests that the learned visual representations generalize well across various tasks and robot platforms, demonstrating the broad applicability of self-supervised pretraining for real-world robotic applications. Both studies emphasize the advantages of self-supervised visual pretraining, where models are trained on large amounts of unlabeled data to learn useful visual representations. By leveraging real-world images and videos, these approaches enable learning from diverse and unstructured visual data, leading to more robust and transferable representations for motor control tasks in robotic systems.

Another example of a Transformer-based policy model is the work on Robotics Transformer (RT-1) [44], where the authors demonstrate a model that shows promising scalability properties. To train the model, the authors use a large dataset of over 130k real-world robotic experiences, comprising more than 700 tasks, that was collected over 17 months using a ﬂeet of 13 robots. RT-1 receives images and natural language instructions as inputs and outputs discretized base and arm actions. It can generalize to new tasks, maintain robustness in changing environments, and execute long-horizon instructions. The authors also demonstrate the model’s capability to effectively absorb data from diverse domains, including simulations and different robots.

The follow-up work, called Robotic Transformer 2 (RT2) [45], demonstrates a vision-language-action (VLA) model that takes a step further by learning from both web and robotics data. The model effectively utilizes this data to generate generalized actions for robotic control. To do so, the authors use pre-existing vision-language models and directly co-ﬁnetune them on robot trajectories resulting in a single model that operates as a language model, a vision-language model, and a robot policy. To make co-ﬁne-tuning possible, the actions are represented as simple text strings which are then tokenized using an LLM tokenizer into text tokens. The resulting model, RT-2, enables vision-language models to output low-level closed-loop control. Similarly to RT-1, actions are produced based on robot instructions paired with camera observations and the action space includes 6-DoF positional and rotational displacement of the robot end-effector, gripper extension, and episode termination command. Via extensive experiments, the authors show that utilizing VLMs aids in the enhancement of generalization across visual and semantic concepts and enables the robots to respond to the so-called chain of thought prompting, where the agent performs more complex, multistage semantic reasoning. Both RT-1 and RT-2 consider robot manipulation and navigation tasks using a real-world mobile manipulator robot from Everyday Robots. One key limitation of RT-2 and other related works in robotics is the fact that the range of physical skills exhibited by the robot is limited to the distribution of skills observed within the robot’s data. While one way to approach this limitation is to collect more diverse and dexterous robotic data, there might be other intriguing research directions such as using motion data in human videos, robotic simulations, or other robotic embodiments.

The next work utilizing the Transformer architecture indeed focuses on learning from data that combines multiple robotic embodiments. In RT-X [46], the authors provide a number of datasets in a standardized data format and models to make it possible to explore the possibility of training large cross-embodied robotic models in the context of robotic manipulation. In particular, they assembled a dataset from 22 different robots collected through a collaboration between 21 institutions, demonstrating 527 skills (160266 tasks). With this uniﬁed dataset, RT-X demonstrates that RT-1- and RT-2based models trained on this multi-embodiment, diverse data exhibit positive transfer across robotic domains and improve the capabilities of multiple robots by leveraging experience from other platforms.

Other works have investigated general pretrained transformers for robot control, trained with self-supervised trajectory data from multiple robots. For example, Perception-Action Causal Transformer (PACT) [47] is a generative transformer architecture that builds representations from robot data with self-supervision. This work considers robot navigation tasks. PACT pretrains a representation useful for multiple tasks on a given robot. Similar to how large language models learn from extensive text data, PACT is trained on abundant safe state-action data (trajectories) from a robot, learning to predict appropriate safe actions. By predicting states and actions over time in an autoregressive manner, the model implicitly captures dynamics and behaviors speciﬁc to a robot. PACT was tested in experiments involving mobile agents: a wheeled robot with a LiDAR sensor (MuSHR) and a simulated agent using ﬁrst-person RGB images (Habitat). The results show that this robot-speciﬁc representation can serve as a starting point for tasks like safe navigation, localization, and mapping. Additionally, the experiments demonstrated that ﬁne-tuning smaller task-speciﬁc networks on the pre-trained model leads to signiﬁcantly better performance compared to training a single model from scratch for all tasks simultaneously, and comparable performance to training a separate large model for each task independently.

Another work in this space is Self-supervised Multi-task pretrAining with contRol Transformer (SMART) [134], which introduces a self-supervised multi-task pertaining to control transformers, providing a pretraining-ﬁnetuning approach tailored for sequential decision-making tasks. During the pretraining phase, SMART captures information essential for both short-term and long-term control, facilitating transferability across various tasks. Subsequently, the ﬁnetuning process can adapt to a wide variety of tasks spanning diverse domains. Experimentation underscores SMART’s ability to enhance learning efﬁciency across tasks and domains. This work considers cart pole-swing-up, cart pole-balance, hopper-hop, hopperstand, cheetah-run, walker-stand walker-run, and walker-walk tasks. The approach demonstrates robustness against distribution shifts and proves effective with low-quality pretraining datasets.

Some works have investigated transformer models in conjunction with classical planning and control layers as part of a modular robot control architecture. For example, in [50], a multi-modal transformer (LATTE) is presented that allows a user to reshape robot trajectories using language instructions. This work considers both robot manipulation and navigation tasks. LATTE transformer takes as input geometrical features of an initial trajectory guess along with the obstacle map conﬁguration, language instructions from a user, and images of each object in the environment. The model’s output is modiﬁed for each waypoint in the trajectory so that the ﬁnal robot motion can adhere to the user’s language instructions. The initial trajectory plan can be generated using any geometric planner such as A∗, RRT∗, or model predictive control. Subsequently, this plan is enriched with the semantic objectives within the model. LATTE leverages pretrained language and visual-language models to harness semantic representations of the world.

F. Open-Vocabulary Robot Navigation and Manipulation

1) Open-Vocabulary Navigation: Open-vocabulary navigation addresses the challenge of navigating through unseen environments. The open-vocabulary capability signiﬁes that the robot possesses the capacity to comprehend and respond to language cues, instructions, or semantic information, without being restricted to a predeﬁned dataset. In this section, we explore papers that examine the integration of LLMs, VLMs, or a combination of both in a plug-and-play manner for robot navigation tasks. Additionally, we discuss papers that take a different approach by constructing foundation models explicitly tailored for robot navigation tasks.

In VLN-BERT [135], the authors present a visual-linguistic transformer-based model that leverages multi-modal visual and language representations for visual navigation using web data. The model is designed to score the compatibility between an instruction, such as “...stop at the brown sofa,” and a sequence of panoramic RGB images captured by the agent.

Similarly, LM-Nav [136] considers visual navigation tasks. LM-Nav is a system that utilizes pretrained models of images and language to provide a textual interface to visual navigation. LM-Nav demonstrates visual navigation in a real-world outdoor environment from natural language instructions. LMNav utilizes an LLM (GPT-3 [2]), a VLM (CLIP [4]), and a VNM (Visual Navigation Model). First, LM-Nav constructs a topological graph of the environment via the VNM estimating the distance between images. The LLM is then used to translate the natural instructions to sequences of intermediate language landmarks. The VLM is used to ground the visual observations in landmark descriptions via a joint probability distribution over landmarks and images. Using the VLM’s probability distribution, the LLM instructions, and the VNM’s graph connectivity, the optimal path is planned using the search algorithm. Then the plan is executed by the goalconditioned policy of VNM.

While LM-Nav makes use of LLMs and VLMs as plug-andplay for visual navigation tasks, the authors of ViNT [137] propose to build a foundation model for visual navigation tasks. ViNT is an image goal-conditioned navigation policy trained on diverse training data and can control different robots in zero-shot. It can be ﬁne-tuned to be adapted for different robotic platforms and various downstream tasks. ViNT is trained on various navigation datasets from different robotic platforms. It is trained with goal-reaching objectives and utilizes a Transformer-based architecture to learn navigational affordances. ViNT encodes visual observations and visual goals using an EfﬁcientNet CNN and predicts temporal distance and normalized actions in an embodiment-agnostic manner. Additionally, ViNT can be augmented with diffusionbased sub-goal proposals to help explore environments not encountered during training. An image-to-image diffusion generates sub-goal images, which the ViNT then navigates toward while building a topological map in the background.

Another work that considers zero-shot navigation tasks is Audio Visual Language Maps (AVLMaps) [138]. AVLMaps presents a 3D spatial map representation for cross-modal information from audio, visual, and language cues. AVLMaps re-

ceives multi-modal prompts and performs zero-shot navigation tasks in the real world. The inputs are depth and RGB images, camera pose, and audio. Visual features are encoded using pretrained foundation models. Visual localization features (using NetVLAD [139], SuperPoint [140]), visual-language features (using LSeg [58]), and audio-language features (using AudioCLIP [141]) are computed and predictions from different modalities are combined into 3D heatmaps. The pixel-wise joint probability of the heatmap is computed and used for planning. Additionally, navigation policies are generated as executable codes with the help of GPT-3. Finally, 3D heatmaps are predicted indicating the location of multimodal concepts such as objects, sounds, and images.

Many roboticists may wonder about the comparative strengths of classical modular robot navigation systems versus end-to-end learned systems. Semantic navigation [142] seeks to address this question by presenting an empirical analysis of semantic visual navigation methods. The study compares representative approaches from classical, modular, and endto-end learning paradigms across six different homes, without any prior knowledge, maps, or instrumentation. The ﬁndings of the study reveal that modular learning methods perform well in real-world scenarios. In contrast, the end-to-end learning approaches face challenges due to a signiﬁcant domain gap between simulated and real-world images. This domain gap hinders the effectiveness of end-to-end learning methods in real-world navigation tasks. For practitioners, the study emphasizes that modular learning is a reliable approach to object navigation. The modularity and abstraction in policy design enable successful transfer from simulation to reality, making modular learning an effective choice for practical implementations. For researchers, the study also highlights two critical issues that limit the reliability of current simulators as evaluation benchmarks. Firstly, there exists a substantial Simto-Real gap in images, which hampers the transferability of learned policies from simulation to the real world. Secondly, there is a disconnect between simulation and real-world error modes, which further complicates the evaluation process.

Another line of work in open-vocabulary navigation is object navigation tasks. In this task, the robot must be able to ﬁnd the object described by humans and navigate towards the object. The navigation task is decomposed into exploration when the language target is not detected and exploitation when the target is detected and the robot navigates toward the target. As the robot moves in the environment, it creates a top-down map using RGB-D observations and poses estimates. In [143], the authors introduce a zero-shot object navigation setting that uses an open-vocabulary classiﬁer such as CLIP [4] to compute the cosine similarity between an image and a user-speciﬁed description.

Common datasets and benchmarks for these types of problems are Matterport3D [144], [145], Gibson [146] and Habitat [147]. L3MVN [148] enhances visual target navigation by constructing an environment map and selecting long-term goals using the inference capabilities of large language models. The system can determine appropriate long-term goals for navigation by leveraging pretrained language models such as RoBERTa-large [149], enabling efﬁcient exploration and

searching. Chen et al. [150] presents a training-free and modular system for object goal navigation, which constructs a structured scene representation through active exploration. The system utilizes semantic information in the scene graphs to deduce the location of the target object and integrates semantics with the geometric frontiers to enable the agent to navigate effectively to the most promising areas for object search while avoiding detours in unfamiliar environments. HomeRobot [151] introduces a benchmark for the OpenVocabulary Mobile Manipulation (OVMM) task. OVMM task is the problem of ﬁnding an object in any unseen environment, navigating towards the object, picking it up, and navigating towards a goal location to place the object. HomeRobot provides a benchmark in simulation and the real world for OVMM tasks.

2) Open-Vocabulary Manipulation: Open-vocabulary manipulation refers to the problem of manipulating any object in a previously unseen environment. VisuoMotor Attention Agent (VIMA) [152] learns robot manipulation from multi-modal prompts. VIMA is a transformer-based agent that predicts motor commands conditioned on a task prompt and a history of interactions. VIMA It introduces a new form of task speciﬁcations that combines textual and visual tokens. Multi-modal prompting converts different robot manipulation tasks, such as visual goal-reaching, learning from visual demonstrations, and novel concept grounding into one sequence modeling problem. It offers the training of a uniﬁed policy across diverse tasks, potentially allowing for zero-shot generalization to previously unseen ones. VIMA-BENCH is introduced as a benchmark for multi-modal robot learning. The VIMA-BENCH simulator supports collections of objects and textures that can be utilized in multi-modal prompting. RoboCat [153] is a self-improving AI agent. It uses a 1.18B-parameter decoder-only transformer. It learns to operate different robotic arms, solves tasks from as few as 100 demonstrations, and improves from self-generated data. RoboCat is based on Gato [154] architecture and is trained with a self-improvement cycle.

For robots to operate effectively in the real world they must be able to manipulate previously unseen objects. Liu et al. present StructDiffusion [155], which seeks to enable robots to use partial viewpoint clouds and natural language instructions to construct a goal conﬁguration for objects that were previously seen or unseen. They accomplish this by ﬁrst using segmentation to break up the scene into objects. Then they use a multi-model transformer to combine word and point cloud embeddings and output a 6-DoF goal pose prediction. The predictions are iteratively reﬁned via diffusion and a discriminator that is trained to determine if a sampled conﬁguration is feasible. Manipulation of Open-World Objects (MOO) [156] leverages a pretrained vision-language model to extract object-centric information from the language command and the image and conditions the robot policy on the current image, the instructions, and the extracted object information in a form of a single-pixel overlaid onto the image. MOO

- uses Owl-ViT for object detection and RT-1 for languageconditioned policy learning.

Another task in robot manipulation involves autonomous scene rearrangement and in-painting. DALL-E-Bot [157] per-

forms zero-shot autonomous rearrangement in the scene in a human-like way using pretrained image diffusion model DALL-E2 [88]. DALL-E-Bot autonomous object rearrangement does not require any further data collection or training. First, the initial observation image (of the disorganized scene) is converted into a per-object representation including a segmentation mask using Mask R-CNN [158], an object caption, and a CLIP visual feature vector. Then a text prompt is generated by describing the object in the scene and is given to DALL-E to create a goal image for the rearrangement task (the objects should be rearranged in a human-like way). Next, the objects in the initial and generated images are matched using their CLIP visual features. Poses are estimated by aligning their segmentation masks. The robot rearranges the scene based on the estimated poses to create the generated arrangement.

In Table II some robotic-speciﬁc foundation models are reported along with information about their size and architecture, pretrained task, inference time, and hardware setup.

IV. PERCEPTION

Robots interacting with their surrounding environments receive raw sensory information in different modalities such as images, video, audio, and language. This high-dimensional data is crucial for robots to understand, reason, and interact in their environments. Foundation models, including those that have been developed in the vision and NLP domains, are promising tools for converting these high-dimensional inputs into abstract, structured representations that can be more easily interpreted and manipulated. Particularly, multimodal foundation models enable robots to integrate different sensory inputs into a uniﬁed representation encompassing semantic, spatial, temporal, and affordance information. These multi-modal models reﬂect cross-modal interactions, often by aligning elements across modalities to ensure coherence and correspondence. For example, text and image data are aligned for image captioning tasks. This section will explore a range of tasks related to robot perception that are improved through aligning modalities using foundation models, with a focus on vision and language. There is an extensive body of literature studying multi-modality in the machine learning community, and an interested reader is referred to the survey paper [161] that presents a taxonomy of multi-modal learning. We focus on applications of multi-modal models to robotics.

A. Open-Vocabulary Object Detection and 3D Classiﬁcation

1) Object Detection: Zero-shot object detection allows robots to identify and locate objects they have never encountered previously. Grounded Language-Image Pre-training (GLIP) [52] integrates object detection and grounding by redeﬁning object detection as phrase grounding. This reformulation enables the learning of a visual representation that is both language-aware and semantically rich at the object level. In this framework, the input to the detection model comprises not only an image but also a text prompt that describes all the potential categories for the detection task. To train GLIP, a dataset of 27 million grounding instances was compiled,

TABLE II PRETRAINED MODELS FOR ROBOTICS

[Figure 39]

Hardware *

Paper Backbone Size (Parameters)

Pretrained Task Inference Speed

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

RoboCat [153] decoder-only transformer 1.18B manipulation 10-20Hz Gato [154] decoder-only transformer 1.2B generalist agent 20Hz 4 days on 16x16 TPU v3 slice PaLM-E-562B [6] decoder-only transformer 562B 1Hz for Language

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

5-6Hz runs on multi-TPU cloud service

subgoals + 5Hz low-level control policies

ViNT [137] EfﬁcientNet+ decoder transformer 31M visual navigation 4Hz variety of GPU conﬁgurations is used including 2×4090, 3×Titan Xp, 4×P100, 8×1080Ti, 8×V100, and 8×A100

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

VPT [72] a temporal convolution layer, a ResNet 62 image processing stack, and residual unmasked attention layers,

0.5B embodied agent in Minecraft

20Hz 9 days on 720 V100 GPUs

- RT-1 [44] Conditioned EfﬁcientNet + TokenLearner + decoder-only transformer

35M real-world robotics tasks

3Hz

[Figure 59]

- RT-2 [45] PaLI-X 55B real-world robotics tasks

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

1-3Hz runs on multi-TPU cloud service RT-2-X [46] ViT and Language model UL2

[Figure 64]

[159]

55B real-world robotics tasks

1-3Hz runs on multi-TPU cloud service

[Figure 65]

LIV [37] CLIP reward learning 15Hz 8 NVIDIA V100 GPUs SMART [134] decoder-only transformer 11M bidirectional

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

dynamics prediction and masked hindsight control

1 Hz 8 Nvidia V100 GPUs

[Figure 71]

COMPASS [160] 3D-Resnet encoder 20M Contrastive loss 30 Hz 8 Nvidia V100 GPUs PACT [47] decoder-only transformer 12M forward dynamics

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

and next action prediction

10 Hz (edge) / 50 Hz

Nvidia Xavier NX (edge) / 8 Nvidia V100 GPUs

[Figure 77]

* Empty ﬁelds in the table denote no data is reported.

consisting of 3 million human-annotated pairs and 24 million image-text pairs obtained by web crawling. The results of the study demonstrate the remarkable zero-shot and few-shot transferability of GLIP to a wide range of object-level recognition tasks. Recently, PartSLIP [162] demonstrated that GLIP can be used for low-shot part segmentation on 3D objects. PartSLIP renders a 3D point cloud of an object from multiple views and combines 2D bounding boxes in these views to detect object parts. To deal with noisy 2D bounding boxes from different views, PartSLIP runs a voting and grouping method on super points from 3D, assigns multi-view 2D labels to super points, and ﬁnally groups super points to obtain a precise part segmentation. To enable few-shot learning of

- 3D part segmentation, prompt tuning, and multi-view feature aggregation are proposed to improve performance.

set detection by fusing vision and language. Grounding DINO outperforms GLIP in open-set object detection. This superior performance is mainly due to the transformer architecture of Grounding DINO, which facilitates multi-modal feature fusion at multiple stages.

2) 3D Classiﬁcation: Zero-shot 3D classiﬁers can enable robots to classify objects in their environments without explicit training data. Foundation models are strong candidates for performing 3D classiﬁcation. PointCLIP [54] transfers CLIP’s pre-trained knowledge of 2D images to 3D point cloud understanding by aligning point clouds with text. The authors propose to project each point onto a series of pre-deﬁned image planes to generate depth maps. Then, the CLIP visual encoder is used to encode multi-view features of the point cloud and predict labels in natural language for each view. The ﬁnal prediction for the point cloud is computed via weighted aggregation of the predictions for each view. PointBERT [55] uses a transformer-based architecture to extract features from point clouds, generalizing the concept of BERT into 3D point clouds.

OWL-ViT [51] is an open-vocabulary object detector. OWLViT uses a vision transformer architecture with contrastive image-text pre-training and detection end-to-end ﬁne-tuning. Unlike GLIP, which frames detection as a phrase grounding problem with a single text query and limits the number of possible object categories, OWL-ViT can handle multiple textbased or image-driven queries. OWL-ViT has been applied to robot learning for example in VoxPoser [34] as the openvocabulary object detector to ﬁnd “entities of interest” (e.g., vase or drawer handles) and ultimately deﬁne value maps for optimizing manipulation trajectories.

Unlike PointCLIP which converts the task of matching point clouds and text to image-text alignment, ULIP [56], [57] is a Uniﬁed representation of Language, Images, and Point clouds for 3D understanding. It achieves this by pre-training with object triplets (image, text, point cloud). The model is trained using a small number of automatically synthesized triplets from ShapeNet55 [163], which is a large-scale 3D model repository. ULIP uses CLIP as the vision-language

Grounding DINO [53] combines DINO [105] with grounded pre-training, extending the closed-set DINO model to open-

model. During pretraining, the CLIP model is kept frozen and a 3D encoder is trained by aligning the 3D features of an object with its associated textual and visual features from CLIP using contrastive learning. The pretraining process allows ULIP to learn a joint embedding space where the three modalities are aligned. One of the major advantages of ULIP is that it can substantially improve the recognition ability of 3D backbone models. This is because the pretraining process allows ULIP to learn more robust and discriminative features for each modality, which can then be used to improve the performance of 3D models. Another advantage of ULIP is that it is agnostic to the 3D model architecture, and thus can be easily integrated into the pretraining process of existing 3D pipelines. ULIP adopts masked language modeling from BERT to 3D by tokenizing 3D patches randomly masking out 3D tokens and predicting them back during pretraining. ULIP [56], [57] has shown that the performance of recognition capability of models such as PointBERT can be improved by using a uniﬁed multimodal representation of ULIP.

B. Open-Vocabulary Semantic Segmentation

Semantic segmentation classiﬁes each pixel in an image into semantic classes. This provides ﬁne-grained information about object boundaries and locations within an image and enables embodied agents to understand and interact with the environment at a more granular level. Several works explore how foundation models such as CLIP can enhance the generalizability and ﬂexibility of semantic segmentation tasks.

LSeg is a language-driven semantic segmentation model [58] that associates semantically similar labels to similar regions in an embedding space. LSeg uses a text encoder based on the CLIP architecture to compute text embeddings and an image encoder with the underlying architecture of Dense Prediction Transformer (DPT) [164]. Similar to CLIP, LSeg creates a joint embedding space using text and image embeddings. LSeg freezes the text encoder at training time and trains the image encoder to maximize the correlation between the text embedding and the image pixel embedding of the ground-truth pixel class. It allows users to arbitrarily shrink, expand, or rearrange the label set (with unseen categories) for any image at test time.

Segment Anything Model (SAM) [59] introduces a framework for promptable segmentation consisting of the task deﬁnition for promptable segmentation, a segmentation foundation model (the Segment Anything Model, or SAM), and a data engine. SAM adapts a pretrained Vision Transformer from Masked Auto-Encoder (MAE) [113] as an image encoder while using a text encoder from CLIP [114] for sparse prompts (points, boxes, and text) and a separate dense prompt encoder for masks. In contrast to other foundation models that are trained in an unsupervised manner on web-scale data, SAM is trained using supervised learning with data engines that help scale the number of available annotations. Along with the model, the authors released the Segment Anything 1 Billion (SA-1B) dataset. It consists of 11M images and 1.1B segmentation masks. In this work, the authors conducted experiments on ﬁve zero-shot transfer tasks, including point-valid

mask evaluation, edge detection, object proposal, instance segmentation, and text-to-mask. The system’s composable design, facilitated by prompt engineering techniques, enables a broader range of applications compared to systems trained speciﬁcally for ﬁxed task sets. However, one limitation of this work that is particularly relevant to robotic applications is that SAM cannot run in real-time.

FastSAM [60] and MobileSAM [61] achieve comparable performance to SAM at faster inference speeds. The Track Anything Model (TAM) [62] combines SAM and XMem [165], an advanced video object segmentation (VOS) model, to achieve interactive video object tracking and segmentation. Anything-3D [166] employs a collection of visuallanguage models and SAMs to elevate objects into the realm of 3D. It uses BLIP [108] to generate textual descriptions while using SAM to extract objects of interest from visual input. Then, Anything-3D lifts the extracted objects into a Neural Radiance Field (NeRF) [167] representation using a text-to-image diffusion model, enabling their integration into 3D scenes.

Amidst these remarkable advancements, achieving ﬁnegrained detection with real-time performance still remains challenging. For example, LSeg [58] reports failure cases related to misclassiﬁcation, when the test time input labels do not include the true label for the pixel, and the model thus assigns the highest probability to the closest label. Another failure case occurs when multiple labels can be correct for a particular pixel, and the model must classify it as just one of the categories. For example “window” and “house” may both be deﬁned as labels, but during inference, a pixel representing a “window” may be labeled instead as “house”. SAM also does not provide precise segmentation for ﬁne structures and often fails to produce crisp boundaries. All models that use SAM as a sub-component may encounter similar limitations. In the future, ﬁne-grained semantic segmentation models that can assign multiple labels to a pixel when there are multiple correct descriptions should be considered. Additionally, developing models that can run in real-time will be critical for robotics applications.

C. Open-Vocabulary 3D Scene and Object Representations

Scene representations allow robots to understand their surroundings, facilitate spatial reasoning, and provide contextual awareness. Language-driven scene representations align textual descriptions with visual scenes, enabling robots to associate words with objects, locations, and relationships. In this section, we study recent works that use foundation models to enhance scene representations.

1) Language Grounding in 3D Scene: Language grounding refers to combining geometric and semantic representations of an environment. One type of representation that can provide an agent with a strong geometric prior is an implicit representation. One example of an implicit representation is a Neural Radiance Field (NeRF) [167]–[169]. NeRF creates high-quality 3D reconstructions of scenes and objects from a set of 2D images captured from different viewpoints (without the need for explicit depth information). The NeRF neural

network takes camera poses as input and predicts the 3D geometry of the scene as well as color and intensity. Most NeRF-based models memorize the light ﬁeld in a single environment and are not pre-trained on a large data set, hence they are not foundation models. However, foundation models such as CLIP can be combined with NeRFs to extract semantic information from an agent’s environment.

Kerr et al. [64] propose language-embedded radiance ﬁelds (LERFs) that ground CLIP embeddings into a dense multiscale 3D ﬁeld. This results in a 3D representation of the environment that can be queried to produce semantic relevancy maps. The LERF model takes 3D position (x,y,z), viewing direction (φ, θ), and a scaling factor as input and outputs an RGB value, density (σ), as well as DINO [105] and CLIP features. The LERF is optimized in two stages: initially, a multi-scale feature pyramid of CLIP embeddings over training views is computed; then, the pyramid is interpolated using the image scale and pixel location to obtain the CLIP embedding; and ﬁnally, the CLIP embeddings are supervised through cosine similarity and the RGB and density are supervised using the standard mean squared-error.

Models such as LERF inherit the shortcomings of CLIP and NeRF. For example, CLIP exhibits difﬁculty in capturing spatial relationships between objects. In addition, language queries from CLIP can highlight a signiﬁcant issue similar to the bag-of-words model, which struggles to distinguish terms with opposite sentiments. Also, NeRF relies on known camera poses associated with pre-captured multi-view images.

In CLIP-Fields [170], an implicit scene representation g(x,y,z) : R3 → Rd is trained by decoding a d-dimensional latent vector to different modality-speciﬁc outputs. The model distills information from pretrained image models by backprojecting the pixel labels to 3D space and training the output heads to predict semantic labels from an open-vocab object detector called Detic, the CLIP visual representation, and one-hot instance labels using a contrastive loss. The scene representation can then be used as a spatial database for segmentation, instance identiﬁcation, semantic search over space, and 3D view localization from images.

Another related work is VLMaps [171], which projects pixel embeddings from LSeg to grid cells in a top-down grid map. This method does not require training and instead directly backprojects pixel embeddings to grid cells and averages the values in overlapping regions. By combining a VLMap with a code-writing LLM, the authors demonstrate spatial goal navigation using landmarks (e.g., move to the plant) or spatial references with respect to landmarks (between the keyboard and the bowl). Semantic Abstraction (SemAbs) [172] presents another approach for 3D scene understanding by decoupling visual-semantic reasoning and 3D reasoning. In SemAbs, given an RGB-D image of a scene, a semantic-aware 2D VLM extracts 2D relevancy maps for each queried object while semantic-abstracted 3D modules predict the 3D occupancy of each object using the relevancy maps. Because the 3D modules are trained irrespective of the speciﬁc object labels, the system demonstrates strong generalization capabilities, including generalization to new object categories and from simulation to the real world.

Current VLMs can reason about 2D images, however, they are not grounded in the 3D world. The main challenge for building 3D VLM foundation models is the scarcity of 3D data. Particularly, 3D data paired with language description is scarce. One strategy to circumvent this issue is to take advantage of 2D models trained on large-scale data to supervise 3D models. For instance, the authors of FeatureNeRF [173] propose to learn 3D semantic representations by distilling 2D vision foundation models (i.e., DINO or Latent Diffusion) into 3D space via neural rendering. FeatureNeRF predicts a continuous 3D semantic feature volume from a single or few images which can be used for downstream tasks such as keypoint transfer or object part co-segmentation.

In 3D-LLM [11], the authors propose to use 2D VLMs as backbones to train a 3D-LLM that can take 3D representations (i.e., 3D point clouds with their features) as inputs and accomplish a series of diverse 3D-related tasks. The 3D features are extracted from 2D multi-view images and mapped to the feature space of 2D pretrained VLMs. To overcome 3D data scarcity, the authors propose an efﬁcient prompting procedure for ChatGPT to generate 3D-language data encompassing a diverse set of tasks. These tasks include 3D captioning, dense captioning, 3D question answering, 3D task decomposition, 3D grounding, 3D-assisted dialog, and navigation. Also, to capture 3D spatial information, the authors propose a 3D localization mechanism by 1) augmenting 3D features with position embedding and 2) augmenting LLM vocabularies with 3D location tokens. In the ﬁrst part, the position embeddings of the three dimensions are generated and concatenated with 3D features. In the second part, the coordinates of the bounding box representing the grounded region are discretized to voxel integers as location tokens < xmin,ymin,zmin,xmax,ymax,zmax >. It is important to highlight that, typically, creating 3D representations necessitates the use of 2D multi-view images and camera matrices. These resources are not as readily available as the vast amounts of internet-scale text and image data that current foundation models are trained on.

2) Scene Editing: When an embodied agent relies on an implicit representation of the world, the capability to edit and update this representation enhances the robot’s adaptability. For instance, consider a scenario where a robot utilizes a pretrained NeRF model of an environment for navigation and manipulation. If a portion of the environment changes, being able to adjust the NeRF without retraining the model from scratch saves time and resources.

In the case of NeRFs, Wang et al. [63] propose a text and image-driven method for manipulating NeRFs called CLIPNeRF. This approach uses CLIP to disentangle the dependence between shape and appearance in conditional neural radiance ﬁelds. CLIP-NeRF facilitates the editing of the shape and appearance of NeRFs using either image or text prompts. It is composed of two modules: the disentangled conditional NeRF and CLIP-driven manipulation. The former takes the positional encoding γ(x,y,z), a shape code zs, viewing direction v(φ,θ), and appearance code za as an input and outputs color and density. The disentanglement is achieved using a deformation network that is appended as input to the traditional NeRF MLP

that produces density, and by taking the output from this MLP and concatenating it with an appearance code to attain the color value. The CLIP-driven manipulation module takes an image example or text prompt as an input and outputs a shape deformation ∆zs and an appearance deformation ∆za from shape mapping and appearance mapping MLPs respectively. These deformation values aim to perturb the shape code and appearance code in the disentangled conditional NeRF module to produce the desired output.

- A key limitation of the CLIP-NeRF approach is that prompt-

ing can impact the entire scene rather than a selected region. For example, prompting to change the color of a ﬂower’s petals might also impact the shape and color of its leaves. To address this limitation, Kobayashi et al. propose to train distilled feature ﬁelds (DFFs) [65] and then manipulate DFFs through query-based scene decomposition and editing. Pre-trained 2D VLMs (such as LSeg [58] and DINO [105]) are employed as teacher networks and distilled into 3D distilled feature ﬁelds via volume rendering. Editing is achieved by alpha compositing the density and color values of the two NeRF scenes. When combined with CLIP-NeRF, this method enables CLIP-NeRF to selectively edit speciﬁc regions of multi-object scenes. A similar approach was explored by Tschernezki et al. in [174] where the authors show that enforcing the 3D consistency of features in the NeRF embedding improved segmentation performance compared to using features from the original 2D images.

Another approach to more controlled 3D scene editing is to use structured 3D scene representations. Nerﬂets [175] represent a 3D scene as a combination of local neural radiance ﬁelds where each maintains its own spatial position, orientation, and dimension. Instead of employing a single large MLP to predict colors and densities as standard NeRF, individual Nerﬂets are combined to predict these values, modulated by their weights. After optimizing posed 2D images and segmentations, Nerﬂets reﬂect the decomposed scene and support more controlled editing.

One application of image editing in robotics is for data augmentation during policy learning. ROSIE [176] use the Imagen editor [177] to modify training images to add additional distractors and unseen objects and backgrounds to train robust imitation learning policies. GenAug [178] similarly generates images with in-category and cross-category object substitutions, visual distractors, and diverse backgrounds. The CACTI [14] pipeline includes a step in-painting different plausible objects via Stable-Diffusion [117] onto training images. These approaches generate photorealistic images for training robust policies; however, generating images with sufﬁcient diversity while also maintaining physical realism, e.g. for object contacts, remains a challenge. Existing approaches use learned or provided masks to specify areas of the image to keep, or heuristics based on the particular robotic task.

Another direction is to use generative models to deﬁne goal images for planning. DALL-E-Bot [157] uses DALL-E

- 2 to deﬁne a goal image of human-like arrangements from observations.
- 3) Object Representations: Learning correspondences be-

tween objects can facilitate manipulation by enabling skill

transfer from trained objects to novel object instances in known categories or novel object categories at test time. Traditionally, object correspondences have been learned using strong supervision such as keypoints and keyframes. Neural descriptor ﬁelds (NDFs) [179] remove the need for dense annotation by leveraging layer-wise activations from an occupancy network; however, this approach still requires many training shapes for each target object category. Additional works have started to build object representations directly from image features of pretrained vision models.

Feature Fields for Robotic Manipulation (F3RM) [180] builds on DFF to develop scene representations that support ﬁnding corresponding object regions. F3RM uses a similar feature representation for 6-DoF poses relative to objects (e.g., a grasp on the handle of the mug) to NDF. Besides allowing corresponding 6-DoF poses to be found from a few demonstrations, the pose embeddings can also be directly compared to text embeddings from CLIP to leverage language guidance (e.g., pick up the bowl). Correspondences between objects have also been directly extracted from DINO features [181] without training. This method ﬁrst extracts dense ViT feature maps of two objects using multiple views. Similar regions on the two objects are found by computing the cyclical distance metric [182] on the feature maps. With the 2D patch correspondences, a 7-D rigid body transform (i.e., a SO(3) pose, a translation, and a scaling scalar) between the objects can be solved together with RANSAC and Umeyama’s method [183].

D. Learned Affordances

Affordances refer to the potential of objects, environments, or entities to offer speciﬁc functions or interactions to an agent. They can include actions such as pushing, pulling, sitting, or grasping. Detecting affordances bridges the gap between perception and action.

Affordance Diffusion [66] synthesizes complex interactions of e.g. an articulated hand with a given object. Given an RGB image, Affordance Diffusion aims to generate images of human hands for hand-object interaction (HOI). The authors propose a two-step generative approach based on large-scale pretrained diffusion models based on where to interact (layout) and how to interact (content). The layout network generates a 2D spatial arrangement of hand and object. The content network then synthesizes images of a hand grasping the object conditioned on the given object and the sampled HOI layout. Affordance Diffusion outputs both the hand articulation and approach orientation.

Vision-Robotic Bridge (VRB) [67] trains a visual affordance model on internet videos of human behavior. Particularly, it estimates the likely location and manner in which a human interacts within a scene. This model captures the structural information of these behavioral affordances. The authors seamlessly integrate the affordance model with four different robot learning paradigms. Firstly, they apply ofﬂine imitation learning, where the robot learns by imitating the observed human interactions from the videos. Secondly, they use exploration techniques to enable the robot to actively

discover and learn new affordances in its environment. Thirdly, the authors incorporate goal-conditioned learning, allowing the robot to learn how to achieve speciﬁc objectives by leveraging the estimated affordances. Finally, they integrate action parameterization for reinforcement learning, enabling the robot to learn complex behaviors by optimizing its actions based on the estimated affordances.

E. Predictive Models

Predictive dynamics models, or world models, predict how the state of the world changes given particular agent actions, that is, they attempt to model the state transition function of the world [184]. When applied to visual observations, dynamics modeling can be formulated as a video prediction problem [185], [186]. While video generation and prediction, particularly over long horizons, is a longstanding challenge with many prior efforts, recent models based on vision transformers and diffusion models have demonstrated improvements [187], [188]. For instance, the Phenaki model [189] generates variable length video up to minutes in length conditioned on text prompts.

Several approaches apply these models to robotics in the literature. Note that while learned dynamics or world models in robotics have been explored in constrained or smaller-data regimes, we focus in this section on works that train on a diversity or volume of data that is characteristic of foundation models. One strategy is to learn an action-conditioned model that may be used directly for downstream planning by optimizing an action sequence [190], i.e. performing modelpredictive control, or for policy learning via training on simulated rollouts. One example is the GAIA-1 model which generates predictions of driving video conditioned on arbitrary combinations of video, action, and text [191]. It was trained on 4700 hours of proprietary driving data. Another approach is to use a video prediction model to generate a plan of future states, and then learn a separate goal-conditioned policy or inverse dynamics model to infer control actions based on the current and target state. One line of work instantiates this by combining text-conditioned video diffusion models with image-goal-conditioned policies to solve manipulation tasks in simulated and real tabletop settings [192]. This approach has been extended to longer-horizon object manipulation tasks by using the PaLM-E VLM to break down a high-level language goal into smaller substeps, leveraging feedback between the VLM and video generation models [193].

Another example is COMPASS [160], which ﬁrst constructs a comprehensive multimodal graph to capture crucial relational information across diverse modalities. The graph is then used to construct a rich spatio-temporal and semantic representation. Pretrained on the TartanAir multimodal dataset, COMPASS was demonstrated to address multiple robotic tasks including drone navigation, vehicle racing, and visual odometry.

V. EMBODIED AI

Recently, researchers have shown that the the success of LLMs can be extended to embodied AI domains [32], [33],

[42], [194], where “embodied” typically refers to a virtual embodiment in a world simulator, not a physical robot embodiment. Statler [69] is a framework that endows LLMs with an explicit representation of the world state as a form of “memory” that is maintained over time. Statler uses two instances of general LLMs: a world-model reader and a worldmodel writer, that interface with and maintain the world state. Statler improves the ability of existing LLMs to reason over longer time horizons without the constraint of context length.

Large Scale Language Models (LSLMs) have exhibited strong reasoning ability and the ability to adapt to new tasks through in-context learning. Dasgupta et al. [195] combine these complementary abilities in a single system consisting of three parts: a Planner, an Actor, and a Reporter. The Planner is a pretrained language model that can issue commands to a simple embodied agent (the Actor), while the Reporter communicates with the Planner to inform its next command. Mu et al. [70] build EgoCOT, a dataset consisting of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. EmbodiedGPT [70] utilizes preﬁx adapters to augment the 7B language model’s capacity to generate high-quality planning, training it on the EgoCOT dataset to avoid overly divergent language model responses. Comprehensive experiments were conducted, demonstrating that the model effectively enhances the performance of embodied tasks such as Embodied Planning, Embodied Control, Visual Captioning, and Visual Q&A. Embodied agents should autonomously and endlessly explore the environment. They should actively seek new experiences, acquire new skills, and improve themselves.

The game of Minecraft [196] provides a platform for designing intelligent agents capable of operating in the open world. MineDojo [71] is a framework for developing generalist agents in the game of Minecraft. MineDojo offers thousands of open-ended and language-prompted tasks, where the agent can navigate in a progressively generated 3D environment to mine, craft tools, and build structures. As part of this work, the authors introduce MiniCLIP, a video-language model that learns to capture the correlations between a video clip and its time-aligned text that describes the video. The MineCLIP model, trained on YouTube videos, can be used as a reward function to train the agent with reinforcement learning. By maximizing this reward function, it incentivizes the agent to make progress toward solving tasks speciﬁed in natural language.

Voyager [73] introduces an LLM-powered embodied lifelong learning agent in the realm of Minecraft. Voyager uses GPT-4 to continuously explore the environment. It interacts with GPT-4 through in-context prompting and does not require model parameter ﬁne-tuning. Exploration is maximized by querying GPT-4 to provide a stream of new tasks and challenges based on the agent’s history interactions and current situations. Also, the iterative prompting mechanism generates code as the action space to control the Minecraft agent. Iterative prompting incorporates environment feedback provided by Minecraft, execution errors, and a self-veriﬁcation scheme. For self-veriﬁcation, GPT-4 acts as a critic by checking task success and providing suggestions for task completion in

the case of failure. The GPT-4 critic can be replaced by a human critic to provide on-the-ﬂy human feedback during task execution. Ghost in the Minecraft (GITM) [197] leverages LLM to break down goals into sub-goals and map them to structured actions for generating control signals. GITM consists of three components: an LLM Decomposer, an LLM Planner, and an LLM Interface. The LLM Decomposer is responsible for dividing the given Minecraft goal into a subgoal tree. The LLM Planner then plans an action sequence for each sub-goal. Finally, the LLM Interface executes each action in the environment using keyboard and mouse operations.

Reinforcement learning in embodied AI virtual environments has the potential to improve the capabilities of realworld robotics by providing efﬁcient training and optimizing control policies in a safe and controlled setting. Reward design is a crucial aspect of RL that inﬂuences the robot’s learning process. Rewards should be aligned with the task’s objective and guide the robot to achieve the desired task. Foundation models can be leveraged to design rewards. Kwon et al. [16] investigate the simpliﬁcation of reward design by utilizing a large language model (LLM), such as GPT-3, as a proxy reward function. In this approach, users provide a textual prompt that contains a few examples (few-shots) or a description (zero-shot) of the desired behavior. The proposed method incorporates this proxy reward function within a reinforcement learning framework. Users specify a prompt at the start of the training process. During training, the RL agent’s behavior is evaluated by the LLM against the desired behavior outlined in the prompt, resulting in a corresponding reward signal generated by the LLM. Subsequently, the RL agent employs this reward to update its behavior through the learning process.

In [74], the authors propose a method called Exploring with LLMs (ELLM) that rewards an agent for achieving goals suggested by a language model. The language model is prompted with a description of the agent’s current state. Therefore, without having a human in the loop, ELMM guides agents toward meaningful behavior.

Zhang et al. [198] explore the potential relationship between ofﬂine reinforcement learning and language modeling. They hypothesize that RL and LM share similarities in predicting future states based on current and past states, considering both local and long-range dependencies across states. To validate this assumption, the authors pre-train Transformer models on different ofﬂine RL tasks and assess their performance on various language-related tasks. Tarasov et al. [199] present an approach to harness pretrained language models in deep ofﬂine reinforcement learning scenarios that are not inherently compatible with textual representations. The authors suggest a method that involves transforming the RL states into humanreadable text and performing ﬁne-tuning of the pretrained language model during training with deep ofﬂine RL algorithms.

Advances in model architecture (e.g. transformer) for foundation models allow the model to effectively model and predict sequences. To harness the power of these models, some recent studies investigate exploiting these architectures for sequence modeling in RL problems. Reid et al. [200] explore the potential of leveraging the sequence modeling formulation of reinforcement learning and examine the transferability of

pretrained sequence models across different domains, such as vision and language. They speciﬁcally focus on the effectiveness of ﬁne-tuning these pretrained models on ofﬂine RL tasks, including control and games. In addition to investigating the transferability of pretrained sequence models, the authors propose techniques to enhance the transfer of knowledge between these domains. These techniques aim to improve the adaptability and performance of the pretrained models when applied to new tasks or domains.

High-level task planning using LLMs is demonstrated in embodied AI environments. Huang et al. [68] propose employing pretrained Language Models (LMs) as zero-shot planners. The approach is evaluated in the VirtualHome [129] environment. In this work, ﬁrst, an autoregressive LLM such as GPT-3 [2] or Codex [201] is quarried to generate action plans for high-level tasks. Some of these action plans might not be executable by the agent due to ambiguity in language or referring to objects that are not present or grounded in the environment. So, to select the admissible action plans, admissible environment actions, and generated actions by the causal LLM are embedded using a BERT-style LM. Then for each admissible environment action, its semantic distance to the generated action is computed using cousin similarity.

Chain of thought reasoning and action generation are proposed for embodied agents as well. ReAct [202] combines reasoning (e.g. chain of thought) and acting (e.g. sequence of action generation) within LLM. Reasoning traces enhance the model’s ability to deduce, monitor, and revise action plans, along with managing exceptions effectively. Actions facilitate interaction with external resources, like knowledge bases or environments, enabling it to acquire supplementary information. ReAct showcases its proﬁciency across a wide array of language and decision-making tasks, including questionanswering and fact veriﬁcation. It enhances interpretability and trust for users by transparently illustrating the process through which it searches for evidence and formulates conclusions. Unlike prior methods that depend on a singular chain-ofthought, ReAct engages with a Wikipedia API for pertinent information retrieval and belief updating. This strategy effectively mitigates the issues commonly associated with chain-ofthought reasoning, such as hallucination and error propagation.

VPT [72] presents video pretraining in which the agent learns to act by watching unlabeled online videos. It is shown that an inverse dynamic model can be trained with a small labeled dataset and the model can be used to label a huge unlabeled data of the internet. Videos of people who have played Minecraft are used to train an embodied AI agent to play Minecraft. The model exhibits zero-shot performance and can be ﬁne-tuned for more complex skills using imitation learning or reinforcement learning. The VPT model is trained with a standard behavioral cloning loss (9) (negative loglikelihood) while the actions are drawn from the inverse dynamic model.

A. Generalist AI

A long-standing challenge in robotics research is deploying robots or embodied AI agents in a variety of non-factory realworld applications, performing a range of tasks. To make

generalist robots that can operate in diverse environments with diverse tasks, some researchers have proposed generative simulators for robot learning. For example, Generative Agents [203] discusses how generative agents can produce realistic imitations of human behavior for interactive applications, creating a miniature community of agents similar to those found in games like The Sims. The authors connect their architecture with the ChatGPT large language model to create a game environment with 25 agents. The study includes two evaluations, a controlled evaluation and an end-to-end evaluation, which demonstrate the causal effects of the various components of their architecture. Xian et al. [204], authors propose a fully automated generative pipeline, known as a generative simulation for robot learning, which utilizes models to generate diverse tasks, scenes, and training guidance on a large scale. This approach can facilitate the scaling up of lowlevel skill learning, ultimately leading to a foundational model for robotics that empowers generalist robots.

An alternative method for developing generalist AI involves using generalizable multi-modal representations. Gato [154] is a generalist agent that works as a multi-modal, multi-task, multi-embodiment generalist policy. Using the same neural network with the same set of weights, Gato can sense and act with different embodiments in various environments across different tasks. Gato can play Atari, chat, caption images, stack blocks with a real robot arm, navigate in a 3D simulated environment, and more. Gato is trained on 604 different tasks with various modalities, observations, and actions. In this setting, language acts as a common grounding across different embodiments. Gato has 1.2B parameters and is trained ofﬂine in a supervised way. Positioned at the conﬂuence of representation learning and reinforcement learning (RL), RRL [205] learns behaviors directly from proprioceptive inputs. By harnessing pre-trained visual representations, RRL is able to learn from visual inputs, which typically pose challenges in conventional RL settings.

- B. Simulators

High-quality simulators or benchmarks are crucial for robotics development. Hence, we put the “simulator” section here to highlight its essential role. To facilitate generalization from simulation to the real world, Gibson [206] emphasizes real-world perception for embodied agents. To bridge the gap between simulation and real-world, iGibson [146] and BEHAVIOR-1K [207] further support the simulation of a more diverse set of household tasks and reach high levels of simulation realism. As a simulation platform for research in Embodied AI, Habitat [208] consists of Habitat-Sim and Habitat-API. Habitat-Sim can achieve several thousand frames per second (fps) running single-threaded. Rather than modeling into low-level physics, Habitat-Lab [147], is a highlevel library for embodied AI, giving a modular framework for end-to-end development. It facilitates the deﬁnition of embodied AI tasks, such as navigation, interaction, instruction following, and question answering. Additionally, it enables the conﬁguration of embodied agents, encompassing their physical form, sensors, and capabilities. The library supports various

training methodologies for these agents, including imitation learning, reinforcement learning, and traditional non-learning approaches like the SensePlanAct pipelines. Furthermore, it provides standard metrics for evaluating agent performance across these tasks. In line with this, the recent release of Habitat 3.0 [209] further expands these capabilities.

Similarly, RoboTHOR [210] serves as a platform for the development and evaluation of embodied AI agents, offering environments in both simulated and physical settings. Currently, RoboTHOR includes a training and validation set comprising 75 simulated scenes. Additionally, there are 14 scenes each for test-dev and test-standard in the simulation, with corresponding physical counterparts. Key features of RoboTHOR include its reconﬁgurability and benchmarking capabilities. The physical environments are constructed using modular, movable components, enabling the creation of diverse scene layouts and furniture conﬁgurations in a single physical area. Another simulator, VirtualHome [129], models complex activities that occur in a typical household. It supports program descriptions for a variety of activities that happen in people’s homes. Huang et al. [33] use VirtualHome to evaluate the robot planning ability with language models. These simulators have the potential to be applied for evaluating LLMs on robotics tasks.

VI. CHALLENGES AND FUTURE DIRECTIONS

In this section, we examine challenges related to integrating foundation models into robotics settings. We also explore potential future avenues to address some of these challenges.

A. Overcoming Data Scarcity in Training Foundation Models for Robotics

One main challenge is that compared to the internet-scale text and image data that large models are trained on, roboticspeciﬁc data is scarce. We discuss various techniques to overcome data scarcity. For example, to scale up robot learning, some recent works suggest the use of play data instead of expert data for imitation learning. Another technique is data augmentation using in-painting techniques.

1) Scaling Robot Learning Using Unstructured Play Data and Unlabeled Videos of Humans: Language-conditioned learning such as language-conditioned behavioral cloning, or language-conditioned affordance learning requires having access to large annotated datasets. To scale up learning, in Play-LMP [26], the authors suggest using teleoperated human-provided play data instead of fully annotated expert demonstrations. Play data is unstructured, unlabeled, cheap to collect, but rich. Collecting play data does not require scene staging, task segmenting, or resetting to an initial state. Also, in MimicPlay [118] a goal-conditioned trajectory generation model is trained based on human-play data. The play data includes unlabeled video sequences of humans interacting with the environment with their hands. Recently works such as [125] have shown a very small percentage (as little as 1%) of language-annotated data is needed to train a visuo-lingual affordance model for robot manipulation tasks.

- 2) Data Augmentation using Inpainting: Collecting

robotics data requires the robot to interact with the real physical world. This data collection process can be associated with signiﬁcant costs and potential safety concerns. One way to tackle this challenge is to use generative AI such as text-to-image diffusion models for data augmentation. For example, ROSIE (Scaling Robot Learning with Semantically Imagined Experience) [176] presents a diffusion-based data augmentation. Given a robot manipulation dataset, they use inpainting to create various unseen objects, backgrounds, and distractors with textual guidance. One important challenge for these methods is developing inpainting strategies that can generate sufﬁcient semantically and visually diverse data, while at the same time ensuring that this data is physically feasible and accurate. For instance, using inpainting to modify an image of an object within a robot’s gripper may result in an image with a physically unrealistic grasp, leading to poor downstream training performance. Additional investigation into generative foundation models that are evaluated not only for visual quality but also for physical realism may improve the generality of these methods.

- 3) Overcoming 3D Data Scarcity for Training 3D Foun-

dation Models: Currently, multi-modal Vision-and-Language Models (VLMs) can analyze 2D images, but they lack a connection to the 3D world, which encompasses 3D spatial relationships, 3D planning, 3D affordances, and more. The primary obstacle in developing foundational 3D VLM models lies in the scarcity of 3D data, especially data that is paired with language descriptions. As discussed, language-driven perception tasks such as language-driven 3D scene representation, language-driven 3D scene editing, language-driven 3D scene or shape generation, language-driven 3D classiﬁcation, and affordance prediction require access to 3D data or multi-view images with camera matrices which are not readily available data types. New datasets or data generation methods need to be created in the future to overcome data scarcity in the 3D domain.

- 4) Synthetic Data Generation via High-Fidelity Simulation:

High-ﬁdelity simulation via gaming engines can provide an efﬁcient means to collect data, especially to solve multimodal and 3D perception tasks on robots. For example, TartanAir

- [211], a dataset for robot navigation tasks, was collected in
- [212] with the presence of moving objects, changing light, and various weather conditions. By collecting data in simulations, it was possible to obtain multi-modal sensor data and precise ground truth labels such as the stereo RGB image, depth image, segmentation, optical ﬂow, camera poses, and LiDAR point cloud. A large number of environments were set up with various styles and scenes, covering challenging viewpoints and diverse motion patterns that are difﬁcult to achieve by using physical data collection platforms. An extension TartanAirV2 (https://tartanair.org) furthers the dataset by incorporating additional environments and modalities, such as ﬁsheye, panoramas, and pinholes, with arbitrary camera intrinsic and rotations.

5) Data Augmentation using VLMs: Data augmentation can be provided using Visual-Language Models (VLMs). In DIAL [213], Data-driven Instruction Augmentation for

Language-conditioned control is introduced. DIAL uses VLM to label ofﬂine datasets for language-conditioned policy learning. DIAL performs instruction augmentation using VLMs to weakly relabel ofﬂine control datasets. DIAL consists of three steps 1) Contrastive ﬁne-tuning of a VLM such as CLIP [4] on a small robot manipulation dataset of trajectories with crowdsourced annotation, 2) producing new instruction labels by using the ﬁne-tuned VLM to score relevancy of crowd-sourced annotations against a larger dataset of trajectories, 3) training a language-conditioned policy using behavior cloning on both, the original and re-annotated dataset.

6) Robot Physical Skills are Limited to Distribution of Skills: One key limitation of the existing robot transformers and other related works in robotics is that robot physical skills are limited to the distribution of skills observed within the robot data. Using these transformers, the robot lacks the capability to generate new movements. To address this constraint, an approach involves using motion data from videos that humans performing various tasks. The inherent motion information within these videos can then be employed to facilitate the acquisition of physical skills in robotics.

B. Real Time Performance (High Inference Time of Foundation Models)

Another bottleneck for deploying foundation models on robots is the inference time of these models. In Table II, the inference time for some of these models is reported. As seen, the inference time for some of the models still needs to be improved for reliable real-time deployment of the robotic systems. As real-time capability is an essential requirement for any robotic system, more research needs to be performed to improve the computational efﬁciency of foundation models.

Furthermore, foundation models are most often stored and run in remote data centers, and accessed through APIs that require network connectivity. Many foundation models (e.g., the GPT models, the Dall-E models) can only be accessed this way, while others are usually accessed this way, but can also be downloaded and run locally with sufﬁcient local computing power (such as SAM [59], LLaMA [214], and DINOv2 [107]). Given this cloud-service paradigm, the latencies and service times in response to an API call for a foundation model depend on the underlying network over which the data is routed and the data center where the computation takes place—factors that are beyond the control of a robot. So network reliability should be taken into account before integrating a foundation model into a robot’s autonomy stack.

For some robotics domains reliance on the network and 3rd party computing may not be a safe or realistic operating paradigm. In autonomous driving, autonomous aircraft, search and rescue or emergency response applications, and defense applications the robot cannot rely on network connectivity for time-critical perception or control computations. One option is to have a safe fall-back mode that relies on classical autonomy tools using only local computation, that can take over if access to the cloud is interrupted for some reason. Another potential longer-term solution for network-free autonomy is the distillation of large foundation models into smaller-sized

specialized models that run on onboard robot hardware. Some recent work has attempted this approach (though without an explicit link to robotics) [215]. Such distilled models would likely give up some aspect of the full model, e.g. restricting operation to a certain limited context, in exchange for smaller size and faster compute. This could be an interesting future direction for bringing the power of foundation models to safety-critical robotics systems.

- C. Limitations in Multimodal Representation

Multimodal interaction implicitly assumes that the modality is tokenizable and can be standardized into input sequences without losing information. The Multimodal models provide information sharing between multiple modalities and are some variation of multimodal transformers with cross-modal attention between every pair of inputs. In multimodal representation learning, it is assumed that cross-modal interactions and the dimension of heterogeneity between different modalities can all be captured by simple embeddings. In other words, a simple embedding is assumed to be sufﬁcient to identify the modality or for example, how different language is from vision. In the realm of multimodal representation learning, the question of whether a single multimodal model can accommodate all modalities remains an open challenge.

Additionally, when paired data between a modality and text is available one can embed that modality into text directly. In robotics applications there are some modalities for which sufﬁcient data is not available and to be able to align them with other modalities, they need to be ﬁrst converted to other modalities and then be used. For example, 3D point cloud data has various applications in robotics but training a foundation model using this type of data is challenging since data is scarce and is not aligned with text. So, one way to overcome this challenge is ﬁrst converting this 3D point cloud data to other modalities such as images and subsequently images to text as the secondary step of alignment. Then they can be used in foundation model training. As another example, in Socratic models [194], each modality, whether visual or auditory, is initially translated into language, after which language models attempt to respond to these modalities.

- D. Uncertainty Quantiﬁcation

How can we provide assurances on the reliability of foundation models when they are deployed in potentially safetycritical robotics applications [188]? Current foundation models such as LLMs often hallucinate, i.e., produce outputs that are factually incorrect, logically inconsistent, or physically infeasible. While such failures may be acceptable in applications where the outputs from the model can be checked by a human in real-time (e.g., as is often the case for LLM-based conversational agents), they are not acceptable when deploying autonomous robots that use the outputs of foundation models in order to act in human-centered environments. Rigorous uncertainty quantiﬁcation is a key step toward addressing this challenge and safely integrating foundation models into robotic systems. Below, we highlight challenges and recent progress in uncertainty quantiﬁcation for foundation models in robotics.

1) Instance-Level Uncertainty Quantiﬁcation: How can we quantify the uncertainty in the output of a foundation model for a particular input? As an example, consider the problem of image classiﬁcation; given a particular image, one may quantify uncertainty in the output by producing a set of object labels that the model is uncertain among or a distribution over object labels. Instance-level uncertainty quantiﬁcation can inform the robot’s decisions at runtime. For example, if an image classiﬁcation model running on an autonomous vehicle produces a prediction set {Pedestrian, Bicyclist} representing that it is uncertain whether a particular agent is a pedestrian or a bicyclist, the autonomous vehicle can take actions that consider both possibilities.

- 2) Distribution-Level Uncertainty Quantiﬁcation: How can

we quantify the uncertainty in the correctness of a foundation model that will be deployed on a distribution of possible future inputs? For the problem of image classiﬁcation, one may want to compute or bound the probability of errors over the distribution of inputs that a robot may encounter when deployed. Distribution-level uncertainty quantiﬁcation allows us to decide whether a given model is sufﬁciently reliable to deploy in our target distribution of scenarios. For example, we may want to collect additional data or ﬁne-tune the model if the computed probability of error is too high.

- 3) Calibration: In order to be useful, estimates of un-

certainty (both at the instance-level and distribution level) should be calibrated. If we perform instance-level uncertainty quantiﬁcation using prediction sets, calibration asks for the prediction set to contain the true label with a user-speciﬁed probability (e.g., 95%) over future inputs. If instance-level uncertainty is quantiﬁed using a distribution over outputs, it should be the case that outputs that are assigned conﬁdence p are in fact correct with probability p over future inputs. Similarly, distribution-level uncertainty estimates should bound the true probability of errors when encountering inputs from the target distribution.

We highlight a subtle, but important, point that is often overlooked when performing uncertainty quantiﬁcation in robotics: it can be crucial to pay attention to the distinction between Frequentist and Bayesian interpretations of probabilities. In many robotics contexts — particularly safety-critical ones the desired interpretation is often Frequentist in nature. For example, if we produce a bound ǫ for the probability of collision of an autonomous vehicle, this should bound the actual observed rate of collisions when the vehicle is deployed. Bayesian techniques (e.g., Gaussian processes or Bayesian ensembles) do not necessarily produce estimates of uncertainty that are calibrated in this Frequentist sense (since the estimates depend on the speciﬁc prior that is used to produce the estimates). Trusting the resulting uncertainty estimates may lead one astray if the goal is to provide statistical guarantees on the safety or performance of the robotic system when it is deployed.

4) Distribution Shift: An important challenge in performing calibrated uncertainty quantiﬁcation is distribution shift. A foundation model trained on a particular distribution of inputs may not produce calibrated estimates of uncertainty when deployed on a different distribution for a downstream task. A

more subtle cause of distribution shift in robotics arises from closed-loop deployment of a model. For example, imagine an autonomous vehicle that chooses actions using the output of a perception system that relies on a pretrained foundation model; since the robot’s actions inﬂuence future states and observations, the distribution of inputs the perception system receives can be potentially very different from the one it was trained on.

5) Case Study: Uncertainty Quantiﬁcation for LanguageInstructed Robots: Recently, there has been exciting progress in performing rigorous uncertainty quantiﬁcation for languageinstructed robots [216]. This work proposes an approach called KNOWNO for endowing language-instructed robots with the ability to know when they don’t know and to ask for help or clariﬁcation from humans in order to resolve uncertainty. KNOWNO performs both instance-level and distribution-level uncertainty quantiﬁcation in a calibrated manner using the theory of conformal prediction. In particular, given a language instruction (and a description of the robot’s environment generated using its sensors), conformal prediction is used to generate a prediction set of candidate actions. If this set is a singleton, the robot executes the corresponding action; otherwise, the robot seeks help from a human by asking them to choose an action from the generated set. Using conformal prediction, KNOWNO ensures that asking for help in this manner results in a statistically guaranteed level of task success (i.e., distribution-level uncertainty quantiﬁcation). KNOWNO tackles potential challenges with distribution shift by collecting a small amount of calibration data from the target distribution of environments, tasks, and language instructions, and using this as part of the conformal prediction calibration procedure. While KNOWNO serves as an example of calibrated instance-level and distribution-level uncertainty quantiﬁcation for LLMs, future research should also explore assessing and ensuring the reliability of various other foundation models, such as vision-language models, vision-navigation models, and vision-language-action models, commonly employed in robotics. In addition, exploring how Bayesian uncertainty quantiﬁcation techniques (e.g., ensembling [217], [218]) can be combined with approaches such as conformal prediction to produce calibrated estimates of instance-level and distributionlevel uncertainty is a promising direction.

- E. Safety Evaluation

The problem of safety evaluation is closely related to uncertainty quantiﬁcation. How can we rigorously test for the safety of a foundation model-based robotic system (i) before deployment, (ii) as the model is updated during its lifecycle, and (iii) as the robot operates in its target environments? We highlight challenges and research opportunities related to these problems below.

1) Pre-deployment safety tests: Rigorous pre-deployment testing is crucial for ensuring the safety of any robotic system. However, this can be particularly challenging for robots that incorporate foundation models. First, foundation models are trained on vast amounts of data; thus, a rigorous testing procedure should ensure that test scenarios were not seen by

the model during training. Second, foundation models often commit errors in ways that are hard to predict a priori; thus, tests need to cover a diverse enough range of scenarios to uncover ﬂaws. Third, foundation models such as LLMs are often used to produce open-ended outputs (e.g., a plan for a robot described in natural language). The correctness of such outputs can be challenging to evaluate in an automated manner if these outputs are evaluated in isolation from the entire system.

The deployment cycle of current foundation models (in non-robotics applications) involves thorough red-teaming by human evaluators [3], [219]. Recent work has also considered partially automating this process by using foundation models themselves to perform red-teaming [220], [221]. Developing ways to perform red-teaming (both by humans and in a partially automated way) for foundation models in robotics is an exciting direction for future research.

In addition to evaluating the foundation model in isolation, it is also critical to assess the safety of the end-to-end robotic system. Simulation can play a critical role here, and already does so for current ﬁeld-deployed systems such as autonomous vehicles [222], [223]. The primary challenges are to ensure that (i) the simulator has high enough ﬁdelity for results to meaningfully transfer to the real world, and (ii) test scenarios (manually speciﬁed, replicated from real-world scenarios, or automatically generated via adversarial methods [224]) are representative of real-world scenarios and are diverse enough to expose ﬂaws in the underlying foundation models. In addition, ﬁnding ways to augment large-scale simulation-based testing with smaller-scale real-world testing is an important direction for future work. We emphasize the need for performing such testing throughout the lifecycle of a ﬁeld-deployed robotic system, especially as updates are made to different components (which may interact in unpredictable ways with foundation models).

2) Runtime monitoring and out-of-distribution detection:

In addition to performing rigorous testing ofﬂine, robots with foundation model-based components should also perform runtime monitoring. This can take the form of failure prediction in a given scenario, which can allow the robot to deploy a safety-preserving fallback policy [225]–[229]. Alternately, the robot can perform out-of-distribution (OOD) detection using experiences collected from a small batch of scenarios in a novel distribution [230]–[233]; this can potentially trigger the robot to cease its operations and collect additional training data in the novel distribution in order to re-train its policy. Developing techniques that perform runtime monitoring and OOD detection with statistical guarantees on false positive/negative error rates in a data-efﬁcient manner remains an important research direction.

F. Using Existing Foundation Models as Plug-and-Play or Building New Foundation Models for Robotics

To incorporate foundation models into robotics, either the existing pretrained large models can be employed as plug-andplay or new foundation models can be built using robotics data. Using foundation models as plug-and-play refers to integrating foundation models into various applications without

the need for extensive customization. A large body of the existing literature on foundation models in robotics is centered around the use of foundation models from other domains such as language or vision as plug-and-play. The plug-and-play approach simpliﬁes and facilitates the integration of recent AI advances into the robotics domain. While employing these models as plug-and-play offers a convenient way to harness the power of AI and provide rapid implementation, versatility, and scalability, they are not always customized to speciﬁc applications. When speciﬁc domain expertise is needed, it is necessary to build a foundation model from scratch or ﬁnetune the existing models. Building a foundation model from scratch is resource-intensive and demands signiﬁcant computational power. However, it provides ﬁne-grained control over the architecture, training parameters, and overall behavior.

- G. High Variability in Robotic Settings

Another challenge is the high variability in robotic settings. Robot platforms are inherently diverse with different physical characteristics, conﬁgurations, and capabilities. Realworld environments that robots operate in are also diverse and uncertain with a wide range of variations. Due to all these variabilities, robotic solutions are usually tailored to speciﬁc robot platforms with speciﬁc layouts, environments, and objects for speciﬁc tasks. These solutions are not generalizable across various embodiments, environments, or tasks. Hence, to build general-purpose pretrained robotic foundation models, a key factor is to pre-train large models that are task-agnostic, cross-embodiment, and open-ended and capture diverse robotic data. In ROSIE [176] a diverse dataset is generated for robot learning by performing inpainting of various unseen objects, backgrounds, and distractors with semantic textual guidance. To overcome variability in robotic settings and improve generalization, another solution as ViNT [137] presents is to train foundation models on diverse robotic data across various embodiments. RT-X [46] also investigates the possibility of training large cross-embodied robotic models in the domain of robotic manipulation. RT-X is trained using a multi-embodiment dataset which is created by collecting data from different robot platforms collected through a collaboration between 21 institutions, demonstrating 160266 tasks. RTX demonstrates transfer across embodiment improves robot capabilities by employing experience from diverse robotic platforms.

- H. Benchmarking and Reproducibility in Robotics Settings

Another signiﬁcant obstacle in incorporating foundation models into robotics research is the necessary reliance on real-world hardware experiments. This creates challenges for reproducibility, as replicating results obtained from hardware experiments may necessitate access to the exact equipment employed. Conversely, many recent works have relied on nonphysics-based simulators (e.g., ignoring or greatly simplifying contact physics in gasping) that instead focus on high-level, long-term tasks and visual environment models. Examples of this class of simulators are common and include many of the simulators described above in Sec. V. For example the Gibson

family of simulators [146], [206], the Habitat family [147], [208], [209], RobotTHOR [210], and VirtualHome [129] all neglect low-level physics in favor of simulating higher level tasks with high visual ﬁdelity. This leads to a large sim-toreal gap and introduces variability in real-world performance based on how low-level planning and control modules handle the true physics of the scenario. Even when physics-based simulators are used (e.g., PyBullet or MuJoCo), the absence of standardized simulation settings, computing environments, and a persistent sim-to-real gap impede efforts to benchmark and compare performance across various research endeavors. A combination of open hardware, benchmarking in physicsbased simulators, and promoting transparency in experimental and simulation setups can signiﬁcantly alleviate the challenges associated with reproducibility in the integration of foundation models into robotics research. These practices contribute to the development of a more robust and collaborative research ecosystem within the ﬁeld.

VII. CONCLUSION

Through examination of the recent literature, we have surveyed the diverse and promising applications of foundation models in robotics. We have delved into how these models have enhanced the capabilities of robots in areas such as decision-making, planning and control, and perception. We also discussed the literature on embodied AI and generalist AI, with an eye toward opportunities for roboticists to extend the concepts in that research ﬁeld to real-world robotic applications. Generalization, zero-shot capabilities, multimodal capabilities, and scalability of foundation models have the potential to transform robotics. However, as we navigate through this paradigm shift in incorporating foundation models in robotics applications, it is imperative to recognize the challenges and potential risks that must be addressed in future research. Data scarcity in robotics applications, high variability in robotics settings, uncertainty quantiﬁcation, safety evaluation, and realtime performance remain signiﬁcant concerns that demand future research. We have delved into some of these challenges and have discussed potential avenues for improvement.

ACKNOWLEDGMENTS

The ﬁrst author was supported on an ASEE e-Fellows postdoctoral fellowship. J.T. and S.T. were partially supported by NSF Graduate Research Fellowships. This project was also partially supported by DARPA project HR001120C0107 and by a gift from Meta. We are grateful for this support. Anirudha Majumdar was supported by the NSF CAREER Award [#2044149] and the Ofﬁce of Naval Research [N00014-23-1-2148].

REFERENCES

- [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020.

- [3] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [4] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, ICML, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021.
- [5] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot textto-image generation. In ICML, pages 8821–8831. PMLR, 2021.
- [6] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-E: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023.
- [7] Jiankai Sun, De-An Huang, Bo Lu, Yun-Hui Liu, Bolei Zhou, and Animesh Garg. PlaTe: Visually-grounded planning with transformers in procedural tasks. IEEE Robotics and Automation Letters, 7(2):4924– 4930, 2022.
- [8] Jianing Qiu, Lin Li, Jiankai Sun, Jiachuan Peng, Peilun Shi, Ruiyang Zhang, Yinzhao Dong, Kyle Lam, Frank P.-W. Lo, Bo Xiao, Wu Yuan, Ningli Wang, Dong Xu, and Benny Lo. Large ai models in health informatics: Applications, challenges, and the future. IEEE Journal of Biomedical and Health Informatics, 27(12):6074–6087, 2023.
- [9] Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. Reasoning with foundation models: Concepts, methodologies, and outlook. In Zenodo preprint 10.5281/zenodo.10298866, 2023.
- [10] Dingyuan Zhang, Dingkang Liang, Hongcheng Yang, Zhikang Zou, Xiaoqing Ye, Zhe Liu, and Xiang Bai. SAM3D: Zero-shot 3D object detection via segment anything model. arXiv preprint arXiv:2306.02245, 2023.
- [11] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D world into large language models. arXiv preprint arXiv:2307.12981, 2023.
- [12] William Chen, Siyi Hu, Rajat Talak, and Luca Carlone. Leveraging large language models for robot 3D scene understanding. arXiv preprint arXiv:2209.05629, 2022.
- [13] Sherry Yang, Oﬁr Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023.
- [14] Zhao Mandi, Homanga Bharadhwaj, Vincent Moens, Shuran Song, Aravind Rajeswaran, and Vikash Kumar. CACTI: A framework for scalable multi-task multi-scene visual imitation learning. arXiv preprint arXiv:2212.05711, 2022.
- [15] Norman Di Palo, Arunkumar Byravan, Leonard Hasenclever, Markus Wulfmeier, Nicolas Heess, and Martin Riedmiller. Towards a uniﬁed agent with foundation models. In Workshop on Reincarnating Reinforcement Learning at ICLR 2023, 2023.
- [16] Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. Reward design with language models. In ICLR, 2023.
- [17] Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. ChessGPT: Bridging policy learning and language modeling. arXiv preprint arXiv:2306.09200, 2023.
- [18] Yifan Du, Junyi Li, Tianyi Tang, Wayne Xin Zhao, and Ji-Rong Wen. Zero-shot visual question answering with language model feedback. arXiv preprint arXiv:2305.17006, 2023.
- [19] Xingwei He, Zhenghao Lin, Yeyun Gong, Alex Jin, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, Weizhu Chen, et al. AnnoLLM: Making large language models to be better crowdsourced annotators. arXiv preprint arXiv:2303.16854, 2023.
- [20] Xuan Xiao, Jiahang Liu, Zhipeng Wang, Yanmin Zhou, Yong Qi, Qian Cheng, Bin He, and Shuo Jiang. Robot learning in the era of foundation models: A survey. arXiv preprint arXiv:2311.14379, 2023.
- [21] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023.
- [22] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM Computing Surveys, 54(10s):1–41, 2022.

- [23] Muning Wen, Runji Lin, Hanjing Wang, Yaodong Yang, Ying Wen, Luo Mai, Jun Wang, Haifeng Zhang, and Weinan Zhang. Large sequence models for sequential decision-making: a survey. Frontiers of Computer Science, 17(6):176349, 2023.
- [24] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.
- [25] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. CLIPort: What and where pathways for robotic manipulation. In CoRL, pages 894–906. PMLR, 2022.
- [26] Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and Pierre Sermanet. Learning latent plans from play. In CoRL, pages 1113–1132. PMLR, 2020.
- [27] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-Actor: A multi-task transformer for robotic manipulation. In CoRL, pages 785–

799. PMLR, 2023.

- [28] Corey Lynch and Pierre Sermanet. Language conditioned imitation learning over unstructured data. Robotics: Science and Systems, 2021.
- [29] Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. In RSS, 2023.
- [30] Adaptive Agent Team, Jakob Bauer, Kate Baumli, Satinder Baveja, Feryal Behbahani, Avishkar Bhoopchand, Nathalie Bradley-Schmieg, Michael Chang, Natalie Clay, Adrian Collister, Vibhavari Dasagi, Lucy Gonzalez, Karol Gregor, Edward Hughes, Sheleem Kashem, Maria Loks-Thompson, Hannah Openshaw, Jack Parker-Holder, Shreya Pathak, Nicolas Perez-Nieves, Nemanja Rakicevic, Tim Rockta¨schel, Yannick Schroecker, Jakub Sygnowski, Karl Tuyls, Sarah York, Alexander Zacherl, and Lei Zhang. Human-timescale adaptation in an open-ended task space. In ICML, 2023.
- [31] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.
- [32] Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. Do as I can, not as I say: Grounding language in robotic affordances. In CoRL, pages 287–318. PMLR, 2023.
- [33] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner monologue: Embodied reasoning through planning with language models. In arXiv preprint arXiv:2207.05608, 2022.
- [34] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. VoxPoser: Composable 3D value maps for robotic manipulation with language models. In CoRL, 2023.
- [35] Parsa Mahmoudieh, Deepak Pathak, and Trevor Darrell. Zero-shot reward speciﬁcation via grounded natural language. In ICML, pages 14743–14752. PMLR, 2022.
- [36] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. VIP: Towards universal visual reward and representation via value-implicit pre-training. In ICLR, 2023.
- [37] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. LIV: Language-image representations and rewards for robotic control. In ICML, 2023.
- [38] Suraj Nair, Eric Mitchell, Kevin Chen, brian ichter, Silvio Savarese, and Chelsea Finn. Learning language-conditioned robot behavior from ofﬂine data and crowd-sourced annotation. In CoRL, pages 1303–1315. PMLR, 08–11 Nov 2022.
- [39] Yongchao Chen, Rujul Gandhi, Yang Zhang, and Chuchu Fan. NL2TL: Transforming natural languages to temporal logics using large language models. arXiv preprint arXiv:2305.07766, 2023.
- [40] Yongchao Chen, Jacob Arkin, Yang Zhang, Nicholas Roy, and Chuchu Fan. AutoTAMP: Autoregressive task and motion planning with llms as translators and checkers. arXiv preprint arXiv:2306.06531, 2023.
- [41] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. ProgPrompt: Generating situated robot task plans using large language models. In ICRA, pages 11523–11530. IEEE, 2023.
- [42] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as Policies: Language model programs for embodied control. In ICRA, pages 9493–9500. IEEE, 2023.

- [43] Sai Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. ChatGPT for Robotics: Design principles and model abilities. Technical Report MSR-TR-2023-8, Microsoft, 2023.
- [44] Anthony Brohan, Noah Brown, Justice Carbajal, and et al. RT-1: Robotics transformer for real-world control at scale. In RSS, 2023.
- [45] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R Sanketi, Grecia Salazar, Michael S Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J Joshi, Alex Irpan, brian ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen, Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: Vision-languageaction models transfer web knowledge to robotic control. In CoRL, 2023.
- [46] Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv preprint arXiv:2310.08864, 2023.
- [47] Rogerio Bonatti, Sai Vemprala, Shuang Ma, Felipe Frujeri, Shuhang Chen, and Ashish Kapoor. PACT: Perception-action causal transformer for autoregressive robotics pretraining. In IROS. IEEE, 2023.
- [48] Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik. Masked visual pre-training for motor control. arXiv preprint arXiv:2203.06173, 2022.
- [49] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In CoRL, pages 416–426. PMLR, 2023.
- [50] Arthur Bucker, Luis Figueredo, Sami Haddadin, Ashish Kapoor, Shuang Ma, Sai Vemprala, and Rogerio Bonatti. LATTE: LAnguage Trajectory TransformEr. In ICRA, pages 7287–7294. IEEE, 2023.
- [51] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple openvocabulary object detection with vision transformers. In ECCV, pages 728–755. Springer, 2022.
- [52] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, JenqNeng Hwang, et al. Grounded language-image pre-training. In CVPR, pages 10965–10975, 2022.
- [53] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [54] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. PointCLIP: Point cloud understanding by CLIP. In CVPR, pages 8552–8562, 2022.
- [55] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-BERT: Pre-training 3D point cloud transformers with masked point modeling. In CVPR, pages 19313–19322, 2022.
- [56] Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. ULIP: Learning uniﬁed representation of language, image and point cloud for 3D understanding. arXiv preprint arXiv:2212.05171, 2022.
- [57] Le Xue, Ning Yu, Shu Zhang, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. ULIP-2: Towards scalable multimodal pre-training for 3d understanding. arXiv preprint arXiv:2305.08275, 2023.
- [58] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In ICLR, 2022.
- [59] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In ICCV, pages 4015–4026, October 2023.
- [60] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. arXiv preprint arXiv:2306.12156, 2023.
- [61] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289, 2023.

- [62] Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023.
- [63] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. CLIP-NeRF: Text-and-image driven manipulation of neural radiance ﬁelds. In CVPR, pages 3835–3844, 2022.
- [64] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance ﬁelds. In ICCV, pages 19729–19739, 2023.
- [65] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing NeRF for editing via feature ﬁeld distillation. In NeurIPS, 2022.
- [66] Yufei Ye, Xueting Li, Abhinav Gupta, Shalini De Mello, Stan Birchﬁeld, Jiaming Song, Shubham Tulsiani, and Sifei Liu. Affordance diffusion: Synthesizing hand-object interactions. In CVPR, 2023.
- [67] Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak. Affordances from human videos as a versatile representation for robotics. In CVPR, 2023.
- [68] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In ICML, 2022.
- [69] Takuma Yoneda, Jiading Fang, Peng Li, Huanyu Zhang, Tianchong Jiang, Shengjie Lin, Ben Picker, David Yunis, Hongyuan Mei, and Matthew R. Walter. Statler: State-maintaining language models for embodied reasoning. arXiv preprint arXiv:2306.17840, 2023.
- [70] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. EmbodiedGPT: Vision-language pre-training via embodied chain of thought. arXiv preprint arXiv:2305.15021, 2023.
- [71] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. MineDojo: Building open-ended embodied agents with internet-scale knowledge. In NeurIPS Datasets and Benchmarks Track, 2022.
- [72] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video PreTraining (VPT): Learning to act by watching unlabeled online videos. In NeurIPS, 2022.
- [73] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv: Arxiv-2305.16291, 2023.
- [74] Yuqing Du, Olivia Watkins, Zihan Wang, Ce´dric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, and Jacob Andreas. Guiding pretraining in reinforcement learning with large language models. In ICML, pages 8657–8677. PMLR, 23–29 Jul 2023.
- [75] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In ACL, 2016.
- [76] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.
- [77] Daniel Dugas. The gpt-3 architecture, on a napkin. [Online; accessed 28-November-2023].
- [78] Wikipedia. GPT-3. [Online; accessed 28-November-2023].
- [79] John Thickstun. The Transformer Model in Equations. [Online; accessed 28-November-2023].
- [80] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015.
- [81] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. https://openai.com/research/language-unsupervised, 2018.
- [82] Peter J Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating Wikipedia by summarizing long sequences. In ICLR, 2018.
- [83] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning of medical visual representations from paired images and text. In MLHC, 2022.
- [84] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.
- [85] Kihyuk Sohn. Improved deep metric learning with multi-class N-pair loss objective. In NeurIPS, 2016.
- [86] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, 2018.

- [87] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
- [88] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.
- [89] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.
- [90] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019.
- [91] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [92] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI Blog, 2019.
- [93] Hector Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In KR, 2012.
- [94] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.
- [95] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothe´e Lacroix, Baptiste Rozie`re, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efﬁcient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [96] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garc´ıa, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark D´ıaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [97] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130B: An open bilingual pre-trained model. In ICLR, 2023.
- [98] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [99] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
- [100] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM Computing Surveys, 2022.
- [101] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, 2022.
- [102] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. In NeurIPS, 2022.
- [103] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023.
- [104] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. PaLI-X: On scaling up a multilingual vision and language model. arXiv preprint arXiv:2305.18565, 2023.

- [105] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve´ Je´gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.
- [106] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.
- [107] Maxime Oquab, Timothe´e Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [108] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for uniﬁed vision-language understanding and generation. In ICML, 2022.
- [109] Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, and Hang Xu. CLIP2: Contrastive language-image-point pretraining from real-world point cloud data. In CVPR, 2023.
- [110] Lewei Yao, Runhu Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. FILIP: Fine-grained interactive language-image pre-training. In ICLR, 2022.
- [111] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In CVPR, 2023.
- [112] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot textto-image generation. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8821–8831. PMLR, 18–24 Jul 2021.
- [113] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dolla´r, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 16000–16009, 2022.
- [114] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [115] Ge Yuying, Macaluso Annabella, Erran Li Li, Luo Ping, and Wang Xiaolong. Policy adaptation from foundation model feedback. In CVPR, 2023.
- [116] Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong, Ivan Krasin, Dan Duong, Vikas Sindhwani, and Johnny Lee. Transporter networks: Rearranging the visual world for robotic manipulation. In CoRL, 2020.
- [117] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjo¨rn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [118] Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. MimicPlay: Long-horizon imitation learning by watching human play. In CoRL, 2023.
- [119] Rutav Shah, Roberto Mart´ın-Mart´ın, and Yuke Zhu. MUTEX: Learning uniﬁed policies from multimodal task speciﬁcations. In CoRL, 2023.
- [120] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. RLˆ2: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.
- [121] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a ﬁxed-length context. In ACL, 2019.
- [122] Open Ended Learning Team, Adam Stooke, Anuj Mahajan, Catarina Barros, Charlie Deck, Jakob Bauer, Jakub Sygnowski, Maja Trebacz, Max Jaderberg, Michae¨l Mathieu, Nat McAleese, Nathalie BradleySchmieg, Nathaniel Wong, Nicolas Porcel, Roberta Raileanu, Steph Hughes-Fitt, Valentin Dalibard, and Wojciech Marian Czarnecki. Openended learning leads to generally capable agents. arXiv preprint arXiv:2107.12808, 2021.
- [123] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The EPIC-KITCHENS dataset. In ECCV, 2018.
- [124] Kevin Lin, Christopher Agia, Toki Migimatsu, Marco Pavone, and Jeannette Bohg. Text2Motion: From natural language instructions to feasible plans. Autonomous Robots. Special Issue: Large Language Models in Robotics, 2023.

- [125] Oier Mees, Jessica Borja-Diaz, and Wolfram Burgard. Grounding language with visual affordances over unstructured data. In ICRA, pages 11576–11582. IEEE, 2023.
- [126] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a uniﬁed text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485– 5551, 2020.
- [127] Junning Huang, Sirui Xie, Jiankai Sun, Qiurui Ma, Chunxiao Liu, Dahua Lin, and Bolei Zhou. Learning a decision module by imitating driver’s control behaviors. In CoRL, pages 1–10. PMLR, 2021.
- [128] Jiankai Sun, Hao Sun, Tian Han, and Bolei Zhou. Neuro-symbolic program search for autonomous driving decision module design. In CoRL, pages 21–30. PMLR, 2021.
- [129] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. VirtualHome: Simulating household activities via programs. In CVPR, pages 8494–8502, 2018.
- [130] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022.
- [131] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 35:24824–24837, 2022.
- [132] Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, and Andy Zeng. Large language models as general pattern machines. In arXiv preprint arXiv:2307.04721, 2023.
- [133] Zhiwei Jia, Fangchen Liu, Vineet Thumuluri, Linghao Chen, Zhiao Huang, and Hao Su. Chain-of-thought predictive control. arXiv preprint arXiv:2304.00776, 2023.
- [134] Yanchao Sun, Shuang Ma, Ratnesh Madaan, Rogerio Bonatti, Furong Huang, and Ashish Kapoor. SMART: Self-supervised multi-task pretraining with control transformers. In ICLR, 2023.
- [135] Arjun Majumdar, Ayush Shrivastava, Stefan Lee, Peter Anderson, Devi Parikh, and Dhruv Batra. Improving vision-and-language navigation with image-text pairs from the web. In ECCV, pages 259–274. Springer, 2020.
- [136] Dhruv Shah, Błaz˙ej Osin´ski, Sergey Levine, et al. LM-Nav: Robotic navigation with large pre-trained models of language, vision, and action. In CoRL, pages 492–504. PMLR, 2023.
- [137] Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, and Sergey Levine. ViNT: A Foundation Model for Visual Navigation. In CoRL, 2023.
- [138] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Audio visual language maps for robot navigation. arXiv preprint arXiv:2303.07522, 2023.
- [139] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. NetVLAD: CNN architecture for weakly supervised place recognition. In CVPR, pages 5297–5307, 2016.
- [140] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPR Deep Learning for Visual SLAM Workshop, 2018.
- [141] Andrey Guzhov, Federico Raue, Jo¨rn Hees, and Andreas Dengel. AudioCLIP: Extending clip to image, text and audio. In ICASSP, pages 976–980. IEEE, 2022.
- [142] Theophile Gervet, Soumith Chintala, Dhruv Batra, Jitendra Malik, and Devendra Singh Chaplot. Navigating to objects in the real world. arXiv preprint arXiv:2212.00922, 2023.
- [143] Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Ilharco, Ludwig Schmidt, and Shuran Song. CoWs on pasture: Baselines and benchmarks for language-driven zero-shot object navigation. In CVPR, pages 23171–23181, 2023.
- [144] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niebner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In 3DV, pages 667–676, 2017.
- [145] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Su¨nderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In CVPR, pages 3674– 3683, 2018.
- [146] Chengshu Li, Fei Xia, Roberto Mart´ın-Mart´ın, Michael Lingelbach, Sanjana Srivastava, Bokui Shen, Kent Elliott Vainio, Cem Gokmen, Gokul Dharan, Tanish Jain, Andrey Kurenkov, Karen Liu, Hyowon Gweon, Jiajun Wu, Li Fei-Fei, and Silvio Savarese. iGibson 2.0:

- Object-centric simulation for robot learning of everyday household tasks. In CoRL, 2021.
- [147] Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. In NeurIPS, 2021.
- [148] Yu Bangguo, Kasaei Hamidreza, and Cao Ming. LL3MVN: Leveraging large language models for visual target navigation. arXiv preprint arXiv:2304.05501, 2023.
- [149] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
- [150] Junting Chen, Guohao Li, Suryansh Kumar, Bernard Ghanem, and Fisher Yu. How to not train your dragon: Training-free embodied object goal navigation with semantic frontiers. arXiv preprint arXiv:2305.16925, 2023.
- [151] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton. Homerobot: Open vocabulary mobile manipulation. arXiv preprint arXiv:2306.11565, 2023.
- [152] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. VIMA: General robot manipulation with multimodal prompts. In ICML, 2023.
- [153] Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X Lee, Maria Bauza, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, et al. RoboCat: A self-improving foundation agent for robotic manipulation. arXiv preprint arXiv:2306.11706, 2023.
- [154] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.
- [155] Weiyu Liu, Yilun Du, Tucker Hermans, Sonia Chernova, and Chris Paxton. StructDiffusion: Language-guided creation of physically-valid structures using unseen objects. In RSS, 2023.
- [156] Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, KuangHuei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, Chelsea Finn, and Karol Hausman. Open-world object manipulation using pre-trained vision-language model. arXiv preprint arXiv:2303.00905, 2023.
- [157] Ivan Kapelyukh, Vitalis Vosylius, and Edward Johns. DALL-E-Bot: Introducing web-scale diffusion models to robotics. IEEE Robotics and Automation Letters, 8(7):3956–3963, 2023.
- [158] Kaiming He, Georgia Gkioxari, Piotr Dolla´r, and Ross Girshick. Mask R-CNN. In ICCV, pages 2980–2988, 2017.
- [159] Tay Yi, Dehghani Mostafa, Tran Vinh Q., Garcia Xavier, Wei Jason, Wang Xuezhi, Chung Hyung Won, Shakeri Siamak, Bahri Dara, Schuster Tal, Steven Zheng Huaixiu, Zhou Denny, Houlsby Neil, and Donald Metzler. UL2: Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2023.
- [160] Shuang Ma, Sai Vemprala, Wenshan Wang, Jayesh K Gupta, Yale Song, Daniel McDufft, and Ashish Kapoor. COMPASS:: Contrastive multimodal pretraining for autonomous systems. In IROS, pages 1000–

1007. IEEE, 2022.

- [161] Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy. IEEE Trans. Pattern Anal. Mach. Intell., 41(2):423–443, feb 2019.
- [162] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In CVPR, pages 21736– 21746, 2023.
- [163] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. ShapeNet: An information-rich 3D model repository. arXiv preprint arXiv:1512.03012, 2015.
- [164] Rene´ Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, pages 12179–12188, 2021.
- [165] Ho Kei Cheng and Alexander G Schwing. XMem: Long-term video object segmentation with an atkinson-shiffrin memory model. In ECCV, pages 640–658. Springer, 2022.
- [166] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Anything-3D: Towards single-view anything reconstruction in the wild. arXiv preprint arXiv:2304.10261, 2023.

- [167] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance ﬁelds for view synthesis. Communications of the ACM, 65(1):99–106, 2021.
- [168] Jiankai Sun, Yan Xu, Mingyu Ding, Hongwei Yi, Chen Wang, Jingdong Wang, Liangjun Zhang, and Mac Schwager. NeRF-Loc: Transformerbased object localization within neural radiance ﬁelds. IEEE Robotics and Automation Letters, 8(8):5244–5250, 2023.
- [169] Jiankai Sun, Jianing Qiu, Chuanyang Zheng, John Tucker, Javier Yu, and Mac Schwager. Aria-NeRF: Multimodal egocentric view synthesis. arXiv preprint arXiv:2311.06455, 2023.
- [170] Nur Muhammad Mahi Shaﬁullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam. CLIP-Fields: Weakly supervised semantic ﬁelds for robotic memory. In RSS, 2023.
- [171] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 10608–10615. IEEE, 2023.
- [172] Huy Ha and Shuran Song. Semantic abstraction: Open-world 3D scene understanding from 2D vision-language models. In CoRL, 2022.
- [173] Jianglong Ye, Naiyan Wang, and Xiaolong Wang. FeatureNeRF: Learning generalizable nerfs by distilling pre-trained vision foundation models. arXiv preprint arXiv:2303.12786, 2023.
- [174] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea Vedaldi. Neural Feature Fusion Fields: 3D distillation of self-supervised 2D image representations. In 3DV, pages 443–453. IEEE, 2022.
- [175] Xiaoshuai Zhang, Abhijit Kundu, Thomas Funkhouser, Leonidas Guibas, Hao Su, and Kyle Genova. Nerﬂets: Local radiance ﬁelds for efﬁcient structure-aware 3d scene representation from 2d supervision. In CVPR, pages 8274–8284, 2023.
- [176] Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, and Fei Xia. Scaling robot learning with semantically imagined experience. In arXiv preprint arXiv:2302.11550, 2023.
- [177] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In CVPR, pages 18359–18369, 2023.
- [178] Zoey Chen, Sho Kiami, Abhishek Gupta, and Vikash Kumar. GenAug: Retargeting behaviors to unseen situations via generative augmentation. In RSS, 2023.
- [179] Anthony Simeonov, Yilun Du, Andrea Tagliasacchi, Joshua B Tenenbaum, Alberto Rodriguez, Pulkit Agrawal, and Vincent Sitzmann. Neural Descriptor Fields: SE(3)-equivariant object representations for manipulation. In ICRA, 2022.
- [180] William Shen, Ge Yang, Alan Yu, Jensen Wong, Leslie Pack Kaelbling, and Phillip Isola. Distilled feature ﬁelds enable few-shot manipulation. In CoRL, 2023.
- [181] Walter Goodwin, Ioannis Havoutis, and Ingmar Posner. You only look at one: Category-level object representations for pose estimation from a single example. In CoRL, 2022.
- [182] Walter Goodwin, Sagar Vaze, Ioannis Havoutis, and Ingmar Posner. Zero-shot category-level object pose estimation. In ECCV, 2022.
- [183] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis & Machine Intelligence, 13(04):376–380, 1991.
- [184] Jiankai Sun, Lantao Yu, Pinqian Dong, Bo Lu, and Bolei Zhou. Adversarial inverse reinforcement learning with self-attention dynamics model. IEEE Robotics and Automation Letters, 6(2):1880–1886, 2021.
- [185] Jiankai Sun, Shreyas Kousik, David Fridovich-Keil, and Mac Schwager. Connected autonomous vehicle motion planning with video predictions from smart, self-supervised infrastructure. arXiv preprint arXiv:2309.07504, 2023.
- [186] Jiankai Sun, Shreyas Kousik, David Fridovich-Keil, and Mac Schwager. Self-supervised trafﬁc advisors: Distributed, multi-view trafﬁc prediction for smart cities. In IEEE ITSC, 2022.
- [187] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for ﬂexible behavior synthesis. In ICML, 2022.
- [188] Jiankai Sun, Yiqi Jiang, Jianing Qiu, Parth Talpur Nobel, Mykel Kochenderfer, and Mac Schwager. Conformal prediction for uncertainty-aware planning with diffusion dynamics model. In NeurIPS, 2023.
- [189] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro,

- Julius Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual description. In ICLR, 2023.
- [190] Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. RoboNet: Large-scale multi-robot learning. In CoRL, 2019.
- [191] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
- [192] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Oﬁr Nachum, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In NeurIPS, 2023.
- [193] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, and Jonathan Tompson. Video language planning. arXiv preprint arXiv:2310.10625, 2023.
- [194] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. Socratic Models: Composing zero-shot multimodal reasoning with language. arXiv, 2022.
- [195] Ishita Dasgupta, Christine Kaeser-Chen, Kenneth Marino, Arun Ahuja, Sheila Babayan, Felix Hill, and Rob Fergus. Collaborating with language models for embodied reasoning. In Second Workshop on Language and Reinforcement Learning, 2022.
- [196] Herman A. Engelbrecht and Gregor Schiele. Transforming minecraft into a research platform. In IEEE CCNC, 2014.
- [197] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and Jifeng Dai. Ghost in the Minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023.
- [198] Ziqi Zhang, Yile Wang, Yue Zhang, and Donglin Wang. Can ofﬂine reinforcement learning help natural language understanding? arXiv preprint arXiv:2212.03864, 2022.
- [199] Denis Tarasov, Vladislav Kurenkov, and Sergey Kolesnikov. Prompts and pre-trained language models for ofﬂine reinforcement learning. In ICLR Workshop on Generalizable Policy Learning in Physical World, 2022.
- [200] Machel Reid, Yutaro Yamada, and Shixiang Shane Gu. Can Wikipedia help ofﬂine reinforcement learning? arXiv preprint arXiv:2201.12122, 2022.
- [201] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [202] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In ICLR, 2023.
- [203] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative Agents: Interactive simulacra of human behavior. In ACM Symposium on User Interface Software and Technology, 2023.
- [204] Zhou Xian, Theophile Gervet, Zhenjia Xu, Yi-Ling Qiao, Tsun-Hsuan Wang, and Yian Wang. Towards generalist robots: A promising paradigm via generative simulation. arXiv preprint arXiv:2305.10455, 2023.
- [205] Rutav Shah and Vikash Kumar. RRL: Resnet as representation for reinforcement learning. In ICML, 2021.
- [206] Fei Xia, Amir R. Zamir, Zhi-Yang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson Env: real-world perception for embodied agents. In CVPR, 2018.
- [207] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1K: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation. In CoRL, 2022.
- [208] Manolis Savva*, Abhishek Kadian*, Oleksandr Maksymets*, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A Platform for Embodied AI Research. In ICCV, 2019.

- [209] Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, et al. Habitat 3.0: A co-habitat for humans, avatars and robots. arXiv preprint arXiv:2310.13724, 2023.
- [210] Matt Deitke, Winson Han, Alvaro Herrasti, Aniruddha Kembhavi, Eric Kolve, Roozbeh Mottaghi, Jordi Salvador, Dustin Schwenk, Eli VanderBilt, Matthew Wallingford, et al. RoboTHOR: An open simulationto-real embodied AI platform. In CVPR, 2020.
- [211] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual SLAM. In IROS, 2020.
- [212] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. AirSim: High-ﬁdelity visual and physical simulation for autonomous vehicles. In Field and Service Robotics, 2017.
- [213] Ted Xiao, Harris Chan, Pierre Sermanet, Ayzaan Wahid, Anthony Brohan, Karol Hausman, Sergey Levine, and Jonathan Tompson. Robotic skill acquisition via instruction augmentation with vision-language models. In RSS, 2023.
- [214] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothe´e Lacroix, Baptiste Rozie`re, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efﬁcient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [215] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. AWQ: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978, 2023.
- [216] Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, and Anirudha Majumdar. Robots that ask for help: Uncertainty alignment for large language model planners. In CoRL, 2023.
- [217] Young-Jin Park, Hao Wang, Shervin Ardeshir, and Navid Azizan. Representation reliability and its impact on downstream tasks. arXiv preprint arXiv:2306.00206, 2023.
- [218] Meiqi Sun, Wilson Yan, Pieter Abbeel, and Igor Mordatch. Quantifying uncertainty in foundation models via ensembles. In NeurIPS Workshop on Robustness in Sequence Modeling, 2022.
- [219] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.
- [220] Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. Red teaming language models with language models. In EMNLP, 2022.
- [221] Shengbang Tong, Erik Jones, and Jacob Steinhardt. Mass-producing failures of multimodal systems with language models. arXiv preprint arXiv:2306.12105, 2023.
- [222] Kristofer D Kusano, Kurt Beatty, Scott Schnelle, Francesca Favaro, Cam Crary, and Trent Victor. Collision avoidance testing of the waymo automated driving system. arXiv preprint arXiv:2212.08148, 2022.
- [223] Nick Webb, Dan Smith, Christopher Ludwick, Trent Victor, Qi Hommes, Francesca Favaro, George Ivanov, and Tom Daniel. Waymo’s safety methodologies and safety readiness determinations. arXiv preprint arXiv:2011.00054, 2020.
- [224] Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li, and Ding Zhao. A survey on safety-critical driving scenario generation—a methodological perspective. IEEE ITSC, 2023.
- [225] Rachel Luo, Shengjia Zhao, Jonathan Kuck, Boris Ivanovic, Silvio Savarese, Edward Schmerling, and Marco Pavone. Sample-efﬁcient safety assurances using conformal prediction. In WAFR, 2022.
- [226] Alec Farid, David Snyder, Allen Z Ren, and Anirudha Majumdar. Failure prediction with statistical guarantees for vision-based robot control. arXiv preprint arXiv:2202.05894, 2022.
- [227] Alec Farid, Sushant Veer, Boris Ivanovic, Karen Leung, and Marco Pavone. Task-relevant failure detection for trajectory predictors in autonomous vehicles. In CoRL, 2022.
- [228] Kai-Chieh Hsu, Allen Z Ren, Duy P Nguyen, Anirudha Majumdar, and Jaime F Fisac. Sim-to-Lab-to-Real: Safe reinforcement learning with shielding and generalization guarantees. Artiﬁcial Intelligence, 314:103811, 2023.
- [229] Kai-Chieh Hsu, Haimin Hu, and Jaime Ferna´ndez Fisac. The safety ﬁlter: A uniﬁed view of safety-critical control in autonomous systems. arXiv preprint arXiv:2309.05837, 2023.

- [230] Alec Farid, Sushant Veer, and Anirudha Majumdar. Task-driven outof-distribution detection with statistical guarantees for robot learning. In CoRL, 2021.
- [231] Ido Greenberg and Shie Mannor. Detecting rewards deterioration in episodic reinforcement learning. In ICML, 2021.
- [232] Feiyang Cai and Xenofon Koutsoukos. Real-time out-of-distribution detection in learning-enabled cyber-physical systems. In ICCPS, 2020.
- [233] Rohan Sinha, Apoorva Sharma, Somrita Banerjee, Thomas Lew, Rachel Luo, Spencer M Richards, Yixiao Sun, Edward Schmerling, and Marco Pavone. A system-level view on out-of-distribution data in robotics. arXiv preprint arXiv:2212.14020, 2022.

