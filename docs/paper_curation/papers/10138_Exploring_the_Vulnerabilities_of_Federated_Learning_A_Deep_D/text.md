## Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks

Pengxin Guo*, Runxi Wang*, Shuang Zeng, Jinjing Zhu, Haoning Jiang, Yanran Wang, Yuyin Zhou, Feifei Wang, Hui Xiong, Fellow, IEEE, and Liangqiong Qu

Project page: pengxin-guo.github.io/FLPrivacy

### arXiv:2503.11514v2[cs.CR]9Jan2026

Abstract—Federated Learning (FL) has emerged as a promising privacy-preserving collaborative model training paradigm without sharing raw data. However, recent studies have revealed that private information can still be leaked through shared gradient information and attacked by Gradient Inversion Attacks (GIA). While many GIA methods have been proposed, a detailed analysis, evaluation, and summary of these methods are still lacking. Although various survey papers summarize existing privacy attacks in FL, few studies have conducted extensive experiments to unveil the effectiveness of GIA and their associated limiting factors in this context. To fill this gap, we first undertake a systematic review of GIA and categorize existing methods into three types, i.e., optimization-based GIA (OP-GIA), generationbased GIA (GEN-GIA), and analytics-based GIA (ANA-GIA). Then, we comprehensively analyze and evaluate the three types of GIA in FL, providing insights into the factors that influence their performance, practicality, and potential threats. Our findings indicate that OP-GIA is the most practical attack setting despite its unsatisfactory performance, while GEN-GIA has many dependencies and ANA-GIA is easily detectable, making them both impractical. Finally, we offer a three-stage defense pipeline to users when designing FL frameworks and protocols for better privacy protection and share some future research directions from the perspectives of attackers and defenders that we believe should be pursued. We hope that our study can help researchers design more robust FL frameworks to defend against these attacks.

Index Terms—Federated Learning, Data Privacy, Gradient Inversion Attacks.

I. INTRODUCTION

Pengxin Guo, Runxi Wang, and Liangqiong Qu are with the School of Computing and Data Science, The University of Hong Kong, Hong Kong 999077, China (e-mail: {guopx, u3637153}@connect.hku.hk, liangqqu@hku.hk).

Shuang Zeng is with the Department of Mathematics, The University of Hong Kong, Hong Kong 999077, China (e-mail: zengsh9@connect.hku.hk).

Jinjing Zhu and Hui Xiong are with the Thrust of Artificial Intelligence, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou 511458, China (email: jzhu706@connect.hkust-gz.edu.cn, xionghui@ust.hk).

Haoning Jiang is with the Department of Electronic and Electrical Engineering, Southern University of Science and Technology, Shenzhen 518055, China (e-mail: 12210308@mail.sustech.edu.cn).

Yanran Wang is with the Department of Biomedical Data Science, Stanford University, Stanford, CA 94305, USA (e-mail: joycewyr@stanford.edu).

Yuyin Zhou is with the Department of Computer Science and Engineering, University of California, Santa Cruz, CA 95064, USA (e-mail: yzhou284@ucsc.edu).

Feifei Wang is with the Department of Electrical and Electronic Engineering, The University of Hong Kong, Hong Kong 999077, China, and also with the Materials Innovation Institute for Life Sciences and Energy (MILES), HKU-SIRI, Shenzhen 518055, China (e-mail: ffwang@eee.hku.hk).

Pengxin Guo and Runxi Wang contributed equally to this work. Corresponding author: Liangqiong Qu.

# F

EDERATED Learning (FL) [1]–[4] is a framework that enables multiple clients to collaboratively train a model

without the need to disclose their private local data. The distinctive features of FL make it particularly suitable for developing machine learning models in privacy sensitive scenarios such as healthcare [5]–[7] and financial services [8], [9]. Although FL is designed to protect data privacy by only sharing model gradients, several studies have shown that attackers can still extract sensitive information about private training data through Gradient Inversion Attacks (GIA) [10], [11], Membership Inference Attacks (MIA) [12], [13], or Property Inference Attacks (PIA) [14], [15]. Among these privacy attacks, GIA, which focuses on reconstructing input data via either an honest-but-curious server that follows the FL protocol but is interested in uncovering input data [10], [11], [16], or a malicious server that may modify the model architecture or parameters sent to the user [17]–[19], has emerged as the most powerful and is the focus of this work.

Many GIA methods have been proposed, but a detailed analysis, evaluation, and summary of these methods is still lacking. Although there are various surveys [20]–[27] summarizing existing privacy attacks in FL, few studies have conducted extensive experiments to reveal the effectiveness of GIA and their associated limiting factors in this context. Among the limited research on this topic, the works [11], [28], [29] are the most closely related to ours. However, these methods focus solely on analyzing partial GIA methods, leaving the applicability of their findings to other types of GIA uncertain.

To fill this gap, we aim to conduct a comprehensive study of GIA, including a literature review, categorization of existing methods, in-depth analysis, and extensive evaluation. Specifically, we first undertake a systematic review of current GIA methods and divide them into three categories (Section II): (1) Optimization-based GIA (OP-GIA): OP-GIA works by minimizing the distance between the received gradients and the gradients computed from dummy data [10], [16], [30]–[35]. (2) Generation-based GIA (GEN-GIA): GEN-GIA utilizes a generator to reconstruct input data [36]–[45]. (3) Analyticsbased GIA (ANA-GIA): ANA-GIA aims to recover input data in closed form using a malicious server [17]–[19], [46]–[50]. Additionally, based on the different optimization components of the GEN-GIA methods, they can be further categorized into (i) optimizing latent vector z [36]–[38], (ii) optimizing gener-

0000–0000/00$00.00 © 2021 IEEE

TABLE I COMPARISON OF DIFFERENT TYPES OF GIA METHODS IN TERMS OF INFLUENCE FACTORS, RECONSTRUCTION RESULTS, AND EXTRA RELIANCE OF EACH TYPE OF METHOD. INFLUENCE FACTORS INCLUDE BATCH SIZE, IMAGE RESOLUTION, MODEL TRAINING STATE, THE NUMBER OF THE SAME LABELS IN ONE BATCH DATA, NETWORK ARCHITECTURE, AND PRACTICAL FEDAVG WITH MULTIPLE LOCAL TRAINING STEPS. RECONSTRUCTION RESULTS INCLUDE WHETHER THE RECONSTRUCTION RESULTS ARE THE ORIGINAL INPUTS AND THE VISUAL QUALITY OF THE RECONSTRUCTION RESULTS.

Influence Factors Reconstruction Results

Taxonomy

Extra Reliance Batch Size

Image Resolution

# Same Label

Model Training State

Network Architecture

Practical FedAvg

Original Inputs? Visual Quality OP-GIA - ✓ ✓ ✓ ✓ ✓ ✓ Yes Low No GEN-GIA

Opti. Lat. Vec. ✗ ✗ ✗ ✗ ✗ ✗ No High Trained Generator Opti. Gen. Para. ✓ ✓ ✓ ✓ ✓ ✓ Yes Middle Sigmoid Activation Train. Inv. Mod. ✓ ✓ ✓ ✗ ✓ ✓ Yes Low Auxiliary Dataset

Manip. Mod. Arch. ✗ ✗ ✗ ✗ ✗ ✗ Yes High Malicious Server Manip. Mod. Para. ✗ ✓ ✗ ✓ ✓ ✗ Yes Middle Malicious Server

ANA-GIA

ator’s parameters W [39]–[42], and (iii) training an inversion generation model using an auxiliary dataset [43], [44]. Based on the type of modification, the ANA-GIA methods can be further categorized into (i) manipulating model architecture

- [17], [19], [46], and (ii) manipulating model parameters [18], [47]–[50]. Concurrent work [27] also classifies existing GIA methods into Optimization-based, Generative Model-based, and Analytic-based attacks, further validating the rationale

behind our classification.

Moreover, we conduct a comprehensive analysis and evaluation of the three types of GIA in FL, with the goal of providing actionable insights into the factors that influence their performance, practicality, and potential risks. Our study is structured to address the following research questions, progressing from foundational factors to a comparative analysis of threats, and finally focusing on strategies for efficient fine-tuning and their implications:

- R1. What are the crucial factors that impact the per-

formance of different GIA methods and their associated underlying mechanisms?

- R2. Among all types of GIA with both honest-but-curious

and malicious adversaries, which type is the most practical and poses the greatest threat to FL?

- R3. With Parameter-Efficient Fine-Tuning (PEFT) tech-

nologies being widely used in FL to fine-tune foundation models [51]–[54], what is the potential privacy leakage of FL under PEFT?

To answer the first two research questions, we conduct extensive experiments on the three types of GIA to uncover the factors that influence GIA performance (Section III). Our results, summarized in Table I, reveal that:

###### (I) OP-GIA is the most practical attack setting, but the

performance is not satisfactory. Specifically, OP-GIA relies on minimal assumptions (i.e., NO extra reliance), making it the most practical attack setting among the three types of GIA. However, it is influenced by common FL training parameters, making it challenging to achieve satisfactory attack performance. Moreover, practical FedAvg [1], where clients train the model locally for multiple iterations before sending updates, itself has the ability to resist OP-GIA (Section III-B).

###### (II) GEN-GIA has many dependencies, making it pose a

minimal threat to FL. Specifically, some GEN-GIA methods (i.e., optimizing latent vector z) can only achieve semantic-

level recovery and heavily rely on the pre-trained generator. Other GEN-GIA methods (i.e., optimizing generator’s parameters W and training an inversion model) can perform pixel-level attacks, but they have strong dependencies, such as reliance on the Sigmoid function and an auxiliary dataset (Section III-C).

(III) ANA-GIA can achieve satisfactory attack performance but is easily detected and defended against by clients. Specifically, while ANA-GIA can achieve satisfactory performance by manipulating model architecture or parameters, the modifications it makes to the network structure make it easily detectable and defendable by clients, thus rendering it impractical (Section III-D).

For the research question R3, we construct the attack methods for FL under PEFT (Section II-D) and evaluate the privacy leakage (Section III-E). Our experimental results reveal that:

(IV) Attackers can breach privacy on low-resolution images but fail with high-resolution ones under PEFT. Specifically, as shown in Figure 12, the attackers achieve relatively good performance on CIFAR-10, CIFAR-100, and CelebA with a small resolution but perform poorly on ImageNet with a large resolution (Section III-E).

Based on our experimental findings, we provide a threestage defense pipeline for users when designing FL frameworks and protocols for better privacy protection: (1) avoid the Sigmoid activation function and use more complicated network architectures during network design, (2) adopt larger batch sizes and multi-step local training in the local training protocol, and (3) implement client-side validation to check for any potential malicious modification to the model architecture and parameters upon receiving the model from the server. By following this pipeline, users can better protect their data privacy when using FL without worrying about being attacked by current GIA methods.

We summarize our contributions as follows.

• We undertake a systematic review of GIA and categorize existing methods into three types: optimization-based GIA (OP-GIA), generation-based GIA (GEN-GIA), and analytics-based GIA (ANA-GIA) (Section II). We also provide a public repository to continually track developments in this fast-evolving field: Awesome-GradientInversion-Attacks.

DLG [10], iDLG [30], SAPAG [56], IG [16], Geng et al. [57], GradInversion [31], CAFE [58], APRIL [59], GradViT [32], Dimitrov et al. [60], ROG [34], CPA [33], iLRG [61], Wang et al. [62], HFG [35], SEER [63], MGIC [64], AFGI [65], GI-SMN [66], GI-NAS [67], DLG-FB [68], HF-GradInv [69], GI-PIP, [70], TGIAs-RO [71], Mj¨olnir [72], NL-SME [73]

Optimization-based GIA (OP-GIA §II-A)

Optimizing Latent Vector z (§II-B1) GIAS [36], GGL [37], GIFD [38]

Generation-based GIA (GEN-GIA §II-B)

###### GIA

Optimizing Generator’s Parameters W (§II-B2) GRNN [39], CGIR [40], CI-Net [41], GIRG [42]

Training an Inversion Generation Model (§II-B3) LTI [43], GLFA [44]

Manipulating Model Architecture (§II-C1) Robbing the Fed [17], Zhao et al. [46], LOKI [19]

Analytics-based GIA (ANA-GIA §II-C)

Pasquini et al. [47], Fishing [48], Trap Weights [18], MKOR [49], Scale-MIA [50]

Manipulating Model Parameters (§II-C2)

Fig. 1. Taxonomy of existing GIA methods. The existing GIA methods can be divided into three types: optimization-based GIA (OP-GIA), which works by minimizing the distance between received gradients and gradients computed from dummy data; generation-based GIA (GEN-GIA), which utilizes a generator to reconstruct input data; and analytics-based GIA (ANA-GIA), which aims to recover input data in closed form. Moreover, GEN-GIA can be further divided into three categories: optimizing the latent vector z, optimizing the generator’s parameters W , and training an inversion generation model. ANA-GIA can be further divided into two categories: manipulating model architecture and manipulating model parameters.

- • We introduce an error bound analysis (Theorem 1) for data reconstruction in OP-GIA, which, for the first time, theoretically proves that the OP-GIA performance is linearly related to the square root of the batch size and image resolution. Furthermore, we propose a gradient similarity proposition (Proposition 1) to investigate the impact of complex FL training parameters, such as model training state and label distributions, on OP-GIA performance (Section II-A).
- • We conduct extensive experiments on the three types of GIA to uncover the factors influencing their performance and many key interesting findings are provided and summarized (Section III). Based on our experimental findings, we provide a three-stage defense pipeline for users when designing FL frameworks and protocols for better privacy protection (Section IV).
- • We further investigate the privacy leakage in FL with PEFT in this work, revealing that attackers can breach privacy on low-resolution images but fail with highresolution ones under PEFT (Sections II-D & III-E).

II. ANALYSIS OF GRADIENT INVERSION ATTACKS

Gradient Inversion Attacks (GIA) are tailored attacks intended for shared gradients in FL [10]. They aim to reconstruct input data using the shared gradients, typically by a potential adversary seeking to reconstruct clients’ private data. Adversaries are classified as honest-but-curious [55], following the FL training protocol while trying to recover private data, or malicious [17], modifying model architecture or parameters to worsen privacy leakage. We divide existing GIA methods into three categories: optimization-based GIA (OP-GIA), generation-based GIA (GEN-GIA), and analyticsbased GIA (ANA-GIA), as illustrated in Figure 1.

A. Optimization-based GIA

Optimization-based GIA (OP-GIA) typically operates under the threat model of an honest-but-curious server to recover

Algorithm 1 Optimization-based GIA

Input: Leaked gradients ∇θL(x∗, y∗), model weight θ, loss function L, distance metric D(·, ·), regulation term ω(·), regularization coefficient λ, optimization learning rate η, and optimization iterations I.

Output: Restored batch data (xˆ, yˆ).

- 1: Label Restoration: ∇θL(x∗, y∗) → yˆ;
- 2: Initialize xˆ0;
- 3: for i = 1 to I do
- 4: f(xˆi−1) = D(∇θL(x∗, y∗), ∇θL(xˆi−1, yˆ)) + λω(xˆi−1);
- 5: xˆi = xˆi−1 − η∇xˆi−1f(xˆi−1);
- 6: end for
- 7: xˆ = xˆI;
- 8: return (xˆ, yˆ).

private data from clients [10]. It aims to reconstruct input data by initializing random dummy data and minimizing the distance between the received gradients and those computed from the dummy data, a process also known as gradient matching [10]. Formally, given a neural network parameters θ and the gradients ∇θL(x∗,y∗) computed with a private data batch (x∗,y∗), the attacker randomly initializes (x,y) and aims to solve the following optimization problem:

D(∇θL(x∗,y∗),∇θL(x,y)) + λω(x), (1)

arg min

x,y

where D(·,·) denotes a distance metric, ω(·) is a regularization term, and λ is a regularization coefficient. Since simultaneously optimizing the input x and the label y in Eq. (1) is challenging [10], [30], [61], this has led to the development of methods that restore the label first, followed by input optimization [30], [31], [61], [62], [74] 1. Then, the whole procedure of OP-GIA can be summarized in Algorithm 1.

Since the introduction of GIA in [10], various methods have been proposed to enhance attack performance by designing more powerful distance metrics or incorporating more complex regularization terms. For example, DLG [10] uses ℓ2distance as D(·,·) and does not use a regularization term.

1Label restoration is not the focus of this work, and we assume that label information is obtainable [16], [30], [31], [61].

IG [16] adopts cosine similarity as D(·,·) and the Total Variance (TV) as ω(·). GradInversion [31] adopts ℓ2-distance as D(·,·) and divides ω(·) into four terms, TV, ℓ2-distance, a Batch Normalization (BN) prior on the input x, and a group consistency regularization term. GI-PIP [70] utilizes cosine similarity as D(·,·) and divides ω(·) into two terms: TV and Anomaly Score (AS). Moreover, CPA [33] uses Independent Component Analysis (ICA) to reconstruct features and aid further reconstruction. ROG [34] employs an encoder to reduce input resolution, shrinking the original optimization space. HFG [35] implements GIA stepwise to enhance reconstruction.

Despite the progress, recent studies reveal that OP-GIA is often influenced by various FL training parameters, such as image resolution and batch size [10], [11]. However, a comprehensive theoretical understanding detailing the reasons behind OP-GIA’s susceptibility to these parameters as well as the extent of their impact (whether linear, exponential, or otherwise), remains unclear. To fill this gap, we provide the following theoretical analysis (the proof is provided in Section I-A in the Supplementary Material) obtained by Algorithm 1, aiming to shed light on the relationship between FL parameters and attack performance.

Theorem 1. If f is µ strong convex and L-smooth, choose step-size η ≤ µ+2L, then Algorithm 1 obtains xˆ satisfying the following convergence guarantees:

2BCHW(µ + L) µ

µ µ + L

∥xˆ−x∗∥2 ≤ (1−

)I∥xˆ0−x∗∥2+

κ,

where C,H,W denote the image resolution, B is the batch size, and κ is the upper bound of ∥∇xˆf(xˆ) −

T t=1 ∇xˆft(xˆ)∥2 for T ≥ 2, where ft is constructed based on model weights θt at t temporal. Here we assume there are T communication rounds in FL, and θt denotes the model weights at time t.

1 T

- Remark 1. Intuitively, Theorem 1 provides a convergence analysis and details the quantitative relationship between the reconstruction error upper bound and both image resolution and batch size. Specifically, the reconstruction error is linearly related to the square root of the batch size and image resolution. Therefore, as image resolution and batch size increase, the attack performance deteriorates, which is also demonstrated in Section III-B in our experiments and other works [10], [16], [71].

Proposition 1. For any two model weights θt

1

and θt

2

, if the leaked gradients of different batch data on θt

1

are more similar than those on θt

2

, i.e., the cardinality of the set {x∗,j : D(∇θt1

L(x∗,i,y∗,i),∇θt1

L(x∗,j,y∗,j) < ϵ} is greater than the cardinality of the set {x∗,j : D(∇θt2

L(x∗,i,y∗,i),∇θt2

L(x∗,j,y∗,j) < ϵ} for any i and ϵ > 0, then recovering the input data using the leaked gradients by Algorithm 1 on θt

1

is harder than on θt

2

.

- Remark 2. Proposition 1 states that for two different models, if the leaked gradients of different batch data on one model are more similar than those on another, then recovering input data using the former’s leaked gradients is much harder. This implies that, in addition to the batch size and image

resolution, the performance of OP-GIA is also influenced by the model training state and the label distribution on each batch size. Specifically, a well-trained model tends to exhibit more similarity in the gradients of different data points compared to an untrained model, resulting in a worse attack performance on the well-trained model. Furthermore, when a batch contains a higher number of identical labels, the gradients become more similar to those of a single image from that particular class, further degrading attack performance. These findings are also validated in Section III-B.

In summary, Theorem 1 and Proposition 1 offer valuable insights into the factors affecting the performance of OPGIA. Theorem 1 lays a strong theoretical foundation for understanding the impact of critical training parameters, such as batch size and image resolution, on a given fixed model. In contrast, Proposition 1 delves deeper into the influence of more complex and general parameters on attack performance by comparing the shared gradients of two distinct models. This insight is particularly valuable as it reveals the interplay between the model training state and attack performance, enabling researchers to identify potential vulnerabilities and devise appropriate countermeasures. Together, these theoretical analyses serve as a powerful tool for guiding future work on OP-GIA methods, allowing researchers to evaluate the effectiveness and limitations of their models’ attack performance more comprehensively.

B. Generation-based GIA

Another type of GIA utilizes a generator to reconstruct the input data, which we refer to as generation-based GIA (GEN-GIA) [36]–[45]. Differing from OP-GIA, which directly optimizes the input data, this type of method uses a generator to produce the input data. Based on the different optimization components of these methods, we divide them into the following categories: 1) optimizing the latent vector z; 2) optimizing the generator’s parameters W; and 3) training an inversion generation model using an auxiliary dataset.

1) Optimizing Latent Vector z: Due to the large search space complicating the optimization process in OP-GIA methods, optimizing the latent vector z of a pre-trained generator is proposed to mitigate this issue, as it significantly reduces the search space [36]–[38]. Formally, the optimizing process can be illustrated as:

D(∇θL(x∗, y∗), ∇θL(G(z, y∗), y∗)) + λω(G(z, y∗)),

arg min

z

(2)

where z denotes the latent vector fed into the generator and the dimension is usually small, G is a pre-trained generator. Here they [36]–[38] assume the label y can be accurately recovered by other method [30], [31], [61], [75]. After reconstructing zˆ using Eq. (2), it is fed into the pre-trained generator G to obtain reconstruction images xˆ via xˆ = G(zˆ,y∗). Despite its promise, there are two limitations for this type of method that cannot be ignored. First, the data distribution of the recovered data should be similar to the data that the generator G was pretrained on [37]. Moreover, since the generator G is pre-

trained and fixed, only semantically similar images can be reconstructed [36]–[38], as illustrated in Section III-C.

2) Optimizing Generator’s Parameters W: Since fixing the generator’s parameters can only produce semantically similar images, some works attempt to optimize the generator’s parameters W to achieve pixel-level attacks [39]–[42], resulting in the following optimization problem:

D(∇θL(x∗, y∗), ∇θL(G(W ; z, y∗), y∗)) + λω(G(W ; z, y∗)), (3)

arg min

W

where W denotes the parameters of the generator, and z denotes the randomly initialized latent vector that remains fixed during training. Differing from the optimization of the latent vector z in Eq. (2), the generator’s parameters are optimized in Eq. (3), allowing for pixel-level similar image reconstruction, as demonstrated in Section III-C. However, such methods heavily rely on the activation functions of the target model [41]. Specifically, this approach only works when the target model adopts the Sigmoid activation function and fails with other activation functions, as validated in Section III-C. This strong dependence limits its practicality.

3) Training an Inversion Generation Model: Unlike the previous two types of GEN-GIA methods that rely on gradient matching, another approach to utilizing the generator model is to train an inversion generation model to generate the input data [43], [44]. This involves optimizing the model’s parameters to learn the mapping from a given set of gradients or other information to reconstruct the original input data. For example, Wu et al. [43] propose Learning to Invert (LTI), which trains an inversion model to reconstruct training samples from their gradients with the assistance of an auxiliary dataset. Xue et al. [44] introduce FGLA, which first extracts the features of each sample in a batch and then directly generates the user data based on a trained generator. Both methods can recover the input images directly based on the given gradients by solving an inverse problem. However, these methods require an auxiliary dataset to train the inversion model, which poses a significant limitation.

C. Analytics-based GIA

In contrast to the OP-GIA and GEN-GIA that recover input data using a gradient matching or inversion function, analytics-based GIA (ANA-GIA) aims to reconstruct the input data in closed form [17]–[19], [47]–[49], [74], [76], [77]. These approaches leverage the characteristic of the linear layer, where the linear composition of the input can be calculated based on the gradients of the weight and bias [76]. Specifically, consider a linear layer y = Wx + b, where W is a weight matrix, b is a bias, and x is the layer’s input. When only a single image xi activates a neuron i, the input xi can be directly computed as xi = ∇WiL ⊘ ∇biL. Here L is the loss function, ⊘ denotes entry-wise division, i is the activated neuron, xi is the input that activates neuron i, and ∇WiL, ∇biL are the weight gradient and bias gradient of the neuron respectively. If the linear layer is at the beginning of a network, the reconstructed data will be the original input. However, exact reconstruction only occurs when a single data sample

activates a neuron. If multiple inputs activate the same neuron, the reconstruction becomes a combination of all activating images, as illustrated in Figure 2.

###### Inputs Linear Layer

###### Reconstruction Results

|[Figure 1]| |
|---|---|
|[Figure 2]<br><br>[Figure 3]| |

|[Figure 4]|
|---|

(Gradients)

|[Figure 5]|
|---|

Fig. 2. Reconstruction results of the linear layer. Neuron i is activated by a single image, resulting in an accurate reconstruction, while neuron j, activated by two images, leads to a reconstruction that is a combination of both images.

To address this issue, a malicious adversary is employed to manipulate model architecture or parameters, ensuring that each neuron is activated by only a single sample whenever possible, thereby facilitating such attacks [17]–[19], [47]–[49]. Based on the type of modification, these ANA-GIA methods can be categorized into 1) manipulating model architecture [17], [19], [46] and 2) manipulating model parameters [18], [47]–[50].

1) Manipulating Model Architecture: Fowl et al. [17] first introduce Robbing the Fed which inserts a specifically designed linear layer at the network’s outset to exactly recover the input data, called “binning”. This layer, formulated as M(x) = ReLU(W∗x+b∗), uses weights to measure a known continuous cumulative density function (CDF) of the input data (e.g., image brightness), with each neuron’s bias acting as a cutoff. The goal of this method is to ensure that only one input activates each “bin”, where the “bin” is defined as the neuron with the largest cutoff that is activated. With welldesigned weights W∗ and biases b∗, the original input xi can be reconstructed as:

xi = ∇Wi

L − ∇bi∗+1L , (4)

L − ∇W∗i+1L ⊘ ∇bi

∗

∗

where W∗i is the i-th row of W∗, i is the activated bin and i+1 is the bin with the next higher cutoff bias. The proposition 1 in [17] (also provided in Section II in the Supplementary Material) shows that the number of exactly recovered data relates to the relationship between number of print bins k and batch size B. However, if we choose a relatively large number of bins, the influence of batch size becomes minimal,

- as shown in Table IV in our experiments. Zhao et al. [46] discuss a fundamental issue in prior work [17] on privacy attacks against FL with secure aggregation, highlighting that treating the aggregate update as a single large batch incurs unnecessary overheads, and they demonstrate that viewing the update as the aggregation of multiple individual updates allows the application of sparsity to alleviate resource overhead. Later, Zhao et al. [19] propose LOKI, which inserts an attack module
- at the start of a model. The module consists of a convolutional layer followed by two fully connected layers, designed to perform successful attacks even under secure aggregation.

LOKI sends customized convolutional kernels to each client as an identity mapping set, allowing the server to separate weight gradients from the clients despite the use of secure aggregation. The server then uses these weight gradients to reconstruct the original data points. Although these methods can achieve exact recovery, they involve modifications to the model architecture, making them easily detectable and defensible by clients [63].

2) Manipulating Model Parameters: Since modifying model architecture is easily detectable, some works propose manipulating only the model parameters to facilitate such attacks [18], [47]–[50]. These methods can be understood as attacks based on gradient sparsity. In this context, while the aggregation protocol nominally runs as intended, all but one of the data points return a zero gradient for certain model parameters, allowing the averaging process to still produce these entries directly [48]. Formally, for a batch of data x = {x1,x2,...,xB} and the corresponding labels y = {y1,y2,...,yB}, the gradients of the loss function L with respect to the model parameters θ on this batched data are:

B

1 B

∇θL xk,yk

∇θL(x,y) =

k=1

 ∇θL xt,yt +

 .

B

1 B

∇θL xk,yk

=

k=1,k̸=t

(5) By deliberately modifying the model parameters θ, the malicious server can make Bk=1,k̸=t ∇θL xk,yk → 0. Thus, it can achieve:

1 B ∇θL xt,yt . (6)

∇θL(x,y) ≈

As a result, the gradient of a single data point is “isolated” from the large batch, breaking the aggregation. For example, Pasquini et al. [47] propose distributing different model weights to different users to tamper with the updates so that their aggregation will leak information about the update of a target user. Wen et al. [48] introduce Fishing, in which the malicious server artificially decreases the confidence of the network’s predictions for the target class, significantly increasing the contribution of the gradient information calculated from data belonging to the selected class. Boenisch et al.

- [18] propose Trap Weights, which re-scale components in the model’s weight matrix to extract individual training data points from a chosen subset of participating users. Shi et al. [50] introduce Scale-MIA, which decomposes the reconstruction task into a two-step process involving the reconstruction of latent space representations, leveraging specially crafted linear layers, followed by the use of a fine-tuned generative decoder to reconstruct the entire input batch. These methods share a similar idea: by manipulating model weights sent to the client to isolate a data point from batch data. However, they can only reconstruct the original input data when the first layer of the target model is a fully connected layer; otherwise, they can only isolate the gradients of a single data point and must be combined with OP-GIA methods to continue recovering

[18], [48]. Moreover, due to the modifications to the model parameters, they can also be detected by clients [63].

- D. Attacks under Parameter-Efficient Fine-Tuning

The aforementioned works have primarily focused on the traditional scenario where clients share the gradients of the entire model with the server. In this section, we consider a more prevalent situation where clients fine-tune a foundation model using Parameter-Efficient Fine-Tuning (PEFT) technologies, such as Low-Rank Adaptation (LoRA) [78]. These approaches involve sharing only the gradients of a small subset of parameters with the server, and there is a lack of research on privacy leakage analysis in these scenarios. To fill this gap, we will explore these issues in this work.

When a client model is fine-tuned using LoRA, we can adopt the following optimization problem to recover the input data:

arg min

W

D(∇θLL(x∗, y∗), ∇θLL(G(W ; z), y∗)) + λω(G(W ; z)), (7)

where G denotes a generator parameterized by W as used in GEN-GIA (Section II-B2), θL represents the parameters of the LoRA matrices, and ω(·) is the regularization term. The regularization term consists of two components: 1) Total Variation, which enhances the quality of reconstructed images, and 2) Patch Consistency, which addresses the misalignment of patches caused by the Vision Transformer [32].

- E. Extension to Practical FedAvg

The aforementioned works focused on reconstructing raw data from known gradients in ideal settings, rather than considering practical environments in production FL. Specifically, these studies often assume training with FedSGD, where clients compute a gradient update on a single local batch of data and then send it to the server. In contrast, real-world FL applications typically build on FedAvg [1], where clients train the model locally for multiple iterations before sending updates. Under FedAvg, reconstructing clients’ private data becomes significantly more challenging [60], [79], which is also demonstrated in our later evaluation section.

III. EVALUATION

In this section, we provide a comprehensive evaluation of privacy leakage in FL across various types of GIA to answer the following research questions:

- R1. What are the crucial factors that impact the perfor-

mance of different types of GIA?

- R2. Among all types of GIA, which type is the most practical

and poses the greatest threat to FL?

- R3. What’s the privacy leakage of FL under PEFT? To answer the research question R1, we divide the influence

factors into two types: data level and model level. At the data level, we investigate the influence of batch size, image resolution, and the number of the same labels within one batch on the performance of each type of GIA. At the model level, we explore the influence of model training state and network architectures on the performance of each type of GIA. For

the research question R2, we utilize the additional reliance on each type of GIA to measure its practicality and combine this with the reconstruction results to demonstrate its threat to FL. For the research question R3, we test the privacy leakage of FL under LoRA fine-tuning. In the following sections, we examine each type of GIA separately in terms of their influence factors, practicality, and threats to FL. Then, we discuss them collectively and provide guidance for users on designing FL models and training protocols to enhance data privacy protection.

A. Experimental Setup

We evaluate the privacy leakage of FL in three nature image classification datasets: CIFAR-10 [80], CIFAR-100 [80], and ImageNet [81], and a facial image dataset: CelebA [82]. For each dataset, we select a subset of 64 images to evaluate the attack performance. There are no same labels in the subset of CIFAR-100, ImageNet, and CelebA. For the CelebA dataset, we choose the images of the first 1,000 celebrities and assign the corresponding number of the celebrity as the label. The illustration of these selected subsets is shown in Figure III.1 in the Supplementary Material. We adopt ResNet18 [83] as the baseline network, except for the experiments comparing different model architectures and attacks under PEFT. Evaluation on other backbones, such as LeNet [84], AlexNet [85], VGG [86], and GoogLeNet [87], is provided in Section V of the Supplementary Material.

We experiment with both honest-but-curious and malicious adversary settings, and focus on image reconstruction tasks due to its widespread interest. For OP-GIA and GEN-GIA, the adversary is an honest-but-curious server, while for ANAGIA, the adversary is a malicious server.

For the OP-GIA, we choose IG [16] as the attack method for evaluation. For optimization, we optimize the attack for 24,000 iterations using Adam optimizer [88], with an initial learning rate 0.1. The learning rate is decayed by a factor of 0.1 at 3/8, 5/8, 7/8 of the optimization. The coefficient of the TV regularization term is 1e-2 for CIFAR-10 and CIFAR-100, and 1e-6 for ImageNet.

For the GEN-GIA, we employ GGL [37], which optimizes the latent vector z, CI-Net [41], which optimizes the generator’s parameters W, and LTI [43], which learns an inversion generation model to evaluate the privacy leakage in FL. For GGL, similar to the experimental settings in [37], we use a pre-trained BigGAN [89] as the generator and the gradientfree Covariance Matrix Adaptation Evolution Strategy (CMAES) [90] as the optimizer. For CI-Net, following the settings in [41], we use the CI-Net as the generator and optimize the attack for 2000 iterations using the Adam optimizer with learning rate 1e-3. For LTI, following the settings in [43], we adopt an MLP model as the generator and optimize the attack for 5000 epochs using the Adam optimizer with learning rate 1e-4.

For the ANA-GIA, we choose Robbing the Fed [17], which manipulates model architectures, and Fishing [48], which manipulates model parameters, as the attack method for evaluation. For Robbing the Fed, we modify the ResNet18 to include an imprint module with a different number of

bins in front to perform attacks. For Fishing, we modify the parameters of the ResNet-18 model, enabling the isolation of gradients and the execution of optimization-based attacks on a single image.

All experiments are conducted on NVIDIA L40S and GeForce RTX 4090 GPUs. The peak signal to noise ratio (PSNR) [91], structural similarity (SSIM) [92], learned perceptual image patch similarity (LPIPS) [93], Jaccard similarity (Jaccard) [34], and relative data leakage value (RDLV) [94] are adopted as the metrics for evaluating the attack performance. Lower LPIPS, higher PSNR, SSIM, Jaccard, and RDLV of reconstructed images indicate better attack performance.

B. Optimization-based GIA

To validate the influence factors of OP-GIA, we evaluate the attack performance of IG [16] on various datasets with different image resolutions and batch sizes. Additionally, we conduct attacks using different network architectures and models at different training states. The reconstruction results are shown in Figures 3(a) and 3(b). Additional evaluation metrics, such as PSNR, LPIPS, Jaccard, and RDLV, are provided in Figures IV.2 and IV.3 in the Supplementary Material. The visualization of reconstruction results are provided in Section IV-A1 in the Supplementary Material. Based on these results, we conclude that:

Takeaway 1: OP-GIA has no additional reliance, but its performance is not satisfactory and is affected by many factors. Larger batch size, higher image resolution, more complicated network architecture, better model training state, and more same labels in one batch lead to worse OP-GIA performance.

0.8

0.8

CIFAR-10-Untrained

ResNet-18 ResNet-34 ResNet-50 ResNet-101

0.7

CIFAR-10-Trained

0.7

CIFAR-100-Untrained

0.6

CIFAR-100-Trained

0.6

ImageNet-Untrained

0.5

SSIM

SSIM

ImageNet-Trained

0.5

0.4

0.3

0.4

0.2

0.3

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a)

(b)

Fig. 3. (a) Reconstruction results of IG evaluated on models in different training states on various datasets with different image resolutions and batch sizes. (b) Reconstruction results of IG with different network architectures on the CIFAR-100 dataset. The shaded region represents the standard deviation. These results show that a larger batch size, higher image resolution, more complicated network architecture, and better model training state lead to worse OP-GIA performance.

Larger batch size, worse OP-GIA performance. According to Figure 3(a), we can see that with the increase in batch size, the attack performance decreases, which is consistent with Theorem 1. Intuitively, the size of the optimization problem in Eq. (1) is O(B × din), where B is the batch size and din denotes the dimensionality of the input data. The difficulty of performing this optimization scales with the dimensionality of the input, preventing OP-GIA from scaling

to high-dimensional inputs. Therefore, as B increases, the attack performance will decrease [10], [16], [71], [95].

Higher image resolution, worse OP-GIA performance. Comparing the attack performance on CIFAR-10/100 (with resolution 3∗32∗32) and ImageNet (with resolution 3∗224∗ 224) in Figure 3(a), we observe that higher image resolution leads to worse attack performance. This phenomenon is consistent with Theorem 1, and the reasoning is similar to the influence of batch size. Moreover, the image resolution has a more significant impact than the batch size. As as shown in Theorem 1, the reconstruction error for OP-GIA is linearly related to the square root of the batch size and image resolution. Since increasing the image resolution involves increasing both the height (H) and width (W), the reconstruction error can be considered to have a linear correlation with the image resolution. This suggests that the image resolution has a more significant impact than the batch size. Moreover, the experimental results on ImageNet with different image resolutions in Figure IV.4 in the Supplementary Material further support this point.

More complicated network architecture, worse OP-GIA performance. From the results in Figure 3(b), we can observe that as the model architecture becomes more complicated (i.e., transitioning from ResNet-18 to ResNet-101), the attack performance degrades. Notably, when the client adopts ResNet50, the attack effectiveness against a batch size of 1 is even worse than using ResNet-18 with a batch size of 64. This is because more complicated network architectures can make the optimization process (as expressed in Eq. (1)) more prone to getting trapped in local minima [71]. As a result, the attack performance decreases.

Better model training state, worse OP-GIA performance. By comparing the attack performance on untrained and welltrained models in Figure 3(a), we observe that the attack performance is worse on the well-trained model. This is because the gradients tend to be more similar for well-trained models, as illustrated in Figure 4, making it difficult to recover private data by comparing gradient values. This observation is consistent with Proposition 1, which states that for two different models, if the leaked gradients of different batch data on one model are more similar than those on another, then recovering input data using the former’s leaked gradients is much harder.

Untrained

40

Trained

30

0.18

20

0.20

0.22

10

0.24

0

1.50 1.45 1.40

10

20 10 0 10 20 30 40

- Fig. 4. t-SNE visualization of gradients of different CIFAR-100 data points on untrained and trained models. It shows that the gradients are more similar for the trained model than the untrained model.

More same labels in one batch, worse OP-GIA performance. We find that the attack performance on CIFAR-10 is inferior to that on CIFAR-100 when the batch size is large,

as shown in Figure 3(a). Since both datasets share the same resolution, we hypothesize that this is due to the higher number of images with the same label in CIFAR-10. To validate this point, we compare the attack performance with different numbers of images having the same label within a batch, and the results are shown in Figure 5. As illustrated in this figure, as the number of images with the same label increases, the attack performance decreases, which is consistent with our hypothesis.

To analyze the reasons for this finding, we compare the gradient relationships between these batch images and individual images, which are shown in Table II. Ii denotes that there are i identical labels in the batch data. This also corresponds to each batch images from left to right in Figure 5. Iij denotes the j-th image in Ii. From Table II, we can observe that as the number of images with the same label increases, the cosine similarity between the gradients of a single image and these batch images also increases (i.e., the first row in Table II). Furthermore, the cosine similarity between the gradients of individual images and the batch images with the same labels are all large (i.e., the last column in Table II). Therefore, we conclude that the reason for the difficulty in recovering batch images with the same labels comes from the similarity of gradients between the individual images and the batch images, which is consistent with the objective Eq. (1) and our gradient similarity Proposition 1. Since the objective function is based on gradient matching, when the gradient between batch images and a single image is similar, it’s hard to distinguish them, which makes recovering the input based on gradient matching more difficult.

TABLE II COSINE SIMILARITY BETWEEN PAIRWISE GRADIENTS. Ii DENOTES THAT THERE ARE i IDENTICAL LABELS IN THE BATCH DATA. Iij DENOTES THE j-TH IMAGE IN Ii. IT SHOWS THAT AS THE NUMBER OF IMAGES WITH THE SAME LABEL INCREASES, THE COSINE SIMILARITY BETWEEN THE GRADIENTS OF A SINGLE IMAGE AND THESE BATCH IMAGES ALSO INCREASES (I.E., THE FIRST ROW).

I1 I2 I3 I4 I11/2/3/4 -0.0305 0.7443 0.9029 0.9541

- I42 - - - 0.9715
- I43 - - - 0.9623
- I44 - - - 0.9454

a) Extension to Practical FedAvg.: Previous evaluations are based on an ideal setting where training is performed using FedSGD where clients compute a gradient update on a single local batch of data, and then send it to the server. In this part, we evaluate the privacy leakage in a practical FedAvg scenario [1] in which clients train the model locally for multiple iterations before sending the updated model back to the server. In this way, the server only observes the aggregates of the client’s local updates. The experimental results of attacking practical FedAvg on the CIFAR-100 dataset are provided in Table III. Strong Simulation means that the server knows the local training information (i.e., learning rate, batch size, epochs, etc.) that can simulate the training process at the server side. Weak Simulation denotes that the server does not know the local training information and adopts different training

horse frog truck ship horse frog truck horse horse frog horse horse horse horse horse horse

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

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

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

PSNR ↑: 22.03, SSIM ↑: 0.7410, LPIPS ↓: 0.0559. PSNR ↑: 19.95, SSIM ↑: 0.6260, LPIPS ↓: 0.0731. PSNR ↑: 18.36, SSIM ↑: 0.5593, LPIPS ↓: 0.0901. PSNR ↑: 14.03, SSIM ↑: 0.4137, LPIPS ↓: 0.1821.

- Fig. 5. Reconstruction results of IG on the CIFAR-10 dataset with a batch size of 4. From left to right, the number of images with the same label are 0, 2, 3, and 4. The first row represents the ground truth, while the second row shows the reconstruction results. These results indicate that more same labels in one batch lead to worse OP-GIA performance.

6, while the complete results can be found in Section IVB1 in the Supplementary Material. From these results we can conclude that when optimizing latent vector z, GEN-GIA can even generate semantically similar images when using random Gaussian noise instead of real gradients, as long as the label information is available, indicating that it is not affected by the factors influencing OP-GIA. However, it heavily relies on the pre-trained generator and only can achieve semantic-level recovery.

hyperparameters to simulate the local training process. No Simulation means that the server makes no simulation at the server side and considers the local updates as a FedSGD gradient to conduct attacks. From these results, we can conclude that practical FedAvg itself has the ability to resist OP-GIA.

TABLE III RECONSTRUCTION RESULTS OF IG ON THE CIFAR-100 DATASET UNDER PRACTICAL FEDAVG. E DENOTES THE NUMBER OF LOCAL TRAINING EPOCHS, AND B REPRESENTS THE BATCH SIZE. THESE RESULTS DEMONSTRATE THAT PRACTICAL FEDAVG ITSELF HAS THE ABILITY TO RESIST OP-GIA.

[Figure 38]

Ground Truth

- E=1, B=8

[Figure 39]

[Figure 40]

[Figure 41]

- E=2, B=1

Strong Simulation Weak Simulation No Simulation E B PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

[Figure 42]

B=1 (untrained)

- 1 1 12.02 0.3031 0.2049 12.26 0.3971 0.1522 11.40 0.2301 0.2252

- 1 8 14.11 0.5026 0.1295 11.75 0.3615 0.1681 11.53 0.1824 0.2512

- 1 16 14.50 0.5050 0.1315 12.26 0.3971 0.1522 11.57 0.1503 0.2751
- 2 1 12.13 0.3653 0.1701 11.08 0.2672 0.2120 11.30 0.2011 0.2372

- 2 8 12.99 0.4252 0.1486 12.36 0.3871 0.1624 11.44 0.1573 0.2877

- 2 16 15.17 0.5261 0.1138 13.67 0.4502 0.1409 11.62 0.1375 0.2880 5 1 11.51 0.3462 0.1816 10.93 0.2647 0.2363 11.48 0.1898 0.2482 5 8 14.58 0.5210 0.1232 11.63 0.3321 0.1928 10.48 0.1666 0.3045 5 16 14.85 0.5253 0.1137 12.22 0.3486 0.1702 11.03 0.1311 0.3281

[Figure 43]

B=64 (trained)

E=5, B=8

(a)

(b)

[Figure 44]

[Figure 45]

B=1 (noise)

Ground Truth

[Figure 46]

[Figure 47]

B=1

B=64 (noise)

(d)

(c)

Specifically, as shown in Table III, when the server does not have access to the training information on the client (i.e., weak simulation and no simulation), the attack performance is poor. Even when the server has knowledge of the local training information and can precisely simulate the training process on the server side, the reconstruction results are still unsatisfactory. This implies that practical FedAvg itself has the ability to resist OP-GIA.

Fig. 6. Reconstruction results of GGL. (a)-(c) Reconstruction results on the ImageNet dataset: (a) with different batch sizes and model training states; (b) under practical FedAvg scenario; (c) with random Gaussian noise. The ground truth for (b) and (c) is similar to (a) and is omitted. (d) Reconstruction results on the CIFAR-100 dataset with a batch size of one and an untrained model. These results show that when optimizing the latent vector z, GEN-GIA can generate semantically similar images and is not affected by the factors influencing OP-GIA. However, it heavily relies on the pre-trained generator and only can achieve semantic-level recovery.

As illustrated in Figure 6, the reconstruction results of the GEN-GIA are not influenced by the factors affecting OPGIA. This is likely because GEN-GIA only recovers a latent vector and then use this vector along with label information as input to a pre-trained generator to produce the reconstructed images via xˆ = G(zˆ,y∗). Consequently, the inferred label information and the pre-trained generator are crucial, while the obtained gradients are less important. To verify this hypothesis, we replace the real gradients in Eq. (2) with random Gaussian noise for gradient matching. As shown in Figure 6(c), even with random Gaussian noise, as long as the label information is available, it is still possible to reconstruct the images, supporting our hypothesis. Note that we assume the label information can be accurately recovered by other methods [30], [31], [61], [75], which is consistent with the assumptions in most GEN-GIA methods [36]–[39], [41], [42]. Moreover, the recovery of label information is not the focus of this work.

C. Generation-based GIA

GEN-GIA can be further divided into three categories: 1) optimizing the latent vector z; 2) optimizing the generator’s parameters W; and 3) training an inversion generation model using an auxiliary dataset. In this section, we evaluate each type of method separately and then provide an overall observation.

1) Optimizing Latent Vector z: As mentioned above, by optimizing the input latent vector z in Eq. (2), GEN-GIA can only generate semantically similar images, not pixellevel reconstructions. This limitation makes numerical metrics such as PSNR, SSIM, and LPIPS unsuitable for evaluating the attack performance. Thus, we primarily rely on visual comparisons. We choose GGL [37] to evaluate the privacy leakage of FL under GEN-GIA when optimizing the latent vector z. Partial reconstruction results are provided in Figure

Additionally, the results for the CIFAR-100 dataset in

Figure 6(d) show that the reconstructed images are not even semantically similar to the input data. This suggests that GENGIA heavily relies on the pre-trained generator. In detail, the generator used in our experiment is pre-trained on the ImageNet dataset, which is dissimilar to CIFAR-100 data, resulting in poor reconstruction for CIFAR-100. This underscores the importance of the pre-trained generator for GENGIA.

Therefore, despite the advantages of GEN-GIA with optimizing latent vector z in not being influenced by many factors, it heavily relies on the pre-trained generator and can only generate data that is semantically similar, rather than reconstructing the original inputs. This implies that GEN-GIA, when optimizing the latent vector z, suffers from significant constraints and poses little threat to FL.

2) Optimizing Generator’s Parameters W: When optimizing the generator’s parameters W, GEN-GIA can reconstruct pixel-wise similar images, thereby enabling pixel-level attacks. We choose CI-Net [41] to evaluate the privacy leakage of FL under GEN-GIA when optimizing the generator’s parameters W. The reconstruction results of CI-Net are provided in Figure 7. Additional evaluation metrics, such as PSNR, LPIPS, Jaccard, and RDLV, are provided in Figures IV.12 and IV.13 in the Supplementary Material. More experimental results regarding varying numbers of samples that share the same label in one batch and other activation functions (e.g., LeakyReLU, RReLU, and GeLU) are shown in Figures IV.15 and IV.14 in the Supplementary Material. These results show that when optimizing the generator’s parameters W, GEN-GIA can achieve pixel-level attacks, but is affected by the factors that influence OP-GIA. Moreover, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions.

1.0

CelebA-Sigmoid

CelebA-ReLU

0.55

0.8

CelebA-Tanh

CIFAR-100-Sigmoid

0.50

CIFAR-100-ReLU

0.6

SSIM

CIFAR-100-Tanh

SSIM

0.45

ImageNet-Sigmoid

0.4

ImageNet-ReLU

0.40

ImageNet-Tanh

ImageNet-64

0.2

ImageNet-128 ImageNet-224

0.35

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a)

(b)

Fig. 7. (a) Reconstruction results of CI-Net evaluated on ResNet-18 with different activation functions on various datasets with different batch sizes. (b) Reconstruction results of CI-Net on ImageNet with different resolutions under the Sigmoid activation function. These results show that GEN-GIA with optimizing the generator’s parameters W is affected by the factors that influence OP-GIA. Moreover, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions.

As shown in Figure 7(a), CI-Net achieves satisfactory reconstruction results on the model with Sigmoid activation function, while it fails to reconstruct with other activation functions. To explore the reasons behind the vulnerability of the Sigmoid activation function, we start by analyzing the input to the activation function, as shown in Figure 8(a). These results show that the inputs to the activation function mostly lie in the range [−2,2]. Surprisingly, the Sigmoid activation function is approximately linear within

this range, while other activation functions (e.g., ReLU and Tanh) are not, as illustrated in Figure 8(b). Thus, we attribute the vulnerability of the Sigmoid activation function to its local linearity. Specifically, the data transformation under the Sigmoid activation function is linear, whereas it is non-linear for other activation functions, making it more vulnerable under the Sigmoid activation function.

- 0

- 1

- 2

6

Sigmoid

ReLU Tanh

4

2

f(x)

0

2

4

1

2 1 0 1 2 x

6

Sigmoid ReLU Tanh

(a)

(b)

- Fig. 8. (a) Distributions of inputs to different activation functions. It shows that the inputs mostly fall within the range of [-2, 2]. (b) Data transformation of different activation functions under the input range [-2, 2]. It shows that the Sigmoid activation function is approximately linear within this range, whereas other activation functions are not.

Therefore, although GEN-GIA optimizes the generator’s parameters W can achieve pixel-level attacks, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions. Since most current models rarely utilize Sigmoid activation functions, this type of attack poses little threat to FL.

1 4 8 16 32 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

SSIM

LeNet-Untrained

LeNet-Trained

ResNet-20-Untrained

ResNet-20-Trained

(a)

1 4 8 16 32 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

SSIM

CIFAR

ImageNet-64

ImageNet-128

(b)

- Fig. 9. (a) Reconstruction results of LTI evaluated on different models with different training states on CIFAR-10 with different batch sizes. (b) Reconstruction results of LTI on different datasets with different resolutions on LeNet. These results show that when training an inversion generation model, GEN-GIA can achieve pixel-level attacks but is influenced by most of the factors that affect OP-GIA, except for the model’s training state.

3) Training an Inversion Generation Model: Here, we choose LTI [43] to evaluate the privacy leakage of FL under GEN-GIA with training an inversion generation model. Reconstruction results are shown in Figure 9. Additional evaluation metrics, such as PSNR, LPIPS, Jaccard, and RDLV, are provided in Figures IV.16 and IV.17 in the Supplementary Material. Reconstruction results for the varying numbers of samples that share the same label in one batch are provided in Figure IV.18 in the Supplementary Material. From these results, we can conclude that when training an inversion generation model, GEN-GIA can achieve pixel-level attacks but is influenced by most of the factors that affect OPGIA, except for the model’s training state. Moreover, such a paradigm relies on an auxiliary dataset with a data distribution similar to the local data to train the inversion

model [43], which is difficult to achieve in real-world applications.

In summary, combining all the experimental results of GENGIA, including optimizing of latent vector z and generator’s parameters W, and training an inversion generation model using an auxiliary dataset, we conclude that:

Takeaway 2: GEN-GIA has many dependencies, which makes it pose a minimal threat to FL. Some GEN-GIA methods (i.e., optimizing latent vector z) can only achieve semantic-level recovery and heavily rely on the pre-trained generator. Other GEN-GIA methods (i.e., optimizing generator’s parameters W and training an inversion model) can perform pixellevel attacks, but they have strong dependencies, such as reliance on the Sigmoid function and an auxiliary dataset.

- D. Analytics-based GIA

The success of ANA-GIA relies on a malicious server that alters the model architecture or the model parameters sent to the client. In this section, we use Robbing the Fed [17], which manipulates model architecture, and Fishing [48], which manipulates model parameters, as examples to demonstrate the attack performance of ANA-GIA.

1) Manipulating Model Architecture: Since ANA-GIA with manipulating model architecture can achieve exact recovery, resulting in reconstruction results that are exactly the same as the original images, we choose the number of reconstruction images as the metric here. The experimental results of Robbing the Fed [17] are illustrated in Table IV. Reconstruction results for the varying numbers of samples that share the same label in one batch are provided in Figure IV.19 in the Supplementary Material. These results show that ANA-GIA, when manipulating model architecture, can achieve great attack performance irrespective of batch size, image resolution, model training state, or the number of same labels in one batch, provided it adopts a relatively large number of bins.

TABLE IV THE NUMBER OF RECONSTRUCTION IMAGES OF ROBBING THE FED WITH 1000 BINS ON DIFFERENT BATCH SIZES. IT SHOWS THAT ANA-GIA, WHEN MANIPULATING MODEL ARCHITECTURE, CAN ACHIEVE GREAT ATTACK PERFORMANCE IRRESPECTIVE OF BATCH SIZE, IMAGE RESOLUTION, OR MODEL TRAINING STATE, PROVIDED IT ADOPTS A RELATIVELY LARGE NUMBER OF BINS.

Batch Size CIFAR-10 CIFAR-100 ImageNet

1 64/64 64/64 64/64 8 63/64 64/64 64/64

32 60/64 61/64 64/64 64 60/64 60/64 61/64

As shown in Table IV, Robbing the Fed achieves impressive attack performance regardless of image resolution or model training state, which aligns with Proposition 1 in [17]. Furthermore, the influence of batch size is minimal if we choose a relatively large number of bins, i.e., 1000. This results in the batch size being smaller than the number of bins, thereby

minimizing the batch size’s influence. Additionally, unlike the OP-GIA which is affected by the number of same labels in a batch, Robbing the Fed is not influenced by this factor, as illustrated in Figure IV.19 in the Supplementary Material. Furthermore, in the case of practical FedAvg scenario [1], the authors further propose a sparse variant of Robbing the Fed utilizing a two-sided activation function (such as Hardtanh) and weight/bias scaling to maintain the same leakage as in FedSGD [17].

Parameters (M)

0.310.620.921.231.85 3.08 4.92 6.15

0.9

0.8

ReconstructedData

0.7

CIFAR-10

0.6

CIFAR-100 ImageNet

0.5

0.4

0.3

0.2

50100150200 300 500 800 1000

Bins

Fig. 10. Proportion of exactly reconstructed images for a batch size of 64 with different numbers of bins, where the upper horizontal axis represents the number of additional parameters introduced. This indicates that there exists a trade-off between attack performance and model size overhead in ANA-GIA when manipulating model architecture.

However, despite these advantages, ANA-GIA with manipulating model architecture is easily detectable due to the modifications it makes to the network structure, rendering it impractical [63]. Moreover, introducing a linear layer can lead to storage and communication overhead. As shown in Figure 10, increasing the number of perfectly reconstructed images requires increasing the number of imprint bins, which, in turn, necessitates an increase in the number of parameters. From these results, we can see that there exists a trade-off between the attack performance and model size overhead in ANA-GIA when manipulating model architecture.

2) Manipulating Model Parameters: As discussed in Section II-C2, ANA-GIA that manipulates model parameters can isolate the gradients of a single data point from a batch, making its performance independent of batch size, and the number of same labels in one batch. In this section, we utilize Fishing [48] as the attack method on the ImageNet dataset to examine other impact factors, such as image resolution, model training state, and network architectures. We further utilize several large batch sizes to show that the attack performance is indeed not influenced by batch size. To test the influence of image resolutions, we resize the data in the ImageNet dataset to 64∗64 and 128∗128 to compare the performance with the original resolution of 224 ∗ 224. The experimental results of Fishing [48] on the ImageNet dataset are shown in Figure 11. Additional evaluation metrics, such as PSNR, LPIPS, Jaccard, and RDLV, are provided in Figures IV.20 and IV.21 in the Supplementary Material.

As the results in Figure 11(a) show, ANA-GIA that manipulates model parameters can achieve satisfactory attack

| |ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.575

0.6

0.550

ImageNet-64-Trained

0.525

ImageNet-64-Untrained

0.5

ImageNet-128-Trained

SSIM

0.500

SSIM

ImageNet-128-Untrained

0.4

ImageNet-224-Trained

0.475

ImageNet-224-Untrained

0.450

0.3

0.425

0.400

16 32 64 Batch Size

16 32 64 Batch Size

(a)

(b)

Fig. 11. Reconstruction results of Fishing on ImageNet with different (a) image resolutions and model training states, and (b) network architectures. These results show that the attack performance of ANA-GIA, which manipulates model parameters, is not affected by batch size but worsens with larger image resolutions, worse model training states, and more complicated model architectures.

performance regardless of batch size. However, performance decreases with increasing image resolution and from trained to untrained models. The impact of the model’s training state has an inverse influence compared to OP-GIA, which may be because a well-trained model has better feature extraction ability, making it more compatible with the feature-fishing strategy used in the Fishing attack. Moreover, as shown in Figure 11(b), the attack performance will also decrease with more complicated network architectures, which is similar to OP-GIA. Additionally, under practical FedAvg scenario [1], performance is maintained because breaking the aggregation makes the batch size equivalent to 1, which eliminates the protection offered by practical FedAvg compared to OP-GIA [48].

Despite achieving satisfactory attack performance, the modifications to the model parameters can also make it detectable and defensible by clients [63]. Moreover, it can only isolate the gradients of a single data point from a batch and must be combined with OP-GIA methods to continue recovering [18], [48], making it not a purely analytics-based method.

Based on the aforementioned experimental results and the analysis of ANA-GIA involving the manipulation of model architecture and parameters, we conclude that:

Takeaway 3: ANA-GIA can achieve satisfactory attack performance but is easily detected and defended against by clients.

- E. Attacks under Parameter-Efficient Fine-Tuning

In the previous sections, we focused on the traditional scenario where clients share the gradients of the entire model with the server. In this section, we evaluate the potential privacy leakage under Parameter-Efficient Fine-Tuning (PEFT) for foundation models to answer the research question R3.

We choose pre-trained ViT [96] as the base models and utilize Low-Rank Adaptation (LoRA) [78] to fine-tune them. In this paradigm, the server receives only the gradients of the LoRA parameters. The reconstruction results using Eq. (7) are shown in Figures 12, while the visualization results are provided in Section IV-D1 in the Supplementary Material. Additional evaluation metrics, such as PSNR, LPIPS, Jaccard, and RDLV, are provided in Figures IV.22 and IV.23 in the

Supplementary Material. More experimental results on ImageNet with different image resolutions are shown in Figure IV.24 in the Supplementary Material. From these results, we can conclude that:

Takeaway 4: Attackers can breach privacy on lowresolution images but fail with high-resolution ones under PEFT. Moreover, smaller pre-trained models are better at protecting privacy.

CIFAR-10

0.55

ViT-tiny

0.550

CIFAR-100 ImageNet CelebA

ViT-base ViT-large

0.50

0.525

0.45

0.500

SSIM

SSIM

0.40

0.475

0.450

0.35

0.425

0.30

0.400

0.25

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a)

(b)

Fig. 12. (a) Reconstruction results of Eq. (7) evaluated on the ViT-base fine-tuned with LoRA on different datasets with different batch sizes. (b) Reconstruction results of Eq. (7) evaluated on different ViT architectures finetuned with LoRA on the CIFAR-100 dataset. These results show that attackers can breach privacy on low-resolution images but fail with high-resolution ones under PEFT. Moreover, smaller models are better at protecting privacy.

Specifically, as shown in Figure 12(a), the attackers achieve relatively good performance on CIFAR-10, CIFAR-100, and CelebA with a small resolution but perform poorly on ImageNet with a large resolution. This indicates that attackers can breach privacy on low-resolution images but fail with highresolution ones. The reconstruction results on ImageNet at different image resolutions, shown in Figure IV.24 in the Supplementary Material, further support this point. Moreover, we evaluate the privacy leakage across different ViT architectures fine-tuned by LoRA, with the results shown in Figure 12(b). According to these results, we find that smaller pre-trained models are better at protecting privacy. This may be because, with a smaller pre-trained model, the LoRA parameters are fewer, resulting in relatively small leaked gradients and less information leakage.

IV. OUTLOOK A. Discussion

From the evaluation, we understand the influence factors of each type of GIA method and their corresponding practicality and threat to FL. Here, we provide an overall discussion of all GIA methods and offer insights on how to defend against each type of method.

OP-GIA is the most practical setting since it has no additional reliance. However, its effectiveness is heavily influenced by many FL training factors, such as batch size, image resolution, the number of same labels in one batch, model training state, and network architectures. To defend against these attacks, we can increase the batch size and choose more complicated network architectures. Furthermore, we find that practical FedAvg with multiple local training steps can inherently resist OP-GIA, indicating that the actual threat posed

by OP-GIA to FL is limited. Therefore, it is recommended to perform multiple local training iterations when using the FL algorithm.

As for GEN-GIA, when optimizing the latent vector z, it can achieve semantic-level recovery as long as the label information is provided. If users do not care about semantic privacy leakage, they do not need to be concerned about such attacks. When optimizing the generator’s parameters W, although it can achieve pixel-level attacks, it only works when the target model uses Sigmoid activation functions and fails with other activation functions. Users can simply choose different activation functions when designing local models to defend against this attack. The third variance of GENGIA, which involves training an inversion generation model, demands an auxiliary dataset with a distribution similar to the local data. This requirement is difficult to satisfy in real applications since the server may not know the distribution of the local data. Furthermore, even with a distributionally aligned auxiliary dataset, the performance of such attacks is unsatisfactory. In summary, these constraints–the generation of semantically similar results, specificity to the Sigmoid activation function, and the unrealistic requirement for an auxiliary dataset- significantly limit GEN-GIA’s real-world threat to FL systems.

For ANA-GIA, although it can achieve satisfactory attack performance, both manipulating model architecture and parameters are easily detected by clients. The client only needs to conduct some simple checks when receiving the model sent from the server to defend against such attacks [63].

Overall, current GIA methods all have their limitations, and if users use the FL training protocol carefully, the privacy leakage of local data can be minimized.

B. Defensive Guidelines

Based on our extensive evaluation and analysis, we propose actionable strategies to protect FL systems against GIA without using complicated defense mechanisms, which usually lead to privacy-utility trade-offs [97]. Existing defense mechanisms against GIA can be broadly categorized into cryptographic methods, such as Secure Multi-party Computing (SMC) [98]– [100] and Homomorphic Encryption (HE) [101]–[103], and perturbation-based methods like Differential Privacy (DP) [104]–[107]. Cryptographic solutions offer strong security guarantees by performing computations on encrypted data, but their high computational and communication overheads make them impractical for many large-scale deep learning applications [99], [102]. DP-based methods, which involve adding calibrated noise to gradients, are more lightweight but often lead to a degradation in model performance, creating a difficult trade-off between privacy and utility [105], [108]. Beyond these theoretically grounded defenses, there are several empirical strategies, such as gradient pruning or masking [10], [11], [37], which zero out small-magnitude gradients; gradient perturbation [19], [109], [110], which adds Gaussian noise to gradients; learning algorithm modification [35], [111], which augments the training data with a designed vicinal distribution [35] or obfuscates the gradients of the sensitive data with

concealed samples [111]; and model modification [112], which enhances the local architecture of arbitrary models by inserting PRECODE, a PRivacy EnhanCing mODulE. However, as demonstrated in their respective studies, most empirical methods also struggle with the inherent privacy-utility tradeoff. In contrast, our work proposes a set of practical guidelines derived from an extensive empirical analysis of GIA vulnerabilities. Rather than applying external defense strategies, our guidelines focus on strategically designing and optimizing the FL training protocol itself to enhance data privacy without compromising model performance.

Specifically, the defender we are discussing here is how to design FL training protocols to safeguard local data privacy. Our recommendations are organized into a three-stage defense pipeline. First, for network design, users should avoid adopting the Sigmoid activation function, as it is particularly vulnerable to GEN-GIA. Moreover, employing more complicated network architectures can make many attack methods more challenging. Second, during the local training protocol, users are encouraged to use a larger batch size and train locally for multiple steps before sending model updates to the server, as this can also hinder attacks. Third, simple client-side validation should be adopted: when receiving the model from the server, users should verify the model architecture and parameters to avoid being attacked by ANA-GIA. By integrating these practices, users can better protect their data privacy when using FL without worrying about being attacked by current GIA methods. The above guidelines can be summarized into the following key takeaway tips:

Takeaway 5: Three-stage defense pipeline: (1) avoid the Sigmoid activation function and use more complicated network architectures during network design, (2) adopt larger batch sizes and multi-step local training in the local training protocol, and (3) implement clientside validation to check for any potential malicious modification to the model architecture and parameters upon receiving the model from the server.

C. Implications for Attack Design

While this work advocates for privacy protection, the experiments conducted may also provide some inspiration for improving attack strategies. Since attack and defense are interactive games that complement each other, a better attack method can promote the development of a better defense. For designing attack methods, researchers should not only consider attack performance but also practical applicability. For example, although current ANA-GIA methods can achieve high attack performance, modifications to the model architecture or parameters make them easily detected and defended against by clients, thus limiting their practicality [63]. Moreover, current GEN-GIA methods depend on factors such as pretrained generators, the Sigmoid activation function, and an auxiliary dataset, which limits their practicality. Thus, when designing attack methods in the future, researchers should prioritize their practicality and minimize reliance on external

factors, similar to OP-GIA. Furthermore, researchers should pay more attention to practical scenarios like FedAvg, where clients train the model locally for multiple iterations before sending updates, rather than focusing solely on FedSGD.

V. CONCLUSION

In this work, we conduct a comprehensive study on various types of GIA in FL. We thoroughly examine and assess existing GIA methods, dividing them into three categories: optimization-based GIA (OP-GIA), generation-based GIA (GEN-GIA), and analytics-based GIA (ANA-GIA). Theoretically, we provide an error bound analysis for data reconstruction and a gradient similarity proposition in the context of OP-GIA. These theoretical analyses serve as powerful tools for guiding future work on OP-GIA methods, allowing researchers to evaluate the effectiveness and limitations of their models’ attack performance more comprehensively. Empirically, we conduct extensive evaluations, revealing that: (1) OP-GIA is the most practical attack setting, but its performance is not satisfactory; (2) GEN-GIA has many dependencies, posing a minimal threat to FL; (3) ANA-GIA can achieve satisfactory attack performance but is easily detected and defended against by clients. Based on these experimental findings, we provide insights on how to defend against GIA in FL. We hope our study can assist researchers in designing more robust FL frameworks to defend against these attacks.

ACKNOWLEDGMENTS

This work was supported by National Natural Science Foundation of China (62306253, T2522030), Early Career Scheme from the Research Grants Council of Hong Kong SAR (27207025, 27204623), and Guangdong Natural Science Fund-General Programme (2024A1515010233).

REFERENCES

- [1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. Int. Conf. Artif. Intell. Stat., 2017, pp. 1273–1282.
- [2] O. Gupta and R. Raskar, “Distributed learning of deep neural network over multiple agents,” J. Network Comput. Appl., vol. 116, pp. 1–8, 2018.
- [3] J. Zhang, S. Zeng, M. Zhang, R. Wang, F. Wang, Y. Zhou, P. P. Liang, and L. Qu, “Flhetbench: Benchmarking device and state heterogeneity in federated learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 12098–12108.
- [4] P. Guo, Y. Wang, W. Li, M. Liu, M. Li, J. Zheng, and L. Qu, “Exploring federated pruning for large language models,” arXiv:2505.13547, 2025.
- [5] A. Sadilek, L. Liu, D. Nguyen, M. Kamruzzaman, S. Serghiou, B. Rader, A. Ingerman, S. Mellem, P. Kairouz, E. O. Nsoesie et al., “Privacy-first health research with federated learning,” npj Digital Med., vol. 4, no. 1, p. 132, 2021.
- [6] R. Yan, L. Qu, Q. Wei, S.-C. Huang, L. Shen, D. Rubin, L. Xing, and Y. Zhou, “Label-efficient self-supervised federated learning for tackling data heterogeneity in medical imaging,” IEEE Trans. Med. Imaging, 2023.
- [7] S. Zeng, P. Guo, S. Wang, J. Wang, Y. Zhou, and L. Qu, “Tackling data heterogeneity in federated learning via loss decomposition,” in Proc. Int. Conf. Med. Image Comput. Comput. Assist. Interv., 2024, pp. 707–717.
- [8] G. Long, Y. Tan, J. Jiang, and C. Zhang, “Federated learning for open banking,” in Federated Learning: Privacy and Incentive. Springer, 2020, pp. 240–254.

- [9] P. Chatterjee, D. Das, and D. B. Rawat, “Federated learning empowered recommendation model for financial consumer services,” IEEE Trans. Consum. Electron., 2023.
- [10] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 32, 2019.
- [11] Y. Huang, S. Gupta, Z. Song, K. Li, and S. Arora, “Evaluating gradient inversion attacks and defenses in federated learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 34, 2021, pp. 7232–7241.
- [12] M. Nasr, R. Shokri, and A. Houmansadr, “Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning,” in Proc. IEEE Symp. Secur. Priv., 2019, pp. 739–753.
- [13] O. Zari, C. Xu, and G. Neglia, “Efficient passive membership inference attack in federated learning,” in Proc. Int. Conf. Neural Inf. Process. Syst. Workshops, 2021.
- [14] Z. Wang, Y. Huang, M. Song, L. Wu, F. Xue, and K. Ren, “Poisoningassisted property inference attack against federated learning,” IEEE Trans. Dependable Secure Comput., 2022.
- [15] X. Luo, Y. Wu, X. Xiao, and B. C. Ooi, “Feature inference attack on model predictions in vertical federated learning,” in Proc. Int. Conf. Data Eng., 2021, pp. 181–192.
- [16] J. Geiping, H. Bauermeister, H. Dr¨oge, and M. Moeller, “Inverting gradients-how easy is it to break privacy in federated learning?” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 33, 2020, pp. 16937– 16947.
- [17] L. Fowl, J. Geiping, W. Czaja, M. Goldblum, and T. Goldstein, “Robbing the fed: Directly obtaining private data in federated learning with modified models,” in Proc. Int. Conf. Learn. Represent., 2022.
- [18] F. Boenisch, A. Dziedzic, R. Schuster, A. S. Shamsabadi, I. Shumailov, and N. Papernot, “When the curious abandon honesty: Federated learning is not private,” in Proc. IEEE Eur. Symp. Secur. Priv., 2023, pp. 175–199.
- [19] J. C. Zhao, A. Sharma, A. R. Elkordy, Y. H. Ezzeldin, S. Avestimehr, and S. Bagchi, “Loki: Large-scale data reconstruction attack against federated learning through model manipulation,” in Proc. IEEE Symp. Secur. Priv. IEEE Computer Society, 2024, pp. 30–30.
- [20] V. Mothukuri, R. M. Parizi, S. Pouriyeh, Y. Huang, A. Dehghantanha, and G. Srivastava, “A survey on security and privacy of federated learning,” Future Gener. Comput. Syst., vol. 115, pp. 619–640, 2021.
- [21] K. N. Kumar, C. K. Mohan, and L. R. Cenkeramaddi, “The impact of adversarial attacks on federated learning: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., 2023.
- [22] C. Zhang, Y. Xie, H. Bai, B. Yu, W. Li, and Y. Gao, “A survey on federated learning,” Knowledge-Based Syst., vol. 216, p. 106775, 2021.
- [23] R. Zhang, S. Guo, J. Wang, X. Xie, and D. Tao, “A survey on gradient inversion: Attacks, defenses and future directions,” in Proc. Int. Joint Conf. Artif. Intell., 2022, pp. 5678–685.
- [24] J. Wen, Z. Zhang, Y. Lan, Z. Cui, J. Cai, and W. Zhang, “A survey on federated learning: challenges and applications,” Int. J. Mach. Learn. Cybern., vol. 14, no. 2, pp. 513–535, 2023.
- [25] Y. Liu, Y. Kang, T. Zou, Y. Pu, Y. He, X. Ye, Y. Ouyang, Y.-Q. Zhang, and Q. Yang, “Vertical federated learning: Concepts, advances, and challenges,” IEEE Trans. Knowl. Data Eng., 2024.
- [26] Y. Shi, O. Kotevska, V. Reshniak, A. Singh, and R. Raskar, “Dealing doubt: Unveiling threat models in gradient inversion attacks under federated learning, a survey and taxonomy,” arXiv:2405.10376, 2024.
- [27] V. Carletti, P. Foggia, C. Mazzocca, G. Parrella, and M. Vento, “{SoK}: Gradient inversion attacks in federated learning,” in Proc. USENIX Secur. Symp., 2025, pp. 6439–6459.
- [28] J. Du, J. Hu, Z. Wang, P. Sun, N. Z. Gong, and K. Ren, “Sok: On gradient leakage in federated learning,” in Proc. USENIX Secur. Symp., 2025.
- [29] I. Baglin, X. Zhu, and S. Hadfield, “Fedlad: Federated evaluation of deep leakage attacks and defenses,” arXiv:2411.03019, 2024.
- [30] B. Zhao, K. R. Mopuri, and H. Bilen, “idlg: Improved deep leakage from gradients,” arXiv:2001.02610, 2020.
- [31] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and P. Molchanov, “See through gradients: Image batch recovery via gradinversion,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 16337–16346.
- [32] A. Hatamizadeh, H. Yin, H. R. Roth, W. Li, J. Kautz, D. Xu, and P. Molchanov, “Gradvit: Gradient inversion of vision transformers,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10021– 10030.
- [33] S. Kariyappa, C. Guo, K. Maeng, W. Xiong, G. E. Suh, M. K. Qureshi, and H.-H. S. Lee, “Cocktail party attack: Breaking aggregation-based

- privacy in federated learning using independent component analysis,” in Proc. Int. Conf. Mach. Learn., 2023, pp. 15884–15899.
- [34] K. Yue, R. Jin, C.-W. Wong, D. Baron, and H. Dai, “Gradient obfuscation gives a false sense of security in federated learning,” in Proc. USENIX Secur. Symp., 2023, pp. 6381–6398.
- [35] Z. Ye, W. Luo, Q. Zhou, Z. Zhu, Y. Shi, and Y. Jia, “Gradient inversion attacks: Impact factors analyses and privacy enhancement,” IEEE Trans. Pattern Anal. Mach. Intell., 2024.
- [36] J. Jeon, K. Lee, S. Oh, J. Ok et al., “Gradient inversion with generative image prior,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 34, 2021, pp. 29898–29908.
- [37] Z. Li, J. Zhang, L. Liu, and J. Liu, “Auditing privacy defenses in federated learning via generative gradient leakage,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10132–10142.
- [38] H. Fang, B. Chen, X. Wang, Z. Wang, and S.-T. Xia, “Gifd: A generative gradient inversion method with feature domain optimization,” in Proc. IEEE Int. Conf. Comput. Vis., 2023, pp. 4967–4976.
- [39] H. Ren, J. Deng, and X. Xie, “Grnn: generative regression neural network—a data leakage attack for federated learning,” ACM Trans. Intell. Syst. Technol., vol. 13, no. 4, pp. 1–24, 2022.
- [40] X. Xu, P. Liu, W. Wang, H.-L. Ma, B. Wang, Z. Han, and Y. Han, “Cgir: Conditional generative instance reconstruction attacks against federated learning,” IEEE Trans. Dependable Secure Comput., vol. 20, no. 6, pp. 4551–4563, 2022.
- [41] C. Zhang, Z. Xiaoman, E. Sotthiwat, Y. Xu, P. Liu, L. Zhen, and Y. Liu, “Generative gradient inversion via over-parameterized networks in federated learning,” in Proc. IEEE Int. Conf. Comput. Vis., 2023, pp. 5126–5135.
- [42] E. Sotthiwat, L. Zhen, C. Zhang, Z. Li, and R. S. M. Goh, “Generative image reconstruction from gradients,” IEEE Trans. Neural Networks Learn. Syst., 2024.
- [43] R. Wu, X. Chen, C. Guo, and K. Q. Weinberger, “Learning to invert: Simple adaptive attacks for gradient inversion in federated learning,” in Proc. Uncertain. Artif. Intell., 2023, pp. 2293–2303.
- [44] D. Xue, H. Yang, M. Ge, J. Li, G. Xu, and H. Li, “Fast generationbased gradient leakage attacks against highly compressed gradients,” in Proc. IEEE INFOCOM - IEEE Con. Comput. Commun., 2023, pp. 1–10.
- [45] H. Gu, X. Zhang, J. Li, H. Wei, B. Li, and X. Huang, “Federated learning vulnerabilities: Privacy attacks with denoising diffusion probabilistic models,” in Proc. ACM Web Conf., 2024, pp. 1149–1157.
- [46] J. C. Zhao, A. R. Elkordy, A. Sharma, Y. H. Ezzeldin, S. Avestimehr, and S. Bagchi, “The resource problem of using linear layer leakage attack in federated learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 3974–3983.
- [47] D. Pasquini, D. Francati, and G. Ateniese, “Eluding secure aggregation in federated learning via model inconsistency,” in Proc. ACM SIGSAC Conf. Comput. Commun. Secur., 2022, pp. 2429–2443.
- [48] Y. Wen, J. A. Geiping, L. Fowl, M. Goldblum, and T. Goldstein, “Fishing for user data in large-batch federated learning via gradient magnification,” in Proc. Int. Conf. Mach. Learn., 2022, pp. 23668– 23684.
- [49] F. Wang, S. Velipasalar, and M. C. Gursoy, “Maximum knowledge orthogonality reconstruction with gradients in federated learning,” in Proc. Winter Conf. Appl. Comput. Vis., 2024, pp. 3884–3893.
- [50] S. Shi, N. Wang, Y. Xiao, C. Zhang, Y. Shi, Y. T. Hou, and W. Lou, “Scale-mia: A scalable model inversion attack against secure federated learning via latent space reconstruction,” in Proc. Network and Distributed System Security Symposium, 2025.
- [51] Y. Sun, Z. Li, Y. Li, and B. Ding, “Improving lora in privacy-preserving federated learning,” in Proc. Int. Conf. Learn. Represent., 2024.
- [52] Y. Yang, G. Long, T. Shen, J. Jiang, and M. Blumenstein, “Dualpersonalizing adapter for federated foundation models,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2024.
- [53] P. Guo, S. Zeng, Y. Wang, H. Fan, F. Wang, and L. Qu, “Selective aggregation for low-rank adaptation in federated learning,” in Proc. Int. Conf. Learn. Represent., 2025.
- [54] W. Zheng, Z. Lin, P. Guo, Y. Zhou, F. Wang, and L. Qu, “Fedvlmbench: Benchmarking federated fine-tuning of vision-language models,” arXiv:2506.09638, 2025.
- [55] O. Goldreich, Foundations of cryptography: volume 2, basic applications. Cambridge university press, 2009.
- [56] Y. Wang, J. Deng, D. Guo, C. Wang, X. Meng, H. Liu, C. Ding, and S. Rajasekaran, “Sapag: A self-adaptive privacy attack from gradients,” arXiv:2009.06228, 2020.

- [57] J. Geng, Y. Mou, F. Li, Q. Li, O. Beyan, S. Decker, and C. Rong, “Towards general deep leakage in federated learning,” arXiv:2110.09074, 2021.
- [58] X. Jin, P.-Y. Chen, C.-Y. Hsu, C.-M. Yu, and T. Chen, “Cafe: Catastrophic data leakage in vertical federated learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 34, 2021, pp. 994–1006.
- [59] J. Lu, X. S. Zhang, T. Zhao, X. He, and J. Cheng, “April: Finding the achilles’ heel on privacy for vision transformers,” in IEEE Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10051–10060.
- [60] D. I. Dimitrov, M. Balunovic, N. Konstantinov, and M. Vechev, “Data leakage in federated averaging,” Tran. Mach. Learn. Res., 2022.
- [61] K. Ma, Y. Sun, J. Cui, D. Li, Z. Guan, and J. Liu, “Instance-wise batch label restoration via gradients in federated learning,” in Proc. Int. Conf. Learn. Represent., 2023.
- [62] Y. Wang, J. Liang, and R. He, “Towards eliminating hard label constraints in gradient inversion attacks,” in Proc. Int. Conf. Learn. Represent., 2024.
- [63] K. Garov, D. I. Dimitrov, N. Jovanovi´c, and M. Vechev, “Hiding in plain sight: Disguising data stealing attacks in federated learning,” in Proc. Int. Conf. Learn. Represent., 2024.
- [64] C. Liu and J. Wang, “Mgic: A multi-label gradient inversion attack based on canny edge detection on federated learning,” arXiv:2403.08284, 2024.
- [65] C. Liu, J. Wang, and D. Yu, “Raf-gi: Towards robust, accurate and fast-convergent gradient inversion attack in federated learning,” arXiv:2403.08383, 2024.
- [66] J. Qian, K. Wei, Y. Wu, J. Zhang, J. Chen, and H. Bao, “Gi-smn: Gradient inversion attack against federated learning without prior knowledge,” in Proc. Int. Conf. Intell. Comput., 2024, pp. 439–448.
- [67] W. Yu, H. Fang, B. Chen, X. Sui, C. Chen, H. Wu, S.-T. Xia, and K. Xu, “Gi-nas: Boosting gradient inversion attacks through adaptive neural architecture search,” IEEE Trans. Inf. Forensics Secur., 2025.
- [68] L. Leite, Y. Santo, B. L. Dalmazo, and A. Riker, “Federated learning under attack: Improving gradient inversion for batch of images,” arXiv:2409.17767, 2024.
- [69] Z. Ye, W. Luo, Q. Zhou, and Y. Tang, “High-fidelity gradient inversion in distributed learning,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 18, 2024, pp. 19983–19991.
- [70] Y. Sun, G. Xiong, X. Yao, K. Ma, and J. Cui, “Gi-pip: Do we require impractical auxiliary dataset for gradient inversion attacks?” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. IEEE, 2024, pp. 4675–4679.
- [71] B. Li, H. Gu, R. Chen, J. Li, C. Wu, N. Ruan, X. Si, and L. Fan, “Temporal gradient inversion attacks with robust optimization,” IEEE Trans. Dependable Secure Comput., 2025.
- [72] X. Liu, S. Cai, Q. Zhou, S. Guo, R. Li, and K. Lin, “Mj¨olnir: Breaking the shield of perturbation-protected gradients via adaptive diffusion,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 25, 2025, pp. 26308– 26316.
- [73] L. Xia, Z. Liu, S. Huang, W. Tang, and X. Liu, “Non-linear trajectory modeling for multi-step gradient inversion attacks in federated learning,” arXiv:2509.22082, 2025.
- [74] J. Zhu and M. B. Blaschko, “R-gap: Recursive gradient attack on privacy,” in Proc. Int. Conf. Learn. Represent., 2021.
- [75] T. Dang, O. Thakkar, S. Ramaswamy, R. Mathews, P. Chin, and F. Beaufays, “Revealing and protecting labels in distributed training,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 34, 2021, pp. 1727– 1738.
- [76] L. T. Phong, Y. Aono, T. Hayashi, L. Wang, and S. Moriai, “Privacypreserving deep learning: Revisited and enhanced,” in Proc. Appl. Tech. Info. Secur., 2017, pp. 100–110.
- [77] D. I. Dimitrov, M. Baader, M. N. M¨uller, and M. Vechev, “Spear: Exact gradient inversion of batches in federated learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2024.
- [78] E. J. Hu, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent., 2022.
- [79] F. Wang, E. Hugh, and B. Li, “More than enough is too much: Adaptive defenses against gradient leakage in production federated learning,” in Proc. IEEE INFOCOM - IEEE Con. Comput. Commun., 2023, pp. 1– 10.
- [80] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.
- [81] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2009, pp. 248–255.

- [82] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 3730–3738.
- [83] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.
- [84] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 2002.
- [85] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in Proc. Int. Conf. Neural Inf. Process. Syst., vol. 25, 2012.
- [86] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv:1409.1556, 2014.
- [87] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2015, pp. 1–9.
- [88] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. Int. Conf. Learn. Represent., 2015.
- [89] A. Brock, J. Donahue, and K. Simonyan, “Large scale gan training for high fidelity natural image synthesis,” in Proc. Int. Conf. Learn. Represent., 2019.
- [90] N. Hansen, “The cma evolution strategy: A tutorial,” arXiv:1604.00772, 2016.
- [91] A. Hore and D. Ziou, “Image quality metrics: Psnr vs. ssim,” in Proc. Int. Conf. Pattern Recognit., 2010, pp. 2366–2369.
- [92] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.
- [93] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018, pp. 586–595.
- [94] A. Hatamizadeh, H. Yin, P. Molchanov, A. Myronenko, W. Li, P. Dogra, A. Feng, M. G. Flores, J. Kautz, D. Xu et al., “Do gradient inversion attacks make federated learning unsafe?” IEEE Trans. Med. Imaging, vol. 42, no. 7, pp. 2044–2056, 2023.
- [95] A. T¨orn and A. Zilinskas,ˇ Global optimization. Springer, 1989, vol. 350.
- [96] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent., 2021.
- [97] P. Guo, S. Zeng, W. Chen, X. Zhang, W. Ren, Y. Zhou, and L. Qu, “A new federated learning framework against gradient inversion attacks,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 16, 2025, pp. 16969– 16977.
- [98] A. C. Yao, “Protocols for secure computations,” in Proc. IEEE Symp. Found. Comput. Sci. IEEE, 1982, pp. 160–164.
- [99] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for privacy-preserving machine learning,” in Proc. ACM Conf. Comput. Commun. Secur., 2017, pp. 1175–1191.
- [100] Y. Xu, C. Peng, W. Tan, Y. Tian, M. Ma, and K. Niu, “Non-interactive verifiable privacy-preserving federated learning,” Future Gener. Comput. Syst., vol. 128, pp. 365–380, 2022.
- [101] C. Gentry, A fully homomorphic encryption scheme. Stanford university, 2009.
- [102] X. Zhang, A. Fu, H. Wang, C. Zhou, and Z. Chen, “A privacypreserving and verifiable federated learning scheme,” in Proc. IEEE Int. Conf. Commun., 2020, pp. 1–6.
- [103] J. Park and H. Lim, “Privacy-preserving federated learning using homomorphic encryption,” Appl. Sci., vol. 12, no. 2, p. 734, 2022.
- [104] C. Dwork, “Differential privacy,” in Proc. Int. Colloq. Automata, Lang., Program. Springer, 2006, pp. 1–12.
- [105] R. C. Geyer, T. Klein, and M. Nabi, “Differentially private federated learning: A client level perspective,” arXiv:1712.07557, 2017.
- [106] W. Wei, L. Liu, Y. Wu, G. Su, and A. Iyengar, “Gradient-leakage resilient federated learning,” in Proc. Int. Conf. Distrib. Comput. Syst. IEEE, 2021, pp. 797–807.
- [107] J. Han, L. Wang, Z. Liu, B. Qin, K. Zhang, and W. Li, “Ppfl: Privacy-preserving federated learning based on differential privacy and personalized data transformation,” IEEE Internet Things J., 2025.
- [108] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, “Learning differentially private recurrent language models,” in Proc. Int. Conf. Learn. Represent., 2018.
- [109] W. Wei, L. Liu, M. Loper, K.-H. Chow, M. E. Gursoy, S. Truex, and Y. Wu, “A framework for evaluating gradient leakage attacks in federated learning,” arXiv:2004.10397, 2020.

- [110] F. Wang, E. Hugh, and B. Li, “More than enough is too much: Adaptive defenses against gradient leakage in production federated learning,” IEEE/ACM Trans. Netw., vol. 32, no. 4, pp. 3061–3075, 2024.
- [111] J. Wu, M. Hayat, M. Zhou, and M. Harandi, “Concealing sensitive samples against gradient leakage in federated learning,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 19, 2024, pp. 21717–21725.
- [112] D. Scheliga, P. M¨ader, and M. Seeland, “Precode-a generic model extension to prevent deep gradient leakage,” in Proc. Winter Conf. Appl. Comput. Vis., 2022, pp. 1849–1858.
- [113] S. Bubeck et al., “Convex optimization: Algorithms and complexity,” Found. Trends® Mach. Learn., vol. 8, no. 3-4, pp. 231–357, 2015.
- [114] R. Ge, F. Huang, C. Jin, and Y. Yuan, “Escaping from saddle points—online stochastic gradient for tensor decomposition,” in Proc. Conf. Learn. Theory, 2015, pp. 797–842.

#### Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks

##### Supplementary Material

I. THEORETICAL PROOF A. Proof of Theorem 1

Proof. In order to prove Theorem 1, we first assume that there exists T temporal model weights θt and leaked gradients ∇θtL(x∗,y∗). Consequently, multiple ft(·) can be constructed, i.e., ft(x) = D(∇θtL(x∗,y∗),∇θtL(x,yˆ))+λω(x). This assumption is practical since clients usually share gradients with the server multiple times, and any of these leaked gradients can be used to conduct an attack.

Then, according to Algorithm 1, we have:

∥xˆi − x∗∥2

f(xˆi−1) − x∗∥2

= ∥xˆi−1 − η∇xˆi−1

f(xˆi−1) − x∗ − η

= ∥xˆi−1 − η∇xˆi−1

T

T

1 T

1 T

ft(xˆi−1)∥2

∇xˆi−1

∇xˆi−1

ft(xˆi−1) + η

t=1

t=1

T

1 T

≤ ∥xˆi−1 − x∗ − η

∇xˆi−1

ft(xˆi−1)∥2

t=1

T

1 T

+ η∥

∇xˆi−1

f(xˆi−1)∥2

ft(xˆi−1) − ∇xˆi−1

t=1

f¯(xˆi−1)∥2

= ∥xˆi−1 − x∗ − η∇xˆi−1

f¯(xˆi−1) − ∇xˆi−1

f(xˆi−1)∥2,

+ η∥∇xˆi−1

(I.1) where ∇xˆi−1

f¯(xˆi−1) = T1 Tt=1 ∇xˆi−1

ft(xˆi−1). For the first term in Eq. (I.1),

f¯(xˆi−1)∥22

∥xˆi−1 − x∗ − η∇xˆi−1

f¯(xˆi−1)⟩

= ∥xˆi−1 − x∗∥22 − 2η⟨xˆi−1 − x∗,∇xˆi−1

f¯(xˆi−1)∥22.

+ η2∥∇xˆi−1

Since f is L−smooth and µ strong convex, according to Lemma 3.11 in [113], we obtain:

f¯(xˆi−1)⟩ ≥

⟨xˆi−1 − x∗,∇xˆi−1

1 µ + L∥∇xˆi−1

µL µ + L∥xˆi−1 − x∗∥22.

f¯(xˆi−1)∥22 +

Let η ≤ µ+2L, then we get:

f¯(xˆi−1)∥22 ≤ (1 −

∥xˆi−1 − x∗ − η∇xˆi−1

2 µ + L

2µ µ + L

f¯(xˆi−1)∥22 ≤ (1 −

)∥xˆi−1 − x∗∥22 + (η2 −

)∥∇xˆi−1

2µ µ + L

)∥xˆi−1 − x∗∥22.

Since √1 − x ≤ 1 − x2, we get:

µ µ + L

f¯(xˆi−1)∥2 ≤ (1 −

∥xˆi−1 − x∗ − η∇xˆi−1

)∥xˆi−1 − x∗∥2.

(I.2)

For the second term in Eq. (I.1), since the dimension of ∇xf(x) is B × C × H × W, where B is the batch size, C,H,W denote the image resolution, then we have:

f¯(xˆi−1) − ∇xˆi−1f(xˆi−1)∥2

∥∇xˆi−1

f¯(xˆi−1)∥2

= ∥∇xˆi−1f(xˆi−1) − ∇xˆi−1

B×C×H×W

f¯(xˆi−1)]2(j)

[∇xˆi−1f(xˆi−1) − ∇xˆi−1

=

j=1

B×C×H×W

√

f¯(xˆi−1)]2(j)

≤

[∇xˆi−1f(xˆi−1) − ∇xˆi−1

BCHW

j=1

√

=

BCHWκ,

(I.3)

where κ is the upper bound of ∥∇xˆf(xˆ) − ∇xˆf¯(xˆ)∥2, and the inequality is due to Cauchy–Schwarz inequality.

Combine Eqs. (I.1), (I.2), and (I.3), and use η ≤ µ+2L, we have:

µ µ + L

2BCHW µ + L

∥xˆi − x∗∥2 ≤ (1 −

)∥xˆi−1 − x∗∥2 +

κ.

(I.4) Apply Eq. (I.4) recursively for i = 1,...,I, we obtain:

∥xˆ − x∗∥2 ≤ (1 −

2BCHW(µ + L) µ

µ µ + L

)I∥xˆ0 − x∗∥2 +

κ. (I.5)

| |
|---|

B. Proof of Proposition 1

Proof. Suppose the ground truth data (x∗,y∗) and reconstruction results (xˆ,yˆ) obtained by Algorithm 1 satisfy: D(∇θL(x∗,y∗),∇θL(xˆ,yˆ)) < ϵ, ϵ > 0. Then, the set of all possible reconstruction results obtained by Algorithm 1 for model parameters θt

and θt

can be written as:

1

2

- Aˆ1 = {xˆ : D(∇θt1

L(x∗,y∗),∇θt1

L(xˆ,yˆ)) < ϵ},

- Aˆ2 = {xˆ : D(∇θt2

L(x∗,y∗),∇θt2

L(xˆ,yˆ)) < ϵ},

where Aˆ1 and Aˆ2 represent the sets of all possible reconstruction results obtained by Algorithm 1 for θt

, respectively. Then, all the elements in these sets can be considered local minima obtained by Algorithm 1, while only one of them is the optimal solution and the others are not. Thus, the presence of more elements in the set indicates that the reconstruction task is more challenging [114].

and θt

1

2

According to the assumption that the cardinality of the set {x∗,j : D(∇θt1

L(x∗,j,y∗,j) < ϵ} is greater than the cardinality of the set {x∗,j : D(∇θt2

L(x∗,i,y∗,i),∇θt1

L(x∗,j,y∗,j) < ϵ} for any i and

L(x∗,i,y∗,i),∇θt2

ϵ > 0, we have |Aˆ1| > |Aˆ2|, where |A| denotes the cardinality of the set A. This means recovering the input data using the leaked gradients by Algorithm 1 on θt

is harder than on θt

.

1

2

| |
|---|

II. PROPOSITION 1. IN ROBBING THE FED Proposition 2. [17] If the server knows the cumulative density function (assumed to be continuous) of some quantity associated with user data that can be measured with a linear function h : Rm → R, then for a batch of size B and a number of imprint bins k > B > 2, by using an appropriate combination of linear layer and ReLU activation, the server can expect to exactly recover









B−i 2

B−2

1 k + B − 1 k − 1

k − i j

B − i − j − 1 j − 1

k i ·

i ·

+r(B, k)

 

 

 

 

i=1

j=1

samples of user data (where the data is in Rm) perfectly, where

k i

denotes the number of ways to select the i bins that have exactly 1 element, and r(B,k) is a correction term.

III. VISUALIZATION OF SELECTED SUBSET

The illustration of these selected subsets is shown in Figure III.1.

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

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

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

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

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

[Figure 134]

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

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

(a) CIFAR-10. (b) CIFAR-100.

[Figure 176]

[Figure 177]

[Figure 178]

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

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

(c) ImageNet. (d) CelebA.

- Fig. III.1. Visualization of each subset. We resize the images to 64 ∗ 64 for the CelebA dataset. The image resolutions for CIFAR-10 and CIFAR-100 are 32 ∗ 32, while they are 224 ∗ 224 for ImageNet.

IV. MORE EXPERIMENTAL RESULTS A. Optimization-based GIA

The reconstruction results of IG with all evaluation metrics are shown in Figures IV.2 and IV.3. These results show that larger batch size, higher image resolution, more complicated network architecture, and better model training state lead to worse OP-GIA performance.

The reconstruction results of IG on the ImageNet dataset with different resolutions 2 and batch sizes are shown in Figure

2We resize the images to different resolutions.

0.8

| |CIFAR-10-Untrained<br><br>CIFAR-10-Trained<br><br>CIFAR-100-Untrained<br><br>CIFAR-100-Trained<br><br>ImageNet-Untrained<br><br>ImageNet-Trained| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |CIFAR-10-Untrained<br><br>CIFAR-10-Trained<br><br>CIFAR-100-Untrained<br><br>CIFAR-100-Trained<br><br>ImageNet-Untrained<br><br>ImageNet-Trained| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

CIFAR-10-Untrained

22

0.7

CIFAR-10-Trained

0.7

0.6

20

CIFAR-100-Untrained

CIFAR-100-Trained

0.5

0.6

18

ImageNet-Untrained

PSNR

LPIPS

SSIM

0.4

ImageNet-Trained

16

0.5

0.3

14

0.4

0.2

12

0.1

0.3

10

0.0

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

3.0

CIFAR-10-Untrained

CIFAR-10-Untrained

0.8

CIFAR-10-Trained

CIFAR-10-Trained

2.5

CIFAR-100-Untrained

CIFAR-100-Untrained

0.7

CIFAR-100-Trained

CIFAR-100-Trained

2.0

0.6

ImageNet-Untrained

ImageNet-Untrained

Jaccard

RDLV

ImageNet-Trained

ImageNet-Trained

1.5

0.5

0.4

1.0

0.3

0.5

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.2. Reconstruction results of IG evaluated on models in different training states on various datasets with different image resolutions and batch sizes, where the shaded region represents the standard deviation. These results show that a larger batch size, higher image resolution, and better model training state lead to worse OP-GIA performance.

1 4 8 16 32 64 Batch Size

12

14

16

18

20

22

PSNR

ResNet-18 ResNet-34 ResNet-50 ResNet-101

(a) PSNR ↑.

1 4 8 16 32 64 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

SSIM

ResNet-18 ResNet-34 ResNet-50 ResNet-101

(b) SSIM ↑.

1 4 8 16 32 64 Batch Size

0.05

0.10

0.15

0.20

0.25

0.30

LPIPS

ResNet-18 ResNet-34 ResNet-50 ResNet-101

(c) LPIPS ↓.

1 4 8 16 32 64 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

Jaccard

ResNet-18 ResNet-34 ResNet-50 ResNet-101

(d) Jaccard ↑.

1 4 8 16 32 64 Batch Size

0.0

0.5

1.0

1.5

2.0

2.5

RDLV

ResNet-18 ResNet-34 ResNet-50 ResNet-101

(e) RDLV ↑.

- Fig. IV.3. Reconstruction results of IG with different network architectures on the CIFAR-100 dataset, where the shaded region represents the standard deviation. These results show that more complicated network architecture lead to worse OP-GIA performance.

IV.4. From these results, we can see that doubling the image resolution leads to a larger decrease in attack performance compared to doubling the batch size. This suggests that the image resolution has a more significant impact on the performance of OP-GIA than the batch size.

1) Visualization: The visualization of reconstruction results of IG on the CIFAR-10, CIFAR-100, and ImageNet datasets are shown in Figures IV.5, IV.6, and IV.7, respectively.

B. Generation-based GIA

- 1) Optimizing Latent Vector z: The reconstruction results

of GGL are shown in Figures IV.8, IV.9, IV.10, and IV.11. These results show that when optimizing latent vector z, GENGIA can even generate semantically similar images when using random Gaussian noise instead of real gradients, as long as the label information is available, indicating that it is not affected by the factors influencing OP-GIA. However, it heavily relies on the pre-trained generator and only can achieve semantic-level recovery.

- 2) Optimizing Generator’s Parameters W: The reconstruc-

tion results of CI-Net with all evaluation metrics are shown

BS-64 BS-32 BS-16

BS-64 BS-32 BS-16

BS-64 BS-32 BS-16

0.40

0.70

14.0

0.38

0.65

13.5

LPIPS

PSNR

SSIM

0.60

0.36

13.0

0.55

0.34

12.5

0.50

0.32

12.0

0.45

64*64 128*128 224*224 Image Resolution

64*64 128*128 224*224 Image Resolution

64*64 128*128 224*224 Image Resolution

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

0.9

BS-64 BS-32 BS-16

0.50

BS-64 BS-32 BS-16

0.8

0.45

0.7

Jaccard

RDLV

0.40

0.6

0.35

0.5

0.30

0.4

64*64 128*128 224*224 Image Resolution

64*64 128*128 224*224 Image Resolution

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.4. Reconstruction results of IG with different image resolutions and batch sizes on the ImageNet dataset, where the shaded region represents the standard deviation. It shows that the image resolution has a more significant impact on the performance of OP-GIA than the batch size.

[Figure 304]

[Figure 305]

[Figure 306]

(a) Batch size = 1. PSNR ↑: 22.03, SSIM ↑: 0.7410, LPIPS ↓: 0.0559.

(b) Batch size = 32. PSNR ↑: 12.59, SSIM ↑: 0.3561, LPIPS ↓: 0.1634.

(c) Batch size = 64. PSNR ↑: 12.47, SSIM ↑: 0.2985, LPIPS ↓: 0.2110.

- Fig. IV.5. Visualization of reconstruction results of IG on the CIFAR-10 dataset of untrained ResNet-18.

[Figure 307]

(a) Batch size = 1. PSNR ↑: 21.39, SSIM ↑: 0.7448, LPIPS ↓: 0.0572.

[Figure 308]

(b) Batch size = 32. PSNR ↑: 16.72, SSIM ↑: 0.5787, LPIPS ↓: 0.1172.

[Figure 309]

(c) Batch size = 64. PSNR ↑: 14.99, SSIM ↑: 0.5131, LPIPS ↓: 0.1222.

- Fig. IV.6. Visualization of reconstruction results of IG on the CIFAR-100 dataset of untrained ResNet-18.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

(a) Batch size = 1. PSNR ↑: 15.75, SSIM ↑: 0.3979, LPIPS ↓: 0.5933.

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

(b) Batch size = 32. PSNR ↑: 12.32, SSIM ↑: 0.3136, LPIPS ↓: 0.7005.

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

(c) Batch size = 64. PSNR ↑: 12.17, SSIM ↑: 0.3087, LPIPS ↓: 0.7204.

- Fig. IV.7. Visualization of reconstruction results of IG on the ImageNet dataset of untrained ResNet-18.

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

(a) Batch size = 1. (b) Batch size = 32. (c) Batch size = 64.

- Fig. IV.8. Reconstruction results of GGL on the ImageNet dataset of untrained ResNet-18. These results show that the attack performance of GGL is not affected by batch size.

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

(a) Batch size = 1. (b) Batch size = 32. (c) Batch size = 64.

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

- Fig. IV.9. Reconstruction results of GGL on the ImageNet dataset of trained ResNet-18. These results show that the attack performance of GGL is not affected by model training states and batch size.

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

(a) Epoch = 1, batch size

= 8.

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

(b) Epoch = 2, batch size = 1.

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

(c) Epoch = 5, batch size

= 8.

- Fig. IV.10. Reconstruction results of GGL on the ImageNet dataset of untrained ResNet-18 under practical FedAvg scenario. These results show that the attack performance of GGL is not affected by practical FedAvg scenario.

in Figures IV.12, IV.13. Reconstruction results of CI-Net on the CIFAR-100 dataset with a batch size of 64 under different numbers of images with the same label within one batch are shown in Figure IV.15. Reconstruction results of CI-Net on the CIFAR-100 dataset under various activation functions (e.g., Sigmoid, ReLU, Tanh, Leaky-ReLU, RReLU, and GeLU) are shown in Figure IV.14. These results show that when optimizing the generator’s parameters W, GENGIA can achieve pixel-level attacks, but is affected by the factors that influence OP-GIA. Moreover, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions.

3) Training an Inversion Generation Model: The reconstruction results of LTI with all evaluation metrics are shown in Figures IV.16 and IV.17. Reconstruction results of LTI for the varying numbers of samples that share the same label in one batch are provided in Figure IV.18. These results show that when training an inversion generation model, GEN-GIA can achieve pixel-level attacks but is influenced by most of the factors that affect OP-GIA, except for the model’s training state. Furthermore, such a paradigm relies on an auxiliary

indigo bunting eft anole tench

indigo bunting eft tench tench

indigo bunting tench tench

tench tench tench tench tench

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

- Fig. IV.11. Reconstruction results of GGL on the ImageNet dataset with a batch size of 4. From left to right, the number of images with the same label is 0, 2, 3, and 4. The first row represents the ground truth, while the second row shows the reconstruction results. It shows that the reconstruction results of GGL are not affected by the number of images with the same label within one batch.

1 4 8 16 32 64 Batch Size

10

15

20

25

30

PSNR

CelebA-Sigmoid

CelebA-ReLU

CelebA-Tanh

CIFAR-100-Sigmoid

CIFAR-100-ReLU

CIFAR-100-Tanh

ImageNet-Sigmoid

ImageNet-ReLU

ImageNet-Tanh

(a) PSNR ↑.

1 4 8 16 32 64 Batch Size

0.2

0.4

0.6

0.8

1.0

SSIM

CelebA-Sigmoid

CelebA-ReLU

CelebA-Tanh

CIFAR-100-Sigmoid

CIFAR-100-ReLU

CIFAR-100-Tanh

ImageNet-Sigmoid

ImageNet-ReLU

ImageNet-Tanh

(b) SSIM ↑.

|CelebA-Sigmoid CelebA-ReLU CelebA-Tanh| |
|---|---|
|CIFAR-100-Sigmoid CIFAR-100-ReLU CIFAR-100-Tanh ImageNet-Sigmoid ImageNet-ReLU ImageNet-Tanh| |

1 4 8 16 32 64 Batch Size

0.0

0.2

0.4

0.6

0.8

LPIPS

Cele Cele Cele CIFA CIFA CIFA

Ima Ima Ima

(c) LPIPS ↓.

1 4 8 16 32 64 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

Jaccard

CelebA-Sigmoid

CelebA-ReLU

CelebA-Tanh

CIFAR-100-Sigmoid

CIFAR-100-ReLU

CIFAR-100-Tanh

ImageNet-Sigmoid

ImageNet-ReLU

ImageNet-Tanh

(d) Jaccard ↑.

| | |
|---|---|
| | |
| | |

1 4 8 16 32 64 Batch Size

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

RDLV

CelebA-Sigmoid CelebA-ReLU CelebA-Tanh CIFAR-100-Sigmoid CIFAR-100-ReLU CIFAR-100-Tanh ImageNet-Sigmoid ImageNet-ReLU ImageNet-Tanh

(e) RDLV ↑.

- Fig. IV.12. Reconstruction results of CI-Net evaluated on ResNet-18 with different activation functions on various datasets with different batch sizes. These results show that GEN-GIA with optimizing the generator’s parameters W is affected by the factors that influence OP-GIA. Moreover, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions.

1 4 8 16 32 64 Batch Size

13.0

13.5

14.0

14.5

15.0

15.5

16.0

16.5

PSNR

ImageNet-64

ImageNet-128 ImageNet-224

(a) PSNR ↑.

1 4 8 16 32 64 Batch Size

0.35

0.40

0.45

0.50

0.55

SSIM

ImageNet-64

ImageNet-128 ImageNet-224

(b) SSIM ↑.

| |ImageNet-64<br><br>ImageNet-128 ImageNet-224<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1 4 8 16 32 64 Batch Size

0.1

0.2

0.3

0.4

0.5

0.6

0.7

LPIPS

(c) LPIPS ↓.

1 4 8 16 32 64 Batch Size

0.30

0.35

0.40

0.45

0.50

0.55

0.60

0.65

Jaccard

ImageNet-64

ImageNet-128 ImageNet-224

(d) Jaccard ↑.

| |ImageNet-64<br><br>ImageNet-128 ImageNet-224<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1 4 8 16 32 64 Batch Size

0.6

0.8

1.0

1.2

1.4

1.6

RDLV

(e) RDLV ↑.

- Fig. IV.13. Reconstruction results of CI-Net on ImageNet with different resolutions under the Sigmoid activation function. These results show that images with lower resolution are relatively easier to reconstruct for CI-Net.

20

0.9

| |Sigmoid<br><br>ReLU Tanh Leaky-ReLU<br><br>RReLU<br><br>GeLU| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Sigmoid

Sigmoid

0.8

0.35

ReLU Tanh Leaky-ReLU

ReLU Tanh Leaky-ReLU

18

0.7

0.30

0.25

0.6

RReLU

16

RReLU

PSNR

SSIM

GeLU

0.20

LPIPS

0.5

GeLU

0.15

0.4

14

0.10

0.3

0.05

12

0.2

0.00

0.1

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

2.5

| |Sigmoid<br><br>ReLU Tanh Leaky-ReLU<br><br>RReLU<br><br>GeLU| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Sigmoid

0.9

ReLU Tanh Leaky-ReLU

0.8

2.0

0.7

RReLU

1.5

Jaccard

RDLV

0.6

GeLU

1.0

0.5

0.4

0.5

0.3

0.0

0.2

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.14. Reconstruction results of CI-Net on CIFAR-100 under various activation functions. These results further show that when optimizing the generator’s parameters W , GEN-GIA only succeeds when the target model adopts the Sigmoid activation function.

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

(a) No same label. (b) Randomly selected. (c) All same label.

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

(d) No same label. PSNR ↑: 18.15, SSIM ↑: 0.7556, LPIPS ↓: 0.0184.

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

(e) Randomly selected. PSNR ↑: 17.87, SSIM ↑: 0.7365, LPIPS ↓: 0.0221.

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

(f) All same label. PSNR ↑: 14.40, SSIM ↑: 0.4712, LPIPS ↓: 0.0769.

- Fig. IV.15. (a)-(c): Original images on CIFAR-100 with varying numbers of samples that share the same label. (d)-(f): Reconstruction results of CI-Net on CIFAR-100 with varying numbers of samples that share the same label in batches of size 64. These results indicate that more same labels in one batch lead to worse CI-Net performance.

dataset where the data distribution is similar to the local data to train the inversion model, which is difficult to satisfy in real applications.

C. Analytics-based GIA

1) Manipulating Model Architecture: Reconstruction results of Robbing the Fed for the varying numbers of samples that share the same label in one batch are provided in Figure IV.19. These results show that the reconstruction results of

ANA-GIA with manipulating model architecture are now affected by the number of same labels in one batch.

2) Manipulating Model Parameters: The reconstruction results of Fishing with all evaluation metrics are shown in Fig-

LeNet-Untrained

24

LeNet-Untrained

0.8

0.5

LeNet-Trained

LeNet-Trained

22

ResNet-20-Untrained

0.7

ResNet-20-Untrained

ResNet-20-Trained

0.4

ResNet-20-Trained

20

0.6

PSNR

LPIPS

SSIM

18

0.3

0.5

16

0.4

0.2

LeNet-Untrained

LeNet-Trained

14

0.3

0.1

ResNet-20-Untrained

12

ResNet-20-Trained

0.2

1 4 8 16 32 Batch Size

1 4 8 16 32 Batch Size

1 4 8 16 32 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

3.0

LeNet-Untrained

LeNet-Untrained

0.9

LeNet-Trained

LeNet-Trained

0.8

2.5

ResNet-20-Untrained

ResNet-20-Untrained

0.7

ResNet-20-Trained

ResNet-20-Trained

2.0

Jaccard

0.6

RDLV

1.5

0.5

0.4

1.0

0.3

0.5

0.2

1 4 8 16 32 Batch Size

1 4 8 16 32 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.16. Reconstruction results of LTI on CIFAR-10 and different model architectures under different model training states. These results show that a more complicated model architecture and better training state lead to worse LTI performance.

1 4 8 16 32 Batch Size

12

14

16

18

20

22

24

PSNR

CIFAR

ImageNet-64

ImageNet-128

(a) PSNR ↑.

1 4 8 16 32 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

SSIM

CIFAR

ImageNet-64

ImageNet-128

(b) SSIM ↑.

| |CIFAR<br><br>ImageNet-64<br><br>ImageNet-128| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1 4 8 16 32 Batch Size

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

LPIPS

(c) LPIPS ↓.

1 4 8 16 32 Batch Size

0.3

0.4

0.5

0.6

0.7

0.8

0.9

Jaccard

CIFAR

ImageNet-64

ImageNet-128

(d) Jaccard ↑.

1 4 8 16 32 Batch Size

0.5

1.0

1.5

2.0

2.5

3.0

RDLV

CIFAR

ImageNet-64

ImageNet-128

(e) RDLV ↑.

- Fig. IV.17. Reconstruction results of LTI on various datasets with different resolutions. These results show that larger resolutions lead to worse LTI performance.

ures IV.20 and IV.21. These results show that ANA-GIA which manipulates model parameters can achieve satisfactory attack performance regardless of batch size. However, performance decreases with increasing image resolution, from trained to untrained models, and complicated network architectures.

[85], VGG [86], and GoogLeNet [87], to demonstrate the generalizability of our findings.

- A. Optimization-based GIA

The reconstruction results of IG with all evaluation metrics on other backbones are shown in Figure V.29. These results show a similar phenomenon to that observed with ResNet-18, which further demonstrates the generalizability of our findings.

- B. Generation-based GIA

Since the performance of GEN-GIA, when optimizing the latent vector z, is unaffected by the leaked gradients, it indicates that the results are not influenced by the backbones used in the local setup. Additionally, experiments involving the training of an inversion generation model have utilized LeNet. Therefore, we will only present the results of GEN-GIA with optimizing generator parameters W on other backbones in this section.

1) Optimizing Generator’s Parameters W: The reconstruction results of CI-Net with all evaluation metrics using LeNet as backbone are shown in Figure V.30. These results exhibit a phenomenon similar to that observed with ResNet-18, further highlighting the generalizability of our findings.

- C. Analytics-based GIA

Since the performance of ANA-GIA, when manipulating the model architecture, is not affected by the local model—given that it directly adds a linear layer before the local model—we will only present the results of ANA-GIA with manipulated model parameters on other backbones in this section.

1) Manipulating Model Parameters: The reconstruction results of Fishing with all evaluation metrics using GoogLeNet as backbone are shown in Figures V.31. These results reveal a comparable phenomenon to what was seen with ResNet-18, further emphasizing the generalizability of our findings.

D. Attacks under Parameter-Efficient Fine-Tuning

The reconstruction results using Eq. (7) evaluated on different ViT architectures fine-tuned with LoRA on different datasets with all evaluation metrics are shown in Figures IV.22 and IV.23. Reconstruction results on ImageNet with different image resolutions are shown in Figure IV.24. These results show that attackers can breach privacy on low-resolution images but fail with high-resolution ones under PEFT. Moreover, smaller pre-trained models are better at protecting privacy.

1) Visualization: The visualization of reconstruction results using Eq. (7) is shown in Figures IV.25, IV.26, IV.27, and IV.28.

V. EVALUATION ON OTHER BACKBONES

We further evaluate the attack performance of various GIA methods on other backbones, such as LeNet [84], AlexNet

horse frog truck ship horse frog truck horse horse frog horse horse horse horse horse horse

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

PSNR ↑: 21.73, SSIM ↑: 0.6369, LPIPS ↓: 0.1710. PSNR ↑: 18.82, SSIM ↑: 0.5624, LPIPS ↓: 0.1947. PSNR ↑: 16.75, SSIM ↑: 0.4855, LPIPS ↓: 0.1919. PSNR ↑: 12.07, SSIM ↑: 0.3858, LPIPS ↓: 0.2563.

- Fig. IV.18. Reconstruction results of LTI on the CIFAR-10 dataset with a batch size of 4. From left to right, the number of images with the same label are 0, 2, 3, and 4. The first row represents the ground truth, while the second row shows the reconstruction results. These results indicate that more same labels in one batch lead to worse LTI performance.

4/4 images are exactly reconstructed.

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

[Figure 1542]

horse frog truck ship horse frog truck horse horse frog horse horse horse horse horse horse

4/4 images are exactly reconstructed. 4/4 images are exactly reconstructed. 4/4 images are exactly reconstructed.

- Fig. IV.19. Reconstruction results of Robbing the Fed on the CIFAR-10 dataset with a batch size of 4. From left to right, the number of images with same label is 0, 2, 3, and 4. The first row represents the ground truth, while the second row shows the reconstruction results. These results show that the reconstruction results of ANA-GIA with manipulating model architecture are now affected by the number of same labels in one batch.

| |ImageNet-64-Trained<br><br>ImageNet-64-Untrained<br><br>ImageNet-128-Trained<br><br>ImageNet-128-Untrained<br><br>ImageNet-224-Trained<br><br>ImageNet-224-Untrained| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

16 32 64 Batch Size

- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19

PSNR

(a) PSNR ↑.

16 32 64 Batch Size

0.3

0.4

0.5

0.6

SSIM

ImageNet-64-Trained

ImageNet-64-Untrained

ImageNet-128-Trained

ImageNet-128-Untrained

ImageNet-224-Trained

ImageNet-224-Untrained

(b) SSIM ↑.

| |ImageNet-64-Trained<br><br>ImageNet-64-Untrained<br><br>ImageNet-128-Trained<br><br>ImageNet-128-Untrained|
|---|---|
| |ImageNet-224-Trained<br><br>ImageNet-224-Untrained|

16 32 64 Batch Size

0.1

0.2

0.3

0.4

0.5

0.6

0.7

LPIPS

(c) LPIPS ↓.

16 32 64 Batch Size

0.35

0.40

0.45

0.50

0.55

0.60

Jaccard

ImageNet-64-Trained

ImageNet-64-Untrained

ImageNet-128-Trained

ImageNet-128-Untrained

ImageNet-224-Trained

ImageNet-224-Untrained

(d) Jaccard ↑.

| |ImageNet-64-Trained<br><br>ImageNet-64-Untrained<br><br>ImageNet-128-Trained| |
|---|---|---|
| |ImageNet-128-Untrained<br><br>ImageNet-224-Trained<br><br>ImageNet-224-Untrained| |

16 32 64 Batch Size

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

RDLV

(e) RDLV ↑.

- Fig. IV.20. Reconstruction results of Fishing on ImageNet with different image resolutions and model training states. These results show that the attack performance of ANA-GIA, which manipulates model parameters, is not affected by batch size but worsens with larger image resolutions and worse model training states.

16 32 64 Batch Size

14.0

14.5

15.0

15.5

16.0

16.5

17.0

17.5

PSNR

ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50

(a) PSNR ↑.

| |ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

16 32 64 Batch Size

0.400

0.425

0.450

0.475

0.500

0.525

0.550

0.575

SSIM

(b) SSIM ↑.

16 32 64 Batch Size

0.20

0.25

0.30

0.35

0.40

0.45

LPIPS

ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50

(c) LPIPS ↓.

16 32 64 Batch Size

0.30

0.35

0.40

0.45

0.50

0.55

Jaccard

ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50

(d) Jaccard ↑.

| | |
|---|---|
| | |

16 32 64 Batch Size

0.6

0.8

1.0

1.2

1.4

1.6

1.8

RDLV

ImageNet-ResNet-18 ImageNet-ResNet-34 ImageNet-ResNet-50

(e) RDLV ↑.

- Fig. IV.21. Reconstruction results of Fishing on ImageNet with different network architectures. These results show that the attack performance of ANAGIA, which manipulates model parameters, is not affected by batch size but worsens with more complicated model architecture.

CIFAR-10

0.7

CIFAR-10

- 9

- 10

- 11

- 12

- 13

- 14

CIFAR-10

0.55

CIFAR-100 ImageNet CelebA

CIFAR-100 ImageNet CelebA

CIFAR-100 ImageNet CelebA

0.6

0.50

0.5

0.45

PSNR

LPIPS

0.4

SSIM

0.40

0.3

0.35

0.2

0.30

0.1

0.25

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

2.5

| |CIFAR-10<br><br>CIFAR-100 ImageNet CelebA<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.9

0.8

2.0

0.7

CIFAR-10

Jaccard

CIFAR-100 ImageNet CelebA

RDLV

0.6

1.5

0.5

1.0

0.4

0.3

0.5

0.2

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.22. The reconstruction results using Eq. (7) evaluated on ViT-base fine-tuned with LoRA on different datasets with all evaluation metrics. These results show that attackers can breach privacy on low-resolution images but fail with high-resolution ones under PEFT.

1 4 8 16 32 64 Batch Size

12.0

12.5

13.0

13.5

14.0

PSNR

ViT-tiny

ViT-base ViT-large

(a) PSNR ↑.

1 4 8 16 32 64 Batch Size

0.400

0.425

0.450

0.475

0.500

0.525

0.550

SSIM

ViT-tiny

ViT-base ViT-large

(b) SSIM ↑.

1 4 8 16 32 64 Batch Size

0.06

0.08

0.10

0.12

0.14

0.16

LPIPS

ViT-tiny

ViT-base ViT-large

(c) LPIPS ↓.

1 4 8 16 32 64 Batch Size

0.300

0.325

0.350

0.375

0.400

0.425

0.450

0.475

0.500

Jaccard

ViT-tiny

ViT-base ViT-large

(d) Jaccard ↑.

1 4 8 16 32 64 Batch Size

0.7

0.8

0.9

1.0

1.1

1.2

1.3

1.4

1.5

RDLV

ViT-tiny

ViT-base ViT-large

(e) RDLV ↑.

- Fig. IV.23. The reconstruction results using Eq. (7) evaluated on different ViT architectures fine-tuned with LoRA on CIFAR-100 with all evaluation metrics. These results show that smaller pre-trained models are better at protecting privacy.

ImageNet-64

ImageNet-64

ImageNet-64

0.70

0.38

ImageNet-128 ImageNet-224

ImageNet-128 ImageNet-224

ImageNet-128 ImageNet-224

0.65

12.5

0.36

0.60

12.0

0.55

0.34

LPIPS

PSNR

SSIM

0.50

0.32

11.5

0.45

0.30

0.40

11.0

0.35

0.28

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

0.9

| |ImageNet-64<br><br>ImageNet-128 ImageNet-224<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.55

0.8

0.50

0.45

0.7

0.40

Jaccard

RDLV

0.6

0.35

0.30

0.5

ImageNet-64

0.25

ImageNet-128 ImageNet-224

0.4

0.20

0.15

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

- Fig. IV.24. The reconstruction results using Eq. (7) evaluated on different ViT-base fine-tuned with LoRA on ImageNet with different resolutions with all evaluation metrics. These results show that attackers can breach privacy on low-resolution images but fail with high-resolution ones under PEFT.

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

[Figure 1592]

[Figure 1593]

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

(a) Batch size = 1. PSNR ↑: 13.96, SSIM ↑: 0.5185, LPIPS ↓: 0.0803.

[Figure 1607]

[Figure 1608]

[Figure 1609]

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

[Figure 1626]

[Figure 1627]

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

[Figure 1652]

[Figure 1653]

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

[Figure 1668]

[Figure 1669]

[Figure 1670]

(b) Batch size = 32. PSNR ↑: 12.61, SSIM ↑: 0.3886, LPIPS ↓: 0.1334.

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

[Figure 1694]

[Figure 1695]

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

[Figure 1704]

[Figure 1705]

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

[Figure 1722]

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

(c) Batch size = 64. PSNR ↑: 12.06, SSIM ↑: 0.3812, LPIPS ↓: 0.1432.

- Fig. IV.25. Visualization of reconstruction results using Eq. (7) evaluated on the ViT-base fine-tuned with LoRA on the CIFAR-10 dataset.

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

(a) Batch size = 1. PSNR ↑: 13.61, SSIM ↑: 0.5539, LPIPS ↓: 0.0781.

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

[Figure 1824]

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

(b) Batch size = 32. PSNR ↑: 12.71, SSIM ↑: 0.4632, LPIPS ↓: 0.1243.

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

[Figure 1898]

[Figure 1899]

[Figure 1900]

[Figure 1901]

[Figure 1902]

[Figure 1903]

[Figure 1904]

[Figure 1905]

[Figure 1906]

[Figure 1907]

[Figure 1908]

[Figure 1909]

[Figure 1910]

[Figure 1911]

[Figure 1912]

[Figure 1913]

[Figure 1914]

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

(c) Batch size = 64. PSNR ↑: 13.26, SSIM ↑: 0.4918, LPIPS ↓: 0.0965.

- Fig. IV.26. Visualization of reconstruction results using Eq. (7) evaluated on the ViT-base fine-tuned with LoRA on the CIFAR-100 dataset.

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

[Figure 1940]

[Figure 1941]

[Figure 1942]

[Figure 1943]

[Figure 1944]

[Figure 1945]

[Figure 1946]

[Figure 1947]

[Figure 1948]

[Figure 1949]

[Figure 1950]

[Figure 1951]

[Figure 1952]

[Figure 1953]

[Figure 1954]

[Figure 1955]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

[Figure 1962]

[Figure 1963]

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

[Figure 1973]

[Figure 1974]

[Figure 1975]

[Figure 1976]

[Figure 1977]

[Figure 1978]

[Figure 1979]

[Figure 1980]

[Figure 1981]

[Figure 1982]

[Figure 1983]

[Figure 1984]

[Figure 1985]

[Figure 1986]

[Figure 1987]

[Figure 1988]

[Figure 1989]

[Figure 1990]

(a) Batch size = 1. PSNR ↑: 12.38, SSIM ↑: 0.3875, LPIPS ↓: 0.3432.

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2014]

[Figure 2015]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

[Figure 2032]

[Figure 2033]

[Figure 2034]

[Figure 2035]

[Figure 2036]

[Figure 2037]

[Figure 2038]

[Figure 2039]

[Figure 2040]

[Figure 2041]

[Figure 2042]

[Figure 2043]

[Figure 2044]

[Figure 2045]

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

(b) Batch size = 32. PSNR ↑: 11.76, SSIM ↑: 0.3874, LPIPS ↓: 0.4295.

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

[Figure 2071]

[Figure 2072]

[Figure 2073]

[Figure 2074]

[Figure 2075]

[Figure 2076]

[Figure 2077]

[Figure 2078]

[Figure 2079]

[Figure 2080]

[Figure 2081]

[Figure 2082]

[Figure 2083]

[Figure 2084]

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2093]

[Figure 2094]

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

[Figure 2102]

[Figure 2103]

[Figure 2104]

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

(c) Batch size = 64. PSNR ↑: 11.54, SSIM ↑: 0.3805, LPIPS ↓: 0.4548.

- Fig. IV.27. Visualization of reconstruction results using Eq. (7) evaluated on the ViT-base fine-tuned with LoRA on the ImageNet dataset.

[Figure 2119]

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

[Figure 2131]

[Figure 2132]

[Figure 2133]

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2174]

[Figure 2175]

[Figure 2176]

[Figure 2177]

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

[Figure 2206]

[Figure 2207]

[Figure 2208]

[Figure 2209]

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

[Figure 2219]

[Figure 2220]

[Figure 2221]

[Figure 2222]

[Figure 2223]

[Figure 2224]

[Figure 2225]

[Figure 2226]

[Figure 2227]

[Figure 2228]

[Figure 2229]

[Figure 2230]

[Figure 2231]

[Figure 2232]

[Figure 2233]

[Figure 2234]

[Figure 2235]

[Figure 2236]

[Figure 2237]

[Figure 2238]

[Figure 2239]

[Figure 2240]

[Figure 2241]

[Figure 2242]

[Figure 2243]

[Figure 2244]

[Figure 2245]

[Figure 2246]

[Figure 2247]

[Figure 2248]

[Figure 2249]

[Figure 2250]

[Figure 2251]

[Figure 2252]

[Figure 2253]

[Figure 2254]

[Figure 2255]

[Figure 2256]

[Figure 2257]

[Figure 2258]

[Figure 2259]

[Figure 2260]

[Figure 2261]

[Figure 2262]

[Figure 2263]

[Figure 2264]

[Figure 2265]

[Figure 2266]

[Figure 2267]

[Figure 2268]

[Figure 2269]

[Figure 2270]

[Figure 2271]

[Figure 2272]

[Figure 2273]

[Figure 2274]

[Figure 2275]

[Figure 2276]

[Figure 2277]

[Figure 2278]

[Figure 2279]

[Figure 2280]

[Figure 2281]

[Figure 2282]

[Figure 2283]

[Figure 2284]

[Figure 2285]

[Figure 2286]

[Figure 2287]

[Figure 2288]

[Figure 2289]

[Figure 2290]

[Figure 2291]

[Figure 2292]

[Figure 2293]

[Figure 2294]

[Figure 2295]

[Figure 2296]

[Figure 2297]

[Figure 2298]

[Figure 2299]

[Figure 2300]

[Figure 2301]

[Figure 2302]

[Figure 2303]

[Figure 2304]

[Figure 2305]

[Figure 2306]

[Figure 2307]

[Figure 2308]

[Figure 2309]

[Figure 2310]

(a) Batch size = 1. PSNR ↑: 10.27, SSIM ↑: 0.4082, LPIPS ↓: 0.4299.

(b) Batch size = 32. PSNR ↑: 8.79, SSIM ↑: 0.3820, LPIPS ↓: 0.4950.

(c) Batch size = 64. PSNR ↑: 8.38, SSIM ↑: 0.3702, LPIPS ↓: 0.5386.

- Fig. IV.28. Visualization of reconstruction results using Eq. (7) evaluated on the ViT-base fine-tuned with LoRA on the CelebA dataset.

| |LeNet-Untrained<br><br>LeNet-Trained<br><br>AlexNet-Untrained<br><br>AlexNet-Trained<br><br>VGG-16-Untrained<br><br>VGG-16-Trained| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1 4 8 16 32 64 Batch Size

7.5

10.0

12.5

15.0

17.5

20.0

22.5

25.0

PSNR

(a) PSNR ↑.

1 4 8 16 32 64 Batch Size

0.3

0.4

0.5

0.6

0.7

0.8

0.9

SSIM

LeNet-Untrained

LeNet-Trained

AlexNet-Untrained

AlexNet-Trained

VGG-16-Untrained

VGG-16-Trained

(b) SSIM ↑.

1 4 8 16 32 64 Batch Size

0.05

0.10

0.15

0.20

0.25

LPIPS

LeNet-Untrained

LeNet-Trained

AlexNet-Untrained

AlexNet-Trained

VGG-16-Untrained

VGG-16-Trained

(c) LPIPS ↓.

1 4 8 16 32 64 Batch Size

0.2

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

Jaccard

LeNet-Untrained

LeNet-Trained

AlexNet-Untrained

AlexNet-Trained

VGG-16-Untrained

VGG-16-Trained

(d) Jaccard ↑.

1 4 8 16 32 64 Batch Size

0.5

1.0

1.5

2.0

2.5

3.0

RDLV

LeNet-Untrained

LeNet-Trained

AlexNet-Untrained

AlexNet-Trained

VGG-16-Untrained

VGG-16-Trained

(e) RDLV ↑.

- Fig. V.29. Reconstruction results of IG evaluated on different models with different training states on CIFAR-100 with different batch sizes, where the shaded region represents the standard deviation. These results show that a larger batch size and better model training state lead to worse OP-GIA performance.

22

0.8

CIFAR-100-Sigmoid

CIFAR-100-Sigmoid

0.8

0.7

CIFAR-100-ReLU

20

CIFAR-100-ReLU

CIFAR-100-Tanh

0.7

CIFAR-100-Tanh

0.6

CIFAR-100-Sigmoid

ImageNet-Sigmoid

18

ImageNet-Sigmoid

CIFAR-100-ReLU

0.5

0.6

ImageNet-ReLU

ImageNet-ReLU

PSNR

CIFAR-100-Tanh

LPIPS

SSIM

ImageNet-Tanh

16

ImageNet-Tanh

0.4

ImageNet-Sigmoid

0.5

0.3

ImageNet-ReLU

14

0.4

ImageNet-Tanh

0.2

12

0.3

0.1

10

0.0

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

0.9

2.5

CIFAR-100-Sigmoid

CIFAR-100-Sigmoid

CIFAR-100-ReLU

CIFAR-100-ReLU

0.8

CIFAR-100-Tanh

CIFAR-100-Tanh

2.0

0.7

ImageNet-Sigmoid

ImageNet-Sigmoid

ImageNet-ReLU

ImageNet-ReLU

0.6

Jaccard

1.5

RDLV

ImageNet-Tanh

ImageNet-Tanh

0.5

1.0

0.4

0.3

0.5

0.2

1 4 8 16 32 64 Batch Size

1 4 8 16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

Fig. V.30. Reconstruction results of CI-Net evaluated on LeNet with different activation functions on various datasets with different batch sizes. These results show that GEN-GIA with optimizing the generator’s parameters W is affected by the factors that influence OP-GIA. Moreover, it only works when the target model adopts the Sigmoid activation function and fails with other activation functions.

1.0

0.5

- 12

- 13

- 14

- 15

- 16

0.9

0.8

0.4

ImageNet-64-Trained ImageNet-64-Untrained ImageNet-128-Trained ImageNet-128-Untrained ImageNet-224-Trained ImageNet-224-Untrained

ImageNet-64-Trained

| | |
|---|---|
| | |

0.7

ImageNet-64-Untrained

0.3

PSNR

ImageNet-128-Trained

0.6

LPIPS

SSIM

ImageNet-64-Trained

ImageNet-128-Untrained

0.5

ImageNet-64-Untrained

0.2

ImageNet-224-Trained

ImageNet-128-Trained

0.4

ImageNet-224-Untrained

ImageNet-128-Untrained

0.1

0.3

ImageNet-224-Trained

0.2

ImageNet-224-Untrained

0.0

16 32 64 Batch Size

16 32 64 Batch Size

16 32 64 Batch Size

(a) PSNR ↑.

(b) SSIM ↑.

(c) LPIPS ↓.

0.6

1.5

0.5

1.0

ImageNet-64-Trained

0.4

| |ImageNet-64-Trained ImageNet-64-Untrained| |
|---|---|---|
| |ImageNet-128-Trained<br><br>ImageNet-128-Untrained<br><br>ImageNet-224-Trained<br><br>ImageNet-224-Untrained| |

ImageNet-64-Untrained

Jaccard

0.5

0.3

ImageNet-128-Trained

RDLV

ImageNet-128-Untrained

0.2

0.0

ImageNet-224-Trained

ImageNet-224-Untrained

0.1

0.5

0.0

1.0

16 32 64 Batch Size

16 32 64 Batch Size

(d) Jaccard ↑.

(e) RDLV ↑.

Fig. V.31. Reconstruction results of Fishing evaluated on GoogLeNet on ImageNet with different image resolutions and model training states. These results show that the attack performance of ANA-GIA, which manipulates model parameters, is not affected by batch size but worsens with larger image resolutions and worse model training states.

