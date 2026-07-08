# arXiv:2306.07280v3[cs.CV]14Mar2024

## Controlling Text-to-Image Diffusion by Orthogonal Finetuning

Zeju Qiu1,* Weiyang Liu1,2,*,† Haiwen Feng1 Yuxuan Xue3 Yao Feng1 Zhen Liu1,4 Dan Zhang3,5 Adrian Weller2,6 Bernhard Schölkopf1 1MPI for Intelligent Systems - Tübingen 2University of Cambridge 3University of Tübingen 4Mila, Université de Montréal 5Bosch Center for Artificial Intelligence 6The Alan Turing Institute *Equal contribution †Project lead oft.wyliu.com

#### Abstract

Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method – Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.

#### 1 Introduction

Recent text-to-image diffusion models [45, 50, 53] achieve impressive performance in text-guided control for high-fidelity image generation. Despite strong results, text guidance can still be ambiguous and insufficient to provide fine-grained and accurate control to the generated images. To address this shortcoming, we target two types of text-to-image generation tasks in this paper:

- • Subject-driven generation [51]: Given just a few images of a subject, the task is to generate images of the same subject in a different context using the guidance of a text prompt.
- • Controllable generation [38, 68]: Given an additional control signal (e.g., canny edges, segmentation maps), the task is to generate images following such a control signal and a text prompt.

Both tasks essentially boil down to how to effectively finetune text-to-image diffusion models without losing the pretraining generative performance. We summarize the desiderata for an effective finetuning method as: (1) training efficiency: having fewer trainable parameters and number of training epochs, and (2) generalizability preservation: preserving the high-fidelity and diverse generative performance. To this end, finetuning is typically done either by updating the neuron weights by a small learning rate (e.g., [51]) or by adding a small component with re-parameterized neuron weights (e.g., [22, 68]). Despite simplicity, neither finetuning strategy is able to guarantee the preservation of pretraining generative performance. There is also a lack of principled understanding towards designing a good finetuning strategy and finding suitable hyperparameters such as the number of training epochs. A key difficulty is the lack of a measure for quantifying the preservation of pretrained generative ability. Existing finetuning methods implicitly assume that a smaller Euclidean distance between the

This work was finished when ZQ was a research intern hosted by WL at MPI for Intelligent Systems.

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Input images

+199

| |Dream|Boo| | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |LoRA OFT| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Hypersphericalenergy

0.965

0.96

0.955

0.95

0 500 1000 1500 2000 2500 3000

Number of iterations

Text prompt: a [V] dog with a mountain in the background

Text prompt: a house

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

DreamBoothLoRAOFT

ControlNetLoRAOFT

Ground truth

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Control signal

5% data

10% data 50% data

Hypersphericalenergy

+199

Initial energy

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

0.946

0.945

LoRA OFT 5% data 10% data 50% data

Iteration 400 Iteration 1400 Iteration 2200 Iteration 3000

(a) Subject-driven generation (b) Controllable generation

Figure 1: (a) Subject-driven generation: OFT preserves the hyperspherical energy and yields more stable finetuning performance across different number of iterations, while both DreamBooth [51] and LoRA [22] do not. OFT can preserve hyperspherical energy and perform stable finetuning, while both LoRA and DreamBooth are unable. (b) Controllable generation: OFT is more sample-efficient in training and converges well with only 5% of the original dataset, while both ControlNet [68] and LoRA [22] cannot converge until 50% of the data is present. The hyperspherical energy comparison between LoRA and OFT is fair, since they finetune the same layers. ControlNet uses a different layer finetuning strategy, so its hyperspherical energy is not comparable. The detailed settings are given in the experiment section and Appendix A.

finetuned model and the pretrained model indicates better preservation of the pretrained ability. Due to the same reason, finetuning methods typically work with a very small learning rate. While this assumption may occasionally hold, we argue that the Euclidean difference to the pretrained model alone is unable to fully capture the degree of semantic preservation, and therefore a more structural measure to characterize the difference between the finetuned model and the pretrained model can greatly benefit the preservation of pretraining performance as well as finetuning stability.

Inspired by the empirical observation that hyperspherical similarity encodes semantic information well [7, 35, 36], we use hyperspherical energy [32] to characterize the pairwise relational structure among neurons. Hyperspherical energy is defined as the sum of hyperspherical similarity (e.g., cosine similarity) between all pairwise neurons in the same layer, capturing the level of neuron uniformity on the unit hypersphere [34]. We hypothesize that a good finetuned model should have a minimal difference in hyperspherical energy compared to the pretrained model. A naive way is to add a regularizer such that the hyperspherical energy remains the same during the finetuning stage, but there is no guarantee that the hyperspherical energy difference can be well minimized. Therefore, we take advantage of an invariance property of hyperspherical energy – the pairwise hyperspherical similarity is provably preserved if we apply the same orthogonal transformation for all neurons. Motivated by such an invariance, we propose Orthogonal Finetuning (OFT) which adapts large text-to-image diffusion models to a downstream task without changing its hyperspherical energy. The central idea is to learn a layer-shared orthogonal transformation for neurons such that their pairwise angles are preserved. OFT can also be viewed as adjusting the canonical coordinate system for the neurons in the same layer. By jointly taking into consideration that smaller Euclidean distance between the finetuned model and the pretrained model implies better preservation of pretraining performance, we further propose an OFT variant – Constrained Orthogonal Finetuning (COFT) which constrains the finetuned model within the hypersphere of a fixed radius centered on the pretrained neurons.

The intuition for why orthogonal transformation works for finetuning neurons partially comes from 2D Fourier transform, with which an image can be decomposed as magnitude and phase spectrum. The phase spectrum, which is angular information between input and basis, preserves the major part of semantics. For example, the phase spectrum of an image, along with a random magnitude spectrum, can still reconstruct the original image without losing its semantics. This phenomenon suggests that changing the neuron directions is the key to semantically modifying the generated image (which is the goal of both subject-driven and controllable generation). However, changing neuron directions with a large degree of freedom will inevitably destroy the pretraining generative performance. To constrain the degree of freedom, we propose to preserve the angle between any pair of neurons, largely based on the hypothesis that the angles between neurons are crucial for representing the knowledge of neural networks. With this intuition, it is natural to learn layer-shared orthogonal transformation for neurons in each layer such that the hyperspherical energy stays unchanged.

We also draw inspiration from orthogonal over-parameterized training [33] which trains classification neural networks from scratch by orthogonally transforming a randomly initialized neural network. This is because a randomly initialized neural network yields a provably small hyperspherical energy in

expectation and the goal of [33] is to keep hyperspherical energy small during training (small energy leads to better generalization in classification [30, 32]). [33] shows that orthogonal transformation is sufficiently flexible to train generalizable neural networks for classification problems. In contrast, we focus on finetuning text-to-image diffusion models for better controllability and stronger downstream generative performance. We emphasize the difference between OFT and [33] in two aspects. First, while [33] is designed to minimize the hyperspherical energy, OFT aims to preserve the same hyperspherical energy as the pretrained model so that the intrinsic pretrained structure will not be destroyed by finetuning. In the case of finetuning diffusion models, minimizing hyperspherical energy could destroy the original semantic structures. Second, OFT seeks to minimize the deviation from the pretrained model, which leads to the constrained variant. In contrast, [33] imposes no such constraints. The key to finetuning is to find a good trade-off between flexibility and stability, and we argue that our OFT framework effectively achieves this goal. Our contributions are listed below:

- • We propose a novel finetuning method – Orthogonal Finetuning for guiding text-to-image diffusion models towards better controllability. To further improve stability, we propose a constrained variant which limits the angular deviation from the pretrained model.
- • Compared to existing finetuning methods, OFT performs model finetuning while provably preserving the hyperspherical energy, which we empirically find to be an important measure of the generative semantic preservation of the pretrained model.
- • We apply OFT to two tasks: subject-driven generation and controllable generation. We conduct a comprehensive empirical study and demonstrate significant improvement over prior work in terms of generation quality, convergence speed and finetuning stability. Moreover, OFT achieves better sample efficiency, as it converges well with a much smaller number of training images and epochs.
- • For controllable generation, we introduce a new control consistency metric to evaluate the controllability. This core idea is to estimate the control signal from the generated image and then compare it with the origin control signal. The metric further validates the strong controllability of OFT.

#### 2 Related Work

Text-to-image diffusion models. Tremendous progress [16, 39, 45, 50, 53] has been made in text-to-image generation, largely thanks to the rapid development in diffusion-based generative

- models [12, 20, 55, 56] and vision-language representation learning [1, 28, 29, 37, 44, 54, 57, 61]. GLIDE [39] and Imagen [53] train diffusion models in the pixel space. GLIDE trains the text encoder jointly with a diffusion prior using paired text-image data, while Imagen uses a frozen pretrained text encoder. Stable Diffusion [50] and DALL-E2 [45] train diffusion models in the latent space. Stable Diffusion uses VQ-GAN [14] to learn a visual codebook as its latent space, while DALLE2 adopts CLIP [44] to construct a joint latent embedding space for representing images and text. Other than diffusion models, generative adversarial networks [27, 48, 65, 67] and autoregressive
- models [13, 46, 62, 66] have also been studied in text-to-image generation. OFT is inherently a model-agnostic finetuning approach and can be applied to any text-to-image diffusion model.

Subject-driven generation. To prevent subject modification, [2, 39] consider a given mask from users as an additional condition. Inversion methods [8, 12, 15, 45] can be applied to modify the context without changing the subject. [18] can perform local and global editing without input masks. The methods above are unable to well preserve identity-related details of the subject. In Pivotal Tuning [49], a generator is finetuned around an initial inverted latent code with an additional regularization to preserve the identity. Similarly, [41] learns a personalized generative face prior from a collection of a person’s face images. [6] can generate difference variations of an instance, but it may lose the instance-specific details. With a customized token and a few subject images, DreamBooth [51] finetunes the text-to-image diffusion model using a reconstruction loss and a classspecific prior preservation loss. OFT adopts the DreamBooth framework, but instead of performing naive finetuning with a small learning rate, OFT finetunes the model with orthogonal transformations.

Controllable generation. The task of image-to-image translation can be viewed as a form of controllable generation, and previous methods mostly adopt conditional generative adversarial networks [9, 23, 42, 60, 71]. Diffusion models are also used for image-to-image translation [52, 58, 59]. More recently, ControlNet [68] proposes to control a pretrained diffusion model by finetuning and adapting it to additional control signals and achieves impressive controllable generation performance. Another concurrent and similar work, T2I-Adapter [38], also finetunes a pretrained diffusion model in order to gain stronger controllability for the generated images. Following the same task setting

in [38, 68], we apply OFT to finetune pretrained diffusion models, yielding consistently better controllability with fewer training data and less finetuning parameters. More significantly, OFT does not introduce any additional computational overhead during test-time inference.

Model finetuning. Finetuning large pretrained models on downstream tasks has been increasingly popular nowadays [3, 11, 17]. As a form of finetuning, adaptation methods (e.g., [21, 22, 43]) are heavily studied in natural language processing. LoRA [22] is the most relevant work to OFT, and it assumes a low-rank structure for the additive weight update during finetuning. In contrast, OFT uses layer-shared orthogonal transformation to update neuron weights in a multiplicative manner, and it provably preserves the pair-wise angles among neurons in the same layer, yielding better stability.

#### 3 Orthogonal Finetuning

##### 3.1 Why Does Orthogonal Transformation Make Sense?

We start by discussing why orthogonal transformation is desirable in finetuning text-to-image diffusion models. We decompose this question into two smaller ones: (1) why we want to finetune the angle of neurons (i.e., direction), and (2) why we adopt orthogonal transformation to finetune angles.

For the first question, we draw inspiration from the empirical observation in [7, 35] that angular feature difference well characterizes the semantic gap. SphereNet [36] shows that training a neural network with all neurons normalized onto a unit hypersphere yields comparable capacity and even better generalizability, implying that the direction of neurons can fully capture the most important information from data. To better demonstrate the importance of neuron angles, we conduct a toy experiment in Figure 2 where we train a standard convolutional autoencoder on some flower images. In the training stage, we use the standard inner product to produce the feature map (z denotes the element output of the convolution kernel w and x is the input in the sliding window). In the testing stage, we compare three ways to generate the feature map: (a) the inner product used in training, (b) the magnitude information, and (c) the angular information. The results in Figure 2 show that the angular information of neurons can almost perfectly recover the input images, while the magnitude of neurons contains no useful information. We emphasize that we do not apply the cosine activation (c) during training, and the training is done only based on inner product. The results imply that the angles (directions) of neurons play the major role in storing the semantic information of the input images. In order to modify the semantic information of images, finetuning the neuron directions will likely be more effective.

z z z

w x

- (a)
- (b)
- (c)

=

T

= || ||.|| ||

w x

Convolution Deconvolution

cos(θ)

=

z

Feature map

[Figure 29]

[Figure 30]

[Figure 31]

(a) Inner product (b) Magnitude (c) Angle

Figure 2: A toy experiment to demonstrate the importance of angular information. The autoencoder is trained in a standard way using inner product activation, and (a) shows the standard reconstruction. In testing, the angular information of neurons alone can well recover the input image, even if the autoencoder is not trained with angles.

w/ orthogonal w/o orthogonal

For the second question, the simplest way to finetune direction of neurons is to simultaneously rotate / reflect all the neurons (in the same layer), which naturally brings in orthogonal transformation. It may be more flexible to use some other angular transformation that rotates different neurons with different angles, but we find that orthogonal transformation is a sweet spot between flexibility and regularity. Moreover, [33] shows that orthogonal transformation is sufficiently powerful for learning neural networks. To support our argument, we perform an experiment to demonstrate the effective regularization induced by the orthogonality constraint. We perform the controllable generation experiment using the setting of ControlNet [68], and the results are given in Figure 3. We can observe that our standard OFT performs quite stably and achieves accurate control after the training is finished (epoch 20). In comparison, OFT without the orthogonality constraint fails to generate any realistic image and achieve no control effect. The experiment validates the importance of the orthogonality constraint in OFT.

[Figure 32]

[Figure 33]

[Figure 34]

ControlsignalOriginalimage

Epoch1Epoch20

[Figure 35]

[Figure 36]

[Figure 37]

Figure 3: Controllable generation with or without orthogonality. Middle column is from the original OFT, and the right column is given by OFT without the orthogonality constraint.

##### 3.2 General Framework

The conventional finetuning strategy typically uses gradient descent with a small learning rate to update a model (or certain layers of a model). The small learning rate implicitly encourages a small

deviation from the pretrained model, and the standard finetuning essentially aims to train the model while implicitly minimizing ∥M −M0∥ where M is the finetuned model weights and M0 is the pretrained model weights. This implicit constraint makes intuitive sense, but it can still be too flexible for finetuning a large model. To address this, LoRA introduces an additional low-rank constraint for the weight update, i.e., rank(M −M0)=r′ where r′ is set to be some small number. Different from LoRA, OFT introduces a constraint for the pair-wise neuron similarity: ∥HE(M)−HE(M0)∥=0 where HE(·) denotes hyperspherical energy of a weight matrix. As an illustrative example, we consider a fully connected layer W ={w1,··· ,wn}∈Rd×n where wi ∈ Rd is the i-th neuron (W0 is the pretrained weights). The output vector z∈Rn of this fully connected layer is computed by z=W⊤x where x∈Rd is the input vector. OFT can be interpreted as minimizing the hyperspherical energy difference between the finetuned model and the pretrained model:

∥wˆi − wˆj∥−1 −

∥wˆi0 − wˆj0∥−1 (1)

HE(W) − HE(W0) ⇔ min

min

W

###### W

i̸=j

i̸=j

where wˆi:=wi/∥wi∥ denotes the i-th normalized neuron, and the hyperspherical energy of a fully connected layer W is defined as HE(W):= i̸=j ∥wˆi−wˆj∥−1. One can easily observe that the attainable minimum is zero for Eq. (1). The minimum can be achieved as long as W and W0 differ only up to a rotation or reflection, i.e., W =RW0 in which R∈Rd×d is an orthogonal matrix (The determinant 1 or −1 means rotation or reflection, respectively). This is exactly the idea of OFT, that we only need to finetune the neural network by learning layer-shared orthogonal matrices to transform neurons in each layer. Formally, OFT seeks to optimize the orthogonal matrix R∈Rd×d for a pretrained fully connected layer W0∈Rd×n, changing the forward pass from z=(W0)⊤x to

z = W⊤x = (R · W0)⊤x, s.t. R⊤R = RR⊤ = I (2)

where W denotes the OFT-finetuned weight matrix and I is an identity matrix. OFT is illustrated in Figure 4. Similar to the zero initialization in LoRA, we need to ensure OFT to finetune the pretrained model exactly from W0. To achieve this, we initialize the orthogonal matrix R to be an identity matrix so that the finetuned model starts with the pretrained weights. To guarantee the orthogonality of the matrix R, we can use differential orthogonalization strategies discussed in [26, 33]. We will discuss how to guarantee the orthogonality in a computationally efficient way.

##### 3.3 Efficient Orthogonal Parameterization

Standard orthogonalization such as GramSchmidt method, despite differentiable, is often too expensive to compute in practice [33]. For better efficiency, we adopt Cayley parameterization to generate the orthogonal matrix. Specifically, we construct the orthogonal matrix with R=

n d/r

|R<br><br>Orthogonal Matrix<br><br>d<br><br>|
|---|

|Pretrained Weight Matrix<br><br>W0<br><br>d<br><br>n<br><br>|
|---|

- R1
- R2
- R3

Pretrained Weight Matrix

x

x

d

...

W0

Orthogonal Matrix

Rr (a) Original OFT (b) OFT with block-diagonal R

Figure 4: (a) Original OFT without a diagonal structure. (b) OFT with r diagonal blocks of the same size. When r = 1, the case of (b) recovers the case of (a).

(I +Q)(I −Q)−1 where Q is a skew-symmetric matrix satisfying Q=−Q⊤. Such an efficiency comes at a small price – the Cayley parameterization can only produce orthogonal matrices with determinant 1 which belongs to the special orthogonal group. Fortunately, we find that such a limitation does not affect the performance in practice. Even if we use Cayley transform to parameterize the orthogonal matrix, R can still be very parameter-inefficient with a large d. To address this, we propose to represent R with a block-diagonal matrix with r blocks, leading to the following form:

  

   ∈ O(d) (3)

R1 ∈ O(dr)

...

R = diag(R1, R2, · · · , Rr) =

Rr ∈ O(dr)

where O(d) denotes the orthogonal group in dimension d, and R∈Rd×d and Ri∈Rd/r×d/r,∀i are orthogonal matrices. When r=1, then the block-diagonal orthogonal matrix becomes a standard unconstrained one. For an orthogonal matrix with size d×d, the number of parameters is d(d−1)/2, resulting in a complexity of O(d2). For an r-block diagonal orthogonal matrix, the number of parameter is d(d/r−1)/2, resulting in a complexity of O(d2/r). We can optionally share the block matrix to further reduce the number of parameters, i.e., Ri=Rj,∀i̸=j. This reduces the parameter complexity to O(d2/r2). Despite all these strategies to improve parameter efficiency, we note that the resulting matrix R remains orthogonal, so there is no sacrifice in preserving hyperspherical energy.

We discuss how OFT compares to LoRA in terms of parameter efficiency. For LoRA with a low-rank parameter r′, we have its number of trainable parameters as r′(d+n). If we consider both r and r′ to be dependent on the neuron dimension d (e.g., r=r′=αd where 0<α≤1 is some constant), then the parameter complexity of LoRA becomes O(d2 + dn) and the parameter complexity of OFT becomes O(d). We illustrate the difference in complexity between OFT and LoRA with a concrete example. Suppose we have a weight matrix with size 128×128, LoRA has 2,048 trainable parameters with r′=8, while OFT has 960 trainable parameters with r=8 (no block sharing is applied).

##### 3.4 Constrained Orthogonal Finetuning

We can further limit the flexibility of original OFT by constraining the finetuned model to be within a small neighborhood of the pretrained model. Specifically, COFT uses the following forward pass:

z = W⊤x = (R · W0)⊤x, s.t. R⊤R = RR⊤ = I, ∥R − I∥ ≤ ϵ (4)

which has an orthogonality constraint and an ϵ-deviation constraint to an identity matrix. The orthogonality constraint can be achieved with the Cayley parameterization introduced in Section 3.3. However, it is nontrivial to incorporate the ϵ-deviation constraint to the Cayley-parameterized orthogonal matrix. To gain more insights on the Cayley transform, we apply the Neumann series to approximate R=(I +Q)(I −Q)−1 as R≈I +2Q+O(Q2) (under the assumption that the Neumann

series converges in the operator norm). Therefore, we can move the constraint ∥R−I∥≤ϵ inside the Cayley transform, and the equivalent constraint is ∥Q−0∥≤ ϵ′ where 0 denotes an all-zero matrix and ϵ′ is another error hyperparameter (different than ϵ). The new constraint on the matrix Q can be easily enforced by projected gradient descent. To achieve identity initialization for the orthogonal matrix R, we initialize Q as an all-zero matrix. COFT can be viewed as a combination of two explicit constraints: minimal hyperspherical energy difference and constrained deviation from the pretrained model. The second constraint is usually implicitly used by existing finetuning methods, but COFT makes it an explicit one. Despite the excellent performance of OFT, we observe that COFT yields even better finetuning stability than OFT due to this explicit deviation constraint. Figure 5 provides an example on how ϵ affects the performance of COFT. We can observe that ϵ controls the flexibility of finetuning. With larger ϵ, the COFT-finetuned model resembles the OFT-finetuned model. With smaller ϵ, the COFT-finetuned model behaves increasingly similar to the pretrained text-to-image diffusion model.

Text prompt: a [V] toy on the beach

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

ε= 0 ε=1e-7.d ε=1e-6.d ε=1e-5.d ε=1e-4.d OFT (ε=∞) Input image

Figure 5: How ϵ affects the flexibility of COFT in subject-driven generation.

##### 3.5 Re-scaled Orthogonal Finetuning

We propose a simple extension to the original OFT by additionally learning a magnitude scaling coefficient for each neuron. This is motivated by the fact that re-scaling neurons does not change the hyperspherical energy (the magnitude will be normalized out). Specifically, we use the forward pass: z=(RW0D)⊤x1 where D=diag(s1,··· ,sn)∈Rn×n is a learnable diagonal matrix with all the diagonal element s1,··· ,sn larger than zero. In contrast to OFT’s original forward pass in Eq. (2) where only R is learnable, we have both the diagonal matrix D and the orthogonal matrix R learnable. The re-scaled OFT further improves the flexibility of OFT with a neglectable number of additional parameters. We stick to the original OFT in the experiment to show the effectiveness of orthogonal transformation alone, but we find that the re-scaled OFT is generally better (see Appendix C).

#### 4 Intriguing Insights and Discussions

OFT is agnostic to different architectures. We can apply OFT to any type of neural network in principle. For Transformers, LoRA is typically applied to the attention weights [22]. To compare fairly to LoRA, we only apply OFT to finetune the attention weights in our experiments. Besides fully connected layers, OFT is also well suited for finetuning convolution layers, because the blockdiagonal structure of R has interesting interpretations in convolution layers (unlike LoRA). When we use the same number of blocks as the number of input channels, each block only transforms a unique neuron channel, similar to learning depth-wise convolution kernels [10]. When all the blocks

1Errata: In the NeurIPS camera ready version, the forward pass of re-scaled OFT is mistakenly written as z=(DRW0)⊤x. The original implementation is correct, so the results in Appendix C are unaffected.

in R are shared, OFT transforms the neurons with an orthogonal matrix shared across channels. We conduct a preliminary study on finetuning convolution layers with OFT in Appendix D

Connection to LoRA. By adding a low-rank matrix, LoRA prevents the information in the pretrained weight matrix from shifting dramatically. In contrast, OFT controls the transform that applies to the pretrained weight matrix to be orthogonal (full-rank), which prevents the transform to destroy the pretraining information. We can rewrite OFT’s forward pass as z=(RW0)⊤x=(W0+(R− I)W0)⊤x where (R−I)W0 is analogous to LoRA’s low-rank weight update. Since W0 is typically full-rank, OFT also performs low-rank weight update when R−I is low-rank. Similar to LoRA that has a rank parameter r′, OFT has a diagonal block parameter r to reduce the number of trainable parameters. More interestingly, LoRA and OFT represent two distinct ways to be parameter-efficient. LoRA exploits the low-rank structure to reduce the number of trainable parameters, while OFT takes a different route by exploiting the sparsity structure (i.e., block-diagonal orthogonality).

Why OFT converges faster. On one hand, we can see from Figure 2 that the most effective update to modify the semantics is to change neuron directions, which is exactly what OFT is designed for. On the other hand, OFT can be viewed as finetuning neurons on a smooth hypersphere manifold, which yields better optimization landscape. This is also empirically verified in [33].

Why not minimize hyperspherical energy. A key difference to [33] is that we do not aim to minimize hyperspherical energy. In classification problems, neurons without redundancy are desired. The minimum hyperspherical energy means all neurons are uniformly spaced around the hypersphere. This is not a meaningful objective for finetuning, as it may destroy the pretraining information.

Trade-off between flexibility and regularity in finetuning. We discover an underlying trade-off between flexibility and regularity. Standard finetuning is the most flexible method, but it yields poor stability and easily causes model collapse. Being surprisingly simple, OFT finds a good balance between flexibility and regularity by preserving the pairwise neuron angles. The block-diagonal parameterization can also be viewed as a stronger regularization of the orthogonal matrix.

No additional inference overhead. Unlike ControlNet, our OFT framework introduces no additional inference overhead to the finetuned model. In the inference stage, we can simply multiply the learned orthogonal matrix R into the pretrained weight matrix W0 and obtain an equivalent weight matrix W =RW0. Thus the inference speed is the same as the pretrained model.

#### 5 Experiments and Results

General settings. In the experiment, we use Stable Diffusion v1.5 [50] as the pretrained text-to-image model. For fairness, we randomly pick generated images from each method. For subject-driven generation, we generally follow DreamBooth [51]. For controllable generation, we generally follow ControlNet [68] and T2I-Adapter [38]. To ensure a fair comparison to LoRA, we only apply OFT or COFT to the same layer where LoRA is used. More results and details are given in Appendix A.

##### 5.1 Subject-driven Generation

###### Text prompt: a [V] cat wearing a yellow shirt

Inputimages

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Settings. We use DreamBooth [51] and LoRA [22] as the baselines. All the methods adopt the same loss function as in DreamBooth. For DreamBooth and LoRA, we generally follow the original paper and use the best hyperparameter setup. More results are provided in Appendix A,E,F,J.

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

Iter.400Iter.2000Iter.1000

Finetuning stability and convergence. We first evaluate the finetuning stability and the convergence speed for DreamBooth, LoRA, OFT and COFT. Results are given in Figure 1 and Figure 6. We can observe that both COFT and OFT are able to finetune the diffusion model quite stably. After 400 iterations, both DreamBooth and OFT variants achieve good control, while LoRA fails to preserve the subject identity. After 2000 iterations, DreamBooth starts to generate collapsed images, and LoRA fails to generate yellow shirt (and instead generates yellow fur). In contrast, both OFT and COFT are still able to achieve stable and consistent control over the generated image. These results validate the fast yet stable convergence of our OFT framework in subject-driven genera-

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

DreamBooth LoRA COFT

OFT

Figure 6: Generated images across different iterations.

Input images

Input images

a [V] bowl with a blue house in the background

A [V] clock on the beach

[Figure 62]

[Figure 63]

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

A [V] clock on top of green gras with sunflower

a [V] bowl with a tree and autumn leaves in the background

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

[Figure 88]

[Figure 89]

OFT DreamBooth LoRA COFT

OFT

DreamBooth LoRA COFT

Input images

Input images

a [V] dog in a police outfit

a [V] stuffed animal on top of a pink fabric

[Figure 90]

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

a [V] dog wearing a black top and a monocle

a [V] stuffed animal in the snow

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

[Figure 114]

[Figure 115]

DreamBooth LoRA COFT OFT

DreamBooth LoRA COFT OFT

Figure 7: Qualitative comparison of subject-driven generation among DreamBooth, LoRA, COFT and OFT. Results are generated with the same finetuned model from each method. All examples are randomly picked. The figure is best viewed digitally, in color and significantly zoomed in.

tion. We note that the insensitivity to the number of finetuning iteration is quite important, since it can effectively alleviate the trouble of tuning the iteration number for different subjects. For both OFT and COFT, we can directly set a relatively large iteration number without carefully tuning it. For COFT with a proper ϵ, both the learning rate and the iteration number become effortless to set.

Quantitative comparison. Following [51], we conduct a quantitative comparison to evaluate subject fidelity (DINO [5], CLIP-I [44]), text prompt fidelity (CLIPT [44]) and sample diversity (LPIPS [69]). CLIP-I computes the average pairwise cosine similarity of CLIP embeddings between generated and real images. DINO is similar to CLIP-I, except that we use ViT S/16 DINO embeddings. CLIP-T is the average cosine similarity of CLIP embeddings between text prompt and generated images. We also evaluate average LPIPS cosine similarity between generated images of the same subject with the same text prompt. Table 1 show that both COFT and OFT outperforms DreamBooth and LoRA in the DINO and CLIP-I metrics by a considerable margin, while achieving slightly better or comparable performance in prompt fidelity and diversity metric. For each method, we repeatedly finetune the same pretrained model with 30 different random seeds to minimize randomness. The results show that our OFT framework not only achieves better convergence and stability, but also yields consistently better final performance.

Method DINO ↑ CLIP-I ↑ CLIP-T ↑ LPIPS ↑

Real Images 0.703 0.864 - 0.695 DreamBooth 0.614 0.778 0.239 0.737 LoRA 0.613 0.765 0.237 0.744 COFT 0.630 0.783 0.235 0.744 OFT 0.632 0.785 0.237 0.746

Table 1: Quantitative comparison of subject fidelity (DINO, CLIP-I), prompt fidelity (CLIP-T) and diversity metric (LPIPS). The evaluation images and prompts are the same as [51] (25 subjects with 30 text prompts each subject).

Qualitative comparison. To have a more intuitive understanding of OFT’s benefits, we show some randomly picked examples for subject-driven generation in Figure 7. For a fair comparison, all the examples are generated from the same finetuned model using each method, so no text prompt will be separately optimized for its final results. For each method, we select the model that achieves the best validation CLIP metrics. From the results in Figure 7, we can observe that both OFT and COFT deliver excellent semantic subject preservation, while LoRA often fails to preserve the subject identity (e.g., LoRA completely loses the subject identity in the bowl example). In the meantime, both OFT and COFT have much more accurate control using text prompts, while DreamBooth, despite its preservation of subject identity, often fails to generate the image following the text prompt (e.g., the first row of the bowl example). The qualitative comparison demonstrates that our OFT framework achieves better controllability and subject preservation at the same time. Moreover, the number of iterations is not sensitive in OFT, so OFT performs well even with a large number of iterations, while neither DreamBooth nor LoRA can. More qualitative examples are given in Appendix F. Moreover, we conduct a human evaluation in Appendix H which further validates the superiority of OFT.

##### 5.2 Controllable Generation

Settings. We use ControlNet [68], T2I-Adapter [38] and LoRA [22] as the baselines. We consider three challenging controllable generation tasks in the main paper: Canny edge to image (C2I) on the

COCO dataset [31], segmentation map to image (S2I) on the ADE20K dataset [70] and landmark to face (L2F) on the CelebA-HQ dataset [25, 63]. All the methods are used to finetune Stable Diffusion (SD) v1.5 on these three datasets for 20 epochs. More results are given in Appendix F,G,J.

Convergence. We evaluate the convergence speed of ControlNet, T2IAdapter, LoRA and COFT on the L2F task. We provide both quantitative and qualitative evaluation. Specifically for the evaluation metric, we compute the mean ℓ2 distance between control face landmarks and predicted face landmarks. In Figure 8, we plot the face landmark error obtained by the model finetuned with different number of epochs. We can observe that both COFT and OFT achieve significantly faster convergence. It takes 20 epochs for LoRA to converge to the performance of our OFT framework at the 8-th epoch. We note that OFT and COFT use a similar number of trainable parameters to LoRA (much fewer than ControlNet), while being much more efficient to converge than existing methods. On the other hand, the fast convergence of OFT is also validated by the results in Figure 1. The right example in Figure 1 shows that OFT is much more data-efficient than ControlNet and LoRA, since OFT can converge well with only 5% of the ADE20K dataset. For qualitative results, we focus on comparing OFT, COFT and ControlNet, because ControlNet achieves the closest landmark error to ours. Results in Figure 9 show that both OFT and COFT converge stably and the generated face pose is gradually aligned with the control landmarks. In contrast to our stable and smooth convergence, the controllability in ControlNet suddenly emerges after the 8-th epoch, which perfectly matches the sudden convergence phenomenon observed in [68]. Such a convergence stability makes our OFT framework much easier to use in practice, since the training dynamics of OFT is far more smooth and predictable. Thus it will be easier to find good OFT’s hyperparameters.

35

30

Landmarkerror

25

T2I-Adapter

ControlNet

20

LoRA COFT OFT

15

10

5 10 15 20

Number of epochs

Figure 8: Face landmark error.

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

a man with long hair wearing a red shirt

ControlNetCOFT

Quantitative comparison. We introduce a control consistency metric to evaluate the performance of controllable generation. The basic idea is to compute the control signal from the generated image and then compare it with the original input control signal. For the C2I task, we compute IoU and F1 score. For the S2I task, we compute mean IoU, mean and overall accuracy. For the L2F task, we compute the mean ℓ2 distance between control landmarks and predicted landmarks. More details regarding the consistency metrics are given in Appendix A. For all the compared method, we use the best possible hyperparameter settings. Results in Table 2 show that both OFT and COFT yield much stronger and accurate control than the other methods. We observe that the adapter-based approaches (e.g., T2I-Adapter and ControlNet) converge slowly and also yield worse final results. Compared to ControlNet, LoRA performs better in the S2I task and worse in the C2I and L2F tasks. In general, we find that the performance ceiling of LoRA is relatively low, even if we have carefully tuned its hyperparameters. As a comparison, the performance of our OFT framework has not yet saturated, since we empirically find that it still gets better as the number of trainable parameters gets large. We emphasize that our quantitative evaluation in controllable generation is one of our novel contributions, since it can accurately evaluate the control performance of the finetuned models (up to the accuracy of the off-the-shelf segmentation/detection model).

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

Control

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

OFT

Ground truth

Epoch 4 Epoch 6 Epoch 8 Epoch 10

Figure 9: Qualitative examples with different number of epochs.

Qualitative comparison. We also qualitatively compare OFT and COFT to current state-of-the-art methods, including ControlNet, T2I-Adapter and LoRA. Randomly generated images in Figure 10 show that OFT and COFT not only yield high-fidelity and realistic image quality, but also achieve accurate control. In the S2I task, we can see that LoRA completely fails to generate images following the input segmentation map, while ControlNet, OFT and COFT can well control the generated images. In contrast to ControlNet, both OFT and COFT are able to generate high-fidelity images with more vivid details and more reasonable

Task Metric SD ControlNet T2I-Adapter LoRA COFT OFT C2I

IoU ↑ 0.049 0.189 0.078 0.168 0.195 0.193 F1 ↑ 0.093 0.317 0.143 0.286 0.325 0.323

mIoU ↑ 7.72 20.88 16.38 22.98 26.92 27.06 mAcc ↑ 14.40 30.91 26.31 35.52 40.08 40.09 aAcc ↑ 33.61 61.42 51.63 58.03 62.96 62.42

S2I

L2F Error ↓ 146.19 7.61 23.75 7.68 6.92 7.07

Table 2: Quantitative comparison of control signal consistency for three control tasks (Canny edge to image, segmentation to image and landmark to face).

###### a tropical island

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

||[Figure 137]|
|---|
<br><br>|[Figure 138]|
|---|
<br><br>|[Figure 139]|
|---|
<br><br>|[Figure 140]|
|---|
<br><br>a building<br><br>[Figure 141]<br><br>[Figure 142]<br><br>a building||[Figure 143]|
|---|
<br><br>|[Figure 144]|
|---|
<br><br>|[Figure 145]|
|---|
<br><br>|[Figure 146]|
|---|
<br><br>|[Figure 147]|
|---|
<br><br>|[Figure 148]|
|---|
||[Figure 149]|
|---|
<br><br>|[Figure 150]|
|---|
<br><br>|[Figure 151]|
|---|
<br><br>|[Figure 152]|
|---|
||[Figure 153]|
|---|
<br><br>|[Figure 154]|
|---|
<br><br>|[Figure 155]|
|---|
<br><br>|[Figure 156]|
|---|
|
|---|---|---|---|
|a cat<br><br>a truck<br><br>|[Figure 157]|
|---|
<br><br>|[Figure 158]|
|---|
<br><br>|[Figure 159]|
|---|
<br><br>|[Figure 160]|
|---|
||[Figure 161]|
|---|
<br><br>|[Figure 162]|
|---|
<br><br>|[Figure 163]|
|---|
<br><br>|[Figure 164]|
|---|
<br><br>|[Figure 165]|
|---|
<br><br>|[Figure 166]|
|---|
||[Figure 167]|
|---|
<br><br>|[Figure 168]|
|---|
<br><br>|[Figure 169]|
|---|
<br><br>|[Figure 170]|
|---|
||[Figure 171]|
|---|
<br><br>|[Figure 172]|
|---|
<br><br>|[Figure 173]|
|---|
<br><br>|[Figure 174]|
|---|
|
|a women with curly hair<br><br>a man with long hair wearing a red shirt<br><br>|[Figure 175]|
|---|
<br><br>|[Figure 176]|
|---|
<br><br>|[Figure 177]|
|---|
<br><br>|[Figure 178]|
|---|
||[Figure 179]|
|---|
<br><br>|[Figure 180]|
|---|
<br><br>|[Figure 181]|
|---|
<br><br>|[Figure 182]|
|---|
<br><br>|[Figure 183]|
|---|
<br><br>|[Figure 184]|
|---|
||[Figure 185]|
|---|
<br><br>|[Figure 186]|
|---|
<br><br>|[Figure 187]|
|---|
<br><br>|[Figure 188]|
|---|
||[Figure 189]|
|---|
<br><br>|[Figure 190]|
|---|
<br><br>|[Figure 191]|
|---|
<br><br>|[Figure 192]|
|---|
|

LandmarktofaceSegmentationtoImgeCannyedgetoImage

Ground truth Control ControlNet T2I-Adapter LoRA COFT OFT

Figure 10: Qualitative comparison of controllable generation. The figure is best viewed digitally, in color and significantly zoomed in.

geometric structures with far less model parameters. In the C2I task, both OFT and COFT are able to hallucinate semantically similar images based on a rough Canny edges, while T2I-Adapter and LoRA perform much worse. In the L2F task, our method produces the most accurate pose control for the generated faces even under challenging face poses. In all three control tasks, we show that both OFT and COFT produce qualitatively better images than the state-of-the-art baselines, demonstrating the effectiveness of our OFT framework in controllable generation. To give a more comprehensive qualitative comparison, we provide more qualitative examples for all the three control tasks in Appendix F.2, and moreover, we demonstrate OFT can perform well on more control tasks (including dense pose to human body, sketch to image and depth to image) in Appendix G.

#### 6 Concluding Remarks and Open Problems

Motivated by the observation that angular information among neurons crucially determines visual semantics, we propose a simple yet effective finetuning method – orthogonal finetuning for controlling text-to-image diffusion models. Specifically, we target two text-to-image applications: subject-driven generation and controllable generation. Compared to existing methods, OFT demonstrates stronger controllability and finetuning stability with fewer number of finetuning parameters. More importantly, OFT does not introduce additional inference overhead, leading to an efficient deployable model.

OFT also introduces a few interesting open problems. First, OFT guarantees the orthogonality via Cayley parametrization which involves a matrix inverse. It slightly limits the scalability of OFT. Although we address this limitation using block diagonal parametrization, how to speed up this matrix inverse in a differentiable way remains a challenge. Second, OFT has unique potential in compositionality, in the sense that the orthogonal matrices produced by multiple OFT finetuning tasks can be multiplied together and remains an orthogonal matrix. Whether this set of orthogonal matrices preserve the knowledge of all the downstream tasks remains an interesting direction to study. Finally, the parameter efficiency of OFT is largely dependent on the block diagonal structure which inevitably introduces additional biases and limits the flexibility. How to improve the parameter efficiency in a more effective and less biased way remains an important open problem.

#### Acknowledgement

The authors would like to sincerely thank Luigi Gresele, Yandong Wen, Yuliang Xiu and many other colleagues at Max Planck Institute for Intelligent Systems for many helpful suggestions.

This work was supported by the German Federal Ministry of Education and Research (BMBF): Tubingen AI Center, FKZ: 01IS18039B, and by the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645. WL was supported by the German Research Foundation (DFG): SFB 1233, Robust Vision: Inference Principles and Neural Mechanisms, TP XX, project number: 276693517. AW acknowledges support from a Turing AI Fellowship under grant EP/V025279/1, The Alan Turing Institute, and the Leverhulme Trust via CFI.

#### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022. 3
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. arXiv preprint arXiv:2206.02779, 2022. 3
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 4
- [4] Adrian Bulat and Georgios Tzimiropoulos. How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks). In ICCV, 2017. 17
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 8
- [6] Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano. Instance-conditioned gan. In NeurIPS, 2021. 3
- [7] Beidi Chen, Weiyang Liu, Zhiding Yu, Jan Kautz, Anshumali Shrivastava, Animesh Garg, and Animashree Anandkumar. Angular visual hardness. In ICML, 2020. 2, 4
- [8] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. In ICCV, 2021. 3
- [9] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In CVPR, 2018. 3
- [10] François Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017. 6
- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 4
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS,

2021. 3

- [13] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS,

2021. 3

- [14] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021. 3
- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [16] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022. 3
- [17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 4

- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 17
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [21] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In ICML,

2019. 4

- [22] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 1, 2, 4, 6, 7, 8, 20
- [23] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017. 3
- [24] Yuming Jiang, Shuai Yang, Haonan Qiu, Wayne Wu, Chen Change Loy, and Ziwei Liu. Text2human: Text-driven controllable human image generation. ACM Transactions on Graphics (TOG), 41(4):1–11,

2022. 16

- [25] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In ICLR, 2018. 9, 16
- [26] Mario Lezcano-Casado and David Martınez-Rubio. Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. In ICML, 2019. 5
- [27] Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip Torr. Controllable text-to-image generation. In NeurIPS, 2019. 3
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3
- [29] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 3, 16
- [30] Rongmei Lin, Weiyang Liu, Zhen Liu, Chen Feng, Zhiding Yu, James M Rehg, Li Xiong, and Le Song. Regularizing neural networks via minimizing hyperspherical energy. In CVPR, 2020. 3
- [31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 9, 16
- [32] Weiyang Liu, Rongmei Lin, Zhen Liu, Lixin Liu, Zhiding Yu, Bo Dai, and Le Song. Learning towards minimum hyperspherical energy. In NeurIPS, 2018. 2, 3
- [33] Weiyang Liu, Rongmei Lin, Zhen Liu, James M Rehg, Liam Paull, Li Xiong, Le Song, and Adrian Weller. Orthogonal over-parameterized training. In CVPR, 2021. 2, 3, 4, 5, 7
- [34] Weiyang Liu, Rongmei Lin, Zhen Liu, Li Xiong, Bernhard Schölkopf, and Adrian Weller. Learning with hyperspherical uniformity. In AISTATS, 2021. 2
- [35] Weiyang Liu, Zhen Liu, Zhiding Yu, Bo Dai, Rongmei Lin, Yisen Wang, James M Rehg, and Le Song. Decoupled networks. In CVPR, 2018. 2, 4
- [36] Weiyang Liu, Yan-Ming Zhang, Xingguo Li, Zhiding Yu, Bo Dai, Tuo Zhao, and Le Song. Deep hyperspherical learning. In NIPS, 2017. 2, 4
- [37] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS, 2019. 3
- [38] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 1, 3, 4, 7, 8
- [39] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022. 3

- [40] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, 2008. 16
- [41] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. ACM Transactions on Graphics (TOG), 41(6):1–10, 2022. 3
- [42] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatiallyadaptive normalization. In CVPR, 2019. 3
- [43] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. Adapterfusion: Non-destructive task composition for transfer learning. arXiv preprint arXiv:2005.00247, 2020. 4
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 8
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1, 3
- [46] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 3
- [47] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 16
- [48] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016. 3
- [49] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics, 42(1):1–13, 2022. 3
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 3, 7
- [51] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 1, 2, 3, 7, 8, 16
- [52] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In SIGGRAPH 2022, pages 1–10, 2022. 3
- [53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. In NeurIPS, 2022. 1, 3
- [54] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 3
- [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [56] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 3
- [57] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. In ICLR, 2020. 3
- [58] Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. arXiv preprint arXiv:2211.13752, 2022. 3
- [59] Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, and Fang Wen. Pretraining is all you need for image-to-image translation. arXiv preprint arXiv:2205.12952, 2022. 3
- [60] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Highresolution image synthesis and semantic manipulation with conditional gans. In CVPR, 2018. 3
- [61] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021. 3

- [62] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. Nüwa: Visual synthesis pre-training for neural visual world creation. In ECCV, 2022. 3
- [63] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Baoyuan Wu. Tedigan: Text-guided diverse face image generation and manipulation. In CVPR, 2021. 9
- [64] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. In NeurIPS, 2021. 17
- [65] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In CVPR,

2018. 3

- [66] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. 3
- [67] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In ICCV, 2017. 3
- [68] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 1, 2, 3, 4, 7, 8, 9, 16
- [69] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8
- [70] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017. 9, 16
- [71] Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A Efros, Oliver Wang, and Eli Shechtman. Toward multimodal image-to-image translation. In NeurIPS, 2017. 3

### Appendix

#### Table of Contents

- A Experimental Details 16
- B Effect of Different Number of Diagonal Blocks 18
- C Experiments on Re-scaled OFT 19
- D Applying OFT to Convolution Layers 20
- E Comparison between COFT and OFT 21
- F More Qualitative Results 23

- F.1 Subject-driven Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- F.2 Controllable Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- G More Controllable Generation Tasks 34

- G.1 Dense Pose to Human Body . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- G.2 Sketch to Image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- G.3 Depth to Image . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38

- H Human Evaluation 40
- I Style Transfer by Adapting Stable Diffusion with Orthogonal Finetuning 41
- J Failure Cases 42

- J.1 Failure Cases in Subject-driven Generation . . . . . . . . . . . . . . . . . . . 42
- J.2 Failure Cases in Controllable Generation . . . . . . . . . . . . . . . . . . . . . 43

#### A Experimental Details

To verify the effectiveness of our Orthogonal Fine-tuning (OFT) method, we extensively evaluate the performance of our method in two common text-to-image generation tasks: subject-driven generation and controllable generation. More specifically, we use the exact same task setting as ControlNet [68] and Dreambooth [51] and the baseline implementations were sourced from the GitHub repository Diffusers2 and ControlNet3.

Data and Model. For training the convolutional autoencoder from Figure 2, we use 1000 random images from the Oxford 102 Flower dataset [40]. For the task of subject-driven generation, we use the official DreamBooth dataset, which consists of 30 subjects from 15 different classes. For each subject, there are several images and 25 different text prompts. For generating the image-control-caption combinations, we use BLIP [29] to automatically caption the images (pre-trained model weight and code for captioning based on the GitHub repository BLIP4). Note, although COCO provides captions for the training and validation split, to be consistent with other image-control-caption combinations, we instead use the BLIP-generated captions as text prompts. For the C2I task, we use the whole COCO 2017 dataset [31] with in total of 180K images; we generate canny edge images as the control signal using the same canny edge detector as ControlNet. For the S2I task, we use the semantic segmentation dataset ADE20K [70] with in total of 24K image-segmentation mask pairs. For the L2F dataset, we use the CelebA-HQ dataset [25], which contains 30K images. Additionally, we demonstrate that OFT also works well in other controllable generation tasks, including depth-toimage (D2I), densepose-to-image (P2I), and sketch-to-image (Sk2I). For the D2I task, we also use the COCO dataset and employ MiDaS [47] to generate depth maps; the pre-trained weights are obtained from the GitHub repository MiDaS5. For the P2I task, we use the DeepFashion-MultiModal dataset [24] with in total of 44K clothed human images with the corresponding densepose. For the Sk2I task, we use a subset of the LAION-Aesthetics dataset with approximately 350K images to learn sketch-guided image generation. We use the Stable Diffusion v1.56 as the base model.

Subject-driven generation. For training our subject-driven generation diffusion model, we follow the training objective of Dreambooth. More specifically, we use the class-specific prior preservation loss to fine-tune our orthogonal matrices:

Ex,c,ϵ,ϵ′,t[wt∥xˆθ(αtx + σtϵ,c) − x∥22 + λwt′∥xˆθ(αt′xpr + σt′ϵ′,cpr) − xpr∥22], (5)

with cpr being the class conditioning vector. For calculating the prior-preservation loss, we additionally need to generate 200 images using the subject’s class prompt. Similar to LoRA, we inject our trainable orthogonal matrices into the attention modules of the stable diffusion model. To be comparable with LoRA, we choose the exact same linear layers as LoRA to affect upon: the linear layers Wq, Wk, Wv and Wo. We perform training on 1 Tesla V100-SXM2-32GB GPU using a learning rate of

- 6 × 10−5, batch size of 1, and train for approximately 1000 iterations. In the case of COFT, we use ϵ = 6 × 10−5 to constrain the orthogonal matrices.

Controllable generation. Apart from injecting the trainable OFT weights into the stable diffusion model, we need to add a small encoding model to stable diffusion to encode the control signal. To be comparable with ControlNet [68], we use the same encoding module, which is a shallow 8-layer convolutional network with Scaled Exponential Linear Unit (SELU) activation functions. We also the same training objective as ControlNet. The control signal is encoded and concatenated once with the input to the stable diffusion U-Net. For the LoRA baseline, we use the same encoding module to encode the control signal. For S2I, L2I and P2I, we fine-tune the model for 20 epochs; for C2I and D2I we fine-tune the model for 10 epochs; for Sk2I we fine-tune the model for 8 epochs. We perform training on 4 NVIDIA A100-SXM4-80GB GPUs using a learning rate of 1 × 10−5, batch size of 4 for L2I and batch size of 16 for the rest of tasks. For fine-tuning with COFT, we use ϵ = 1 × 10−3.

- 2https://github.com/huggingface/diffusers
- 3https://github.com/lllyasviel/ControlNet
- 4https://github.com/salesforce/BLIP
- 5https://github.com/isl-org/MiDaS
- 6https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt

Evaluation. When evaluating the effectiveness of controllable generation, we primarily focus on evaluating the controllability. Using the consistency metrics introduced in the main paper, we can effectively compute the difference between the control signal and the generated image. For the C2I task, we apply the identical canny filter on the generated image to determine a canny image of the predicted image. Both the control signal canny image and the canny image obtained from the generated images are black-and-white images, with pixel values being either 0 or 1. We evaluate the pixel-wise Intersection over Union (IoU) and F1 score between these two canny predictions. For the S2I task, we compute mean IoU, mean and overall accuracy by deploying a pre-trained semantic segmentation model. More specifically, we use the Segformer7 [64] model, which is trained on ADE20K (Segformer-B4), to perform semantic segmentation on our generated images. We use the segmentation accuracy as an indication for the overall semantically and structural resemblance of the generated images to the ground truth image. For the L2F task, we compute the mean ℓ2 distance between the input control landmarks and the landmarks estimated from generated images using facial landmark detector [4].

We also evaluate the generation performance by calculating Fréchet Inception Distance (FID) [19], we use the default setting of the GitHub repository pytorch-fid8. The FID is a metric quantifying the similarity between two image dataset. It utilizes 2048-dimensional features, which are derived from the final average pooling layer of a pretrained InceptionV3 network trained on ImageNet dataset. A lower FID score indicates a higher similarity between the datasets.

- 7https://github.com/NVlabs/SegFormer
- 8https://github.com/mseitzer/pytorch-fid

#### B Effect of Different Number of Diagonal Blocks

We note that the number of diagonal blocks r is an important hyperparameter that effectively controls the number of trainable parameters. It is necessary to perform a sensitivity study on this hyperparameter. Following the same settings as the main paper, we evaluate how r affects OFT in the S2I task. Results in Table 3 show that smaller r (closer to recovering the standard orthogonal matrix) generally works better than larger r. However, we find that a good trade-off between flexibility and parameter-efficiency indeed exists. Empirically, we find that we can use a much bigger r if the dataset is simple, leading to better parameter-efficiency and faster convergence. In the main paper, we always use r = 4 because we find that r = 4 works well across datasets and tasks. Note that, in terms of the number of inference parameters, both LoRA and OFT have the exact same number of parameters, which is equal to the number of parameters of the underlying stable diffusion model, while ControlNet has an additional control model with 361M parameters.

ControlNet r = 2 r = 4 r = 8 r = 16

Trainable Parameters 361.3 M 29.5 M 16.3 M 9.7 M 6.4 M Inference Parameters 1.42 B 1.06 B 1.06 B 1.06 B 1.06 B

mIoU ↑ 20.88 27.18 27.06 24.09 21.0 mAcc ↑ 30.91 39.39 40.09 36.95 32.55 aAcc ↑ 61.42 65.24 62.96 60.25 55.5

Table 3: How the number of diagonal blocks affects the control capability of OFT.

#### C Experiments on Re-scaled OFT

Since both OFT and COFT transform neurons with orthogonal matrices and do not affect the magnitude of neurons, their magnitude may be sub-optimal with their updated orientations. To address this issue, we propose a re-scaled OFT where the neuron magnitude is refined using the same set of data in the downstream task. Specifically, re-scaled OFT further finetunes the magnitude of neurons without changing their directions. re-scaled OFT can be performed in two manners: (1) joint fitting: magnitude fitting can be used simultaneously with OFT or COFT, and (2) Post-stage fitting: magnitude fitting can be used after OFT or COFT is finished. An important motivation for re-scaled OFT comes from Figure 2, where we observe that constructing images only with angular information perfectly preserves visual structures, but it also results in a certain degree of color distortion. We hypothesize that this minor color distortion is caused by magnitude loss and fixing this issue can potentially improve the visual quality of generated images.

Notably, re-scaled OFT does not change the hyperspherical energy since it does not change the direction of neurons - all the nice properties of OFT and COFT on hyperspherical energy are still perfectly preserved. Therefore, the advantage of structural preservation is also inherited.

To simplify the experiments and validate the effectiveness of re-scaled OFT, we perform post-stage magnitude fitting on the COFT model and compare the FID between the original validation images and the generated images (using the control signals extracted from validation images). The reason we use FID here is that FID is more sensitive to color distortion, while the consistency metric only measures the structural preservation. Table 4 shows that magnitude fitting can indeed improve the FID of COFT and is beneficial to COFT.

Magnitude fitting is lightweight and can be implemented easily by simply adding one trainable parameter for each layer we modify; the parameter has the shape of (N × 1), with N corresponds to the number of neurons in that specific layer. The performance gain illustrated in Table 4 is achieved by performing Post-stage fitting on a COFT-fine-tuned model for only one additional epoch. Moreover, we expect that the joint fitting re-scaled OFT can lead to better performance.

SD ControlNet T2I LoRA COFT Re-scaled COFT FID ↓ 41.2 30.9 33.1 30.9 30.8 30.2

- Table 4: FID on the segmentation to image task (ADE20K). r = 4 is used here.

#### D Applying OFT to Convolution Layers

In the original setting [22], LoRA is only applied to the linear layers of the attention modules. To be a fair comparison, we also apply OFT to these weights. However, OFT is not limited to linear layers but can easily be adapted to convolution layers by transforming the convolutional neurons. We highlight the compatibility of OFT and COFT for finetuning convolution layers. More interestingly, sharing the parameters of diagonal blocks in R becomes interpretable in convolution layers. With a suitable setup, orthogonal matrices with sharing diagonal blocks can transform the convolution kernel in a channel-sharing manner (or in a spatial manner), implying that the same orthogonal transformation is applied to all channels. This shares similar intuition with depth-wise convolution.

For this ablation experiment, we study the performance of applying OFT to the convolution layers in the ResNet blocks of the stable diffusion model. In this experiment, we use COFT as the baseline method and consider the controllable generation (segmentation to image) as an example. We have both quantitative (Table 5) and qualitative results (Figure 11). We can empirically observe that by only fine-tuning the convolutional layers, we can also achieve some degree of control. By simultaneously fine-tuning both linear and convolutional layers, we achieve a slightly better FID score. Note, for fine-tuning convolutional layers, we let r be equal to the number of channels of convolutional neurons in that layer.

COFT (attention) COFT (conv) COFT (extended) FID ↓ 30.8 39.8 30.4

- Table 5: FID results of applying COFT to different types of layers. (with r = 4)

Text prompt: the inside of an airplane

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

Text prompt: a room with paintings on the wall

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

Text prompt: a sunset

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

Text prompt: a city

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

Original image Control Convolution Attention Both

Figure 11: Controllable generation results of applying COFT to different types of layers.

#### E Comparison between COFT and OFT

We have already provided many qualitative examples for COFT and OFT in the main paper. One may question the fundamental difference between OFT and COFT. Based on the intuition behind COFT, the deviation constraint is introduced to improve the training stability. We demonstrate the training stability of COFT with a qualitative example in subject-driven generation. Results in Figure 12 and Figure 13 show that, despite being much more stable than existing methods, OFT will eventually generate collapsed images at the 9000-th iteration. In contrast, COFT still produces visually appealing images. We train both OFT and COFT with a learning rate of 1 × 10−5 and constrain COFT with ϵ = 1 × 10−5.

Text prompt: a [V] dog on top of a wooden floor

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

COFTOFTCOFTOFT

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Text prompt: a [V] dog wearing a yellow shirt

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

COFTOFTCOFTOFT

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Input images

Text prompt: a [V] teapot on top of a dirt road

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

Text prompt: a [V] teapot on top of a dirt road

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Input images

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Iteration 2000 Iteration 9000

Figure 12: Qualitative comparison between COFT and OFT on subject-driven generation.

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Input images

Text prompt: a [V] vase on top of the sidewalk in a crowded street

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

COFTOFTCOFTOFT

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Text prompt: a [V] vase with a city in the back ground

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Text prompt: a [V] vase on top of a dirt road

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

COFTOFT

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Text prompt: a [V] vase with a tree and autumn leaves in the background COFTOFT

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Iteration 2000 Iteration 9000

Figure 13: Qualitative comparison between COFT and OFT on subject-driven generation.

#### F More Qualitative Results

##### F.1 Subject-driven Generation

Text prompt: a [V] backpack floating on top of water

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

Text prompt: a [V] backpack in the snow

[Figure 297]

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Text prompt: a [V] backpack on the beach

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

[Figure 310]

Text prompt: a purple [V] backpack

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

Text prompt: a [V] cat in a firefighter outfit

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

|[Figure 318]|
|---|

Text prompt: a [V] cat on top of a purple rug in a forest

[Figure 319]

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

|[Figure 323]|
|---|

[Figure 324]

[Figure 325]

Text prompt: a [V] cat wearing a red hat

[Figure 326]

|[Figure 327]|
|---|

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

[Figure 331]

###### Text prompt: a wet [V] cat

Input images

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

DreamBooth LoRA COFT OFT

Figure 14: More qualitative results on subject-driven generation.

Text prompt: a [V] dog in the snow

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

Text prompt: a [V] dog wearing a yellow shirt

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

[Figure 344]

[Figure 345]

[Figure 346]

Text prompt: a [V] dog with a blue house in the background

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

[Figure 351]

[Figure 352]

Text prompt: a [V] in a police outfit

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

Text prompt: a [V] vase on a cobblestone street

|[Figure 357]|
|---|

|[Figure 358]|
|---|

|[Figure 359]|
|---|

|[Figure 360]|
|---|

[Figure 361]

Text prompt: a [V] vase on the beach

[Figure 362]

[Figure 363]

|[Figure 364]|
|---|

|[Figure 365]|
|---|

|[Figure 366]|
|---|

|[Figure 367]|
|---|

[Figure 368]

[Figure 369]

Text prompt: a [V] vase with a wheat field in the background

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]|
|---|

|[Figure 373]|
|---|

[Figure 374]

Text prompt: a purple [V] vase

Input images

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

|[Figure 378]|
|---|

DreamBooth LoRA COFT OFT

Figure 15: More qualitative results on subject-driven generation.

- F.2 Controllable Generation

- F.2.1 Segmentation to Image

|[Figure 379]|
|---|

|[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

Text prompt: an auditorium

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

|[Figure 389]|
|---|

Text prompt: two beds in a room

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

###### Text prompt: an art gallery

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

|[Figure 397]|
|---|

Text prompt: a thomass train at disneyland world

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

Text prompt: a small room with a bed and shelves

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

Text prompt: a bathroom with a sink and toilet

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

###### Text prompt: a house

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

Original image Control ControlNet COFT OFT

Figure 16: More qualitative results of OFT and COFT on the segmentation to image generation task.

Text prompt: a living room

|[Figure 414]|
|---|

|[Figure 415]|
|---|

|[Figure 416]|
|---|

|[Figure 417]|
|---|

|[Figure 418]|
|---|

Text prompt: a living room

|[Figure 419]|
|---|

|[Figure 420]|
|---|

|[Figure 421]|
|---|

|[Figure 422]|
|---|

|[Figure 423]|
|---|

Text prompt: a road in the middle of nowhere

|[Figure 424]|
|---|

|[Figure 425]|
|---|

|[Figure 426]|
|---|

|[Figure 427]|
|---|

|[Figure 428]|
|---|

Text prompt: a hallway in a building

|[Figure 429]|
|---|

|[Figure 430]|
|---|

|[Figure 431]|
|---|

|[Figure 432]|
|---|

|[Figure 433]|
|---|

Text prompt: the sun shining through the clouds over a lake

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

Text prompt: a person in a kitchen

|[Figure 439]|
|---|

|[Figure 440]|
|---|

|[Figure 441]|
|---|

|[Figure 442]|
|---|

|[Figure 443]|
|---|

###### Text prompt: bears

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

|[Figure 447]|
|---|

|[Figure 448]|
|---|

Original image Control ControlNet COFT OFT

Figure 17: More qualitative results of OFT and COFT on the segmentation to image generation task.

###### Text prompt: a bedroom

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Text prompt: a dining room with a table and chairs

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

Text prompt: the living room

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

Text prompt: a living room with a vase of flowers

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Text prompt: a volleyball court

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Text prompt: a bedroom

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

Text prompt: a bedroom

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Original image Control ControlNet COFT OFT

Figure 18: More qualitative results of OFT and COFT on the segmentation to image generation task.

##### F.2.2 Canny Edge to Image

|[Figure 484]|
|---|

|[Figure 485]|
|---|

|[Figure 486]|
|---|

|[Figure 487]|
|---|

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

Text prompt: a bus with the words scott brown on it

|[Figure 491]|
|---|

|[Figure 492]|
|---|

|[Figure 493]|
|---|

Text prompt: a sign on a pole

|[Figure 494]|
|---|

|[Figure 495]|
|---|

|[Figure 496]|
|---|

Text prompt: elephants in the wild

|[Figure 497]|
|---|

|[Figure 498]|
|---|

|[Figure 499]|
|---|

Text prompt: a man holding a skateboard

|[Figure 500]|
|---|

|[Figure 501]|
|---|

|[Figure 502]|
|---|

Text prompt: a woman with red hair

|[Figure 503]|
|---|

|[Figure 504]|
|---|

|[Figure 505]|
|---|

Text prompt: big ben and the houses of parliament

|[Figure 506]|
|---|

|[Figure 507]|
|---|

|[Figure 508]|
|---|

###### Text prompt: a kitchen

|[Figure 509]|
|---|

|[Figure 510]|
|---|

|[Figure 511]|
|---|

|[Figure 512]|
|---|

|[Figure 513]|
|---|

|[Figure 514]|
|---|

|[Figure 515]|
|---|

|[Figure 516]|
|---|

|[Figure 517]|
|---|

|[Figure 518]|
|---|

Original image Control ControlNet COFT OFT

Figure 19: More qualitative results of OFT and COFT on the Canny edge to image generation task.

Text prompt: a train going through a tunnel

|[Figure 519]|
|---|

|[Figure 520]|
|---|

|[Figure 521]|
|---|

|[Figure 522]|
|---|

|[Figure 523]|
|---|

###### Text prompt: a sandwich

|[Figure 524]|
|---|

|[Figure 525]|
|---|

|[Figure 526]|
|---|

|[Figure 527]|
|---|

|[Figure 528]|
|---|

Text prompt: people on the beach at sunset

|[Figure 529]|
|---|

|[Figure 530]|
|---|

|[Figure 531]|
|---|

|[Figure 532]|
|---|

|[Figure 533]|
|---|

Text prompt: a slice of pizza on a table

|[Figure 534]|
|---|

|[Figure 535]|
|---|

|[Figure 536]|
|---|

|[Figure 537]|
|---|

|[Figure 538]|
|---|

Text prompt: a bathroom with a toilet and a shower

|[Figure 539]|
|---|

|[Figure 540]|
|---|

|[Figure 541]|
|---|

|[Figure 542]|
|---|

|[Figure 543]|
|---|

Text prompt: a bathroom with a toilet and a bucket

|[Figure 544]|
|---|

|[Figure 545]|
|---|

|[Figure 546]|
|---|

|[Figure 547]|
|---|

|[Figure 548]|
|---|

Text prompt: a living room

|[Figure 549]|
|---|

|[Figure 550]|
|---|

|[Figure 551]|
|---|

|[Figure 552]|
|---|

|[Figure 553]|
|---|

Original image Control ControlNet COFT OFT

Figure 20: More qualitative results of OFT and COFT on the Canny edge to image generation task.

###### Text prompt: a train

|[Figure 554]|
|---|

|[Figure 555]|
|---|

|[Figure 556]|
|---|

|[Figure 557]|
|---|

|[Figure 558]|
|---|

Text prompt: a bowl of food on a table

|[Figure 559]|
|---|

|[Figure 560]|
|---|

|[Figure 561]|
|---|

|[Figure 562]|
|---|

|[Figure 563]|
|---|

Text prompt: two white horses in a field

|[Figure 564]|
|---|

|[Figure 565]|
|---|

|[Figure 566]|
|---|

|[Figure 567]|
|---|

|[Figure 568]|
|---|

Text prompt: a vase on a table

|[Figure 569]|
|---|

|[Figure 570]|
|---|

|[Figure 571]|
|---|

|[Figure 572]|
|---|

|[Figure 573]|
|---|

Text prompt: a fire truck parked in a parking

|[Figure 574]|
|---|

|[Figure 575]|
|---|

|[Figure 576]|
|---|

|[Figure 577]|
|---|

|[Figure 578]|
|---|

###### Text prompt: a bathroom

|[Figure 579]|
|---|

|[Figure 580]|
|---|

|[Figure 581]|
|---|

|[Figure 582]|
|---|

|[Figure 583]|
|---|

Text prompt: a bus parked in a park

|[Figure 584]|
|---|

|[Figure 585]|
|---|

|[Figure 586]|
|---|

|[Figure 587]|
|---|

|[Figure 588]|
|---|

Original image Control ControlNet COFT OFT

Figure 21: More qualitative results of OFT and COFT on the Canny edge to image generation task.

##### F.2.3 Landmark to Face

Text prompt: a woman with red hair

|[Figure 589]|
|---|

|[Figure 590]|
|---|

|[Figure 591]|
|---|

|[Figure 592]|
|---|

|[Figure 593]|
|---|

Text prompt: a man wearing a hat

|[Figure 594]|
|---|

|[Figure 595]|
|---|

|[Figure 596]|
|---|

|[Figure 597]|
|---|

|[Figure 598]|
|---|

Text prompt: a man wearing glasses

|[Figure 599]|
|---|

|[Figure 600]|
|---|

|[Figure 601]|
|---|

|[Figure 602]|
|---|

|[Figure 603]|
|---|

Text prompt: a man in a tuxedo

|[Figure 604]|
|---|

|[Figure 605]|
|---|

|[Figure 606]|
|---|

|[Figure 607]|
|---|

|[Figure 608]|
|---|

Text prompt: a man with glasses smiling

|[Figure 609]|
|---|

|[Figure 610]|
|---|

|[Figure 611]|
|---|

|[Figure 612]|
|---|

|[Figure 613]|
|---|

Text prompt: a woman with blonde hair

|[Figure 614]|
|---|

|[Figure 615]|
|---|

|[Figure 616]|
|---|

|[Figure 617]|
|---|

|[Figure 618]|
|---|

Text prompt: a woman with long brown hair

|[Figure 619]|
|---|

|[Figure 620]|
|---|

|[Figure 621]|
|---|

|[Figure 622]|
|---|

|[Figure 623]|
|---|

Original image Control ControlNet COFT OFT

Figure 22: More qualitative results of OFT and COFT on the landmark to face generation task.

Text prompt: a man with a beard

|[Figure 624]|
|---|

|[Figure 625]|
|---|

|[Figure 626]|
|---|

|[Figure 627]|
|---|

|[Figure 628]|
|---|

Text prompt: a man wearing glasses

|[Figure 629]|
|---|

|[Figure 630]|
|---|

|[Figure 631]|
|---|

|[Figure 632]|
|---|

|[Figure 633]|
|---|

Text prompt: a man in a suit and tie

|[Figure 634]|
|---|

|[Figure 635]|
|---|

|[Figure 636]|
|---|

|[Figure 637]|
|---|

|[Figure 638]|
|---|

Text prompt: a man in a suit and tie

|[Figure 639]|
|---|

|[Figure 640]|
|---|

|[Figure 641]|
|---|

|[Figure 642]|
|---|

|[Figure 643]|
|---|

Text prompt: a man with long hair

|[Figure 644]|
|---|

|[Figure 645]|
|---|

|[Figure 646]|
|---|

|[Figure 647]|
|---|

|[Figure 648]|
|---|

Text prompt: a man in a suit and tie

|[Figure 649]|
|---|

|[Figure 650]|
|---|

|[Figure 651]|
|---|

|[Figure 652]|
|---|

|[Figure 653]|
|---|

Text prompt: a woman with long brown hair

|[Figure 654]|
|---|

|[Figure 655]|
|---|

|[Figure 656]|
|---|

|[Figure 657]|
|---|

|[Figure 658]|
|---|

Original image Control ControlNet COFT OFT

Figure 23: More qualitative results of OFT and COFT on the landmark to face generation task.

Text prompt: a woman with long brown hair

|[Figure 659]|
|---|

|[Figure 660]|
|---|

|[Figure 661]|
|---|

|[Figure 662]|
|---|

|[Figure 663]|
|---|

Text prompt: a woman with red lipstick

|[Figure 664]|
|---|

|[Figure 665]|
|---|

|[Figure 666]|
|---|

|[Figure 667]|
|---|

|[Figure 668]|
|---|

Text prompt: a woman with blonde hair

|[Figure 669]|
|---|

|[Figure 670]|
|---|

|[Figure 671]|
|---|

|[Figure 672]|
|---|

|[Figure 673]|
|---|

Text prompt: a young man with brown hair

|[Figure 674]|
|---|

|[Figure 675]|
|---|

|[Figure 676]|
|---|

|[Figure 677]|
|---|

|[Figure 678]|
|---|

Text prompt: a young man in a suit and tie

|[Figure 679]|
|---|

|[Figure 680]|
|---|

|[Figure 681]|
|---|

|[Figure 682]|
|---|

|[Figure 683]|
|---|

Text prompt: a beautiful asian woman

|[Figure 684]|
|---|

|[Figure 685]|
|---|

|[Figure 686]|
|---|

|[Figure 687]|
|---|

|[Figure 688]|
|---|

Text prompt: a woman eating food

|[Figure 689]|
|---|

|[Figure 690]|
|---|

|[Figure 691]|
|---|

|[Figure 692]|
|---|

|[Figure 693]|
|---|

Original image Control ControlNet COFT OFT

Figure 24: More qualitative results of OFT and COFT on the landmark to face generation task.

#### G More Controllable Generation Tasks G.1 Dense Pose to Human Body

Text prompt: a young man wearing a grey t-shirt and black jeans

|[Figure 694]|
|---|

|[Figure 695]|
|---|

|[Figure 696]|
|---|

|[Figure 697]|
|---|

|[Figure 698]|
|---|

|[Figure 699]|
|---|

|[Figure 700]|
|---|

|[Figure 701]|
|---|

|[Figure 702]|
|---|

Text prompt: a man wearing a grey t-shirt

|[Figure 703]|
|---|

|[Figure 704]|
|---|

|[Figure 705]|
|---|

|[Figure 706]|
|---|

|[Figure 707]|
|---|

|[Figure 708]|
|---|

|[Figure 709]|
|---|

|[Figure 710]|
|---|

|[Figure 711]|
|---|

Text prompt: a man in a white shirt and green pants

|[Figure 712]|
|---|

|[Figure 713]|
|---|

|[Figure 714]|
|---|

|[Figure 715]|
|---|

|[Figure 716]|
|---|

|[Figure 717]|
|---|

|[Figure 718]|
|---|

|[Figure 719]|
|---|

|[Figure 720]|
|---|

Text prompt: a man in a white top and black jeans

|[Figure 721]|
|---|

|[Figure 722]|
|---|

|[Figure 723]|
|---|

|[Figure 724]|
|---|

|[Figure 725]|
|---|

|[Figure 726]|
|---|

|[Figure 727]|
|---|

|[Figure 728]|
|---|

|[Figure 729]|
|---|

Text prompt: a man in a black jacket and jeans

|[Figure 730]|
|---|

|[Figure 731]|
|---|

|[Figure 732]|
|---|

|[Figure 733]|
|---|

|[Figure 734]|
|---|

|[Figure 735]|
|---|

|[Figure 736]|
|---|

|[Figure 737]|
|---|

|[Figure 738]|
|---|

Text prompt: a young man wearing a black t-shirt and white jeans

|[Figure 739]|
|---|

|[Figure 740]|
|---|

|[Figure 741]|
|---|

|[Figure 742]|
|---|

|[Figure 743]|
|---|

|[Figure 744]|
|---|

|[Figure 745]|
|---|

|[Figure 746]|
|---|

|[Figure 747]|
|---|

Text prompt: a young man in a white t-shirt and red pants

|[Figure 748]|
|---|

|[Figure 749]|
|---|

|[Figure 750]|
|---|

|[Figure 751]|
|---|

|[Figure 752]|
|---|

|[Figure 753]|
|---|

|[Figure 754]|
|---|

|[Figure 755]|
|---|

|[Figure 756]|
|---|

Text prompt: a man wearing a white t-shirt and jeans

|[Figure 757]|
|---|

|[Figure 758]|
|---|

|[Figure 759]|
|---|

|[Figure 760]|
|---|

|[Figure 761]|
|---|

|[Figure 762]|
|---|

|[Figure 763]|
|---|

|[Figure 764]|
|---|

|[Figure 765]|
|---|

Text prompt: a man wearing a white shirt and black jeans

|[Figure 766]|
|---|

|[Figure 767]|
|---|

|[Figure 768]|
|---|

|[Figure 769]|
|---|

|[Figure 770]|
|---|

|[Figure 771]|
|---|

|[Figure 772]|
|---|

|[Figure 773]|
|---|

|[Figure 774]|
|---|

Original image Control T2I-Adapter LoRA COFT OFT

ControlNet

Figure 25: Qualitative comparison among different methods on the dense pose to human body generation task.

|[Figure 775]|
|---|

Text prompt: a man in a black sweater and jeans

|[Figure 776]|
|---|

|[Figure 777]|
|---|

|[Figure 778]|
|---|

|[Figure 779]|
|---|

|[Figure 780]|
|---|

|[Figure 781]|
|---|

Text prompt: a man in a grey sweater and blue jeans

|[Figure 782]|
|---|

|[Figure 783]|
|---|

|[Figure 784]|
|---|

|[Figure 785]|
|---|

|[Figure 786]|
|---|

|[Figure 787]|
|---|

Text prompt: a man wearing a denim shirt and white pants

|[Figure 788]|
|---|

|[Figure 789]|
|---|

|[Figure 790]|
|---|

|[Figure 791]|
|---|

|[Figure 792]|
|---|

|[Figure 793]|
|---|

Text prompt: a man wearing a white t-shirt and jeans

|[Figure 794]|
|---|

|[Figure 795]|
|---|

|[Figure 796]|
|---|

|[Figure 797]|
|---|

|[Figure 798]|
|---|

|[Figure 799]|
|---|

Text prompt: a man wearing a denim jacket and jeans

|[Figure 800]|
|---|

|[Figure 801]|
|---|

|[Figure 802]|
|---|

|[Figure 803]|
|---|

|[Figure 804]|
|---|

Reconstruction Control signal COFT OFT

Figure 26: More qualitative results of COFT and OFT on the dense pose to human body task.

##### G.2 Sketch to Image

Text prompt: a group of people standing in front of the white house

|[Figure 805]|
|---|

|[Figure 806]|
|---|

|[Figure 807]|
|---|

|[Figure 808]|
|---|

|[Figure 809]|
|---|

|[Figure 810]|
|---|

|[Figure 811]|
|---|

Text prompt: a boathouse on the water at sunset

|[Figure 812]|
|---|

|[Figure 813]|
|---|

|[Figure 814]|
|---|

|[Figure 815]|
|---|

|[Figure 816]|
|---|

|[Figure 817]|
|---|

|[Figure 818]|
|---|

Text prompt: a boat on the beach

|[Figure 819]|
|---|

|[Figure 820]|
|---|

|[Figure 821]|
|---|

|[Figure 822]|
|---|

|[Figure 823]|
|---|

|[Figure 824]|
|---|

|[Figure 825]|
|---|

Text prompt: a mountain with snow on it

|[Figure 826]|
|---|

|[Figure 827]|
|---|

|[Figure 828]|
|---|

|[Figure 829]|
|---|

|[Figure 830]|
|---|

|[Figure 831]|
|---|

|[Figure 832]|
|---|

Text prompt: an old car driving down a snowy road

|[Figure 833]|
|---|

|[Figure 834]|
|---|

|[Figure 835]|
|---|

|[Figure 836]|
|---|

|[Figure 837]|
|---|

|[Figure 838]|
|---|

|[Figure 839]|
|---|

Text prompt: a painting of a castle

|[Figure 840]|
|---|

|[Figure 841]|
|---|

|[Figure 842]|
|---|

|[Figure 843]|
|---|

|[Figure 844]|
|---|

|[Figure 845]|
|---|

|[Figure 846]|
|---|

Text prompt: a family with a dog

|[Figure 847]|
|---|

|[Figure 848]|
|---|

|[Figure 849]|
|---|

|[Figure 850]|
|---|

|[Figure 851]|
|---|

|[Figure 852]|
|---|

|[Figure 853]|
|---|

Text prompt: the golden gate bridge

|[Figure 854]|
|---|

|[Figure 855]|
|---|

|[Figure 856]|
|---|

|[Figure 857]|
|---|

|[Figure 858]|
|---|

|[Figure 859]|
|---|

|[Figure 860]|
|---|

Text prompt: a deer in the snow

|[Figure 861]|
|---|

|[Figure 862]|
|---|

|[Figure 863]|
|---|

|[Figure 864]|
|---|

|[Figure 865]|
|---|

|[Figure 866]|
|---|

|[Figure 867]|
|---|

Original image Control ControlNet T2I-Adapter LoRA COFT OFT

Figure 27: Qualitative comparison among different methods on the sketch to image generation task.

|[Figure 868]|
|---|

|[Figure 869]|
|---|

|[Figure 870]|
|---|

|[Figure 871]|
|---|

|[Figure 872]|
|---|

Text prompt: a river scene with a ship in the distance

|[Figure 873]|
|---|

|[Figure 874]|
|---|

|[Figure 875]|
|---|

|[Figure 876]|
|---|

|[Figure 877]|
|---|

Text prompt: a river in the mountains

|[Figure 878]|
|---|

|[Figure 879]|
|---|

|[Figure 880]|
|---|

|[Figure 881]|
|---|

|[Figure 882]|
|---|

Text prompt: an old town with a church in the background

|[Figure 883]|
|---|

|[Figure 884]|
|---|

|[Figure 885]|
|---|

|[Figure 886]|
|---|

|[Figure 887]|
|---|

Text prompt: a castle with a dragon flying over it

|[Figure 888]|
|---|

|[Figure 889]|
|---|

|[Figure 890]|
|---|

|[Figure 891]|
|---|

|[Figure 892]|
|---|

Text prompt: a boat with lightning in the sky

|[Figure 893]|
|---|

|[Figure 894]|
|---|

|[Figure 895]|
|---|

|[Figure 896]|
|---|

|[Figure 897]|
|---|

Text prompt: a man with blonde hair

|[Figure 898]|
|---|

|[Figure 899]|
|---|

|[Figure 900]|
|---|

|[Figure 901]|
|---|

|[Figure 902]|
|---|

Original image Control ControlNet COFT OFT

Figure 28: More qualitative comparison on the sketch to image generation task.

##### G.3 Depth to Image

###### Text prompt: a bike

|[Figure 903]|
|---|

|[Figure 904]|
|---|

|[Figure 905]|
|---|

|[Figure 906]|
|---|

|[Figure 907]|
|---|

|[Figure 908]|
|---|

|[Figure 909]|
|---|

Text prompt: a motorcycle on display

|[Figure 910]|
|---|

|[Figure 911]|
|---|

|[Figure 912]|
|---|

|[Figure 913]|
|---|

|[Figure 914]|
|---|

|[Figure 915]|
|---|

|[Figure 916]|
|---|

Text prompt: a toilet in a bathroom

|[Figure 917]|
|---|

|[Figure 918]|
|---|

|[Figure 919]|
|---|

|[Figure 920]|
|---|

|[Figure 921]|
|---|

|[Figure 922]|
|---|

|[Figure 923]|
|---|

###### Text prompt: a city

|[Figure 924]|
|---|

|[Figure 925]|
|---|

|[Figure 926]|
|---|

|[Figure 927]|
|---|

|[Figure 928]|
|---|

|[Figure 929]|
|---|

|[Figure 930]|
|---|

Text prompt: a hospital room

|[Figure 931]|
|---|

|[Figure 932]|
|---|

|[Figure 933]|
|---|

|[Figure 934]|
|---|

|[Figure 935]|
|---|

|[Figure 936]|
|---|

|[Figure 937]|
|---|

###### Text prompt: a painting

|[Figure 938]|
|---|

|[Figure 939]|
|---|

|[Figure 940]|
|---|

|[Figure 941]|
|---|

|[Figure 942]|
|---|

|[Figure 943]|
|---|

|[Figure 944]|
|---|

Text prompt: an airplane at the airport

|[Figure 945]|
|---|

|[Figure 946]|
|---|

|[Figure 947]|
|---|

|[Figure 948]|
|---|

|[Figure 949]|
|---|

|[Figure 950]|
|---|

|[Figure 951]|
|---|

Text prompt: a cat in a suitcase

|[Figure 952]|
|---|

|[Figure 953]|
|---|

|[Figure 954]|
|---|

|[Figure 955]|
|---|

|[Figure 956]|
|---|

|[Figure 957]|
|---|

|[Figure 958]|
|---|

###### Text prompt: a kitchen

|[Figure 959]|
|---|

|[Figure 960]|
|---|

|[Figure 961]|
|---|

|[Figure 962]|
|---|

|[Figure 963]|
|---|

|[Figure 964]|
|---|

|[Figure 965]|
|---|

Original image Control ControlNet T2I-Adapter LoRA COFT OFT

Figure 29: Qualitative comparison among different methods on the depth to image generation task.

|[Figure 966]|
|---|

|[Figure 967]|
|---|

|[Figure 968]|
|---|

|[Figure 969]|
|---|

|[Figure 970]|
|---|

Text prompt: a river scene with a ship in the distance

|[Figure 971]|
|---|

|[Figure 972]|
|---|

|[Figure 973]|
|---|

|[Figure 974]|
|---|

|[Figure 975]|
|---|

Text prompt: a river in the mountains

|[Figure 976]|
|---|

|[Figure 977]|
|---|

|[Figure 978]|
|---|

|[Figure 979]|
|---|

|[Figure 980]|
|---|

Text prompt: an old town with a church in the background

|[Figure 981]|
|---|

|[Figure 982]|
|---|

|[Figure 983]|
|---|

|[Figure 984]|
|---|

|[Figure 985]|
|---|

Text prompt: a castle with a dragon flying over it

|[Figure 986]|
|---|

|[Figure 987]|
|---|

|[Figure 988]|
|---|

|[Figure 989]|
|---|

|[Figure 990]|
|---|

Text prompt: a boat with lightning in the sky

|[Figure 991]|
|---|

|[Figure 992]|
|---|

|[Figure 993]|
|---|

|[Figure 994]|
|---|

|[Figure 995]|
|---|

Text prompt: a man with blonde hair

|[Figure 996]|
|---|

|[Figure 997]|
|---|

|[Figure 998]|
|---|

|[Figure 999]|
|---|

|[Figure 1000]|
|---|

Original image Control ControlNet COFT OFT

Figure 30: More qualitative comparison on the depth to image generation task.

#### H Human Evaluation

Human evaluation settings. We also carried out a structured human evaluation for the subjectdriven generation task, involving 50 participants. Here’s a breakdown of our evaluation process:

- • Selection of subjects: we picked 7 subjects from the DreamBooth dataset9 at random.
- • Image and prompt: for every subject, 4 unique text prompts were chosen at random. This resulted in a total of 28 distinct subject-prompt combinations. For every single one of the 28 tasks, we randomly sampled an image generated by each of the three methods - DreamBooth, LoRA, and OFT.

Every participant was asked to answer three single-selection questions for each task:

- • Subject fidelity: which image best preserves the identity of the subject? In other words, which generated image resembles the most the original subject?
- • Text alignment: which image matches the given text description the best?
- • Overall image quality: out of the options, which image has the best overall quality?

The methods were assessed at two specific points during their fine-tuning phase: at the 1000th iteration, a checkpoint where these methods typically exhibit best performance, and at the 10,000th iteration, a checkpoint used to measure the stability of the finetuning process over an extended period.

Results. The results are given in Table 6, indicating the proportion of participants who chose a particular method based on the above criteria. We can see that OFT is more favored after finetuning Stable Diffusion with 1000 iterations and after 10000 iterations. We note that OFT delivers significantly better image quality and text-following ability than both DreamBooth and LoRA after a relatively large number of finetuning iterations.

|Metric|Iteration 1000<br><br>DreamBooth LoRA OFT<br><br>|Iteration 10000 DreamBooth LoRA OFT<br><br>|
|---|---|---|
|Subject Fidelity Text Alignment Overall Image Quality|42.0% 15.4% 42.6% 18.6% 24.7% 56.7% 35.7% 19.2% 45.1%<br><br>|22.4% 1.4% 76.2%<br><br>2.6% 1.4% 96.0% 11.6% 0.8% 87.6%<br><br>|

Table 6: Participant voting percentages for subject fidelity, text alignment and overall image quality.

###### 9https://github.com/google/dreambooth

#### I Style Transfer by Adapting Stable Diffusion with Orthogonal Finetuning

Stable Diffusion can generate images based on the input text prompts. Without any adaptation, inputting text prompts to a pre-trained Stable Diffusion model will result in images that resemble natural images. We can finetune the pre-trained Stable Diffusion model on a custom dataset, to adapt the style of the generated images to the custom dataset. To demonstrate the effectiveness of orthogonal finetuning, we show qualitative results of adapting Stable Diffusion on the Sketch Scene dataset10 after finetuning for 20000 iterations in Figure 31 and on the Wikiart dataset11 after finetuning for 30000 iterations in Figure 32. We train on a single NVIDIA A100-SXM4-80GB GPU using a learning rate of 1 × 10−4, batch size of 1, and 4 as the number of gradient accumulation steps.

Generated image Before OFT finetuning After OFT finetuning

Text prompt

|[Figure 1001]|
|---|

|[Figure 1002]|
|---|

|[Figure 1003]|
|---|

|[Figure 1004]|
|---|

|[Figure 1005]|
|---|

|[Figure 1006]|
|---|

Bench kept on the hill

A House in the field

a street in the snow

|[Figure 1007]|
|---|

|[Figure 1008]|
|---|

|[Figure 1009]|
|---|

|[Figure 1010]|
|---|

|[Figure 1011]|
|---|

|[Figure 1012]|
|---|

cows walking in the grass

a dog grabbing a flying disc

a herd of zebras

|[Figure 1013]|
|---|

|[Figure 1014]|
|---|

|[Figure 1015]|
|---|

|[Figure 1016]|
|---|

|[Figure 1017]|
|---|

|[Figure 1018]|
|---|

a big tree and a bench near the hillside

A boy is riding bike on the road

a train

(a) Example image-text pairs in the sketch-scene dataset

(b) Text-to-image generation after finetuning on the sketch-scene dataset

Figure 31: Direct OFT Finetuning of Stable Diffusion on the sketch-scene dataset.

Generated image Before OFT finetuning After OFT finetuning

Text prompt

|[Figure 1019]|
|---|

|[Figure 1020]|
|---|

|[Figure 1021]|
|---|

|[Figure 1022]|
|---|

|[Figure 1023]|
|---|

|[Figure 1024]|
|---|

a cityscape in the style of post impressionism

a restaurant on a city street

a landscape in the style of impressionism

|[Figure 1025]|
|---|

|[Figure 1026]|
|---|

|[Figure 1027]|
|---|

|[Figure 1028]|
|---|

|[Figure 1029]|
|---|

|[Figure 1030]|
|---|

a landscape in the style of realism

a living room with a fire place

a portrait in the style of realism

|[Figure 1031]|
|---|

|[Figure 1032]|
|---|

|[Figure 1033]|
|---|

|[Figure 1034]|
|---|

|[Figure 1035]|
|---|

|[Figure 1036]|
|---|

a stree at night time

a portrait in the style of impressionism

a landscape in the style of impressionism

(a) Example image-text pairs in the wikiart-caption dataset (b) Text-to-image generation after finetuning on the wikiart-caption dataset

Figure 32: Direct OFT Finetuning of Stable Diffusion on the wikiart-caption dataset.

- 10https://huggingface.co/datasets/zoheb/sketch-scene
- 11https://huggingface.co/datasets/fusing/wikiart_captions

#### J Failure Cases

We also show a few failure cases of OFT and COFT. Figure 33 gives three failure cases in subjectdriven generation. Figure 34 gives three failure cases in controllable generation.

##### J.1 Failure Cases in Subject-driven Generation

In subject-driven generation, OFT and COFT will sometimes fail to ground the text attribute to the intended object. In the cat example, both OFT and COFT will sometimes generate other red objects, instead of generating a red cat.

Text prompt: a cube shaped [V] dog

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

OFTOFTCOFTCOFT

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

Text prompt: a [V] vase floating in an ocean of milk

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

Text prompt: a red [V] cat

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

OFTCOFT

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

Input images

Figure 33: Some failure cases in subject-driven generation.

##### J.2 Failure Cases in Controllable Generation

Both OFT and COFT will sometimes hallucinate complicated structural details in a large region with the same semantics. Despite still being visually plausible, these generated images cannot match the original segmentation maps.

Text prompt: a field with trees and mountains in the background

|[Figure 1077]|
|---|

|[Figure 1078]|
|---|

|[Figure 1079]|
|---|

|[Figure 1080]|
|---|

|[Figure 1081]|
|---|

Text prompt: a living room

|[Figure 1082]|
|---|

|[Figure 1083]|
|---|

|[Figure 1084]|
|---|

|[Figure 1085]|
|---|

Text prompt: a building with stairs leading up to it

|[Figure 1086]|
|---|

|[Figure 1087]|
|---|

|[Figure 1088]|
|---|

Original image Control COFT OFT

Figure 34: Some failure cases in controllable generation.

