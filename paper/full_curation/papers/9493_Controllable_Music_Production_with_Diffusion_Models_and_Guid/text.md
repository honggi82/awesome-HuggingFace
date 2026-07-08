# arXiv:2311.00613v2[cs.SD]5Dec2023

## Controllable Music Production with Diffusion Models and Guidance Gradients

Mark Levy Apple mark_levy@apple.com

Bruno Di Giorgi Apple bdigiorgi@apple.com

Floris Weers Apple floris_weers@apple.com

Angelos Katharopoulos Apple a_katharopoulos@apple.com

Tom Nickson Apple tnickson@apple.com

### Abstract

We demonstrate how conditional generation from diffusion models can be used to tackle a variety of realistic tasks in the production of music in 44.1kHz stereo audio with sampling-time guidance. The scenarios we consider include continuation, inpainting and regeneration of musical audio, the creation of smooth transitions between two different music tracks, and the transfer of desired stylistic characteristics to existing audio clips. We achieve this by applying guidance at sampling time in a simple framework that supports both reconstruction and classification losses, or any combination of the two. This approach ensures that generated audio can match its surrounding context, or conform to a class distribution or latent representation specified relative to any suitable pre-trained classifier or embedding model. Audio samples are available at https://machinelearning.apple.com/research/controllablemusic.

### 1 Introduction

Recent work has shown great progress in addressing the challenging problem of generating musical audio with high enough quality for real world applications. Language modelling approaches such as MusicLM and MusicGen [1, 2] tackle the problem of sequence length by working with compressed, tokenised representations originally developed for efficient audio encoding, and by cascading coarse and fine models to achieve realistic sounding audio at up to 32kHz. Diffusion models such as Moûsai and Noise2Music [3, 4] also show promising results, again using a cascade of models. These systems focus on conditional generation from descriptions of the desired content, such as “a calming violin melody backed by a distorted guitar riff”. This allows the creation of an impressive variety of sounds but limits control over the musical output to generalised concepts, and conditioning relies on suitable paired data being available at training time. In our work, we scale up waveform and latent diffusion to reach high audio quality, and then explore some of the approaches to creative editing that become possible with a pre-trained diffusion model. Noting the similarity between the reconstruction guidance of [5] and the classifier guidance first introduced in [7], we combine both in a single framework, allowing us to tackle a wide range of useful music production tasks where control is provided in the form of example audio. Conditioning on audio prompts provides intuitive and fine-grained control over the musical characteristics of generated output, while applying conditioning with guidance gradients at sampling time removes the requirement to have paired data when training our diffusion models. By analogy with previous similar work on controllable image modification, we see huge potential for music production incorporating a diffusion model as a generative prior, and our work only scratches the surface of possible methods and applications.

Preprint. Under review.

### 2 Related work

##### 2.1 Creative editing with diffusion models

Diffusion models for high resolution images now provide the basis for a wide range of methods for creative editing of photos and videos. Inpainting can be implemented most simply by replacing the estimate of unmasked regions required during sampling with the original pixel values [8]. This approach can be improved by fine-tuning with masking, or even training from scratch with a suitable masking strategy [9, 10]. Individual elements of an image can be edited starting from a rough guide, where for example part or all of the desired final image is provided in the form of a sketch or shape mask [11]. If a paired dataset of sketch and image examples is available, then a separate control network can be fine-tuned to provide conditioning at sampling time [12]. With models trained to generate images conditioned on the embedding of a text prompt, semantic editing can be implemented by modifying the prompt and generating pixels within an explicitly specified region, or by manipulating cross-attention maps between text and image [13–15].

A related line of research uses diffusion models to solve inverse problems, where the aim is to reconstruct a complex signal as precisely as possible from a degraded version [16, 5, 10]. While the motivation for these methods is reconstruction of missing or corrupted data, they can easily be applied to creative settings, where we wish to regenerate some part of an original signal.

##### 2.2 Diffusion models for musical audio

Diffusion models have been applied to music by fine-tuning a pre-trained image model to generate audio spectrograms by treating them directly as images [17, 18]. Most recent work focuses on diffusion models trained directly on audio signals to ensure higher quality results. CRASH works directly in the waveform domain to generate short drum hits at 44.1kHz [19]. Noise2Music trains a cascade of diffusion models to generate 30 second clips of 24kHz audio, conditioned on text prompts learned from a large paired dataset of music and synthetic pseudo-labels [4]. Moûsai introduces a latent diffusion model for audio, using an independently trained diffusion autoencoder to compress spectrograms and then generating in the resulting latent domain [20, 3]. CQTDiff applies previous work on inverse problems to audio reconstruction with a diffusion model operating in the Constant-Q transform domain, including inpainting short sections of piano music recorded at 22kHz [21].

#### 3 Conditional generation with guidance gradients Similarly to [22], we formalise conditional sampling as a multi-objective optimization problem:

max

- J1(x) = pdata(x) min x
- J2(x) = d(y,A(x))

x

(1)

where x ∈ Rn is the sample, A ∈ Rn → V is a non-linear, differentiable measurement operator, y ∈ V is the measured output and d is a distance function V×V → R+. J1 is maximised for samples that have high probability under the data distribution pdata and is solved using the reverse process of a diffusion model. J2 is minimized for samples that are consistent with the measurement y.

Diffusion models solve the first sub-problem with an iterative algorithm, starting from noisy xT ∼ N(0,I) and refining the sample xt at every iteration, with t decreasing from T to 0. It is therefore a natural choice to solve the second problem by including a gradient descent step xt := xt−ξ∇xt

J2(xt) at each iteration of the denoising process, where ξ is the step size. This strategy can be interpreted as the alternation of unconditional updates and projections towards the measurement subspace [5].

Depending on A, specific variations can be applied. When A is defined for noiseless inputs only, a different “denoising” measurement operator A′ can be used instead. A′ is obtained as the composition of the original measurement operator A and the differentiable denoising operator of the diffusion model xˆ0(xt), which estimates the noiseless sample at each iteration. When A is linear A(x) = Ax with A ∈ Rm×n having full row rank, we can apply a data consistency step xt := xt + AT(AAT)−1(y−Axt) at the end of each iteration, which exactly projects xt onto the measurement subspace y = Ax. The resulting algorithms are provided in Section B of the appendix.

Unconditional Infill/Regeneration Continuation Transitions Cl. Guidance

Model Sampling Steps FAD↓ KLD↓ FAD↓ KLD↓ MR↓ FAD↓ KLD↓ MR↓ Realism↑ FAD↓ KLD↓ Latent DDIM 50 1.13 0.33 0.45/0.53 0.19/0.19 2.68/1.76 0.49 0.22 3.11 0.95±0.12 1.22 0.30

DDPM 50 1.10 0.34 0.39/0.42 0.17/0.17 2.63/2.30 0.50 0.18 3.00 0.95±0.12 1.31 0.21

500 0.72 0.28 0.55/0.49 0.18/0.19 2.63/1.72 0.46 0.20 3.03 0.94±0.10 0.96 0.08 Waveform DDPM 50 8.42 1.72 1.36/0.47 0.36/0.25 1.67/2.87 2.73 0.93 4.42 1.05±0.17 8.68 1.57

500 4.35 1.35 0.56/0.58 0.28/0.27 2.84/2.69 0.77 0.72 3.82 1.00±0.14 6.28 0.30 CQTDiff DDPM 50 1.98/ – 0.42/ – 3.10/ – VAE 0.47 0.13 0.51 0.15 1.48 0.40 0.12 1.48 0.96±0.09 0.47 0.13 Test Set 0.23 0.04 0.25 0.08 0.00 0.18 0.07 0.00 1.00±0.00 0.23 0.04

Table 1: Experimental results

### 4 Experiments

##### 4.1 Model architectures

We train two classes of diffusion models, a waveform model and a latent diffusion model. Our waveform model is a one-dimensional Unet [23] with 440M parameters. Our latent diffusion model comprises a Variational Autoencoder (VAE) with a downsampling ratio of 128 in the time dimension and a transformer diffusion model [24] with 1B parameters. See Section C of the appendix for full details. As a baseline we also train a CQT diffusion model [21], using the authors’ reference code.

##### 4.2 Dataset and metrics

We train all our diffusion models on the Free Music Archive dataset [25], which consists of 100k tracks totalling 8k hours of music. We hold out 80 hours as a test set.

To evaluate the quality of the generated audio we compute the Fréchet Audio Distance (FAD) [26] in the VGGish [27] embedding space; and the KL Divergence [28] in the AudioSet class space [29], using the Patchout classifier [30]. Both metrics are computed with respect to reference statistics computed on the training set.

###### 4.3 Creative applications We evaluate our models in a set of creative applications described below (further details in Section

- B.2 of the appendix). Results are shown in Table 1.

Unconditional generation: We sample 7k five second audio clips starting from Gaussian noise. As a reference we include the scores obtained by evaluating the clips in the test set and their reconstruction using the VAE, which serves as an upper bound for the Latent model.

Continuation: We take the first 2.4s from each test set example, and use the model to reconstruct a possible continuation up to 6s. We include the mel-reconstruction distance [31] (“MR” in Table 1) as a measure of consistency between the generated and the original continuation.

Infill / Regeneration: An audio segment with duration equal to 6s is extracted from each test set example. In the infill task, the middle 2 seconds of each segment are masked out and then generated using the model. In the regeneration task the original audio is partially noised using the forward process, which ensures that basic rhythmic structure is maintained while other details are obscured, and used as the starting sample instead of Gaussian noise. In both cases the left and right contexts are used to condition the generation as done in the continuation task.

Transitions: The transition task is a variant of the regeneration task: we start with two different tracks on the left and the right sides, and a 2.5s constant-power crossfade between them in the middle segment to be regenerated. Ideally we want the regenerated section to sound musical even when the raw crossfade contains rhythmic and harmonic clashes. We evaluate this with the realism score introduced in [32], normalised track-by-track by the score of the constant-power crossfade. We evaluate this task over a set of 100 transitions between randomly extracted pairs of tracks, with the reference manifold computed by projecting the training set on the Patchout classifier’s embedding space [30].

Mel-spectrogramdistance

- 0.0 0.2 0.4 0.6 0.8 1.0 Relative transition time

- 1

- 2

- 3

- 4

signal: gain-crossfade

signal: VAE

model: Latent

model: Waveform

Figure 1: Average mel-spectrogram distance of generated transitions with respect to the first track, over the duration of the transition. The distance increases linearly for all methods, implying smooth transitions. The ML-generated transitions start at non-zero distance because of the imperfect reconstruction.

| |
|---|

| |
|---|

| |
|---|

Original

Latent

Waveform

CQTDiﬀ

300

Figure 2: Subjective evaluation results. We show the number of wins in head-to-head preference comparisons between samples generated by each class of model and also the original i.e. the prompt audio itself.

Numberofwins

200

100

0

Inﬁll Regenerate Continuation Transition Guidance

We illustrate transition smoothness in Fig. 1, which shows the average mel-reconstruction error relative to the first track over the duration of the transition.

Classifier guidance: A common technique to improve the realism of conditionally generated samples is to use pre-trained classifiers to compute the gradients. We use the gradient of the L2 loss on the embedding space of the Patchout classifier [30] to guide our generation.

##### 4.4 Subjective evaluation

We run a blind pairwise comparison test where we present the rater two samples generated by two models (or one sample vs the reference audio) for the same audio prompt and creative task. The raters are asked to choose the preferred sample from each pair based on perceived quality. We also encourage the reader to listen to the samples available at https://machinelearning.apple.com/research/controllablemusic.

##### 4.5 Discussion

Results in Table 1 show that, regardless of the sampling method, the model operating in a latent space generally produces musical audio that is higher quality and closer to the original than the waveform diffusion model. This general trend confirms that cascading generation is beneficial, as previously suggested in [4]. Both models improve over infilling with the baseline CQTDiff model by a large margin. We summarise the results of the subject evaluation in Fig. 2. These confirm the superiority of our models over the baseline and suggest that the latent representation gives an advantage in tasks where a data consistency step is unavailable (guidance) or weaker (continuation, where we have only a one-sided prompt).

### 5 Conclusions

In this paper we explored a simple framework that enables different applications of diffusion generative models in the context of high-fidelity music production. Applications such as continuation, inpainting, regeneration, transition and latent conditioning have been evaluated for two model architectures showing the relative importance of architectural and sampling choices for the different tasks.

### References

- [1] Andrea Agostinelli, Timo I. Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, Matt Sharifi, Neil Zeghidour, and Christian Frank. MusicLM: Generating music from text, 2023.
- [2] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. Simple and controllable music generation, 2023.
- [3] Flavio Schneider, Zhijing Jin, and Bernhard Schölkopf. Moûsai: Text-to-music generation with long-context latent diffusion, 2023.
- [4] Qingqing Huang, Daniel S. Park, Tao Wang, Timo I. Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, Jesse Engel, Quoc V. Le, William Chan, Zhifeng Chen, and Wei Han. Noise2Music: Text-conditioned music generation with diffusion models, 2023.
- [5] Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems, 2023.
- [6] Eloi Moliner, Jaakko Lehtinen, and Vesa Välimäki. Solving Audio Inverse Problems with a Diffusion Model, 2022.
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.
- [8] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution, 2020.
- [9] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [10] Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alexandros G. Dimakis, and Adam Klivans. Ambient diffusion: Learning clean distributions from corrupted data, 2023.
- [11] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [12] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.
- [13] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022.
- [14] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [15] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [16] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems, 35:23593–23606, 2022.
- [17] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

- [18] Seth Forsgren and Hayk Martiros. Riffusion - Stable diffusion for real-time music generation. 2022.
- [19] Simon Rouard and Gaëtan Hadjeres. CRASH: Raw Audio Score-based Generative Modeling for Controllable High-resolution Drum Sound Synthesis, June 2021. arXiv:2106.07431 [cs, eess].
- [20] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10619–10629, 2022.
- [21] Eloi Moliner, Jaakko Lehtinen, and Vesa Välimäki. CQT-Diff: Solving Audio Inverse Problems with a Diffusion Model, November 2022. arXiv:2210.15228 [cs, eess].
- [22] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated MRI. Medical image analysis, 80:102479, 2022.
- [23] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
- [24] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers, 2022.
- [25] Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. In 18th International Society for Music Information Retrieval Conference, number CONF, 2017.
- [26] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fréchet audio distance: A metric for evaluating music enhancement algorithms. arXiv preprint arXiv:1812.08466, 2018.
- [27] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan Seybold, et al. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 131–135. IEEE, 2017.
- [28] Vladimir Iashin and Esa Rahtu. Taming visually guided sound generation. arXiv preprint arXiv:2110.08791, 2021.
- [29] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 776–780, 2017.
- [30] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. In Interspeech 2022. ISCA, sep 2022.
- [31] Hugo Flores Garcia, Prem Seetharaman, Rithesh Kumar, and Bryan Pardo. VampNet: Music generation via masked acoustic token modeling, 2023.
- [32] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019.
- [33] Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A Generative Model for Music, April 2020. arXiv:2005.00341 [cs, eess, stat].
- [34] Curtis Hawthorne, Andrew Jaegle, C˘at˘alina Cangea, Sebastian Borgeaud, Charlie Nash, Mateusz Malinowski, Sander Dieleman, Oriol Vinyals, Matthew Botvinick, Ian Simon, Hannah Sheahan, Neil Zeghidour, Jean-Baptiste Alayrac, João Carreira, and Jesse Engel. General-purpose, long-context autoregressive modeling with perceiver ar, 2022.

- [35] Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, and Yonghui Wu. w2v-BERT: Combining contrastive learning and masked language modeling for self-supervised speech pre-training, 2021.
- [36] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked generative image transformer, 2022.
- [37] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2021.
- [38] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.
- [39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020.
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. StableDiffusion: High-Resolution Image Synthesis with Latent Diffusion Models, April 2022. arXiv:2112.10752 [cs].
- [41] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [42] Ryuichi Yamamoto, Eunwoo Song, and Jae-Min Kim. Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6199–6203. IEEE, 2020.
- [43] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High Fidelity Neural Audio Compression, October 2022. arXiv:2210.13438 [cs, eess, stat].
- [44] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155, 2018.

### A Other related work

- A.1 Language models for music generation

We focus in this paper on diffusion models, but there is a growing body of parallel work on language modelling for musical audio, operating in the discrete token space of VQ-VAEs. Jukebox uses a cascade of three transformers trained with metadata and song lyrics as well as encoded audio, to enable generation conditioned on song texts and artist identity [33]. PerceiverAR and MusicGen model interleaved sequences of audio tokens from the multiple residual VQ-VAE codebooks of neural codecs, while MusicLM extends this approach by conditioning on “semantic" tokens from a pre-trained first stage model which models audio at a lower sample rate [34, 1, 2, 35]. MusicLM is conditioned on text captions and also on melody, by concatenating automatically extracted melody tokens with the text tokens, and this supports generation of samples conditioned on a provided melody as well as a text prompt.

In contrast to these autoregressive models, VampNet introduces a language model trained on a masked token prediction objective which supports inpainting, and also employs an efficient iterative sampling scheme to generate interleaved codebook tokens [31, 36]. The authors experiment with enforcing the rhythmic consistency of generated sections by retaining tokens around the expected beat positions in the region to be inpainted.

B Conditional generation

- B.1 Diffusion models

A Gaussian denoising diffusion probabilistic model (DDPM) defines a forward corruption process through time, which gradually applies noise to real data x0 ∼ pdata

q(xt|x0) = N(xt;αtx0,σt2I),

where αt and σt are constants defining a noise schedule with monotonically decreasing signal-tonoise ratio αt/σt, such that xT ∼ N(0,I). We use a cosine schedule with continuous t ∈ [0,1], αt = cos(πt/2) and αt2 + σt2 = 1. We can sample noised data directly with xt = αtx0 + σtϵt and ϵt ∼ N(0,I).

DDPMs are trained to learn a reverse process pθ(xs|xt) = N(µθ(xt,t),σt→s) where σt→s = (σs2/σt2)(1 − αt2/αs2) are constants computed from the noise schedule and µθ is the output of a denoising model that estimates x0 or equivalently ϵt. We choose the so called v prediction parameterization for our estimator, learning to predict a combination of the added noise and the original signal vt = αtϵt − σtx0 [37]. We train our model to minimise a reconstruction loss

0∼pdata,ϵ∼N(0,I) vt − vθ(xt,t) 2 .

L(θ) = Et∼U(0,1),x

Sampling from a DDPM starts with Gaussian noise xT and uses pθ(xs|xt) to produce gradually less noisy samples until reaching a final sample x0. An important result from [38] is that the output of the denoising model is equivalent to estimating an evidence lower bound on the score function ∇xt

log p(xt), which enables the use of samplers derived from denoising score matching, such as DDIM [39].

##### B.2 Sampling with guidance gradients

With reference to Sect. 3, Algorithms 1 and 2 illustrate the DDPM and DDIM iterative methods for conditional generation. We model different conditional generation tasks by varying specific parameters of the algorithm: the measurement operator A, the distance function d and the starting sample xT, as shown in Table 2.

Task A(x) d(y, A(x)) y, xT

y = Ax¯ xT = AT y + (I − AT A)z

Continuation A(x) = Ax, A = AL ∥y − Ax∥1

###### AL AR

Infill A(x) = Ax, A =

as Continuation as Continuation

y = Ax¯ xT = AT y + (I − AT A)(kz + (1 − k)¯x)

Regenerate as Infill as Continuation

  

  

AL¯xL Fout¯xL + Fin¯xR AR¯xR

¯x =

Transition as Infill as Continuation

y = Ax¯ xT = AT y + (I − AT A)(kz + (1 − k)¯x) Fout = [0, diag(fout), 0] Fin = [0, diag(fin), 0]

Embedder guidance A ∈ Rn → Rm ∥y − Ax∥2 y = A(¯x), xT = z Classifier guidance A(x)i = p(ci|x) ∈ [0, 1]m BCE(y, A(x)) y ∈ [0, 1]m, xT = z

Table 2: Task specific parameters. AL = [IC

R×n; CL and CR are the sample lengths of the left and right contexts respectively, CL + CR < n; x¯, x¯L, x¯R are target signals; z is a noise signal z ∼ N(0,I), k ∈ [0,1] is a scalar coefficient that regulates the amount of noise in the infill region of the initial sample for the regenerate and transition tasks, BCE stands for Binary Cross-Entropy. In the transition task fout and fout represent fade out and fade in coefficients of a constant-power cross-fade. The audio channel dimension is not considered for simplicity.

L×n, AR = [0,IC

,0] ∈ RC

] ∈ RC

L

R

- Algorithm 1 Sampling with guidance gradients

Require: initial sample xT, measurement y, distance function d, measurement operator A(·), gradient step size ξ, N timesteps ti in (T,0) for i = N to 1 do

t ← ti, s ← ti−1 xˆ0 ← αtxt − σtvθ(xt,t) ϵ ∼ N(0,I)

xs ←

αs σt2

(1 −

αt2 αs2

)xˆ0 +

αtσs2 αsσt2

xt + σt→sϵ xs ← xs − ξ∇xt

d(y,A(xt)) ▷ Alternatively A(xˆ0)

▷ Apply data consistency step if possible (see Sect. 3) end for return x0

- Algorithm 2 DDIM with guidance gradients

Require: initial sample xT, measurement y, distance function d, measurement operator A(·), gradient step size ξ, N timesteps ti in (T,0) for i = N to 1 do

t ← ti, s ← ti−1 xˆ0 ← αtxt − σtvθ(xt,t) ˆϵt ← (xt − αtxˆ0)/σt ˆϵt ← ˆϵt − ξσt∇xt

d(y − A(xt)) ▷ Alternatively A(xˆ0) xs ← αsxˆ0 + (1 − αs2)ˆϵt

▷ Apply data consistency step if possible (see Sect. 3) end for return x0

### C Model architecture and training details

##### C.1 Models

Our waveform diffusion model uses a similar Unet implementation to [40] to generate 11.9 seconds of 44.1kHz stereo audio. To deal efficiently with the large input size we “fold” the 2-dimensional stereo input into the channel dimension and “unfold” the output. We minimize possible aliasing artifacts by using overlapping windows with frame size 32 and hop size 16, and multiplying with a hamming window function before aggregating.

Our latent diffusion model comprises a Variational Autoencoder (VAE) to extract the latent representations and a transformer diffusion model to generate 5.9 seconds of audio. Contrary to previous works on images, we use a much higher downsampling ratio for our VAE to make the input size tractable for the transformer. In particular, our VAE consists of 8 layers in the encoder and decoder. After every layer but the first one we use a 2× down- or upsampling operation respectively, resulting in a latent sequence 128× smaller than the input. The layers are implemented as 1D ConvNext [41] layers, which we observed perform better than the traditional dilated convolutions. Finally, we train our VAE using frequency losses as well as L1 and L2 losses [42]. To improve the fidelity of the audio reconstructed from the latent space, we finetune the decoder while keeping the encoder fixed using a multi-scale frequency based discriminator for a small number of steps (similar to [43]). Our transformer model for latent diffusion follows [24] closely, the main difference being the addition of a relative positional encoding [44]. We use 32 layers with 1536 feature dimensions resulting in 1B parameters.

##### C.2 Training

The models are trained with the AdamW optimizer, β1 = 0.9, β2 = 0.999, and no weight decay. A cosine learning rate schedule with a warmup of 5000 steps is used at the beginning of training. Training uses fp16 mixed precision and distributed data parallelism. The waveform model is trained for 600k steps with a batch size of 384 on 24 A100 GPUs, and the latent model for 300k steps with a batch size of 96 on 8 A100 GPUs.

##### C.3 Sampling parameters

We set the guidance gradient step size ξ differently for latent (ξ = 3 × 10−2) and waveform (ξ = 3 × 10−3) models, based on informal qualitative evaluation. The waveform models appeared more sensitive to ξ and degraded to generating noisy signals for higher values ξ > 3 × 10−2. We kept the step size fixed throughout the sampling process. We created initial samples for the regenerate and transition tasks using a noise mixing coefficient k = 0.85.

