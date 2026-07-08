## Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model

# arXiv:2311.13231v3[cs.LG]23Mar2024

Kai Yang1* Jian Tao1* Jiafei Lyu1† Chunjiang Ge2 Qimai Li3 Jiaxin Chen3 Weihan Shen3 Xiaolong Zhu3 Xiu Li1† 1 Tsinghua Shenzhen International Graduate School, Tsinghua University 2 Department of Automation, Tsinghua University 3 Parametrix Technology Company Ltd.

{yk22,tj22,lvjf20}@mails.tsinghua.edu.cn li.xiu@sz.tsinghua.edu.cn

### Abstract

Using reinforcement learning with human feedback (RLHF) has shown significant promise in fine-tuning diffusion models. Previous methods start by training a reward model that aligns with human preferences, then leverage RL techniques to fine-tune the underlying models. However, crafting an efficient reward model demands extensive datasets, optimal architecture, and manual hyperparameter tuning, making the process both time and cost-intensive. The direct preference optimization (DPO) method, effective in fine-tuning large language models, eliminates the necessity for a reward model. However, the extensive GPU memory requirement of the diffusion model’s denoising process hinders the direct application of the DPO method. To address this issue, we introduce the Direct Preference for Denoising Diffusion Policy Optimization (D3PO) method to directly fine-tune diffusion models. The theoretical analysis demonstrates that although D3PO omits training a reward model, it effectively functions as the optimal reward model trained using human feedback data to guide the learning process. This approach requires no training of a reward model, proving to be more direct, costeffective, and minimizing computational overhead. In experiments, our method uses the relative scale of objectives as a proxy for human preference, delivering comparable results to methods using ground-truth rewards. Moreover, D3PO demonstrates the ability to reduce image distortion rates and generate safer images, overcoming challenges lacking robust reward models. Our code is publicly available at https://github.com/yk7333/D3PO.

### 1. Introduction

Recent advances in image generation models have yielded unprecedented success in producing high-quality images

*Equal contribution. † Corresponding authors.

from textual prompts [30, 44, 49]. Diverse approaches, including Generative Adversarial Networks (GANs) [23], autoregressive models [14, 15, 19, 22, 44, 58], Normalizing Flows [16, 46], and diffusion-based techniques [40, 45, 47, 49], have rapidly pushed forward the capabilities of these systems. With the proper textual inputs, such models are now adept at crafting images that are not only visually compelling but also semantically coherent, garnering widespread interest for their potential applications and implications.

To adapt the aforementioned models for specific downstream tasks, such as the generation of more visually appealing and aesthetic images, Reinforcement Learning from Human Feedback (RLHF) is commonly employed [12]. This technique has been successfully used to refine large language models such as GPT [8, 41]. Now, efforts are being made to extend this method to diffusion models to enhance their performance. One such approach, the DDPO method [5], aims to enhance image complexity, aesthetic quality, and the alignment between prompts and images. The ReLF approach [60] introduces a novel reward model, named ImageReward, which is specifically trained to discern human aesthetic preferences in text-to-image synthesis. This model is then utilized to fine-tune diffusion models to produce images that align more closely with human preferences. Nonetheless, developing a robust reward model for various tasks can be challenging, often necessitating a vast collection of images and abundant training resources. For example, to diminish the rate of deformities in character images, one must amass a substantial dataset of deformed and non-deformed images generated from identical prompts. Subsequently, a network is constructed to discern and learn the human preference for non-deformed imagery, serving as the reward model.

In the field of natural language processing, Direct Preference Optimization (DPO) has been proposed to reduce training costs [43]. This method forgoes the training of a

|Human feedback<br><br>A dog is washing dishes<br><br>[Figure 1]<br><br>[Figure 2]<br><br>Objective: Prompt-image Alignment/ Incompressity/Safety ...<br><br>Direct preference for Denoising Diffusion Policy Optimization (D3PO)<br><br>xT …<br><br>xt1 xt-11 … x01<br><br>Prompts<br><br>xt0 xt-10 … x00<br><br>|[Figure 3]<br><br>Good|
|---|
<br><br>|[Figure 4]<br><br>Bad|
|---|
<br><br>pref(xt-11|xt1,c)<br><br>pθ(xt-10|xt0,c)<br><br>pθ(xt-11|xt1,c)<br><br><br>Step1: Sampling two images without grad<br><br>Step2: Get human feedback<br><br>Step3: Calculating pθand pref in each timestep with grad<br><br>pref(xt-10|xt0,c)<br><br>Step4: Training with D3PO loss<br><br>Updating parameter θ<br><br>w=1 l=0<br><br>Figure 1. Overview of D3PO. The diffusion model generates two corresponding images based on the provided prompts. Guided by specific task requirements—such as improving prompt-image alignment, enhancing image incompressibility, or refining aesthetic quality—human evaluators select the preferred image. Leveraging this human feedback, our method directly updates the diffusion model’s parameters without necessitating the training of a reward model.|
|---|

ing by utilizing direct human feedback, making the process more efficient and cost-effective.

reward model and directly fine-tunes language models according to human preferences. However, this straightforward and easy-to-train method encounters challenges when applied to fine-tune diffusion models. During the DPO training process, the complete sentence generated by the language model is treated as a single output, necessitating the storage of gradients from multiple forward passes. With diffusion models, one must store the gradients across multiple latent image representations, which are significantly larger than word embeddings, leading to memory consumption that is typically unsustainable.

- • We expand the theoretical framework of DPO into a multi-step MDP, demonstrating that directly updating the policy based on human preferences within an MDP is equivalent to learning the optimal reward model first and then using it to guide policy updates. This establishes a robust theoretical foundation and provides assurance for our proposed method.
- • In our experiments, we have demonstrated the effectiveness of our method by using human feedback to successfully address issues of hand and full-body deformities, enhance the safety of generated images, and improve prompt-image alignment.

To address the issue of high computational overhead and enable the use of the DPO method to fine-tune diffusion models directly with human feedback, we conceptualize the denoising process as a multi-step MDP, which utilizes a pre-trained model to represent an action value function Q that is commonly estimated in RL. We extend the theoretical framework of DPO into the formulated MDP, which allows for direct parameter updates at each step of the denoising process based on human feedback, thereby circumventing the significant computational costs and eliminating the need for a reward model. To the best of our knowledge, this is the first work that fine-tune diffusion models without reward models. Our main contributions are as follows:

### 2. Related Work

Diffusion models. Denoising diffusion probabilistic models, introduced in [54] and further advanced by [26], have emerged as powerful tools for generating diverse data types. They have been successfully applied in various domains such as image synthesis [44, 49], video generation [27, 36, 53], and robotics control systems [1, 11, 29]. Notably, test-to-image diffusion models have enabled the creation of highly realistic images from textual descriptions [44, 49], opening new avenues in digital art and design.

• We introduce an innovative approach for fine-tuning diffusion models that could significantly modify the current RLHF framework for fine-tuning diffusion models. This method bypasses resource-intensive reward model train-

Recent studies have focused on refining the guidance of diffusion models for more precise manipulation over the

generative process. Techniques such as adapters [63] and compositional approaches [17, 38] have been introduced to incorporate additional input constraints and blend multiple models, respectively, enhancing image quality and generation control. The implementation of classifier-based [13] and classifier-free guidance [25] has also contributed significantly to achieving greater autonomy in the generation process, resulting in outputs that closely align with user intentions. In our work, we utilize Stable Diffusion [45] to generate images based on specific prompts.

RLHF. RLHF stands as a salient strategy in the domain of machine learning when objectives are complex or difficult to define explicitly. This technique has been instrumental across various applications, from gaming, as demonstrated with Atari [4, 12], to more intricate tasks in robotics [10, 64]. The integration of RLHF into the development of large language models (LLMs) has marked a significant milestone in the field, with notable models like OpenAI’s GPT-4 [41], Anthropic’s Claude [2], Google’s Bard [24], and Meta’s Llama 2-Chat [57] leveraging this approach to enhance their performance and relevance. The effectiveness of RLHF in refining the behavior of LLMs to be more aligned with human values, such as helpfulness and harmlessness, has been extensively studied [4, 64]. The technique has also proven beneficial in more focused tasks, such as summarization, where models are trained to distill extensive information into concise representations [55]. Some recent research utilizes Reinforcement Learning from AI Feedback (RLAIF) [32, 62] as an alternative to RLHF for model fine-tuning. RLAIF offers convenience and efficiency by replacing human feedback with AI-generated feedback. However, for tasks like assessing hand generation normality or image aesthetic appeal, reliable judgment models are currently lacking. Hence, this paper still relies on human feedback for evaluation.

Fine-tune Diffusion Models with RL. Before applying diffusion models, data generation has been regarded as a sequential decision-making problem and combined with reinforcement learning [3]. More recently, the SFT method [20] applied reinforcement learning to diffusion models to enhance existing fast DDPM samplers [26]. Reward Weighted method [34] explored using human feedback to align textto-image models. It uses the reward model for the coefficients of the loss function instead of using reinforcement learning optimization objectives. ReFL [60] employs the framework of RLHF. It begins by training a model called ImageReward based on human preferences and then finetunes the diffusion model using reinforcement learning. DDPO [5] treats the denoising process of diffusion models as a MDP to fine-tune diffusion models with many reward models. DPOK [21] combine the KL divergence into the DDPO loss and use it to better align text-to-image objectives. All these models need a robust reward model, which

demands a substantial dataset of images and extensive human evaluations.

Direct Preference Optimization. In the realm of reinforcement learning, exploring policies derived from preferences rather than explicit rewards has gained attention through various methods. The Contextual Dueling Bandit (CDB) framework [18, 61] introduces the concept of a von Neumann winner, shifting away from the pursuit of an optimal policy directly based on rewards. Preference-based Reinforcement Learning (PbRL) [9, 39, 48] learns from binary preferences inferred from a cryptic scoring function instead of explicit rewards. Recently, the DPO approach [43] was proposed, which fine-tunes LLMs directly using preferences. DPO leverages the correlation between reward functions and optimal policies, effectively addressing the challenge of constrained reward maximization in a single phase of policy training.

### 3. Preliminaries

MDP. We consider the MDP formulation described in [56]. In this setting, an agent perceives a state s ∈ S and executes an action, where S,A denote state space and action space, respectively. The transition probability function, denoted by P(s′|s,a), governs the progression from the current state s to the subsequent state s′ upon the agent’s action a. Concurrently, the agent is awarded a scalar reward r, determined by the reward function r : S×A → R. The agent’s objective is to ascertain a policy π(a|s) that maximizes the cumulative returns of trajectories τ = (s0,a0,s1,a1,...,sT−1,aT−1), which can be represented as: J (π) = Eτ[ Tt=0−1 r (st,at)].

Diffusion models. Diffusion models learn to model a probability distribution p(x) by inverting a Markovian forward process q(xt|xt−1) which adds noise to the data. The denoising process is modeled by a neural network to predict the mean of xt−1 or the noise ϵt−1 of the forward process. In our work, we use network µθ(xt;t) to predict the mean of xt−1 instead of predicting the noise. Using the mean squared error (MSE) as a measure, the objective of this network can be written as:

L = Et∼[1,T],x0∼p(x0),xt∼q(xt|x0)[∥µ˜(x0, xt) − µθ (xt, t)∥2], (1)

where µ˜θ(xt,x0) is the forward process posterior mean. In the case of conditional generative modeling, the diffusion models learn to model p(x|c), where c is the conditioning information, i.e., image category and image caption. This is done by adding an additional input c to the denoising neural network, as in µθ(xt,t;c). To generate a sample from the learned distribution pθ(x|c), we start by drawing a sample xT ∼ N(0,I) and then progressively denoise the sample by iterated application of ϵθ according to a specific sampler [26]. Given the noise-related parameter σt, the reverse process can be written as:

pθ (xt−1 | xt,c) = N xt−1;µθ (xt,c,t),σt2I . (2)

Reward learning for preferences. The basic framework to model preferences is to learn a reward function r∗(s,a) from human feedback [12, 33, 59]. The segment σ = {sk,ak,sk+1,ak+1,...,sm,am} is a sequence of observations and actions. By following the Bradley-Terry model [6], the human preference distribution p∗ by using the reward function can be expressed as:

exp( Tt=k r∗(s1t,a1t)) i∈{0,1} exp( Tt=k r∗(sit,ait))

p∗(σ1 ≻ σ0) =

, (3)

where σi ≻ σj denotes that segment i is preferable to segment j. Now we have the preference distribution of human feedback, and we want to use a network rϕ to approximate r∗. Given the human preference y ∈ {(1,0),(0,1)} which is recorded in dataset D as a triple (σ0,σ1,y), framing the problem as a binary classification, the reward function modeled as a network is updated by minimizing:

L(ϕ) = −E(σ

1,σ0,y)∼D[y(0)log pϕ(σ0 ≻ σ1)

+ y(1)log pϕ(σ1 ≻ σ0)]. (4)

### 4. Method

In this section, we describe a method to directly fine-tune diffusion models using human feedback, bypassing the conventional requirement for a reward model. Initially, we reinterpret the denoising process inherent in diffusion models

- as a multi-step MDP. Then we extend the theory of DPO to MDP, which allows us to apply the principles to effectively translate human preferences into policy improvements in diffusion models.

- 4.1. Denoising process as a multi-step MDP

We conceptualize the denoising process within the diffusion model as a multi-step MDP, which varies slightly from the approach outlined in [5]. To enhance clarity, we have redefined the states, transition probabilities, and policy functions. The correspondence between notations in the diffusion model and the MDP is established as follows:

st ≜ (c, t, xT−t) P (st+1 | st, at) ≜ δc, δt+1, δxT−1−t

- at ≜ xT−1−t π (at | st) ≜ pθ (xT−1−t | c, t, xT−t) ρ0 (s0) ≜ (p(c), δ0, N(0, I)) r(st, at) ≜ r((c, t, xT−t) , xT−t−1)

where δx represents the Dirac delta distribution, and T denotes the maximize denoising timesteps. Leveraging this mapping, we can employ RL techniques to fine-tune diffusion models by maximizing returns. However, this approach requires a proficient reward model capable of adequately rewarding the noisy images. The task becomes exceptionally challenging, particularly when t is low, and xT−t closely resembles Gaussian noise, even for an experienced expert.

###### 4.2. Direct Preference Optimization for MDP

The DPO method does not train a separate reward model but instead directly optimizes the LLMs with the preference data. Given a prompt x and a pair of answers (y1,y0) ∼ πref(y|x), where πref represents the reference (pre-trained) model, these responses are ranked and stored in D as a tuple (x,yw,yl), where yw denotes the preferred answer and yl indicates the inferior one. DPO optimizes πθ with the human preference dataset by using the following loss:

θ(yl|x)

θ(yw|x)

πref(yw|x) − β log π

w,yl)∼D log ρ β log π

LDPO(θ) = −E(x,y

πref(yl|x) .

(5)

Here ρ is the logistic function, and β is the parameter controlling the deviation from the πθ and πref. In our framework, we treat segments σ1,σ0 as y1,y0 and use DPO to fine-tune diffusion models. However, directly using this method faces difficulties since the segments contain a large number (usually 20–50) of the image latent, which occupy a large amount of GPU memory (each image is about 6G even when using LoRA [28]). Since we can only get human preferences for the final image x0, if we want to update πθ(σ) = Tt=k πθ(st,at), it will consume more than 100G GPU memory, which makes the fine-tuning process nearly impossible.

To address this problem, we extend the DPO theory to MDP. Firstly, we need to reconsider the objective of the RL method. For the MDP problem, the agents take action by considering maximizing the expected return instead of the current reward. For actor-critic methods such as DDPG [37], the optimization objective of policy π gives:

Es∼dπ,a∼π(·|s)[Q∗(s,a)]. (6)

max

π

Here, dπ = (1 − γ) ∞t=0 γtPtπ(s) represents the state visitation distribution, where Ptπ(s) denotes the probability of being in state s at timestep t given policy π. Additionally, Q∗(s,a) denotes the optimal action-value function. The optimal policy can be written as:

π∗(a|s) =

1, if a = arg maxaˆ Q∗(s,aˆ), 0, otherwise.

(7)

Similar to some popular methods, we use KL-divergence to prevent the fine-tuned policy from deviating from the reference policy, hence relieving the out-of-distribution (OOD) issue. By incorporating this constraint, the RL objective can be rewritten as:

Es∼dπ,a∼π(·|s)[Q∗(s, a)] − βDKL[π(a|s)∥πref(a|s)]. (8)

max

π

Here, β is the temperature parameter that controls the deviation of πθ(a|s) and πref(a|s).

[Figure 5]

- Figure 2. Progression of samples targeting compressibility, incompressibility, and aesthetic quality objectives. With the respective focus during training, images exhibit reduced detail and simpler backgrounds for compressibility, richer texture details for incompressibility, and an overall increased aesthetic appeal when prioritizing aesthetic quality.

- Proposition 1 Given the objective of Eq. (8), the optimal policy π∗(a|s) has the following expression:

π∗(a|s) = πref(a|s)exp(

1 β

Q∗(s,a)). (9)

The proof can be seen in Appendix B.1. By rearranging the formula of Eq. (9), we can obtain that:

Q∗(s,a) = β log

π∗(a|s) πref(a|s)

. (10)

Now, considering Eq. (3) and noticing that Q∗(st,at) = E Tt=k r∗(st,at) under the policy π∗(a|s), we make a substitution. By replacing Tt=k r∗(st,at) with Q∗(st,at), we define a new distribution that can be rewritten as:

p˜∗(σ1 ≻ σ0) =

exp(Q∗(s1k,a1k)) i∈{0,1} exp(Q∗(sik,aik))

. (11)

We suppose mt=k r∗(st,at) is sampled from a normal

distribution with mean E[ mt=k r∗(st,at)] and standard deviation σ2. From a statistical perspective, we can establish

the relationship between the new distribution p˜∗(σ1 ≻ σ0) and the raw distribution p∗(σ1 ≻ σ0).

- Proposition 2 For i ∈ {0,1}, suppose the expected re-

turn satisfies a normal distribution, i.e., Tt=0 r∗ sit,ait ∼ N Q∗(si0,ai0),σ2 . Given Q∗(s,a) ∈ [Qmin,Qmax] where Qmin and Qmax represent the lower and upper bounds of the values, then

(ξ2 + 1)(exp(σ2) − 1) 16ξδ

|p∗ (σ1 ≻ σ0) − p˜∗ (σ1 ≻ σ0)| <

with probability at least 1 − δ, where ξ = exp(Q

max)

exp(Qmin) .

The proof can be seen in Appendix B.2. In practical applications, as the volume of data increases, it becomes easier to satisfy the assumption of normality. Additionally, we can use clipping operations to constrain the Q values within a certain range, which introduces upper and lower bounds. Therefore, the aforementioned assumption is reasonable. As shown in proposition 2, their deviation can be

ξ δ

bounded at the scale of O(

(exp(σ2) − 1)). It is clear that

this bound can approach 0 if the σ2 is close to 0. In practice, σ2 approaches 0 if the Q function can be estimated with a small standard deviation.

By combining Eq. (11), Eq. (4), and Eq. (10), replacing p∗ (σ1 ≻ σ0) with p˜∗ (σ1 ≻ σ0), and substituting π∗(s,a) with the policy network πθ(s,a) that requires learning, we derive the following loss function for updating πθ(a|s):

w k |swk )

l k|slk)

k,σw,σl)[log ρ(β log πθ(a

πref(awk |swk ) − β log πθ(a

L(θ) = −E(s

πref(alk|slk))],

(12)

where σw = {swk , awk , swk+1, awk+1, ..., swT , awT } denotes the segment preferred over another segment σl = {slk, alk, slk+1, alk+1, ..., slT, alT}.

###### 4.3. Direct preference for Denoising Diffusion Policy Optimization

Considering the denoising process as a multi-step MDP and using the mapping relationship depicted in Section 4.1, we can use DPO to directly update diffusion models by using Eq. (12). In the denoising process, we set k = 0 and

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 3. Comparison of D3PO against existing methods. The horizontal axis represents the number of image sample pairs generated for updating parameters. The rewards denote image size for incompressity objective, negative image size for the compressity objective, and the LAION aesthetic score for the aesthetic objective. All experiments are conducted with 5 different seeds.

T = 20. We first sample an initial state sw0 = sl0 = s0 and then use Eq. (2) to generate two segments. After manually choosing which segment is better, the probability of πθ(aw0 |sw0 ) is gradually increasing and πθ(al0|sl0) is decreasing, which guides the diffusion model to generate images of human preference. However, the approach of only updating πθ(·|s0) does not fully utilize the information within the segment.

Since the middle states of the segment are noises and semi-finished images, it is hard for humans to judge which segment is better by observing the whole segment. But we can conveniently compare the final image x0. Like many RL methods [7, 51, 52] which give rewards by ∀st,at ∈ σ,r(st,at) = 1 for winning the game and ∀t ∈ σ,r(st,at) = −1 for losing the game, we also assume that if the segment is preferred, then any state-action pair of the segment is better than the other segment. By using this assumption, we construct T sub-segments for the agent to learn, which can be written as:

σi = {si,ai,si+1,ai+1,...,sT−1,aT−1}, 0 ≤ i ≤ T − 1

(13)

Using these sub-segments, the overall loss of the D3PO algorithm gives:

w i |swi )

l i|sli)

i,σw,σl)[log ρ(β log πθ(a

πref(awi |swi ) − β log πθ(a

πref(ali|sli))],

Li(θ) = −E(s

(14) where i ∈ [0,T − 1]. Compared to Eq. (12), Eq. (14) uses every state-action pair for training, effectively increasing the data utilization of the segment by a factor of T.

The overview of our method is shown in Fig. 1. The pseudocode of D3PO can be seen in Appendix A.

### 5. Experiment

In our experiments, we evaluate the effectiveness of D3PO in fine-tuning diffusion models. Initially, we conduct tests on measurable objectives to verify if D3PO can increase these metrics, which quickly ascertain the algorithm’s effectiveness by checking for increases in the target measures.

Next, we apply D3PO to experiments aimed at lowering the rate of deformities in hands and full-body images generated by diffusion models. Moreover, we utilize our method to increase image safety and enhance the concordance between generated images and their corresponding prompts. These tasks pose considerable obstacles for competing algorithms, as they often lack automated capabilities for detecting which image is deformed or safe, thus relying heavily on human evaluation. We use Stable Diffusion v1.5 [47] to generate images in most of the experiments.

###### 5.1. Pre-defined Quantifiable Objectives Results

We initially conduct experiments with D3PO using quantitative objectives (alternatively referred to as optimal reward models). In these experiments, the relative values of the objectives (rewards) are used instead of human preference choices. Preferences are established based on these objectives, meaning A is preferred if its objective surpasses that of B. After training, we validate the effectiveness of our approach by measuring the growth of metrics.

In our experimental setup, we benchmark against several popular methods: DDPO [5], DPOK [21], and Reward Weighted [34], all of which necessitate a reward model. Note that our approach only used the relative sizes of the rewards corresponding to the objectives for preference choices, rather than the rewards themselves, whereas the other methods employed standard rewards during training. During testing, we used the rewards corresponding to the objectives as the evaluation criterion for all methods. This generally ensures a fair and unified comparison of finetuning with and without the reward model.

We first use the size of the images to measure the preference relationship between two pictures. For the compressibility experiment, an image with a smaller image size is regarded as better. Conversely, for the incompressibility experiment, we consider larger images to be those preferred by humans. As the training progresses, we can obtain the desired highly compressible and low compressible images. We then utilize the LAION aesthetics predictor [50] to pre-

dict the aesthetic rating of images. This model can discern the aesthetic quality of images, providing a justifiable reward for each one without requiring human feedback. The model can generate more aesthetic images after fine-tuning. We conducted a total of 400 epochs during the training process, generating 80 images samples in each epoch. The progression of the training samples is visually presented in Figure 2. More quantitative samples are shown in Figure 8. The testing curves of D3PO and other methods are shown in Figure 3. We are surprised to find that the D3PO method, which solely relies on relative sizes for preference choices, achieves results nearly on par with the methods trained using standard rewards, delivering comparable performance. This indicates that even in the presence of a reward model, D3PO can effectively fine-tune the diffusion model, continually increasing the reward to achieve the desired results.

###### 5.2. Experiments without Any Reward Model

We conduct experiments for some objectives without any reward model. We decide manually whether an image is deformed or if it is safe without a predefined reward model. During the training process, the model fine-tuned after each epoch serves as the reference model for the subsequent epoch. For each prompt, we generate 7 images with the same initial state xT ∼ N(0,I).

###### 5.2.1 Reduce Image Distortion

We use the prompt “1 hand” to generate images, manually selecting those that are deformed. Diffusion models often struggle to produce aesthetically pleasing hands, resulting in frequent deformities in the generated images. In this experiment, we focus on the normalcy rate of the images instead of the deformity rate, as depicted in Figure 4. We categorize 1,000 images for each epoch, over a total of five epochs, and track the normalcy rate of these hand images. After fine-tuning, the model shows a marked reduction in the deformity rate of hand images, with a corresponding increase in the production of normal images. Additionally, the fine-tuned model shows a higher probability of generating hands with the correct number of fingers than the pretrained model, as demonstrated in Figure 9.

To assess the generality of our method, we generated images with the Anything v5 model 1, renowned for creating anime character images. With Anything v5, there’s a risk of generating characters with disproportionate head-tobody ratios or other deformities, such as an incorrect number of limbs (as shown in Figure 6 left). We categorize such outputs as deformed. We assume that non-selected images are more favorable than the deformed ones, though we do not rank preferences within the deformed or non-deformed

1https://huggingface.co/stablediffusionapi/anything-v5

sets. The diminishing distortion rates across epochs are illustrated in Figure 4, showing a significant decrease initially that stabilizes in later epochs. The visual examples are provided in Figure 6.

[Figure 9]

Figure 4. Testing Curves depicting the Normalcy Rate of Hand Images and Deformity Rate of Anime Characters. As the training advances, there is a notable enhancement in the normalcy rate of hand generation, accompanied by a discernible decline in the deformity rate of anime characters.

###### 5.2.2 Enhance Image Safety

In this experiment, we utilized unsafe prompts to generate images using a diffusion model. These prompts contained edgy terms that could lead to the creation of both normal and Not Safe for Work (NSFW) images, examples being ‘ambiguous beauty’ and ‘provocative art’. The safety of the images was assessed via human annotations, and the diffusion model was fine-tuned based on these feedbacks. Across 10 epochs, we generated 1,000 images per epoch. Given the relatively minor variations in human judgments about image safety, we engaged just two individuals for the feedback task—one to annotate and another to doublecheck. The image safety rate during the training process is illustrated in Figure 5. After fine-tuning, the model consistently produced safe images, as evidenced in Figure 10.

[Figure 10]

Figure 5. Safety rate curves of the training procession.

###### 5.2.3 Prompt-Image Alignment

We employ human feedback to evaluate the alignment preference between two images generated from each prompt.

[Figure 11]

- Figure 6. Image samples of the pre-trained model and the fine-tuned model. The images on the left side of the arrows are distorted images (such as having 3 legs in the image) generated by the pre-trained model, while the images on the right side are normal images generated after fine-tuning the model. Both sets of images used the same initial Gaussian noise, prompts, and seeds.

[Figure 12]

- Figure 7. Comparative evaluation of 300 text prompts: Our study involved generating images from the pre-trained diffusion model, the fine-tuned models using other methods, and our finetuned model—using the same text prompts. For each prompt, human evaluators are tasked with determining which image is better aligned with the text. Each image was assessed by 5 human raters, and we report the percentage of images that received favorable evaluations. We also highlight the percentage with more than half vote in the black box.

The preference comparisons between images from the pretrained and fine-tuned models are conducted by an additional 5 evaluators, with the comparative preferences depicted in Figure 7. We also execute quantitative evaluations of models using metrics that measure the congruence between prompts and images, including CLIP [42], BLIP [35], and ImageReward [60], as presented in Table 1.

### 6. Conclusion

In this paper, we propose a direct preference denoising diffusion policy optimization method, named D3PO, to finetune diffusion models purely from human feedback without learning a separate reward model. D3PO views the denoising process as a multi-step MDP, making it possible to utilize the DPO-style optimization formula by formulating the action-value function Q with the reference model and the fine-tuned model. D3PO updates parameters at each step of denoising and consumes much fewer GPU memory overheads than directly applying the DPO algorithm. The empirical experiments illustrate that our method achieves competitive or even better performance compared with a diffusion model fine-tuned with a reward model that is trained with a large amount of images and human preferences in terms of image compressibility, image compressibility and aesthetic quality. We further show that D3PO can also benefit challenging tasks such as reducing image distortion rates, enhancing the safety of the generated images, and aligning prompts and images.

Table 1. Evaluations of prompt-image alignment. We compare D3PO with those that (a) do not fine-tune, (b) use the preferred images to fine-tune, and (c) use Reward Weighted method to finetune. D3PO achieve the best performance. D3PO demonstrated superior performance across all comparisons.

Methods CLIP score (↑) BLIP score (↑) ImageReward score (↑) Human preference (↑)

No fine-tune 30.7 1.95 0.04 14.7% Preferred images 31.0 1.97 0.08 11% Reward Weighted 31.5 2.01 0.17 18.7% D3PO 31.9 2.06 0.27 55.7%

For each epoch, we use 4,000 prompts, generate two images per prompt, and assess preferences with feedback from 16 different evaluators. The training spans 10 epochs.

### Acknowledgements

This work was supported by the STI 2030-Major Projects under Grant 2021ZD0201404.

### References

- [1] Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B Tenenbaum, Tommi S Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision making? In The Eleventh International Conference on Learning Representations, 2022. 2
- [2] Anthropic. Introducing claude, 2023. 3
- [3] Philip Bachman and Doina Precup. Data generation as sequential decision making. Advances in Neural Information Processing Systems, 28, 2015. 3
- [4] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. 3
- [5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 1, 3, 4, 6
- [6] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 4
- [7] Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365(6456):885–890, 2019. 6
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1
- [9] R´obert Busa-Fekete, Bal´azs Sz¨or´enyi, Paul Weng, Weiwei Cheng, and Eyke H¨ullermeier. Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine learning, 97: 327–351, 2014. 3
- [10] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, J´er´emy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023. 3
- [11] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. arXiv preprint arXiv:2303.04137, 2023. 2
- [12] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 1, 3, 4
- [13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [14] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:19822–19835, 2021. 1

- [15] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022. 1
- [16] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016. 1
- [17] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International Conference on Machine Learning, pages 8489–8510. PMLR, 2023. 3
- [18] Miroslav Dud´ık, Katja Hofmann, Robert E Schapire, Aleksandrs Slivkins, and Masrour Zoghi. Contextual dueling bandits. In Conference on Learning Theory, pages 563–587. PMLR, 2015. 3
- [19] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1
- [20] Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362,

2023. 3

- [21] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. 3, 6
- [22] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022. 1
- [23] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 1
- [24] Google. Bard, 2023. 3
- [25] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 3
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [27] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 4
- [29] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, pages 9902–9915. PMLR, 2022. 2

- [30] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 1
- [31] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 16

- [32] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023. 3
- [33] Kimin Lee, Laura Smith, and Pieter Abbeel. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. arXiv preprint arXiv:2106.05091, 2021. 4
- [34] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 3, 6
- [35] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–

12900. PMLR, 2022. 8

- [36] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10234– 10243, 2023. 2
- [37] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. 4
- [38] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 3
- [39] Runze Liu, Yali Du, Fengshuo Bai, Jiafei Lyu, and Xiu Li. Zero-shot preference learning for offline rl via optimal transport. arXiv preprint arXiv:2306.03615, 2023. 3
- [40] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1
- [41] OpenAI. Gpt-4 technical report, 2023. 1, 3
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 8
- [43] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct

- preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023. 1, 3
- [44] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 1, 2
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1, 3

- [46] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR, 2015. 1
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 1, 6
- [48] Aadirupa Saha, Aldo Pacchiano, and Jonathan Lee. Dueling rl: Reinforcement learning with trajectory preferences. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pages 6263–6289. PMLR,

2023. 3

- [49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 1, 2
- [50] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 6
- [51] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016. 6
- [52] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676): 354–359, 2017. 6
- [53] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [54] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2

- [55] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020. 3
- [56] Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning. 1998. 3
- [57] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [58] A¨aron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747–1756. PMLR, 2016. 1
- [59] Aaron Wilson, Alan Fern, and Prasad Tadepalli. A bayesian approach for policy learning from trajectory preference queries. Advances in neural information processing systems, 25, 2012. 4
- [60] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. arXiv preprint arXiv:2304.05977,

2023. 1, 3, 8, 14

- [61] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012. JCSS Special Issue: Cloud Computing 2011. 3
- [62] Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Jianquan Li, Guiming Chen, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, et al. Huatuogpt, towards taming language model to be a doctor. arXiv preprint arXiv:2305.15075, 2023. 3
- [63] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3
- [64] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 3

### A. D3PO Pseudo-code

The pseudocode of the D3PO method can be seen in Algorithm 1.

Algorithm 1 D3PO pseudo-code Require: Number of inference timesteps T, number of training epochs N, number of prompts per epoch K, pre-trained

diffusion model ϵθ.

- 1: Copy a pre-trained diffusion model ϵref = ϵθ. Set ϵref with requires grad to False.

- 2: for n = 1 : N do
- 3: # Sample images
- 4: for k = 1 : K do
- 5: Random choose a prompt ck and sample xT ∼ N(0,I)
- 6: for i = 0 : 1 do
- 7: for t = T : 1 do
- 8: no grad: xik,t−1 = µ(xik,t,t,ck) + σtz, z ∼ N(0,I)
- 9: end for
- 10: end for
- 11: end for
- 12: # Get Human Feedback
- 13: for k = 1 : K do
- 14: Get human feedback from ck, x0k,0, and x1k,0.
- 15: if x00 is better than x10 then
- 16: hk = [1,−1]
- 17: else if x01 is better than x00 then
- 18: hk = [−1,1]
- 19: else
- 20: hk = [0,0]
- 21: end if
- 22: end for
- 23: # Training
- 24: for k = 1 : K do
- 25: for t = T : 1 do
- 26: for i = 0 : 1 do
- 27: with grad:
- 28: µθ(xik,t,t,ck) = √1α

t

xik,t − β

√1−tα¯tϵθ xik,t,t,ck

- 29: µref(xik,t,t,ck) = √1α

t

xik,t − β

√1−tα¯tϵref xik,t,t,ck

- 30: πθ(xik,t−1|xik,t,t,ck) = √21πσ

t

exp(−(x

i k,t−1−µθ(xik,t,t,ck))2

2σt2 )

- 31: πref(xik,t−1|xik,t,t,ck) = √21πσ

t

exp(−(x

i k,t−1−µref(xik,t,t,ck))2

2σt2 )

- 32: end for
- 33: Update θ with gradient descent using

∇θ log ρ(hk(0)β log

πθ(x0k,t−1|x0k,t,t,c) πref(xk,t−1|x0k,t,t,c)

+ hk(1)β log

πθ(x1k,t−1|x1k,t,t,c) πref(xk,t−1|x1k,t,t,c)

)

- 34: end for
- 35: end for
- 36: end for

### B. Proof

- B.1. Proof of Proposition 1 The RL objective can be written as:

max

π

Es∼dπ,a∼π(a|s)[Q∗(s,a)] − βDKL[π(a|s)∥πref(a|s)]

= max

π

Es∼dπ,a∼π(a|s)[Q∗(s,a) − β log

π(a|s) πref(a|s)

]

= min

π

Es∼dπ,a∼π(a|s)[log

π(a|s) πref(a|s) −

1 β

Q∗(s,a)]

= min

π

Es∼dπ,a∼π(a|s)[log

π(a|s) πref(a|s)exp(

1 β

Q∗(s,a))

]

= min

π

Es∼dπ[DKL[π(a|s)∥π˜(a|s)]]

where π˜(a|s) = πref(a|s)exp(

1 β

Q∗(s,a)). Note that the KL-divergence is minimized at 0 iff the two distributions are identical, so the optimal solution is:

π(a|s) = π˜(a|s) = πref(a|s)exp(

1 β

Q∗(s,a)).

- B.2. Proof of Proposition 2 For simplicity, we define Qi = Q∗(si0,ai0) and Xi = Tt=0 r∗ sit,ait i ∈ {0,1}. Using the Eq. (3) we can obtain that:

E[exp(X1)] E[exp(X1) + exp(X0)]

E[p∗ (σ1 ≻ σ0)] =

exp(Q1 + 1/2σ) exp(Q1 + 1/2σ) + exp(Q0 + 1/2σ)

=

exp(Q1) exp(Q1) + exp(Q0)

=

= E[˜p∗ (σ1 ≻ σ0)].

E[exp(2X1)] E[exp(2X1)] + E[exp(2X0)] + E[2exp(X0)exp(X1)]

E[(p∗ (σ1 ≻ σ0))2] =

exp(2Q1 + 2σ2) exp(2Q1 + 2σ2) + exp(2Q0 + 2σ2) + exp(Q0 + Q1 + σ2)

=

exp(2Q1 + σ2) exp(2Q1 + σ2) + exp(2Q0 + σ2) + 2exp(Q0 + Q1)

=

.

Var[p∗(σ1 ≻ σ0)] = E[(p(σ1 ≻ σ0))2] − (E[p(σ1 ≻ σ0)])2

2exp(3Q1 + Q0)(exp(σ2) − 1) [exp(2Q1 + σ2) + exp(2Q0 + σ2) + 2exp(Q0 + Q1)][exp(Q1) + exp(Q0)]2 ≤

=

2exp(3Q1 + Q0)(exp(σ2) − 1) [exp(Q1) + exp(Q0)]4

.

Similarly, we have:

2exp(Q1 + 3Q0)(exp(σ2) − 1) [exp(Q1) + exp(Q0)]4

Var[p∗ (σ0 ≻ σ1)] ≤

.

Note that Var[p∗ (σ1 ≻ σ0)] = Var[1 − p∗ (σ0 ≻ σ1)] = Var[p∗ (σ0 ≻ σ1)], considering these two inequalities, we have:

[exp(Q1 + 3Q0) + exp(Q0 + 3Q1)](exp(σ2) − 1) [exp(Q1) + exp(Q0)]4 ≤

Var[p∗ (σ1 ≻ σ0)] ≤

[exp(Q1 + 3Q0) + exp(Q0 + 3Q1)](exp(σ2) − 1) 16[exp(2Q1)exp(2Q0)]

[exp(Q0 − Q1) + exp(Q1 − Q0)](exp(σ2) − 1) 16

=

1 ξ

)(exp(σ2) − 1) 16

(ξ +

≤

. By using the Chebyshev inequality, we can obtain:

(ξ2 + 1)(exp(σ2) − 1) 16ξt

P(|p∗ (σ1 ≻ σ0) − p˜∗ (σ1 ≻ σ0)| < t) > 1 −

.

(ξ2 + 1)(exp(σ2) − 1) 16ξδ

We choose t =

so that:

(ξ2 + 1)(exp(σ2) − 1) 16ξδ

P(|p∗ (σ1 ≻ σ0) − p˜∗ (σ1 ≻ σ0)| <

) > 1 − δ.

### C. Prompts of Experiments

During the quantitative experiments in Section 5.1, we utilized prompts related to 45 common animals, outlined as follows: We simply use “1 hand” to generate hand images in the image distortion experiments. In the experiment of reducing distor-

cat dog horse monkey rabbit zebra spider bird sheep deer cow goat lion tiger bear raccoon fox wolf lizard beetle kangaroo fish butterfly shark whale dolphin squirrel mouse rat snake turtle frog chicken duck goose bee pig turkey fly llama camel gorilla bat ant hedgehog

tions in generated anime images using the Anything V5 model, we initially collected 483 illustrations from anime games. Subsequently, non-character roles within these illustrations were filtered, resulting in the retention of 442 images. We then employed autotagging from Stable-diffusion-webui 2 to perform reverse inference for each image using prompts, generating 442 prompts in total. Here is an example of our prompts: “1girl, bangs, bare shoulders, blue eyes, blue hair, blunt bangs, blush, boots, braid, breasts, bridal gauntlets, bug, butterfly, butterfly hair ornament, butterfly wings, center opening, cleavage, collarbone, dark-skinned female, dark skin, detached sleeves, dress, elbow gloves, full body, gloves, hair ornament, halterneck, high heel boots, high heels, holding, holding staff, jewelry, long hair, looking at viewer.”

In the experiment aimed at enhancing image safety in Section 5.2.2, we employed somewhat ambiguous and potentially misleading terms to prompt the diffusion model for image generation. The prompts we used are as follows:

For the prompt-image alignment experiments mentioned in Section 5.2.3, we employed 10,000 prompts extracted from [60]. These prompts cover diverse categories including arts, people, outdoor scenes, animals, and more.

2https://github.com/AUTOMATIC1111/stable-diffusion-webui

provocative art forbidden pleasures intimate moments gentle intimacy sexy pose ambiguous beauty seductive allure sensual elegance artistic body provocative aesthetics whispers of desire artful sensuality seductive grace ostentatious temptation

### D. More Samples

In this section, we give more samples from our models. Figure 8 shows the samples after using the objective of compressibility, and aesthetic quality. Figure 10 shows the image samples with unsafe prompts following training on enhancing image safety tasks. Figure 11 shows the image samples of the pre-trained diffusion model and our fine-tuned model after training with the prompt-image alignment objective. The images generated by D3PO fine-tuned, preferred image fine-tuned, reward weighted fine-tuned are also depicted in Figure 14.

[Figure 13]

- Figure 8. Image samples of pre-trained models, fine-tuned models for compressibility objectives, incompressibility objectives, and aesthetic quality objectives using the same prompts. It can be observed that the images generated after fine-tuning more closely align with the specified objectives.

#### E. Implementation Details and Experimental Settings Our experiments are performed by using the following hardware and software:

- • GPUs: 32G Tesla V100 × 4
- • Python 3.10.12
- • Numpy 1.25.2
- • Diffusers 0.17.1
- • Accelerate 0.22.0
- • Huggingface-hub 0.16.4
- • Pytorch 2.0.1
- • Torchmetrics 1.0.2 In our experiments, we employ the LoRA technique to fine-tune the UNet weights, preserving the frozen state of the text

encoder and autoencoder weights, which substantially mitigates memory consumption. Our application of LoRA focuses solely on updating the parameters within the linear layers of keys, queries, and values present in the attention blocks of the UNet. For detailed hyperparameters utilized in Section 5.1, please refer to Figure 2.

In the experiments of Section 5.2.1 and Section 5.2.2, we generate 7 images per prompt and choose the distorted images

Table 2. Hyperparameters of D3PO method

Name Description Value lr learning rate of D3PO method 3e-5 optimizer type of optimizer Adam [31] ξ weight decay of optimizer 1e-4 ϵ Gradient clip norm 1.0

- β1 β1 of Adam 0.9

- β2 β2 of Adam 0.999 T total timesteps of inference 20 β temperature 0.1 bs batch size per GPU 10 n number of batch samples per epoch 2 η eta parameter for the DDIM sampler 1.0 G gradient accumulation steps 1 w classifier-free guidance weight 5.0 N epochs for fine-tuning with reward model 400 mp mixed precision fp16

(unsafe images) by using an open-source website 3, which can be seen in Figure 15. We set different tags for different tasks. In the experiment of prompt-image alignment, we generate 2 images per prompt instead of 7 images and choose the better one by using the same website.

To calculate the CLIP score in the section 5.2.3, we use the ‘clip score’ function of torchmetrics. We calculate the Blip score by using the ‘model base.pth’ model 4. The ImageReward model we use to assess the quality of prompt-image matching is available at the website 5.

- 3https://github.com/zanllp/sd-webui-infinite-image-browsing
- 4https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model base.pth

- 5https://github.com/THUDM/ImageReward

[Figure 14]

- (a) Samples from pre-trained model

[Figure 15]

- (b) Samples from fine-tuned model

- Figure 9. Image samples from the hand distortion experiments comparing the pre-trained model with the fine-tuned model. The pre-trained model predominantly generates hands with fewer fingers and peculiar hand shapes. After fine-tuning, although the generated hands still exhibit some deformities, they mostly depict a normal open-fingered position, resulting in an increased occurrence of five-fingered hands.

[Figure 16]

###### Figure 10. Image samples generated from the fine-tuned model with unsafe prompts. All generated images are safe, and no explicit content images are produced.

[Figure 17]

- (a) prompt:a robot with long neon braids, body made from porcelain and brass, neon colors, 1 9 5 0 sci - fi, studio lighting, calm, ambient occlusion, octane render

[Figure 18]

- (b) prompt:highly detailed anime girl striking a dramatic pose at night with bright lights behind, hands on shoulders. upper body shot, beautiful face and eyes.

[Figure 19]

(c) prompt:medieval temple in fantasy jungle, pond, statue, sculpture

- Figure 11. Image samples of the fine-tuned model after using human feedback to align prompt and image. After fine-tuning, the images better match the description in the prompt, and the generated images become more aesthetically pleasing.

[Figure 20]

(a) prompt:alien in banana suit

[Figure 21]

(b) prompt:a very cool cat

[Figure 22]

- (c) prompt:futuristic technologically advanced solarpunk planet, highly detailed, temples on the clouds, one massive perfect sphere, bright sun magic hour, digital painting, hard edges, concept art, sharp focus, illustration, 8 k highly detailed, ray traced

[Figure 23]

###### (a) prompt:portrait photo of a giant huge golden and blue metal humanoid steampunk robot with a huge camera, gears and tubes, eyes are glowing red lightbulbs, shiny crisp finish, 3 d render, insaneley detailed, fluorescent colors

[Figure 24]

###### (b) prompt:fighter ornate feminine cyborg in full body skin space suit, arab belt helmet, concept art, gun, intricate, highlydetailed, space background, 4 k raytracing, shadows, highlights, illumination

[Figure 25]

###### (c) prompt:a masked laboratory technician man with cybernetic enhancements seen from a distance, 1 / 4 headshot, cinematic lighting, dystopian scifi outfit, picture, mechanical, cyboprofilerg, half robot

[Figure 26]

- Figure 14. Image samples from the pretrained model and the fine-tuned models. Prompts: (a) highly detailed vfx portrait of a oriental mage, stephen bliss, unreal engine, greg rutkowski, loish, rhads, beeple, makoto shinkai and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, alphonse mucha, global illumination, detailed and intricate environment. (b) pixar animation of an anthropomorphic genz cat. (c) a detailed sculpture of god crushing satan with his hand, demonic, demon, viking, by greg rutkowski and justin gerard, digital art, monstrous, art nouveau, baroque style, realistic painting, very detailed, fantasy, dnd, character design, top down lighting, trending on artstation. (d) style artstation, style greg rutkowsk, ciberpunk, comic art book, biopunk, octane render, unreal engine 6, epic game graphics. (e) a futuristic visiom of artificial intelligence, unreal engine, fantasy art by greg rutkowski, loish, rhads, ferdinand knab, makoto shinkaib and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, global illumination, radiant light, detailed and intricate environment by fromsoftware, spiritual, colorful, fantasy landscape.

[Figure 27]

###### Figure 15. The website we use. We can tag each image according to different tasks, such as using the ‘deformed’ tag to denote an image is deformed and the ’unsafe’ tag to record an image is unsafe.

