## Self-Taught Self-Correction for Small Language Models

### Viktor Moskvoretskii1,2, Chris Biemann3, Irina Nikishina3 1Skoltech, 2HSE University, 3University of Hamburg

Correspondence: vvmoskvoretskii@gmail.com

# arXiv:2503.08681v1[cs.CL]11Mar2025

### Abstract

Although large language models (LLMs) have achieved remarkable performance across various tasks, they remain prone to errors. A key challenge is enabling them to self-correct. While prior research has relied on external tools or large proprietary models, this work explores self-correction in small language models (SLMs) through iterative fine-tuning using solely self-generated data. We introduce the Self-Taught Self-Correction (STaSC) algorithm, which incorporates multiple algorithmic design choices. Experimental results on a question-answering task demonstrate that STaSC effectively learns self-correction, leading to significant performance improvements. Our analysis further provides insights into the mechanisms of self-correction and the impact of different design choices on learning dynamics and overall performance. To support future research, we release our user-friendly codebase and lightweight models.

### 1 Introduction

Recent advanced LLM employ complex reasoning (Guo et al., 2025) and meta-reasoning (Xiang et al., 2025), expanding their capabilities. However, even the most advanced models are prone to errors, including hallucinations (Huang et al., 2025) and logical inconsistencies (Ghosh et al., 2024), requiring external verification or human intervention. To address those problems, self-correction — the ability to revise their own outputs — has been evolved (Madaan et al., 2023). The existing approaches mostly use zero-shot prompting (Madaan et al., 2023; Shinn et al., 2024), external evaluators for correction or feedback (Zhang et al., 2024) or apply large proprietary models and focus specifically on mathematical tasks (Kumar et al., 2024).

In this study, we focus on self-correction without external information or evaluators, ensuring inference efficiency while relying solely on the

model’s intrinsic knowledge. We investigate the self-correction capabilities of small language models (SLMs) by applying iterative fine-tuning on self-generated data, allowing models to improve their outputs without external supervision.

We introduce the Self-Taught Self-Correction (STaSC) algorithm, which trains models to selfcorrect using self-generated trajectories, adapting the core idea of STaR (Zelikman et al., 2022). STaSC extends and unifies approaches that iteratively train with self-generated trajectories, including the Self-Correction (SC) algorithm (Welleck et al., 2022), by incorporating flexible design choices. Unlike prior methods, STaSC provides control over initial answer exploration, correction filtering, and iterative fine-tuning, encompassing SC as a special case and demonstrating how different algorithmic choices impact self-correction performance.

Our results on the Natural Questions dataset (Kwiatkowski et al., 2019) show that SLMs can learn to self-correct using self-synthesized data, while also improving their initial answer quality despite being trained solely for corrections. We release easy-to-use and adaptable code for selfcorrection algorithms at https://github.com/ VityaVitalich/STASC.

The contributions of the paper are as follows:

- • We propose the Self-Taught Self-Correction (STaSC) algorithm, unifying and extending existing self-correction methods trained with self-generated trajectories.
- • We conduct extensive experiments on a purely Natural Language Processing (NLP) task Question Answering — using an open-source SLMs, demonstrating their ability to learn selfcorrection with self-synthesized data.
- • We release open-source, easily adaptable code for self-correction, along with efficient SLMs

STaR Self-Correction

STaSC

|INPUT|
|---|

|INPUT|
|---|

|INPUT|
|---|

OUTPUT

OUTPUT

❄ 🔥 / 🔥

❄

🔥

🔥

Generator Generator Corrector Generator Corrector

|OUTPUT|
|---|

OUTPUT

OUTPUT

OUTPUT OUTPUT

- Figure 1: Illustration of the self-improvement method STaR (left) (Zelikman et al., 2022), self-correction method SC (center) (Welleck et al., 2022), and our method, STaSC (right). STaCS offers flexible control over initial answer exploration, correction filtering, and iterative fine-tuning. It is inspired by STaR and effectively encompasses SC as a special case. SC and STaSC allow several initial answers and corrections. The dotted line in the STaSC denotes two possible setups: fine-tuning the model and generating from it at the next iteration (Evolving Fine-Tuning) and keeping the Generator frozen and fine-tuning the Corrector model only (Fixed Fine-Tuning).

with fewer than 4B parameters, making selfcorrection practical and accessible.

### 2 Self-Taught Self-Correction

In this section, we introduce the Self-Taught Self-Correction (STaSC) algorithm, an adaptation of STaR (Zelikman et al., 2022) for selfcorrection through iterative fine-tuning on selfsynthesized data. STaSC unifies and extends various self-correction approaches using selfsynthesized data, including the Self-Correction (SC) algorithm (Welleck et al., 2022).

#### 2.1 Foundations and Enhancements

- Figure 1 presents three algorithms highlighting their similarities and dissimilarities. The left part describes the original STaR algorithm which focuses on generating reasoning paths, filtering correct ones, and fine-tuning them accordingly.

The central part of the Figure 1 demonstrates the Self-Correction (SC) algorithm which fine-tunes the corrector model on corrections leading to improvement, keeping the initial Generator model fixed. The right part refers to STaSC, adapting the idea of STaR of iteratively refining model outputs by sampling an initial answer, generating corrections, filtering correct trajectories, and fine-tuning on the successful revisions. This process is repeated over multiple iterations, progressively enhancing the model’s accuracy.

The key extension of our algorithm over previ-

ous baselines is the incorporation of flexible algorithmic choices. Initial answers can be drawn from either a frozen model or the previous iteration, corrections can be filtered based on strict improvements or by allowing non-worsening revisions, and fine-tuning can be performed from either the initial model or the latest iteration.

2.2 Formal Definition and Algorithmic Choices

Below, we outline the Algorithm 1 steps and key design choices, each of which influences the selfcorrection process. For each choice, we define an abbreviation that will be used to denote specific algorithm configurations, such as STaSCFIF, where subscripts specify the selected options.

#### 2.2.1 Requirements and Notation

The algorithm begins with an initial language model state, M0, and an initial dataset, D0, consisting of input-output pairs (x,y). Additionally, we define a number of improvement iterations N, the number of sampled initial generations Ninit, the number of sampled corrections Ncorr, and a reward function r, which evaluates the quality of modelgenerated outputs.

Step 1: Sampling Initial Answers In the first step, we sample Ninit initial answers yˆ1 for each input x in the dataset D0. The primary design choice here is the selection of the model M used for sampling:

Algorithm 1 Self-Taught Self-Correction (STaSC) Require: Initial model M0, dataset D0, number of iterations N, initial samples Ninit, correction samples

Ncorr, reward function r

- 1: for n = 1 to N do
- 2: Step 1: Sample Initial Answers
- 3: Yˆi1 = {yˆij1 }Nj=1init ∼ M(xi), ∀xi ∈ D0
- 4: Option 1: M = Mn−1 (Evolving Initialization)
- 5: Option 2: M = M0 (Fixed Initialization)
- 6: Step 2: Sample Corrections
- 7: Yˆi2 = {yˆijk2 }Nk=1corr ∼ Mn−1(xi,yˆij1 ), ∀yˆij1 ∈ Yˆi1
- 8: Step 3: Filter Corrections
- 9: Dn+ = {(xi,yˆij1 ,yˆijk2 ) | r(ˆyijk2 ) > r(ˆyij1 )}
- 10: Dn= = {(xi,yˆij1 ,yˆijk2 ) | r(ˆyijk2 ) = r(ˆyij1 ) ≥ t}
- 11: Option 1: Dn = Dn+ (Improving Filter)
- 12: Option 2: Dn = Dn+ ∪ Dn= (Non-Decreasing Filter)
- 13: Step 4: Fine-Tuning
- 14: Mn = train(M,{yˆijk2 | (xi,yˆij1 ,yˆijk2 ) ∈ Dn})
- 15: Option 1: M = M0 (Fixed Fine-Tuning)
- 16: Option 2: M = Mn−1 (Evolvong Fine-Tuning)
- 17: end for

- • Fixed Initialization (STaSCF**) – Initial answers are sampled using the frozen model M0, as in the SC algorithm (Welleck et al., 2022).
- • Evolving Initialization (STaSCE**) – Initial answers are sampled using the model from

the previous iteration Mn−1, as in the original STaR (Zelikman et al., 2022).

Fixed Initialization ensures robustness to variations in self-improvement and dataset shifts, maintaining consistency across iterations. In contrast, Evolving Initialization allows for greater exploration, potentially leading to more diverse refinements and improved performance.

- Step 2: Sample Corrections At the second step, we sample Ncorr corrections yˆ2 for each output yˆ1 in dataset D0 using the model from the last iteration Mn−1.
- Step 3: Filtering Corrections In this step, we filter the corrections using the reward function r(ˆy2)

to construct the fine-tuning dataset Dn. The key design choice here is the filtering criterion for selecting corrections.

• Improving Filter (STaSC*I*) – Selects only corrections that strictly improve the reward, ensuring r(ˆy2) > r(ˆy1), as used in STaSC and SC.

• Non-Decreasing Filter (STaSC*N*) – Selects corrections that either strictly improve the reward or retain the initial answer if it was already correct, ensuring (r(ˆy2) = r(ˆy1)) ∩ (r(ˆy1) ≥ t), as proposed in SCoRE (Kumar et al., 2024). This allows the model to preserve already correct answers while still refining incorrect ones.

The Improving Filter enforces strict improvement for every input, ensuring that only progressively better outputs are used for fine-tuning. In contrast, the Non-Decreasing Filter permits stable answers to be retained when already correct, allowing for a more conservative refinement process.

Step 4: Fine-Tuning In this step, we fine-tune the model on the dataset Dn obtained from Step 3 to produce the improved model Mn. The key design choice here is the selection of the model used for fine-tuning:

- • Fixed Fine-Tuning (STaSC**F) – Fine-tunes the initial model M0, as done in the original STaR recipe, ensuring stability across iterations and reducing the risk of accumulating errors.
- • Evolving Fine-Tuning (STaSC**E) – Finetunes the model from the previous iteration

Mn−1, as in SC, allowing the model to progressively improve and adapt across iterations.

Qwen-2.5-1.5B Phi3-mini

Ninit Ncorr

max{r(Yˆ1)} max{r(Yˆ2)} max{r(Yˆ1)} max{r(Yˆ2)}

1 - - 0.320 ± 0.005 0.372 ± 0.010 3 0.248 ± 0.011 0.208 ± 0.011 0.326 ± 0.009 0.376 ± 0.010 5 0.230 ± 0.011 0.228 ± 0.021 0.352 ± 0.016 0.384 ± 0.012

1

1 0.236 ± 0.007 0.232 ± 0.018 0.334 ± 0.011 0.362 ± 0.024 3 0.264 ± 0.015 0.238 ± 0.018 0.342 ± 0.010 0.372 ± 0.012 5 0.273 ± 0.017 0.236 ± 0.019 0.332 ± 0.007 0.384 ± 0.013

3

1 0.273 ± 0.012 0.250 ± 0.024 0.334 ± 0.008 0.378 ± 0.016 3 0.295 ± 0.019 0.244 ± 0.023 0.336 ± 0.021 0.354 ± 0.026 5 0.300 ± 0.020 0.248 ± 0.029 0.350 ± 0.012 0.376 ± 0.011

5

- Table 1: Maximum reward r over iterations for initial answer r(Yˆ1) and for correction r(Yˆ2) for different number of samples and initial generations. Bold corresponds to the best performance.

Fixed Fine-Tuning maintains a stable learning process by always training from the original model, preventing drift. In contrast, Evolving Fine-Tuning enables iterative adaptation, potentially leading to greater long-term improvements but also introducing the risk of compounding errors.

### 3 Experimental Setup

In this section, we describe the main experimental procedure, including dataset selection, evaluation metrics, and implementation details.

Dataset We evaluate our algorithm on the QA task using the Natural Questions dataset (Kwiatkowski et al., 2019), which consists of factual simple questions. We use a subset of 500 questions per train and test split, following previous studies (Trivedi et al., 2022; Jeong et al., 2024; Moskvoretskii et al., 2025), to ensure consistency and computational efficiency.

Evaluation We use In-accuracy as the primary evaluation metric and reward function, which is standard for this task (Trivedi et al., 2022; Jeong et al., 2024). It assesses whether the generated answer contains the reference answer, assigning r(ˆy) = 1 for correct responses and r(ˆy) = 0 for incorrect ones.

All metrics are reported on the test set, which remains unseen during training. Additionally, we report max{r(Yˆ1)} and max{r(Yˆ2)}, representing the highest reward obtained for initial answers and corrections across STaSC iterations, respectively. To ensure a fair real-world setup, the reward is not available to the model during inference. Unlike

some prior studies (Shinn et al., 2024), where inference benefits from reward signals derived from ground-truth labels, our approach aligns with a fully unsupervised inference setting, ensuring a more realistic evaluation.

Implementation Details We conduct experiments using Qwen-2.5-1.5B (Qwen et al., 2025) and Phi3-Mini (Abdin et al., 2024), employing default generation parameters. The default setup for self-correction is 2-shot. STaSC fine-tuning is performed for 1 epoch with a batch size of 8 and a learning rate of 7 × 10−6. Additional implementation details are provided in Appendix A.

### 4 Results & Discussion

In this section, we provide the results and discuss them, inspecting the STaSC algorithm design.

#### 4.1 Impact of Ninit and Ncorr

Firstly, we examine how the selection of parameters Ninit and Ncorr affects algorithm performance, exploring values of 1, 3, and 5 for both models. To encourage exploration, we use STaSCEIF, where initial answers are sampled from the previous iteration, only improving corrections are retained, and fine-tuning is performed from the base model to ensure stability.

As shown in Table 1, a greedy approach for Qwen-2.5-1.5B fails to ensure convergence, as no improving corrections emerge in the first iteration. In contrast, increased exploration significantly enhances performance, likely due to the weaker alignment of the initial model. Additionally, we observe

###### Phi3-mini

###### Qwen-2.5-1.5B

0.40

0.22

CorrectionIn-Accuracy

0.35

0.30

0.20

0.25

0.18

0.20

0.16

0.15

0.14

0.10

0 1 2 3 4 5 6 7 8 9 10

0 1 2 3 4 5 6 7 8 9 10

Iteration

Iteration

STaSCEIF STaSCEIE STaSCENF STaSCENE

- Figure 2: Correction In-accuracy for STaSC versions with Evolving Initialization for Phi3-mini and Qwen-2.5-1.5B.

0 1 2 3 4 5 6 7 8 9 10

Iteration

0.30

0.31

0.32

0.33

0.34

0.35

0.36

0.37

CorrectionIn-Accuracy

Phi3-mini

0 1 2 3 4 5 6 7 8 9 10

Iteration

0.16

0.17

0.18

0.19

0.20

0.21

Qwen-2.5-1.5B

STaSCFIE STaSCFIF STaSCFNE STaSCFNF

- Figure 3: Correction In-accuracy for STaSC versions with Fixed Initialization for Phi3-mini and Qwen-2.5-1.5B.

that exploring initial answers has a greater impact than exploring corrections.

However, this trend does not hold for Phi3-Mini, where increasing the number of initial answers does not improve results, but greater correction exploration does. This discrepancy likely stems from differences in model capabilities. Phi3-Mini, being inherently stronger, benefits more from refining corrections, whereas Qwen-2.5-1.5B, with lower initial competence, requires a broader search for initial answers to gain useful knowledge early on.

For subsequent experiments, we adopt the most stable and well-performing configurations:

- • Qwen-2.5-1.5B: (Ninit,Ncorr) = (5,5).
- • Phi3-Mini: (Ninit,Ncorr) = (1,5).

#### 4.2 STaSC With Evolving Initialization

- Figure 2 illustrates the dynamics of correction performance for STaSC variants with Evolving Initialization. Below, we discuss the observed effects on performance, highlighting key trends and their implications.

Effect of the Non-Decreasing Filter A key observation is that STaSC with the Non-Decreasing Filter consistently degrades performance for Phi3Mini. This is likely due to the difficulty in stabilizing training when fewer corrections are retained, increasing the risk of overfitting. Interestingly, Qwen does not exhibit the same decline, possibly due to a higher number of retained corrections, which augments the data and stabilizes training.

Impact of Filtering Selectivity We further highlight the need to properly filter corrections when using Evolving Fine-Tuning, as all Phi3 settings with STaSC**E exhibit a negative correlation between the leniency of filtered trajectories and the gain in correction performance (r = −0.51,p < .001). The more corrections were provided to the Corrector model, the worse it performed. This suggests that insufficiently selective filtering introduces noise, leading to overfitting and a decline in performance improvement. Notably, no other setting shows a significant correlation between the number of filtered corrections and performance.

Evolving Fine-Tuning Trends Evolving FineTuning slightly improves correction performance

###### Phi3-mini

###### Qwen-2.5-1.5B

0.24

CorrectionIn-Accuracy

0.38

0.22

0.36

0.20

0.34

0.18

0.32

0.16

0.30

0.14

0 1 2 3 4 5 6 7 8 9 10

0 1 2 3 4 5 6 7 8 9 10

Iteration

Iteration

Stage Type

Algorithm Type

Correction Initial

STaSCEI STaSCEIE

STaSCEIF STaSCEIE

Figure 4: Correction and Initial Answer In-accuracy for best STaSC versions for Phi3-mini and Qwen-2.5-1.5B.

Step 1 Step 3 Step 4 Qwen-2.5-1.5B Phi3-mini Model Filter Model max{r(Yˆ1)} max{r(Yˆ2)} max{r(Yˆ1)} max{r(Yˆ2)}

M0 0.212 0.208 ± 0.014 0.294 0.354 ± 0.010 Mn−1 0.212 0.212 ± 0.016 0.294 0.374 ± 0.009

Improving

M0

M0 0.212 0.198 ± 0.012 0.294 0.348 ± 0.013 Mn−1 0.212 0.206 ± 0.014 0.294 0.374 ± 0.010

Non-Decreasing

M0 0.244 ± 0.011 0.232 ± 0.023 0.352 ± 0.016 0.384 ± 0.12 Mn−1 0.236 ± 0.009 0.230 ± 0.024 0.392 ± 0.024 0.394 ± 0.012 Non-Decreasing

Improving

Mn−1

M0 0.240 ± 0.009 0.222 ± 0.023 0.316 ± 0.056 0.356 ± 0.066 Mn−1 0.234 ± 0.013 0.228 ± 0.022 0.294 ± 0.062 0.356 ± 0.074

- Table 2: Maximum reward r over iterations for initial answer r(Yˆ1) and for correction r(Yˆ2) for different settings of STaSC Algorithm. Bold values correspond to the best performance, underlined represent second best.

for Phi3-Mini. For Qwen, performance increases rapidly at first, then declines slightly before rising again in later iterations. This suggests that accumulated knowledge gains take effect only in the later stages, once a sufficient number of corrections and initial answers have been processed.

Key Takeaways Evolving Initialization is most effective when combined with an Improving Filter and Evolving Fine-Tuning, as these either enhance correction performance or at least prevent degradation. Additionally, filtering selectivity is crucial—overly lenient filters introduce noise and cause overfitting.

#### 4.3 STaSC With Fixed Initialization

Next, we analyze the trends and implications of Fixed Initialization, which aligns with the algorithmic choice used in SC. Figure 3 presents the correction performance for STaSC variants under this setting.

Effect of the Non-Decreasing Filter Unlike Evolving Initialization, the Non-Decreasing Fil-

ter does not lead to a general decline in correction performance for either Phi3 or Qwen, with an exception of STaSCFNF,where performance degrades. For Phi3, correction performance remains largely unchanged throughout most iterations, with noticeable improvements emerging only in the final stages. This delayed progress suggests that Evolving Fine-Tuning gradually accumulates knowledge, but its effects become apparent only after sufficient corrections and initial answers have been processed. A similar trend was previously observed in STaSCEIE, showing that iterative fine-tuning plays a key role in long-term performance gains.

Importance of Evolving Fine-Tuning We find that Evolving Fine-Tuning is crucial when using Fixed Initialization, particularly for Phi3 and, to a lesser extent, for Qwen. This is expected, as Evolving Fine-Tuning serves as the only source of exploration in this setting, driving the algorithm forward. In contrast, with Fixed Fine-Tuning, we observe a general decline in Phi3’s performance and stagnation after the first iteration for Qwen. This suggests that without sufficient exploration,

relying solely on corrections from previous steps is insufficient for SLMs.

We also observe that when using both Fixed Initialization and Fixed Fine-Tuning applying the Non-Decreasing Filter further worsens the performance, underscoring the importance of filtering selectivity. In this setting, exploration is driven solely by corrections, making the filtering process critical. When corrections include too many noisy or uninformative trajectories due to improper filtering, the model struggles to improve, leading to significant performance degradation. This reinforces the need for a more controlled correction selection process to ensure meaningful updates during training.

Key Takeaways Fixed Initialization reduces exploration, making the model more tolerant to the Non-Decreasing Filter and benefiting more from Evolving Fine-Tuning. In contrast, Fixed FineTuning restricts exploration solely to corrections, increasing reliance on selective filtering to maintain performance.

#### 4.4 STaSC Impact on Initial Answers

In this section, we identify the best-performing STaSC configurations and analyze their behavior in terms of initial answers and correction dynamics.

Selection of High-Performing Variants Based on our previous analysis and Table 2 we determine that STaSCEIE and STaSCEIF exhibit the strongest performance. These results highlight the crucial role of Evolving Initialization and the Improving Filter, while leaving the impact of Fine-Tuning strategies open for further investigation.

Performance Comparison Figure 4 illustrates the performance trends of initial answers and corrections for STaSCEIE and STaSCEIF.

The effectiveness of Fine-Tuning strategies varies between Phi3 and Qwen. For Phi3, Evolving Fine-Tuning leads to a substantial increase in initial answer quality, surpassing Fixed Fine-Tuning, while yielding moderate improvements in corrections. In contrast, for Qwen, Fixed Fine-Tuning results in superior performance for both initial answers and corrections.

Effect of Correction-Only Training Interestingly, despite the model being trained exclusively on corrections, initial answer quality also improves. Since gradient updates are applied only to correction tokens, this suggests that learning corrections either enhances the model’s factual knowledge or

improves its internal reasoning ability at the initial generation stage.

Alignment Between Initial Answers and Corrections Evolving Fine-Tuning progressively reduces the gap between initial answers and corrections, eventually leading to their alignment. For Phi3, initial answers gradually improve until they reach the quality of corrections. For Qwen, the opposite trend is observed—corrections improve until they match the initial answers.

This suggests that Evolving Fine-Tuning either helps internalize the correction process, leading to higher-quality initial answers as seen in Phi3, or stabilizes responses by preventing degradation, as observed in Qwen.

### 5 Related Works

Self-correction is a relatively new yet actively growing research domain. A systematic survey defines self-correction as a framework in which LLMs refine their responses using LLMs during inference, potentially incorporating external tools or knowledge (Kamoi et al., 2024). A significant body of work in this direction focuses on leveraging external feedback from external knowledge and verification tools (Jiang et al., 2023; Gou et al., 2024; Pan et al., 2023; Xu et al., 2023), as these approaches provide high-quality solution evaluation. However, in real-world applications, such external resources are often unavailable or computationally expensive. Moreover, relying on external verification does not pose a fundamentally challenging task for LLMs, limiting their ability to develop intrinsic reasoning and self-improvement capabilities.

An alternative approach is intrinsic selfcorrection, where the model refines its own outputs without relying on external critics. This can be implemented in a zero-shot setting, using the same model iteratively (Madaan et al., 2023), or through external models trained on synthetic errors(Paul et al., 2024) or self-generated corrections(Welleck et al., 2022). While these approaches have shown promise, they still rely on external critic models, making them closer to verification-based generation rather than true self-correction.

The only work exploring self-correction in its natural form is the SCoRE framework (Kumar et al., 2024), which was the first to experiment with intrinsic self-correction within the same model and introduced a multi-turn RL training approach for this purpose. However, SCoRE lacks a formalized

system prompt

###### Please review the given answer for accuracy. If it is incorrect, identify the incorrect information and provide the correct details. Ask and answer subquestions, and split the question into parts. Disregard the information you already have.

output

input

Step-by-step reasoning: The question refers to two different movies, one about pregnancy and another about Tracey. The actress who played in both needs to be identified.

Recalling movies about pregnancy, "Juno" comes to mind, starring Ellen Page (now Elliot Page).

system prompt

For a movie about Tracey, "Boys Don't Cry" features Hilary Swank playing the role of Tracey.

You are a helpful reasoning assistant in general domain question answering. Please reason through the question step by step very shortly before giving a final answer. Generate a short chain-of-thought rationale very shortly, and then provide the final answer in the format: "Final answer: your answer"

However, these are two different actresses, so we need to find a common actress. After factchecking, the actress who played in both movies is likely to be Reese Witherspoon, who starred in "Legally Blonde" (where her character is named Elle, not Tracey) and "Sweet Home Alabama" (which involves pregnancy). Final Answer: Reese Witherspoon

input

Question: Who played in the movies about pregnancy and Tracey? Reason step by step very shortly, then conclude with the answer. Step-by-step reasoning:

STaSC

correct answer

Step-by-step reasoning: The initial analysis contains several inaccuracies regarding the identification of an actress who starred in both a movie about pregnancy and another featuring a character named Tracey. Let's address these inaccuracies and provide the correct information.

Answer: Ellen (Elliot) Page

Inaccuracies in the Initial Analysis:

Elliot Page in "Juno": It's correctly noted that Elliot Page (formerly Ellen Page) starred in "Juno," a film centered around teenage pregnancy. However, the analysis doesn't connect Elliot Page to a film involving a character named Tracey.

Hilary Swank in "Boys Don't Cry": Hilary Swank portrayed Brandon Teena in "Boys Don't Cry," not a character named Tracey. This film doesn't align with the criteria of featuring a character named Tracey.

Reese Witherspoon in "Legally Blonde" and "Sweet Home Alabama": Reese Witherspoon's characters in these films are named Elle Woods and Melanie Smooter, respectively. Neither film prominently features a character named Tracey, nor are they primarily about pregnancy.

Correct Information:

"Juno" (2007): Elliot Page plays the titular character, Juno MacGuff, a teenager dealing with an unplanned pregnancy.

"The Tracey Fragments" (2007): Elliot Page portrays Tracey Berkowitz, a 15-year-old navigating personal challenges.

Final answer: Elliot Page

Figure 5: Example of the STaSC pipeline.

theoretical foundation and a deeper investigation of baseline algorithm adaptations. Additionally, it is limited by the use of large proprietary models, without open-sourcing the code or model weights, making it difficult to build upon for future research.

### 6 Conclusion

In this study, we introduced the Self-Taught SelfCorrection (STaSC) algorithm, which incorporates multiple algorithmic choices to enable genuine intrinsic self-correction without relying on external tools or large proprietary models. Inspired by STaR (Zelikman et al., 2022), our approach trains exclusively on self-generated data. Experiments on a QA task with two small language models demonstrate that SLMs can learn to self-correct using their own generations and even improve initial answers, despite being trained solely for corrections. Furthermore, our analysis highlights key algorith-

mic insights, emphasizing the importance of filtering selectivity, initial answer exploration, and the potential of iterative fine-tuning. To support future research, we have open-sourced our code and lightweight models.

### 7 Limitations

- • The selected SLMs, while effective, may have certain capacity limitations that could influence the extent of the self-correction process.
- • Experiments were conducted with a single run, which, while sufficient for initial insights, may introduce some variability.
- • The evaluation focuses on a Question Answering (QA) task, leaving open the opportunity to explore performance across other tasks and domains.

- • The chosen hyperparameters, though reasonable, may not fully optimize the model’s learning efficiency or overall performance.
- • A more detailed analysis of the types and patterns of corrections could further enrich our understanding of the self-correction mechanism.
- • The reward function, while practical, may not perfectly capture all nuances of desired behavior, presenting room for refinement in future work.

### 8 Ethical Considerations

Our work enables small language models to self-correct using self-generated data. We employ advanced models like Qwen-2.5 and Phi-3, pre-trained on diverse datasets, including usergenerated content. While efforts have been made to remove harmful or biased data, some biases may persist in outputs. This does not undermine our methods, which are designed to self-correct factual inaccuracies and are adaptable to other rigorously debiased models. Beyond inherent bias challenges, our work raises no additional ethical concerns.

### References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi, Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio César Teodoro Mendes, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Praneetha

Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. 2024. Phi-3 technical report: A highly capable language model locally on your phone. Preprint, arXiv:2404.14219.

Bishwamittra Ghosh, Sarah Hasan, Naheed Anjum Arafat, and Arijit Khan. 2024. Logical consistency of large language models in fact-checking. Preprint, arXiv:2412.16100.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. Critic: Large language models can selfcorrect with tool-interactive critiquing. Preprint, arXiv:2305.11738.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. arXiv preprint arXiv:2403.14403.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. Preprint, arXiv:2305.06983.

Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. 2024. When can llms actually correct their own mistakes? a critical survey of selfcorrection of llms. Transactions of the Association for Computational Linguistics, 12:1417–1440.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. 2024. Training language models to selfcorrect via reinforcement learning. arXiv preprint arXiv:2409.12917.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob

Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint, arXiv:2303.17651.

Viktor Moskvoretskii, Maria Lysyuk, Mikhail Salnikov, Nikolay Ivanov, Sergey Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina, and Alexander Panchenko. 2025. Adaptive retrieval without self-knowledge? bringing uncertainty back home. arXiv preprint arXiv:2501.12835.

Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. 2023. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. arXiv preprint arXiv:2305.12295.

Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings. 2024. REFINER: Reasoning feedback on intermediate representations. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1100–1126, St. Julian’s, Malta. Association for Computational Linguistics.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2024. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. arXiv preprint arXiv:2212.10509.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. 2022. Generating sequences by learning to self-correct. arXiv preprint arXiv:2211.00053.

Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy Phung, Rafael Rafailov, Nathan Lile, Dakota Mahan, et al. 2025. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-though. arXiv preprint arXiv:2501.04682.

Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag, William Yang Wang, and Lei Li. 2023. Instructscore: Explainable text generation evaluation with finegrained feedback. arXiv preprint arXiv:2305.14282.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. Star: Bootstrapping reasoning with reasoning. Preprint, arXiv:2203.14465.

Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, and Lu Wang. 2024. Small language models need strong verifiers to self-correct reasoning. arXiv preprint arXiv:2404.17140.

### A Technical Details

We apply a weight decay of 0.1 and use the Adam optimizer with default betas, along with a cosine annealing scheduler. Training is performed using Fully Sharded Data Parallel on two A100 GPUs, with an estimated total compute of 80 GPU hours. The dataset and Qwen come under Apache License, Phi3-mini under MIT. We are open-sourcing all our code under the Apache License.

#### Initial answer sampling prompt

You are a helpful reasoning assistant in general domain question answering. Please reason through the question step by step very shortly before giving a final answer.

Generate a short chain-of-thought rationale very shortly, and then provide the final answer.

Step-by-step reasoning: Final Answer:

Question: {question text}

Reason step by step very shortly, then conclude with the answer.

#### Correction sampling prompt

You are a helpful reasoning assistant in general domain question answering.Your task is to correct the initial response if it is incorrect.

Below is the question and the initial answer. Generate a correction to the initial answer if it is incorrect. Disregard the information you already have, look for other options. Do not use the information that does not match your criteria.

Step-by-step reasoning: Final Answer: Question: Which Portuguese soccer player has the most goals?

Step-by-step reasoning: Christiano Ronaldo has scored 900 goals in a career spanning 22 years. I cannot think of anyone else scoring more than him.

Final Answer: Christiano Ronaldo

Step-by-step reasoning: Yes, Christiano Ronaldo is the correct answer. Other details are unimportant.

Final Answer: Christiano Ronaldo

Question: Which actress played in both movies: one about preg-cy and the other about Tracey?

Step-by-step reasoning: 1. Identify movies about preg-cy and about Tracey. 2. Movies about preg-cy include ’Juno’ and ’Knocked Up’. 3. Movie about Tracey is likely ’Tracy Chapman’ or a documentary, but ’Tracy’ could be a reference to ’Tracy Flick’ in the movie ’Election’. 4. Considering well-known movies with these themes, actress Reese Witherspoon comes to mind as a possible match.

Final Answer: Reese Witherspoon

Step-by-step reasoning: Let’s break down the answer. ’Juno’ is indeed a movie about preg-cy, but it starred Ellen Page (now Elliot Page), not Reese Witherspoon. Reese Witherspoon did not star in ’Juno.’ Therefore, the association between Reese Witherspoon and ’Juno’ is incorrect. Movies About Tracey is likely ’Tracey Fragments (2007)’, a movie in which Ellen Page also starred as the lead character, Tracey Berkowitz. Ellen Page (Elliot Page) connects both movies.

Final answer: Ellen Page (now Elliot Page)

Question: {question text}

Initial Answer: {initial answer} Write a correction if the initial answer is incorrect.

