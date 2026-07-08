!

## Discovering Hidden Gems in Model Repositories

Jonathan Kahana and Eliahu Horwitz and Yedid Hoshen Department of Computer Science School of Computer Science and Engineering The Hebrew University of Jerusalem, Israel jonathan.kahana@mail.huji.ac.il https://jonkahana.github.io/hidden_gems

!

[Figure 1]

### Abstract

[Figure 2]

[Figure 3]

Public repositories host millions of fine-tuned models, yet community usage remains disproportionately concentrated on a small number of foundation checkpoints. We investigate whether this concentration reflects efficient market selection or if superior models are systematically overlooked. Through an extensive evaluation of over 2,000 models, we show the prevalence of “hidden gems”, unpopular fine-tunes that significantly outperform their popular counterparts. Notably, within the Llama-3.1-8B family, we find rarely downloaded checkpoints that improve math performance from 83.2% to 96.0% without increasing inference costs. However, discovering these models through exhaustive evaluation of every uploaded model is computationally infeasible. We therefore formulate model discovery as a Multi-Armed Bandit problem and accelerate the Sequential Halving search algorithm by using shared query sets and aggressive elimination schedules. Our method retrieves top models with as few as 50 queries per candidate, accelerating discovery by over 50×.

Best Math:

# arXiv:2601.22157v1[cs.LG]29Jan2026

[Figure 4]

###### +12.8% on GSM8K

[Figure 5]

Llama3.1 8B

!

|aisingapore/ Llama-SEA-LION-v3-8B-IT<br><br>Best Overall (+3.4%)<br><br>[Figure 6]|
|---|

[Figure 7]

[Figure 8]

Llama3.1 8B Instruct

[Figure 9]

[Figure 10]

Best Coder:

Root Full Fine-tune Adapter

+3.2% on MBPP

[Figure 11]

[Figure 12]

💎 Hidden Gem

[Figure 13]

Figure 1: Hidden Gems & Repository Inefficiency. Center: The Llama3.1-8B Model Tree, where node size reflects downloads (log scale). Our evaluation reveals “hidden gems” (circled): unpopular models that significantly outperform widely used baselines. Top Left: Cumulative download rates showing usage is extremely concentrated in a tiny fraction of top models. Bottom Right: The vast majority of models (over 90%) are rarely explored, receiving ≤ 15 monthly downloads.

### 1 Introduction

Public model repositories, such as Hugging Face, have democratized access to new, high-performing models (Casper et al., 2025). However, as these repositories scale to millions of checkpoints, selecting the right model for each task becomes a critical bottleneck. Relying on documentation is often ineffective, as model cards (Mitchell et al., 2019) are frequently incomplete or missing entirely (Kahana et al., 2025b; Horwitz et al., 2025b). Consequently, the vast majority of users default to the foundation model at the root of the Model Tree (Horwitz et al., 2025a), such as the official Qwen or Llama checkpoints or to its official instruction fine-tuning. While this strategy is simple, its efficacy is unproven. This paper tackles two questions:

i) Are the most popular models actually the best performers available? ii) If not, how can we efficiently identify superior models among many candidates?

First, we investigate whether users are successful at identifying the best models by analyzing the relation between download counts and performance. By evaluating more than 2,000 models from several popular model families on diverse tasks, we consistently identify unpopular models that significantly outperform their popular counterparts. We term these lost high-performers “hidden gems”.

Discovering such hidden gems presents a signifi-

Table 1: Model Discovery Results. Comparison of popular (Base) and gem models performance. For each model tree, we present the base performance, the discovered gem performance, and the resulting gain. RouterBenchs is the combined performance of all other tasks.

Tree ARC-Cs WinoG.s MMLUs MBPPs GSM8Ks RouterB.s

Base 79.3 66.0 65.5 68.6 83.5 71.6 Gem 84.1 67.1 67.8 73.3 89.0 73.2 Gain +4.8 +1.1 +2.3 +4.7 +5.5 +1.6

Qwen 3B

Base 90.0 67.1 71.3 80.9 89.7 78.3 Gem 91.1 72.5 73.9 82.1 92.1 79.1 Gain +1.1 +5.4 +2.6 +1.2 +2.4 +0.8

Qwen 7B

Base 74.6 56.4 56.6 47.0 40.6 55.6 Gem 81.8 66.9 66.2 58.8 80.7 69.6 Gain +7.2 +10.5 +9.6 +11.8 +40.1 +14.0

Mistral 7B

Base 80.9 59.7 64.1 65.6 83.2 71.3 Gem 86.0 67.1 71.6 68.8 96.0 74.7 Gain +5.1 +7.4 +7.5 +3.2 +12.8 +3.4

Llama 8B

cant challenge. Exhaustively evaluating all candidate models may require billions of inferences. Unlike existing methods that rank new models within a predefined leaderboard (Perlitz et al., 2024; Tamura et al., 2025; Ashury-Tahan et al., 2024), we aim to rank an entire model population from scratch without knowing the rank of previous models. Our approach can be easily integrated with query selection methods (Polo et al., 2024; Zouhar et al., 2025) although we do not focus on this here. Concretely, we formulate model discovery as a Multi-Armed Bandit (MAB) problem (Katehakis and Veinott Jr, 1987). We adapt the Sequential Halving algorithm (Karnin et al., 2013) to exploit the unique characteristics of model evaluation, such as the ability to share query sets across models for variance reduction. Our approach consistently finds a top-3 model with just 50 queries per model, making the search over 50× faster than exhaustive baselines, while improving average performance by over 4.5%.

### 2 Do Hidden Gems Exist?

The Centralization of Usage. Public model repositories are expanding rapidly, with Hugging Face alone hosting over two million models. Despite this abundance, usage remains heavily centralized: As shown in Fig. 1, a mere 0.0015% of models account for 95% of all downloads while the vast majority of the ecosystem is inactive. This disparity suggests two competing hypotheses. The Efficient Discovery hypothesis posits that usage concentration is causal: users successfully converge to the best models. Alternatively, the Information Asymmetry hypothesis argues that the best models remain “hidden” in the long tail, and that users have no real choice but using them due to a lack of reli-

able signals. Concretely, as evaluating millions of models on each task is infeasible, the default option becomes well-tested but sub-optimal checkpoints.

Defining Hidden Gems. We propose a test to decide between these hypotheses. If we can identify unpopular models that strictly outperform the popular choice, we can reject the efficient discovery hypothesis. We term such models Hidden Gems.

Formally, let T = {m1,...,mK} be a set of models. Let ri denote the performance of model mi on dataset D. We define the Popular Consensus group, P ⊂ T , as the top 1% of models by download count, and the Elite group, ED ⊂ T , as the top 1% by performance. A model mi is a Hidden Gem if it satisfies three criteria: i) Obscurity: mi ∈/ P (it is unpopular); ii) Excellence: mi ∈ ED (it is top-tier); and iii) Dominance: ri > maxmj∈P rj. Condition (iii) is crucial: Gems must strictly outperform the best popular models to justify the search. Popularity ̸= Performance. We search for hidden gems by evaluating over 2,000 models. To ensure identical inference costs and leave capability as the sole differentiator, we compare fine-tunes only within the same Model Trees (Horwitz et al., 2025c,a), i.e., sets of fine-tunes derived from a shared ancestor. We test four major trees: Qwen2.5 (3B & 7B) (Yang et al., 2024a), Mistral-7B (Jiang et al., 2023), and Llama3.1-8B (Dubey et al., 2024). To assess general quality, we use RouterBench (Hu et al., 2024), which aggregates diverse benchmarks including MBPP, Winograd, MMLU, ARCChallenge, and GSM8K (Sakaguchi et al., 2021; Clark et al., 2018; Hendrycks et al., 2020; Cobbe et al., 2021; Austin et al., 2021). Due to computational constraints, we evaluate on a subsampled set of 2,500 randomly chosen queries, denoted by a subscript s, e.g., RouterBenchs (see App. C). Unsurprisingly, usage is dominated by the official base releases; e.g., in the Qwen-3B tree, the official base and instruct variants alone capture over 80% of all downloads.

Tab. 1 compares the performance of the popular consensus against the best-performing gems. Across all trees and tasks, we consistently uncover models that are much better than their popular predecessors. A striking example is the Qwen-3B tree, where a math-oriented fine-tune boosts GSM8Ks accuracy from 83.5% up to 89.0%, nearing the performance of the best Qwen-7B base version, with less than half the parameters. Crucially, we reveal that hidden gems are not limited to niche tasks; every tree also includes unpopular models that are

|Overall Performance<br><br>[Figure 14]<br><br>Best Overall<br><br>+0.8% 80 Monthly Downloads<br><br>Qwen-7B Instruct (Most Popular) 9,600,000 Monthly Downloads<br><br>[Figure 15]<br><br>|
|---|

Qwen2.5 3B

|Math<br><br>[Figure 16]<br><br>Best in Math<br><br>+12.8% on GSM8K 47 Monthly Downloads<br><br>Qwen2.5 7B Math 96,000 Monthly Downloads<br><br>[Figure 17]<br><br>|
|---|

[Figure 18]

[Figure 19]

Second Best: 72.7% on RouterBench 1,000 Monthly Downloads

Best Model: 74.7% on RouterBench 3,500 Monthly Downloads

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Qwen2.5 3B Instruct

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Best Coder

[Figure 34]

73.3% on MBPP (+4.7%) 8 Monthly Downloads

[Figure 35]

[Figure 36]

Continuous Linear Color Scale (Values at every 20%)

Continuous Linear Color Scale (Values at every 20%)

[Figure 37]

0.75

0.73

| |
|---|

| |
|---|

0.71

0.66

0.60 0.64 0.67 0.71 0.75

0.45 0.52 0.59 0.66 0.73

[Figure 38]

0.67

0.59

Llama3.1-8B Instruct (Most Popular) 71.3% on RouterBench 6,000,000 Monthly Downloads

Qwen2.5-3B Coder Instruct 68.6% on MBPP 22,000 Monthly Downloads

!

|Coding<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Best Coder<br><br>+1.2% on MBPP 6 Monthly Downloads<br><br>Qwen-7B Coder Instruct 250,000 Monthly Downloads<br><br>[Figure 41]<br><br>|
|---|

0.52

0.64

≤0.45

0.60

Value

Value

(a) Qwen-3B Coding Acc.

(b) Qwen 7B Hidden Gems.

(c) Llama 3.1 8B RouterB.s Acc.

Figure 2: Visualizing Hidden Gems in Model Trees. Nodes represent models, with size corresponding to monthly downloads in log-scale. (a) Qwen-3B tree colored by MBPPs coding performance; the best coder (circled) remains unpopular despite significantly outperforming the base instruct model. (b) Qwen-7B tree highlighting gems in math, coding and overall (RouterBenchs) performance. (c) Llama-3.1-8B tree colored by overall performance; the best-performing model has orders of magnitude fewer downloads than the official Llama3.1 8B Instruct version.

better generalists than the popular base versions.

The Failure of Heuristics. Why are these models missed? We found that over 90% of the identified gems lacked performance documentation relevant to their specific strengths, leaving users with no signal to identify them (see App. G). We then map the locations of these gems in Fig. 2. The bestperforming models do not cluster near the root or along predictable trajectories. This implies that simple search heuristics based on popularity or graph centrality are likely to fail. These findings refute the efficient discovery hypothesis and motivate active search methods as we propose next.

### 3 Efficient Model Discovery

Problem Formulation. Evaluating millions of models may require billions of queries which is infeasible. We therefore formulate practical model discovery as a budget-constrained problem. Given a model tree T = {m1,...,mK} and dataset D, our goal is to identify the best-performing model m∗ under a global query budget B.

Best Arm Identification (BAI). We frame this as a Fixed-Budget Best-Arm Identification (Audibert and Bubeck, 2010) problem: the Arms are candidate models in T ; an Action is a single query evaluation; and Rewards are binary (+1 for correct, else 0). Unlike standard MAB, where an agent maximizes total reward during learning, our goal is pure exploration. We aim to minimize the simple

regret (gap between the chosen model and the best one) after the fixed budget B is exhausted.

Sequential Halving (SH) (Karnin et al., 2013). We adopt SH as our base algorithm. SH operates in rounds s = 1...S. First, it uniformly allocates a small budget (5-25 queries) to all models. At the end of each round, models are ranked by empirical accuracy; the bottom 50% are eliminated, and the survivors are tested with more queries. This repeats until a single model remains. While SH is theoretically sound, standard implementations are sample-inefficient in our domain. We therefore introduce two domain-specific modifications. We illustrate our proposed approach in Fig. 3.

Variance Reduction via Correlated Sampling. A standard SH implementation samples queries independently for each model. However, benchmarks contain a mix of trivial and even unsolvable problems. If Model A is tested with “easy” queries and Model B with “hard” ones, the ranking can become noisy. To mitigate this, we employ Correlated Sampling. In every round, we enforce that all surviving models are tested with the exact same queries, thus minimizing the variance of the difference estimator. Aggressive Elimination Schedule. We present the distribution of model quality in Fig. 4, where we notice that the vast majority of uploads are low-quality or broken. These under-performers can be detected with very few queries. Standard SH, which eliminates only 50% of candidates per round, wastes budget refining estimates for obviously poor mod-

Table 2: Model Discovery Results. We evaluate the top retrieval of each method. For each query budget, we report the mean rank and accuracy of the retrieved models for each model tree. We report the mean out of 100 repetitions.

Queries Per Model

Qwen-3B Qwen-7B Mistral-7B Llama-8B Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑

Method

Random Selection 233.8 0.588 318.4 0.599 144.6 0.431 270.8 0.601 Best Base 56.5 0.716 30.0 0.783 62.0 0.556 41.0 0.713

-

Uniform 166.3 0.671 173.8 0.724 29.5 0.656 222.0 0.665 UCB 88.0 0.708 83.9 0.770 14.1 0.684 110.2 0.702 UCB-StdDev. 81.8 0.710 75.2 0.772 11.8 0.686 96.0 0.707 TTTS 92.6 0.708 75.7 0.772 14.0 0.685 103.6 0.702 Successive Rejects 94.4 0.706 128.3 0.752 25.6 0.662 171.5 0.683 BayesElim 56.4 0.717 58.9 0.778 13.5 0.685 79.9 0.708 UCB-E 78.4 0.710 83.7 0.769 13.6 0.684 106.5 0.703 Sequential Halving 69.3 0.714 62.9 0.777 15.2 0.684 75.2 0.710 Ours 11.3 0.726 15.8 0.786 2.6 0.694 15.4 0.725

10

Uniform 91.2 0.707 81.9 0.770 17.4 0.683 114.9 0.705 UCB 41.2 0.719 39.8 0.782 8.9 0.689 55.8 0.718 UCB-StdDev. 34.7 0.721 28.9 0.784 4.0 0.693 37.1 0.722 TTTS 53.1 0.717 51.7 0.780 10.8 0.687 70.6 0.712 Successive Rejects 83.1 0.710 78.4 0.773 16.5 0.682 92.0 0.707 BayesElim 30.0 0.721 31.0 0.784 6.7 0.691 34.4 0.721 UCB-E 37.4 0.720 33.1 0.783 4.3 0.693 43.4 0.720 Sequential Halving 41.0 0.719 29.6 0.784 7.7 0.690 29.9 0.720 Ours 3.5 0.729 3.6 0.790 1.6 0.695 3.0 0.736

50

Best Available 1.0 0.732 1.0 0.791 1.0 0.696 1.0 0.747

els. We therefore modify the schedule to perform aggressive elimination already in the first round. Specifically, we reduce the candidate pool to a fixed size of 100 models immediately. This "fail-fast" strategy reallocates a larger chunk of the compute budget to distinguishing between the elite candidates in later rounds. For more details see App. C. App. D confirms our modified scheduler is highly important, significantly boosting retrieval ranks and accuracies through improved query allocation.

### 4 Experiments

Experimental Setting. We evaluate our method and baselines on searching for the optimal model m∗. We use four distinct model pools derived from the Qwen-3B, Qwen-7B, Mistral-7B, and Llama38B trees. We define the total query budget as B = N × K, where K is the number of models and N ∈ {10,50} is the average queries per model (for more see App. E.). We repeat each experiment 100 times and report the mean rank and top-1 accuracy. Baselines. We compare against 8 established algorithms. Uniform allocates budget equally. UCB (Auer et al., 2002), UCB-E (Audibert and Bubeck, 2010), and UCB-StdDev greedily sample based on upper confidence bounds. Successive Rejects (Audibert and Bubeck, 2010) and Sequential Halving (Karnin et al., 2013) iteratively eliminate the worst models. We also include TTTS (Russo, 2016), which uses top-two Thompson sampling,

and Bayesian Elimination (Atsidakou et al., 2022), which prunes based on posterior probabilities.

Results. Table 2 summarizes the performance for 10,50 queries per model budgets (for more see App. E). In the low-budget regime (N = 10), standard baselines struggle where in the Qwen and Llama trees, most methods fail to even outperform the popular base versions. The exception is the Mistral-7B tree, where many models surpass the base (see App. F). In contrast, our method consistently identifies “Hidden Gems” that significantly outperform the popular consensus across all trees.

In the mid-budget regime (N = 50), eliminationbased baselines begin to improve, occasionally beating the base models, but often finding models significantly weaker than the best available. Our approach, uses correlated sampling and aggressive pruning and consistently converges to a top-3 model with much better accuracy. See App. D for ablations confirming both design choices.

### 5 Conclusion

We demonstrated that public repositories contain “Hidden Gems”, unpopular models that strictly outperform foundation checkpoints. To address the computational intractability of finding them, we proposed an accelerated Sequential Halving algorithm. Our method consistently identifies elite models with only 50 queries per candidate, making model selection practically achievable.

### References

Shir Ashury-Tahan, Ariel Gera, Benjamin Sznajder, Leshem Choshen, Liat Ein-Dor, and Eyal Shnarch. 2024. Label-efficient model selection for text generation. arXiv preprint arXiv:2402.07891.

Alexia Atsidakou, Sumeet Katariya, Sujay Sanghavi, and Branislav Kveton. 2022. Bayesian fixedbudget best-arm identification. arXiv preprint arXiv:2211.08572.

Jean-Yves Audibert and Sébastien Bubeck. 2010. Best arm identification in multi-armed bandits. In COLT23th Conference on learning theory-2010, pages 13– p.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer.

2002. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1 others. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Omri Avrahami, Dani Lischinski, and Ohad Fried. 2022. Gan cocktail: mixing gans without dataset access. In European Conference on Computer Vision, pages 205–221. Springer.

Guy Bar-Shalom, Fabrizio Frasca, Yaniv Galron, Yftah Ziser, and Haggai Maron. 2025a. Beyond token probes: Hallucination detection via activation tensors with act-vit. arXiv preprint arXiv:2510.00296.

Guy Bar-Shalom, Fabrizio Frasca, Derek Lim, Yoav Gelberg, Yftah Ziser, Ran El-Yaniv, Gal Chechik, and Haggai Maron. 2025b. Beyond next token probabilities: Learnable, fast detection of hallucinations and data contamination on llm output distributions. Preprint, arXiv:2503.14043.

Rishi Bommasani, Dilara Soylu, Thomas I. Liao, Kathleen A. Creel, and Percy Liang. 2025. Ecosystem Graphs: Documenting the Foundation Model Supply Chain, page 196–209. AAAI Press.

Stephen Casper, Kyle O’Brien, Shayne Longpre, Elizabeth Seger, Kevin Klyman, Rishi Bommasani, Aniruddha Nrusimha, Ilia Shumailov, Sören Mindermann, Steven Basart, and 1 others. 2025. Open technical problems in open-weight ai model risk management. Social Science Research Network.

Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim, and Yoav Katz. 2023. Where to start? analyzing the potential value of intermediate models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1446–1470, Singapore. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind

Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, and Leshem Choshen. 2023. ColD fusion: Collaborative descent for distributed multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 788–806, Toronto, Canada. Association for Computational Linguistics.

Amil Dravid, Yossi Gandelsman, Alexei A Efros, and Assaf Shocher. 2023. Rosetta neurons: Mining the common units in a model zoo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1934–1943.

Amil Dravid, Yossi Gandelsman, Kuan-Chieh Wang, Rameen Abdal, Gordon Wetzstein, Alexei A Efros, and Kfir Aberman. 2024. Interpreting the weight space of customized diffusion models. arXiv preprint arXiv:2406.09413.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Gabriel Eilertsen, Daniel Jönsson, Timo Ropinski, Jonas Unger, and Anders Ynnerman. 2020. Classifying the classifier: dissecting the weight space of neural networks. arXiv:2002.05688.

Ziya Erkoç, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. 2023. Hyperdiffusion: Generating implicit neural fields with weight-space diffusion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14300–14310.

Damian Falk, Konstantin Schürholt, Konstantinos Tzevelekakis, Léo Meynent, and Damian Borth. 2025. Learning model representations using publicly available model hubs. arXiv preprint arXiv:2510.02096.

Shangbin Feng, Yike Wang, Weijia Shi, and Yulia Tsvetkov. 2025. Data swarms: Optimizable generation of synthetic evaluation data. arXiv preprint arXiv:2506.00741.

Shangbin Feng, Zifeng Wang, Yike Wang, Sayna Ebrahimi, Hamid Palangi, Lesly Miculicich, Achin Kulshrestha, Nathalie Rauschmayr, Yejin Choi, Yulia Tsvetkov, and 1 others. 2024. Model swarms: Collaborative search to adapt llm experts via swarm intelligence. arXiv preprint arXiv:2410.11163.

Sarah Gao and Andrew Kean Gao. 2023. On the origin of llms: An evolutionary tree and graph for 15,821 large language models. arXiv preprint arXiv:2307.09793.

Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. 2023. Knowledge is a region in weight space for fine-tuned language models. In The 2023 Conference on Empirical Methods in Natural Language Processing.

David Ha, Andrew Dai, and Quoc V Le. 2016. Hypernetworks. arXiv preprint arXiv:1609.09106.

Robert Hecht-Nielsen. 1990. On the algebraic structure of feedforward network weight spaces. In Advanced Neural Computers, pages 129–135. Elsevier.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Vincent Herrmann, Francesco Faccio, and Jürgen Schmidhuber. 2024. Learning useful representations of recurrent neural network weight matrices. In Fortyfirst International Conference on Machine Learning.

Aspen Hopkins, Sarah H Cen, Isabella Struckman, Andrew Ilyas, Luis Videgaray, and Aleksander Ma˛dry. 2025. Ai supply chains: An emerging ecosystem of ai actors, products, and services. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 8, pages 1266–1277.

Eliahu Horwitz, Bar Cavia, Jonathan Kahana, and Yedid Hoshen. 2025a. Learning on model weights using tree experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20468–20478.

Eliahu Horwitz, Jonathan Kahana, and Yedid Hoshen. 2024. Recovering the pre-fine-tuning weights of generative models. In International Conference on Machine Learning, pages 18882–18904. PMLR.

Eliahu Horwitz, Nitzan Kurer, Jonathan Kahana, Liel Amar, and Yedid Hoshen. 2025b. We should chart an atlas of all the world’s models. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems Position Paper Track.

Eliahu Horwitz, Asaf Shul, and Yedid Hoshen. 2025c. Unsupervised model tree heritage recovery. In The Thirteenth International Conference on Learning Representations.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. 2024. Routerbench: A benchmark for multi-llm routing system. arXiv preprint arXiv:2403.12031.

Ruoxi Jia, Luis Oala, Wenjie Xiong, Suqin Ge, Jiachen T. Wang, Feiyang Kang, and Dawn Song. 2025. A sustainable AI economy needs data deals that work

for generators. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems Position Paper Track.

Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. ArXiv, abs/2310.06825.

Jonathan Kahana, Eliahu Horwitz, Imri Shuval, and Yedid Hoshen. 2025a. Deep linear probe generators for weight space learning. In The Thirteenth International Conference on Learning Representations.

Jonathan Kahana, Or Nathan, Eliahu Horwitz, and Yedid Hoshen. 2025b. Can this model also recognize dogs? zero-shot model search from weights. arXiv preprint arXiv:2502.09619.

Zohar Karnin, Tomer Koren, and Oren Somekh. 2013. Almost optimal exploration in multi-armed bandits. In International conference on machine learning, pages 1238–1246. PMLR.

Michael N Katehakis and Arthur F Veinott Jr. 1987. The multi-armed bandit problem: decomposition and computation. Mathematics of Operations Research, 12(2):262–268.

Miltiadis Kofinas, Boris Knyazev, Yan Zhang, Yunlu Chen, Gertjan J Burghouts, Efstratios Gavves, Cees GM Snoek, and David W Zhang. 2024. Graph neural networks for learning equivariant representations of neural networks. arXiv preprint arXiv:2403.12143.

Rohith Kuditipudi, Jing Huang, Sally Zhu, Diyi Yang, Christopher Potts, and Percy Liang. 2025. Blackbox model provenance via palimpsestic membership inference. arXiv preprint arXiv:2510.19796.

Derek Lim, Haggai Maron, Marc T Law, Jonathan Lorraine, and James Lucas. 2023. Graph metanetworks for processing diverse neural architectures. arXiv preprint arXiv:2312.04501.

Derek Lim, Moe Putterman, Robin Walters, Haggai Maron, and Stefanie Jegelka. 2024. The empirical impact of neural parameter symmetries, or lack thereof. arXiv preprint arXiv:2405.20231.

Shayne Longpre, Christopher Akiki, Campbell Lund, Atharva Kulkarni, Emily Chen, Irene Solaiman, Avijit Ghosh, Yacine Jernite, and Lucie-Aimée Kaffee. 2025. Economies of open intelligence: Tracing power & participation in the model ecosystem. arXiv preprint arXiv:2512.03073.

Daohan Lu, Sheng-Yu Wang, Nupur Kumari, Rohan Agarwal, Mia Tang, David Bau, and Jun-Yan Zhu. 2023. Content-based search for deep generative models. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12.

Michael Luo, Justin Wong, Brandon Trabucco, Yanping Huang, Joseph E Gonzalez, Ruslan Salakhutdinov, Ion Stoica, and 1 others. 2024. Stylus: Automatic adapter selection for diffusion models. Advances in Neural Information Processing Systems, 37:32888– 32915.

Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency, pages 220–229.

Mohammed Muqeeth, Haokun Liu, Yufan Liu, and Colin Raffel. 2024. Learning to route among specialized experts for zero-shot generalization. arXiv preprint arXiv:2402.05859.

Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, and Haggai Maron. 2023a. Equivariant architectures for learning in deep weight spaces. In International Conference on Machine Learning, pages 25790–25816. PMLR.

Aviv Navon, Aviv Shamsian, Ethan Fetaya, Gal Chechik, Nadav Dym, and Haggai Maron. 2023b. Equivariant deep weight space alignment. arXiv preprint arXiv:2310.13397.

Cuong Nguyen, Tal Hassner, Matthias Seeger, and Cedric Archambeau. 2020. Leep: A new measure to evaluate transferability of learned representations. In International Conference on Machine Learning, pages 7294–7305. PMLR.

Tuomas Oikarinen and Tsui-Wei Weng. 2023. CLIPdissect: Automatic description of neuron representations in deep vision networks. In The Eleventh International Conference on Learning Representations.

Cailean Osborne, Jennifer Ding, and Hannah Rose Kirk. 2024. The ai community building the future? a quantitative analysis of development activity on hugging face hub. Journal of Computational Social Science, 7(2):2067–2105.

Koyena Pal, David Bau, and Renée J Miller. 2024. Model lakes. arXiv preprint arXiv:2403.02327.

William Peebles, Ilija Radosavovic, Tim Brooks, Alexei A Efros, and Jitendra Malik. 2022. Learning to learn with generative models of neural network checkpoints. arXiv preprint arXiv:2209.12892.

Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv, Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal Shmueli-Scheuer, and Leshem Choshen. 2024. Efficient benchmarking (of language models). In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2519–2536, Mexico City, Mexico. Association for Computational Linguistics.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. 2024. tinybenchmarks: evaluating llms with fewer examples. arXiv preprint arXiv:2402.14992.

Theo Putterman, Derek Lim, Yoav Gelberg, Stefanie Jegelka, and Haggai Maron. 2024. Learning on loras: Gl-equivariant processing of low-rank weight spaces for large finetuned models. arXiv preprint arXiv:2410.04207.

Daniel Russo. 2016. Simple bayesian algorithms for best arm identification. In Conference on learning theory, pages 1417–1418. PMLR.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106.

Mohammad Salama, Jonathan Kahana, Eliahu Horwitz, and Yedid Hoshen. 2025. Dataset size recovery from fine-tuned model weights. In Workshop on Neural Network Weights as a New Data Modality.

Konstantin Schürholt, Giorgos Bouritsas, Eliahu Horwitz, Derek Lim, Yoav Gelberg, Bo Zhao, Allan Zhou, Damian Borth, and Stefanie Jegelka. 2024. Neural network weights as a new data modality. In ICLR 2025 Workshop Proposals.

Konstantin Schürholt, Boris Knyazev, Xavier Giró-i Nieto, and Damian Borth. 2022. Hyper-representations as generative models: Sampling unseen neural network weights. Advances in Neural Information Processing Systems, 35:27906–27920.

Konstantin Schürholt, Dimche Kostadinov, and Damian Borth. 2021. Self-supervised representation learning on neural network weights for model characteristic prediction. Advances in Neural Information Processing Systems, 34:16481–16493.

Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. 2023. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600.

Aviv Shamsian, Aviv Navon, David W Zhang, Yan Zhang, Ethan Fetaya, Gal Chechik, and Haggai Maron. 2024. Improved generalization of weight space networks via augmentations. arXiv preprint arXiv:2402.04081.

George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. 2024. Model merging with svd to tie the knots. arXiv preprint arXiv:2410.19735.

Takuya Tamura, Taro Yano, Masafumi Enomoto, and Masafumi Oyamada. 2025. Can a crow hatch a falcon? lineage matters in predicting large language model performance. arXiv preprint arXiv:2504.19811.

Anh T Tran, Cuong V Nguyen, and Tal Hassner. 2019. Transferability and hardness of supervised classification tasks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1395–1405.

Mengying Wang, Moming Duan, Yicong Huang, Chen Li, Bingsheng He, and Yinghui Wu. 2025. Ml-asset management: Curation, discovery, and utilization. arXiv preprint arXiv:2509.23577.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and 1 others. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. 2024. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36.

Qwen An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxin Yang, Jingren Zhou, Junyang Lin, and 25 others. 2024a. Qwen2.5 technical report. ArXiv, abs/2412.15115.

Xinyu Yang, Weixin Liang, and James Zou. 2024b. Navigating dataset documentations in ai: A large-scale analysis of dataset cards on hugging face. arXiv preprint arXiv:2401.13822.

Nicolas Yax, Pierre-Yves Oudeyer, and Stefano Palminteri. 2025. PhyloLM: Inferring the phylogeny of large language models and predicting their performances in benchmarks. In The Thirteenth International Conference on Learning Representations.

- Runpeng Yu and Xinchao Wang. 2024. Neural lineage. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4797–4807.
- Runpeng Yu and Xinchao Wang. 2025. Neural phylogeny: Fine-tuning relationship detection among neural networks. In The Thirteenth International Conference on Learning Representations.

Amir R Zamir, Alexander Sax, , William B Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. 2018. Taskonomy: Disentangling task transfer learning. In 2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.

Yi-Kai Zhang, Ting-Ji Huang, Yao-Xiang Ding, DeChuan Zhan, and Han-Jia Ye. 2023. Model spider: Learning to rank pre-trained models efficiently. Advances in Neural Information Processing Systems, 36:13692–13719.

Bo Zhao, Robin Walters, and Rose Yu. 2025. Symmetry in neural network parameter spaces. arXiv preprint arXiv:2506.13018.

Allan Zhou, Chelsea Finn, and James Harrison. 2024a. Universal neural functionals. arXiv preprint arXiv:2402.05232.

Allan Zhou, Kaien Yang, Kaylee Burns, Adriano Cardace, Yiding Jiang, Samuel Sokota, J Zico Kolter, and Chelsea Finn. 2024b. Permutation equivariant neural functionals. Advances in neural information processing systems, 36.

Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J Zico Kolter, and Chelsea Finn. 2024c. Neural functional transformers. Advances in neural information processing systems, 36.

Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, and Percy Liang. 2025. Independence tests for language models. arXiv preprint arXiv:2502.12292.

Vilém Zouhar, Peng Cui, and Mrinmaya Sachan. 2025. How to select datapoints for efficient human evaluation of nlg models? Preprint, arXiv:2501.18251.

First steps eliminate most models

💎

💎

💎

✅ ❌ ✅ ❌ ✅ ❌ ✅ ❌ ✅ ❌ ❌ ❌ ✅ ✅ ✅ ❌ ✅ ❌

✅ ❌ ✅ ✅ ✅

Test remaining models on the same subset of unseen queries

Keep best performing models

❌ ✅ ✅ ❌ ❌

❌ ✅ ❌ ❌ ❌

...

✅ ❌ ❌ ✅ ❌ ✅ ✅ ✅

✅ ✅ ✅

Figure 3: Our proposed Model Search Algorithm

### A Limitations

Evaluation Resources. While our approach expedites the search by over 50×, it still requires evaluating all models on a small number of queries. An alternative direction is weight-space learning, which aims to learn semantic representations of model weights, or subsets of weights, and retrieve models directly in weight space (Kahana et al., 2025b; Horwitz et al., 2025a; Lu et al., 2023). Although weight-space learning has the potential to further accelerate discovery, it is currently limited to small-scale networks and benchmarks.

Other Tasks. While we found hidden gems in math, coding, question answering and general performance, we did not evaluate all possible tasks. Finding gems for new task would require to reevaluate the models on this task.

### B Related Works and Model Populations

Selecting an appropriate model for a given use case has long been a central challenge in machine learning (Zamir et al., 2018; Lu et al., 2023; Zhang et al., 2023; Luo et al., 2024; Tran et al., 2019; Nguyen et al., 2020; Choshen et al., 2023; Muqeeth et al., 2024). The recent explosive growth in the number of publicly available models has further intensified this challenge, catalyzing a growing body of work on model populations. One prominent direction is Weight-Space Learning, which aims to train neural networks that treat other neural networks not only as functions, but as data points (Schürholt et al., 2024, 2021, 2022; Horwitz et al., 2025b; Pal et al., 2024). The objective is to predict a wide range of model attributes (Salama et al., 2025), including performance (Kahana et al., 2025a; Eilertsen et al., 2020), functionality (Horwitz et al., 2025a; Falk et al., 2025; Gueta et al., 2023; Herrmann et al., 2024), lineage (Horwitz et al., 2025c; Zhu et al.,

2025; Yu and Wang, 2025; Yax et al., 2025; Yu and Wang, 2024), and training trajectories (Horwitz et al., 2024; Kuditipudi et al., 2025), directly from model weights, without access to in distribution data. More broadly, weight space learning seeks to learn compact, structured representations of neural networks that enable search (Kahana et al., 2025b), comparison, and discovery directly in weight space. This is challenging due to the extreme dimensionality of model parameters and the presence of many parameter space symmetries (Hecht-Nielsen, 1990; Zhao et al., 2025), such as permutation invariances (Lim et al., 2024; Shamsian et al., 2024; Lim et al.,

- 2023; Putterman et al., 2024; Navon et al., 2023b,a; Kofinas et al., 2024; Zhou et al., 2024c,a,b).

Beyond predicting attributes of individual models, recent works study model populations as structured objects in their own right (Horwitz et al., 2025b; Wang et al., 2025). This line of research asks how populations are organized, what insights can be derived from their global structure, how to identify gaps or communities within the population (Horwitz et al., 2025b), and how AI model supply chains (Hopkins et al., 2025; Jia et al., 2025; Bommasani et al., 2025) shape the distribution of available models (Yang et al., 2024b; Osborne et al.,

- 2024; Gao and Gao, 2023; Longpre et al., 2025). Model populations have also been leveraged as

a resource for improving existing models. A notable example is model merging (Yadav et al., 2024; Stoica et al., 2024; Avrahami et al., 2022; Shah et al., 2023; Wortsman et al., 2022), where the weights of two or more models are combined to produce a model with improved or complementary capabilities. More broadly, population level approaches have been explored for collaborative optimization (Feng et al., 2024, 2025; Don-Yehiya et al., 2023), collaborative generation, weight generation (Dravid et al., 2024; Peebles et al., 2022; Ha et al., 2016; Erkoç et al., 2023), and interpretability (Dravid et al., 2024, 2023; Bar-Shalom et al., 2025a,b; Oikarinen and Weng, 2023). In contrast, we show that even without applying such population level methods, existing model populations already contain highly valuable models that can substantially outperform commonly used models.

### C Implementation Details

Model Selection. We select models from 4 highly popular model trees created from 4 popular base models: Qwen2.5-3B, Qwen2.5-7B, Mistral-

7B and Llama3.1-8B. As these trees are very large we cannot evaluate them all. Therefore, for each tree we select 400 downstream fine-tunings of the base model (including itself) and 400 adapters (if existing). To ensure we dont miss a highperforming popular version, we select the 300 most popular models of the tree first and then an additional 100 random ones for both the fine-tunings and adapters. However, many models fail to download or run due to various mismatches, such as missing weight tensors or hugging-face library nonsupport for older models. We therefore are left with much less models in each tree. We detail the number of models in each tree in Tab. 3.

Tree #Fine-tunes #Adapters Qwen-2.5 (3B) 331 148 Mistral (7B) 240 40 Qwen-2.5 (7B) 317 333 Llama-3.1 (8B) 320 297

- Table 3: Number Evaluated Models from Each Tree.
- Table 4: Our custom scheduler. Queries per-model for different rounds (s1 ...s5) across different budgets.

Queries-per-model s1 s2 s3 s4 s5

Budget

10 6 16 30 60 120 25 15 40 75 150 300 50 30 75 150 300 600 100 60 150 300 600 1200 200 120 300 600 1200 2400

Query Selection. To facilitate the evaluation of over 2,000 models, we subsample the RouterBench evaluation suite to a total of 2,500 queries and allocate the budget uniformly across the tasks (ARC-C, Winogrande, MMLU, MBPP, and GSM8K). Queries within each task are selected via random sampling to ensure an unbiased representation of the original benchmarks.

Model Evaluation. To ensure a fair comparison, we wish to evaluate all models under consistent conditions. However, we observed that some models might benefit from using their system prompt and some would not. Moreover, there could be inconsistencies within the same model across different tasks. To overcome these inconsistencies, we evaluated each model both with and without its

system prompt and, given a task, took the version with the best accuracy for it.

For reproducibility, we evaluate models by greedy decoding (do_sample=False). Lastly, we set a maximum response length of max_length=50 tokens at multiple-choice and short answer queries (ARC-Challenge, MMLU and Winogrande) which typically require only a single response token. In coding (MBPP) and math (GSM8K) queries, which may require longer answers or some thinking, we set a limit of max_length=512 tokens.

Scheduler. Driven by models’ skewed accuracy distributions(see App. F), we design a custom scheduler for efficient model discovery. First, as the vast majority of models are significantly worse than the best available option, we can eliminate them quickly. Therefore, we make an aggressive first elimination round of keeping a fixed number of just 100 models, which translates to approximately 20% of the model tree on average. However, it is critical we do not eliminate high-performing models in this round. Therefore, we also modify the query allocation schedule between rounds. Specifically, we allocate 60% of the budget to the first step, making it the most resource intensive. As we are working with highly constraint budgets of up to just 10 queries per model, this translates to as few as 6 evaluations. We then double the amount of queries per-model at each round. Using this scheduler, we avoid early eliminations of good models, while leaving a substantial amount of the budget to the remaining ones as their number quickly decreases. We detail our exact scheduler for different budgets at Tab. 4. Finally, Tab. 5 and Tab. 7 show this scheduler is highly effective, and greatly improves the accuracy and rank of the detected models.

### D Ablation Studies

We ablate the main two parts of our method: (i) correlated sampling and (ii) elimination schedule. We present the results in Tab. 5. For our modified scheduler, the results are clear: our scheduler can retrieve significantly better models under the same budget, focusing the search on the more “interesting” comparisons. Specifically, at a 10 queries (per-model) budget we observe an retrieval rank decrease of over 30 when averaged across trees using our custom schedule. However, this is less clear in the case of correlated sampling. In highly small budgets, we find both options to be comparable, as the evaluation is very noisy in the first place.

- Table 5: Ablation Studies. We ablate both parts of our approach. We report the mean rank and accuracy of the retrieved models for each model tree. Each experiment is repeated 100 times.

Queries Per Model

Method

Qwen-3B Qwen-7B Mistral-7B Llama-8B Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑

-

Random Selection 233.8 0.588 318.4 0.599 144.6 0.431 270.8 0.601 Best Base 56.5 0.716 30.0 0.783 62.0 0.556 41.0 0.713

10

Sequential Halving 69.3 0.714 62.9 0.777 15.2 0.684 75.2 0.710 Ours w. SH Scheduler 51.9 0.717 45.4 0.780 12.7 0.684 56.2 0.718 Ours w/o Corr. Sampling 12.6 0.725 10.4 0.788 2.8 0.694 11.1 0.725 Ours 11.3 0.726 15.8 0.786 2.6 0.694 15.4 0.725

50

Sequential Halving 41.0 0.719 29.6 0.784 7.7 0.690 29.9 0.720 Ours w. SH Scheduler 10.6 0.726 13.5 0.787 5.3 0.692 11.2 0.731 Ours w/o Corr. Sampling 5.4 0.728 5.5 0.789 2.0 0.695 6.5 0.729 Ours 3.5 0.729 3.6 0.790 1.6 0.695 3.0 0.736

Best Available 1.0 0.732 1.0 0.791 1.0 0.696 1.0 0.747

However, this changes as the budget increases: in a larger budget of 50 we can see that in all trees, correlated sampling achieves better results, allowing our approach to retrieve a top-3 available model.

E Additional Budgets Results

We evaluate our method’s robustness across three additional query budgets: 25, 100, and 200 queries per model. Results are presented in Tab. 7. As expected, increasing the budget size leads to saturated performance, where the baseline approaches also begin to consistently identify gem models that are significantly better than the best popular base versions. Nevertheless, our approach remains superior, often matching or surpassing the baselines performance while requiring only half the query budget. For instance, our method’s performance with a budget of 25 queries per-model exceeds that of all baselines given a doubled budget of 50. We observe a similar trend when comparing our 50 query results against the baselines 100 query results. Furthermore, under equal budget constraints, our approach consistently identifies superior models with higher average accuracies and lower average ranks.

- Table 6: Gem Documentation. We analyze which identified hidden gems include performance documentation.

### F Model Accuracy Distributions

We present the accuracy distributions across different models trees for math, coding and overall performances. For each tree and task we plot the cumulative histogram, and mark 2 important lines: one at −10% from the best gem model, and another at −5%. In all tasks and trees the vast majority of models are more than 10% worse than the best model, and thus can be easily detected and ignored.

### G Gem Documentation

We analyze whether identified hidden gems include performance documentation. Results are show in Tab. 6. We find that 19 out of the 24 gems has no performance documentation at all making them completely undetectable for text-based search. Moreover, out of the remaining 5, 3 of them are documentations targeted for South-Eastern Asian languages evaluation which are irrelevant for their gem tasks. This means only 2 out of the 24 gems could have been found by text search. Specifically, the first is Nvidia’s official fine-tuning of Llama for math evaluated on the same evaluation dataset and the other model is a Qwen-3B fine-tuning which reports Fortran code evaluation, where we looked for Python coding skills. As many different coding models and evaluations exist, it is likely that textbased search engines could not have known this model is also the best available for Python tasks, and would not return it as their prime candidate.

✓ indicates relevant documentation, ✗ indicates no documentation, and ● indicates documentation for irrelevant tasks only (e.g., multilingual scores for a math model).

Tree ARC-Cs WinoG.s MMLUs MBPPs GSM8Ks RouterB.s Qwen 3B ● ✗ ✗ ✓ ✗ ✗

### H Additional Atlas Visualizations

Mistral 7B ✗ ✗ ✗ ✗ ✗ ✗ Qwen 7B ✗ ✗ ✗ ✗ ✗ ✗ Llama 8B ✗ ✗ ● ✗ ✓ ●

In Fig. 5–8, we provide extended atlas visualizations for the selected model trees, color-coded by performance on GSM8Ks (math), MBPPs (cod-

ing), and RouterBenchs (overall). These views again show that “hidden gems” often possess limited download counts compared to the popular base versions, making them nearly invisible when observing the tree’s structure alone. Furthermore, gem models are often scattered across the tree rather than clustered in predictable branches. A notable exception is the Qwen-7B tree, where highperformers are nested within relevant official finetuning branches. For instance, the top coding models reside within the Qwen-7B Coder Instruct subtree. However, such curated official versions are not universally available and even when present, their sub-trees can be very large, making an exhaustive search computationally prohibitive.

- Table 7: Extended Model Discovery Results. We evaluate the top retrievals of each method. For each budget (10, 25, 50, 100, 200), we report the rank and accuracy of the retrieved models for each model tree. Each experiment is repeated 100 times, and mean results are reported.

Queries Qwen-3B Qwen-7B Mistral-7B Llama-8B Per Model Method Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑ Rank ↓ Acc. ↑

Random Selection 233.8 0.588 318.4 0.599 144.6 0.431 270.8 0.601 Best Base 56.5 0.716 30.0 0.783 62.0 0.556 41.0 0.713

-

Uniform 166.3 0.671 173.8 0.724 29.5 0.656 222.0 0.665 UCB 88.0 0.708 83.9 0.770 14.1 0.684 110.2 0.702 UCB-StdDev. 81.8 0.710 75.2 0.772 11.8 0.686 96.0 0.707

TTTS 92.6 0.708 75.7 0.772 14.0 0.685 103.6 0.702 Successive Rejects 94.4 0.706 128.3 0.752 25.6 0.662 171.5 0.683

10

BayesElim 56.4 0.717 58.9 0.778 13.5 0.685 79.9 0.708 UCB-E 78.4 0.710 83.7 0.769 13.6 0.684 106.5 0.703 Sequential Halving 69.3 0.714 62.9 0.777 15.2 0.684 75.2 0.710 Ours 11.3 0.726 15.8 0.786 2.6 0.694 15.4 0.725

Uniform 115.6 0.697 126.4 0.752 19.4 0.679 167.9 0.687 UCB 63.6 0.715 64.7 0.777 12.4 0.686 74.1 0.712 UCB-StdDev. 54.9 0.717 53.0 0.780 8.8 0.689 55.4 0.718

TTTS 76.4 0.712 62.5 0.777 15.3 0.684 96.2 0.705 Successive Rejects 88.6 0.707 76.7 0.773 19.1 0.678 122.3 0.699

25

BayesElim 47.2 0.719 43.9 0.782 9.8 0.689 56.3 0.715 UCB-E 55.0 0.717 58.4 0.777 7.6 0.690 63.7 0.715 Sequential Halving 52.4 0.717 35.7 0.783 11.6 0.687 50.9 0.717 Ours 5.3 0.728 12.3 0.787 1.9 0.695 8.0 0.730

Uniform 91.2 0.707 81.9 0.770 17.4 0.683 114.9 0.705 UCB 41.2 0.719 39.8 0.782 8.9 0.689 55.8 0.718 UCB-StdDev. 34.7 0.721 28.9 0.784 4.0 0.693 37.1 0.722

TTTS 53.1 0.717 51.7 0.780 10.8 0.687 70.6 0.712 Successive Rejects 83.1 0.710 78.4 0.773 16.5 0.682 92.0 0.707

50

BayesElim 30.0 0.721 31.0 0.784 6.7 0.691 34.4 0.721 UCB-E 37.4 0.720 33.1 0.783 4.3 0.693 43.4 0.720 Sequential Halving 41.0 0.719 29.6 0.784 7.7 0.690 29.9 0.720 Ours 3.5 0.729 3.6 0.790 1.6 0.695 3.0 0.736

Uniform 63.2 0.714 60.7 0.777 14.5 0.685 86.2 0.713 UCB 30.6 0.721 29.2 0.784 3.5 0.694 27.7 0.728 UCB-StdDev. 16.8 0.725 19.9 0.787 2.3 0.695 13.2 0.734

TTTS 47.2 0.719 35.5 0.783 5.6 0.692 46.6 0.718 Successive Rejects 69.3 0.714 57.3 0.779 13.8 0.685 63.3 0.713

100

BayesElim 10.9 0.726 16.5 0.786 3.4 0.694 11.6 0.729 UCB-E 12.4 0.726 14.7 0.787 2.8 0.694 13.2 0.735 Sequential Halving 17.9 0.724 16.1 0.786 3.9 0.693 14.8 0.726 Ours 2.1 0.731 2.0 0.791 1.4 0.695 1.9 0.740

Uniform 30.5 0.722 45.0 0.781 10.6 0.688 54.8 0.723 UCB 20.0 0.726 20.4 0.786 1.2 0.696 8.4 0.741 UCB-StdDev. 10.0 0.729 6.3 0.789 1.3 0.695 1.8 0.743

TTTS 23.4 0.723 30.0 0.784 1.2 0.696 16.1 0.734 Successive Rejects 52.4 0.717 52.0 0.780 7.3 0.690 52.2 0.716

200

BayesElim 4.7 0.728 5.5 0.789 2.0 0.695 4.0 0.738 UCB-E 5.7 0.730 4.2 0.790 2.0 0.695 1.9 0.743 Sequential Halving 6.2 0.728 7.3 0.788 2.3 0.695 4.3 0.739 Ours 1.4 0.731 1.7 0.791 1.1 0.696 1.3 0.744

Optimal 1.0 0.732 1.0 0.791 1.0 0.696 1.0 0.747

- 0.8
- 1

-5% from best 95.8% models

%ofmodels

0.6

-10% from best 69.1% of models

0.4

0.2

0

0 0.2 0.4 0.6 0.8

Accuracy

Qwen-3B

GSM8Ks

- 0.8
- 1

-5% from best 98.3% models

%ofmodels

0.6

-10% from best 92.3% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Accuracy

Qwen-3B

MBPPs

- 0.8
- 1

-5% from best 99.6% models

%ofmodels

0.6

-10% from best 68.5% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Accuracy

Qwen-3B

#### RouterBenchs

- 0.8
- 1

-5% from best 83.4% models

%ofmodels

0.6

-10% from best 69.4% of models

0.4

0.2

0

0 0.2 0.4 0.6 0.8

Accuracy

Qwen-7B

GSM8Ks

- 0.8
- 1

-5% from best 87.8% models

%ofmodels

0.6

-10% from best 74.8% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Accuracy

Qwen-7B

MBPPs

- 0.8
- 1

-5% from best 78.5% models

%ofmodels

0.6

-10% from best 62.9% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Accuracy

Qwen-7B

#### RouterBenchs

- 0.8
- 1

-5% from best 93.6% models

%ofmodels

0.6

-10% from best 87.5% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Accuracy

Mistral-7B

GSM8Ks

- 0.8
- 1

-5% from best 98.9% models

%ofmodels

0.6

-10% from best 92.1% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5

Accuracy

Mistral-7B

MBPPs

- 0.8
- 1

-5% from best 87.1% models

%ofmodels

0.6

-10% from best 84.6% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Accuracy

Mistral-7B

#### RouterBenchs

- 0.8
- 1

-5% from best 99.8% models

%ofmodels

0.6

-10% from best 96.4% of models

0.4

0.2

0

0 0.2 0.4 0.6 0.8

Accuracy

Llama3-8B

GSM8Ks

- 0.8
- 1

-5% from best 83.3% models

%ofmodels

0.6

-10% from best 70.8% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6

Accuracy

Llama3-8B

MBPPs

- 0.8
- 1

-5% from best 86.2% models

%ofmodels

0.6

-10% from best 60.3% of models

0.4

0.2

0

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Accuracy

Llama3-8B

#### RouterBenchs

Figure 4: Cumulative Accuracy Distributions across different model architectures and specialized tasks.

###### Table 8: Qwen-3B Hidden Gems.

Tree Task Best Found Performer Accuracy

Coding GiuLeo01/FortranCodeGen-3B-SynthData 73.3 Math watermelonhjg/Qwen2.5-3B-Instruct-EN-Zero 89.0 General watermelonhjg/Qwen2.5-3B-Instruct-EN-Zero 73.2

Qwen-3B

Qwen2.5 3B

Qwen2.5 3B

Most Popular 9M Downloads

Most Popular 9M Downloads

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Qwen2.5 3B Instruct

Qwen2.5 3B Instruct

[Figure 50]

Continuous Linear Color Scale (Values at every 20%)

[Figure 51]

Best Coder

+4.7% Coding 8 Downloads

73.2

| |
|---|

71.2

65.00 67.05 69.10 71.15 73.20

69.1

Root Full Fine-tune Adapter Hidden Gem !

Best Overall and in Math

[Figure 52]

Best Overall

[Figure 53]

67.0

+5.5% on GSM8K +1.6% on RouterBench 9 Downloads

+1.6% on RouterBench 9 Downloads

[Figure 54]

≤65.0

Value

#### Qwen-3B Model Tree RouterB.s Performance

Qwen2.5 3B

Qwen2.5 3B

[Figure 55]

Most Popular 9M Downloads

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Qwen2.5 3B Instruct

Qwen2.5 3B Instruct

[Figure 61]

Continuous Linear Color Scale (Values at every 20%)

Continuous Linear Color Scale (Values at every 20%)

[Figure 62]

Best Coder

+4.7% Coding 8 Downloads

89.0

73.3

| |
|---|

| |
|---|

81.8

66.2

60.00 67.25 74.50 81.75 89.00

45.00 52.08 59.15 66.22 73.30

74.5

59.1

Qwen2.5-3B Coder Instruct 20K Downloads

Best in Math

[Figure 63]

67.2

52.1

+5.5% on GSM8K 9 Downloads

≤60.0

≤45.0

Value

Value

#### GSM8Ks Performance MBPPs Performance

Figure 5: Different performance views for the Qwen-3B model tree.

###### Table 9: Llama3.1-8B Hidden Gems.

Tree Task Best Found Performer Accuracy

Coding plandes/sdoh-llama-3-1-8b 68.8 Math nvidia/OpenMath2-Llama3.1-8B 96.0 General aisingapore/Llama-SEA-LION-v3-8B-IT 74.7

Llama3.1-8B

[Figure 64]

Best in Math

[Figure 65]

+12.8% on GSM8K

[Figure 66]

Llama3.1 8B

Llama3.1 8B

Best Overall

Best Overall

+3.4% on RouterBench 2,800 Downloads

[Figure 67]

+3.4% on RouterBench

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Continuous Linear Color Scale (Values at every 20%)

[Figure 73]

[Figure 74]

Llama3.1 8B Instruct

Llama3.1 8B Instruct

74.7

[Figure 75]

| |
|---|

[Figure 76]

Best Coder

[Figure 77]

[Figure 78]

+3.2% on MBPP

[Figure 79]

71.0

60.00 63.67 67.35 71.03 74.70

67.3

Most Popular 5,000,000 Downloads

Most Popular 5,000,000 Downloads

Root Full Fine-tune Adapter Hidden Gem !

63.7

≤60.0

[Figure 80]

Value

#### Llama3-8B Model Tree RouterB.s Performance

[Figure 81]

Best in Math

[Figure 82]

+12.8% on GSM8K 930 Downloads

[Figure 83]

Llama3.1 8B

Llama3.1 8B

Llama3.1 8B

Continuous Linear Color Scale (Values at every 20%)

Continuous Linear Color Scale (Values at every 20%)

[Figure 84]

[Figure 85]

Llama3.1 8B

Llama3.1 8B Instruct

Llama3.1 8B Instruct

96.0

68.8

[Figure 86]

| |
|---|

| |
|---|

[Figure 87]

Best Coder:

[Figure 88]

[Figure 89]

+3.2% on MBPP 2 Downloads

[Figure 90]

87.0

64.1

60.00 69.00 78.00 87.00 96.00

50.00 54.70 59.40 64.10 68.80

78.0

59.4

Most Popular 5,000,000 Downloads

Most Popular 5,000,000 Downloads

69.0

54.7

≤60.0

≤50.0

Value

Value

#### GSM8Ks Performance MBPPs Performance

Figure 6: Different performance views for the Llama3-8B model tree.

###### Table 10: Mistral-7B Hidden Gems.

###### Tree Task Best Found Performer Accuracy

58.8

mlfoundations-dev/mistral_7b_0-3_ohdcft-v3.1-claude-3-5-sonnet-20241022

Coding

80.7

mlfoundations-dev/oh-mistral-bs512_lr5_00E-06_ schedulercosine_with_min_lr_warmup1_00E-01

Mistral-7B Math

69.6

mlfoundations-dev/mistral_7b_0-3_ohdcft-v3.1-gpt-4o-mini

General

Best in Math +40% on GSM8K

[Figure 91]

[Figure 92]

Mistral 7B

Mistral 7B

[Figure 93]

[Figure 94]

Best Coder

[Figure 95]

[Figure 96]

[Figure 97]

+11.8% on MBPP

[Figure 98]

Most Popular 1.2M Downloads

Most Popular 1.2M Downloads

Continuous Linear Color Scale (Values at every 20%)

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Best Overall

Best Overall

69.6

+14.0% on RouterBench 1 Download

[Figure 103]

+14.0% on RouterBench

| |
|---|

[Figure 104]

64.7

[Figure 105]

[Figure 106]

50.00 54.90 59.80 64.70 69.60

Mistral-7B Instruct

Mistral-7B Instruct

59.8

54.9

Root Full Fine-tune Adapter Hidden Gem !

≤50.0

Value

[Figure 107]

#### Mistral-7B Model Tree RouterB.s Performance

Best in Math

[Figure 108]

+40% on GSM8K 4 Downloads

[Figure 109]

Mistral 7B

Mistral 7B

[Figure 110]

[Figure 111]

Best Coder

[Figure 112]

+11.8% on MBPP 5 Downloads

[Figure 113]

Most Popular 1.2M Downloads

Most Popular 1.2M Downloads

Continuous Linear Color Scale (Values at every 20%)

Continuous Linear Color Scale (Values at every 20%)

[Figure 114]

[Figure 115]

80.7

58.8

| |
|---|

| |
|---|

75.5

49.1

[Figure 116]

[Figure 117]

60.00 65.17 70.35 75.53 80.70

20.00 29.70 39.40 49.10 58.80

Mistral-7B Instruct

Mistral-7B Instruct

70.3

39.4

65.2

29.7

≤60.0

≤20.0

Value

Value

#### GSM8Ks Performance MBPPs Performance

Figure 7: Different performance views for the Mistral-7B model tree.

###### Table 11: Qwen-7B Hidden Gems.

Tree Task Best Found Performer Accuracy

Coding mlx-community/Josiefied-Qwen2.5-Coder-7B-Instruct-abliterated-v1 82.1 Math zwhe99/DeepMath-Zero-Math-7B 92.1 General langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld 79.1

Qwen-7B

|Overall Performance<br><br>[Figure 118]<br><br>Best Overall<br><br>+0.8% 80 Monthly Downloads<br><br>Qwen-7B Instruct (Most Popular) 9,600,000 Monthly Downloads<br><br>[Figure 119]<br><br>|
|---|

Best Overall

[Figure 120]

|Math<br><br>[Figure 121]<br><br>Best in Math<br><br>+12.8% on GSM8K 47 Monthly Downloads<br><br>Qwen2.5 7B Math 96,000 Monthly Downloads<br><br>[Figure 122]<br><br>|
|---|

Qwen-7B

+0.8% 80 Monthly Downloads

[Figure 123]

[Figure 124]

[Figure 125]

Most Popular 9.6M Downloads

[Figure 126]

[Figure 127]

Qwen-7B Math

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Qwen-7B Instruct

Continuous Linear Color Scale (Values at every 20%)

[Figure 133]

[Figure 134]

[Figure 135]

Qwen-7B Coder Instruct

[Figure 136]

79.1

| |
|---|

74.3

60.00 64.78 69.55 74.32 79.10

69.5

Root Full Fine-tune Adapter Hidden Gem !

[Figure 137]

!

|Coding<br><br>[Figure 138]<br><br>[Figure 139]<br><br>Best Coder<br><br>+1.2% on MBPP 6 Monthly Downloads<br><br>Qwen-7B Coder Instruct 250,000 Monthly Downloads<br><br>[Figure 140]<br><br>|
|---|

64.8

[Figure 141]

≤60.0

Value

#### Qwen-7B Model Tree RouterB.s Performance

Qwen-7B

Qwen-7B

Qwen2.5 7B Math 96,000 Downloads

[Figure 142]

Best in Math

[Figure 143]

[Figure 144]

[Figure 145]

Qwen-7B Math

+12.8% on GSM8K 50 Downloads

[Figure 146]

Qwen-7B Instruct

Qwen-7B Instruct

Continuous Linear Color Scale (Values at every 20%)

Continuous Linear Color Scale (Values at every 20%)

[Figure 147]

Qwen-7B Coder Instruct

[Figure 148]

92.1

82.1

| |
|---|

| |
|---|

84.1

77.8

60.00 68.03 76.05 84.07 92.10

65.00 69.28 73.55 77.82 82.10

76.0

73.5

[Figure 149]

!

[Figure 150]

68.0

69.3

Best Coder

[Figure 151]

Qwen-7B Coder Instruct 250,000 Monthly Downloads

+1.2% on MBPP 6 Monthly Downloads

≤60.0

≤65.0

Value

Value

#### GSM8Ks Performance MBPPs Performance

Figure 8: Different performance views for the Qwen-7B model tree.

