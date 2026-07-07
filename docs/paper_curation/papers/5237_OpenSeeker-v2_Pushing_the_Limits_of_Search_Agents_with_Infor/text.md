# arXiv:2605.04036v1[cs.AI]5May2026

## OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories

Yuwen Du1,*, Rui Ye1,*,#,†, Shuo Tang1, Keduan Huang1, Xinyu Zhu1, Yuzhu Cai1, Siheng Chen1,† 1Shanghai Jiao Tong University, *Equal Core Contributions, #Project Lead †Corresponding Authors: yr991129@sjtu.edu.cn, sihengc@sjtu.edu.cn

#### Abstract

Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet their development remains dominated by industrial giants. The typical industry recipe involves a highly resource-intensive pipeline spanning pre-training, continual pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL). In this report, we show that when fueled with informative and high-difficulty trajectories, a simple SFT approach could be surprisingly powerful for training frontier search agents. By introducing three simple data synthesis modifications: scaling knowledge graph size for richer exploration, expanding the tool set size for broader functionality, and strict low-step filtering, we establish a stronger baseline. Trained on merely 10.6k data points, our OpenSeeker-v2 achieves state-of-the-art performance across 4 benchmarks (30B-sized agents with ReAct paradigm): 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity’s Last Exam, and 78.0% on xbench, surpassing even Tongyi DeepResearch trained with heavy CPT+SFT+RL pipeline, which achieves 43.4%, 46.7%, 32.9%, and 75.0%, respectively. Notably, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm to be developed by a purely academic team using only SFT. We are excited to open-source the OpenSeeker-v2 model weights and share our simple yet effective findings to make frontier search agent research more accessible to the community.

[Figure 1]

Code https://github.com/PolarSeeker/OpenSeeker Model https://huggingface.co/PolarSeeker/OpenSeeker-v2-30B-SFT

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

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

Figure 1: OpenSeeker-v2 achieves state-of-the-art performance within its model scale and paradigm on four representative benchmarks, remarkably accomplishing this via simple SFT and outperforming Tongyi DeepResearch that is trained via extensive continual pre-training, SFT, and RL.

Preprint. Technical Report.

### 1 Introduction

In the era of information explosion, deep search has emerged as a non-negotiable competency for frontier Large Language Model (LLM) agents (OpenAI, 2025a). However, the development of these highperformance agents has long remained a "closed-door game" played almost exclusively by well-funded corporate entities (OpenAI, 2025b; Anthropic, 2025). The typical industry recipe to achieve state-of-theart (SOTA) performance is highly resource-intensive, typically involving Continual Pre-Training (CPT) on massive corpora (Team et al., 2025b, 2026; Chu et al., 2026), followed by Supervised Fine-Tuning (SFT) (Ye et al., 2025), and culminating in complex Reinforcement Learning (RL) stages (Li et al., 2025). This heavy reliance on immense compute and proprietary data pipelines has created a massive barrier, fundamentally hindering the academic and open-source communities from innovating within this domain.

We challenge this prevailing reliance on complex, multi-stage training pipelines. Building upon our initial exploration in OpenSeeker (Du et al., 2026), we shift the focus entirely back to the quality of the training trajectories themselves and ask a crucial question: can we push the limits of search agents and rival the performance of heavy industrial pipelines using only a straightforward SFT approach?

In this report, we introduce OpenSeeker-v2, an upgraded search agent that proves a straightforward SFT approach could be sufficiently powerful when fueled by high-quality data of high difficulty and richness. Specifically, we introduce two simple yet highly effective modifications to our data synthesis pipeline: (1) Scaling graph size for richer exploration: We significantly expand the topological graph size during data generation. This expansion injects a much richer and more diverse set of source information into the context, enabling the synthesis of highly complex tasks that structurally mandate deep, multi-hop exploration to solve. (2) Expanding the tool set for broader functionality: We increase the number of available tools, allowing the agent to learn more versatile strategies and handle a wider variety of queries. (3) Strict low-step filtering: We filter out any trajectory that can be resolved in too few tool-call steps. By intentionally dropping these simple queries, we guarantee a strict minimum difficulty floor for the training set, forcing the agent to learn sustained reasoning and information seeking over long horizons.

By applying these two strategies, we curate a highly condensed dataset of merely 10k high-difficulty trajectories. Strikingly, training a 30B parameter model on this small dataset via a single SFT run yields surprising results. OpenSeeker-v2 achieves a new SOTA1 across four representative agentic benchmarks: 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity’s Last Exam, and 78.0% on xbench. Notably, this simple SFT baseline decisively outperforms prominent industrial models such as Tongyi DeepResearch, which relies on an extensive CPT+SFT+RL pipeline and achieves 43.4%, 46.7%, 32.9%, and 75.0%, respectively.

Ultimately, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm (ReAct) to be developed entirely by a purely academic team using only SFT. To democratize frontier search agent research and provide an easily reproducible baseline for the community, we are excited to fully open-source the OpenSeeker-v2 model weights.

### 2 Methodology and Results

#### 2.1 Methodology

We introduce OpenSeeker-v2, an upgraded search-agent training framework based on supervised finetuning (SFT). Our central hypothesis is that, given sufficiently difficult and information-rich training data, a straightforward SFT objective is enough to induce strong long-horizon search and reasoning abilities.

1While some works focus on context management (Ye et al., 2025; Team et al., 2026), our work focuses on general ReAct-based paradigm with emphasis on data quality.

Scaling graph size for richer exploration. Let G = (V,E) denote the source graph used for task synthesis. For each seed node vseed ∈ V, the original pipeline constructs a local subgraph Gsub around vseed. In OpenSeeker-v2, we increase the expansion budget from k to K, where K > k, and obtain a larger evidence subgraph:

##### Gsub(K) = Expand(G,vseed,K).

The enlarged subgraph contains a richer set of topologically related sources, which increases the number and diversity of feasible reasoning paths. A synthetic query is then generated conditioned on this expanded context:

q ∼ Pgen q | Gsub(K) . By scaling K, the generated question is more likely to require evidence aggregation over multiple nodes rather than relying on few source.

Expanding the tool set for broader functionality. Given a generated question q, we equip the search agent with an expanded set of tools A larger than that used in OpenSeeker-v1 (Du et al., 2026) following Team et al. (2026) and let it produce a multi-step ReAct-style trajectory:

##### τ = (r1,a1,o1,r2,a2,o2,...,rT,aT,oT,rT+1,y),

where each action at ∈ A corresponds to a tool call selected from the enlarged tool set, and ot denotes the observation returned by the invoked tool. rt represents the reasoning trace before each action. The trajectory consists of T tool-call steps, followed by a final reasoning step rT+1 and the answer y. By expanding A, the agent is encouraged to learn more diverse interaction patterns and leverage complementary tools, resulting in more flexible and functionally rich problem-solving behaviors.

Strict low-step filtering. To remove overly simple instances, we apply a strict low-step filtering rule:

##### Dv2 = {(q,τ) ∈ Draw | T(τ) ≥ Tmin}.

Here, Tmin is a predefined minimum tool-call threshold. Trajectories with T(τ) < Tmin are discarded because they can often be solved by direct lookup or shallow keyword matching.

Finally, OpenSeeker-v2 trains the search agent with a standard SFT objective over the filtered dataset.

The expanded graph increases contextual richness and multi-hop dependency, while low-step filtering enforces a minimum difficulty floor. Together, these two modifications produce high-quality SFT data that encourages the agent to learn sustained reasoning, robust information extraction, and long-horizon search behavior.

#### 2.2 Experimental Setup

Implementation. We instantiate OpenSeeker-v2 from Qwen3-30B-A3B-Thinking-2507 (Team, 2025), which has 30B total parameters and 3B activated parameters during inference. The agent uses a 256k context window and allows up to 200 tool calls per trajectory. OpenSeeker-v2 is trained with SFT, without RL or additional hyperparameter tuning.

Benchmarks. We evaluate OpenSeeker-v2 on five challenging agentic benchmarks: BrowseComp (Wei et al., 2025), BrowseComp-ZH (Zhou et al., 2025), Humanity’s Last Exam (HLE) (Phan et al., 2025), and xbench-DeepSearch (Xbench-Team, 2025). These benchmarks cover diverse deep research tasks. We mask the hugging-face-related links when calling the web search tools to avoid potential leakage.

Baselines. We compare OpenSeeker-v2 with representative systems in Table 1, with a primary focus on comparable-scale ReAct-based search agents. Tongyi DeepResearch (Team et al., 2025b) and RedSearcher (Chu et al., 2026) are strong 30B-scale search agents trained with heavier CPT+SFT+RL pipelines. They provide direct references for evaluating whether our SFT-only approach can rival more resourceintensive training recipes. For completeness, we also include closed-source proprietary models (Anthropic,

Table 1: Comparisons among our OpenSeeker and other ReAct-based search agents. ‘# Samples’ denotes the number of total training data samples; ‘Training’ denotes training techniques (CPT: continual pretraining, SFT: supervised fine-tuning, RL: reinforcement learning); ‘Academic’ denotes whether conducted by pure academic team (✓: Yes, ×: No); ‘BC-ZH’ denotes BrowseComp-ZH. Notably, with simple SFT only, OpenSeeker-v2-30B-SFT consistently outperforms models of comparable scale trained with more complex pipelines involving CPT, SFT, and RL. OpenSeeker-v2 comprehensively outperforms pure ReAct-based models of comparable scale.

Model Name # Samples Training Academic BrowseComp BC-ZH HLE xbench

Closed-Source Proprietary Models Claude-4-Opus ? ? × 18.8 37.4 - Claude-4.5-Sonnet ? ? × 24.1 42.4 32.0 Gemini-3-pro ? ? × 37.8 66.8 45.8 OpenAI-o3 ? ? × 49.1 68.7 20.2 65.0 OpenAI Deep Research ? ? × 51.5 42.9 26.6 GPT-5-High ? ? × 54.9 63.0 41.7 -

###### Open-Source Models > 30B

- DeepSeek-V3.1-671B ? ? × 30.0 49.2 29.8 71.2
- DeepSeek-V3.2-671B ? ? × 51.4 65.0 40.8 -

- GLM-4.6-357B ? ? × 45.1 49.5 30.4 -
- GLM-4.7-357B ? ? × 52.0 66.6 42.8 Minimax-M2-230B ? ? × 44.0 48.5 - -

∼30B Models WebSailor-V2-30B-SFT ? SFT × 24.4 28.3 23.9 61.7 WebSailor-V2-30B-RL ? SFT + RL × 35.3 44.1 30.6 73.7 WebLeaper-30B-SFT 15 k SFT × 27.7 - - 66.0 WebLeaper-30B-RL ? RL × 38.8 - - 72.0 Tongyi DeepResearch ? CPT + SFT + RL × 43.4 46.7 32.9 75.0 RedSearcher-30B ? CPT + SFT + RL × 42.1 49.8 34.3 OpenSeeker-v1-30B-SFT 11.7 k SFT ✓ 29.5 48.4 - 74.0 OpenSeeker-v2-30B-SFT 10.6 k SFT ✓ 46.0 58.1 34.6 78.0

2025; OpenAI, 2025b,a; Singh et al., 2025) and large open-source models (DeepSeek-AI et al., 2025; Team et al., 2025a; MiniMax AI Team, 2025) as broader reference points. Baseline results are taken from their technical reports or public leaderboards.

#### 2.3 Main Results

Surpassing comparable-scale agents trained with heavier pipelines. The central question behind OpenSeeker-v2 is whether simple SFT can push the limits of search agents and rival heavier industrial pipelines. As shown in Table 1, OpenSeeker-v2-30B-SFT achieves the strongest overall performance among ∼30B ReAct-based search agents while using SFT only. OpenSeeker-v2 achieves 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity’s Last Exam, and 78.0% on xbench. (1) Notably, with simple SFT, OpenSeeker-v2 outperforms Tongyi DeepResearch developed by Alibaba Tongyi Lab (Team et al., 2025b) and RedSearcher developed by RedNote, which are trained by the extensive CPT+SFT+RL pipeline Specifically, on the challenging benchmarks BrowseComp and HLE, OpenSeekerv2 outperforms these two by at least 2.6% and 0.3%, respectively; while on the BrowseComp-ZH and xbench, OpenSeeker-v2 significantly outperforms Tongyi DeepResearch by 11.4% and 3%, respectively. (2) Comparing with larger models, OpenSeeker-v2 also outperforms DeepSeek-V3.1-671B, GLM-4.6-357B, Minimax-M2-230B, Claude-4.5-Sonnet, indicating its strong capability. These results demonstrate that a straightforward SFT approach can be sufficiently powerful when fueled by high-quality data of high

difficulty and richness, suggesting that data quality could be a critical path towards training intelligent long-horizon search agents.

Demonstrating the scaling potential of OpenSeeker. OpenSeeker-v2 substantially improves upon OpenSeeker-v1 (Du et al., 2026) under the same model scale and SFT-only training recipe, highlighting the development potential of the OpenSeeker framework through higher-quality data construction. OpenSeekerv2 raises BrowseComp from 29.5 to 46.0, BrowseComp-ZH from 48.4 to 58.1, xbench from 74.0 to 78.0. These gains suggest that OpenSeeker has not yet saturated under the current SFT setting. More importantly, they show that increasing the difficulty and richness of synthesized QA tasks and enhancing the overall quality of synthesized trajectories can lead to substantial capability gains, indicating that scalable high-quality data synthesis is a promising path for further advancing search agents.

OpenSeeker-v2 demonstrates higher data difficulty than prior counterparts. OpenSeeker-v2 is built upon substantially longer search trajectories, with an average of 64.67 steps per trajectory, compared with 46.97 for OpenSeeker-v1 and 36.01 for RedSearcher. This suggests that the OpenSeeker-v2 training data requires more complex multi-step reasoning and longerhorizon information seeking. We hypothesize that such long and difficult synthetic trajectories are crucial for enabling the model to acquire stronger longhorizon retrieval and search capabilities, which further explains the superior performance of OpenSeeker-v2 on challenging deep-research benchmarks.

###### Tool Call Distribution

REDSearcher OpenSeeker-v1 OpenSeeker-v2 REDSearcher (avg)

0.12

| |
|---|

SampleProportion

| |
|---|

0.10

OpenSeeker-v1 (avg) OpenSeeker-v2 (avg)

0.08

0.06

0.04

0.02

0.00

0 25 50 75 100 125 150 175 200

Number of Tool Calls

Figure 2: Comparison of average tool call counts across search-agent training data.

### 3 Conclusion

In this report, we share that when fueled by high-quality data of high-difficulty and richness, a search agent trained with simple SFT could rival the performance of agents trained with extensive resources. Specifically, we share three simple yet effective modifications on the data collection pipeline: scaling graph size, expanding tool set, and low-step filtering, and train our final search agent: OpenSeeker-v2. Though trained with only 10.6k samples, OpenSeeker-v2 achieves a new SOTA across four representative benchmarks: 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity’s Last Exam, and 78.0% on xbench, significantly outperforms Tongyi DeepResearch and RedSearcher that are extensively trained via CPT, SFT, and RL. Our report highlight the critical role of data quality, suggesting that carefully designed data alone can unlock substantial performance gains.

What’s next. Our internal observations suggest strong scaling potential of high-quality synthesized data. Moving forward, we will continue to push in this direction by scaling up data quantity, quality, and diversity, with the goal of further pushing the limits of search agents.

### References

Anthropic. Introducing claude 4, 2025. URL https://www.anthropic.com/news/claude-4. Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao,

Cheng Xiang, Shengchao Hu, et al. Redsearcher: A scalable and cost-efficient framework for longhorizon search agents. arXiv preprint arXiv:2602.14234, 2026.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun

Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, et al. Deepseek-v3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, December 2025. doi: 10.48550/arXiv.2512.02556. URL https://arxiv.org/abs/2512.02556.

Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, and Siheng Chen. Openseeker: Democratizing frontier search agents by fully open-sourcing training data. arXiv preprint arXiv:2603.15594, 2026.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, et al. Websailor-v2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning. arXiv preprint arXiv:2509.13305, 2025.

MiniMax AI Team. MiniMax M2 & Agent: Ingenious in Simplicity, October 2025. URL https:// www.minimax.io/news/minimax-m2. Open-sourced model weights on Hugging Face: https: //huggingface.co/MiniMaxAI/MiniMax-M2.

OpenAI. Deep research system card, 2025a. URL https://cdn.openai.com/ deep-research-system-card.pdf.

OpenAI. Introducing openai o3 and o4-mini, 2025b. URL https://openai.com/index/ introducing-o3-and-o4-mini/.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, Alex Makelov, Alex Neitz, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, December 2025. doi: 10.48550/arXiv.2601.03267. URL https://arxiv.org/abs/2601.03267.

GLM Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jianyang Yan, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai, Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shijie Li, Shuang Li, Shuang-Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao,

Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zihan Wang, Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025a. URL https://arxiv.org/abs/2508.06471.

MiroMind Team, S Bai, L Bing, L Lei, R Li, X Li, X Lin, E Min, L Su, B Wang, et al. Mirothinker-1.7 & h1: Towards heavy-duty research agents via verification. arXiv preprint arXiv:2603.15726, 2026.

Qwen Team. Qwen3-30b-a3b-thinking-2507, 2025. URL https://huggingface.co/Qwen/ Qwen3-30B-A3B-Thinking-2507.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025b.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

Xbench-Team. Xbench-deepsearch, 2025. URL https://xbench.org/agi/aisearch. Rui Ye, Zhongwang Zhang, Kuan Li, Huifeng Yin, Zhengwei Tao, Yida Zhao, Liangcai Su, Liwen Zhang,

Zile Qiao, Xinyu Wang, et al. Agentfold: Long-horizon web agents with proactive context management. arXiv preprint arXiv:2510.24699, 2025.

Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang, Yifan Shao, Qichen Ye, Dading Chong, Zhiling Jin, Chenxuan Xie, Meng Cao, et al. Browsecomp-zh: Benchmarking web browsing ability of large language models in chinese. arXiv preprint arXiv:2504.19314, 2025.

