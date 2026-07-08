## SWE-chat: Coding Agent Interactions From Real Users in the Wild

Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang∗ & Sanmi Koyejo∗ Stanford University {baumann,diyiy,sanmi}@cs.stanford.edu

Data Website Code

[Figure 1]

[Figure 2]

[Figure 3]

# arXiv:2604.20779v1[cs.AI]22Apr2026

#### Abstract

AI coding agents are being adopted at scale, yet we lack empirical evidence on how people actually use them and how much of their output is useful in practice. We present SWE-chat, the first large-scale dataset of real coding agent sessions collected from open-source developers in the wild. The dataset currently contains 6,000 sessions, comprising more than 63,000 user prompts and 355,000 agent tool calls. SWE-chat is a living dataset; our collection pipeline automatically and continually discovers and processes sessions from public repositories. Leveraging SWE-chat, we provide an initial empirical characterization of real-world coding agent usage and failure modes. We find that coding patterns are bimodal: in 41% of sessions, agents author virtually all committed code (“vibe coding”), while in 23%, humans write all code themselves. Despite rapidly improving capabilities, coding agents remain inefficient in natural settings. Just 44% of all agent-produced code survives into user commits, and agent-written code introduces more security vulnerabilities than code authored by humans. Furthermore, users push back against agent outputs—through corrections, failure reports, and interruptions—in 44% of all turns. By capturing complete interaction traces with human vs. agent code authorship attribution, SWE-chat provides an empirical foundation for moving beyond curated benchmarks towards an evidence-based understanding of how AI agents perform in real developer workflows.

We collect coding agent interaction logs in the wild:

Our data collection combines agent session logs with git data versioning:

SWE-chat dataset statistics:

[Figure 4]

|User<br><br>starts a new coding agent session<br><br>󰠁| | |
|---|---|---|
| | | |
| | | |

|Agent explains / writes / edits some code<br><br>🤖|
|---|

|User or agent commits the changes<br><br>🤖 󰠁|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

terminalbackground

205 public GitHub repos 6K coding agent sessions 13K checkpoints 63K user prompts 355K tool calls

1.

2. 3.

### ?

Entire.io CLI installs git hooks

Full transcript is captured

Automatic sync of agent log and git diff

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

prompts traces tool calls code attribution

Figure 1: We present SWE-chat, a continually growing dataset of real human-coding agent interactions collected from public GitHub repositories. Developers opt in via installing Entire.io, an open-source tool that automatically logs coding agent sessions and links them to code commits with line-level human vs. agent attribution. As of April 2026, SWE-chat contains 2.7M logged events from 200+ repositories, including 63,000+ user prompts and 355,000+ tool calls.

∗Equal advising.

###### RQ1: Human-AI coding agent interaction patterns

- Sec. 3.1: agent use cases beyond code generation

RQ2: Agent failures and how users respond

- ● 41% of sessions are "vibe coding” (i.e., agents author >99% of code)
- ● Only 23% still write all code themselves

- Sec. 3.2: vibe coding is trending

- ● Average session success = 82%
- ● Session fail when users interrupt, system errors block the agent, …

- Sec. 4.1: many, though not all, sessions are successful

- ● <50% of AI-authored code survives
- ● Vibe coding is less efficient and less safe than human-AI collaborative coding

- Sec. 4.2 & 4.3: vibe coding is costly, slow and unsafe

- ● Users often provide very nitpicky (40%) and sometimes vague (33%) instructions
- ● Less ‘mind changing’ during vibe coding

- Sec. 3.3: user behavior varies during agent interactions

[Figure 13]

- ● Understanding code most common intent
- ● Agents mostly execute bash commands (33%) and read/edit/search files (48%)

[Figure 14]

[Figure 15]

[Figure 16]

Sec. 4.4: Agents rarely stop, but users often push back

[Figure 17]

- ● Users interrupt in 5% and push back in 41% of turns
- ● Agents rarely stop to ask user a question

[Figure 18]

Figure 2: Usage patterns and failure modes in SWE-chat. Using the SWE-chat dataset, we analyze how people use coding agents in the wild (left) and when and how they fail (right). Text colorings correspond to figure components. Results reflect the SWE-chat population of open-source developers using public repositories and opting into session logging.

#### 1 Introduction

AI coding agents have taken the world by storm. Enhancing Large Language Models (LLMs) with a simple set of actions for interacting with a coding environment autonomously—socalled tool calls for editing files, executing terminal commands, and invoking subagents—has accelerated their ability to complete long and difficult programming tasks (Yang et al., 2024a). Lately, AI agents are reported to succeed on 50% of coding tasks that humans take 12 hours to complete (METR, 2026; Kwa et al., 2025). As a result, developers increasingly delegate coding to agents (Mürtz and Müller, 2025; Anthropic, 2026), with unprecedented impacts on the global workforce (Peng et al., 2023; Demirci et al., 2025; Massenkoff et al., 2026).

Despite massive adoption, our understanding of how humans and AI coding agents interact remains largely anecdotal. While recent work has begun evaluating code completion models in realistic settings (Chi et al., 2025), no comparable effort exists for full agentic coding sessions. No public dataset captures how developers prompt, steer, override, and ultimately commit (or discard) agent-produced code. When it comes to software engineering (SWE) tasks, most AI benchmarks consist of a fairly limited set of curated problems with well-defined, verifiable solutions (Jimenez et al., 2024; Yang et al., 2024b; Deng et al., 2026; Kottamasu et al., 2026). Even more recent benchmarks fixate on task difficulty (Merrill et al., 2026), but still neglect the human-agent interaction dimension (Wang et al., 2026b). But strong performance on curated GitHub issues with meticulous instructions does not translate to real-world, iterative usage (Pan et al., 2025; Wang et al., 2026a). There is growing recognition that the next frontier lies in evaluating agents on the collaborative workflows that characterize actual development (Patwardhan et al., 2025; Cursor Research Team, 2026; Anthropic, 2025; 2026). Understanding how developers use coding agents in practice is a prerequisite for building genuinely helpful agents. Collecting real usage data in the wild is the only way to close this gap:

- RQ1 How do users interact with coding agents in real-world coding tasks?

Coding agents are increasingly deployed as autonomous problem solvers, even though we have no empirical evidence on how much of their output developers actually use, how often they fail, or how users cope when they do.

- RQ2 How do coding agents fail in practice, and how do users respond?

- Table 1: Comparison of SWE-chat with existing AI agent datasets. SWE-chat is the first dataset combining real user interactions with coding agent trajectories and rich contextual information, including detailed code authorship attribution.

Agent tool-use trajectories

Code diffs

Code attribution

Human prompts

Dataset

SWE-smith-trajectories (Yang et al., 2025) ✗ ✓ ✓ ✗ CoderForge-Preview (Ariyak et al., 2026) ✗ ✓ ✓ ✗ SERA (Shen et al., 2026) ✗ ✓ ✓ ✗ nex-agi-agent-sft (Cai et al., 2025) ✗ ✓ ✓ ✗ SWE-rebench-openhands-trajectories (Trofimova et al., 2025) ✗ ✓ ✓ ✗ Agent Trajectories (Bouzenia and Pradel, 2025) ✗ ✓ ✓ ✗ Multi-SWE-bench_trajs (Zan et al., 2025) ✗ ✓ ✓ ✗ Agent Data Protocol (Song et al., 2025) ✗ ✓ ✓ ✗ AIDev (Li et al., 2025) ✗ ✗ ✓ ✗ AgentPack (Zi et al., 2025) ✗ ✗ ✓ ✗

SWE-chat (ours) ✓ ✓ ✓ ✓

##### 1.1 Our contributions

We present SWE-chat, the first large-scale dataset of real coding agent sessions from actual users on real repositories (Figure 1). SWE-chat includes complete interaction traces between humans and AI coding agents, with full tool-call trajectories and code diffs with human vs. agent authorship attribution (Table 1). This enables researchers to study not just what code agents produce, but how users prompt, steer, and override them. We describe the data collection pipeline and aggregate statistics in Section 2.

Using SWE-chat, we contribute an initial sweep of empirical insights from real-world coding agent usage, summarized in Figure 2. Our analysis of interaction behavior in Section 3 (addressing RQ1) reveals that humans rely on coding agents for a broad range of tasks beyond writing patches to fix bugs or implement features: Understanding existing code is the most common user intent, and agents spend a third of their tool calls executing bash commands rather than editing files (Figures 19a and 19b). This suggests that benchmarks focused narrowly on patch generation underestimate the operational diversity and complexity of real agent workflows. Users’ coding mode is extremely bimodal: in most sessions, the AI agent either writes none or all of the code (Figure 5). But despite the emerging trend toward vibe coding (Figure 25), fully autonomous one-shot problem-solving remains far from reality. In fact, interactions typically span multiple turns, and users are often very nitpicky about what they want an agent to do and how they want it done (Figures 4 and 24).

Our analysis of failure modes and user responses in Section 4 (addressing RQ2) reveals lots of room for improvement. We identify sessions with a low success rating, revealing cases where agents fail to complete the user requests appropriately (Figure 6). In addition to that, we find that less than half of all agent-produced code survives into user commits

- (Table 3). Vibe coding is particularly inefficient, consuming roughly 3× more tokens and dollars per committed line than collaborative coding (Figures 7 and 29). Vibe-coded code is also substantially less safe. It introduces roughly 9× more security vulnerabilities per committed line than code that humans write themselves and about 5× more than code they co-author with the agent (Table 4). Agents are working autonomously for longer—the 99.9th-percentile turn duration now exceeds 100 minutes—yet they rarely stop to ask users for clarification (Figure 30). Users compensate by interrupting agents in 5% of turns and by pushing back against agent outputs in 39% of turns, often providing corrections and failure reports (Figure 8).

In Section 5.1, we outline a roadmap of how SWE-chat can help close some of these gaps—be it through realistic benchmarks, better interaction designs, or open-source user simulators evaluated on real session data.

#### 2 SWE-chat

##### 2.1 Data collection

We build the dataset from public GitHub repositories whose developers have opted into Entire.io’s CLI checkpoint logging, which records coding agent session transcripts on a dedicated branch. Each checkpoint is linked to a commit with line-level code authorship attribution. When enabled by the developer, Entire automatically records session transcripts for various coding agents (Claude Code, OpenCode, Gemini CLI, Cursor, and Factory AI Droid). These session logs capture user prompts, agent responses, tool calls (file edits, shell commands, code searches, etc.), and token usage. We provide more details on the data collection pipeline and its rapid growth trajectory in Appendix C.1.

The resulting SWE-chat dataset provides a comprehensive look into real-world humanagent collaboration, comprising almost 6,000 coding sessions across more than 200 repositories (Figure 1). At the time of writing, the data includes more than 13,000 checkpoints, 63,000 user prompts, and 355,000 agent tool calls. The full dataset contains 2.7 million logged events—these also include streamed progress events, return values from tool calls, and a small set of reasoning traces from 200 sessions with extended thinking. This trend is clearly visible in the steep trajectory shown in Figure 1. We plan to update our Website and Data frequently as we continue to collect new data. An example SWE-chat session is shown in Figure 3 to illustrate the session structure. Because SWE-chat captures only developers who actively opt into Entire’s public checkpoint logging, the dataset reflects an early-adopter population and may not generalize to all coding agent users; we discuss this and other limitations in Appendix A.

[Figure 19]

prompt

turn1turn2

agent tool calls

agent response

session

Figure 3: Structure of a coding agent session in SWE-chat. Each session consists of alternating user prompts and agent responses with tool calls (file reads, edits, shell commands) and text output.

##### 2.2 Data statistics

Agent answers a clarification question

Where should I create the test repo?

Simplify Go plugin to JS-only (55 tool calls)

(no ﬁle reads / edits needed)

(reads .go ﬁle)

Implement new feature, then debug errors

Remove all traces of `task_timeout`

My python trading bot won’t exit on Ctrl+C

(106 tool calls total)

(ﬁle read + grep → ﬁle edits)

(grep `_state.*=|ThreadSafeState`)

Docker compose deployment, but then undo release (106 tool calls total)

Implement plugin from the plan document

Add a new copy icon buton

(plan → read → implement → ﬁx)

(edit .css)

[Figure 20]

###### Figure 4: SWE-chat user-agent interaction statistics. (a) Distribution of turns per session. (b) Distribution of agent tool calls per turn. (c) Top 15 file types touched by agent tool calls.

SWE-chat consists of multi-turn coding agent sessions, collected from hundreds of real users in the wild (Figure 4)1, interacting with five widely used coding agents.2 Agents often make multiple tool calls for any user request (Figure 4b) and interact with a wide variety of programming languages, as reflected by the file types touched during sessions (Figure 4c). We present more detailed dataset statistics in Appendix D.1 and explore task topic distributions in Appendix D.2.

##### 2.3 Data analysis methodology

The true value of SWE-chat lies in unlocking an understanding of complex human-agent behaviors at scale, going beyond aggregate statistics to characterize how developers interact with coding agents in the long tail and why sessions succeed or fail. To facilitate this, we enrich the dataset with annotations that provide signal for both researchers studying humanAI collaboration (RQ1) and model developers seeking to build more helpful agents (RQ2). We classify sessions and user prompts using the annotation rubrics listed in Table 2, each designed to capture a specific dimension of real-world agent usage.

We developed clear annotation codebooks for each task and evaluated inter-annotator agreement, which was moderate to high across all tasks (see Appendix E for details). We rely on LLM judges to annotate the full dataset. It is important to note that LLMs can make mistakes and are thus not reliable data annotators (Baumann et al., 2025). However, we chose this approach for its scalability, enabling continuous annotation as new data is collected. For each task, we evaluated the zero-shot performance of various open-weight and proprietary LLMs using multiple prompt paraphrases against human expert gold labels, and then annotated the full dataset with the best-performing model and prompt. We describe the full LLM-as-a-judge validation approach in Appendix E.1.

Additionally, we leverage rich information from raw session logs and code attribution data, which capture all agent events—what tools they call, how much code they produce, and how long they take. To quantify how efficiently they do it, we define a suite of metrics (detailed in Appendix C.2) that quantify the fraction of agent-produced code that survives into user commits (code survival rate), the overhead of agent self-rewrites (coding efficiency), and the tokens, cost, time, and user effort required per committed line of code. To assess code safety, we additionally run the static-analysis tool Semgrep3 on the pre- and post-commit snapshots of each committed change and count the security findings introduced by the commit. This lets us compare the rate of introduced vulnerabilities per committed line across coding modes (see Section 4.3 and details in Appendix D.5). These metrics allow us to answer RQ2 by revealing where agents waste effort and where their output falls short of what developers actually commit.

#### 3 How do humans interact with coding agents in the wild? (RQ1)

##### 3.1 Task types: agents assist with a broad range of tasks beyond writing code

User requests are diverse Figure 19a illustrates the distribution of user intents. While a large portion of prompts (26.6%) falls into a broad “other” category, the most common specific request is to understand existing code or behavior, accounting for 19.0% of all prompts. Creating new code is another frequent intent at 13.4%. Routine development, such as git operations (13.4%) and debugging (13.0%), is also prevalent, while code refactoring, writing tests, and setting up connections occur less frequently.

Coding agents must be optimized not only for code generation, but for code comprehension and routine development tasks. These capabilities are underrepresented in existing benchmarks, which focus narrowly on patch generation.

1For our data analysis, we filter out any data that appears to be generated by automated bots. 2In practice, ∼85% comes from Claude Code usage data, as this is currently one of the most widely

used coding agents and the first one that was supported by Entire.io’s CLI tool. 3https://github.com/semgrep/semgrep

- Table 2: Annotations applied to the SWE-chat dataset. We show different examples in Appendix B. See Appendix E for implementation details and LLM annotator validations.

Level Task Description Classification input Why this matters Session Session suc-

Rates each session’s overall success on a 0–100 scale.

Full conversation with a summary of all tool calls.

Enables identification of failure patterns, helpful as a training signal for reward modeling.

cess

User persona Assigns each session one of four behavioral personas (expert nitpicker, vague requester, mind changer, or other).

Chronological summary of session events with descriptions (conversations & tool calls).

Characterizes how developers interact with agents, which could inform the design of more adaptive agent interfaces.

Prompt intent Labels prompts with primary user intent: create new code, refactor, debug, understand, connect, git, test, or other.

Raw prompt text without context.

Reveals operational diversity of real workflows.

User prompt

User pushback Classifies non-interruption prompts into pushback categories: correction, rejection, failure report, or non-pushback.

Full conversation transcript preceding the prompt.

Directly measures friction points that degrade the user experience, signaling where agents fall short.

Agents invoke many tools within a single turn One third of all agent tool calls are bash commands—predominantly git operations—followed by file reads, edits, and grep searches (see Figure 19b and Table 5). Agent trajectories typically begin with reading and searching tools before transitioning to file modifications and build commands (Figure 21a).

##### 3.2 Coding modes: vibe coding is increasingly common

55.8% of all committed lines of code are written by coding agents, but this distribution is extremely bimodal—see Figure 5. We therefore introduce three different coding modes:

- • Human-only coding (22.7% of sessions): All committed code is written by the human. The agent serves as an assistant for code comprehension, debugging, or git operations.
- • Collaborative coding (36.5% of sessions): Human and agent jointly contribute to committed code, with the agent authoring >0% but <99% of lines.
- • Vibe coding (40.8% of sessions): More than 99% of the committed code is authored by the agent.

Vibe coding is becoming more prevalent: over our three-month observation window, its share has doubled from 20% to over 40% of sessions (Figure 25).

mean:55.8%

median:73.3%

2000

Numberofsessions

1500

Vibecoding

Human-only

1000

500

Collaborative

0

0 20 40 60 80 100

Agent-authored code (%)

- Figure 5: Vibe coding in the wild. % of agent-authored code, structured into three coding modes: human-only (0% agent-authored code), collaborative (0–99%), and vibe coding (≥99%).

- 3.3 User types: expert nitpicking behavior dominates

To characterize how users interact with agents beyond single prompts, we classify each session into a behavioral persona based on the full transcript (Table 2): expert nitpickers who meticulously correct agent output while maintaining a stable goal, vague requesters who underspecify tasks and delegate decisions to the agent, and mind changers who redirect goals mid-session. Most users act as expert nitpickers (Figure 24). This holds even in vibe coding sessions (47%). Mind changing is less common during vibe coding (5% vs. 10% in other modes). This stands in contrast to current benchmarks, which provide complete instructions up front. In reality, users iteratively refine their instructions after seeing the agent’s outputs.

- 4 How do coding agents fail and how do users respond? (RQ2)

- 4.1 Most coding agent sessions successfully complete user requests

[Figure 21]

User interrupted before agent responded

42 tool calls searching, codebase never found

“You’re out of extra usage · resets 1am” [no work done]

Diagnosed bug but refused to fix it

Delegated to sub-agent, nothing implemented

- Figure 6: Distribution of LLM-annotated session success rating. The distribution is leftskewed, indicating that most sessions are rated as largely successful.

Figure 6 shows that 90% of sessions receive success ratings of 50+, indicating that coding agents generally fulfill users’ requests. Human-only sessions have a slightly lower average session success rating than collaborative and vibe coding sessions.

The tail of the distribution with low success ratings is more interesting, which is why we manually inspected the 50 sessions with the lowest success ratings (2–15). The most common failure modes in these sessions are user interruptions that end the session before the agent can deliver meaningful output, and agents producing work or commits that are entirely unrelated to the user’s actual request. We provide one such example in Figure B.1.

##### 4.2 Coding agents are inefficient

Users discard most AI-written code Less than half (44.3%) of all agent-produced code survives into user commits (Table 3). During vibe coding sessions, users are more accepting, committing 59% of AI-authored lines of code on average. However, this higher survival rate is difficult to interpret causally: it may reflect genuinely better-targeted agent output, or it may reflect lower user scrutiny.

The main source of inefficiency is agent-authored code that the human decides not to commit (see human deletions in Table 3). If the user directly changes the code themselves, it is

captured under human overwrites. Note that agents’ self-overwrites typically occur when the user pushes back and instructs the agent to reimplement something before committing.

Coding Code survival

Coding mode

Detailed attribution breakdown efficiency rate

All modes 44.3% 50.3% 44.3% 9.3% 42.2% Collaborative 38.2% 44.1% 38.2% 10.1% 46.9% Vibe coding 59.0% 64.6% 59.0% 7.1% 30.9%

Survived Agent self-overwrite Human overwrite Human deletion

- Table 3: Agent coding efficiency, code survival rate, and detailed attribution of agentproduced code by coding mode, excluding human-only. Coding efficiency measures what fraction of total agent effort ended up in the commit; survival rate measures what fraction of the agent’s net output the human kept (i.e., it does not penalize agent self-overwrites).

Vibe coding is costly and slow While more of the agent’s output survives into commits in vibe-coding mode, this comes at a substantially higher cost per committed line. Vibe-coded sessions consume a median of 204K tokens per 100 committed lines of code—roughly 3× more than collaborative sessions and 2× more than human-only sessions. Translated to dollar costs, vibe coding has a median cost of $0.13 per 100 committed lines, compared to $0.07 for human-only and $0.05 for collaborative sessions. Furthermore, users invest more effort in prompting when vibe coding (Figures 7 and 29).

0.0 0.2 0.4 0.6 0.8 1.0 1.2

$ spent per 100 committed lines of code

Human-only

Collaborative

Vibe coding

µ=1.1

µ=0.574

µ=0.916

Figure 7: Cost efficiency per 100 committed lines of code. µ indicates means.

In terms of time, collaborative sessions are the most efficient at a median of 4.8 minutes per 100 committed lines, while vibe coding (12.6 minutes) and humanonly sessions (8.6 minutes) are slower in comparison. The agent runtime metric, which excludes time spent waiting for user input, closely tracks session runtime across all modes. However, it is important to note that both time and agent runtime are imperfect proxies, as they do not account for the user’s time spent coding before or after a coding session.

- 4.3 Vibe coding introduces more security vulnerabilities per line

Table 4 reports the rate at which each coding mode introduces security vulnerabilities. For every commit we run the static analyzer Semgrep on the pre- and post-commit repository snapshots and count findings that appear in post but not in pre, restricted to files the commit modified (details in Appendix D.5). Vibe-coded commits introduce vulnerabilities at a rate of 0.76 per 1,000 committed lines, roughly 9× higher than human-only (0.08) and 5× higher than collaborative (0.14) commits. Vibe-coded commits also fix vulnerabilities at a higher rate (0.52 per 1K lines vs. 0.04 for human-only and 0.08 for collaborative), reflecting more security-relevant code-changes overall. But there are more introductions than fixes in every mode, and the difference is biggest for vibe coding.

We observe a range of vulnerability types, including path traversal, command injection, unsafe format strings, and SQL injection (see Appendix Figures 26 and 27). If vibe coding continues to grow as a share of real-world development (Figure 25), the absolute volume of newly introduced security issues might increase, making production code less safe.

- 4.4 Agents work autonomously for longer, but users push back frequently

We now turn to session stops initiated by either the agent or the user. For comparability with McCain et al. (2026), we only include data from Claude Code for the results in Figure 8.

Table 4: Security-relevant findings per coding mode. Introduced counts Semgrep findings present after a commit but not before; Fixed counts findings present before but not after. Rates are per 1,000 added lines. Vibe-coded commits introduce vulnerabilities at roughly 9× the human-only rate and 5× the collaborative rate, but also fix more vulnerabilities.

Vulnerabilities fixed Vulnerabilities introduced Coding mode (per 1K lines) (per 1K lines)

Human-only 0.04 0.08 Collaborative 0.08 0.14 Vibe coding 0.52 0.76

Overall 0.06 0.11

Agents work autonomously for longer Most Claude Code interactions are short. The median turn lasts under one minute, and even the 90th percentile stays below seven minutes (Figure 30). This is broadly consistent with the trends reported by McCain et al. (2026). While the 99.9th percentile turn duration remains well below the 12-hour human-equivalent task difficulty that METR estimates Claude Code can solve at a 50% success rate (Kwa et al., 2025), we observe a clear upward trend over the data-collection period.

Humans frequently interrupt the agent and push back Figure 8 breaks down agentinitiated stops, user interruptions, and user pushback by coding mode. Across all modes, Claude Code rarely proactively asks the user for clarification (1.1%–2.6%). The higher agent autonomy of vibe coding sessions is reflected in fewer agent questions. Surprisingly, the share of agent stops is much lower than what McCain et al. (2026) report.

In contrast, users interrupt the agent more frequently (3.3%–6.0%). This effect is stable over time (see Figure 31) and across coding modes (Figure 8). When users interrupt an ongoing trajectory, the interruption most frequently occurs when the agent exits the plan mode, makes a git operation, or edits a file (Figure 21c).

Even more common than hard user interruptions are soft user pushbacks in the form of correction prompts after an agent’s turn has finished. Overall, users push back after 39% of turns, regardless of coding mode. The observation that vibe coding sessions still exhibit substantial pushback rates suggests that users are not entirely passive, even when fully relying on the AI agent for code writing.

56.0%

55.0%

Turn outcome

50.7%

50

| |
|---|

Uninterrupted Agent stops and asks user for clariﬁcation User pushes back against agent output (total) User interrupts agent

| |
|---|

41.9%

39.6%

38.4%

40

| |
|---|

%ofturns

8.9%

6.7%

| |
|---|

7.3%

0.9%

1.3%

1.0%

30

Pushback types

20

| |
|---|

Correction Rejection Failure report

32.0%

31.6%

| |
|---|

30.2%

| |
|---|

10

6.0%

5.0%

3.3%

1.6%

1.4%

1.1%

0

Human-only Collaborative Vibe coding

- Figure 8: Turn-level oversight in Claude Code sessions. Fraction of turns in which the agent stops to ask for clarification, the user interrupts the agent, or the user pushes back against the agent’s response—broken down by coding mode.

#### 5 Discussion

Together, our findings suggest that coding agents, despite their enormous potential, have substantial room for improvement in efficiency and human-agent collaboration. Our analysis of SWE-chat offers an empirical grounding for this understanding: we surface interaction

patterns, efficiency gaps, and failure modes that are invisible in controlled evaluations. These findings are not meant to be definitive. Rather, they are a starting point for a broader research agenda around in-the-wild agent evaluation and human-agent interaction studies.

Autonomy is outpacing oversight Vibe coding is becoming the new norm. In more than 40% of cases, agents author more than 99% of committed code (Figure 5). At the same time, agents like Claude Code stop to ask users a clarifying question in only 1.4% of turns. Users, on the other hand, interrupt and push back frequently, in roughly 44% of turns (Figure 8). This asymmetry suggests that agents may be gaining autonomy faster than they are learning when to seek guidance, leaving users to compensate through manual oversight.

Agents are powerful but brittle Agents are working independently for longer and writing more code (Figure 30), but more autonomy does not translate into more efficient delivery. Agents author more than half of all committed code, yet less than half of their total output survives into commits (Table 3). Agents rarely signal uncertainty, and errors are typically caught only when users actively inspect outputs (Section E.2.4). This is consistent with the broader observation that AI models often fail silently (Potts and Sudhof, 2026). Notably, collaborative sessions where humans and agents co-author code are the most cost-efficient mode we observe (Figure 29), suggesting that the current push toward full autonomy may be counterproductive. Importantly, these findings do not argue against the use of coding agents. Rather, they reveal that agents are less efficient than they could be.

Agent-written code introduces more security vulnerabilities Prior work has shown that LLMs can produce insecure code even from benign prompts (Pearce et al., 2025; Bhatt et al., 2023; Fu et al., 2025). Developers using AI assistants are more likely to produce insecure code while feeling more confident about its security (Perry et al., 2023). SWE-chat extends this to real developer workflows with coding agents: vibe-coded commits introduce Semgrepdetected vulnerabilities at roughly 9× the human-only rate and 5× the collaborative rate

- (Table 4). Combined with our finding that agents rarely signal uncertainty (Figure 8), this suggests that as autonomy grows, the burden of catching unsafe patterns shifts entirely to the user. Existing mitigations, such as secure fine-tuning and system-prompt hardening (He and Vechev, 2023; He et al., 2024; Xu et al., 2025), have largely been evaluated on synthetic benchmarks. SWE-chat provides a natural testbed for whether such interventions are effective for realistic coding agent tasks.
- 5.1 Outlook: implications for building better coding agents

Realistic benchmarks grounded in real workflows Current benchmarks evaluate agents on isolated, curated tasks that reward one-shot patch generation. But the most common realworld intent we observe is understanding existing code, not writing it, and most sessions involve iterative multi-turn interaction rather than single-shot problem solving. SWE-chat enables the construction of benchmarks grounded in actual developer workflows (Zhou et al., 2026). For example, session trajectories can be used to evaluate whether an agent proposes appropriate next actions given real conversation context.

Designing more adaptive human-agent interaction Users push back against agent output in nearly every other turn, yet they rarely abandon sessions entirely. They correct, redirect, and steer agents iteratively until the result is acceptable. At the same time, agents proactively ask for clarification in <2% of turns. SWE-chat captures these correction-response cycles at scale, providing researchers with the data needed to study how human oversight actually unfolds in practice and where current agent interaction design falls short (Guan et al., 2025).

User simulators for offline evaluation Evaluating coding agents currently requires either curated benchmarks or live user studies, both of which are expensive and limited in scope (Naous et al., 2025; Buening et al., 2026). SWE-chat provides the raw material for a new evaluation paradigm: training user simulators on real interaction trajectories. The dataset captures a wide range of the diverse behavioral patterns that a realistic simulator would need to reproduce.

Benchmarks are fixed at the moment of their creation, but how developers use coding agents is changing rapidly. SWE-chat is designed as a living dataset that evolves with the technology it measures. By providing continual updates, it enables longitudinal analysis and ensures our understanding of agents remains grounded in how they are actually used.

#### Ethics statement

All data in SWE-chat is collected from public GitHub repositories where developers have explicitly opted in to Entire CLI tracking and pushed session logs to public branches. We only include repositories whose licenses allow redistribution. We do not collect images attached to user prompts. Before release, we remove personally identifiable information (PII) from every user prompt and assistant response in the dataset, following the WildChat data processing pipeline (Zhao et al., 2024). First, we run Microsoft Presidio’s named-entity recognizer with a SpaCy transformer model over every user/assistant turn to redact PII (e.g., email addresses, phone numbers, person names). Second, we remove credentials (API keys, OAuth tokens, database URIs, etc.) with TruffleHog. The study procedure was reviewed and deemed exempt by the Stanford Institutional Review Board (IRB).

#### Acknowledgments

We are thankful to the members of SALT Lab, the STAIR Lab, the Stanford NLP Group, and the MilaNLP Lab for their helpful feedback, particularly Chenglei Si, David Anugraha, Hao Zhu, Ricardo Dominguez-Olmedo, and Steven Dillmann. This work is partially supported by Open Philanthropy, ONR N000142412532, Schmidt Sciences, NSF 2046795 and 2205329, IES R305C240046, the MacArthur Foundation, Stanford HAI, and the Swiss National Science Foundation (SNSF grant 235328).

#### References

Anthropic. How anthropic teams use claude code. https://claude.com/blog/ how-anthropic-teams-use-claude-code, 2025. Companion technical report available at https://www-cdn.anthropic.com/58284b19e702b49db9302d5b6f135ad8871e7658. pdf. Accessed: 2026-03-31.

Anthropic. 2026 agentic coding trends report: How coding agents are reshaping software development. https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding% 20Trends%20Report.pdf, 2026.

Alpay Ariyak, Junda Zhang, Junxiong Wang, Shang Zhu, Federico Bianchi, Sanjana Srivastava, Ashwinee Panda, Siddhant Bharti, Chenfeng Xu, John Heo, Xiaoxia Shirley Wu, James Zhou, Percy Liang, Leon Song, Ce Zhang, Ben Athiwaratkun, Zhongzhu Zhou, and Qingyang Wu. Coderforge-preview: Sota open dataset for training efficient agents, February 2026. URL https://www.together.ai/blog/coderforge-preview. Project core leads: Alpay Ariyak; Zhongzhu Zhou; Qingyang Wu.

Joachim Baumann, Paul Röttger, Aleksandra Urman, Albert Wendsjö, Flor Miriam Plaza-del Arco, Johannes B Gruber, and Dirk Hovy. Large language model hacking: Quantifying the hidden risks of using llms for text annotation. arXiv preprint arXiv:2509.08825, 2025.

Joel Becker, Nate Rush, Elizabeth Barnes, and David Rein. Measuring the impact of early-2025 ai on experienced open-source developer productivity. arXiv preprint arXiv:2507.09089, 2025.

Manish Bhatt, Sahana Chennabasappa, Cyrus Nikolaidis, Shengye Wan, Ivan Evtimov, Dominik Gabi, Daniel Song, Faizan Ahmad, Cornelius Aschermann, Lorenzo Fontana, et al. Purple llama cyberseceval: A secure coding benchmark for language models. arXiv preprint arXiv:2312.04724, 2023.

Islem Bouzenia and Michael Pradel. Understanding software engineering agents: A study of thought-action-result trajectories. arXiv preprint arXiv:2506.18824, 2025.

Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, and Andreas Krause. Aligning language models from user interactions. arXiv preprint arXiv:2603.12273, 2026.

Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, et al. Nex-n1: Agentic models trained via a unified ecosystem for large-scale environment construction. arXiv preprint arXiv:2512.04987, 2025.

Ricardo JGB Campello, Davoud Moulavi, and Jörg Sander. Density-based clustering based on hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and data mining, pages 160–172. Springer, 2013.

Wayne Chi, Valerie Chen, Anastasios Nikolas Angelopoulos, Wei-Lin Chiang, Aditya Mittal, Naman Jain, Tianjun Zhang, Ion Stoica, Chris Donahue, and Ameet Talwalkar. Copilot arena: A platform for code llm evaluation in the wild. arXiv preprint arXiv:2502.09328, 2025.

Jacob Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46, 1960. doi: 10.1177/001316446002000104. URL https://doi. org/10.1177/001316446002000104.

Cursor Research Team. Composer 2 Technical Report. 2026. URL https://cursor.com/ resources/Composer2.pdf.

Ozge Demirci, Jonas Hannane, and Xinrong Zhu. Who is ai replacing? the impact of generative ai on online freelancing platforms. Management Science, 71(10):8097–8108, 2025. doi: 10.1287/mnsc.2024.05420. URL https://doi.org/10.1287/mnsc.2024.05420.

Xiang Deng, Jeff Da, Edwin Pan, Yannis Y. He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa R Kundurthy, Sean M. Hendryx, Zifan Wang, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. SWE-bench pro: Can AI agents solve long-horizon software engineering tasks?, 2026. URL https://openreview.net/forum?id=9R2iUHhVfr.

Yujia Fu, Peng Liang, Amjed Tahir, Zengyang Li, Mojtaba Shahin, Jiaxin Yu, and Jinfu Chen. Security weaknesses of copilot-generated code in github projects: An empirical study. ACM Trans. Softw. Eng. Methodol., 34(8), October 2025. ISSN 1049-331X. doi: 10.1145/3716848. URL https://doi.org/10.1145/3716848.

Melody Y Guan, Miles Wang, Micah Carroll, Zehao Dou, Annie Y Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian Kivlichan, Mia Glaese, et al. Monitoring monitorability. arXiv preprint arXiv:2512.18311, 2025.

Jingxuan He and Martin Vechev. Large language models for code: Security hardening and adversarial testing. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, CCS ’23, page 1865–1879, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400700507. doi: 10.1145/3576915.3623175. URL https://doi.org/10.1145/3576915.3623175.

Jingxuan He, Mark Vero, Gabriela Krasnopolska, and Martin Vechev. Instruction tuning for secure code generation. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.

Abhi Kottamasu, Akul Datta, Aakash Barthwal, Chirag Mahapatra, Ajay Arun, Adarsh Hiremath, Brendan Foody, and Bertie Vidgen. Apex-swe. arXiv preprint arXiv:2601.08806, 2026.

Thomas Kwa, Ben West, Joel Becker, Amy Deng, Katharyn Garcia, Max Hasin, Sami Jawhar, Megan Kinniment, Nate Rush, Sydney Von Arx, et al. Measuring ai ability to complete long tasks. arXiv preprint arXiv:2503.14499, 2025.

Hao Li, Haoxiang Zhang, and Ahmed E. Hassan. The rise of ai teammates in software engineering (se) 3.0: How autonomous coding agents are reshaping software engineering,

- 2025. URL https://arxiv.org/abs/2507.15003.

Robert A. Martin and Sean Barnum. Common weakness enumeration (cwe) status update. Ada Lett., XXVIII(1):88–91, April 2008. ISSN 1094-3641. doi: 10.1145/1387830.1387835. URL https://doi.org/10.1145/1387830.1387835.

Maxim Massenkoff, Eva Lyubich, Peter McCrory, Ruth Appel, and Ryan Heller. Anthropic economic index report: Learning curves, 2026. URL https://www.anthropic. com/research/economic-index-march-2026-report.

Miles McCain, Thomas Millar, Saffron Huang, Jake Eaton, Kunal Handa, Michael Stern, Alex Tamkin, Matt Kearney, Esin Durmus, Judy Shen, Jerry Hong, Brian Calvert, Jun Shern Chan, Francesco Mosconi, David Saunders, Tyler Neylon, Gabriel Nicholas, Sarah Pollack, Jack Clark, and Deep Ganguli. Measuring ai agent autonomy in practice, 2026. URL https://anthropic.com/research/measuring-agent-autonomy.

Kenneth O McGraw and Seok P Wong. Forming inferences about some intraclass correlation coefficients. Psychological methods, 1(1):30–46, 1996. doi: https://doi.org/10.1037/ 1082-989X.1.1.30.

Leland McInnes, John Healy, Steve Astels, et al. hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11):205, 2017.

Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026.

METR. Time horizon 1.1. https://metr.org/blog/2026-1-29-time-horizon-1-1/, 01 2026. Christian Mürtz and Mark Niklas Müller. Agents in the wild - dashboard. https://insights.

logicstar.ai, 2025. URL https://doi.org/10.5281/zenodo.15846865. Interactive web dashboard. Code available at https://github.com/logic-star-ai/insights.

Tarek Naous, Philippe Laban, Wei Xu, and Jennifer Neville. Flipping the dialogue: Training and evaluating user language models. arXiv preprint arXiv:2510.06552, 2025.

Jane Pan, Ryan Shar, Jacob Pfau, Ameet Talwalkar, He He, and Valerie Chen. When benchmarks talk: Re-evaluating code LLMs with interactive feedback. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 24672–24700, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10. 18653/v1/2025.findings-acl.1267. URL https://aclanthology.org/2025.findings-acl. 1267/.

Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. arXiv preprint arXiv:2510.04374, 2025.

Hammond Pearce, Baleegh Ahmad, Benjamin Tan, Brendan Dolan-Gavitt, and Ramesh Karri. Asleep at the keyboard? assessing the security of github copilot’s code contributions. Commun. ACM, 68(2):96–105, January 2025. ISSN 0001-0782. doi: 10.1145/3610721. URL https://doi.org/10.1145/3610721.

Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. The impact of ai on developer productivity: Evidence from github copilot. arXiv preprint arXiv:2302.06590, 2023.

Neil Perry, Megha Srivastava, Deepak Kumar, and Dan Boneh. Do users write more insecure code with ai assistants? In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, CCS ’23, page 2785–2799, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400700507. doi: 10.1145/3576915.3623157. URL https://doi.org/10.1145/3576915.3623157.

Christopher Potts and Moritz Sudhof. Invisible failures in human-ai interactions. arXiv preprint arXiv:2603.15423, 2026.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/.

Suproteem K Sarkar. Ai agents, productivity, and higher-order thinking: Early evidence from software development. Available at SSRN 5713646, 2025.

Ethan Shen, Danny Tormoen, Saurabh Shah, Ali Farhadi, and Tim Dettmers. Sera: Softverified efficient repository agents. arXiv preprint arXiv:2601.20789, 2026.

Patrick E Shrout and Joseph L Fleiss. Intraclass correlations: uses in assessing rater reliability. Psychological bulletin, 86(2):420–428, 1979. doi: https://doi.org/10.1037/0033-2909.86.2. 420.

Yueqi Song, Ketan Ramaneti, Zaid Sheikh, Ziru Chen, Boyu Gou, Tianbao Xie, Yiheng Xu, Danyang Zhang, Apurva Gandhi, Fan Yang, et al. Agent data protocol: Unifying datasets for diverse, effective fine-tuning of llm agents. arXiv preprint arXiv:2510.24702, 2025.

Charles Spearman. The proof and measurement of association between two things. pages 45–58, 1961.

Maria Trofimova, Anton Shevtsov, Badertdinov Ibragim, Konstantin Pyaev, Simon Karasik, and Alexander Golubev. Openhands trajectories with qwen3-coder-480b-a35b-instruct. Nebius blog, 2025.

Zora Zhiruo Wang, Sanidhya Vijayvargiya, Aspen Chen, Hanmo Zhang, Venu Arvind Arangarajan, Jett Chen, Valerie Chen, Diyi Yang, Daniel Fried, and Graham Neubig. How well does agent development reflect real-world work? arXiv preprint arXiv:2603.01203, 2026a.

Zora Zhiruo Wang, John Yang, Kilian Lieret, Alexa Tartaglini, Valerie Chen, Yuxiang Wei, Zijian Wang Lingming Zhang, Karthik Narasimhan, Ludwig Schmidt, Graham Neubig, Daniel Fried, and Diyi Yang. Position: Humans are missing from ai coding agent research. https://zorazrw.github.io/files/position-haicode.pdf, 2026b.

Xiangzhe Xu, Zian Su, Jinyao Guo, Kaiyuan Zhang, Zhenting Wang, and Xiangyu Zhang. Prosec: Fortifying code LLMs with proactive security alignment. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= Ym19zWky7W.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering, 2024a. URL https://arxiv.org/abs/2405.15793.

John Yang, Carlos E. Jimenez, Alex L. Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R. Narasimhan, Diyi Yang, Sida I. Wang, and Ofir Press. Swe-bench multimodal: Do ai systems generalize to visual software domains?, 2024b. URL https://arxiv.org/abs/2410.03859.

John Yang, Kilian Lieret, Carlos E Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. SWE-smith: Scaling data for software engineering agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/ forum?id=63iVrXc8cC.

Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, and Liang Xiang. Multi-swe-bench: A multilingual benchmark for issue resolving, 2025. URL https://arxiv.org/abs/2504.

02605.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=Bl8u7ZRlbM.

Xuhui Zhou, Weiwei Sun, Qianou Ma, Yiqing Xie, Jiarui Liu, Weihua Du, Sean Welleck, Yiming Yang, Graham Neubig, Sherry Tongshuang Wu, et al. Mind the sim2real gap in user simulation for agentic tasks. arXiv preprint arXiv:2603.11245, 2026.

Yangtian Zi, Zixuan Wu, Aleksander Boruch-Gruszecki, Jonathan Bell, and Arjun Guha. Agentpack: A dataset of code changes, co-authored by agents and humans, 2025. URL https://arxiv.org/abs/2509.21891.

#### A Limitations

SWE-chat is a first-of-its-kind dataset (Figure 1). However, it only contains data from developers who use the Entire CLI with public repositories and opt into checkpoint logging. This selects for early adopters of a new open-source tool and does not cover proprietary enterprise codebases. Agent performance and interaction patterns may differ substantially in such settings (e.g., agents may struggle more with undocumented legacy code, or less with well-structured internal libraries). At this stage, findings based on SWE-chat may not generalize. Additionally, a large fraction of data comes from Entire.io’s own code repository. However, as more open-source developers adopt the tool, the dataset becomes increasingly diverse (see Appendix D.1.4).

Most failed sessions are not captured by our data. If the user abandons the agent’s output entirely, session logs are not committed and thus not captured by our data. This likely leads to an overestimation of session success rates and agent efficiency. On the other hand, we treat agent-authored code that is deleted by the human as inefficient output. However, some of this code may survive semantically, for instance, when a user rewrites an agent suggestion in a different file or refactors it into a different form. Our line-level attribution approach does not capture such cases, potentially underestimating the true usefulness of agent contributions.

The diversity of our data makes it difficult to assess the quality of the code produced. Some of the metrics we use (e.g., number of committed lines) should be understood only as proxies of users’ satisfaction with AI-generated outputs. Similarly, our efficiency metrics capture only what is observable in the session logs and may not reflect the full picture. For instance, cognitive efficiency, measured as prompt characters per committed line does not account for the time users spend reading and reviewing agent output, or planning their instructions. Future research can build on this to develop more robust measurements that can be used as optimization objectives.

LLMs are imperfect data annotators (see Appendix E.1). For results based on LLM-generated labels, we do not draw conclusive statements, given the inherent unreliability of such annotations and the risk of LLM hacking (Baumann et al., 2025). Rather, we use these annotations to enable easy filtering of the large dataset we introduce, for example, to surface specific cases of unsuccessful sessions such as the one presented in the Appendix B.1. We caution against taking these labels at face value and recommend further validation before using them in downstream analyses.

#### B SWE-chat examples

This appendix presents representative examples from SWE-chat illustrating key interaction patterns between users and coding agents. Each example is drawn from a real session in the dataset.

##### B.1 Low session success score

Session score: 10/100 Repository: nuttycc/LuminTime

User: Fix the slow appearance of history list card blocks. The animation feels sluggish when scrolling.

Agent: I’ll fix the animation performance.

▷ Read HistoryListView.swift

▷ Edit HistoryListView.swift — removed unused variable, adjusted stagger delay I’ve optimized the stagger timing for the card animations. The delay between each card appearing has been reduced.

User: That’s the wrong parameter — the container animation is the slow part, not the individual card stagger.

Agent: I see, let me adjust the container timing instead.

▷ Edit HistoryListView.swift — modified item animation duration I’ve updated the animation timing parameters.

User: You changed the item timing again, not the container! Please verify which parameter actually controls the container appearance before making more changes.

No commits made. Session ended without resolution.

Figure 9: Example of a low-success session (score: 10/100). The agent repeatedly modified the wrong animation parameter despite user corrections, failing to verify its assumptions before making edits.

##### B.2 User pushback

User pushback captures moments where the user redirects, corrects, or rejects the agent’s output. We distinguish three subtypes: corrections (the user provides missing information or redirects the approach), rejections (the user explicitly undoes or refuses the agent’s work), and failure reports (the user reports that the agent’s output is broken or incorrect).

- B.2.1 Correction Pushback type: correction Repository: entireio/cli

Agent: The challenge is that PrepareCommitMsg doesn’t have access to the transcript path

— that’s passed to the Stop hook by Claude Code. What we do have in PrepareCommitMsg: session state (FilesTouched, BaseCommit, etc.) and staged changes from git. So we’d need to find another approach to access the transcript content.

User: but we do have access to the session.CondensedTranscriptLines...?

- Figure 10: Example of user correction pushback. The user points out that the agent overlooked an available API field, redirecting the approach without rejecting the overall goal.

- B.2.2 Rejection

Pushback type: rejection Repository: armelhbobdad/bmad-module-skill-forge

Agent: Done. Issue #53 committed (766901b) and closed. Ready for the next one.

User: I undid the commit. We need to find a better approach from scratch. Activate the party mode.

Figure 11: Example of user rejection pushback. The user explicitly reverts the agent’s committed work and requests a completely different approach.

- B.2.3 Failure report

- Figure 12: Example of user failure report pushback. The agent reports a successful fix, but the user observes the feature is still broken and reports it with a screenshot.

Pushback type: failure_report Repository: obsessiondb/rudel

Agent: Now instead of adding padding to the element (which breaks the chart layout), it wraps the element in a temporary container div with 24px padding, captures that, then restores the DOM. The chart content stays untouched.

▷ Edit ChartExport.tsx — wrapped capture target in padding container

User: This is broken [User sends screenshot]

##### B.3 Hard user interruptions

Event: hard interruption Repository: pskoett/pskoett-ai-skills

User: ok, now update the readme. also make sure the install information is in the new skill

Agent: I’ll verify the install steps first.

▷ Bash brew tap entireio/tap && brew install entire

▷ Bash entire –version

▷ Bash entire enable –agent claude-code

▷ Bash entire status

[Request interrupted by user]

User: ok, now update the readme

- Figure 13: Example of a hard user interruption. The user asked to update a README file, but the agent began executing shell installation commands instead of editing the file. The user interrupted and repeated the original request verbatim.

- B.4 Agent stops to ask for clarification (AskUserQuestion)

Event: agent stops to ask for clarification Repository: navapbc/lockpick-xlator

User: The tool currently outputs .graph.md files wrapping mermaid in markdown fences. I want it to output raw .mmd files instead.

Agent:

▷ Read tools/generate-graph.ts

▷ Read commands/graph.ts

The change looks straightforward. Currently the tool writes a .graph.md file wrapping mermaid in markdown fences (with a header and generated date). The .mmd file should contain raw mermaid only — no markdown wrapper. Two command files also reference .graph.md and will need updating.

Should I skip brainstorming and go straight to planning?

|Yes, go to planning|
|---|

|Brainstorm first|
|---|

|Just implement it|
|---|

User: Yes, go to planning

Figure 14: Example of agent-initiated clarification (AskUserQuestion). The agent pauses execution to confirm the user’s preferred workflow, presenting structured options.

- B.5 Prompt intent categories

Each user prompt is classified by its primary developer intent. Below we show one representative prompt per category, drawn from entireio/cli.

Intent: create

User: “Perfect let’s add support for those agents as well”

Intent: refactor

User: “I’d also like to remove all instances of ‘does the result contain A?’ — instead, replace those assertions with a single comparison to what the result should be, literally.”

###### Intent: debug

User: “did our timeout kill and retry work? it looks like it just got killed and we failed?”

Intent: understand

User: “What would happen if we keep the hooks going if he continues after the warning was shown?”

Intent: connect

User: “give steps for setting this up in slack / github”

###### Intent: git

User: “commit this — but let’s test out making some changes after”

###### Intent: test

User: “add test cases to manual_commit_condensation_test.go which use the following cursor sample:”

###### Intent: other

User: “I don’t like ‘turn’, any other suggestions?”

- Figure 15: Example user prompts for each intent category. Notice that the last prompt lacks context, which is why it is classified as ’other’.

##### B.6 User persona categories

Each session’s user is classified into a behavioral persona based on their interaction patterns across the full session.

Persona: Expert Nitpicker Repository: marin-community/marin

User: instead of arg hash for the name, let’s just get short uuid

Agent: [edits remote.py: replaces arg hash with short UUID...]

User: don’t create separate function for _short_uuid

Agent: [removes helper function, inlines uuid4() call...]

User: change DEFAULT_JOB_NAME to “remote_job”

Agent: [renames constant in remote.py...]

User: In the base_name let’s include current user name

- Figure 16: Example of the Expert Nitpicker persona. The user maintains a stable goal while issuing a series of precise, targeted corrections to the implementation. Each prompt refines how the agent executes, not what it builds.

Persona: Vague Requester Repository: entireio/cli

User: pull comments on PR and if real issues the fix

Agent: [fetches PR comments, applies 3 fixes across 2 workflow files...]

User: commit and push

- Figure 17: Example of the Vague Requester persona. The user provides broad, underspecified instructions and delegates all implementation decisions to the agent.

Persona: Mind Changer Repository: entireio/cli

User: let’s make the “entire start” command hidden

Agent: [edits start.go: sets command hidden, verifies build...]

User: thinking again: let’s completely remove the start command

- Figure 18: Example of the Mind Changer persona. The user reverses the overall goal midsession — from hiding a CLI command to removing it entirely — changing what should be built, not just how.

#### C Experimentation details

##### C.1 Data processing pipeline

Raw session log data from AI agents is stored on each repository’s entire/checkpoints/v1 branch, containing checkpoint and session metadata, user prompts, and full conversation transcripts. From each transcript, we extract structured conversation turns (user prompts, assistant responses, thinking traces, tool calls, and tool results), per-turn token usage, and tool-call metadata, including file paths and shell commands.

SWE-chat data growth trajectory As coding agents make it increasingly easy to generate large volumes of code, developers face growing challenges in reviewing, understanding, and validating AI-generated contributions (Sarkar, 2025; Becker et al., 2025; Anthropic, 2026). Entire addresses this need by letting developers track how their codebase evolved not only as a function of commits, but as a function of prompts, creating a searchable record of every AI-assisted change. This utility incentivizes continued adoption, and we expect the dataset to keep growing, a trend already visible in the steep trajectory shown in Figure 1. Our pipeline discovers Entire-enabled public repositories by querying the GitHub Code Search API and, for each repository, downloads all checkpoint directories from the metadata branch and parses the raw transcripts into structured tables.

##### C.2 Metrics

Session duration, tool call duration, number of in- and output tokens, and files touched during agent actions are all measured directly from coding agent session logs. We quantify coding agent efficiency using several complementary approaches, all computed from raw data without the need for annotations.

Agent-authored code percentage The Entire CLI computes code attribution at commit time using temporary checkpoints on shadow branches. It constructs checkpoints stored on a shadow branch to obtain all committed human vs. agent-written lines.

agent lines survived total committed lines × 100 (1)

Agent-authored % =

Agent coding efficiency and code survival To measure the fraction of agent-produced code that survives into the final commit, we perform a post-hoc analysis, since the agentauthored code percentage does not record per-tool-call provenance or agent self-overwrites. We analyze three states for all changed files: the base version (parent commit), the agent actions (sequential tool calls), and the committed version. We reconstruct agentic changes by replaying every file-modifying tool call (e.g., write, edit) in chronological order. After each tool call, we compute a line-level diff between the file’s previous and new state using Python’s difflib.SequenceMatcher. Each line carries a provenance tag—either base (present before the agent acted) or agent (introduced by the agent)—which is updated as we proceed along the agent trajectory. With this approach, we can track all agentic code additions, edits, and deletions—and compute which changes survive, as measured by the file state at the time of commit. We derive two rates from the per-commit aggregate counts:

agent lines survived

agent cumulative lines produced × 100 (2) Code survival rate =

Coding efficiency =

agent lines survived

agent lines in final state × 100 (3)

Coding efficiency measures the fraction of the agent’s total effort (including lines it later rewrote) that ended up in the commit. The code survival rate measures the fraction of the agent’s net output (after self-overwrites) that the human kept unchanged. Note that concurrent changes, where the human and the agent modify the same file simultaneously, may cause the transcript to reflect inconsistent file states and attributions.

Token, cost, and cognitive efficiency We also quantify several cost-per-output metrics that capture the resources consumed to produce each committed line of code. For each session with a clean mapping to committed code (see Appendix C.3), we compute:

total tokens (in + out + cache)

Token efficiency =

total committed lines × 100 (4) Cost efficiency =

∑API call tokens × pricemodel

total committed lines × 100 (5) Cognitive efficiency =

∑ user prompt characters

- total committed lines × 100 (6)

Time efficiency =

∑ session runtimes

- total committed lines × 100 (7)

Agent runtime efficiency =

∑ agent runtimes

- total committed lines × 100 (8)

For time efficiency, we consider complete session runtimes but exclude all idle periods lasting more than 2 minutes, i.e., when neither the agent nor the user performs any action. For agent runtime efficiency, we sum the completion times of all agent turns, where a turn starts with a user prompt and ends with an agent response.

##### C.3 Combining session-level statistics with commit-level outcomes

Sessions may span multiple commits, and multiple sessions may contribute to the same commit (checkpoint). To combine session-level statistics with commit-level results, we restrict the analyses in Table 3 and Figure 29 to sessions where the commit-level lines can be unambiguously attributed. This includes 48.6% of sessions.

#### D Additional results

##### D.1 Dataset statistics

###### (a) Prompt intent

###### (b) Tool call type

###### (c) Repo domain

###### (d) Repo audience

(e) Coding mode

TodoWrite 0.8% mcp 1.7% ToolSearch 0.9%

education 2.5% researchers 3.5%

create new code

other

agent 6.7%

Human-only

other 2.0%

13.4%

15.1%

enduser

22.7%

###### read 19.7%

other 26.6%

debug

35.2%

edit

Vibe coding 40.8%

13.0%

19.3%

application 52.8%

grep

10.4%

developer 58.8%

32.0%

13.6%

glob 1.3%

refactor 9.2%

devtools

12.5%

36.5%

write 2.6%

git

19.0%

bash:ﬁle 7.0% bash:build 8.4% bash:net 1.4%

git/gh

test 4.0%

Collaborative

bash 5.4%

understand connect 1.0%

- Figure 19: Distributions of human user intents (a), agent tool calls (b), repository domains (c), repository audiences (d), and coding modes (e).

- D.1.1 Prompt languages User prompts are predominantly in English (Figure 20). We detect the language of each user prompt using lingua-py4 and retain languages appearing in at least 100 prompts. We manually verified 2,000 classifications where the detector reported low confidence or predicted an extremely low-resource language. In most such cases, the prompt mixed code snippets with English instructions, causing misclassification, and we corrected the label accordingly.
- D.1.2 Tool calls

Table 5 provides a full breakdown of agent tool call types. We group some of the tool calls into aggregate categories for simplicity.

- D.1.3 Agent trajectories

- Figure 21a shows the tool call composition at each sequential position within an agent trajectory after a user makes a request. In early positions, the agent often uses research

4https://github.com/pemistahl/lingua-py

80

%ofprompts

60

40

20

0

EnglishJapaneseChineseRussianPortugueseKorean

Figure 20: Top 6 prompt languages.

Table 5: Tool call type distribution across all agent tool calls.

Rank Category Count % Includes

- 1 read 60,855 19.8% Read, read_file
- 2 grep 31,238 10.1% Grep, bash grep/rg
- 3 glob 4,318 1.4% Glob
- 4 bash:file 21,130 6.9% cd, ls, cat, find, mkdir, rm, bd, echo, tail, wc, lsof, head, ...
- 5 bash:build 24,699 8.0% mise, bun, go, npx, cargo, pnpm, uv, python, npm, xcodebuild, node, make, ...
- 6 bash:net 3,541 1.2% curl, ssh, dig, scp, nc, rsync, nslookup, ping, nmap
- 7 bash 16,527 5.4% sleep, gcloud, for, source, docker, agent-browser, sed, nix, vendor/bin/phpunit, which, ./gradlew, rtk, ...
- 8 git/gh 36,537 11.9% git, gh
- 9 write 9,025 2.9% Write, write_file
- 10 edit 60,205 19.6% Edit, MultiEdit
- 11 web 1,435 0.5% WebFetch, WebSearch
- 12 agent 20,495 6.7% Task, TaskCreate, TaskUpdate, Agent, TaskOutput, SendMessage, ...
- 13 mcp 6,001 1.9% mcp__* (user-installed MCP server tools)
- 14 TodoWrite 3,217 1.0% TodoWrite
- 15 ToolSearch 2,804 0.9% ToolSearch
- 16 AskUserQuestion 2,171 0.7% AskUserQuestion
- 17 Skill 1,679 0.5% Skill
- 18 EnterPlanMode 300 0.1% EnterPlanMode
- 19 ExitPlanMode 1,089 0.4% ExitPlanMode
- 20 other 511 0.2% miscellaneous (apply_patch, LSP, KillShell, ...)

tools (read, grep, glob, and git/gh) as it orients itself in the codebase. As the trajectory progresses, action tools such as edit, write, and bash:build become more prominent.

- Figure 21b examines the same trajectories from the opposite direction, showing tool call composition counting backward from the natural end of a turn (position −1 = last tool call before the agent writes its text response, shown in the rightmost bar). The last tool calls in natural turns are most frequently git/gh commands (committing or pushing results), bash:build (executing bash commands), and edit (final code modifications). Notably, AskUserQuestion rarely appears at position −1, because it is non-blocking, i.e., a turn is completed only with an agent response.
- Figure 21c applies the same reverse trajectory but for a turn that ended with a hard user interruption. ExitPlanMode is the most frequent last tool call (32%), indicating that users often interrupt right at the transition from planning to execution. In such cases, the agent has just finalized its plan, and the user decides to redirect before any code changes are made.

##### D.1.4 Code repository types

To further contextualize the environment in which these interactions occur, we analyze the domains and target audiences of the repositories. We classify each repository into one of three domains (application, devtools, other) and one of four audiences (enduser, developer,

| | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|13<br><br>7|15<br><br>6|18<br><br>6 6|6|6|6|6|6|6|6|6|6|6|6|6|6|7 6 6|7<br><br>6|6 6|6 7|6 7|
|20|18|15<br><br>19|21|21|21|22|23|24|24|24|24|24|24|24|25|24 25 25|24<br><br>24|24 25|26<br><br>24|24 25|
|5| |6 6<br><br>13|7<br><br>12|8<br><br>5<br><br>11|8<br><br>5<br><br>10|9<br><br>6<br><br>10<br>|9<br><br>6<br><br>10<br>|9<br><br>5<br><br>9|10<br><br>10|9<br><br>6<br><br>9|10<br><br>6<br><br>9|10<br><br>6<br><br>9|11<br><br>5<br><br>9|11<br><br>6<br><br>8|11<br><br>5<br><br>9|11 11<br><br>11<br><br>6 6<br><br>6<br><br>9 8<br><br>9|11 12<br><br>6<br><br>6<br><br>9<br><br>8|13 12<br><br>6<br><br>6<br><br>8<br><br>8|11<br><br>12<br><br>6<br><br>6<br><br>8<br><br>9|12 12<br><br>6 6<br><br>8 8|
|9<br><br>7|11<br><br>7|11 11<br><br>7 7|11<br><br>7|11<br><br>7|11<br><br>7|11<br><br>7|11<br><br>7|10<br><br>7|11<br><br>7|10<br><br>7|11<br><br>7|11<br><br>7|11<br><br>7|11<br><br>7|11<br><br>6|11 11<br><br>11<br><br>6 7 7|11 11<br><br>7 7|9 11<br><br>7 7|10 11<br><br>7<br><br>7|10 11<br><br>7 7|
|23|23|22 21|20|19|19|18|18|19|18|18|18|18|18|18|17|18 18 17|18 18|18 17|17 17|17 17|

n=33,980

n=27,938

n=23,444

n=20,058

n=17,307

n=15,150

n=13,525

n=12,035

n=10,853

n=57,766

n=9,855

n=9,008

n=8,210

n=7,534

n=6,918

n=6,416

n=5,962

n=5,522

n=5,161

n=4,828

n=4,501

n=4,218

n=3,971

n=3,752

n=3,582

n=3,390

n=3,221

n=3,030

n=2,861

n=2,691

100

6

11

80

23

###### Shareoftoolcalls(%)

21

8

60

6

5

14

40

8

6

8

11

20

23

17

0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30+

Tool call position within agent response

Tool

read grep glob bash:ﬁle

bash:build

bash:net

bash

git/gh

write

edit web agent

mcp

TodoWrite ToolSearch AskUserQuestion

Skill

EnterPlanMode

ExitPlanMode

other

(a) Tool call composition by position within the agent trajectory (left to right).

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|22 23<br><br>9 9|22<br><br>23<br><br>9 9|22<br><br>8|22<br><br>8|23 24<br><br>8 8|24<br><br>7|23<br><br>7|23<br><br>7|23<br><br>7|23 23<br><br>7 7|23<br><br>23 23<br><br>6<br><br>6 6|23 23<br><br>6 5|23|23 22|22|21|20|18|14|10<br>11<br>| | | |
|5 5<br>6<br><br><br>6<br><br>6<br><br>7<br>|5<br><br>5<br><br>6<br><br>6<br><br>6<br><br>7<br><br><br>|5<br><br>7<br><br>6<br>|6<br><br>6<br><br>6|6 6<br>7 7<br><br><br>6 7|6<br>7<br><br><br>7|6<br>7<br><br><br>7|6<br>7<br><br><br>7|6<br>7<br><br><br>7|6 6<br>7 7<br>7 8<br>|7<br><br>7 7<br><br>8<br><br>8 8<br><br><br><br><br>7<br><br>7 8|7 7<br>8 8<br><br><br>8 8|7<br>8<br>9<br>|7 7<br>8 9<br><br><br>5 5<br><br>10 11|7<br><br>9<br><br>5<br><br>12|8<br><br>10<br><br>5<br><br>14|10<br><br>5<br><br>17|12<br><br>6<br><br>21|13<br><br>6<br><br>25|7<br><br>27| |response| |
|13<br><br>14|13<br><br>14|14|13|13 12|14|14|13|14|14 13|13<br><br>12 13|13 12|12|11<br><br>10|10|9|8<br><br>8|9|8|11| | | |
|23 22|24 22|23|23|22 22|22|22|22|22|22 23|23 22 23|22 22|22|21 21|21|19|18|15<br><br>7|13<br><br>6|9<br><br>8| | | |

n=54,245

n=11,016

n=12,372

n=13,863

n=15,830

n=18,419

n=21,632

n=25,885

n=31,614

n=31,614

n=2,489

n=2,649

n=2,804

n=2,982

n=3,134

n=3,312

n=3,471

n=3,668

n=3,898

n=4,160

n=4,461

n=4,767

n=5,109

n=5,511

n=5,928

n=6,382

n=6,946

n=7,561

n=8,280

n=9,051

n=9,956

100

10 9

80

22 22

###### Shareoftoolcalls(%)

60

5 6

- 5

5

- 6

5

40

14

13

20

24 23

0

≤-30 -29 -28 -27 -26 -25 -24 -23 -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 (turn end)

Tool call position from end of agent response (natural turns)

Tool

read grep glob bash:ﬁle

bash:build

bash:net

bash

git/gh

write

edit web agent

mcp

TodoWrite ToolSearch AskUserQuestion

Skill

EnterPlanMode

ExitPlanMode

other

(b) Tool call composition counting from the end of natural (non-interrupted) turns.

| |6|7|7|5|6<br><br>7|5<br><br>6|6<br><br>6<br><br>8|7|7<br><br>n=426<br><br>n=474<br><br>n=522<br><br>n=586<br><br>n=657<br><br>n=738| | | | |6| | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8|14|17<br><br>14| |17|14|16 14<br><br>15|15<br><br>16|13 12<br><br>8<br><br>5|13<br><br>8 8|9|10 9|9<br><br>8|13<br><br>7| |24| | |
|6<br><br>11<br><br>6<br><br>15|12<br><br>12<br><br>6|10<br><br>5<br><br>11<br><br>9<br><br>6<br>|12<br><br>6<br><br>18|14<br><br>6|6 6<br><br>8<br><br>11<br><br>7<br><br><br>6<br><br>18|6<br><br>5<br><br>9<br><br>11<br><br>11<br><br>9<br><br>7<br><br>7|6<br><br>6 5<br><br>10<br><br>9<br><br>10<br><br>6<br><br>6 8<br><br>14|5 6<br><br>5<br><br>10<br><br>10<br><br>9<br><br>6<br><br><br>8<br><br>7<br><br>13|6<br><br>5<br><br>8<br><br>9<br><br>8<br><br>9<br><br><br>7<br><br>9<br><br>14<br><br>11|7<br><br>9<br><br>11|6<br><br>8<br><br>8<br><br>8<br><br>8<br><br>10<br><br>10|8 8<br>8 9<br><br><br>11<br><br>11|6<br><br>8<br><br>10|11<br><br>17<br><br>13|8<br>9<br>| | |
|15<br><br>8|14|18<br><br>15<br><br>9|14<br><br>7<br><br>6|15<br><br>10|18<br><br>17<br><br>9 8|19<br><br>19<br><br>15<br><br>9 12<br><br>10|17<br><br>13<br><br>16<br><br>10<br><br>8<br><br>7|13<br><br>15 15<br><br>9 8<br><br>10|14<br><br>14<br><br>14<br><br>9<br><br>10<br><br>10|13<br><br>10|13<br><br>13<br><br>10<br><br>10|13 11<br><br>8<br><br>9|8<br><br>8|9<br><br>6<br><br>9|7<br>8<br><br><br>12| | |
|21|27|22<br><br>25|19|21|18<br><br>20|18<br><br>16<br><br>24|22 23 23|23 24 24|24<br><br>19<br><br>22|23|23 24|24 24|26|13<br><br>5|9<br>10<br>| | |

n=2,446

n=152

100

9

12

80

20

14

###### Shareoftoolcalls(%)

5

7

60

12

- 8 7

7

- 9

40

13

12

20

21 22

0

≤-30 -29 -28 -27 -26 -25 -24 -23 -22 -21 -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 (turn end)

Tool call position from end of agent response (interrupted turns)

Tool

read grep glob bash:ﬁle

bash:build

bash:net

bash

git/gh

write

edit web agent

mcp

TodoWrite ToolSearch AskUserQuestion

Skill

EnterPlanMode

ExitPlanMode

other

(c) Tool call composition counting from the end of interrupted turns.

- Figure 21: Agent tool call trajectories. (a) Tool composition by sequential position within a single agent trajectory. (b,c) Tool composition counting backward from the end of the trajectory, split by natural vs. interrupted turns.

researchers, education) based on its name, description, and README file. As shown in Figures 19c and 19d, most repositories are user-facing applications or developer tools. This distribution highlights that SWE-chat primarily reflects practical, software-engineeringfocused environments rather than purely academic or exploratory programming tasks.

##### D.1.5 Dataset diversity over time

Following the public launch of Entire.io on February 10, 2026, open-source developers quickly started using the tool and pushing their coding agent session data to public GitHub repositories. Figure 22 tracks the cumulative fraction of sessions originating from Entire.io’s own repository. At the time of writing, this repository contributes less than 20% of all sessions in SWE-chat, and the share declines with continuing adoption.

| | |Entire.io launch| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

100%

Cumulativefraction(%)

80%

60%

40%

20%

0%

01 2026− 01− 15 2026− 02− 01 2026− 02− 15 2026− 03− 01 2026− 03− 15 2026− 04− 01 2026− 04− 15

01−

2026−

Date

- Figure 22: Cumulative fraction of sessions originating from the entireio/cli repository over time. Each point shows the running proportion of all sessions collected up to that date that came from this single repository. The dashed red line marks the public launch of the Entire.io tool (February 10, 2026).

##### D.2 Topic distribution

To characterize the range of tasks users bring to AI coding assistants, we perform a topic analysis on all English user prompts in SWE-chat.

##### D.2.1 Topic clustering methodology

Starting from all English prompts, we first remove interruption signals (e.g., “[Request interrupted by user]”), system-injected messages (identified by XML-tag prefixes such as and ), Claude skill invocations, and image references. We then stripped fenced and inline code blocks from all remaining prompts and excluded prompts whose stripped text is shorter than 30 or exceeds 1,500 characters. Finally, we deduplicate prompts on case-insensitive stripped content. We generate sentence embeddings using the all-mpnet-base-v2 model from SentenceTransformers (Reimers and Gurevych, 2019). We embed the code-stripped prompt text rather than the raw text so that embeddings reflect the user’s natural language intent rather than the syntactic structure of pasted code. Before clustering, we reduce the embedding dimensionality from 768 to 20 using UMAP We cluster the reduced embeddings using HDBSCAN* (Campello et al., 2013; McInnes et al., 2017) with min_cluster_size=150 and min_samples=5. This yields 20 clusters covering 57.4% of all prompts, with cluster sizes ranging from 152 to 4,329 (median: 256). The remaining 8,265 prompts (42.6%) are classified as noise, reflecting the diversity of coding session prompts that do not form tight semantic groups. For each cluster, we select the 100 prompts with the highest HDBSCAN* membership probability to generate a topic description, which is shown in Figure 23. The descriptions are generated by gpt-5.4-2026-03-05, using the following prompt:

Below are 100 user prompts sampled from a single topic cluster. Each prompt was written

by a software developer using an AI coding assistant. In a <scratchpad >, do the following:

- 1. List the specific technologies , libraries , and project details mentioned.

- 2. Set all of those aside. What **general software engineering activity** unites these prompts? Think in terms of the development lifecycle: planning , prototyping , implementing features , debugging , testing , deploying , configuring infrastructure , designing APIs , refactoring , reviewing code , etc.
- 3. What makes this activity more specific than just "coding"? Is it about a particular layer of the stack (frontend , backend , infra , data , auth)? A particular phase ( greenfield build vs. maintenance)? A particular style of work (exploratory vs. fixing vs. migrating)?

Then produce a single cluster label (4-10 words). The label must:

- - Be understandable to any software engineer without knowledge of these specific projects
- - Name the engineering activity and, if relevant , the stack layer or workflow phase
- - Never mention specific tools , frameworks , or project names

Format: <scratchpad >your reasoning </scratchpad > Label: <your label >

Prompts: {prompts}

- D.2.2 Findings We identify 20 topic clusters that cover 57.4% of all prompts. The results are displayed in Figure 23.5 Frontend coding (cluster 3) has the largest pushback rate (75%). Cluster 17 mostly consists of very long prompts that often specify multiple tasks, explaining the large agent turn durations.

|Debugging and refin|ing source-control|workflow tooling| | |
|---|---|---|---|---|
|Building and debug|ging multi-agent dev|eloper tooling| | |
|Frontend UI polish|and layout bug fixing| | | |
|Authentication integ|ration troubleshooti|ng and security hard|ening| |
|Existing codebase|architecture audit an|d refactor planning| | |
|Developer tooling w|orkflow design and|implementation| | |
|Backend database|migration planning|and troubleshooting| | |
|Automated test suit|e maintenance and|expansion| | |
|Pre-release QA bug|fixing and UX polis|h| | |
|End-to-end testing|and CI failure triage| | | |
|Building and operat|ing ML experiment|pipelines| | |
|Review-driven bug|fixing and test harde|ning| | |
|Implementing and|debugging AI chat fe|atures| | |
|Voice and audio sub|system developme|nt and troubleshoot|ing| |
|Debugging multi-se|ssion lifecycle and|state management| | |
|Troubleshooting co|ntainerized service|deployments and run|time issues| |
|Backend systems c|ode review and bug|triage| | |
|Cross-platform deve|loper environment|configuration and tro|ubleshooting| |
|Backend feature de|sign review and risk|analysis| | |

0 10 20 30 40

Cluster size (%)

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20 Feature-parity debugging for external agent integrations

Size

|46%|54%| | |
|---|---|---|---|
|43%|57%| | |
|25%|75%| | |
|38%|62%| | |
| |75%| |25%|
| |79%| |21%|
| |73%| |27%|
|47%|53%| | |
|29%|71%| | |
|47%|53%| | |
|51|% 49| |%|
| |88%| |12%|
|31%|69%| | |
| |60%| |40%|
|38%|62%| | |
|36%|64%| | |
| |62%| |38%|
|31%|69%| | |
| | | | |
|49 28%|%|51<br><br>72% non-pu pushb<br><br>| |
|---|
|%shback ack|

0% 25% 50% 75% 100%

Proportion

Pushback rate

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 10 20 30

Turn duration (min)

Agent turn duration

0 25 50 75 100

Session success (0 100)

82 82 82 82

88 82

85

82 82 82 82

72

82 82 82 82

76

82

82

Session success

density median

Figure 23: Topic distribution of user prompts. Each bar represents one of the 20 clusters identified by HDBSCAN*, labeled with a GPT-generated topic summary. The remaining panels show, per cluster, pushback rate, agent turn duration in seconds, and session success score distribution for sessions in which at least 20% of prompts fall within the cluster. Several clusters (1, 8, 10, 15, 20) contain a disproportionate amount of prompts originating from Entire.io’s own repository.

- D.3 Distribution of user personas

- Figure 24 shows the full distribution of user personas across all sessions. In most sessions, users act as expert nitpickers.

5Manual inspection revealed that cluster 12 contains a lot of similar prompts that seem to have been generated automatically. Most other clusters have substantial pushback rates.

| | | | | |
|---|---|---|---|---|
|39.7|%<br><br>33.5|%| | |
| | |19.9|%| |
| | | |7.0|%|

3000

Numberofsessions

2000

1000

0

ExpertNitpickerVagueRequester Other MindChanger

Figure 24: Distribution of user personas.

##### D.4 Coding mode distribution over time

Figure 25 shows the temporal evolution of coding modes. The share of vibe coding sessions has roughly doubled since the launch of Entire’s CLI tool, rising from approximately 20% to over 40%.

100%

100%

Human-only

Agent-authoredcode(%)

Collaborative

80%

80%

Vibe coding

Sessions(%)

60%

60%

40%

40%

20%

20%

Mean agent %

Median agent %

0%

0%

15 2026− 03− 01 2026− 03− 15

01 2026− 04− 15

02−

04−

2026−

2026−

Date

Figure 25: 14-day rolling average of coding modes and agent-authored code.

##### D.5 Code vulnerability analysis with Semgrep

We use Semgrep6, an open-source static analyzer that matches community-curated patterns against source code, and run it with its default –config=auto ruleset. This auto-selects rules based on the languages detected in each snapshot, including Common Weakness Enumeration (CWE), which includes known types of security weaknesses (Martin and Barnum, 2008). For every commit, we extract the repository state before and after the commit, scan each state, and keep only findings inside files that the commit actually modified.

Distribution of introduced vulnerabilities Figures 26 and 27 break down the introduced findings by Semgrep rule and by CWE category, respectively. One rule (JavaScript path joining without sanitization) accounts for most detected vulnerabilities, but the remaining

6https://github.com/semgrep/semgrep

findings include a long tail of rules and CWEs, including externally controlled format strings (CWE-134), missing integrity checks (CWE-353), OS command injection (CWE-78), and SQL injection (CWE-89). Hence, a broad set of vulnerability types is being introduced.

javascript: path join resolve traversal

other

javascript: unsafe formatstring

html: missing integrity

python: dynamic urllib use detected

go: dangerous exec command

javascript: detect non literal regexp

trailofbits: automatic memory pinning

python: sqlalchemy execute raw query

python: insecure hash algorithm sha1

javascript: detect child process

dockerﬁle: missing user

python: subprocess shell true

yaml: run shell injection

javascript: detect replaceall sanitization

html: plaintext http link

0 10 20 30 40

Share of Semgrep rules newly triggered in ﬁles changed by the commits (%)

- Figure 26: Distribution of introduced vulnerabilities across Semgrep rule IDs (top 15 plus other).

0 10 20 30 40

Share of CWE tags on Semgrep ﬁndings newly introduced in ﬁles changed by the commits (%)

CWE-915: Improperly Controlled Modiﬁcation of Dynamically-Determined Object Attributes

CWE-732: Incorrect Permission Assignment for Critical Resource

CWE-327: Use of a Broken or Risky Cryptographic Algorithm

CWE-319: Cleartext Transmission of Sensitive Information

CWE-250: Execution with Unnecessary Privileges

CWE-676: Use of Potentially Dangerous Function

CWE-1333: Ineﬃcient Regular Expression Complexity

CWE-79: Improper Neutralization of Input During Web Page Generation (’Cross-site Scripting’)

other

CWE-94: Improper Control of Generation of Code (’Code Injection’)

CWE-939: Improper Authorization in Handler for Custom URL Scheme

CWE-89: Improper Neutralization of Special Elements used in an SQL Command (’SQL Injection’)

CWE-78: Improper Neutralization of Special Elements used in an OS Command (’OS Command Injection’)

CWE-353: Missing Support for Integrity Check

CWE-134: Use of Externally-Controlled Format String

CWE-22: Improper Limitation of a Pathname to a Restricted Directory (’Path Traversal’)

- Figure 27: Distribution of introduced vulnerabilities across CWE categories (top 15 plus other).

Vulnerability example Figure 28 shows a concrete Python example of a vulnerability introduced by a coding agent in our dataset, together with the Semgrep annotation that flags it.

##### D.6 Agent efficiency

- Figure 29 compares efficiency across coding modes along four dimensions. Vibe coding sessions are consistently less efficient: they consume roughly twice as many tokens and require

- 1

|import subprocess def run_build(target: str) -> str:<br><br>"""Run the project's build step and return stdout.""" cmd = f"make {target}" # Semgrep: python.lang.security.audit.subprocess -shell -true (<br><br>CWE -78) # "Found subprocess function with shell=True. ... # A malicious actor can inject arbitrary shell commands. # Use shell=False instead." result = subprocess.run(<br><br>cmd, shell=True , capture_output=True , text=True) return result.stdout<br><br>|
|---|

- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

- Figure 28: Example of a Python vulnerability introduced by an agent in SWE-chat. The agent builds a shell command by interpolating a user-controlled string (target) into an f-string and then calls subprocess.run with shell=True (line 10). The inline comment shows the Semgrep annotation, which flags a CWE-78 OS-Command-Injection risk because an attacker who can influence target could inject arbitrary shell commands (e.g. "rm -rf ˜"). The standard fix is to pass arguments as a list, e.g. subprocess.run(["make", target]) with shell=False, so they are not reparsed by a shell.

more wall-clock time per 100 committed lines than collaborative sessions. Collaborative coding achieves the best trade-off across all metrics, suggesting that human guidance helps agents produce code more economically.

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

Tokensprocessed(inmillions)

per100committedlinesofcode

=1.4µ

=0.740µ

=1.2µ

Token eﬃciency

0

5000

10000

15000

20000

Humanpromptcharacterswritten

per100committedlinesofcode

=22,364µ

=9,903µ

=14,810µ

Cognitive eﬃciency

| | |
|---|---|
|=80.5µ| |
|=51.3| |
|µ| |
| | |
| | |
| | |

0

20

40

60

80

100

Sessionruntime(min)

per100committedlinesofcode

=82.9µ

=51µ

Time eﬃciency

0

20

40

60

80

Agentruntime(min)

per100committedlinesofcode

=74.0µ

=47.7µ

=71.2µ

Agent runtime eﬃciency

| |
|---|

Human-only

| |
|---|

Collaborative

| |
|---|

Vibe coding

- Figure 29: Token, cognitive, and time efficiency per 100 committed lines of code (lower is better). Y-axis labels describe the metrics used for the efficiency dimension. µ indicates means.

- D.7 Agent turn duration over time

Figure 30 tracks agent turn duration over time. While median turn durations have remained relatively stable, the tail has grown since the beginning of the data collection: the 99.9th percentile now exceeds 100 minutes. This trend suggests a gradual shift toward longer autonomous agent runs.

- D.8 Oversight rates over time

Over the entire data collection period (from January to March, 2026), the shares of agentinitiated stops, user interruptions, and user pushback remain relatively stable. We visualize this with average fractions of turn using a 7-day rolling window in Figure 31.

ClaudeOpus4.6

Turnduration(min,7-dayrollingavg)

p50 p90 p99 p99.9

150

125

100

75

50

25

0

15 2026−

02− 01 2026−

02− 15 2026−

03− 01 2026−

15 2026−

04− 01 2026−

15

01−

03−

04−

2026−

- Figure 30: Turn-level autonomy in Claude Code sessions. Agent turn duration in interactive Claude Code sessions. Showing 7-day rolling average of different percentiles (p50, p90, p99, p99.9) over time.

2026−

01−

15

2026−

02−

01

2026−

02−

15

2026−

03−

01

2026−

03−

15

2026−

04−

01

2026−

04−

15

0

20

40

60

%ofturns(7-dayrollingavg)

Per-Turn Friction Rate Over Time

Agent stop (≥1 AskUserQuestion)

Hard interruption

Any pushback

- Figure 31: Agent stops for clarification, user interruptions, and soft user pushback over time.

##### D.9 Development activities

To examine whether agent behavior differs across development activities, we group intents into code writing (create, refactor, connect) and code reviewing (understand, test) prompts. As visible in Figure 32, on average, code writing prompts trigger longer agent turns (mean 4.1 vs. 2.4 minutes) and more file writes (4% of tool calls create a new file from scratch) and edits (24% of tool calls edit an existing file). Furthermore, writing prompts also elicit more friction than code reviewing prompts: agents stop to ask questions nearly three times as often (6.0% vs. 2.6% of turns), and users also interrupt and push back more frequently (see Section 4.4 for more details).

|21%|11% 5%|22%|15%|8%|
|---|---|---|---|---|
|read grep glob bash:ﬁle<br><br>bash:buil|d<br><br>bash:ne<br><br>bash<br><br>git/gh<br><br>write<br><br>edit|Tool type<br><br>t web<br><br>agent<br><br>mcp<br><br>TodoW<br><br>ToolSea|rite rch<br><br>AskU<br><br>Skill Ente<br><br>ExitP other<br><br>|serQuestion rPlanMode lanMode|
|20%|9% 4%|15% 4%|24%|10%|

Mean

###### 2.6%

Code reviewing

0.8% 4.9%

31.7%

Friction type

| |
|---|

Agent asks Hard interruption Pushback (total)

Correction

| |
|---|

Rejection

Failure report

| |
|---|

6.0%

Code writing

5.1% 37.3%

4.2%

0.7%

0 5 10

0 20 40 60 80 100

0 20 40

% of tool calls

Agent turn duration (min)

% of agent turns

Figure 32: Agent behavior by development activity (code writing vs. code reviewing).

#### E Data annotation

Here we provide all prompts used for the final dataset annotation. We include all validation details and prompts for the annotation tasks we crafted. The prompt intent task is inspired by (Becker et al., 2025) and the user persona task is inspired by Wang et al. (2026b).

##### E.1 Validation

Annotation codebook development and annotator agreement To develop the annotation codebook and a dataset to test LLM annotation performance, we proceeded in three stages for each annotation task:

- 1. First, two annotators iteratively refined the codebook until they agreed on all labels for 10 data points.
- 2. Second, the same two humans proceeded to independently annotate NIAA = 90 additional data points. We computed inter-annotator agreement metrics using the results from this stage. The results in Tables 6 and 7 show that agreement was moderate-high for all tasks. This includes a binary version of the prompt pushback tasks that collapses all classification classes except the non-pushback class. Figure 33 shows the full confusion matrices for all tasks. For session success ratings, we discuss all cases where humans disagree by more than 20 points.
- 3. Finally, the same two humans discussed all disagreements and decided on the most appropriate gold label for each data point. Together with the 10 data points

from stage 1, this yielded Ngold = 100 gold labels for evaluating LLM annotation performance.

We use Cohen’s κ to measure human-human and LLM-human agreement for multi-class annotation tasks, and additionally report percentage agreement in Table 6 (Cohen, 1960). Session success is labeled with a 0–100 score, which is why measure absolute agreement with a two-way random effect, single measurement Intraclass Correlation Coefficient, commonly referred to as ICC(2,1) (Shrout and Fleiss, 1979; McGraw and Wong, 1996). We additionally report Spearman ρ in Table 7 and Figures 33–34 (Spearman, 1961). We use the average of the two human annotators’ session success ratings as the gold standard against which we compare the LLMs. If human ratings differ by >20, the human annotators collectively decide on the most appropriate gold rating. For the LLM vs. human gold rating comparison, we additionally report consistency using a two-way mixed effect, single measurement ICC, abbreviated ICC(3,1) (Shrout and Fleiss, 1979). For the repository-level annotations, we take a slightly different approach. Namely, during stage 2, human annotator 2 reviewed the 100 repository domains and repository audience labels set by annotator 1 and either agreed with or overrode them. All annotators are authors of this paper.

LLM annotation performance We tested 9-11 LLMs and 2-4 prompt paraphrases for each task and evaluated them against the 100 human-annotated gold labels. Table 8 shows the performance results, showing only the best-performing prompt for each model-task combination. We then selected the model with the highest performance. The only exception is the prompt pushback task, where we defer to the second-best-performing model, since

Table 6: Inter-annotator agreement for multi-class tasks.

Task NIAA Categories Annotators Agreement Cohen’s κ

Prompt intent 90 7 2 68/90 (75.6%) 0.709 Prompt pushback 90 4 2 80/90 (88.9%) 0.832 Prompt pushback (binary) 90 2 2 85/90 (94.4%) 0.888 User persona 90 4 2 71/90 (78.9%) 0.662

Table 7: Inter-annotator agreement for continuous session success rating task.

Task NIAA Rating scale Annotators Spearman ρ ICC(2,1) Session success 90 0–100 2 0.757 0.503

###### User Persona (Accuracy=0.79, n=90)

###### Prompt Intent (Accuracy=0.76, n=90)

[Figure 22]

[Figure 23]

9 0 0 3 0 0 1

create new code

37 2 1 5

Expert Nitpicker

- 0 11 2 0 0 0 0
- 1 0 14 0 0 0 0

0 1 0 11 0 0 0

3 1 0 1 9 0 0

0 1 0 0 0 1 0

- 2 1 4 1 0 0 13

debug

Humanannotator1

Humanannotator1

git

6 9 0 1

Mind Changer

other

0 0 1 2

Other

refactor

test

2 0 0 24

Vague Requester

understand

ExpertNitpicker MindChanger OtherVagueRequester

createnewcode debug git otherrefactor testunderstand

Human annotator 2

Human annotator 2

###### Prompt Pushback (Accuracy=0.89, n=90)

Session Success ( =0.76, ICC(2,1)=0.50, n=90)

[Figure 24]

100

18 2 3 1

correction

80

Humanannotator1

Humanannotator1

0 23 1 1

failure_report

60

40

- 0 1 38 0
- 1 0 0 1

non_pushback

20

rejection

0

0 20 40 60 80 100

correction failure_report non_pushback rejection

Human annotator 2

Human annotator 2

Figure 33: Inter-annotator agreement confusion matrices.

qwen-3.5-9b offers a much better cost-performance trade-off than gpt-5.4-2026-03-05. The high cost of this task is due to the large context: for each prompt pushback annotation, we provide not only the user message but also the full session transcript up to that point (see Table 2 and Appendix E.2.4). Figure 34 shows the full confusion matrices for all tasks.

- E.2 Annotation prompts We now list all LLM-based annotation tasks applied to the SWE-chat dataset.

Table 8: Annotation model performance against human gold labels. For each task, we indicate the best-performing model and the

|chosen|
|---|

model, i.e., the one we used for

full-dataset annotation. ‘◦’ indicates models that produced invalid labels and ‘—’ denotes models that were too expensive to run. We use accuracy (acc) for multi-class annotations and ICC(2,1) for numeric labeling tasks.

Sessionsuccess(ICC(2,1))

Repositoryaudience(acc)

Repositorydomain(acc)

Pushback(binary)(acc)

Promptpushback(acc)

Promptintent(acc)

Userpersona(acc)

Model

|0.69|
|---|

gpt-5.4-2026-03-05 0.69 0.74 0.83

0.77 0.79 0.56

gpt-5-mini-2025-08-07 0.74 0.64 0.73 0.69 0.75 0.79 0.46 gpt-5-nano-2025-08-07 0.67 0.63 0.69 0.62 0.68 0.79 0.50

|0.81|
|---|

|0.84|
|---|

claude-opus-4-6 0.70 — — 0.57

—

|0.60|
|---|

claude-sonnet-4-6 0.66 — — 0.46 0.78 0.83 claude-haiku-4-5-20251001 0.69 ◦ ◦ 0.60 0.72 0.79 0.56 gpt-oss-120b 0.71 0.66 0.74 0.63 0.66 0.75 0.51 gpt-oss-20b 0.73 0.63 0.71 0.59 0.63 0.69 0.49 qwen-3.5-27b

|0.76|
|---|

0.66 0.73 0.61 0.62 0.76 0.52 qwen-3.5-9b 0.72

|0.67|
|---|

|0.79|
|---|

0.60 0.55 0.76 0.47 olmo-3.1-32b 0.59 ◦ ◦ 0.51 0.62 0.69 0.43 Ngold 100 100 100 100 100 100 100

##### E.2.1 Repository type classifier Model: claude-opus-4-6.

You are a classifier that categorizes software repositories based on their purpose. You are given:

- - Repository metadata: name , description , primary language , topics/tags
- - A README or context excerpt (may be truncated)

- ## Task A - Domain tags

Assign one of the domain tags from:

- - application - end -user software (web/mobile/desktop/CLI apps)
- - library - reusable code packaged for others to consume
- - devtools - tooling for building/testing/releasing software
- - other - any other category Instructions:
- - Choose exactly one , most likely label.
- - If you are unsure , pick other.

- ## Task B - Target audience Assign one of target audience tags from:

- - enduser
- - developer
- - researcher
- - education (i.e., educators or students)

###### Prompt Intent (Accuracy=0.76, n=100)

###### Prompt Pushback (Accuracy=0.67, n=98)

[Figure 25]

[Figure 26]

12 0 0 0 0 1 1

create new code

18 0 6 1

correction

- 0 14 0 0 2 0 0
- 1 0 16 0 0 0 2

0 1 0 7 2 0 4

- 2 1 0 2 8 0 1

debug

Goldlabel(humans)

Goldlabel(humans)

git

4 18 4 1

failure_report

other

10 2 30 0

non_pushback

refactor

- 0 0 0 0 0 2 0
- 1 0 1 2 0 0 17

test

4 0 0 0

rejection

understand

correction failure_report non_pushback rejection

createnewcode debug git otherrefactor testunderstand

Predicted (qwen-3.5-9b)

Predicted (qwen-3.5-27b)

###### User Persona (Accuracy=0.69, n=100)

###### Repo Domain (Accuracy=0.81, n=100)

[Figure 27]

[Figure 28]

43 5 2 5

Expert Nitpicker

43 1 1

application

Goldlabel(humans)

Goldlabel(humans)

12 2 0 1

Mind Changer

9 27 0

devtools

1 0 2 0

Other

7 1 11

other

3 0 2 22

Vague Requester

ExpertNitpicker MindChanger OtherVagueRequester

application devtools other

Predicted (claude-opus-4-6)

Predicted (gpt-5.4-2026-03-05)

###### Repo Audience (Accuracy=0.84, n=100)

###### Session Success ( =0.600, ICC(2,1)=0.605, ICC(3,1)=0.614, n=100)

100

[Figure 29]

y=x

51 1 4 2

developer

80

Goldlabel(humans)

Goldlabel(humans)

- 0 4 0 0

7 1 27 0

- 1 0 0 2

education

60

enduser

40

20

researchers

20 40 60 80 100

developer education enduser researchers

Predicted (claude-sonnet-4-6)

Predicted (claude-opus-4-6)

Figure 34: LLM annotation agreement with gold labels from human expert annotations.

Instructions:

- Choose exactly one , most likely label. ## Output format Respond in valid JSON with the following format (and no other text):

{ ""domain"": <application|library|devtools|other >, ""audience"": <enduser|developer|researchers|education >, }

Repository information: {input}

We aggregate library and devtools into a single category called devtools, since human annotators often disagreed about which to assign.

- E.2.2 Session persona classifier Model: gpt-5.4-2026-03-05. Parameters: reasoning_ef fort = low.

You are given a chronological timeline of a coding session. The timeline contains:

- - User prompts
- - Model responses
- - AI agent coding actions (file edits , tool calls , tests run, etc.)
- - Commits or diffs
- - Any visible user reactions to outputs Your task is to classify the USER (not the model) into one of the following personas:

- 1) Expert Nitpicker

- - Deep domain knowledge; high and consistent standards
- - Gives precise , technically specific instructions or corrections
- - Notices subtle issues and requests exact adjustments
- - Goal remains stable throughout; corrections refine execution , not direction

- 2) Vague Requester

- - Broad strokes only; underspecified goals
- - Missing constraints; leaves many decisions to the model
- - Does not correct or redirect in detail

- 3) Mind Changer

- - Shifts the overall goal or requirements mid-session
- - Contradicts or reverses earlier instructions
- - Corrections change direction , not just execution details

- 4) Other (use only when the session is too brief to judge , or the user 's behavior clearly does not match any of the above)

Disambiguation:

- - Expert Nitpicker vs Mind Changer: an Expert Nitpicker corrects HOW the model executes a stable goal; a Mind Changer revises WHAT the goal is. Repeated precise

corrections within the same goal = Expert Nitpicker.

- - Expert Nitpicker vs Other: prefer Expert Nitpicker over Other when the user shows specific technical demands , even if corrections are infrequent.

INSTRUCTIONS:

- - Analyze the full sequence of actions in the session.
- - Focus on patterns across time , not a single prompt.
- - Use evidence from: constraint specificity , goal stability , revision behavior , error tolerance ,

explicit vs vague expectations.

- - If multiple personas seem plausible , choose the dominant behavioral pattern.
- - Do not over -index on one isolated incident.
- - Reserve "Other" for sessions where no pattern is discernible.

OUTPUT FORMAT (JSON only): Return exactly one JSON with the following keys:

"label": "<one of: Expert Nitpicker | Vague Requester | Mind Changer | Other >", "confidence": 0.0-1.0, "reason": "2-5 sentence explanation referencing specific behavioral patterns observed

in the timeline." Do not include anything outside the JSON. INPUT: <input >

##### E.2.3 Prompt intent classifier

Model: Qwen/Qwen3.5-27B. We use suggested decoding parameters: temperature = 0.7, top_p = 0.8, top_k = 20, presence_penalty = 1.5.

You are a classifier that categorizes developer prompts based on their primary intent. Your task is to read a user prompt and assign it exactly one of the following labels:

- - create new code - The user is asking to generate new functionality , write a script , implement a feature , or build something from scratch.
- - refactor - The user wants to modify , restructure , optimize , clean up, or improve existing code without fundamentally changing its purpose.
- - debug - The user is diagnosing an error , investigating unexpected behavior , fixing a bug , or resolving a crash.
- - understand - The user is asking for an explanation , walkthrough , summary , or clarification of existing code or behavior.
- - connect - The user wants to integrate systems , connect APIs , wire components together , or build glue code between tools/services.
- - git - The user is asking for help with version control , branching , merging , resolving conflicts , commits , or other git-related tasks such as fixing git-related errors.
- - test - The user is asking to write tests , add validation , benchmark , evaluate performance , or verify correctness.
- - other - The prompt does not clearly fall into any of the above categories. Instructions:
- - Choose exactly one label.
- - Select the label that best represents the primary intent of the prompt.
- - If multiple intents are present , prioritize the main action requested.
- - Respond in valid JSON with the following format: { "label": "<one of the labels above >", "reason": "<1-2 sentence explanation of why this label was chosen >" }
- - Do not include any text outside the JSON object.

Now classify the following prompt: {input}

##### E.2.4 User pushback classifier

Model: Qwen/Qwen3.5-9B. We use suggested decoding parameters: temperature = 0.7, top_p = 0.8, top_k = 20, presence_penalty = 1.5.

You are a classifier that determines whether a user prompt in a coding agent session represents pushback against the agent 's preceding action , and if so, what kind.

Pushback is any prompt where the user resists , corrects , redirects , or takes over from the agent - rather than simply continuing the workflow. Use the preceding conversation context to understand what the agent just did.

Classify the prompt into exactly one of the following categories:

- - correction - The user redirects the agent by providing missing context , correcting a misunderstanding , pointing out factual errors , or changing requirements/direction/ scope mid -task.

Examples: "I said X not Y", "you changed the wrong file", "actually the API uses POST

not GET", "actually , let's do X instead", "forget that approach , try Y", "on second thought , skip the tests"

- - rejection - The user explicitly rejects , reverts , or refuses the agent 's output without providing a specific correction.

Examples: "undo that", "revert the last change", "no", "that 's wrong", "I don't want that", "put it back the way it was"

- - failure_report - The user reports that the agent 's output does not work: bugs , errors , test failures , or broken behavior.

Examples: "this still doesn 't work", "it's still crashing", "same error , try again", "the tests are failing", "I get a 404 now"

- - non_pushback - The prompt moves the session forward normally: a new task , building on agent output , asking a question , or routine iteration.

Examples: "now add a login page", "good , also add unit tests", "why did you use a

list here?", "change the button color to blue" Disambiguation:

- - correction vs rejection: correction provides a specific fix, missing information , or new direction; rejection just says "no" or "undo" without explaining what was wrong.
- - failure_report vs rejection: failure_report = "it doesn 't work" (something is broken) ; rejection = "I don't want that" (output is unwanted even if functional).

When uncertain:

- - If the prompt contains words like "undo", "revert", "wrong", "broken", "doesn 't work ", "I said", "you missed", or "never mind", lean toward a pushback category.
- - If the prompt reads like a standalone next step with no negative reaction , lean toward non_pushback.

Respond in valid JSON only: { "label": "<one of: correction , rejection , failure_report , non_pushback >", "reason": "<1-2 sentence explanation >" }

Preceding conversation context: {context}

User prompt to classify: {input}

##### E.2.5 Session success rating Model: claude-sonnet-4-6.

You are scoring the overall success of an interactive coding session between a human

user and an AI coding agent. You will receive:

- 1. The full conversation transcript (user messages and agent responses).
- 2. A summary of tool calls made during the session (category counts , top tools).
- 3. Commit information , if any (commit messages and diff summaries). ## Evaluation procedure First , analyze the session along five dimensions. For each , note concrete evidence from

the transcript.

- 1. **Goal completion**: Did the agent fulfill what the user asked for? Identify every distinct user request or task. For each , judge whether it was fully resolved , partially resolved , or unresolved.
- 2. **Final session state**: How did the session end? A natural conclusion (user confirms satisfaction , moves on to a new topic , or signs off) is positive. An abrupt stop (user abandons mid -task , expresses frustration , or silently disengages

after an unresolved error) is negative. Weigh the ending heavily - a session that goes well for many turns but ends in an unresolved failure should score

substantially lower than one that resolves cleanly.

- 3. **Agent efficiency**: Did the agent make steady progress , or did it spin? Negative signals include: the user repeating the same instruction or correction , the agent retrying a failed approach without changing strategy , and unnecessary tool calls that do not advance the task. Positive signals include: appropriate use of research tools before acting , surfacing uncertainty and offering the user choices ,

and recovering from errors autonomously.

- 4. **Code and commit quality**: If code was produced , does it appear correct and complete based on the available evidence (test results , diff content , user reactions)? Were changes committed? Commits are a strong positive signal for task oriented sessions but are not required for short advisory or exploratory sessions where no code change was the expected outcome.
- 5. **User experience**: Did the user have to fight the agent , or did the interaction flow naturally? Look for signs of satisfaction (thanks , approval , moving to the next task) and dissatisfaction (re-explaining , correcting , expressing frustration)

.

## Scoring rubric After analyzing the five dimensions , assign a single integer score from 0 to 100.

- - **90-100**: Every user request was fully resolved. The final session state is clean ( code works , commits are sound , no dangling errors). The user 's last messages indicate satisfaction or natural conclusion.
- - **70-89**: The core user goal was met, but minor issues remain (e.g., an edge case unhandled , a small follow -up the user would still need to do). No major unresolved

errors. The session ended on a reasonable note.

- - **50-69**: Meaningful progress was made on the primary goal , but at least one significant sub -task is incomplete or the solution has a known defect. The user may have had to redirect the agent more than once on the same point.
- - **30-49**: The agent produced some relevant output , but the primary user goal was not achieved. The codebase may be in a worse or broken state , or the user visibly

gave up on the main task.

- - **10-29**: Very little useful work was accomplished. The agent struggled repeatedly , required heavy user correction , or went down unproductive paths for most of the session.
- - **0-9**: Nothing of value was produced. The session ended with no progress toward any user goal.

When a session is ambiguous (e.g., short , exploratory , no clear deliverable), score based on whether the agent 's responses were the most helpful thing it could have provided given the input. Do not penalize the agent for the user 's request being vague , but do penalize it if it failed to make productive use of whatever information was available.

## Output format Respond with valid JSON only. Do not include any text outside the JSON object. {

"score": <integer 0-100>, "reason": "<2-4 sentences: what was accomplished , what issues were present , and why

this score.>"

} ## Session data {input}

