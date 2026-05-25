# State of GPT

[Microsoft Developer: State of GPT - YouTube](https://www.youtube.com/watch?v=bZQun8Y4L2A)<br>[Slides - Karpathy.ai](https://karpathy.ai/stateofgpt.pdf)<br>Notes by [mk2112](https://github.com/mk2112)

---

**Table of Contents**

- [How To: Train A GPT Assistant](#how-to-train-a-gpt-assistant)
	- [Pretraining](#pretraining)
		- [Data Gathering](#data-gathering)
		- [Tokenization](#tokenization)
		- [Pretraining with Batches](#pretraining-with-batches)
	- [Supervised Finetuning](#supervised-finetuning)
	- [Reward Modeling](#reward-modeling)
	- [Reinforcement Learning](#reinforcement-learning)

- [How To: Use A GPT Assistant](#how-to-use-gpt-assistance)
	- [Self-Consistency](#self-consistency)
	- [Present Notion of Past Mistakes](#present-notion-of-past-mistakes)
	- [Going Experimental](#going-experimental)
	- [Prompt Enrichment and Tool-use](#prompt-enrichment-and-tool-use)
	- [Constrained Prompting](#constrained-prompting)
	- [Finetuning](#finetuning)
	- [Default Recommendations](#default-recommendations)

---

## How To: Train A GPT Assistant

"GPT customization" is an emerging technology to adapt GPTs (Generative Pretrained Transformer networks) to a user's needs and usage patterns.<br>A current approach for customization is based on this multi-stage process:

- **Pretraining:**
	- *Dataset:* Raw internet scraped text, trillions of words with low task-specificity, in high quantity
	- *Algorithm:* Next token prediction
	- *Result:* Base model
- **Supervised Finetuning (SFT):**
	- *Dataset:* Q-A-style behavioral demonstrations (10K to 100K), human-written, high specificity, low quantity
	- *Algorithm:* Next token prediction
	- *Result:* SFT model *(this could be deployed)*
- **Reward Modeling:**
	- *Dataset:* Comparisons, may be written by human contractors
	- *Algorithm:* Binary Classification (Bad Answer vs. Good Answer as labeled by a human)
	- *Result:* RM model
- **Reinforcement Learning (RL):**
	- *Dataset:* Prompts (10K to 100K), may be written by human contractors
	- *Algorithm:* Reinforcement Learning (Generate tokens that maximize a perceived reward)
	- *Result:* RL model *(this could be deployed)*

### Pretraining

Pretraining is the stage where a GPT language model (LM) learns to predict the next token from massive amounts of internet-scale text.<br>
This incentivizes the compressing of patterns of language, knowledge, reasoning, and world structure into the LM weights.<br>

Most of the computational complexity involved in creating aligned LLMs occurs during pretraining.<br>
It can require many 1,000s of GPUs, months of training and millions of dollars in expenses.<br>
Importantly, pretraining on large amounts of data necessitates the availability of said data in suitable quality.

#### Data Gathering

The core competency of the pretraining step arguably lies in data acquisition and curation, i.e. the gathering of large-scale datasets and their transformation into a clean, unified training format, like listed below for LLaMA LM family.

<img src="./img/Pasted%20image%2020231123162506.png" width="300" height="auto"/><br>Source: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

#### Tokenization

<b>Suppose we've got the text. Now what?</b><br>Given that GPTs are mathematical models, requiring numeric inputs, we have to find a way to encode our training data meaningfully into a numeric representation. Tools like the [OpenAI Tokenizer](https://platform.openai.com/tokenizer) help with that. Specifically, algorithms like the state-of-the-art [Byte Pair Encoding](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) are employed.

<img src="./img/Pasted%20image%2020231123163046.png" width="400" height="auto"/>

> A good tokenizer ensures the numeric representation to be both lossless and unique for any text.

<img src="./img/Pasted%20image%2020231123163629.png" width="400" height="auto"/><br>Source: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

Interestingly, LLaMA achieves much higher performance than GPT-3 while being smaller in parameter count (at $175B$ parameters). This is due to longer training runs and factors such as improved data quality and model architecture. LLaMA cost $\$5 \text{million}$ to train, requiring $2,048$ NVIDIA A100 GPUs to be run for $21$ days [\[Touvron, et al. 2023\]](https://arxiv.org/abs/2302.13971) for the base LLaMA model.

#### Pretraining with Batches

Given such a training setup and given we attained the pretraining dataset, we now need to shape and mend the text data so as to most efficiently expose the model to its contained information.

**We define:**
- $B$ as the batch size (e.g. $4$ entries per batch)
- $T$ as the maximum context length provided by an entry in the batch (e.g. $10$ tokens)
- A special token $<|endoftext|>$ to be part of the $T$ tokens within entries in the batch, denoting the contextual end of documents from the training set
	- The special token can appear in an entry, but doesn't have to appear in every entry
	- Equally, the special token can be the last token of an entry, but doesn't have to be

<img src="./img/Pasted%20image%2020231123164549.png" />

$<|endoftext|>$ representation is marked red
<br>

> GPTs (Generative Pre-trained Transformers) are based on the [Transformer](https://arxiv.org/abs/1706.03762) architecture. Within a size-restricted, moving context window, the Transformer takes in previously experienced inputs and evaluates a current input in their context, without the prior inputs from said window losing quality in affecting current input interpretation based on the distance between them.

Think of this as feeding contexts and not (just) individual tokens into the GPT.<br>
Given both a token and its provided predecessors within the context, what's the next token suggested by the model and how does this suggestion compare to what's actually as stated in the dataset?

<img src="./img/Pasted%20image%2020231123170040.png" />

The Transformer-based GPT now gets exposed to contextual, supervised learning. This makes GPT generate a continuous probability distribution over the entire vocabulary for each position in the sequence. The information on what token actually came next, compared to the prediction, causes an iterative optimization. If this is done sensibly and the model actually can take away how to adapt its internal distribution representations, you (hopefully) get see something like this:

<img src="./img/Pasted%20image%2020231123172258.png" width="300" height="auto"/>

The (gradually) lower the training loss, the (gradually) better the model.

### Supervised Finetuning

With all that money and time spent on the large-scale exposure of data to the model, we have ... not attained any task-specificity in the model's behavior yet.<br><b>The model got trained to predict the next token. If it can do that, great, we got it to do what we wanted, but 'predicting the next token from a large dataset' essentially means that now the model 'parrots' the textual structures from said training set as best as possible.</b><br>And because 'parroting well' was the sole objective for pretraining, the model learnt to model the textual information from the training data, but it can't yet answer to given questions or solve tasks like an AI assistant. However, we know that ChatGPT, LlaMA, Open Assistant etc. can indeed respond to questions and actually solve tasks. **What are we missing?**

To provide our model with an assistant-like behavior, we derive an additional Question-Answer-style dataset through human contractors.<br>Think of this as a high quality, low quantity dataset.<br>The model learned general language patterns, token correlations and meaning in a broad, general fashion from pretraining. Now we want to put this broad experience to use through a specific model behavior (answering to questions, not blabbering on). For that, we pretty much continue using the setup we had for pretraining, but now with the new Question-Answer dataset, providing a question as input, and expecting the answer to be generated by the model. This is now called **finetuning**.

> Pretraining allows the model to understand language. Finetuning without pretraining first would be useless as the model wouldn't have a grasp of what words mean, how they relate, or how to structure coherent output. Pretraining teaches exactly those well needed fundamentals and does so across a vast range of contexts, exposing the model to general syntax, semantics, and world knowledge. Finetuning then, on top of that, repurposes that task-agnostic foundation into something narrower, behavior-specific, like answering questions.

<img src="./img/Pasted%20image%2020231123175113.png" width="400" height="auto"/>

Completing this second step results in a "Supervised Fine-Tuning" model (SFT model). This model could be made available as-is. In practice, though, this is not viewed as sufficient. SFT may not fully capture the complexity and diversity needed for successful fine-tuning, especially for tasks requiring specialized knowledge or nuanced understanding. However, this challenge can be addressed through additional **Reward Modeling**.

### Reward Modeling

To continue with the training pipeline, we can expose the SFT model to Reward Modeling.<br>
When combined with Reinforcement Learning, this is also known as [RLHF](https://openai.com/research/learning-from-human-preferences).

Reward Modeling is based on improving through user feedback based on ranking answers. The SFT model produces a set of possible answers for a single prompt. The answers then are compared in their quality by a human, ranking them from best to worst. Think of this as making sure the model is aligned well.

<img src="./img/Pasted%20image%2020231123181520.png" width="650" height="auto"/>

Suppose that now, between all possible pairs of these potential answers, we perform binary classification.<br>To do so, we lay out the (always identical) prompts concatenated with the different responses, and we add a specific $<|reward|>$ token at the end of each response.<br>
We go ahead and feed this human-annotated dataset into *another language model*, the *Reward Model* (RM). This is a smaller model whose sole task is to learn to predict the human-derived ranking of the responses made by our actual language model. The RM effectively compresses human judgments into a scalar reward signal, produced at the $<|reward|>$ token position, which serves as an estimate of response quality by our actual language model.

The human ranking data now becomes relevant for training the RM. The data is used to supervise the scalar outputs, pushing scores up for preferred responses and down for less preferred ones. In this way, the RM is trained to map prompts and responses into a single ordered notion of quality, yielding an optimized Reward Model that approximates human preference over outputs.

Note that the *Reward Model* itself is quite small, and not really useful. However, coupled with our LLM, it shines in what follows now: Reinforcement Learning.

### Reinforcement Learning

Again, a large set of prompts, i.e. inputs to our LLM, is acquired from human contractors. Low in quantity, high in quality.
We expose our LLM to it, again producing multiple answers per prompt. But now we keep the *Reward Model fixed*. It was trained and now serves as reasonably dependable indicator for response quality of our large LLM. We do not need the ranking of answers to be done by a human anymore.

<img src="./img/Pasted%20image%2020231123184003.png" />

> With the predictions of the Reward Model, we attain a guide by which to enforce the prediction of one certain response over others, making the best-ranked answer's associated token prediction more likely to occur. This concludes the RLHF pipeline as applied e.g. to GPT-3.5.

Interestingly, RLHF-trained models gain in perceived quality of response and contextual reference, but tend to play it safe on the entropy side. This means that they become increasingly likely to select the *most preferred* next-token predictions. This partially stems from maximizing positive feedback in the RM/RL stages, which turns an LLM risk-averse, making it favor well-established and commonly accepted responses. This, by the way, is a key indicator for detecting AI-generated text.

## How To: Use A GPT Assistant

Human-written sentences are interesting. They reach deep into both the author's and the reader's perception, experience and skillsets. 

$$California's\ population\ is\ 53\ times\ that\ of\ Alaska.$$
This sentence is the crescendo of a not so trivial thought process:

<img src="./img/Pasted%20image%2020231123185618.png" width="450" height="auto" />

See how the thought process concerns a writing process, but also a process tasked with reassuring factual correctness through tool use and correcting already written text?

This is not how GPTs work. There exists no internal dialogue, no extensive reasoning as such (only shallow at best), no self-correction, no tool-use.<br>
The LLM we trained, finetuned and RLHF'd so far will not reason in its current configuration.

### Self-Consistency

The notion of self-consistency, coming up with several approaches and disregarding some, learning from that, accepting others, learning from those and doing all that independently, is *really* remarkable.

<img src="./img/Pasted%20image%2020231123190203.png" width="450" height="auto" /><br>Source: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)

### Present Notion of Past Mistakes

GPT-4 can actually reflect on past answers and apologize for prior unfavorably sampled responses.<br>Ideally, though, we shouldn't need the model to apologize and instead explore the sampling space by itself in the first place, coming up with a best-aligned, best quality-measured answer as a result of that.

<img src="./img/Pasted%20image%2020231123190700.png" /><br>Source: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

### Going Experimental

A certain sense of self-reflection is emerging already, as can be seen on [AutoGPT](https://arxiv.org/abs/2306.02224).<br>AutoGPT is an application of GPT-4 and GPT-3.5, creating an environment of a task creation agent, a task execution agent and a task prioritization agent, working in conjunction to generate, interlink and process self-prioritized tasks. It's a study on the extents to which unsupervised interaction with the environment is possible with current LLMs.

<img src="./img/Pasted%20image%2020231123211005.png" width="450" height="auto" /><br>Source: [Auto-GPT for Online Decision Making: Benchmarks and Additional Opinions](https://arxiv.org/abs/2306.02224)

### Prompt Enrichment and Tool-use

To get back to the notion of 'How To', we have to be aware that **an LLM by itself is satisfied fully by imitation, not through task-specific contribution.** Additionally, a training dataset might contain multiple different perspectives on a solution to a potential prompt. The LLM on its own has no way of differentiating the qualities of answers. **This is worth remembering when prompting.**

Interestingly, recent advancements worked towards addressing this.<br>**The token vocabulary of ChatGPT contains special, additional tokens.** Given a prompt, an interpreter will read these special tokens, and based on them, trigger the call of external APIs, fetch the results, and concatenate them with the original prompt. **This allows for lifting the restriction of a knowledge cut-off date in the training data. Data now can just be fetched and added from the web.** This approach also works in favor of reducing (not eliminating) factual inconsistencies, e.g. through integration of a calculator API.

> LLMs that incorporate the use of tools are commonly referred to as Retrieval-augmented language models (RAG LMs).

<img src="./img/Pasted%20image%2020231123212252.png" width="300" height="auto" /><br>Source: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

The capabilities of an LLM depend on its memory (context window) as well. We have to place all relevant information about a task in said memory for the LLM to be able to work with it and perform best. Tool-use can help here, but how can tools be most suitably integrated?

Emerging right now, LlamaIndex is a data framework facilitating the integration of custom data sources with LLMs. Serving as a central interface, it enables said LLMs to ingest, structure, and access private or domain-specific data. For this, LlamaIndex provides essential tools, including data connectors, indexes, and application integrations, providing a central streamlining platform for ingestion, structuring, and integration of data with LLMs. Think of LlamaIndex as a bridge, enhancing both accessibility and usability of custom data (sources) for (custom) LLM tasks.

### Constrained Prompting

Another emerging application is *constrained prompting*, meaning requesting very specific, contextually appropriate information.

```json
{
	"id": "{id}",
	"description": "{description}",
	"name": "{gen('name', stop='"')}",
	"age": {gen('age', regex='[0-9]+', stop=',')},
	"armor": "{select(options=['leather', 'chainmail', 'plate'], name='armor')}",
	"weapon": "{select(options=valid_weapons, name='weapon')}",
	"class": "{gen('class', stop='"')}",
	"mantra": "{gen('mantra', stop='"')}",
	"strength": {gen('strength', regex='[0-9]+', stop=',')},
	"items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
}
```

<img src="./img/Pasted%20image%2020231123214126.png" width="350" height="auto" /><br>Source: [Guidance-AI, GitHub](https://github.com/guidance-ai/guidance)

### Finetuning

Finetuning a model means changing its weights through exposure to a comparatively small dataset with the aim of inducing task-specificity in a more broadly trained base model. However, the larger models, the more complex it becomes to finetune them.

However:
- Parameter-Efficient Finetuning (PEFT) emerges, e.g. with [LoRA](https://arxiv.org/abs/2106.09685) making sure to *only partially expose the model and clamp the rest* as needed. This approach works very well and also makes finetuning a lot cheaper.
- High-quality base models emerge, affording more and more specific finetuning, making the resulting models more efficient to use

### Default Recommendations

**Use cases and things to remember:**
1. Models may be biased 
2. Models may fabricate ("hallucinate") information 
3. Models may have reasoning errors 
4. Models may struggle in classes of applications, e.g. spelling-related tasks 
5. Models have knowledge cutoffs (data was only collected until that point, e.g., until September 2024)
6. Models are susceptible to prompt injection, "jailbreak" attacks, data poisoning attacks, etc.<br>
(7. Models are not transparent, and vaguely explainable in their decision-making process at best)<br>
(8. Models may be unstable, leading to different responses on different runs)

**Goal 1: Achieve top possible performance**
- Use GPT-4 (Turbo) or newer available models
- Use prompts with detailed task context, relevant information, instructions
	- "what would you tell a task contractor if they can’t email you back?"
- Retrieve and add any relevant context or information to the prompt
- Experiment with prompt engineering techniques (see above)
- Experiment with few-shot examples that are 
	1) relevant to the test case, 
	2) diverse (if appropriate) 
- Experiment with tools/plugins to offload tasks difficult for LLMs (calculator, code execution, ...) 
- If prompts are well-engineered (work for some time on that): Spend quality time optimizing a pipeline / "chain"
- If you feel confident that you optimized your prompts as much as possible, consider SFT data collection + finetuning
- Expert / fragile / research zone: consider RM data collection, RLHF finetuning 

**Goal 2: Optimize costs to maintain performance** 
- Once you have the top possible performance, attempt cost saving measures (e.g. use GPT-3.5, find shorter prompts, etc.)

**Recommendations:**
- Use in *low-stakes applications*, combine with human oversight 
- Source of inspiration, suggestions 
- Copilots over autonomous agents
