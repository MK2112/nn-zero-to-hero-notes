# The busy person's intro to LLMs - Director's Cut

[YouTube Video - Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)<br>[Slides, PDF - Google Drive](https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view?pli=1)<br>[Slides, Keynote - Google Drive](https://drive.google.com/file/d/1FPUpFMiCkMRKPFjhi9MAhby68MHVqe8u/view)<br>Notes by [mk2112](https://github.com/mk2112)

---

**Table of Contents**

- [Large Language Model (LLM)](#large-language-model-llm)
	- [Inference](#inference)
	- [Training](#training)
	- [Network Interaction](#network-interaction)
	- [Network Architecture](#network-architecture)
	- [Requiring Management Assistance](#requiring-management-assistance)
	- [Performance Evaluation](#performance-evaluation)

- [Back to the future](#back-to-the-future)
	- [Scaling LLMs](#scaling-llms)
	- [Tool-use](#tool-use)
	- [An Academic Perspective](#an-academic-perspective)

- [Jailbreaks and Security Challenges](#jailbreaks-and-security-challenges)

---

## Large Language Model (LLM)

### Inference

A large language model requires just 2 files. If you look at the [Llama-2-70b](https://ai.meta.com/llama/) model by MetaAI, the name tells a lot about its makeup: It's the second iteration of the Llama model series, with 70 billion individual model parameters as part of it. As of now, this exact model is the most powerful one with openly available weights, meaning the specific values to use for all the parameters are known publicly.

> This public approach is in contrast to e.g. OpenAI's ChatGPT, where the only thing a user may see and interact with are the inferences and inputs. Weights are not shared here.

Llama-2-70b consists of just two files: 
- `parameters`: This file is ~140 GB (2 Bytes per weight, datatype is Float-16) and houses the weights, meaning the representations of the 70b parameters
- `run.c`: Within this file, a very compact code allows for training and interaction with the parameters through inference. Here, the programming language used is called `C`, but languages like `Python`, `C++` or `Julia` can theoretically be used here as well

> Beware that this duality of two files is enough to house the entire model. You could download these files (on say your M2 MacBook), run the model and this would work just fine.

To really drive the point home, one could just cut the internet access, then ask the model to describe a specific company or come up with a recipe or anything like that, and the model would answer. This is because inference is done solely based on parameters. Text is solely generated based on parameters. Nothing else matters to Llama.

<img src="images/Pasted%20image%2020231123104403.png" width="250" height="auto" />

> With LLMs, the complexity lies in attaining the parameters based on which the model can generate text perceived as useful.

### Training

The inference process could be perceived as logically simple. Not so with the training process used to attain the parameters. There's no inference without training first. Training is so complex that, other than inference, running it on your laptop is not advised.

Interestingly, MetaAI [published how they trained Llama 2 exactly](https://arxiv.org/abs/2307.09288).<br>First, we need text for the model to get exposed to and to learn based upon. This is done by crawling the web, downloading ~10 TB of text.<br>The untrained model is exposed to this huge set of text on what's called a GPU cluster. Think of this as a set of servers, each running multiple [specialized graphics cards or graphics processing units (GPU)](https://www.nvidia.com/en-us/data-center/a100/) (not obtainable at BestBuy). As it turns out, specialized GPUs are the best hardware we have for training. MetaAI used 6,000 GPUs for 12 days, which cost them around $2 million. Rookie numbers by standards of closed-source models.

> Remarkably, this complex setup aims to distill knowledge about the ~10 TB of text into our desired set of parameters. You can think of this process as lossy knowledge compression.

### Network Interaction

The core task of the LLM is to find the most likely next word, given a context, a set of words.

<img src="images/Pasted%20image%2020231123111305.png" width="500" height="auto" />

There is a close relationship here between the prediction made and the compression/weights used to attain it. A good set of weights can predict the next word from our training set, where we know the next word in each case beforehand, better than a worse set of weights. Thus, the better set of weights more closely represents the data, leading us to be able to use the analogy of compression through weights.

If your objective is next word prediction, your parameters should encode the varying importance of certain words in the input text sequence. If the model can recognize contextually important passages like shown below, they can affect the output likelihood more fittingly.

<img src="images/Pasted%20image%2020231123112449.png" width="350" height="auto" />

> The magic of LLMs is repeating the next word prediction over and over, making the most recent predicted word part of the input sequence to generate a next word again and again. Predictions based on inputs contribute to inputs forming predictions based on inputs contribute to inputs...

<img src="https://api.wandb.ai/files/darek/images/projects/37727390/9ec381c5.gif" width="500" height="auto" />

The process of taking output and concatenating it to the former input to form the next input is referred to as 'dreaming'. This is one of the reasons why e.g. OpenAI states for ChatGPT that "ChatGPT can make mistakes. Consider checking important information." Statements by the LLM like DOIs, ISBNs and dates are not based on fact, as they should be, but entirely on perceived likelihood in a given context. The LLM 'parrots' what it thinks fits best based on what it has seen in the training data. Some outputs thus may be factually correct, some others may only seem like it. It's lossy compression at work, basically.

If this sounds interesting, I refer you to [Karpathy's Makemore series](https://www.youtube.com/watch?v=PaCmpygFfXo), where the process of next character prediction gets implemented and discussed in detail.

### Network Architecture

Buckle up. Today's LLMs share a common building block, making up large parts of the total model:

<img src="images/Pasted%20image%2020231123114140.png" width="450" height="auto" />

This is the [Transformer](https://arxiv.org/abs/1706.03762). This building block is perfectly well described and understood in its mathematical implications. The transformer iteratively affects the model parameters to better represent likelihoods for correct next words. We can measure that the Transformer does that. We only have some barebone ideas as to how the parameters collaborate to come up with the likelihood.

> Seriously, think of LLMs as models that output chains of perceived likelihoods. LLMs are no databases. Think of LLMs as (for now) mostly inscrutable artifacts, and develop correspondingly sophisticated evaluations.

If you're interested in Transformers beyond this, [Karpathy's video on GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) might be a good resource.

### Requiring Management Assistance

Ok, imagine we now have setup a Transformer-based model that we already exposed to Terabytes of scraped training texts, optimizing the model for next word prediction. At best, this makes our LLM a proficient document generator.

<img src="images/Pasted%20image%2020231123115954.png" />

The LLMs behind ChatGPT, Llama or Open Assistant however are not limited to this. You can provide them with a question and receive an answer. To enable this behavior, essentially we continue with the training, but swap out the data. Specifically, a human-written dataset of questions as input and answers as output is derived.

> Think of this as the second stage of a [Transfer Learning](https://www.informatica.si/index.php/informatica/article/view/2828) process. The first stage was high quantity, low task-specific quality. The second stage now provides less quantity, but task specification. This is a special case of transfer learning, called **Finetuning**. For OpenAI, the process is outlined in [this paper](https://arxiv.org/abs/2203.02155).

<img src="images/Pasted%20image%2020231123122521.png" width="300" height="auto" />

The fact that Finetuning works so well is remarkable. The reason as to why however is not well understood. Knowledge and task-specificity come together here.

<img src="images/Pasted%20image%2020231123122701.png" width="400" height="auto" />

On collection of misbehaviors: We monitor an assistant model in its Q-A capabilities. If answers are not as we intend them to be, we make the human feedback loop provide a correct A, given Q. This is then added to the weekly finetuning loop.

The above discussed Llama-2 series was released containing both the base models and already finetuned models, providing you a basis for your own, cheaper, finetuning.

*Wait, there's more.*<br>The state of the art in finetuning involves a third stage. This is based on reasoning that providing two answers and having the user select the more fitting one is cheaper. This behavior can be encountered sometimes e.g. in ChatGPT. At OpenAI, this third stage is called [RLHF](https://openai.com/research/learning-from-human-preferences).

### Performance Evaluation

After this complex, resource-intense training and finetuning process, LLMs can be compared to one another. With [Chatbot Arena](https://chat.lmsys.org/) for example, different LLMs receive what's called an *Elo rating*, similar to chess. 

The derivation process for said ratings is further outlined in [this paper](https://arxiv.org/abs/2306.05685). Essentially, a human user receives two outputs from unknown chatbots and selects the better one.

<img src="images/Pasted%20image%2020231123125556.png" /><br>Source: [huggingface.co](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

> Closed models work a lot better, but you can't work with them freely. 'FOSS vs. Proprietary' is on.

## Back to the future

### Scaling LLMs

The accuracy of the performance in a next word prediction task is a well-predictable function of:
- $N$, the number of parameters in the model
- $D$, the amount of text to train on

<img src="images/Pasted%20image%2020231123131541.png" width="320" height="auto" /><br>Source: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

> With an increased amount of training FLOPS ($D$), the training loss trends downward when scaling the parameter count $N$ up.

So putting algorithmic progress aside, we can boost performance reasonably with tools we have already.
Interestingly, we don't really ever care about next word prediction as-is beyond the initial training step, but the performance of finished models on this task serves as good indicator for higher-level evaluations.

> This is driving the gold rush we see today. It's not a race of algorithmic complexity, but pure resources, time and scale. Confidence is generally high in the field that just these very very simple (not easy) adjustments result in SOTA (for now). 

### Tool-use

With AI tools like [ChatGPT on GPT-4 Turbo](https://chat.openai.com/) or [Perplexity.Ai](https://www.perplexity.ai/), additional sources of information beyond a training or finetuning set include a real-time internet connection. ChatGPT emits special key words that the backend can interpret as a call for wanting to use Bing Search. The backend emits and collects the results of said search and hands them to the model, which incorporates it for the response.

> This 'special word' gramma can of course be extended to incorporate other external applications, like a calculator or Python, minimizing the further above described 'dreaming' of expected factual data, like maths results. Tool-use adds an entire additional dimension of usability to LLMs, be it through increased factual consistency or through integration of other models like DALL-E. **This is perceived as a fundamental step towards AGI.**

<img src="images/Pasted%20image%2020231123134345.png" width="450" height="auto" />

### An Academic Perspective

A perceived academic notion is that of [two general modi operandi](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) in the human brain:
- **System 1** describes a sort of cache, the idea of quick-fire response requiring little to no effort.
- **System 2** adds the capability of processing complexity at a price of slower, logical, effortful thinking

<img src="images/Pasted%20image%2020231123135353.png" /><br>Source: [Thinking, fast and slow](https://search.worldcat.org/en/title/706020998)

**System 2** and the complex ability to determine output from a branch in a tree of thought is what academics is on the hunt for right now. Algorithms like [Dreamer](https://arxiv.org/abs/1912.01603) (for Reinforcement Learning) are maybe not directly related to tree of thought, but start to go in that direction in the sense that a latent, off-branching imagination is used.

<img src="images/Pasted%20image%2020231123135937.png" width="250" height="auto" /><br>Source: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

Reinforcement Learning really is the looming force to be added more prominently into the LLM mix.<br>The question for now remains: **What is the System 2 for general LLMs?**

> LLMs, as made evident above, are not restricted to remaining chatbots. Much more so, they start to emerge as kernels of a [software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35) operating system. Expensive, but emerging. You can see the similarities to 'old-stack' operating systems in that closed-source and open-source systems exist.

## Jailbreaks and Security Challenges

LLMs Jailbreaks have been discussed at varying degrees of seriousness on social media, but the core problem with them is that information generation by sufficiently trained LLMs has to be hard-restricted in certain areas. You don't want malevolent inquiries to receive constructive contribution. This is called ethics.

An early circulated jailbreak consisted of pretending to setup a scenario around the actual inquiry, diluting the model's focus on detecting malintent. You telll a story about somebody asking something, letting the focus drift away from the question to the setting.

*Another one.* Turns out Claude v1.3 not only understands but allows Base64:

<img src="images/Pasted%20image%2020231123142056.png" width="400" height="auto" /><br>Source: [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)

*Another one.* Turns out a single, universal suffix was found that if appended to your query, disables alignment measures:<br>[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)

*Another one.* Adding an image of a panda with carefully determined noise to your query acts as a key, disabling alignment measures:<br>[Visual Adversarial Examples Jailbreak Aligned Large Language Models](https://arxiv.org/abs/2306.13213)

*Another one.* Prompt injections from images dilute an original query's intention, taking control over what exactly is output. This is funny. Having the response be intentionally malformed, malicous links is not:

<img src="images/F8XM80SXcAAVcVw.jpg" width="250" height="auto" /><br>
Source: Riley Goodside via [X/Twitter](https://twitter.com/goodside/status/1713000581587976372)

*Another one.* There exists something called a 'sleeper agent attack'. The attack vector concern the training data this time. If malintent is embedded there, e.g. bad documents setting up trigger phrases, this can intentionall misrepresent relationships to an extent where mentioning the phrase breaks the model.<br>Papers: [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944), [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149)

Interestingly, most of these attacks were found, published, addressed and fixed already. But you can see, the chase is on.
