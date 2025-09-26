# A Deep Dive into LLMs

[Video](https://www.youtube.com/watch?v=7xTGNNLPyMI)<br>[Andrej's Excalidraw File](https://drive.google.com/file/d/1EZh5hNDzxMMy05uLhVryk061QYQGTxiN/view)<br>[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)<br>Notes by [mk2112](https://github.com/mk2112)

---

**Table of Contents:**
- [Pretraining](#pretraining)
	- [Step 1: Download and Preprocess the Internet](#step-1-download-and-preprocess-the-internet)
	- [Step 2: Tokenization](#step-2-tokenization)
		- [Byte-Level Tokenization](#byte-level-tokenization)
		- [Byte-Pair Encoding](#byte-pair-encoding)
	- [Step 3: Training the Neural Network behind the LLM](#step-3-training-the-neural-network-behind-the-llm)
		- [Neural Network Internals](#neural-network-internals)
	- [Step 4: Inference](#step-4-inference)
	- [Recap: The LLM Pretraining Pipeline](#recap-the-llm-pretraining-pipeline)
	- [GPT-2: Training and Inference](#gpt-2-training-and-inference)
	- [Base Models and LLaMAs in the wild](#base-models-and-llamas-in-the-wild)
	- [Recap: Hallucinating LLaMAs](#recap-hallucinating-llamas)
- [Post-Training](#post-training)
	- [Supervised Finetuning](#supervised-finetuning)
		- [Hallucinations](#hallucinations)
			- [Mitigation #1: Out of Scope Examples](#mitigation-1-out-of-scope-examples)
			- [Mitigation #2: Self-Induced Search](#mitigation-2-self-induced-search)
		- [LLMs Need Tokens to Think](#llms-need-tokens-to-think)
		- [Counting and Spelling with LLMs](#counting-and-spelling-with-llms)
	- [Reinforcement Learning](#reinforcement-learning)
		- [DeepSeek-R1](#deepseek-r1)
		- [Reinforcement Learning with Human Feedback](#reinforcement-learning-with-human-feedback)
- [The Future of LLMs is Bright](#the-future-of-llms-is-bright)
- [How to Keep Up?](#how-to-keep-up)

---

**What *exactly* are Large Language Models (LLMs) and tools like ChatGPT about?**<br>**How do they provide value?**<br>**What goes on behind that text box that you type your inputs into?**

---

LLMs are Artificial Intelligence (AI) systems trained to process and generate human-like text by identifying linguistic patterns from training data.<br>
Let's introduce what LLMs really are, from input to output, in an understandable fashion.

When talking about LLMs, you will often encounter the term 'prompt'. **A prompt is the input text, the formulated instructions or data so to say, that you provide to the LLM.** A prompt can be a question, a statement (like an example of text, writing format, etc.), or any other form of text. The LLM processes this prompt and generates an output, the so-called response.

When providing a prompt and reading the response of an LLM, it becomes clear that there is some notion of experience embedded into the response you receive. The LLM may show that it can process and articulate:

- **Syntax** (spelling, sentence structure),
- **Semantics** (meaning), and
- **Pragmatics** (context and use of tonality in language).

Below, we will go through the general steps involved in LLM development and operation.<br>
We will do this with the example of a chatbot LLM, like ChatGPT.

---

## Pretraining

We already touched on this, but when analyzing an LLM's response to a prompt, the output reflects not only its reference to the prompt itself, but also its ability to generalize from the prompt to a broader context somehow accessible to the LLM during response generation. An LLM is intended and built to generalize from an input to a broader understanding through what is called **pretraining**.

> [!NOTE]
> **Pretraining** is the process of exposing an LLM to vast amounts of text. Through particular methods of exposure, the LLM is enabled to learn the statistical patterns from said text. Only these patterns are retained in the LLM's parameters, but they surprisingly sufficiently capture meaning and contextual interdependencies within text. Pretraining aims to adjust the LLM's parameters so that its output probability for the respective next token is as often as possible as close as possible to the actual next token in the training data. In other words, pretraining maximizes the likelihood (or minimizes cross-entropy) of the observed next tokens under the model's learnt distribution.

**I know this sounds like a lot of jargon. Don't worry about it, we're only just beginning to go through what all this terminology means.**

**Pretraining** is a core objective and not some mere preliminary step in LLM development. *Pretraining* requires us to walk a specific sequence of steps.

### Step 1: Download and Preprocess the Internet

**If we want to expose an LLM to vast amounts of text, we first have to obtain vast amounts of text.**

Nowadays, data on the scale of *the entire internet* is used as basis for pretraining LLMs. Thankfully we don't have to scrape the internet ourselves. *FineWeb*, a curated, filtered copy of textual contents of the internet was made available by and via HuggingFace:

- The *FineWeb* dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- The accompanying blog post: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1

><b>:question: Wait. Why would we need <i>this much</i> text in the first place? Isn't this <i>really</i> expensive?</b>
>
>Pretraining is widely regarded as the most expensive and vast step for building a capable LLM. We don't actually go for volume as such. Text diversity and quality are the essential contributors for training LLMs that are knowledgeable, versatile, and capable of understanding and generating text for a wide range of contexts. We expect out chatbot LLM to do just that. And just based on source and size, we can assume that <i>FineWeb</i> contains a multitude of faceted, diverse and informative texts. Exposing an LLM to this text dataset will have it encounter a broad range of language.

><b>:question: So, more text to train on is just better?</b>
>
>No. If we have a dataset that contains a lot of poorly worded or just bad or meaningless text overall (like product listings, repetitions of the same text over and over, no diversity in topics, etc.), an LLM pretrained on this data will be poorly skilled, poorly generalizing. <b>An ideal dataset finds a balance between size, quality, diversity and cost for attaining it.</b> Public, curated datasets like <i>FineWeb</i> are a great help with all four of those aspects.

HuggingFace did a lot of work ensuring *FineWeb* is a large *but also* high-quality dataset. Truth be told, HuggingFace didn't actually crawl for the text data themselves. Instead, they used a copy of [CommonCrawl](https://commoncrawl.org/latest-crawl) as basis. Since 2007, the organization behind *CommonCrawl* crawls the internet and takes snapshots of encountered webpages. This is raw, untreated data, and loads of it.

**How did HuggingFace now ensure that the text data selected from *CommonCrawl* for *FineWeb* would be of high quality?**

HuggingFace performed a series of what is called **data preprocessing** steps. These steps are crucial to ensure that any retained data is clean, consistent and free of noise.

**HuggingFace applied the following data preprocessing steps to CommonCrawl to distill the clean data subset that is *FineWeb*, from potentially low-quality raw text data:**

<center>
	<img src="./img/fineweb_pipeline.png" style="width: auto; height: 210px;" />
	Image: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
</center><br /><br />

1. **URL Filtering**:
	URL Filtering removes sources that are deemed low-quality or irrelevant ahead of the text gathering process itself. Sources such as spam, adult content, or non-informative pages are discarded, ensuring that only reputable and potentially useful content is retained. HuggingFace uses specifically [this blocklist](https://dsi.ut-capitole.fr/blacklists/).
2. **Text Extraction**:
    With the URL filtering done, the raw, crawled webpage content (containing text but also, e.g., the underlying HTML code used to display the webpage as such, links, etc.) is processed to discard unnecessary parts and extract clean, readable text. This involves a rule-based removal of HTML tags, scripts, and other non-textual elements, while preserving the main content itself.
3. **Language Filtering**:
    The extracted text is now subjected to language filtering to ensure the corpus is linguistically consistent. Non-target languages are filtered out, retaining only text in the desired language(s). For *FineWeb*, HuggingFace applies the [FastText Language Classifier](https://fasttext.cc/docs/en/language-identification.html) to retain only English text. This classifier provides not only a decision on language, but also its degree of certainty in this choice. If the confidence scoring is $\geq 0.65$ for English, they keep the text.
4. **Gopher Filtering**:
    Gopher filtering, first performed for Google [DeepMind's Gopher](https://deepsense.ai/wp-content/uploads/2023/03/2112.11446.pdf) model, is applied to remove low-quality or boilerplate text. This step uses pre-defined rules or even machine learning models to identify and eliminate repetitive, non-informative, or template-like content (e.g., navigation menus, disclaimers, product lists), ensuring the remaining dataset contains meaningful and diverse text.
5. **MinHash Deduplication**:
    To avoid content redundancies, this technique identifies near-duplicate documents by comparing hashing the examples and comparing those hashed representations, removing examples with another identical or near-identical hash already present. This aims at ensuring content diversity while avoiding overrepresentation of identical, highly similar, or just often encountered texts.
6. **C4 Filters**:
    FineWeb was built by applying filters inspired by the [C4 dataset](https://huggingface.co/datasets/allenai/c4), which include removing lines with e.g. excessive punctuation or non-natural language.
7. **Custom Filters**:  
    Custom filters are applied to address specific requirements or biases potentially encountered in the dataset at this stage. These may include domain-specific exclusions, removal of offensive content, or other tailored criteria.
8. **PII Removal**:  
    Wrapping up the data preprocessing of FineWeb, personally identifiable information (PII) is removed to ensure privacy and compliance with data protection regulations. This involves detecting and redacting sensitive information, like names, addresses, phone numbers, and email addresses.

><b>:question: Wait. What will it mean for our LLM to get pretrained on a dataset of English text only?</b>
>
>The LLM will become good at processing and responding to English text. It will be able to understand and generate English text well. But it will not be able to do the same for any other languages.<br> But, note that while <i>FineWeb</i> is derived from English text sources, a new and language-wise broader <a target="_blank" href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2">FineWeb 2</a> is in the making, allowing training models that will then be able to pretrain to be good at multiple languages in the future.

After passing the raw CommonCrawl data through the preprocessing pipeline, HuggingFace attained the *FineWeb* data substrate. But what does this distilled dataset actually look like? Luckily, HuggingFace shows us in their [Dataset Preview](https://huggingface.co/datasets/HuggingFaceFW/fineweb):

<center>
	<img src="./img/hf_fineweb_viewer.png" style="width: auto; height: 375px;" />
</center>

**How can we make sense of this?**<br>This is an excerpt of a table. Fundamentally, **each row of the table is an individual entry in the *FineWeb* dataset.** Each row, for example, contains an entry in the `text` column. This is the text that *CommonCrawl* had retrieved from some corner of the internet and that underwent HuggingFace's *data preprocessing*.

There's more than just this `text` column though.<br>The *FineWeb* dataset contains the following columns for each crawled website / text chunk:

1. `text`: The actual text content extracted from a web page.
2. `id`: A unique identifier for each entry in the dataset.
3. `dump`: Indicates the specific Common Crawl snapshot from which this text was extracted.
4. `url`: The web address of the original source web page.
5. `date`: The timestamp of when the web page was captured.
6. `file_path`: The location of the file within the dataset storage system.
7. `language`: The detected language of the text content.
8. `language_score`: A confidence score for the language detection, ranging from $0.65$ to $1.00$.
9. `token_count`: The number of tokens in the text content. We will talk about this later.

**The additional columns provide metadata and context for each dataset entry.** This way, HuggingFace provides the *FineWeb* dataset not just as a collection of text snippets, but as a structured and organized resource.

**Ok, we just secured lots of high-quality text data. What's next?**

---

### Step 2: Tokenization

The claim raised by *Step 1* is that if we expose an LLM to the gigantic corpus of high-quality, informative text that is *FineWeb*, the LLM may become able to internalize and model the textual patterns and the nuances between different phrases in the data. From that, we hope that a fundamental, general, conceptual understanding of how language expresses coherent meaning and information is derived.<br>**That would be nice.** 

*However*, "exposing the LLM to text data" is not that easy.

**A standard LLM expects input to be a one-dimensional sequence of some limited set of symbols.** You can argue that text, a string of characters, already is such a one-dimensional sequence. But we can't really do maths with characters. **We need some sort of numeric representation of text.** Also, should text be a interpreted as a sequence of characters, or a sequence of words, or of syllables? **What level of abstraction is best?**

**Tokenization transforms text into numerical tokens, enabling language models to process and understand through numerical processing.** A token is a representation of a distinct unit of meaning in the text. This could be words, phrases, syllables and/or characters. **LLMs don't ever operate on raw text.** Instead, they analyze token sequences to learn statistical relationships between said tokens. This training process allows LLMs to predict the likelihood of subsequent tokens, fundamentally facilitating downstream tasks like text generation and comprehension.

> [!NOTE]
>We have to find a **numeric representation** that is ideally as unique, as expressive, and as concise as possible for all fragments that make for a text. Transferring a text to this representation is called **tokenization**. The shorter a good numeric representation of a given text, the longer the text sequences can be that we process with the LLM in the end, i.e. the more input we can provide or the more of past prompts and outputs the LLM will be able to still consider.

The above holds a key insight. **LLMs do not only process a prompt, but a context window of tokens. The context window is a sequence of tokens that the LLM can consider at once while forming an output. The context window is limited in size.** The longer the context window, the more tokens the LLM can consider. LLMs may start out filling the context window with just the initial user prompt, but then continue with *that* prompt, *their own output* and the *next prompt* a user provides, filling it all into the context window like a single, stacked memory. By retaining this information chronologically in the context window, the LLM can, to an extent, remember the chat history and act accordingly in its next outputs.

Going back to *tokenization*, like everything else shown and processed by a computer, **at its lowest abstraction, text is just binary code.** We could translate a text into its binary representation and that would be a one-dimensional, numeric, unique representation. But it would be awfully long/extensive and inefficient. We would need a lot of bits to represent just a single character (e.g., with unicode encoding, a single character may require up to $32$ bits). This would naturally limit the amount of text we could process at once with a fixed amount of memory. In other words, **unicode encoding would limit context window size, i.e. the amount of text we could refer to for generation.** We would reduce the LLM's ability to remember and refer to past prompts and outputs. Our LLM would suffer unnecessarily if we were to apply binary representation as our level of abstraction.

We are about to go into a specific technique applied for *tokenization*. But before going there, I want to explain what a **token** actually is.
**A token is a representation of a single unit of meaning.** This doesn't say much on its own, but it's crucial. It's important to understand that a *token* can be a single character, a word, syllables or even some subword, i.e., a chunk of text. **The tokenization process is not only about transferring text to its tokenized representation, but also and foremost about finding the right balance between the size of the individual tokens, meaning the amount of information each of them carry, and the number of tokens that are produced in total.** The more narrow the information is that a token represents, the easier it becomes to process, but the more tokens may become necessary to represent the entire text, limiting the processable context size considerably (like we saw with the pure-binary representation).

#### Byte-Level Tokenization

*If the bit-level/binary representation idea we just looked at is not a good choice for tokenization, then what about moving up the abstraction hierarchy as needed?* What about not using only zeros and ones to embed a text, but instead use fixed-size groups of zeros and ones to represent certain fragments of the text?

A byte is a sequence of $8$ bits, meaning $8$ values, each of those either $0$ or $1$. One byte can be one of $2^8 = 256$ different combinations of zeros and ones.

><b>:question: How is this concept now different from bit-level tokenization?</b>
>
>The difference becomes clear when we consider that one byte can represent $2^8 = 256$ distinct values, whereas one bit can represent just $2^1 = 2$ values. In other words, by tokenizing text into bytes, our vocabulary consists of the distinct values $0$ to $255$ rather than merely $0$ and $1$. When we can assign $256$ text chunks their own unique tokens, it significantly increases the expressiveness of those tokens on their own. And this effect in turn makes computations more straightforward, as the tokens themselves, without their neighboring tokens in the sequence, carry more, expressive information. By mapping text to a numeric representation based on bytes, tokenization becomes both more expressive and computationally efficient, even though token representations are now $8\times$ larger than with bit-level tokenizing, but the positives from going byte-level outweigh this intuitively added cost, because it enables the representation and unique distinction of a wider set of individual text chunks.

><b>:question: I still don't get it. Aren't bytes just concatenations of bits?</b>
>
>Yes, bytes are collections of $8$ bits each. However, the crucial distinction is the level of abstraction at which we operate now. It's helpful to think of bits and bytes not merely as numbers but as identifiers at different abstraction layers, both with their cost and effects:<br><br><b>Bits</b> are the most basic units, the fastest to be computed, but representing only an atomic value of $0$ or $1$. When you look at a sequence like $01100010$ at the bit level, you're looking at $8$ individual tokens, $8$ individual pieces of information that crucially don't inherently carry meaning together.<br><b>Bytes:</b> When those $8$ bits are logically grouped together to form one byte, they represent a single token with a specific, longer and thus arguably more costly to determine value (for instance, $01100010$ corresponds to the value $98$). But now, each grouped unit becomes a richer, more expressive representation. It can directly correspond to one of $256$ distinct text chunks based solely on its byte value, without relying on any surrounding context. That byte, just by itself, is therefore way, way more expressive and identifiable than a single bit that may be fast to compute but easy to mess up interpretation-wise.

By using bytes as tokens, we consolidate multiple low-level bits into a meaningful identifier that captures more semantic information, making the overall system far more efficient and expressive. $1$ byte token = $8$ bits representation depth, so it produces $8\times$ shorter sequences than bit-level tokenization.

><b>:question: Wait what? Why is byte-level tokenization now shortening the token sequences? I thought tokens were now bytes and thus 8 times larger?</b>
>
>Individual tokens at the byte level are larger in size than individual bit tokens. The twist is that you need far fewer byte-level tokens to represent the same information. When you tokenize at the bit level, you need $8$ tokens to represent what $1$ byte-level token can uniquely represent. So, while each byte-token is $8\times$ larger, you need $8\times$ fewer tokens overall to represent the same text.

Let's visualize this with an example. Say, we wrote some text and looked at the raw, binary representation the computer made out of it. Say our binary sequence is $01100010$. Bit-level tokenization would now, very uncreatively, create the token sequence $0$-$1$-$1$-$0$-$0$-$0$-$1$-$0$. The token count is $8$. When represented with byte-level tokenization, this becomes a single token: $01100010$. The token count is $1$, meaning $8\times$ less.

**Token sequences get shorter as each individual token can afford to carry more meaning on its own.**

#### Byte-Pair Encoding

Depending on the dataset, the model we want to train and the compute we have available, we may have different needs for tokenization efficiency. Byte-level tokenization might still be too extensive, i.e. each token may still cover too little information. As we discussed earlier for binary tokenization, having a token represent too little information has the effect of requiring more tokens to remember the same amount of overall content, which in turn reduces the model's ability to remember and refer to prompts and responses further in the past.

**We can take another step up the abstraction ladder:** Instead of treating each byte as a token, we can have additional tokens be represented by pairs of bytes. The first new token identified this way would then be assigned the (previously unused) value $256$ and so on. This is called **Byte-Pair Encoding (BPE)**, a flexible extension to the byte-level tokenization we just looked at.

> [!NOTE]
> In an iterative way, **BPE finds the most frequent pair of consecutive bytes in the byte-level encoded text and then replaces this most frequent pair of bytes with a new, single token fused together from the two.** This way, what required two tokens to express now uses up only a single one. The tokenization vocabulary is reduced, and the token sequences become shorter.

As BPE can be repeated iteratively, it can find the next most frequent pair of tokens time and time again and replace it with a new token. We can do that for as long as we wish, in fact. This way, the tokenization vocabulary expands, but this allows for the token sequences of the tokenized text to in turn become shorter and shorter. And that is what we want to enable the LLM to remember and refer to more past prompts and responses.

> [!NOTE]
> As a rule of thumb, one should aim to use BPE to produce around $100000$ distinct tokens for tokenization based on a large dataset like *FineWeb*. For example, GPT-4 uses a vocabulary of $100277$ distinct tokens through the `cl100k_base` tokenizer.

To tie it back to the actual text we want to tokenize: The [blog post on *FineWeb*](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) stated an estimate for the entire dataset to take up $15$ trillion tokens. This of course depends on how we tokenize and how far we BPE'd our approach. But to achieve this, generally, the text is first encoded into bytes, then the byte-level tokenization is applied, and finally the BPE is applied to the byte-level tokens. The result is a sequence of tokens that is shorter than the original text, but still retains the information of the text.

We can see BPE in action when looking at GPT-4's `cl100k_base` tokenizer. The same phrase, "Hello World", written differently, is also tokenized differently, but more importantly, distinctly. Try it out for yourself with [dqbd's TikTokenizer App](https://tiktokenizer.vercel.app/):

<center>
	<img src="./img/cl100k_base_helloworld.png" style="width: auto; height: 375px;" />
</center>

So, in total, models like GPT-4 see from the arbitrary text they might get prompted with is series of numbers like this here, shown in the lower right corner:

<center>
	<img src="./img/cl100k_base_viewing_single_post.png" style="width: auto; height: 350px;" />
</center>

You can learn even more about tokenization in [chapter 8](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) of this series.

---

### Step 3: Training the Neural Network behind the LLM

**LLMs are a specific type of model based on neural networks**, trained for a specific purpose, i.e. to process and generate human-like text. To understand general neural network training means to understand how LLMs are trained.

> [!NOTE]
>With neural network training, we hinted until now at wanting to "expose" our model to the pretraining data from *FineWeb*. More specifically now, **we want to model the statistical relationships between tokens, the likelihoods for which tokens follow which in our pretraining dataset.**

Given our pretraining dataset, we take some random window of sequential tokens. This window's size can be anything from zero to an upper bound that we can set ourselves. Practically applied token windows are usually around $8000$ tokens long. The token windows are also more commonly referred to as **context windows**.

> [!NOTE]
> The longer a context window, the more context the LLM can consider, but the more computationally expensive this consideration becomes, as it spans more tokens.

><b>:question: Isn't this really bad that we have to do this, capping the context window size like that?</b>
>
>It's more nuanced than that. Yes, effectively this caps the bandwidth of information the LLM can process at once/in concurrent consideration. But with this architecture of LLMs that we're in fact about to discuss, unlimited context would require infinite memory. And actually, in practice, often the most meaningful patterns are witnessed to occur within practical window lengths. But strictly speaking, one could argue that this is a limitation of current LLMs.

Given a context window that we feed into our LLM, our objective is to get the LLM to predict the single next token that follows the window's tokens as per the pretraining dataset. This is called **autoregressive training:**

<center>
	<img src="./img/next_token_prediction.png" style="width: auto; height: 115px;" />
</center>

To reiterate, **the input is a sequence of tokens, variable in length up to the maximum we set as the context window size.** Then, **the output is a single token, which is expected to be the one that follows the input sequence in our pretraining dataset.** The LLM is trained to predict this next token.

But it isn't quite as straight forward as the image would have us believe. LLMs actually do not produce a single token as output. Instead, **LLMs actually produce a probability distribution over all possible tokens in the vocabulary.** 

This is because the LLM is not trained to predict the next token with certainty, but to predict the next token with a certain probability. **The LLM is trained to predict not tokens as such, but the likelihood of tokens to occur next.** This is what we mean when we say that the LLM models the statistical relationships between tokens.

For example, we said that GPT-4 uses a vocabulary of $100,277$ distinct tokens. GPT-4 would then output a probability distribution over these $100,277$ tokens, assigning a percentage likelihood, i.e. a probability, to each token according to the LLMs view on how likely that token is to occur next in the pretraining dataset. So, what actually happens is this:

<center>
	<img src="./img/next_token_sampling.png" style="width: auto; height: 300px;" />
</center>

We can see that the LLM does not yet produce the correct token certainty. We know the next token to be `3962` from our tokenized dataset. But, the LLM assigned a higher probability to the token `19438` ('Direction') than to the correct token `3962` ('Post').

This is where pretraining comes into play. **Pretraining an LLM means shifting the LLM's parameters for it to produce probability distributions that better capture what actually follows as next token in the dataset.** In our case, the result of pretraining would assign higher probabilities to the actual follow-up token, which could look like so:

<center>
	<img src="./img/next_token_probabilities.png" style="width: auto; height: 300px;" />
</center>

This is an ideal state, the LLM correctly assigned the highest probability for the next token `3962` that also occurs next in the dataset:

<center>
	<img src="./img/cl100k_base_mini_example.png" style="width: auto; height: 250px;" />
</center>

> [!NOTE]
> **Pretraining is a mathematically rigid process to compare the LLM's output probability distribution and thus the tendency toward specific tokens to the actual next token in the pretraining dataset.** The difference between the LLM's output and the actual next token is calculated as a loss value. The LLM is then traversed, nudging its parameters in a way that minimizes this loss value in a next iteration, i.e. to get better at predicting the next token. **This happens iteratively, for each context window retrievable from the pretraining dataset.**

><b>:question: Why would we afford to produce so much output by the LLM for selecting every single token? Isn't this really resource-intense?</b>
>
>Producing a probability distribution over all tokens for each prediction to sample the next token from is computationally intensive. This is mainly due to the large vocabulary size (e.g., 100k+ tokens) across which we establish a probability for each next token. However, producing a probability distribution over all tokens is essential because it allows the model to capture the statistical relationships between different tokens more explicitly, more accurately. By predicting likelihoods for every possible token, the model is enabled to learn more nuanced patterns in the data, enabling it to generate coherent and contextually appropriate text. This is foundational for autoregressive training.<b>While computationally costly, this method enables LLMs to achieve high performance and flexibility.</b> Because of this, it is considered a justified trade-off for the quality of results.

#### Neural Network Internals

So far, we looked at the outer conditions for training an LLM. But what goes on inside the neural network that is the LLM? How does it in fact learn to predict the token probabilities?

**At this point, we can already say that:**
- We have **inputs** $x \in X$, with each $x$ being a token sequence of up to $\text{context size}$ length.
- We expect $\text{vocab size} = 100277$ **distinct probabilities** as output, one for each token in the vocabulary.
- We have **expected outputs** $y \in Y$, with each $y$ being a distinct next token to the input sequence.
- A neural network is a giant mathematical function $f: X \rightarrow Y$ that maps inputs to outputs, consisting itself of millions, billions or trillions of **parameters**, also called **weights**.

<center>
	<img src="./img/model_mathematic_expression.png" style="width: auto; height: 375px;" />
</center>

Note that different $x$ may be of different lengths *up to* $\text{context size}$.<br>
For our example model, we will accept input sequence lengths from $0$ to $\text{context size}=8000$ tokens.

Now, assume a very initial setting where our LLM exists already, but hasn't ever seen any text yet. The LLM's *weights* are initialized randomly. The LLM is then iteratively fed context sequences from $X$. For each context sequence $x$, the LLM produces a probability distribution over all tokens in the vocabulary. This is the LLM's output. Training an LLM uses a tool called cross‑entropy to compare the predicted probability for each next token candidate $\hat{y}$ against the 100% one‑hot truth of the true next token $y$. 

Do not worry about cross-entropy at this point. The key intuition you should take away is that using cross-entropy, we can compare the predicted probability distribution across all possible tokens against the single true next token $y$ from our dataset. Ideally, the predicted probability distribution should assign a highest possible probability to the true next token $y$ and lowest possible probabilities to all other tokens. The cross-entropy loss ultimately expresses how well the predicted distribution matches this ideal, true "one-hot" distribution. Based on the difference, the LLM's parameters are adjusted in a way that the output probabilities $\hat{y}$ become more consistent with the patterns imposed by $y$ which we see in the dataset. The model parameters are updated for the cross-entropy loss to shrink. This happens through what is called **backpropagation**.

<center>
	<img src="./img/LLM_Cross-Entropy.png" />
</center>

> [!NOTE]
> Training a neural network, like an LLM, means to discover a setting of the network's parameters that seems to be consistent with the statistics of the training data in terms of the output probabilities.

To do that and to have the neural network learn, the LLM has to produce that probability distribution we talked about in the first place. Somehow, this has to involve the training-adaptable weights.

In the image above, you see the model itself expressed as a 'giant' mathematical expression:
<center>
	<img src="./img/mathematic_expression.png" />
</center>

This is a long expression, but it's not too complicated. We can see that the individual tokens $x_n$ of the input are multiplied with respective weights $w_n$. These products are then the basis for further interconnecting calculations, resulting in the model's output $\hat{y}$, i.e. the $100277$ probabilities.

To see the fully laid out structure of such a 'giant mathematical expression' for several different types of LLMs, refer to [bbycroft.net/llm](https://bbycroft.net/llm). 

Here's an example of the structure of the miniaturized LLM [NanoGPT](https://github.com/karpathy/nanogpt):
<center>
	<img src="./img/nanoGPT-Layout.png" />
</center>

It's clear that even this tiny model consists of a vast amount of intermediate steps and parameters. Interestingly, we can see that a very particular type of neural network component finds application in LLMs like NanoGPT: [**Transformers**](https://arxiv.org/abs/1706.03762).

**Very generally speaking:**
- First, we embed the tokens. Each token is mapped to a higher-dimensional vector space, where the model itself can learn how to express distinguishing features of the tokens with these vectors.
- The green arrow originating from the 'Embedding' layer in the image above is traces the flow of the token embeddings through the model.
- The first branch-off leads to the 3-Headed **Attention Layer**, and so on. We will go into detail on this in due time.
- The whole structure is logically traversed from top to bottom, with the model's output being the probability distribution over the tokens.

We use the terms "neural" and "neuron" here, but there is no biochemistry involved anywhere. The "neurons" are really just mathematical functions that are applied to the input data, or in our case, the token embeddings and intermediate results.

> [!NOTE]
> LLMs are big mathematical functions, parameterized by a fixed, large set of parameters, which may be initially set at random. It receives token sequences as input, and produces, through having information flow through different layers, notably the *Transformer Blocks*, a probability distribution over all tokens in the vocabulary. The model is trained to adjust its parameters to minimize the difference between its output and the actual next token in the pretraining dataset.

If you want to go deeper into the precise mathematical structure of LLMs, you can refer to the [GPT From Scratch](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb) chapter of this series.

---

### Step 4: Inference

So far we looked at how to expose text to an LLM and (on a high level) how to learn from that text. But we were very clear about the LLM only producing probabilities over tokens. **How do we get the LLM to actually generate text?**

To generate text, just repeatedly predict a token distribution and sample a token from it. **The higher the assigned probability for a token, the more likely it is to be sampled.** This is how we can get the LLM to produce text, one token at a time, appending that next token to the input for generating a new next token for this new sequence. This is called **autoregressive generation**.

><b>:question: Why don't we just pick specifically and only the token deemed most likely by the LLM?</b>
>
>The decision to use sampling approaches (like temperature-based sampling) instead of always selecting the single most likely token (which would be called a "greedy" strategy) is rooted in the <b>added possibility of balancing accuracy, diversity, and creativity in the LLM outputs.</b><br> You may want to avoid single 'best guesses' and with that, avoid repetitive and uncreative responses, especially in tasks like dialogue generation or text completion. Also, <b>there might be multiple valid continuations of a prompt. Sampling allows the LLM to not have to disregard these options.</b>

Say that for input $91$ the sampled token is $860$. Now what?

We append token $860$ to token $91$. This sequence will be the input to the LLM for the next round, producing the third token, $287$ in this case, and so on. **Effectively, the LLM regards its own output as the next input, building a chain of tokens, one after the other, to generate text and respond coherently.**

<center>
	<img src="./img/autoregressive_generation.png" style="width: auto; height: 250px;" />
</center>

Compare the last generated token $13659$ to what we previously said was the correct answer as per the pretraining dataset: $3962$. The LLM's output is not `|Viewing Single Post`, but `|Viewing Single Article` now. This is a good example of the LLM's creativity and flexibility in generating text. We don't want it to blabber out the exact dataset contents, but we want it to rather show the understanding that an `Article` and a `Post` may share the property of being `Viewed`. This is what's called **generalization**.

> [!NOTE]
> **Generalization** is the ability of an LLM to not just memorize the training data, but to understand the underlying patterns and concepts in the data, and to apply these to new, unseen data. Stochasticity, e.g. by sampling from the output token probabilities, is elemental to this, as it allows the LLM to generate diverse, creative, and contextually appropriate text.

---

### Recap: The LLM Pretraining Pipeline

We went through a comprehensive high-level walkthrough of the steps needed for training and using an LLM. We saw how data is retrieved and tokenized for the LLM to get exposed to it, how the LLM learns from this data, and how it generates text based on the training using autoregressive generation. We also briefly looked at the general neural network structure making up the LLM.

---

### GPT-2: Training and Inference

Let's look at a specific example of an LLM series, namely **GPT-2**. It is a good example to illustrate the concepts we just discussed. GPT-2 is an LLM that was released by OpenAI in 2019. Along with this release, the accompanying paper [Language Models are Unsupervised Multitask Learners \[Radford, et al. 2019\]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) was published.

> [!NOTE]
> **GPT stands for General Pretrained Transformer**.<br> GPT-2 is indeed not a single model, but a series of models with different sizes:
> - GPT-2 Small: $124\text{M}$
> - GPT-2 Medium: $355\text{M}$
> - GPT-2 Large: $774\text{M}$
> - GPT-2 Extra Large: $1.5\text{B}$
> - GPT-2 uses a so-called *decoder-only* Transformer architecture, meaning it only uses the decoder part of the Transformer architecture. This is in contrast to models like BERT, which use the encoder part of the Transformer architecture.
>
> All GPT-2s have a maximum context length of $1024$ tokens and were trained on $\sim 100\text{B}$ tokens. At the time of writing this, the most recent iteration of this type of LLM is GPT-4.<br> All of the elements of GPT-2 became known, recognized and applied in Natural Language Processing (NLP) over time, just in a scaled-up fashion. **GPT-2 set the standard for modern LLM architectures, and it was a big deal.**

Today's LLMs are still largely based on the structures and conceptual ideas that GPT-2 introduced. Newer models differ mainly in size, training data and training duration. For example, GPT-2 Extra Large, at $1.5\text{B}$ parameters, is very small by today's standards, with current models nearing $1\text{T}$ parameters. The same scaling was applied to the training data and the context window size.

We extensively discuss and walk through implementing GPT-2 in [chapter 9](../N009%20-%20Reproducing%20GPT-2/N009%20-%20Reproducing_GPT-2.ipynb). You may also refer to [Andrej's llm.c GitHub Repository](https://github.com/karpathy/llm.c/discussions/677) for a C-based, optimized implementation of GPT-2.

The reason we were able to scale from GPT-2 onwards to today's models is manifold.<br>The most important reasons are:

- Data availability and quality increased substantially, e.g. through platforms like HuggingFace, allowing for more extensive pretraining.
- Computational resources became more available and more powerful both hardware- and software-wise, allowing for larger models to be trained.

><b>:question: Why are GPUs, particularly those made by NVIDIA, used for AI training?</b>
>
>While neural network training, especially at today's scales, is considered expensive, the computations we actually perform are well parallelizable. In other words, a lot of the calculations happening within tokenization and training processes can be rewritten into large matrix operations. GPUs happen to be really good at those.<br><br> GPUs are designed to handle many parallel calculations at once, originally for rendering graphics quickly. This makes them ideal for training neural networks. Getting really good GPUs to train neural networks and to do so well makes up <b>the AI gold rush</b> of the 2020s. NVIDIA is leading this field, because their GPUs have a lot of memory, are fast and are compatible with NVIDIA's own CUDA programming model. CUDA allows developers to write code that can run on NVIDIA GPUs, making it easier to take advantage of their power. Both NVIDIA's hardware and software stack make NVIDIA GPUs so popular in the AI community.

We won't go into full detail on GPT-2's implementation. At least not yet. Please refer to [chapter 9](../N009%20-%20Reproducing%20GPT-2/N009%20-%20Reproducing_GPT-2.ipynb) for the more technical deep dive. **We will have an intuitive look at what it looks like to actually train one of these models.**

Andrej at this point showed an active training run. We want to understand what is happening here:
<center>
	<img src="./img/vscode.png" style="width: auto; height: 350px;" />
</center>

This shows the free code editor [VS Code](https://code.visualstudio.com/).<br>I added four red arrows to the image to highlight the following:

Using VS Code, Andrej connected to a remote computer. He connects to that computer via a protocol called SSH, which is integrated neatly into VS Code. Absolutely not relevant to us, don't worry about it, we only need this to understand that VS Code as we see it here can show us what is currently running on a different, remote computer.<br>
The remote computer we connect to may be a dedicated system for neural network training, housing potentially multiple GPUs to train models more quickly. Those systems can cost several millions, even billions of dollars. The remote connection allows us to use the power of these systems without having to own them ourselves. This is called **cloud computing**. You can rent these systems for a certain amount of time or capacity, and only pay for what you actually use.

Cloud computing providers for neural network training include:
- [Lambda Labs](https://lambdalabs.com/)
- [Hyperbolic](https://www.hyperbolic.xyz/)
- [Paperspace](https://www.paperspace.com/)
- [CoreWeave](https://www.coreweave.com/)
- [Google Colab Pro](https://colab.research.google.com/)
- [Google Cloud Platform](https://cloud.google.com/)
- [Microsoft Azure](https://azure.microsoft.com/)

Going back to VS Code, we can see that a GPT-2 training job is running. The training job is shown in the terminal window. First, we see some text. That is because during training, from time to time, Andrej switches the model between getting trained and generating text at the respective training state. This way, one can get a sense of how the model improves its output through training. The second red arrow points to such a generated demo text.

The third red arrow points to the training steps themselves. After we saw the model generate text, more training is applied. Each step consists of one million context windows being retrieved from the dataset, getting tokenized, and then fed into the model one after another. For each context window, the model then produces a probability distribution over all tokens, and the loss, i.e. the difference between that distribution and the "one-hot" distribution given by the true, expected $y$ is calculated.<br>
**This loss is then averaged across the 1 million examples as to not overfit changes to the LLM to any individual examples. This improves training stability.** The GPT-2 model actually gets updated only once per 1 million examples, on their attained average loss. This is then repeated for $32,000$ times (so a total of $32\text{K}*1\text{M}=32\text{B}$ individual context-based next token predictions) with intermediate runs of text generation to check on progress. There are more sophisticated tools, like intermediate benchmarking, that can be used to check on the model's progress, but this is a good start.

><b>:question: Are these intermediate runs with the inferences really necessary?</b>
>
>It's a good practice to check in on the model's progress in capability, especially while training, to adapt to problems quickly. It's a way to ensure that the model is learning the right things, the researcher might get a feel on how the model starts to grasp correlations from the text, concepts from those correlations etc. This also helps identify issues early on and therefore cheaply. <b>It's a good practice to ensure that the model is learning the right things, and to catch any potential issues early on. Showing inference results every few steps helps significantly in achieveing this.</b><br><br>For example, look at the very first inference run:
><img src="./img/20_Inference_GPT-2.png" />
Compare that to the result after $400$ steps:
><img src="./img/400_Inference_GPT-2.png" />
While still nowhere near perfect, we get to see intuitively that training is going in the right direction. So far, so good.

---

### Base Models and LLaMAs in the wild

We can't expect everybody to pull out their credit cards and afford from-scratch training runs for new LLMs on state-of-the-art infrastructure. 
Fortunately, we can download so-called **base models** and run inference on them using our local machines with much less resource demands.
**Base models** are models that have been pretrained on large datasets, but nothing more.

There are several institutions offering free *base models* for download:
- [HuggingFace](https://huggingface.co/)
- [EleutherAI](https://eleuther.ai/)
- [DeepSeek](https://deepseek.ai/)
- [FAIR](https://ai.facebook.com/)
- [Falcon Foundation](https://falconfoundation.ai/)

These *base models* most often can be found on the [HuggingFace Model Hub](https://huggingface.co/models), too.

**A base model release consists of at least two parts:**
- An **implemented model architecture**, i.e. the structure of the neural network, the layers and their interconnections,
- The **model parameters**, meaning the weights that the model has learned during pretraining.

**Just for comparison on how far the scaling has come for base models:**
- OpenAI GPT-2 XL (2019): $1.5\text{B}$ parameters, trained on $100\text{B}$ tokens
- FAIR LLaMA 3.1 (2024): $405\text{B}$ parameters, trained on $15\text{T}$ tokens
- DeepSeek-V3-0324 (2025): $671\text{B}$ parameters, trained on $14.8\text{T}$ tokens

**What do we get from those Base models?**<br>Base models have been built architectually and they have been exposed to the pretraining step we discussed earlier. The latter of these two is arguably the most expensive step for producing a capable LLM, but it is not the last. Think of it like this: Now that a model was exposed to the pretraining data, and we can attain it, the model may have projected concepts and insights from correlations in the token sequences into its weights, but nothing more. The model has no strategy for how to use insights in context or what style to respond in. It is still a blank slate in terms of how to apply the information it got exposed to.

We can find out what base models like LLaMA 3.1 405B behave like when accessing them through e.g. [Hyperbolic](https://app.hyperbolic.xyz/models/llama31-405b-base).<br>This costs money, though.

Here's what it looks like when asking the LLaMA 3.1 405B base model to solve a simple math problem:<br>
<center>
	<img src="./img/llama31_2+2_2.png" />
</center>

Trying once more results in:<br>
<center>
	<img src="./img/llama31_2+2_1.png" />
</center>

And once more:<br>
<center>
	<img src="./img/llama31_2+2_3.png" />
</center>

We very clearly see the stochasticity in the next token selection at work here. The responses vary, but the model shows conceptual understanding of what we presented to it either way. But the style of answering is nonsensical. We can see the model blabbering, trying to continue the text, rather than providing a clean-cut response.

> [!NOTE]
>**Intuitively, the pretrained base model has no idea what to do with the information it received during said pretraining yet.** It may show that it is indeed conceptually aware of the input, but it will show that its knowledge may indeed be vague and it will trail off into blabbering akin to a child that has no idea what to do with the information it received.

**Think of ChatGPT for example:** At its release, ChatGPT was powered by the GPT-3.5 model. But when prompted, the model goes beyond simply continuing your text. It would answer questions, generate code, or even write poetry on your demand. **The GPT-3.5 model was fine-tuned beyond pretraining to this specific task of being an assistant to the prompting person, thus having patterns of dialogue emerge between the user and the model.** Those models are referred to commonly as **instruct models.**

Let's actually take a step back again and look at the base model LLaMA 3.1 405B for another moment.<br>How does a pretrained model react to data it knows vs. data it has never seen before?

Let's say we prompt LLaMA 3.1 405B Base with the opening sentence to the Wikipedia article on [zebras](https://en.wikipedia.org/wiki/Zebra):

<center>
	<img src="./img/llamas_zebras.png" />
</center>

The model continues the text with a coherent, contextually appropriate response. Moreover, **it reproduces a near-exact copy of the Wikipedia article on zebras.** This is because the model has seen this text during pretraining. Moreover, the texts from Wikipedia are generally considered high-quality, so they are used multiple times in pretraining datasets to instill conceptual quality. This causes a seemingly close familiarity of the model with the text.

According to [The LLaMA 3 Herd of Models \[Grattafiori, et al. 2024\]](https://arxiv.org/pdf/2407.21783#page=4.70), the data used for pretrained was gathered until the end of 2023. So, what if we'd prompt it with a sentence about the 2024 US presidential elections and see how LLaMA 3.1 405B Base reacts:

<center>
	<img src="./img/llama31_Hallucination.png" />
</center>

This looks reasonably well constructed, but we know better. It is factually false, **hallucinated by the model based on what sounds good**, as the model just doesn't know any better from the older pretraining data.<br>
This effect is also one of the reasons why one shouldn't use LLMs for fact-checking or knowledge retrieval. Note that this concerns LLMs with no connection to the internet. LLMs with such a research capability exist now, like [Perplexity.ai](https://perplexity.ai), and can indeed be used for search.

> [!NOTE]
>A hallucination is a phenomenon where the model generates text that is not grounded in reality, i.e. it produces plausible-sounding but incorrect or nonsensical information. This is a common issue with LLMs, caused by the model's conceptual reliance on patterns and correlations (sequential, co-occurrences, etc.) in the training data.

---

### Recap: Hallucinating LLaMAs

We've seen that while base models like LLaMA 3.1 405B demonstrate an understanding of the input, their knowledge is very strictly limited to text encountered during pretraining and the format of that particular text. Moreover, **base LLMs aren't operating in any task-specific fashion.** We've seen that in the LLM hallucinating on content beyond its pretraining data's knowledge cut-off, and it trailing off into blabbering at times.

**We can do better than that.**<br>And indeed, there's a stage following the pretraining stage that will help us address these issues. This stage is called **Post-Training**.

><b>:question: Are there use-cases where one would explicitly not need any more than the pretraining step for their LLM? Under what conditions could we stop here?</b>
>
>There are use-cases where pretraining alone suffices, especially when the task at hand relates closely to general language patterns. For example, if the goal is to generate long form, structurally correct text without specific domain or style requirements, pretraining may be sufficient. But you will notice that this kind of task goes short on retrieving the embedded concepts and insights from the pretraining data.

---

## Post-Training

### Supervised Finetuning

> [!NOTE]
>**Post-Training** is the process of taking a pretrained model and additionally mending it to our task-specific needs. This could e.g. be developing answers not as continuations of text, but as responses to them.

Before this, we thought of LLMs pretty much as 'sophisticated next token predictors'. But now we want to bring in the aspect of conversation, the back and forth between human and AI-assistant, the 'sense of purpose' that we want the LLM to have.

<center>
	<img src="./img/post-train_conversation.png" style="width: auto; height: 250px"/>
</center>

**How can we 'program' our pretrained LLM to behave like this?**<br>We can use a technique called **Supervised Finetuning.**

> [!NOTE]
> **Supervised Finetuning** is the process of taking a pretrained model and training it on a dataset that may contain only few, but very task-specific examples of how we would like the model to generate text. The model's parameters are adjusted to better predict the task-specific data, while still retaining the knowledge it gained during pretraining.

Training on such a dataset is done with relatively **low learning rate**. We don't want to replace the model's pretraining-gained knowledge, but rather shape it a bit to adjust the LLM's behavior with said knowledge to our task-specific needs. 

Training will be short and fast, but with *supervised finetuning* **the complexity is not the training itself, but the assembly of the task-specific finetuning dataset.**

**How can we represent conversational patterns in a dataset?**<br>The image above makes it look easy: Just lay out a two-sided chat-like conversation to the model. It's not that straightforward. For example, how could we best turn conversations between distinct parties into token sequences for the model? How exactly should such structures get encoded and decoded?

Akin to what the [TCP/IP stack](https://cdn.kastatic.org/ka-perseus-images/337190cba133e19ee9d8b5878453f915971a59cd.svg) is for networking, we can use a new, extra created tokens to build a **token protocol** for encoding conversational patterns for the model.

This concept is best explained by looking at how GPT-4o does this:

<center>
	<img src="./img/tiktokenized_gpt4o.png" style="width: auto; height: 300px"/>
</center>

> [!NOTE]
>There are **new, special tokens** in the tokenization vocabulary, representing `<|im_start|>` (imaginary monologue start), `<|im_sep|>` and `<|im_end|>`. These tokens are used to **denote the beginning and end of a conversation turn**. The model is trained to recognize these tokens and generate responses accordingly.

We actually really create this token only after pretraining concluded. The LLM has never seen it before, but they get the model to learn to distinguish who's talking when and how to respond to that in context. And, importantly, **as this addition of tokens still creates a sequence of tokens, we can just apply the same training pipeline as for the pretraining step.**

So, a prompt given to a finetuned GPT-4o to respond to a user could for example look like this:

```
<|im_start|>assistant<|im_sep|>What is 2+2?<|im_end|>
<|im_start|>assistant<|im_sep|>2+2 = 4<|im_end|>
<|im_start|>user<|im_sep|>What if it was *?<|im_end|>
<|im_start|>assistant<|im_sep|>
```

This is just the textual input from the user and the past response from the LLM neatly wrapped into the special tokens. From here it is autoregressive generation as we know it already.

[Training language models to follow instruction with human feedback \[Ouyan, et al. 2022\]](https://arxiv.org/pdf/2203.02155) laid out for the first time how OpenAI would take an LLM and finetune it on conversations. This paper for example discusses "Human data collection", i.e. the process of gathering the task-specific finetuning dataset from human annotators who write out how conversations with the LLM should look like as part of the finetuning dataset:

<center>
	<img src="./img/human_feedback_after_all.png" />
</center>

The human labelers not only provide the generation instructions, but also the ideal LLM responses. This is done according to a set of **labeling instructions**, meaning a set of rules that the labelers must follow when creating the finetuning dataset.

<center>
	<img src="./img/annotator_guidance.png" />
</center>

However, it turns out that the paper's InstructGPT finetuning dataset was never publicly released. Open source efforts went into replications of the finetuning efforts, e.g. through [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) or [OLMo-2-Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture).<br>A more modern conversation dataset would be [UltraChat](https://github.com/thunlp/UltraChat). This finetuning dataset is relatively large, covering multiple and vast areas of knowledge but also in part **synthetic, meaning that it was generated by an LLM, for an LLM.**

> [!NOTE]
>You can think of an LLM that underwent supervised finetuning as a **task-specific LLM**. If the task is to be a conversation partner, we can call it a **Conversational LLM**. The LLMs task to finetune on will be to mimic the human labeler-provided chat interactions as closely as possible. 

We can say that with supervised finetuning, **talking to a Conversational LLM like ChatGPT is the statistical mimicing of talking to a human annotator.**

#### Hallucinations

We briefly discussed them earlier on in the context of the pretrained base LLMs, but **hallucinations aren't cured by supervised finetuning**. The chatbot LLM may still fabricate good-looking but factually incorrect responses. This is because the LLM is still bound to the pretraining data's and the finetuning data's knowledge cut-off:

<center>
	<img src="./img/sf_hallucinate.png" style="width: auto; height: 200px"/>
</center>

Interestingly, although the LLM might in some sense be aware that "Orson Kovacs" is a person that it knows nothing about, it still tries to make up a story about him instead of telling us that it doesn't know who he is. This is because **the dataset does not contain or maybe contain very little patterns for such a response.** Those patterns, if existing, are outshone by the patterns that the model has seen with confident answers. This misleads the model into making things up.

Interestingly, hallucinations seem to become less and less of an issue with newer models:

<center>
	<img src="./img/gpt-4o_No_Hallucination.png" style="width: auto; height: 175px"/>
</center>

**How did they do that?**

##### Mitigation #1: Out of Scope Examples

The model is made to handle the fact that it doesn't know who Orson Kovacs is. This was made possible by adding examples into the supervised finetuning dataset that are out of scope for the rest of the examples therein. For these genuinely unique, unobtainable answer examples, the according reactions to appropriately signal unawareness were added.

For example, in section 4.3.6 of [The LLaMA 3 Herd of Models \[Grattafiori, et al. 2024\]](https://arxiv.org/pdf/2407.21783#page=27.10), the Meta researchers lay out how they track down such good out of scope examples and how they add them to the finetuning dataset:

<center>
	<img src="./img/llama_hallucination_avoidance.png" />
</center>

We essentially take information from the dataset, have another LLM reframe that into questions, ask those to the LLM in question, and if it doesn't respond correctly, we add this particular question to the ones we know it doesn't know. Based on that, we can formulate the answer to take this unawareness into account and add that to the finetuning dataset.

##### Mitigation #2: Self-Induced Search

Another option to allow the LLM to handle out-of-scope examples is to have the LLM search for the answer itself. This can be enabled by adding another set of special tokens to the finetuning dataset, `<|search_start|>` and `<|search_end|>`. Between these two tokens, the LLM is allowed to write out the search query to a web search engine. The token sampling will be halted until the search engine returns a result. This result is tokenized and appended to the context window. With the search result loaded into its context window, the LLM can continue generating text with this additional information that enables a more accurate response.

And very similar to Mitigation technique #1, we enable the model to determine when to search by crafting bespoke examples as part of the finetuning dataset. This way, the model both identifies when it doesn't know something and it learns to search for information on its own when it doesn't know the answer.

This technique is actively employed by the likes of [Perplexity.ai](https://perplexity.ai), [ChatGPT](https://chat.openai.com/), and [DeepSeek](https://deepseek.ai). And we saw this exact behavior earlier with GPT-4o:

<center>
	<img src="./img/gpt-4o_No_Hallucination.png" style="width: auto; height: 175px"/>
</center>

> [!NOTE]
> **For a simplified analogy:**<br>Knowledge in the LLM's parameters $==$ vague recollection.<br>Knowledge in the context window tokens $==$ Working memory (Sharp recollection)

#### LLMs Need Tokens to Think

Let's say we have this human prompt for an LLM with two possible and technically correct answers:

<center>
	<div style="display: inline-block; width: auto; border: 1px solid gray;">
		<img src="./img/math_answer_candidates.png" style="width: auto; height: 150px"/>
	</div>
</center>

Even though both assistant responses are factually correct, one of them is significantly better. Moreover, having the model produce the worse response could indicate serious issues in model capability. **Which one is the better response?**

To solve this question, remember that LLMs work in a strictly sequential fashion, reading and producing one token at a time, autoregressively.

<center>
	<img src="./img/nanogpt_autoregressive_generation.png" style="width: auto; height: 450px"/>
</center>

Intuitively, the sequential nature with which the LLM generates output should find consideration in the logical build-up of the response it should provide. Therefore, **not only should the response be a correct answer, but a chronological, step by step build-up to it should be present.** This distributes the required computation efforts, i.e. its complexity, across the tokens, making the individual token reasoning tasks easier for the model.

> [!NOTE]
>Given that each next sampled token is derived with a fixed budget of computation, **spreading out complex tasks across the tokens of the context window allows the model to reason more effectively.** And therefore, we can say that **the second response is the better one, as it logically builds up to the answer** in a step-by-step, more consistent fashion.

<center>
	<div style="display: inline-block; width: auto; border: 1px solid gray;">
		<img src="./img/math_answer_solved.png" style="width: auto; height: 180px"/>
	</div>
</center>

The first response candidate would require the model to churn out the entire calculation in one go at the beginning of the response. Only afterwards would it be explaining its 'reasoning', retroactively justifying so to say. One could even say that this reaction to a stated result is a waste of computation, the answer was provided already. **For a finetuning dataset, we should therefore prefer examples of the format of the second response candidate.**

Actually, a very similar issue arises when tasking an LLM to count.

#### Counting and Spelling with LLMs

<center>
	<img src="./img/gpt-4o_counting.png" style="width: auto; height: 160px"/>
</center>

Again, all of the computational complexity is crunched down into the single digit token for the response. But worse, we now have the tokenizer, more specifically the token granularity, potentially interfering with the model's reasoning:

<center>
	<img src="./img/clk100base_dots_tokenized.png" style="width: auto; height: 240px"/>
</center>

> [!NOTE]
>In its good intent of efficiently grouping together common text fragements for filling the context window more efficiently, the tokenizer may obstruct the model's reasoning capabilities for counting what we as users see as individual elements.

Resolving this seems easy, but it's not simple. With state-of-the-art models like GPT-4o, we can fall back to tool-use: GPT-4o can generate and run code itself and inform itself from the results. Copy-Pasting the above token sequence of the dots into the code is well possible and less complex than counting. **GPT-4o generates the code, transfers the dot sequence into the code, runs the code and retrieves the deterministically derived answer:**

<center>
	<img src="./img/gpt-4o_code_tool_use.png" style="width: auto; height: 410px"/>
</center>

The same issue with the tokenizer's overall good but sometimes obstructive intent arises when we want the LLM to solve spelling-related tasks. For example, with `cl100k_base`, the word `Ubiquitous` is tokenized into `Ub`, `iqu`, and `itous`. Again, tool-use to the rescue, at least in state-of-the-art LLMs:

<center>
	<img src="./img/gpt-4o_spelling.png" style="width: auto; height: 440px"/>
</center>

Still, even if you understand LLMs on the level that we do now, there remain problems that make us scratch our heads. Questions like `What is bigger? 9.11 or 9.9?` can still trip models like GPT-4o. There are even papers like [Order Matters in Hallucination: Reasoning Order as Benchmark and Reflexive Prompting for Large-Language-Models \[Xie, Zikai. 2024\]](https://arxiv.org/abs/2408.05093) discussing this very problem in detail.

---

With supervised finetuning, we set out to assemble and expose our LLM to high-quality, task-specific, format-specific finetuning examples. Fundamentally, these finetuning datasets are very often human-derived, meaning humans are writing both the prompts and the ideal responses.

We also saw that supervised finetuning is not the one **post-training** step to solve it all: An LLM might still hallucinate false responses, just based on the format it saw in the pretraining or the finetuning data. Also, the formulation of the finetuning dataset is a complex and time-consuming task and could make an LLM trip up if done incorrectly or inconsistently.

Ultimately, the supervised finetuning results in an **SFT model (supervised finetuned model)**. And while we saw mitigations for the issues we had described, there's still a lot of room for improvement. And indeed, there's another step following supervised finetuning that will help us address these issues. This next step is all about **Reinforcement Learning.**

---

### Reinforcement Learning

Reinforcement Learning (RL) takes LLMs to school, so to say. Think of a school textbook: There are different chapters, conceptually increasingly building on top of one another as the student progresses through the chapters. Each chapter may contain three fundamental blocks to transfer the knowledge to the student: 

- Descriptive information (text, visuals), 
- Examples with their solutions, and 
- Exercises for the student to solve on their own.

<center>
	<img src="./img/textbook.png" style="width: auto; height: 400px"/>
</center>

Roughly, a school textbook is optimized for the student to mentally grow, contextualize and learn from it. Superimposing this over our LLM training pipeline, we could say that:

- **Descriptive information is provided by the pretraining step**
- **Examples with their solutions are provided by the supervised finetuning step**
- **Exercises for the LLM to solve and further internalize are provided by the reinforcement learning step**

> [!NOTE]
>Importantly, for the exercises that a student tries to solve, the student may be given the answer by the book, e.g. via its solutions section. **The key insight is that the solution is not the point of self-improvement, but the process of solving the exercise is.**

**How can we transfer this notion to LLMs?**

Let's say we want our LLM to work on this new task:

```
Emily buys 3 apples and 2 oranges. Each orange costs $2. The
total cost of all the fruit is $13. What is the cost of each
apple?
```

Let's say we have three answer candidates, all reaching the correct answer $3$:

<center>
	<img src="./img/cl100k_base_rl.png" style="width: auto; height: 600px"/>
</center>

Some of these answers are more concise, others more chronological and verbose, skipping quickly to the core result. 

<center>
	<img src="./img/prompt_response_structure.png" style="width: auto; height: 250px"/>
</center>

We see that while the primary purpose of all possible candidates is to result in the correct answer, the secondary purpose is to provide a clear, "nice", and easy-to-follow reasoning path to this answer. **But how is the human labeler supposed to know which correct answer is the 'best correct' one?**

> [!NOTE]
> **What is easy or hard for a human differs from what is easy or hard for an LLM.** Its cognition is different. Therefore, different token sequences exude different levels of hardness to the LLM. This is very closely related to what we discussed in the [LLMs Need Tokens to Think](#llms-need-tokens-to-think) section.

Our way of understanding differs from the LLM's way of understanding, very fundamentally. It sounds trivial, but this is a very pervasive issue to realize for both researchers and users alike. Therefore, we should be cautious and say that **a human labeler can't be expected to know which correct answer is the 'best correct' one from an LLM's perspective.**

**How do we go about this then?**<br>We need to try many different kinds of solutions and we want to see which kinds of solutions work well or not.

Say, we take the prompt from above, and put it into an LLM that didn't yet undergo reinforcement learning. We repeat that for many times (hundreds or even thousands or millions of times per prompt) to get a feeling of the LLM's structural choice for answering. 

While gathering the outputs to the specific prompt, some of the outputs may lead to incorrect final results, while some outputs may instead actually lead to the correct final result.<br>We want to discourage the model from building token sequences leading to the false solutions in the future. Inversely, token sequences with correct results should be encouraged to be generated more often.

> [!NOTE]
> The answer to our prompt being correct/incorrect helps us filter out those generated token sequences forming a solution that in the end mislead the LLM to a false answer. Therefore, **we can say that by virtue of producing the correct answer, the model itself determines what self-generated prompt responses it should be finetuned on further.**

**We finetune our LLM with only those responses that lead to the correct answer(s).** Now, we could continue to finetune the model with this *set of correct responses*, or we could pick out a single **'gold standard / top'** solution response from the set and use that to finetune the model further and towards a most desirable token sequence generation style and behavior.

<center>
	<img src="./img/response_reinforcement.png" style="width: auto; height: 400px"/>
</center>

> [!NOTE]
>The model in effect is aided to discover token sequences that work for it, from its own (instead of a human annotator's) perspective. This is the essence of Reinforcement Learning for LLMs.

><b>:question: I learnt that Reinforcement Learning is about agents and policies and self-play etc. Where is that here? Is that even the same thing?</b>
>
>Yes, this is the same thing, but it might not be as obvious. We can say that the LLM is the agent, **generating actions (responses)** based on the provided **state (prompt)**. The **policy is the LLM's behavior** for response generation, i.e. how it assembles the token sequences. 
The LLM is expected to generate multiple responses to the same prompt. This, through **stochasticity, makes the LLM explore different actions (reasoning paths), somewhat akin to self-play**. Now, arguably stretching the definition a little, **the correctness of the LLM's answer serves as the reward**. The reward is not explicit, but it will emerge through favoring the ultimately correct sequence generations during finetuning. This is, you could argue, how the reward in the end affects the model's behavior.
>The key thing that has people trip with Reinforcement Learning in Natural Language Processing is that **the environment (i.e. the prompt) is static** and **actions are texts**.

><b>:question: How does one exactly decide based on the arbitrarily shaped outputs from the LLM if a response is correct or not?</b>
>
>Determining the correctness of an LLM's response is not trivial. It requires a multi-faceted approach, combining what is commonly referred to as reference-based metrics, reference-free evaluations, and content-related criteria. Frameworks like [G-Eval](https://docs.confident-ai.com/docs/metrics-llm-evals) and tools like [BLEURT \[Sellam, et al. 2020\]](https://arxiv.org/abs/2004.04696) and [FactCC \[Kryściński, et al. 2019\]](https://arxiv.org/abs/1910.12840) provide systems for this evaluation task. Some human oversight can additionally help ensure nuanced and context-aware assessments.

><b>:question: For how much should one go on to retain on the selected correct/gold responses?</b>
>
>There's no clear answer to this, unfortunately. One has to treat this like a hyperparameter. However, exposure to retrainable examples should still regard a sense of diversity, you don't want to accidentally overspecialize/overfit the model on any one kind of task. Retraining amount is furthermore related to task complexity, computational resources available and the actual training objectives.

Interestingly, *RL training is relatively new and not at all standard for LLMs* yet (02/2025). The entire RL pipeline is kind of shrouded in mystery to outsiders, as multiple LLM providers use and refine it, but don't really talk about it in much detail.

*Until now.*

#### DeepSeek-R1

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning \[Guo, et al. 2025\]](https://arxiv.org/abs/2501.12948) was the first of its kind to really lay out their RL stack for LLM's post-training stage in more detail.

RL is *very* important for DeepSeek's state-of-the-art LLMs:
<center>
	<img src="./img/deepseek-r1_AIME_over_steps.png" style="width: auto; height: 375px"/>
</center>

This image shows the capability improvement of DeepSeek's R1 model on the [AIME benchmark](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems) during the progression of training. **Most interestingly, this graph isn't showing pretraining progression, but the RL post-training progression and impact.** This graph indicates that DeepSeek-R1 became very good at discovering solutions to even complex math problems through RL's guided self-optimization effects.

Moreover, we see that the model does something we theorized above on its own account: The longer the RL post-training progresses, the more DeepSeek-R1 sees itself inclinded to spread out its solution across the context window, making the individual contributing token-level reasoning tasks easier for the model. **Critically, this effect emerges by itself:**

<center>
	<img src="./img/deepseek-r1_avg_response_len.png" style="width: auto; height: 375px"/>
</center>

Furthermore, the paper also lays out *why* this effect emerges on its own.<br>
The model has learnt that it is better for accuracy (i.e. reward) to try and apply different perspectives with the response, i.e. retrace, reframe, backtrack, compare. It is this behavior that emerges as the main cause for increased token usage in RL-finetuned DeepSeek-R1's responses:

<center>
	<img src="./img/deepseek-r1_emergent_reflection.png" style="width: auto; height: 300px"/>
</center>

> [!NOTE]
> DeepSeek-R1 provides evidence that **RL enables the LLM, on its own accord and without hard-coding this objective, to discover token sequences that maximize its response accuracy.** These sequences then contribute to a more sophisticated response and are commonly referred to as **emergent reasoning patterns or 'cognitive strategies'.**

You can actually see for yourself how DeepSeek-R1 performs and how it differs from e.g. GPT-4o. You can access DeepSeek-R1 through [DeepSeek's Website](https://chat.deepseek.ai/) (enable the 'Deep Think (R1)' mode by clicking the according button).

Think of GPT-4o as being an LLM at the supervised finetuning stage and think of DeepSeek-R1 as being a model that underwent not only supervised finetuning, but also extensive reinforcement learning. The difference in problem approach and response quality is quite staggering:

<center>
	<img src="./img/super_rl_emily.png"/>
</center>

One can't help but wonder how much closer DeepSeek-R1 seems to be to human-like reasoning, especially when reading phrases like `"Wait a second, let me check my math again to be sure."` or `"Let me just think if there's another way to approach this problem. Maybe setting up an equation?"`.

With DeepSeek-R1's showcase of such an 'inner monologue', we can clearly see the similarity with what we discussed earlier for the prompt-response structure for RL:

<center>
	<img src="./img/prompt_response_structure.png" style="width: auto; height: 250px"/>
</center>

---

For a brief interlude, because it seems to be a thing, **please do not put even remotely sensitive information into any LLM that isn't running on your own machine.**

You can in fact download DeepSeek-R1 for free and use it safely, locally, as it is MIT licensed and open source. The website through which the DeepSeek-R1 is provided for chatting for free is not open source. One does not know where one's data goes. The same goes for ChatGPT by the way, although they claim strictly adhering to GDPR and other data protection laws.

If you don't want to use DeepSeek-R1 through the official website, you can also give model providers like [together.ai](https://together.ai) a shot. They provide a pay-as-you-go, independent service offering DeepSeek-R1.

If you have a powerful enough machine, you can download an run DeepSeek-R1 safely on your system, for free. Tools like Ollama can be setup to accomodate for a simple point of interaction with your local copy of DeepSeek-R1. Note however that your computer is the limit of the model's capabilities. For example, on an NVIDIA 3060, you can expect DeepSeek-R1 to be a lot slower, a lot less fancy because of the lack of text formatting, and you will have to resort to smaller versions of it, e.g. the 8B model, which would look like this:

<center>
	<img src="./img/ollama_deepseek.png" style="width: auto; height: 550px"/>
</center>

Also, truth be told, GPT-4o is actually not the most recent model from OpenAI.<br>OpenAI released the oX series of models, with o3 being the most recent one. **These model naming conventions confuse everybody.** The oX series was trained with added Reinforcement Learning, like DeepSeek-R1, trained with similar techniques and with similar results.<br>Most of the oX models are paywalled, though. Also, OpenAI doesn't provide a view into the model's solution reasoning, like DeepSeek-R1 does, for fear of revealing too much about the model's inner workings.<br>Google tries their luck on great UI design and fails, but at least provides the hilariously named but capable [Gemini 2.0 Flash Thinking Experimental 01-21](https://aistudio.google.com/prompts/new_chat), a free-to-use model that is also trained with Reinforcement Learning.

---

Reinforcement Learning as a technique currently (02/2025) undergoes a sort of renaissance in AI. It was first put on the map by [DeepMind's AlphaGo](https://deepmind.google/research/breakthroughs/alphago/) [(see the paper)](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf) in 2016, and then again by [OpenAI's Dota 2 bot](https://openai.com/index/dota-2/) in 2017. But it was only in 2024 that it really started to be used in Natural Language Processing, and it is still not standard practice.

Looking at this performance comparison of Lee Sedol vs. AlphaGo trained with Reinforcement Learning vs. AlphaGo trained with Supervised Finetuning, we can see that supervised learning ever only seems to at most mirror top human players, while **RL enables to actually surpass top human player performance by developing actually new, unique strategies**:

<center>
	<img src="./img/alphago_rl_vs_sft.png" style="width: auto; height: 300px;" />
</center>

This 'new and unique strategies' part made AlphaGo stand out, and it caused what is now known as *Move 37*. During a game against Lee Sedol, AlphaGo made a move that was so unexpected and so out of the ordinary that it was considered a mistake by Lee Sedol. But it turned out to be a brilliant, never before seen move, and it was this move helped decide the game in AlphaGo's favor.

You can actually see the reactions to *Move 37* for yourself in this excerpt from the documentary "AlphaGo - The Movie": [Lee Sedol vs AlphaGo Move 37 reactions and analysis](https://www.youtube.com/watch?v=HT-UZkiOLv8).

Transfering this back to LLMs, we don't yet know what it will look like when an LLM will reason at a level beyond human capability. But we can say that **RL seems to be a vital step towards unlocking this potential.**

#### Reinforcement Learning with Human Feedback

Reinforcement Learning with Human Feedback (RLHF) [\[Ziegler, et al. 2019\]](https://arxiv.org/pdf/1909.08593) takes the concept of RL and adds a human in the loop.<br>So far, all the problems we looked at are of **verifiable correctness, like math problems**. This means, we can clearly determine if an LLM's response is correct and to be reinforced further, or not.

**What about writing jokes, or poems?** Those problems are of an **unverifiable correctness.** An LLM can generate a joke, but **how could one determine at a massive scale if jokes are funny or not?** Humans can, in a subjective way, but how could we transfer a good scoring mechanism to Reinforcement Learning of an LLM?

In principle, naively, we could have humans review every single one of the millions of jokes an LLM might generate, but that would be very time-consuming and expensive. **The scale becomes an issue:**

<center>
	<img src="./img/rlhf_naive.png" style="width: 500px; height: auto;" />
</center>

The core trick of RLHF is **indirection:** We will get humans involved, but only a little bit. We will train a seperate LLM, a so-called **reward model**, and we train it on the objective of imitating the few, but speaking human scores we gathered. This reward model is then used to bridge the scaling gap between the human judgement and what's needed for the LLM.

> [!NOTE]
> Instead of judging being done by a human, we train a simulator for human judgement that can be used to judge the LLM's responses. This is the essence of Reinforcement Learning with Human Feedback.

Given a prompt, the LLM that we want to apply RL to generates $n$ (attempts at) jokes. A human then rates these jokes in an ascending order of funniness, from 1 (best) to $n$ (worst). Ordering is easier for unverifiable correctness than scoring.

Now, **seperately from what the human just did**, for each of these $n$ generated jokes, the *reward model* gets to see the prompt and this joke and scores it. Note that this *reward model* scores, it doesn't order. **The *reward model's* scores are within a fixed, continuous range**, say $0$ (worst) to $1$ (best). We go on to do this for all $n$ jokes and note each of their scores. From that, a ranking by the *reward model* emerges.

<center>
	<img src="./img/rlhf_rundown.png" style="width: 500px; height: auto;" />
</center>

**The objective now is to minimize the difference in ordering between the human and the *reward model*.** In the image above, for example, you can see that the *reward model* and the human disagree on the ranking of the joke at human-decided rank 2. Based on differences like these, the *reward model* is optimized. As soon as this is deemed to be sufficiently achieved, the *reward model* is used to score the vast amounts of LLM-generated jokes.

> [!NOTE]
> RLHF is a specialized application of RL tailored for aligning AI behavior with human preferences.

Ok. So what do we make of RLHF?<br>Let's look at the upsides and downsides of RLHF:

| Upsides                                                                                                                                                        | Downsides                                                                                                                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Applicability in arbitrary domains**                                                                                                                         | **Lossy Simulation of Human Preferences**                                                                                                                                                                                         |
| - RLHF can be applied in domains where ground truth is unavailable or unverifiable.<br>- Enables training in subjective or creative tasks (e.g. jokes, poetry) | - Human feedback is a noisy and imperfect proxy for true preferences or correctness.<br>- Human annotators may have biases, inconsistencies, or limited expertise.<br>- Feedback may not generalize well to unseen or edge cases. |
| **Bridging the Discriminator-Generator Gap**                                                                                                                   | **Reward Hacking and Adversarial Exploits**                                                                                                                                                                                       |
| - Human feedback acts as a discriminator, guiding the model toward better outputs.<br>- Adding the possibility to train for unverifiable tasks is witnessed to further improve overall model performance<br>- In many cases, discriminating is much easier than generating. It is feasible to attain a reward model based on which the harder generation task can be improved through scale.                                                                             | - RLHF LLMs may accidentally "game" the *reward model* by exploiting its weaknesses.<br>- *Reward models* may fail to capture long-term or contextual quality.                                                                    |
| **Scalability and Efficiency**                                                                                                                                 | **Over-Optimization and Misalignment**                                                                                                                                                                                            |
| - Takes the valuable but expensive human feedback and simulates it through a *reward model*, allowing to scale to the demands of LLMs.                         | - Over-optimization for the reward model can lead to misalignment with true user preferences.                                                                                                                                     |
| **Iterative Improvement**                                                                                                                                      | **Ethical Risks**                                                                                                                                                                                                                 |
| - Allows for continuous improvement through iterative feedback and retraining.<br>- Models can adapt to evolving user preferences or new tasks over time.      | - RLHF may reinforce existing biases present in human feedback.<br>- Risk of misuse in generating (e.g. politically) persuasive or manipulative content.                                                                          |

---

Closing the circle, we can now answer all of the questions we initially set out to investigate:<br><br>
**What *exactly* are Large Language Models (LLMs) and tools like ChatGPT about?**<br>LLMs are a specialized kind of *neural network* models that are trained to process and generate text in a human-like fashion. Nowadays, LLMs are based on *transformer networks* that enable them to learn statistical relationships between tokens, i.e. individual text fragments, from large amounts of text. LLMs are pretrained on huge data sets on the scale of *the entire Internet*. In doing so, they iteratively learn patterns, structures and meanings of language. Tools like ChatGPT work based on LLMs that have been specially finetuned for conversational tasks through what's called *supervised finetuning*. OpenAI achieved this by additionally *post-training* models like GPT-4o on *task-specific* datasets that contain conversational patterns.<br>Note that new techniques are still being developed to make LLMs even better, such as self-induced web search and learning with human feedback (RLHF) for not clearly measurable expectations of the LLM (e.g. writing an actually funny joke). In addition, reinforcement learning is now used as a further post-training stage to improve the LLM at scale and solve tasks in the best possible way. This leads to LLMs generating thought processes before formulating an answer.

**How do they provide value?**<br>LLMs add value through their *tunable* ability to generate text and answer questions, while being *applicable across various tasks*, such as chatting, helping writing code or composing poetry. Most importantly, thanks to their *generalization capability*, LLMs can be applied at scale to unknown data.

**What goes on behind that text box that you type your prompts into?**<br>
A lot. We discussed the six fundamental steps:
1. **Tokenization** (see [chapter 8](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) for more details)**:**
When a user enters text, it is first converted into a sequence of *tokens*. 
These tokens are numerical representations of words or fragments of words.
There are different types of tokenization, such as Byte-Level Tokenization and Byte-Pair Encoding (BPE).

2. **Input to the LLM** (see chapters [7](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb), [8](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) and [9](../N009%20-%20Reproducing%20GPT-2/N009%20-%20Reproducing_GPT-2.ipynb) for more details)**:**
The token sequence is given as input to the trained LLM. Accordingly, the LLM never comes in contact with text as such, but only a *one-dimensional sequence of numbers* (tokens).
which are further processed in a large-scale mathematical process, in which the model's parameters are used to calculate the next token in the sequence. A very prominent component of this process is *transformer* with its *attention mechanisms*.

3. **Probability Distribution** (see [chapter 7](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb) for more details)**:**
The LLM produces not a single token, but a probability distribution over all the possible tokens in its vocabulary.
This distribution is a prediction of what tokens the LLM thinks will likely occur follow on the input sequence.

4. **Token Sampling** (see chapters [7](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb), [8](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) and [9](../N009%20-%20Reproducing%20GPT-2/N009%20-%20Reproducing_GPT-2.ipynb) for more details)**:**
A token is picked out from this probability distribution. This is generally done by sampling, in which the probability of the token is taken into account for the chance of picking it. This retains a chance for unique, more diverse answers to be returned when applied iteratively. However, it is also possible to just directly select the most probable token.

5. **Text Generation** (see chapters [7](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb), [8](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) and [9](../N009%20-%20Reproducing%20GPT-2/N009%20-%20Reproducing_GPT-2.ipynb) for more details)**:** The selected token is appended to the input and the process is repeated until a text is generated (or a specific `<|endoftext|>` gets sampled). This process is called *autoregressive generation*.

6. **Output:** The generated text is returned in response to the prompt.

<br><b>And that's a wrap!</b><br><br>We saw the three steps currently employed to create state-of-the-art LLMs: Pretraining, Supervised Finetuning, and Reinforcement Learning. The key thing you should take away from this is that LLMs should be understood as increasingly sophisticated tools. Even though we have some measures of mitigation, some problems like hallucinations or RLHF loopholes may still persist. **LLMs aren't infallible. Use them for drafting, not for blindly producing production code.**

---

## The Future of LLMs is Bright

Models are getting bigger and more flexible in what they can consume and produce. Multimodality is a big deal right now: LLMs may be fed text, but also images, audio, video. **The future of LLMs is multimodal.** We will probably, foreseeably, still call the models LLMs because at the end of the day, multimodality means to find ways of translating media into token sequences that the model can understand and assume interrelations for.

There is quite the buzz around the concept of **LLM agents**: LLMs that can interact with the world, that can be given tasks and that can solve them within long, coherent, contexts in which such agents may even backtrack and self-correct. **The future of LLMs is interactive** and you will likely increasingly become more of a supervisor to such agents.

LLMs are kind of their own thing right now, a lot of attention is given to standalone, LLM-centered applications like ChatGPT, but **the future of LLMs is in integration and invisibility.** LLMs will be part of many systems, but they will be seen as a part of that system.

Finally, **the future of LLMs is efficient and requires less and less test-time training**. The model should be able to adapt to new tasks and new prompts on the fly, without the need for extensive retraining. This is a big challenge, but it is also a big opportunity.

---

## How to Keep Up?

A great way to stay on top of the latest developments in LLMs is to [follow the research](https://arxiv.org/list/cs.AI/recent). The [Chatbot Arena](https://lmarena.ai) gives a great overview over current LLMs and their capabilities in comparison to one another. Take it with a grain of salt though. It appears that the ranking has been 'gamed' by some LLM providers. Use it as a starting point for exploration and research, not as the final word.

Another source of up-to-date information are newsletters. *Yes, newsletters.* There are some very high-quality ones like [AI News](https://buttondown.com/ainews) or [DeepLearning.Ai's The Batch](https://www.deeplearning.ai/the-batch/). They provide great, concise insights into the latest developments.

YouTube is also a great resource. [3Blue1Brown](https://www.youtube.com/@3Blue1Brown) and [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) provide a great resources for deep dives into maths and AI. And of course [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) is unmatched.

Finally, [X (formerly Twitter)](https://x.com) is still great for AI. A lot of top talent is on there. Follow the likes of [Andrej Karpathy](https://x.com/karpathy), [Ilya Sutskever](https://x.com/ilyasut) and [Will Brown](https://x.com/willccbb) for excellent commentary on the field.

You can access most of the models discussed here via [Hugging Face's Model Hub](https://huggingface.co/models) or the respective website of the LLM provider. For offline use, [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/) are recommended.
<br><br><br><br><br><br><br>
$\tiny{\text{What did Ilya see?}}$
