# metacognition-of-AI

Supporting repository for: "Measuring the metacognition of conversational AI models" (LINK TO BE ADDED). 

In this paper, we carry out two sets of experiments. 

• The meta-d' experiments consists in submitting type 1 and type 2 tasks to the AI models in order to estimate their meta-d' and $M_{ratio}$.

• The $c$-calibration experiments consists in submitting a type 1 task and designing the prompts in order to make the AI models calibrate accordingly their type-1 decision criterion $c$ (e.g., by incentivizing caution when responding either S1 or S2).

## Getting started ##

Clone this repository on your local machine by running:

```bash
git clone git@github.com:sshrichard/metacognition-of-AI.git
``` 
Be sure to add in the folder `metacognition-of-AI` the content of the folder `matlab` from this link: https://github.com/metacoglab/HMeta-d, which is the GitHub page of the paper "HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings" (Fleming, 2017) from which comes from the hierarchical Bayesian model we use in the current paper.

**Please read the usage section below that raises critical points.**

**Meta-d' experiments**
Executing the following line runs a meta-d' experiment under the sentiment analysis task (i.e., task A) via the **OpenAI** API:
```bash
python metacognition-of-AI/meta_d_AI_GPT.py
```
Executing the following line runs a meta-d' experiment under the sentiment analysis task (i.e., task A) via the **DeepSeek** API:
```bash
python metacognition-of-AI/meta_d_AI_deepseek.py
```
Executing the following line runs a meta-d' experiment under the sentiment analysis task (i.e., task A) via the **Mistral** API:
```bash
python metacognition-of-AI/meta_d_AI_mistral.py
```


**c-calibration experiments**
Executing the following line runs a c-calibration experiment under the sentiment analysis task (i.e., task A) via the **OpenAI** API:
```bash
python metacognition-of-AI/c_calibration_GPT.py
```
Executing the following line runs a c-calibration experiment under the sentiment analysis task (i.e., task A) via the **DeepSeek** API:
```bash
python metacognition-of-AI/c_calibration_deepseek.py
```
Executing the following line runs a c-calibration experiment under the sentiment analysis task (i.e., task A) via the **Mistral** API:
```bash
python metacognition-of-AI/c_calibration_mistral.py
```


## Changing task ##

**meta-d' experiments** 

`prompt_metad_A`, `prompt_metad_B` and `prompt_metad_C` are the prompts used for the meta-d' experiments under the sentiment analysis task (i.e., task A), the oral versus written classification task (i.e., task B) and the word depletion detection task (i.e., task C), respectively.

You can change the considered task by changing the prompt there:
```
%-------------------------------------------------------------------------%
% Prompt %
%-------------------------------------------------------------------------%
base_PROMPT = fileread("prompt_metad_A.txt");
%-------------------------------------------------------------------------%
```

Also, when switching to either task B or C, change the "loading dataset" part accordingly. This means: replace

```
dataset_name = 'SST-2';
numSamples = 100; % number of sentences submitted to the charbot

file_name = "train-00000-of-00001.parquet";
testData = parquetread(file_name);

sentences = testData.sentence; % column "sentence"
labels    = testData.label;    % column "label"

idx = randperm(height(testData), numSamples); % we randomly select some entries
sampledSentences = sentences(idx);
sampledLabels    = labels(idx);
```

by 

```
file_name = "Test4Plus_Raw.txt";
T = readtable(file_name, 'Delimiter', '\t', 'ReadVariableNames', false);
% If the separator is ',', use 'Delimiter', ','

sentences = T.Var2;    % column "sentence"
labels    = T.Var1;    % column "label"

idx = randperm(height(T), numSamples);

sampledSentences = sentences(idx);
sampledLabels    = labels(idx);
```

Also, don't forget to change the name of the data folder automatically created.

Finally, note that the current repo does not include the dataset "Test4Plus_Raw" because of its size. It is freely available in the following link: https://zenodo.org/records/7694423

**$c$-calibration experiments**

IN PROGRESS

## Usage ##

By default, `meta_d_AI_GPT.py`, `meta_d_AI_deepseek.py` and `meta_d_AI_mistral.py` interacts tithe OpenAI, DeepSeek and Mistral API, respectively, with the arguments "gpt-5", "deepseek-chat" and "mistral-large-latest", the latter two invoking DeepSeek-V3 and Large-Instruct-2411 at the time of our simulations. **We recomment caution to the users, especially regarding the model's name saved in the data files when running the manuscripts.**

`numSamples` specifies the number of sentences submitted to the model. The larger `numSamples`, the larger the probability that a technical issue arises when interacting with the API (see the Warning section below).

**meta-d' experiments** 

IN PROGRESS

**meta-d' experiments** 

IN PROGRESS

## Warning ##

In case of transient API/server errors (rate limits, timeouts, outages, ...), it is recommended to add some code handling errors and retries, like an exponential backoff retry loop, to maintain continuity in data collecting without a fatal crash. Similarly, a few lines are included to handle potential errors arising from response formatting.
