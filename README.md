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

Executing the following line runs meta-d' experiment under the sentiment analysis task (i.e., task A) via the **OpenAI** API:
```bash
python metacognition-of-AI/meta_d_AI_GPT.py
```
Executing the following line runs meta-d' experiment under the sentiment analysis task (i.e., task A) via the **DeepSeek** API:
```bash
python metacognition-of-AI/meta_d_AI_deepseek.py
```
Executing the following line runs meta-d' experiment under the sentiment analysis task (i.e., task A) via the **Mistral** API:
```bash
python metacognition-of-AI/meta_d_AI_mistral.py
```
Note that, by default, `meta_d_AI_GPT.py`, `meta_d_AI_deepseek.py` and `meta_d_AI_mistral.py` interacts tithe OpenAI, DeepSeek and Mistral API, respectively, with the arguments "gpt-5", "deepseek-chat" and "mistral-large-latest", the latter two invoking DeepSeek-V3 and Large-Instruct-2411 at the time of our simulations. **We recomment caution to the users, especially regarding the model's name saved in the data files when running the manuscripts.**

Executing the following line runs a c-calibration experiment under the sentiment analysis task with GPT-5:
```bash
python metacognition-of-AI/c_calibration_AI.py
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

**c-calibration experiments**

IN PROGRESS

## Usage ##

In the files `T4P_T6SS_interplay_3D.py` and `T4P_T6SS_interplay_2D.py`, the function
`
main
`
simulates a 40^3 (resp. 100^2) large body-centred cubic (resp. triangular) lattice with 50 prey and 50 predators, with matching pili, during 10 minutes, and yields the number of prey, predators and lysing prey over time. These parameters can be tuned.

Besides, if you want to prevent the diffusion of aggregates as whole units, just replace line 245 `elif number_of_free_neighbors < 8:` (resp. `elif number_of_free_neighbors < 6:`) by `elif False:` so that the code dedicated to the diffusion of aggregates as whole units is never executed.

## Warning ##

In case of transient API/server errors (rate limits, timeouts, outages, ...), it is recommended to add some code handling errors and retries, like an exponential backoff retry loop, to maintain continuity in data collecting without a fatal crash.
