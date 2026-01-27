# metacognition-of-AI

Supporting repository for: "Measuring the metacognition of conversational AI models" (LINK TO BE ADDED). 

In this paper, we carry out two sets of experiments. 

    • The meta-d' experiments consists in submitting type 1 and type 2 tasks to the AI models in order to estimate their meta-d' and $M_ratio$.

    • The c-calibration experiments consists in modifying the prompts in order to make the AI models calibrate accordingly their type-1 decision criterion c (e.g., by incentivizing caution when responding either S1 or S2).

## Getting started ##

Clone this repository on your local machine by running:

```bash
git clone git@github.com:Bitbol-Lab/metacognition-of-AI.git
``` 
 

Executing the following line runs a working example of the 3D system:
```bash
python T4P-T6SS-interplay/T4P_T6SS_interplay_3D.py
```

Executing the following line runs a working example of the 2D system:
```bash
python T4P-T6SS-interplay/T4P_T6SS_interplay_2D.py
``` 


## Requirements ##

In order to use the functions `main`, Numba is required.

## Usage ##

In the files `T4P_T6SS_interplay_3D.py` and `T4P_T6SS_interplay_2D.py`, the function
`
main
`
simulates a 40^3 (resp. 100^2) large body-centred cubic (resp. triangular) lattice with 50 prey and 50 predators, with matching pili, during 10 minutes, and yields the number of prey, predators and lysing prey over time. These parameters can be tuned.

Besides, if you want to prevent the diffusion of aggregates as whole units, just replace line 245 `elif number_of_free_neighbors < 8:` (resp. `elif number_of_free_neighbors < 6:`) by `elif False:` so that the code dedicated to the diffusion of aggregates as whole units is never executed.

## Warning ##

Note that in the comments of the code we use the words particles, cells and bacteria in an interchangable fashion.
