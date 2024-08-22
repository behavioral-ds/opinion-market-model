# Opinion Market Model

Code accompanying the paper "Opinion Market Model: Stemming Far-Right Opinion Spread
using Positive Interventions" [[Calderon, et al. '24]](https://ojs.aaai.org/index.php/ICWSM/article/view/31306/33466).

## Description

This repo contains the implementation for the bushfire case study discussed in the paper.

The code is split into two version folders:

`v1/`: version with regularization on two levels

`v2/`: version with platform-specific structure regularization (see Appendix D)

Each folder contains the following:

`opinion_resources_opt1.py` : opinion volume (tier 1) model estimation

`opinion_resources_opt2.py` : opinion share (tier 2) model estimation

`opinion_analysis_v2.py` : generative model sampling and fit evaluation routines

`opinion_resources_twolevel.py` : two-level fitting and post-fitting calculations

`opinion_main_varyT_2D.py`: main routine to be invoked through CLI

## License

Both dataset and code are distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/). If you require a different license, please contact us at <piogabrielle.b.calderon@student.uts.edu.au>
or <Marian-Andrei@rizoiu.eu>.

