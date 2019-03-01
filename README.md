# ASTRO 528 Branch of GP-Research Github

This branch contains the additional work that has been performed on this project Github since the start of the class project.

The purpose of the 528 part of the project is to see the likelihood of different potential keplerian planet signals within a dataset. Eventually we will be fitting with all of the Keplerian orbital parameters, but for now, we are setting eccentricity to zero and inclination to pi/2 to simplify things. The dataset is a draw from a multivariate GP that is meant to be RV noise and some stellar activity indicators. We then can inject a known signal and see how well we recover it, and the hyperparameters of the original GP we drew from.

Most of the work that should be focused on for code review is in the 528, src, and test folders. These are all in the julia folder. Ignore the python folder. The main script in the 528 folder is optimize_periods.jl. All of the functions that are called are held in the src directory and are imported at the start of the script with 

```julia
include("../src/all_functions.jl")"
```

That call also runs all of my tests which are contained in the test folder. If you aren't sure what was there before the class project, you can always check the comparison between the master branch (which I haven't updated) and the 528 branch with the following link: <https://github.com/christiangil/GP-Research/compare/master...christiangil:528>

In src, you should mostly look at general_functions.jl and RV functions.jl

Thanks for checking out my code!

-Christian
