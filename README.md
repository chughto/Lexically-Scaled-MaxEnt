Scale learner
=============
* Code developed at the University of Massachusetts Amherst; Summer 2018	
* Call scaled_weights_MaxEnt.py with Python 2 and all of the necessary command line arguments:

	python scaled_weights_MaxEnt.py METHOD NEG_WEIGHTS NEG_SCALES LAMBDA1 L2_PRIOR INIT_WEIGHT RAND_WEIGHTS ETA EPOCHS TD_FILE LANGUAGE OUTPUT_DIR

Where:
*METHOD is chosen from the set {"GD_CLIP", "GD", "L-BFGS-B"}, for "clipped gradient descent" (Tsuruoka et al. 2009), "gradient descent", and "L-BFGS-B" (Byrd et al. 1995), respectively.
*NEG_WEIGHTS is 0/1, to indicate whether you want to allow negative constraint weights
*NEG_SCALES is 0/1, to indicate whether you want to allow negative scales
*LAMBDA1 is the weight on the prior (to have different weights on the scales' and weights' priors, change the value of the "LAMBDA2" variable on line 16 of the code).
*L2_PRIOR is 0/1 to indicate if you want an L2_Prior (default is L1; L2 only works right now for the L-BFGS-B method
*INIT_WEIGHT is where you want all weights to begin (scales always start at 0.0)
*RAND_WEIGHTS is 0/1 to indicate whether you want initial weights to be randomly sampled from 0-10 (this makes the previous parameter irrelevant)
*ETA is the learning rate
*EPOCHS is how many batch updates you want to run
*TD_FILE is the location+name of the training file (if it's in the same directory as scaled_weights_MaxEnt.py, you won't need to include the location)
*LANGUAGE is the name of the training file without the file extension or location
*OUTPUT_DIR is the directory you want to write the output files to--this needs to include whatever slashes your operating system uses in file paths at the end of it. (e.g. "Output_Files/" for Windows). If you just want to write in the same directory as the script, you can put an empty string in for this parameter.

Data files
-----
* "CCCcategorical.tsv" is a language with categorical deletion, blocked by creation of CCC clusters
* "CCClexical.tsv" is a language with lexically conditioned deletion (with deletion only being blocked by some CCC cluster creation)
* "CCCvariation.tsv" is a language with deletion that's blocked variably by CCC cluster creation
* "CCCvariationlexical1.tsv" is a language where the probability of deletion happening despite CCC clusters is variable and the variability is lexically specified

* "CCCcategoricallexical0.tsv" Same as "CCClexical.tsv", but with 10% of CCC-creating deletion occuring
* "CCCcategoricallexical1.tsv" Same as "CCClexical.tsv", but with 20% of CCC-creating deletion occuring
* "CCCcategoricallexical2.tsv" Same as "CCClexical.tsv", but with 30% of CCC-creating deletion occuring
* "CCCcategoricallexical3.tsv" Same as "CCClexical.tsv", but with 40% of CCC-creating deletion occuring
* "CCCcategoricallexical4.tsv" Same as "CCClexical.tsv", but with 50% of CCC-creating deletion occuring
* "CCCcategoricallexical5.tsv" Same as "CCClexical.tsv", but with 60% of CCC-creating deletion occuring
* "CCCcategoricallexical6.tsv" Same as "CCClexical.tsv", but with 70% of CCC-creating deletion occuring
* "CCCcategoricallexical7.tsv" Same as "CCClexical.tsv", but with 80% of CCC-creating deletion occuring
* "CCCcategoricallexical8.tsv" Same as "CCClexical.tsv", but with 90% of CCC-creating deletion occuring
* "CCCcategoricallexical9.tsv" Same as "CCClexical.tsv", but with 100% of CCC-creating deletion occuring

Data file format
----------------
* Data files largely follow the OTSoft / OTHelp TSV format
* First line: list of morphemes in the data set. The data_files here enumerate prefixes (p0, p1, ...) and stems (s0, s1, ...)
* Second line: list of constraints
* Following lines give input/candidate pairs
+ First line contains the input. Inputs are labeled with the morphemes they contain. Three dollar signs $$$ separate the input from the list and one dollar sign $ separates morphemes from each other 
+ Input is separated from candidate by a tab, probability of that candidate, and then a tab-separated list of constraint violations
+ Inputs are only listed for the first candidate; following candidate lines begin with a tab
* Partial example from TOYcategorical.tsv:

| Labels (not in file) |                   |          |    |     |     |       |     |
|:---------------------|:------------------|:---------|:--:|:---:|:---:|:-----:|:---:|
| *Morphemes*          | p0                | p1	      | p2 | s0	 | s1  | s2    | s3  |
| *Constraints*        |                   |          |    | CCC | Max | Align | SSP |
| *Input 1 - Cand 1*   | ape-taba$$$p0$s0  | apetaba  |	0  | 0	 | 1   | 0     | 0   |
| *Input 1 - Cand 2*   |                   | aptaba   |	1  | 0	 | 1   | 0     | 5   |
| *Input 2 - Cand 1*   | ape-ttaba$$$p0$s1 | apettaba |	1  | 0	 | 0   | 1     | 5   |
| *Input 2 - Cand 2*   |                   | apttaba  |	0  | 1	 | 1   | 0     | 1   |

* For more details, see our paper:

	Hughto, Coral; Lamont, Andrew; Prickett, Brandon; and Jarosz, Gaja (2019) "Learning Exceptionality and Variation with Lexically Scaled MaxEnt," Proceedings of the Society for Computation in Linguistics: Vol. 2 , Article 11. 
	DOI: https://doi.org/10.7275/y68s-kh12 
	Available at: https://scholarworks.umass.edu/scil/vol2/iss1/11
