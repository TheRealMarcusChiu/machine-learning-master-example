requires python 3.6.4

example command
<code> python igh.py 1 2 training_set.csv validation_set.csv test_set.csv yes</code>
<code> python vih.py 1 2 training_set.csv validation_set.csv test_set.csv yes</code>


# Reports

- __data set 1__
  - __Report the accuracy on the test set for decision trees constructed using the two heuristics mentioned above.__
    - igh accuracy: 75.85%
    - vih accuracy: 69.3%
  - __Choose 10 suitable values for L and K (not 10 values for each, just 10 combinations). For each of them, report the accuracies for the post-pruned decision trees constructed using the two heuristics.__
    - L=1 K=2 : vih=69.85% igh=76.15%
    - L=1 K=5 : vih=68.95% igh=76.6%
    - L=1 K=10 : vih=76.55% igh=76.15%
    - L=10 K=2 : vih=69.6% igh=76.15%
    - L=10 K=5 : vih=69.85% igh=76.55%
    - L=10 K=10 : vih=71.35% igh=75.4%
    - L=5 K=2 : vih=69.2% igh=76.6%
    - L=5 K=5 : vih=70.25% igh=75.85%
    - L=5 K=10 : vih=66.75% igh=76.5%
    - L=20 K=20 : vih=68.85% igh=76.25%
    
- __data set 2__
  - __Report the accuracy on the test set for decision trees constructed using the two heuristics mentioned above.__
    - igh accuracy: 72.33%
    - vih accuracy: 67.66%
  - __Choose 10 suitable values for L and K (not 10 values for each, just 10 combinations). For each of them, report the accuracies for the post-pruned decision trees constructed using the two heuristics.__
    - L=1 K=2 : vih=67.66% igh=72.33%
    - L=1 K=5 : vih=67.66% igh=72.33%
    - L=1 K=10 : vih=76.55% igh=72.33%
    - L=10 K=2 : vih=69.6% igh=74%
    - L=10 K=5 : vih=68.33% igh=73.66%
    - L=10 K=10 : vih=67.83% igh=73.66%
    - L=5 K=2 : vih=67.66% igh=73.5%
    - L=5 K=5 : vih=67.66% igh=72.33%
    - L=5 K=10 : vih=65.83% igh=74.33%
    - L=20 K=20 : vih=68.83% igh=74.33%
