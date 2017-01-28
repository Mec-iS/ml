"""
Distance:
 D(S, L) = -SUM(Li * log(Si))

Loss:
 L = 1/N * SUM( D( S(wXi + b), Li ))
 It is a measurement made on all the dataset.

Minimize Loss:
 * Gradient Descent, for two variables: 
     * -alpha * deltaL(w1, w2) 
     * use derivatives
"""