param n;
param ndp; # ndp is the NON delivery probability because we want to maximize the number of replicates and therefore we have to write p=1-ndp
param L;
param B;
param q{i in 1..n};

var x {i in 1..n} integer >= 1, <= L;

maximize deliveryrate: sum{i in 1..n} q[i] * (1-ndp^x[i]);

subject to capacity: sum{i in 1..n} x[i] <= L*B;
