param n;
param L;
param B;
param p{j in 1..L};
param q{i in 1..n};

var x {i in 1..n, j in 1..L} >= 0, <= 1;

maximize deliveryrate: sum{i in 1..n} q[i] * (1 - prod{j in 1..L}(1 - p[j] * x[i, j] ));

subject to min_replicates {i in 1..n}: sum{j in 1..L} x[i, j] >= 1;
subject to capacity {j in 1..L}: sum{i in 1..n} x[i, j] = B;
