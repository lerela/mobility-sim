mobility-sim
============

Nodes placement and requests simulation in real mobility traces (CEDO and CEDO+ demonstration).

Introduction
------------

mobility-sim is a Python tool that goes through a mobility trace. Each epoch must include the coordinates of all the nodes. The trace must follow the following format:
```
timestamp node_id x_coordinate y_coordinate
```

The trace is browsed once and compiled for later use with nodes meetings and inter-meeting times.

The delivery probabilities can be estimated following the heuristic developed in CEDO (uniform Poisson mobility) and CEDO+ (heterogeneous mobility with unknown law).

Random contents can then be generated and assigned to nodes following the global placement algorithms derived in CEDO and CEDO+ (requires AMPL with an integer programming solver like KNITRO or Couenne).

Random requests are then generated along the trace respecting the popularity rates of the contents and delivery statistics are computed (mainly (un)satisfied requests and mean delivery time).
