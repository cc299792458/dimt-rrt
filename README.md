# DIMT-RRT
This repository contains a **Python** implementation of a kinodynamic planning algorithm integrating RRT with a steering method built on the "double-integrator minimum time" formulation from the paper **["Probabilistically Complete Kinodynamic Planning for Robot Manipulators with Acceleration Limits"](https://ieeexplore.ieee.org/document/6943083), IROS 2014**.

It operates several orders of magnitude faster than the [kinodynamic RRT](https://ieeexplore.ieee.org/document/770022) algorithm.

📌 Note: This implementation is a **reproduction** of the original paper. Some notations also refer to the paper [Fast smoothing of manipulator trajectories using optimal bounded-acceleration shortcuts](https://ieeexplore.ieee.org/document/5509683)

## ✨ Example: Solving a 2D kinodynamic planning problem.

<div align=center>
  <img src="dimt_rrt.gif" width="75%"/>
</div>

---

## 🛠 Usage
**[DIMT-RRT](https://github.com/cc299792458/dimt-rrt/blob/main/dimt_rrt.py)** corresponds to Section V of the paper. It differs from conventional kinodynamic RRT by employing a **steering** method that connects two states by solving the **Double-Integrator Minimum Time (DIMT)** problem. This approach enables a "connect" strategy rather than an "extend" strategy, which is one of the key factors behind its enhanced speed.

### Some Core Components
