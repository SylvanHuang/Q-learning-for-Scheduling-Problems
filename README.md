# Q-learning-for-Permutation-Flow-shop-Scheduling-and-Job-Shop-Scheduling

In this project, two Scheduling Problems: Permutation Flowshop Scheduling Problem (P-FSSP), and Job Shop Scheduling Problem are solved by Q-learning.

## Content
The repository contains:

- One instance for each problem
- Two programs in python
- Dockerfile
- Presentation Slide to explain every step of Q-learning for P-FSSP

## Requirements

- numpy==1.20.1
- pandas==1.2.3
- openpyxl==2.6.0

## Usage

### Run directly

```bash
cd src
pip install -r requirements.txt
python Permutation_FSSP.py #For P-FSSP
python JSSP.py # For JSSP
```

### Run with Docker

```bash
sudo docker build -t qlearning .
sudo docker run -e PROBLEM=JSSP.py qlearning #For P-FSSP
sudo docker run -e PROBLEM=Permutation_FSSP.py qlearning # For JSSP
```

## Reference

[Yunior César Fonseca-Reyna(2017): Q-Learning algorithm performance for m-machine, n-jobs flow shop scheduling problems to minimize makespan](https://www.researchgate.net/publication/317371143_Q-learning_algorithm_performance_for_m-machine_n-jobs_flow_shop_scheduling_problems_to_minimize_makespan)

