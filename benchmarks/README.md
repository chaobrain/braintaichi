# Benchmark Tests

This directory contains a series of benchmark tests designed to evaluate performance. The test files are located in the `src/basic` and `src/model` folders, specifically:

- `basic` folder: Contains tests for basic matrix operations, event-driven CSR matrix operations and jitconn matrix operations.
- `model` folder: Contains tests for complex models, such as `test_COBA.py` and `test_MultipleArea.py`.


To run these tests, ensure all necessary dependencies are installed. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

or if you are using a CUDA-enabled GPU, you can install the dependencies using the following command:

```bash
pip install -r requirements_cuda.txt
```
