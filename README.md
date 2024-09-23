# FastSBS
The files used for the experiments in Section6.
To run the codes, you will need Python 3.

## Requirements ##
Before running the codes, please follow the steps below.

### 1. Install g++ and [LEMON](https://lemon.cs.elte.hu/trac/lemon/) ###

```
sudo apt install g++ make cmake 
wget http://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
tar xvzf lemon-1.3.1.tar.gz
cd lemon-1.3.1
mkdir build
cd build
cmake ..
make
sudo make install
```

### 2. Make files ###

```
cd stealth-sampling
make
cd ../wasserstein
make
```


## To Reproduce the Results in the Paper
## 1. Synthetic
The following values may be optionally specified.

```
--N: Number of data in Data Set D (default: 1000)
--Nref: Number of data in Data Set D' (default: 200)
--Nsample: Number of data in Data Set Z (default: 200)
--d: Dimensionality of the data (default: 1)
--spos: Proportion of samples with positive label (default: 0.5)
--ypos_coef: Coefficient for the positive label (default: 0.2)
--n_parallel: Number of parallel jobs to run (default: -1 for all available cores)
--eps: Termination threshold (default: 1e-2)
--n_slice: The number of fixed slices. (default: 10)
--significance: the significance level of the Kolmogorov-Smirnov test (default: 0.05)
--start: Starting seed for data generation (default: 0)
--end: Ending seed for data generation (default: 100)
```

### 1.1 Preprocess
To generate the synthetic data for experiments, run
```
python synthetic.py preprocess
```

If one wishes to change options, run 
```
python synthetic.py preprocess --N <N> --Nref <Nref> --d <d> --spos <spos> --ypos_coef <ypos_coef> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 1.2. Sampling
To sample from the generated data with each method, run
```
python synthetic.py sampling
```

If one wishes to change options, run 
```
python synthetic.py sampling --N <N> --Nsample <Nsample> --spos <sampled_spos> --n_slice <n_slice> --eps <eps> --start <start_seed> --end <end_seed>
```

If you do not want to fix the number of slices, set "n_slice" to -1.

### 1.3. Evaluate
To evaluate the results from each sampling method, run
```
python synthetic.py evaluate
```

If one wishes to change options, run 
```
python synthetic.py evaluate --significance <significance> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 1.4. See Results

To see the results, check `plot_result.ipynb`.


## 2. COMPAS
The following values may be optionally specified.

```
--Nref: Number of data in Data Set D' (default: 1278)
--Nsample: Number of data in Data Set Z (default: 1000)
--n_parallel: Number of parallel jobs to run (default: -1 for all available cores)
--eps: Termination threshold (default: 1e-2)
--n_slice: The the number of fixed slices. (default: 10)
--start: Starting seed for data generation (default: 0)
--end: Ending seed for data generation (default: 100)
```

### 2.1 Preprocess
To generate compas data in experiments, run
```
python compas.py preprocess
```

If one wishes to change options, run 
```
python compas.py preprocess --Nref <Nref> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 2.2. Sampling
To sample from the generated data with each method, run
```
python compas.py sampling
```

If one wishes to change options, run 
```
python compas.py sampling --Nsample <Nsample> --n_slice <n_slice> --eps <eps> --start <start_seed> --end <end_seed>
```

If you do not want to fix the number of slices, set "n_slice" to -1.

### 2.3. Evaluate
To evaluate the results from each sampling method, run
```
python compas.py evaluate
```

If one wishes to change options, run 
```
python compas.py evaluate --Nsample <Nsample> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 2.4. See Results

To see the results, check `plot_result.ipynb`.


## 3. Adult
The following values may optionally be specified.

```
--Ntr: Number of data in train data (default: 10000)
--Nte: Number of data in test data (default: 20000)
--Nsample: Number of data in Data Set Z (default: 2000)
--n_parallel: Number of parallel jobs to run (default: -1 for all available cores)
--eps: Termination threshold (default: 1e-1)
--n_slice: The the number of fixed slices. (default: 10)
--start: Starting seed for data generation (default: 0)
--end: Ending seed for data generation (default: 100)
```

### 3.1 Preprocess
To generate adult data in experiments, run
```
python adult.py preprocess [model]
```

where type is one of {LogReg, Forest}, where LogReg denotes Logistic Regression, and Forest denotes Random Forest.

If one wishes to change options, run 
```
python adult.py preprocess [model] --Ntr <Ntr> --Nte <Nte> model <model> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 3.2. Sampling
To sample from the generated data with each method, run
```
python adult.py sampling [model]
```

where type is one of {LogReg, Forest}, where LogReg denotes Logistic Regression, and Forest denotes Random Forest.

If one wishes to change options, run 
```
python adult.py sampling [model] --Nsample <Nsample> model <model> --n_slice <n_slice> --eps <eps> --start <start_seed> --end <end_seed>
```

If you do not want to fix the number of slices, set "n_slice" to -1.

### 3.3. Evaluate
To evaluate the results from each sampling method, run
```
python adult.py evaluate [model]
```

where type is one of {LogReg, Forest}, where LogReg denotes Logistic Regression, and Forest denotes Random Forest.

If one wishes to change options, run 
```
python adlut.py evaluate [model] --Nsample <Nsample> model <model> --start <start_seed> --end <end_seed> --n_parallel <n_parallel>
```

### 3.4. See Results

To see the results, check `plot_result.ipynb`.

