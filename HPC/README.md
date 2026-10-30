# Student Performance Prediction System (SPS-HPC)

SPS-HPC is a high-performance C++17 application for Windows that trains a logistic regression model in parallel using OpenMP to predict student pass/fail outcomes, benchmarks multi-threaded scaling, and serves an interactive web GUI.

## HPC Concepts Implementation

| Concept | Syllabus | Location | Implementation Details |
|---|---|---|---|
| `parallel for` | Unit 5 | Preprocessing, synthetic generation, training, prediction | `#pragma omp parallel for schedule(static)` |
| `reduction` | Unit 5 | Mean/stddev, loss sum, gradient combine | OpenMP `reduction(+:...)` clauses |
| Thread-local gradient buffers | Unit 2, 5 | Training gradient computation | `std::vector<std::vector<double>> g(threads, ...)`, eliminates false sharing & locks |
| SIMD | Unit 2 | Training dot product loop | `#pragma omp simd reduction(+:dot)` vectorizes feature dot product |
| SoA layout | Unit 2 | `Dataset::X` | Structure of Arrays: one contiguous vector per feature |
| Loop schedules | Unit 5 | Prediction inference | Selectable `static`, `dynamic,1024`, and `guided` schedules |
| Task-level parallelism | Unit 2 | K-fold cross-validation | `#pragma omp parallel for schedule(dynamic)` across folds (1 thread per fold) |
| Checkpointing | Unit 4 | Training persistence | Automatically saves/resumes weights to `data/checkpoint.txt` every 50 epochs; loading a new dataset clears the old checkpoint to prevent reuse across datasets |
| Speedup, efficiency | Unit 2 | Benchmarking suite | Measures scaling metrics: $S = T(1)/T(p)$, $E = S/p$ |
| Karp-Flatt serial fraction | Unit 2 | Benchmarking suite | Quantifies serial bottleneck: $e = (1/S - 1/p)/(1 - 1/p)$ |
| Timing and thread control | Unit 5 | Throughout pipeline | `omp_get_wtime()`, `omp_set_num_threads(p)` across compute sections |

## Windows Setup & Build

1. Install MSYS2 from [msys2.org](https://www.msys2.org/).
2. Open the MSYS2 **UCRT64** terminal and install GCC with OpenMP support:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc
   ```
3. Ensure `student-mat.csv` is placed in `data/student-mat.csv` (UCI Student Performance dataset).
4. Run `build.bat` from the project root in Command Prompt / PowerShell:
   ```bat
   build.bat
   ```
5. Launch `sps.exe` and navigate to `http://127.0.0.1:8080` in your browser.

## GUI Tour

The web GUI provides a four-stage workflow through a left navigation sidebar. **Data & Train** allows ingesting UCI or synthetic datasets, training logistic regression with live SVG loss curves, and evaluating parallel cross-validation. **Predict All** runs multi-threaded batch inference across chosen OpenMP loop schedules and presents paginated student risk breakdowns. **Student Single** provides an interactive form to inspect individual student features and receive tailored intervention suggestions. **Benchmark** evaluates execution time, speedup, efficiency, and Karp-Flatt serial fraction across thread counts with real-time verification.

## Limitations

- The model uses 20 demographic, social, and academic background features. G1 and G2 (prior-period grades) and G3 (final grade) are excluded from the model input feature vector; G3 defines the binary pass/fail target (pass = 1 if G3 >= 10), while G2 and absences are reserved for post-prediction guidance.
- Synthetic large datasets are generated algorithmically for HPC parallel scaling and benchmarking workloads, and should not be interpreted as independent real-world student data.
- Reading CSV input files is bound by serial disk I/O, constraining speedup during initial loading.
- Speedup flattens at higher thread counts due to CPU memory bandwidth limits and serial code sections (Karp-Flatt metric).
- The browser rendering and embedded HTTP server run locally on the host machine and are excluded from benchmark timings.


## Validation and benchmark integrity notes

- The train/test split is created before normalization. Normalization statistics are fitted on training rows only and then applied to all rows.
- Each cross-validation fold independently fits its normalization statistics using only that fold's training rows.
- Benchmark data generation starts from raw feature values, not previously standardized values.
- Benchmark timings are measured locally; results vary by CPU, compiler, OpenMP runtime, workload size, and background load. Rerun the benchmark before reporting numbers.
