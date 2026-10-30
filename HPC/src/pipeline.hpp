#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <cmath>

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// HPC Concept: SoA layout
struct Dataset {
    int n_rows = 0;
    std::vector<std::vector<double>> X;   // X[feature][row]  (SoA layout)
    std::vector<int> y;                   // 0/1
    std::vector<double> G1, G2, absences; // raw values for suggestions
    std::vector<double> mean, stddev;     // filled by preprocess()
    std::vector<std::string> names;       // 20 feature names
};

struct Model {
    std::vector<double> w;                // 20 weights
    double b = 0.0;
    std::vector<double> loss_history;
};

struct Metrics { double accuracy = 0, precision = 0, recall = 0, f1 = 0; int tp = 0, fp = 0, tn = 0, fn = 0; };

struct BenchRow {
    int threads = 0; std::string schedule; int rows = 0;
    double gen_s = 0, preprocess_s = 0, train_s = 0, predict_s = 0;
    double speedup = 0, efficiency = 0, karp_flatt = 0, predict_speedup = 0;
};

using ProgressFn = std::function<void(int epoch, double loss, double elapsed_s)>;

Dataset load_uci(const std::string& path);
Dataset generate_synthetic(const Dataset& seed_ds, int n);
void preprocess(Dataset& ds);
Dataset to_raw_features(const Dataset& source);
void preprocess_from_training_rows(Dataset& ds, const std::vector<int>& train_rows);
void split(const Dataset& ds, std::vector<int>& train, std::vector<int>& test);
Model train(const Dataset& ds, const std::vector<int>& rows, int threads, double lr, double l2, int epochs, const std::atomic<bool>& stop, ProgressFn progress, const std::string& ckpt = "");
std::vector<double> predict_all(const Model& model, const Dataset& ds, int threads, const std::string& schedule);
Metrics evaluate(const Model& model, const Dataset& ds, const std::vector<int>& rows);
Metrics cross_validate(const Dataset& ds, int folds, int threads, std::vector<Metrics>& per_fold);
double karp_flatt(double speedup, int p);
bool save_checkpoint(const Model& model, double lr, double l2, const std::string& path, int epoch, int rows = 0);
bool load_checkpoint(Model& model, double lr, double l2, const std::string& path, int& epochs_done, int rows = 0);
std::vector<BenchRow> run_benchmark(const Dataset& seed, int rows, bool& verified, double& max_abs_diff);
std::string tier_of(double prob);
std::vector<std::string> suggestions_of(double absences, double studytime, double failures, double G2, double higher, double famsup);
std::vector<std::string> suggestions_of(const Dataset& ds, int row);

#endif // PIPELINE_HPP

