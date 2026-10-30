#include "pipeline.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include <map>


static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n\""), e = s.find_last_not_of(" \t\r\n\"");
    return (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
}

Dataset load_uci(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) { std::cerr << "Expected file at: " << path << std::endl; std::exit(1); }
    Dataset ds;
    ds.names = {"sex","age","Medu","Fedu","traveltime","studytime","failures","schoolsup","famsup","paid","activities","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health"};
    ds.X.assign(20, {});
    std::string hline, tok;
    if (!std::getline(file, hline)) return ds;
    std::stringstream hs(hline);
    std::vector<std::string> cols;
    while (std::getline(hs, tok, ';')) cols.push_back(trim(tok));
    auto find_col = [&](const std::string& n) {
        for (int i = 0; i < (int)cols.size(); ++i) if (cols[i] == n) return i;
        return -1;
    };
    std::vector<int> fi(20);
    for (int f = 0; f < 20; ++f) fi[f] = find_col(ds.names[f]);
    int c_abs = find_col("absences"), c_g1 = find_col("G1"), c_g2 = find_col("G2"), c_g3 = find_col("G3");
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<std::string> v;
        while (std::getline(ss, tok, ';')) v.push_back(trim(tok));
        if (v.size() < cols.size()) continue;
        for (int f = 0; f < 20; ++f) {
            double val = 0.0;
            if (ds.names[f] == "sex") val = (v[fi[f]] == "M") ? 1.0 : 0.0;
            else if (v[fi[f]] == "yes") val = 1.0;
            else if (v[fi[f]] == "no") val = 0.0;
            else val = std::stod(v[fi[f]]);
            ds.X[f].push_back(val);
        }
        ds.y.push_back(std::stod(v[c_g3]) >= 10.0 ? 1 : 0);
        ds.absences.push_back(std::stod(v[c_abs]));
        ds.G1.push_back(std::stod(v[c_g1]));
        ds.G2.push_back(std::stod(v[c_g2]));
        ds.n_rows++;
    }
    ds.mean.assign(20, 0.0); ds.stddev.assign(20, 1.0);
    return ds;
}

// Synthetic large dataset: parallel for, one RNG per thread seeded 42 + thread_id
Dataset generate_synthetic(const Dataset& seed_ds, int n) {
    Dataset ds;
    ds.names = seed_ds.names; ds.n_rows = n;
    ds.X.assign(20, std::vector<double>(n));
    ds.y.assign(n, 0); ds.G1.assign(n, 0.0); ds.G2.assign(n, 0.0); ds.absences.assign(n, 0.0);
    std::vector<double> min_v(20), max_v(20), s_std(20);
    std::vector<bool> is_bin(20);
    for (int f = 0; f < 20; ++f) {
        min_v[f] = *std::min_element(seed_ds.X[f].begin(), seed_ds.X[f].end());
        max_v[f] = *std::max_element(seed_ds.X[f].begin(), seed_ds.X[f].end());
        is_bin[f] = (min_v[f] == 0.0 && max_v[f] == 1.0);
        double sum = 0, sq = 0;
        for (double v : seed_ds.X[f]) { sum += v; sq += v * v; }
        double m = sum / seed_ds.n_rows;
        s_std[f] = std::sqrt(std::max(1e-12, (sq / seed_ds.n_rows) - m * m));
    }
    // HPC Concept: Timing and thread control
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(42 + tid);
        std::uniform_int_distribution<int> pick(0, seed_ds.n_rows - 1);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::normal_distribution<double> n01(0.0, 1.0);
        // HPC Concept: parallel for
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            int src = pick(rng);
            for (int f = 0; f < 20; ++f) {
                if (is_bin[f]) {
                    double v = seed_ds.X[f][src];
                    if (u01(rng) < 0.02) v = (v == 0.0 ? 1.0 : 0.0);
                    ds.X[f][i] = v;
                } else {
                    double v = std::round(seed_ds.X[f][src] + n01(rng) * (0.05 * s_std[f]));
                    ds.X[f][i] = std::max(min_v[f], std::min(max_v[f], v));
                }
            }
            ds.absences[i] = std::max(0.0, seed_ds.absences[src] + std::round(n01(rng) * 0.5));
            ds.G1[i] = std::max(0.0, std::min(20.0, seed_ds.G1[src] + std::round(n01(rng) * 0.5)));
            ds.G2[i] = std::max(0.0, std::min(20.0, seed_ds.G2[src] + std::round(n01(rng) * 0.5)));
            double g3_val = std::max(0.0, std::min(20.0, (seed_ds.y[src] ? 12.0 : 8.0) + std::round(n01(rng) * 1.0)));
            ds.y[i] = (g3_val >= 10.0 ? 1 : 0);
        }
    }
    ds.mean.assign(20, 0.0); ds.stddev.assign(20, 1.0);
    return ds;
}

void preprocess(Dataset& ds) {
    for (int f = 0; f < 20; ++f) {
        double sum = 0.0;
        // HPC Concept: reduction
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int i = 0; i < ds.n_rows; ++i) sum += ds.X[f][i];
        double m = sum / ds.n_rows, sq = 0.0;
        // HPC Concept: reduction
        #pragma omp parallel for reduction(+:sq) schedule(static)
        for (int i = 0; i < ds.n_rows; ++i) { double d = ds.X[f][i] - m; sq += d * d; }
        double s = std::max(1e-12, std::sqrt(sq / ds.n_rows));
        ds.mean[f] = m; ds.stddev[f] = s;
        // HPC Concept: parallel for
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ds.n_rows; ++i) ds.X[f][i] = (ds.X[f][i] - m) / s;
    }
}

// Reconstruct raw feature values from a previously normalized dataset.
Dataset to_raw_features(const Dataset& source) {
    Dataset raw = source;
    if ((int)source.mean.size() == 20 && (int)source.stddev.size() == 20) {
        for (int f = 0; f < 20; ++f) {
            for (int i = 0; i < source.n_rows; ++i)
                raw.X[f][i] = source.X[f][i] * source.stddev[f] + source.mean[f];
        }
    }
    raw.mean.assign(20, 0.0);
    raw.stddev.assign(20, 1.0);
    return raw;
}

// Fit normalization statistics only on the supplied training rows, then transform all rows.
void preprocess_from_training_rows(Dataset& ds, const std::vector<int>& train_rows) {
    if (train_rows.empty()) return;
    ds.mean.assign(20, 0.0);
    ds.stddev.assign(20, 1.0);
    for (int f = 0; f < 20; ++f) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int k = 0; k < (int)train_rows.size(); ++k) sum += ds.X[f][train_rows[k]];
        const double mean = sum / train_rows.size();
        double squared = 0.0;
        #pragma omp parallel for reduction(+:squared) schedule(static)
        for (int k = 0; k < (int)train_rows.size(); ++k) {
            const double d = ds.X[f][train_rows[k]] - mean;
            squared += d * d;
        }
        const double sd = std::max(1e-12, std::sqrt(squared / train_rows.size()));
        ds.mean[f] = mean;
        ds.stddev[f] = sd;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ds.n_rows; ++i) ds.X[f][i] = (ds.X[f][i] - mean) / sd;
    }
}

void split(const Dataset& ds, std::vector<int>& train, std::vector<int>& test) {
    std::vector<int> idx(ds.n_rows);
    for (int i = 0; i < ds.n_rows; ++i) idx[i] = i;
    std::mt19937 rng(42); std::shuffle(idx.begin(), idx.end(), rng);
    int n_tr = (int)(0.8 * ds.n_rows);
    train.assign(idx.begin(), idx.begin() + n_tr);
    test.assign(idx.begin() + n_tr, idx.end());
}

// HPC Concept: Checkpointing
bool save_checkpoint(const Model& model, double lr, double l2, const std::string& path, int epoch, int rows) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << lr << " " << l2 << " " << epoch << " " << rows << "\n";
    for (int f = 0; f < 20; ++f) out << model.w[f] << " ";
    out << model.b << "\n";
    return true;
}

bool load_checkpoint(Model& model, double lr, double l2, const std::string& path, int& epochs_done, int rows) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    double c_lr = 0, c_l2 = 0; int ep = 0, c_rows = 0;
    if (!(in >> c_lr >> c_l2 >> ep)) return false;
    if (in >> c_rows) {
        if (rows > 0 && c_rows > 0 && c_rows != rows) return false;
    }
    if (std::fabs(c_lr - lr) > 1e-9 || std::fabs(c_l2 - l2) > 1e-9) return false;
    model.w.assign(20, 0.0);
    for (int f = 0; f < 20; ++f) if (!(in >> model.w[f])) return false;
    if (!(in >> model.b)) return false;
    epochs_done = ep;
    return true;
}

Model train(const Dataset& ds, const std::vector<int>& rows, int threads, double lr, double l2, int epochs, const std::atomic<bool>& stop, ProgressFn progress, const std::string& ckpt) {
    Model m; m.w.assign(20, 0.0); m.b = 0.0;
    int start_ep = 0;
    if (!ckpt.empty()) load_checkpoint(m, lr, l2, ckpt, start_ep, ds.n_rows);
    if (start_ep >= epochs) { start_ep = 0; m.w.assign(20, 0.0); m.b = 0.0; }
    // HPC Concept: Timing and thread control
    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();
    int n = (int)rows.size();
    for (int ep = start_ep + 1; ep <= epochs; ++ep) {
        // HPC Concept: Thread-local gradient buffers
        std::vector<std::vector<double>> g(threads, std::vector<double>(21, 0.0));
        std::vector<double> loss_buf(threads, 0.0);
        // HPC Concept: parallel for
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < n; ++idx) {
            int t = omp_get_thread_num(), i = rows[idx];
            double dot = 0.0;
            // HPC Concept: SIMD
            #pragma omp simd reduction(+:dot)
            for (int f = 0; f < 20; ++f) dot += m.w[f] * ds.X[f][i];
            double p = sigmoid(dot + m.b), err = p - ds.y[i];
            for (int f = 0; f < 20; ++f) g[t][f] += err * ds.X[f][i];
            g[t][20] += err;
            double pc = std::max(1e-12, std::min(1.0 - 1e-12, p));
            loss_buf[t] += -(ds.y[i] * std::log(pc) + (1 - ds.y[i]) * std::log(1.0 - pc));
        }
        for (int f = 0; f < 20; ++f) {
            double sum_f = 0.0;
            for (int t = 0; t < threads; ++t) sum_f += g[t][f];
            m.w[f] -= lr * (sum_f / n + l2 * m.w[f]);
        }
        double sum_b = 0.0, tot_loss = 0.0;
        for (int t = 0; t < threads; ++t) { sum_b += g[t][20]; tot_loss += loss_buf[t]; }
        m.b -= lr * (sum_b / n);
        tot_loss /= n;
        m.loss_history.push_back(tot_loss);
        double elapsed = omp_get_wtime() - t0;
        if (progress) progress(ep, tot_loss, elapsed);
        if (stop.load()) break;
        if (!ckpt.empty() && ep % 50 == 0) save_checkpoint(m, lr, l2, ckpt, ep, ds.n_rows);
    }
    return m;
}

std::vector<double> predict_all(const Model& model, const Dataset& ds, int threads, const std::string& schedule) {
    // HPC Concept: Timing and thread control
    omp_set_num_threads(threads);
    std::vector<double> p(ds.n_rows);
    // HPC Concept: Loop schedules
    if (schedule == "dynamic") {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < ds.n_rows; ++i) {
            double dot = 0.0;
            for (int f = 0; f < 20; ++f) dot += model.w[f] * ds.X[f][i];
            p[i] = sigmoid(dot + model.b);
        }
    } else if (schedule == "guided") {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < ds.n_rows; ++i) {
            double dot = 0.0;
            for (int f = 0; f < 20; ++f) dot += model.w[f] * ds.X[f][i];
            p[i] = sigmoid(dot + model.b);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < ds.n_rows; ++i) {
            double dot = 0.0;
            for (int f = 0; f < 20; ++f) dot += model.w[f] * ds.X[f][i];
            p[i] = sigmoid(dot + model.b);
        }
    }
    return p;
}

Metrics evaluate(const Model& model, const Dataset& ds, const std::vector<int>& rows) {
    Metrics m;
    if (rows.empty()) return m;
    for (int i : rows) {
        double dot = 0.0;
        for (int f = 0; f < 20; ++f) dot += model.w[f] * ds.X[f][i];
        int pred = (dot + model.b >= 0.0) ? 1 : 0, act = ds.y[i];
        if (pred == 1 && act == 1) m.tp++;
        else if (pred == 1 && act == 0) m.fp++;
        else if (pred == 0 && act == 0) m.tn++;
        else m.fn++;
    }
    int tot = m.tp + m.fp + m.tn + m.fn;
    m.accuracy = tot > 0 ? (double)(m.tp + m.tn) / tot : 0.0;
    m.precision = (m.tp + m.fp) > 0 ? (double)m.tp / (m.tp + m.fp) : 0.0;
    m.recall = (m.tp + m.fn) > 0 ? (double)m.tp / (m.tp + m.fn) : 0.0;
    m.f1 = (m.precision + m.recall) > 0 ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    return m;
}

// HPC Concept: Task-level parallelism
Metrics cross_validate(const Dataset& ds, int folds, int threads, std::vector<Metrics>& per_fold) {
    if (ds.n_rows < 2) { per_fold.clear(); return Metrics(); }
    folds = std::max(2, std::min(folds, std::min(20, ds.n_rows)));
    Dataset raw = to_raw_features(ds);
    std::vector<int> idx(raw.n_rows);
    for (int i = 0; i < raw.n_rows; ++i) idx[i] = i;
    std::mt19937 rng(42);
    std::shuffle(idx.begin(), idx.end(), rng);
    per_fold.resize(folds);
    omp_set_num_threads(std::max(1, threads));

    // Each fold gets its own normalization statistics fitted on that fold's training rows.
    // The validation rows never contribute to the means or standard deviations.
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < folds; ++f) {
        std::vector<int> tr, val;
        for (int k = 0; k < raw.n_rows; ++k) {
            if (k % folds == f) val.push_back(idx[k]);
            else tr.push_back(idx[k]);
        }
        Dataset fold_ds = raw;
        preprocess_from_training_rows(fold_ds, tr);
        std::atomic<bool> stop(false);
        Model m = train(fold_ds, tr, 1, 0.1, 0.001, 100, stop, nullptr, "");
        per_fold[f] = evaluate(m, fold_ds, val);
    }
    Metrics mean_m;
    for (const auto& m : per_fold) {
        mean_m.accuracy += m.accuracy;
        mean_m.precision += m.precision;
        mean_m.recall += m.recall;
        mean_m.f1 += m.f1;
    }
    mean_m.accuracy /= folds;
    mean_m.precision /= folds;
    mean_m.recall /= folds;
    mean_m.f1 /= folds;
    return mean_m;
}

// HPC Concept: Karp-Flatt serial fraction
double karp_flatt(double speedup, int p) {
    if (p <= 1 || speedup <= 0) return 0.0;
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p);
}

std::vector<BenchRow> run_benchmark(const Dataset& seed, int rows, bool& verified, double& max_abs_diff) {
    Dataset raw_seed = to_raw_features(seed);
    Dataset bench_ds = (raw_seed.n_rows == rows) ? raw_seed : generate_synthetic(raw_seed, rows);
    preprocess(bench_ds);
    std::vector<int> all_r(bench_ds.n_rows);
    for (int i = 0; i < bench_ds.n_rows; ++i) all_r[i] = i;
    int max_p = omp_get_num_procs();
    std::vector<int> t_counts;
    for (int t : {1, 2, 4, 8}) if (t <= max_p) t_counts.push_back(t);
    if (std::find(t_counts.begin(), t_counts.end(), max_p) == t_counts.end()) t_counts.push_back(max_p);
    std::sort(t_counts.begin(), t_counts.end());
    t_counts.erase(std::unique(t_counts.begin(), t_counts.end()), t_counts.end());
    std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    std::vector<BenchRow> res;
    std::map<std::string, double> base_predict;
    auto med3 = [](double a, double b, double c) {
        double arr[3] = {a, b, c}; std::sort(arr, arr + 3); return arr[1];
    };
    double t_train_1 = 1.0;
    for (int p : t_counts) {
        omp_set_num_threads(p);
        double gen_runs[3], prep_runs[3], tr_runs[3];
        for (int k = 0; k < 3; ++k) {
            double tg = omp_get_wtime();
            Dataset d_temp = generate_synthetic(raw_seed, rows);
            gen_runs[k] = omp_get_wtime() - tg;
            double tp = omp_get_wtime();
            preprocess(d_temp);
            prep_runs[k] = omp_get_wtime() - tp;
        }
        for (int k = 0; k < 3; ++k) {
            std::atomic<bool> s(false);
            double t0 = omp_get_wtime();
            train(bench_ds, all_r, p, 0.1, 0.001, 100, s, nullptr, "");
            tr_runs[k] = omp_get_wtime() - t0;
        }
        double train_s = med3(tr_runs[0], tr_runs[1], tr_runs[2]);
        if (p == 1) t_train_1 = train_s;
        // HPC Concept: Speedup, efficiency
        double speedup = (p == 1) ? 1.0 : t_train_1 / (train_s > 0 ? train_s : 1e-9);
        double efficiency = speedup / p, kf = karp_flatt(speedup, p);
        Model m; m.w.assign(20, 0.05); m.b = 0.0;
        for (const auto& sched : schedules) {
            double pr[3];
            for (int k = 0; k < 3; ++k) {
                double t0 = omp_get_wtime();
                for (int rep = 0; rep < 5; ++rep) predict_all(m, bench_ds, p, sched);
                pr[k] = (omp_get_wtime() - t0) / 5.0;
            }
            double pm = med3(pr[0], pr[1], pr[2]);
            if (p == 1) base_predict[sched] = pm;
            double ps = (p == 1) ? 1.0 : (base_predict[sched] / std::max(pm, 1e-9));
            res.push_back({p, sched, bench_ds.n_rows, med3(gen_runs[0], gen_runs[1], gen_runs[2]),
                med3(prep_runs[0], prep_runs[1], prep_runs[2]), train_s, pm, speedup, efficiency, kf, ps});
        }
    }
    Model m; m.w.assign(20, 0.05); m.b = 0.0;
    auto p1 = predict_all(m, bench_ds, 1, "static");
    auto pmax = predict_all(m, bench_ds, t_counts.back(), "static");
    max_abs_diff = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::abs(p1[i] - pmax[i]));
    verified = (max_abs_diff <= 1e-9);
    std::ofstream csv("results/benchmark.csv");
    if (csv.is_open()) {
        csv << "threads,schedule,rows,gen_s,preprocess_s,train_s,predict_s,speedup,efficiency,karp_flatt,predict_speedup\n";
        for (const auto& r : res) {
            csv << r.threads << "," << r.schedule << "," << r.rows << "," << r.gen_s << "," << r.preprocess_s << ","
                << r.train_s << "," << r.predict_s << "," << r.speedup << "," << r.efficiency << "," << r.karp_flatt
                << "," << r.predict_speedup << "\n";
        }
    }
    return res;
}

std::string tier_of(double prob) {
    if (prob < 0.50) return "At Risk";
    if (prob < 0.80) return "Average";
    return "Excelling";
}

std::vector<std::string> suggestions_of(double absences, double studytime, double failures, double G2, double higher, double famsup) {
    std::vector<std::string> sug;
    if (absences > 10) sug.push_back("Absences are high. Set a weekly attendance goal.");
    if (studytime <= 2) sug.push_back("Study time is low. Aim for at least 5 hours a week.");
    if (failures > 0) sug.push_back("Past failures present. Arrange tutoring in core subjects.");
    if (G2 < 10) sug.push_back("Recent grade is below passing. Review with the teacher.");
    if (higher == 0) sug.push_back("Low interest in higher education. Discuss goals with a counselor.");
    if (famsup == 0) sug.push_back("Limited family support. Consider a peer study group.");
    if (sug.empty()) sug.push_back("Monitor progress at the next assessment.");
    if (sug.size() > 3) sug.resize(3);
    return sug;
}

std::vector<std::string> suggestions_of(const Dataset& ds, int row) {
    double absences = ds.absences[row], G2 = ds.G2[row];
    double studytime = ds.X[5][row] * ds.stddev[5] + ds.mean[5];
    double failures = ds.X[6][row] * ds.stddev[6] + ds.mean[6];
    double higher = std::round(ds.X[11][row] * ds.stddev[11] + ds.mean[11]);
    double famsup = std::round(ds.X[8][row] * ds.stddev[8] + ds.mean[8]);
    return suggestions_of(absences, studytime, failures, G2, higher, famsup);
}
