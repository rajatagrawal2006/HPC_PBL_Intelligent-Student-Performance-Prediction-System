#include "pipeline.hpp"
#include <httplib.h>
#include <json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <omp.h>

using json = nlohmann::json;

static std::mutex g_mtx;
static Dataset g_dataset;
static Model g_model;
static bool g_model_trained = false;
static std::vector<double> g_predictions;
static std::vector<Metrics> g_cv_per_fold;
static Metrics g_cv_mean;
static double g_cv_s = 0.0;
static std::vector<BenchRow> g_bench_rows;
static bool g_bench_verified = false;
static double g_bench_diff = 0.0;
static std::string g_train_status = "idle";
static int g_train_epoch = 0;
static std::vector<double> g_train_loss;
static double g_train_elapsed = 0.0;
static Metrics g_holdout_metrics;
static double g_holdout_eval_s = 0.0;
static bool g_train_resumed = false;
static std::atomic<bool> g_train_stop{false}, g_busy{false};
static std::thread g_worker_thread;

static void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

static void send_err(httplib::Response& res, const std::string& msg, int code = 400) {
    set_cors(res); res.status = code;
    res.set_content(json{{"error", msg}}.dump(), "application/json");
}

int main() {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("web/index.html");
        if (!f.is_open()) { send_err(res, "web/index.html not found", 404); return; }
        std::stringstream ss; ss << f.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    svr.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mtx); set_cors(res);
        res.set_content(json{{"rows", g_dataset.n_rows}, {"model_trained", g_model_trained},
            {"train_status", g_train_status}, {"max_threads", omp_get_num_procs()}}.dump(), "application/json");
    });

    svr.Post("/api/load", [](const httplib::Request& req, httplib::Response& res) {
        if (g_busy.load()) { send_err(res, "System busy with training or benchmark", 409); return; }
        json j;
        try { j = json::parse(req.body); } catch (...) { send_err(res, "Invalid JSON"); return; }
        std::string src = j.value("source", "uci");
        int req_rows = std::max(1000, std::min(j.value("rows", 50000), 500000));
        Dataset seed = load_uci("data/student-mat.csv");
        double t0 = omp_get_wtime();
        Dataset ds = (src == "synthetic") ? generate_synthetic(seed, req_rows) : seed;
        double load_s = omp_get_wtime() - t0, t1 = omp_get_wtime();
        preprocess(ds);
        double prep_s = omp_get_wtime() - t1;
        std::remove("data/checkpoint.txt");
        int pass_c = 0; for (int y : ds.y) if (y == 1) pass_c++;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_dataset = std::move(ds);
            g_model = Model();
            g_model_trained = false;
            g_train_status = "idle";
            g_train_epoch = 0;
            g_train_loss.clear();
            g_train_elapsed = 0.0;
            g_train_resumed = false;
            g_holdout_metrics = Metrics();
            g_holdout_eval_s = 0.0;
            g_cv_per_fold.clear();
            g_cv_mean = Metrics();
            g_cv_s = 0.0;
            g_predictions.clear();
        }
        set_cors(res);
        res.set_content(json{{"rows", g_dataset.n_rows}, {"load_s", load_s}, {"preprocess_s", prep_s}, {"pass_count", pass_c}, {"source", src}}.dump(), "application/json");
    });

    svr.Post("/api/train", [](const httplib::Request& req, httplib::Response& res) {
        if (g_busy.exchange(true)) { send_err(res, "Another training or benchmark is running", 409); return; }
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_dataset.n_rows == 0) { g_busy.store(false); send_err(res, "No dataset loaded. Please load data first."); return; }
        }
        json j;
        try { j = json::parse(req.body); } catch (...) { g_busy.store(false); send_err(res, "Invalid JSON"); return; }
        int threads = j.value("threads", omp_get_num_procs()), epochs = j.value("epochs", 300);
        double lr = j.value("learning_rate", 0.1), l2 = j.value("l2", 0.001);
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_train_status = "running"; g_train_epoch = 0; g_train_loss.clear();
            g_train_elapsed = 0.0; g_train_resumed = false; g_train_stop.store(false);
            g_holdout_metrics = Metrics(); g_holdout_eval_s = 0.0;
        }
        if (g_worker_thread.joinable()) g_worker_thread.join();
        g_worker_thread = std::thread([threads, epochs, lr, l2]() {
            std::vector<int> tr, te; Dataset ds_copy;
            { std::lock_guard<std::mutex> lock(g_mtx); ds_copy = to_raw_features(g_dataset); }
            split(ds_copy, tr, te);
            // Fit normalization on training rows only, then apply the same transform to all rows.
            preprocess_from_training_rows(ds_copy, tr);
            {
                std::lock_guard<std::mutex> lock(g_mtx);
                g_dataset = ds_copy;
            }
            Model chk; int chk_ep = 0;
            if (load_checkpoint(chk, lr, l2, "data/checkpoint.txt", chk_ep, ds_copy.n_rows) && chk_ep < epochs) {
                std::lock_guard<std::mutex> lock(g_mtx); g_train_resumed = true;
            }
            auto prog = [](int ep, double loss, double elap) {
                std::lock_guard<std::mutex> lock(g_mtx);
                g_train_epoch = ep; g_train_loss.push_back(loss); g_train_elapsed = elap;
            };
            Model m = train(ds_copy, tr, threads, lr, l2, epochs, g_train_stop, prog, "data/checkpoint.txt");
            double eval_t0 = omp_get_wtime();
            Metrics holdout = evaluate(m, ds_copy, te);
            double eval_s = omp_get_wtime() - eval_t0;
            {
                std::lock_guard<std::mutex> lock(g_mtx);
                g_model = std::move(m); g_model_trained = true;
                g_holdout_metrics = holdout; g_holdout_eval_s = eval_s;
                g_train_status = g_train_stop.load() ? "stopped" : "done";
            }
            g_busy.store(false);
        });
        set_cors(res); res.set_content(json{{"started", true}}.dump(), "application/json");
    });

    svr.Get("/api/train/progress", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mtx); set_cors(res);
        int sz = (int)g_train_loss.size(), start = std::max(0, sz - 300);
        std::vector<double> l300(g_train_loss.begin() + start, g_train_loss.end());
        res.set_content(json{{"status", g_train_status}, {"epoch", g_train_epoch}, {"loss", l300},
            {"elapsed_s", g_train_elapsed}, {"resumed", g_train_resumed},
            {"holdout", {{"accuracy", g_holdout_metrics.accuracy}, {"precision", g_holdout_metrics.precision},
                         {"recall", g_holdout_metrics.recall}, {"f1", g_holdout_metrics.f1},
                         {"tp", g_holdout_metrics.tp}, {"fp", g_holdout_metrics.fp},
                         {"tn", g_holdout_metrics.tn}, {"fn", g_holdout_metrics.fn}}},
            {"holdout_eval_s", g_holdout_eval_s}}.dump(), "application/json");
    });

    svr.Post("/api/train/stop", [](const httplib::Request&, httplib::Response& res) {
        g_train_stop.store(true); set_cors(res); res.set_content(json{{"stopped", true}}.dump(), "application/json");
    });

    svr.Post("/api/validate", [](const httplib::Request& req, httplib::Response& res) {
        if (g_busy.exchange(true)) { send_err(res, "Another task is running", 409); return; }
        Dataset ds;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_dataset.n_rows == 0) { g_busy.store(false); send_err(res, "No dataset loaded."); return; }
            ds = g_dataset;
        }
        json j;
        try { j = json::parse(req.body); } catch (...) { g_busy.store(false); send_err(res, "Invalid JSON"); return; }
        int folds = j.value("folds", 5), threads = j.value("threads", omp_get_num_procs());
        if (folds < 2 || folds > 20) { g_busy.store(false); send_err(res, "Folds must be between 2 and 20."); return; }
        if (folds > ds.n_rows) { g_busy.store(false); send_err(res, "Folds cannot exceed dataset row count."); return; }
        std::vector<Metrics> pf; double t0 = omp_get_wtime();
        Metrics mean_m = cross_validate(ds, folds, threads, pf);
        double cv_s = omp_get_wtime() - t0;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_cv_per_fold = pf; g_cv_mean = mean_m; g_cv_s = cv_s;
        }
        g_busy.store(false);
        json pf_json = json::array();
        for (size_t i = 0; i < pf.size(); ++i) {
            pf_json.push_back({{"fold", i + 1}, {"accuracy", pf[i].accuracy}, {"precision", pf[i].precision}, {"recall", pf[i].recall}, {"f1", pf[i].f1}});
        }
        set_cors(res);
        res.set_content(json{{"per_fold", pf_json}, {"mean", {{"accuracy", mean_m.accuracy}, {"precision", mean_m.precision}, {"recall", mean_m.recall}, {"f1", mean_m.f1}}}, {"cv_s", cv_s}}.dump(), "application/json");
    });

    svr.Post("/api/predict", [](const httplib::Request& req, httplib::Response& res) {
        if (g_busy.load()) { send_err(res, "System busy with background task", 409); return; }
        Dataset ds; Model m;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_dataset.n_rows == 0) { send_err(res, "Dataset not loaded."); return; }
            if (!g_model_trained) { send_err(res, "Model not trained."); return; }
            ds = g_dataset; m = g_model;
        }
        json j; try { j = json::parse(req.body); } catch (...) { send_err(res, "Invalid JSON"); return; }
        int threads = j.value("threads", omp_get_num_procs());
        std::string sched = j.value("schedule", "static");
        double t0 = omp_get_wtime();
        std::vector<double> probs = predict_all(m, ds, threads, sched);
        double pred_s = omp_get_wtime() - t0;
        int at_risk = 0, average = 0, excelling = 0, pass_count = 0, fail_count = 0;
        for (double p : probs) {
            if (p >= 0.5) pass_count++; else fail_count++;
            std::string t = tier_of(p);
            if (t == "At Risk") at_risk++; else if (t == "Average") average++; else excelling++;
        }
        { std::lock_guard<std::mutex> lock(g_mtx); g_predictions = std::move(probs); }
        set_cors(res);
        res.set_content(json{{"predict_s", pred_s}, {"total", ds.n_rows}, {"pass_count", pass_count}, {"fail_count", fail_count},
            {"tiers", {{"At Risk", at_risk}, {"Average", average}, {"Excelling", excelling}}}}.dump(), "application/json");
    });

    svr.Get("/api/students", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_predictions.empty()) { send_err(res, "No predictions available. Please run prediction first."); return; }
        std::string tier_filter = req.has_param("tier") ? req.get_param_value("tier") : "";
        int page = req.has_param("page") ? std::max(1, std::stoi(req.get_param_value("page"))) : 1;
        int size = req.has_param("size") ? std::max(1, std::stoi(req.get_param_value("size"))) : 50;
        std::vector<int> fids;
        for (int i = 0; i < (int)g_predictions.size(); ++i) {
            std::string t = tier_of(g_predictions[i]);
            if (tier_filter.empty() || tier_filter == "All" || t == tier_filter) fids.push_back(i);
        }
        int total = (int)fids.size(), start = (page - 1) * size, end = std::min(total, start + size);
        json rows = json::array();
        for (int idx = start; idx < end; ++idx) {
            int id = fids[idx]; double p = g_predictions[id];
            rows.push_back({{"id", id + 1}, {"prob", p}, {"tier", tier_of(p)}, {"G2", g_dataset.G2[id]}, {"absences", g_dataset.absences[id]}, {"suggestions", suggestions_of(g_dataset, id)}});
        }
        set_cors(res); res.set_content(json{{"total", total}, {"rows", rows}}.dump(), "application/json");
    });

    svr.Get("/api/student/sample", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_dataset.n_rows == 0) { send_err(res, "No dataset loaded."); return; }
        int id = req.has_param("id") ? std::stoi(req.get_param_value("id")) - 1 : 0;
        if (id < 0 || id >= g_dataset.n_rows) id = 0;
        json feat = json::object();
        for (int f = 0; f < 20; ++f) {
            double raw = g_dataset.X[f][id] * g_dataset.stddev[f] + g_dataset.mean[f];
            feat[g_dataset.names[f]] = std::round(raw * 1000.0) / 1000.0;
        }
        feat["G2"] = g_dataset.G2[id]; feat["absences"] = g_dataset.absences[id];
        set_cors(res); res.set_content(feat.dump(), "application/json");
    });

    svr.Post("/api/student", [](const httplib::Request& req, httplib::Response& res) {
        Dataset ds; Model m;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_dataset.n_rows == 0) { send_err(res, "Dataset not loaded."); return; }
            if (!g_model_trained) { send_err(res, "Model not trained."); return; }
            ds = g_dataset; m = g_model;
        }
        json j; try { j = json::parse(req.body); } catch (...) { send_err(res, "Invalid JSON"); return; }
        json feat = j.value("features", json::object());
        for (const auto& fn : ds.names) {
            if (!feat.contains(fn) || !feat[fn].is_number()) {
                send_err(res, "Missing or non-numeric feature: " + fn); return;
            }
        }
        double age = feat["age"].get<double>();
        if (age < 10.0 || age > 30.0) { send_err(res, "age must be between 10 and 30"); return; }
        double medu = feat["Medu"].get<double>(), fedu = feat["Fedu"].get<double>();
        if (medu < 0.0 || medu > 4.0 || fedu < 0.0 || fedu > 4.0) { send_err(res, "Medu and Fedu must be between 0 and 4"); return; }
        double trtime = feat["traveltime"].get<double>(), sttime = feat["studytime"].get<double>();
        if (trtime < 1.0 || trtime > 4.0 || sttime < 1.0 || sttime > 4.0) { send_err(res, "traveltime and studytime must be between 1 and 4"); return; }
        double fails = feat["failures"].get<double>();
        if (fails < 0.0 || fails > 4.0) { send_err(res, "failures must be between 0 and 4"); return; }
        for (const auto& b : {"sex","schoolsup","famsup","paid","activities","higher","internet","romantic"}) {
            double v = feat[b].get<double>();
            if (v != 0.0 && v != 1.0) { send_err(res, std::string(b) + " must be 0 or 1"); return; }
        }
        for (const auto& sc : {"famrel","freetime","goout","Dalc","Walc","health"}) {
            double v = feat[sc].get<double>();
            if (v < 1.0 || v > 5.0) { send_err(res, std::string(sc) + " must be between 1 and 5"); return; }
        }
        double abs_val = feat.value("absences", 0.0), g2_val = feat.value("G2", 10.0);
        if (abs_val < 0.0 || abs_val > 100.0) { send_err(res, "absences must be between 0 and 100"); return; }
        if (g2_val < 0.0 || g2_val > 20.0) { send_err(res, "G2 must be between 0 and 20"); return; }

        double dot = 0.0;
        for (int f = 0; f < 20; ++f) {
            double raw = feat[ds.names[f]].get<double>();
            double std_val = (raw - ds.mean[f]) / ds.stddev[f];
            dot += m.w[f] * std_val;
        }
        double p = sigmoid(dot + m.b);
        auto sugs = suggestions_of(abs_val, sttime, fails, g2_val, feat["higher"].get<double>(), feat["famsup"].get<double>());
        set_cors(res);
        res.set_content(json{
            {"prob", p},
            {"predicted_pass", p >= 0.5},
            {"tier", tier_of(p)},
            {"suggestions", sugs}
        }.dump(), "application/json");
    });

    svr.Post("/api/benchmark", [](const httplib::Request& req, httplib::Response& res) {
        if (g_busy.exchange(true)) { send_err(res, "System busy", 409); return; }
        Dataset seed;
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            if (g_dataset.n_rows == 0) { g_busy.store(false); send_err(res, "Dataset not loaded. Please load UCI dataset first."); return; }
            seed = g_dataset;
        }
        json j; try { j = json::parse(req.body); } catch (...) { j = json::object(); }
        int rows = std::max(1000, std::min(j.value("rows", 50000), 500000));
        bool verified = false; double max_diff = 0.0;
        auto results = run_benchmark(seed, rows, verified, max_diff);
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_bench_rows = results; g_bench_verified = verified; g_bench_diff = max_diff;
        }
        g_busy.store(false);
        json rows_json = json::array();
        for (const auto& r : results) {
            rows_json.push_back({{"threads", r.threads}, {"schedule", r.schedule}, {"rows", r.rows},
                {"gen_s", r.gen_s}, {"preprocess_s", r.preprocess_s}, {"train_s", r.train_s}, {"predict_s", r.predict_s},
                {"speedup", r.speedup}, {"efficiency", r.efficiency}, {"karp_flatt", r.karp_flatt},
                {"predict_speedup", r.predict_speedup}});
        }
        set_cors(res);
        res.set_content(json{{"rows", rows_json}, {"verified", verified}, {"max_abs_diff", max_diff}}.dump(), "application/json");
    });

    svr.Get("/api/export/predictions", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_predictions.empty()) { send_err(res, "No predictions to export."); return; }
        std::stringstream ss; ss << "id,prob,tier,G2,absences,suggestions\n";
        for (size_t i = 0; i < g_predictions.size(); ++i) {
            double p = g_predictions[i]; auto sugs = suggestions_of(g_dataset, i);
            std::string sug_str = "";
            for (size_t s = 0; s < sugs.size(); ++s) { if (s > 0) sug_str += "; "; sug_str += sugs[s]; }
            ss << (i + 1) << "," << p << "," << tier_of(p) << "," << g_dataset.G2[i] << "," << g_dataset.absences[i] << ",\"" << sug_str << "\"\n";
        }
        std::string csv_data = ss.str();
        std::ofstream f("results/predictions.csv"); if (f.is_open()) f << csv_data;
        res.set_header("Content-Disposition", "attachment; filename=\"predictions.csv\"");
        res.set_content(csv_data, "text/csv");
    });

    svr.Get("/api/export/benchmark", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("results/benchmark.csv");
        if (!f.is_open()) { send_err(res, "No benchmark data available."); return; }
        std::stringstream ss; ss << f.rdbuf();
        res.set_header("Content-Disposition", "attachment; filename=\"benchmark.csv\"");
        res.set_content(ss.str(), "text/csv");
    });

    std::cout << "SPS Server running on http://127.0.0.1:8080\n";
    svr.listen("127.0.0.1", 8080);
    if (g_worker_thread.joinable()) g_worker_thread.join();
    return 0;
}
