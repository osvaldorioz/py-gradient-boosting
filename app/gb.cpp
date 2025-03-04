#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class GradientBoosting {
private:
    std::vector<VectorXd> models;
    double learning_rate;
    int n_estimators;
    
public:
    GradientBoosting(double lr, int n_est) : learning_rate(lr), n_estimators(n_est) {}
    
    void fit(py::array_t<double> X_np, py::array_t<double> y_np) {
        auto X = X_np.unchecked<2>();
        auto y = y_np.unchecked<1>();
        
        int n_samples = X.shape(0);
        int n_features = X.shape(1);
        
        MatrixXd X_eigen(n_samples, n_features);
        VectorXd y_eigen(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            y_eigen(i) = y(i);
            for (int j = 0; j < n_features; ++j) {
                X_eigen(i, j) = X(i, j);
            }
        }
        
        VectorXd residuals = y_eigen;
        for (int i = 0; i < n_estimators; ++i) {
            VectorXd model = X_eigen.colwise().mean();
            models.push_back(model);
            residuals -= learning_rate * model;
        }
    }
    
    py::array_t<double> predict(py::array_t<double> X_np) {
        auto X = X_np.unchecked<2>();
        int n_samples = X.shape(0);
        VectorXd predictions = VectorXd::Zero(n_samples);
        
        for (const auto& model : models) {
            predictions += learning_rate * model;
        }
        
        py::array_t<double> result(n_samples);
        auto r = result.mutable_unchecked<1>();
        for (int i = 0; i < n_samples; ++i) {
            r(i) = predictions(i);
        }
        
        return result;
    }
};

PYBIND11_MODULE(gb_module, m) {
    py::class_<GradientBoosting>(m, "GradientBoosting")
        .def(py::init<double, int>())
        .def("fit", &GradientBoosting::fit)
        .def("predict", &GradientBoosting::predict);
}
