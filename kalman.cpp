#include <iostream>
#include <utility>

#ifdef __linux__
#include <Eigen/Dense>
#elif _WIN32
#include "./Eigen/Dense"    // Needs the eigen library headers in the project directory
#endif

using namespace std;
using Eigen::MatrixXd;

#define Matrix Eigen::MatrixXd

class KalmanFilter {
public:
    
    //Matrix n x n that describes how the state evolves from t-1 to t without controls or noise
    Matrix A;
    
    //Matrix n x l that describes how the control changes the state from t-1 to t
    Matrix B;

    //Matrix k x n that describes how to map the state x_t to an observation z_t.
    Matrix C;

    // Measurement Noise
    Matrix Q;

    // Motion noise
    Matrix R;

    KalmanFilter(Matrix A, Matrix B, Matrix C, Matrix R, Matrix Q) {
        
        /**
         * Constructor
         */
        
        this->A = A;
        this->B = B;
        this->C = C;
        this->R = R;
        this->Q = Q;        
    }

    pair<Matrix, Matrix> iterate(Matrix mu, Matrix cov, Matrix u, Matrix z) {

        /**
         * This function represents one iteration of the Kalman filter algorithm.
         * It takes in the current state estimate, its covariance, the control input,
         * and the measurement, and returns the updated state and covariance.
         * 
         * @param mu - state matrix at time t - 1
         * @param cov - covariance matrix at time t - 1
         * @param u - control matrix at time t
         * @param z - observation matrix at time t
         * 
         * @return pair(mu, cov) - state matrix at time t and covariance matrix at time t
         */
    
        Matrix mu_bel = A * mu + B * u;
        Matrix cov_bel = A * cov * A.transpose() + R; 

        Matrix K = cov_bel * C.transpose() * (C * cov_bel * C.transpose() + Q).inverse();
        mu = mu_bel + K * (z - C * mu_bel);
        
        int n = A.rows();
        Matrix I = Matrix::Identity(n, n);

        cov = (I - K * C) * cov_bel;

        return make_pair(mu, cov);
    }
};
