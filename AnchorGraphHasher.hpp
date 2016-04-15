//
//  AnchorGraphHasher.hpp
//  tryBoost
//
//  Created by 顾秀烨 on 16/4/12.
//  Copyright © 2016年 laoreja. All rights reserved.
//

#ifndef AnchorGraphHasher_hpp
#define AnchorGraphHasher_hpp

#include "lanczos.hpp"
#include <limits>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>


using namespace boost::numeric::ublas;
using std::cin;
using std::cout;
using std::endl;
using std::string;

class AnchorGraphHasher{
private:
    matrix<double> W;
    matrix<double> anchors;
    int nnanchors;
    double sigma;
    // if (sigma < 0) then sigma is None
    
public:
    AnchorGraphHasher(){nnanchors = -1; sigma = -1.0;};
    
    AnchorGraphHasher(matrix<double>& W_in, matrix<double>& anchors_in, int nnanchors_in, double sigma_in){
        this->W.resize(W_in.size1(), W_in.size2());
        this->W = W_in;
        this->anchors.resize(anchors_in.size1(), anchors_in.size2());
        this->anchors = anchors_in;
        this->nnanchors = nnanchors_in;
        this->sigma = sigma_in;
    }
    
    static AnchorGraphHasher train(matrix<double>& traindata, matrix<double>& anchors_in,
                                   matrix<int>& Y_out,
                                   int numhashbits = 12, int nnanchors_in = 2, double sigma_in = -1.0){
        
        int n = (int) traindata.size1();
        int m = (int)anchors_in.size1();
        
        if(numhashbits >= m){
            cout << "The number of hash bits (" << numhashbits << ") must\
            be less than the number of anchors (" << m << ")." << endl;
            return AnchorGraphHasher();
        }
        
        compressed_matrix<double> Z(n, m);
        double sigma_out = _Z(traindata, anchors_in, nnanchors_in, sigma_in, Z);
//        cout << "sigma: " << sigma_out << endl;
//        cout << "Z" << endl;
        
        
        matrix<double> W_out(m, numhashbits);
        _W(Z, numhashbits, W_out);
//        cout << "W" << endl;
        
        matrix<int> Y(n, numhashbits);
        _hash(Z, W_out, Y);
//        cout << "(matrix<int>) Y" << endl;
//        cout << row(Y, 100) << endl;
        
        AnchorGraphHasher agh(W_out, anchors_in, nnanchors_in, sigma_out);
        Y_out = Y;
        
        return agh;
    }
    
    void hash(matrix<double>& data, matrix<int>& Y_out){
        compressed_matrix<double> Z(data.size1(), anchors.size1());
        _Z(data, this->anchors, this->nnanchors, this->sigma, Z);
        _hash(Z, this->W, Y_out);;
    }
    
    static void _hash(compressed_matrix<double>& Z, matrix<double>& W, matrix<int>& Y_out){
        matrix<double> Y_real(Y_out.size1(), Y_out.size2());
        
        axpy_prod(Z, W, Y_real, true);
        
        for (int i = 0; i < Y_real.size1(); i++) {
            for (int j = 0; j < Y_real.size2(); j++) {
                Y_out(i, j) = (Y_real(i, j) > 0);
            }
        }
    }
    
    static void _W(compressed_matrix<double>& Z_in, int numhashbits, matrix<double>& W_out){
        int m = (int)Z_in.size2();
        int n = (int)Z_in.size1();
        
        scalar_vector<double> ones(n);
        vector<double> ZT1(m);
        axpy_prod(trans(Z_in), ones, ZT1, true);
        
        for (int i = 0; i < ZT1.size(); i++) {
            ZT1(i) = 1.0 / sqrt(ZT1(i));
        }
        diagonal_matrix<double> lambda_exp(m, ZT1.data());
        matrix<double> V(m, m);
        vector<double> SIG(m);
        
        matrix<double> M(m, m);
        
        {
            matrix<double> tmp1(m, n);
            axpy_prod(lambda_exp, trans(Z_in), tmp1, true);
            
            matrix<double> tmp2(m, m);
            axpy_prod(tmp1, Z_in, tmp2, true);
            
            axpy_prod(tmp2, lambda_exp, M, true);
        }
        //        noalias(M) = prod( matrix<double>(prod( matrix<double>(prod(lambda_exp, trans(Z_in))) , Z_in)), lambda_exp);
        symmetric_matrix<double, upper> A(M);
        int dimension = Lanczos::symEigenRaw(A, SIG, V);
        
        if (SIG(dimension-1) > 0.99999999) {
            SIG.resize(dimension-1);
            V.resize(V.size1(), dimension-1);
            dimension--;
        }
    
        if (SIG.size() < numhashbits) {
            cout << "the dimension of the eigen vectors is smaller than numhashbits, calculation failed" << endl;
            return;
        }
        
        
        for (int i = 0; i < numhashbits && i < (dimension/2); i++) {
            column(V, i).swap(column(V, dimension - 1 - i));
            double tmp = SIG(i);
            SIG(i) = SIG(dimension - 1 - i);
            SIG(dimension - 1 - i) = tmp;
        }
        
        SIG.resize(numhashbits);
        V.resize(m, numhashbits);
        
//        cout << SIG << endl;
        //        cout <<V << endl;
        
        for (int i = 0; i < SIG.size(); i++) {
            SIG(i) = 1.0 / sqrt(SIG(i));
        }
        diagonal_matrix<double> SIG_exp(numhashbits, SIG.data());
        
        matrix<double> tmp_lambda_exp_V(m, numhashbits);
        axpy_prod(lambda_exp, V, tmp_lambda_exp_V, true);
        axpy_prod(tmp_lambda_exp_V, SIG_exp, W_out, true);
        //        noalias(W_out) = prod( matrix<double>(prod(lambda_exp, V)), SIG_exp);
    };
    
    //return sigma
    
    static double _Z(matrix<double>& data, matrix<double>& anchors_in, int nnanchors_in, double sigma_in, compressed_matrix<double>& Z_out){
        
        int n = (int)data.size1();
        int m = (int)anchors_in.size1();
        
        matrix<double> distances(n, m);
        pdist2(data, anchors_in, std::string("sqeuclidean"), distances);
//        cout << "distances" << endl;
        
        matrix<double> val(n, nnanchors_in);
        matrix<int> pos(n, nnanchors_in);
        
        sigma_in = 0;
        for (int i = 0; i < n; i++) {
            
            int cnt = nnanchors_in;
            double minVal = 0.0;
            while (cnt--) {
                minVal = distances(i, 0);
                int minPos = 0;
                for (int j = 1; j < m; j++) {
                    if (distances(i, j) < minVal) {
                        minVal = distances(i, j);
                        minPos = j;
                    }
                }
                pos(i, nnanchors_in - cnt - 1) = minPos;
                val(i, nnanchors_in - cnt - 1) = minVal;
                distances(i, minPos) = std::numeric_limits<double>::max();
            }
            sigma_in += sqrt(minVal);
            
        }
        
        sigma_in /= (n * sqrt(2.0));
        double c = 2 * pow(sigma_in, 2); //bandwidth parameter
        matrix<double> exponent = val / (-1.0 * c);
        vector<double> shift_column = column(exponent, nnanchors_in-1);
        matrix<double> shift(n, nnanchors_in);
        for (int j = 0; j < nnanchors_in; j++) {
            column(shift, j) = shift_column;
        }
        
        matrix<double> expMinusShift = exponent - shift;
        
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nnanchors_in; j++) {
                expMinusShift(i, j) = exp(expMinusShift(i, j));
            }
        }
        
        
        vector<double> denom(n);
        axpy_prod(expMinusShift, scalar_vector<double>(nnanchors_in), denom, true);
        //        vector<double> denom = prod(expMinusShift, scalar_vector<double>(nnanchors_in));
        
        for (int i = 0; i < n; i++) {
            denom(i) = log(denom(i));
        }
        
        denom += shift_column;
        for (int j = 0; j < nnanchors_in; j++) {
            column(val, j) = column(exponent, j) - denom;
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nnanchors_in; j++) {
                val(i, j) = exp(val(i, j));
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nnanchors_in; j++) {
                Z_out(i, pos(i, j)) = val(i, j);
            }
        }
        
        
        return sigma_in;
    }
    
    template<typename DataType, typename ResType>
    static void pdist2(matrix<DataType>& X, matrix<DataType>& Y, const std::string& metric, matrix<ResType>& distances){
        
        if (metric == "sqeuclidean") {
            int nx = (int)X.size1();
            int ny = (int)Y.size1();
            matrix<double> XX(nx, ny);
            vector<double> sum_X(nx);
            
            
            for (int i = 0; i < nx; i++) {
                double product = inner_prod(row(X, i), row(X, i));
                sum_X(i) = product;
            }
            
            for (int i = 0; i < ny; i++) {
                column(XX, i) = sum_X;
            }
            matrix<double> YY(nx, ny);
            vector<double> sum_Y(ny);
            for (int i = 0; i < ny; i++) {
                double product = inner_prod(row(Y, i), row(Y, i));
                sum_Y(i) = product;
            }
            for (int i = 0; i < nx; i++) {
                row(YY, i) = sum_Y;
            }
            matrix<double> XY(nx, ny);
            
            axpy_prod(X, trans(Y), XY, true);
            
            distances = XX + YY - 2*XY;
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    if (distances(i, j) < 0) {
                        distances(i, j) = 0;
                    }
                }
            }
            
        }else if(metric == "hamming"){
            int hashbits = (int)X.size2();
            int nx = (int)X.size1();
            int ny = (int)Y.size1();
            
            matrix<int> Xint = (2 * (matrix<int>)X) - scalar_matrix<int>(nx, hashbits);
            matrix<int> Yint = (2 * (matrix<int>)Y) - scalar_matrix<int>(ny, hashbits);
            matrix<int> XY(nx, ny);
            axpy_prod(Xint, trans(Yint), XY, true);
            distances = scalar_matrix<int>(nx, ny, hashbits) -
            ((scalar_matrix<int>(nx, ny, hashbits) + XY)/2);
            
        }else{
            cout << "Unsupported Metric: " << metric << endl;
            return;
        }
    };
    
    static double test(matrix<int>& trainY, matrix<int>& testY, matrix<int>&traingnd, matrix<int>& testgnd, int radius = 2){
        
        int ntrain = (int)trainY.size1();
        int ntest = (int)testY.size1();
        
        
        vector<int> testgndRavel(ntest, testgnd.data());
        vector<int> traingndRavel(ntrain, traingnd.data());
        
        matrix<int> hamdis(ntrain, ntest);
        pdist2(trainY, testY, "hamming", hamdis);
        
        vector<double> precision(ntest);
        
        for (int j = 0; j < ntest; j++) {
            vector<int> ham = column(hamdis, j);
            std::vector<int> lst;
            for (int i = 0; i < ntrain; i++) {
                if (ham(i) <= radius) {
                    lst.push_back(i);
                }
            }
            int ln = (int)lst.size();
            if (ln == 0) {
                precision(j) = 0;
            }else{
                precision(j) = 0;
                for (int i = 0; i < ln; i++) {
                    if (traingndRavel(lst[i]) == testgndRavel(j)) {
                        precision(j) += 1.0;
                    }
                }
                precision(j) /= double(ln);
            }
        }
        return sum(precision)/ntest;
    }
    
};


#endif /* AnchorGraphHasher_hpp */
