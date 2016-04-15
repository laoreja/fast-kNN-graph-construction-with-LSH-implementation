//
//  lanczos.hpp
//  tryBoost
//
//  Created by 顾秀烨 on 16/4/8.
//  Copyright © 2016年 laoreja. All rights reserved.
//

#ifndef lanczos_h
#define lanczos_h

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/random.hpp>

#include "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers/clapack.h"
//#include "clapack.h"
#include <iostream>
#include <cmath>

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

class Lanczos{
    constexpr static const int MAX_MAT_SIZE = 1000;
    constexpr static const double EPS = 1e-7;
    
public:
    static void lanczos(symmetric_matrix<double, upper>& A, banded_matrix<double>& T, matrix<double>& Q){
        const unsigned int RANDOM_RANGE = 1000;
        
        size_t vecLen = A.size1();
        
        boost::mt19937 rng((unsigned int)time(0));
        boost::uniform_int<> range(1, RANDOM_RANGE);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<>> unRange(rng, range);
        
        
        vector<double> q_i_minus_1(vecLen, 0), q_i(vecLen), q_i_plus_1(vecLen), u(vecLen);
        double beta_i = 0, beta_i_plus_1 = 0, alpha_i;
        
        for (size_t i = 0; i < vecLen; ++i) {
            q_i(i) = unRange();
        }
        q_i = q_i / norm_2(q_i);
        
        
        for (int i = 1; i <= vecLen; ++i) {
            if(i > 1){
                beta_i = beta_i_plus_1;
                q_i_minus_1 = q_i;
                q_i = q_i_plus_1;
            }
            
            
            u = prod(A, q_i) - q_i_minus_1 * beta_i;
            alpha_i = inner_prod(u, q_i);
            u = u - q_i * alpha_i;
            beta_i_plus_1 = norm_2(u);
            q_i_plus_1 = u / beta_i_plus_1;
            
            T(i-1, i-1) = alpha_i;
            if (i < vecLen) {
                T(i-1, i) = T(i, i-1) = beta_i_plus_1;
            }
            column(Q, i-1) = q_i;
        }
        
    }
    
    static int solveSymmetricTridiagonal(banded_matrix<double>& A, double* eigenvalues, matrix<double>& eigenvectors){
        
        char JOBZ = 'V';
        __CLPK_integer N = (__CLPK_integer)A.size1();
        double E[MAX_MAT_SIZE-1];
        __CLPK_integer LDZ = N;
        double WORK[2*MAX_MAT_SIZE-2];
        double Z[MAX_MAT_SIZE * MAX_MAT_SIZE];
        
        for (int i = 0; i < N; ++i) {
            eigenvalues[i] = A(i, i);
            if (i < N-1) {
                E[i] = A(i, i+1);
            }
        }
        
        int INFO;
        dstev_(&JOBZ, &N, eigenvalues, E, Z, &LDZ, WORK, &INFO);
        if (INFO == 0) {
            for (int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++){
                    eigenvectors(i, j) = Z[j * N + i];
                }
            }
        }
        return INFO;
        /*
         *  INFO    (output) INTEGER
         *          = 0:  successful exit
         *          < 0:  if INFO = -i, the i-th argument had an illegal value
         *          > 0:  if INFO = i, the algorithm failed to converge; i
         *                off-diagonal elements of E did not converge to zero.
         */
    }
    
    static int symEigenRaw(symmetric_matrix<double, upper>& A, vector<double>& eigenvalues, matrix<double>& eigenvectors){
        
        char JOBZ = 'V';
        char UPLO = 'U';
        __CLPK_integer N = (__CLPK_integer)A.size1();
        __CLPK_integer LDA = N;
        
        double AA[LDA * N];
        double W[N];
        __CLPK_integer LWORK = 1 + 6*N + 2*N*N;
        double WORK[LWORK];
        __CLPK_integer LIWORK = 3 + 5*N;
        __CLPK_integer IWORK[LIWORK];
        __CLPK_integer INFO;
        
        
        for (int i = 0; i < N; ++i) {
            for(int j = i; j < N; ++j){
                AA[j * N + i] = A(i, j);
            }
        }
        
        
        int dim = 0;
        eigenvalues.clear();
        
        dsyevd_(&JOBZ, &UPLO, &N, AA, &LDA, W, WORK, &LWORK, IWORK, &LIWORK, &INFO);
        if (INFO == 0) {
            for (int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++){
                    eigenvectors(i, j) = AA[j * N + i];
                }
                eigenvalues(i) = W[i];
                if(std::abs(W[i]) > 1e-7){
                    dim++;
                }
            }
        }
        return dim;
        //    INFO is INTEGER
        //    = 0:  successful exit
        //    < 0:  if INFO = -i, the i-th argument had an illegal value
        //        > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed
        //            to converge; i off-diagonal elements of an intermediate
        //    tridiagonal form did not converge to zero;
        //    if INFO = i and JOBZ = 'V', then the algorithm failed
        //        to compute an eigenvalue while working on the submatrix
        //            lying in rows and columns INFO/(N+1) through
        //            mod(INFO,N+1).
    }
    
    static int symmetricMatEigens(symmetric_matrix<double, upper>& A, vector<double>& eigenvalues, matrix<double>& eigenvectors){
        
        size_t N = A.size1();
        banded_matrix<double> T(N, N, 1, 1);
        matrix<double> Q(N, N);
        
        lanczos(A, T, Q);
        
        matrix<double> W(N, N);
        double D[MAX_MAT_SIZE];
        int INFO = solveSymmetricTridiagonal(T, D, W);
        matrix<double> wholeEigenVec;
        if (INFO == 0) {
            wholeEigenVec = prod(Q, W);
        }
        
        int idx = 0;
        eigenvalues[idx] = D[0];
        column(eigenvectors, idx++) = column(wholeEigenVec, 0);
        for(int i = 1; i < N; i++){
            if( std::abs(D[i] - D[i-1]) < EPS){
                
            }else{
                eigenvalues[idx] = D[i];
                column(eigenvectors, idx++) = column(wholeEigenVec, i);
            }
        }
        
        //    cout << "Symmetrix tridiagonal matrix: " << endl << T << endl << endl;
        //    for (int i= 0; i< N; i++ ) {
        //        cout << "eigenvalue: " << eigenvalues[i] << endl;
        //    }
        //    cout << "Symmetrix tridiagonal eigenvectors: " << W << endl;
        
        return idx--;
    }
};




#endif /* lanczos_h */
