//
//  main.cpp
//  tryBoost
//
//  Created by 顾秀烨 on 16/4/8.
//  Copyright © 2016年 laoreja. All rights reserved.
//

#define BOOST_UBLAS_NDEBUG
#define NDEBUG

#include <iostream>
#include <iomanip>
#include <fstream>


#include "lanczos.hpp"
#include "AnchorGraphHasher.hpp"

//headers for fastkNN
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>

typedef std::pair<int, int> dataPair;
typedef std::pair<double, int> comparePair;

using namespace boost::numeric::ublas;
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::ifstream;

template <typename DataType, typename Size>
void loadMatrix(ifstream& fin, matrix<DataType>& m, Size s){
    
    for (int i = 0; i < m.size1() ; i++) {
        for (int j = 0; j < m.size2(); j++) {
            Size tmp;
            fin.read((char*)&tmp, sizeof(Size));
            m(i, j) = (DataType)tmp;
            
        }
    }
}

void testAghasher();
void testAghSmallData();
void testLanczos();

void load_data(char* filename, float*& data, size_t& num,int& dim){// load data with sift10K pattern
    ifstream in(filename, std::ios::binary);
    if(!in.is_open()){cout<<"open file error"<<endl;exit(-1);}
    in.read((char*)&dim,4);
    cout<<"data dimension: "<<dim<<endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = fsize / (dim+1) / 4;
    data = new float[num*dim];
    
    in.seekg(0, std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}


double distance_compare(vector<double> v1, vector<double> v2){
    vector<double> minus_res = v1 - v2;
    return inner_prod(minus_res, minus_res);
}



class fastKNNIndex{
public:
    matrix<double> dataset;
    int k;
    int l;
    int m;
    int block_sz = 100;
    int data_sz;
    int dim;
    int nanchors = 20;
    const double trainRaio = 1;
    std::vector<std::vector<int>> knn_table_gt;
    std::vector<std::vector<std::vector<int>>> single_knn_table;

    
    fastKNNIndex(){block_sz = 100;};
    fastKNNIndex(matrix<double>& dataset_, int k_, int l_, int block_sz_ = 100, int m_ = 12)
    :dataset(dataset_), k(k_), l(l_), m(m_){
        data_sz = (int) dataset.size1();
        dim = (int) dataset.size2();
        m = ceil( log(data_sz / block_sz) / log(2.0) ) + 1;
        knn_table_gt.resize(data_sz, std::vector<int>(k));
        single_knn_table.resize(l, std::vector<std::vector<int>>(data_sz, std::vector<int>(k)));
    }
    
    void brute_force_kNN(std::vector<dataPair>::iterator begin, std::vector<dataPair>::iterator end, int iteration){
        
        for (std::vector<dataPair>::iterator i = begin; i != end; ++i) {
            std::vector<comparePair> result;
            for (std::vector<dataPair>::iterator j = begin; j != end; ++j) {
                
                if (i != j) {
                    result.push_back(std::make_pair(distance_compare(row(dataset, i->second), row(dataset, j->second)), j->second));
                }
                
            }
            
            std::partial_sort(result.begin(), result.begin() + k, result.end());
            result.resize(k);
            for (int j = 0; j < k; j++) {
                single_knn_table[iteration][i->second][j] = result[j].second;
            }
            
        }
    }
    
    void basic_ann_by_lsh(int iteration){
        
        srand((int)time(0));
        matrix<double> anchors(nanchors, dim);
        for (int i = 0; i < nanchors; i++) {
            row(anchors, i) = row(dataset, rand() % data_sz);
        }
        
        int train_sz = data_sz * trainRaio;
        matrix<double> traindata = project(dataset, range(0, train_sz), range(0, dim));
        matrix<int> trainY(train_sz, m);
        AnchorGraphHasher agh = AnchorGraphHasher::train(traindata, anchors, trainY, m);
        matrix<int> remainY(data_sz-train_sz, m);
        matrix<double> remaindata = project(dataset, range(train_sz, data_sz), range(0, dim));
        agh.hash(remaindata, remainY);
        
        matrix<int> Y(data_sz, m);
        project(Y, range(0, train_sz), range(0, dim)) = trainY;
        project(Y, range(train_sz, data_sz), range(0, dim)) = remainY;
        
        vector<int> w(m);
        for (int i = 0; i < m; i++) {
            w(i) = (rand() % 1000);
        }
        
        vector<int> p(data_sz);
        axpy_prod(Y, w, p, true);
        std::vector<dataPair> data;
        for (int i = 0; i < data_sz; i++) {
            data.push_back(std::make_pair(p(i), i));
        }
        std::sort(data.begin(), data.end());
        
        for (int i = 0; i < data_sz / block_sz; i++) {
            brute_force_kNN(data.begin() + i * block_sz, data.begin() + (i+1) * block_sz, iteration);
        }
    }
    
    void kNNGraphConstruction(){
        
        for (int i = 0; i < l; i++) {
//            single_knn_table.assign(data_sz, std::vector<int>(k));
            basic_ann_by_lsh(i);
        }
        
        for (int i = 0; i < data_sz; i++) {
            
            std::vector<std::pair<int, int>> idx(l);
            for (int tableId = 0; tableId < l; tableId++) {
                idx[tableId] = std::make_pair(tableId, 0);
            }
            
            for (int cnt = 0; cnt < k; cnt++) {

                int minIdx = 0;
                int minId = single_knn_table[idx[0].first][i][idx[0].second];
                double minDis = distance_compare(row(dataset, i), row(dataset, minId));

                for (int j = 1; j < idx.size(); j++) {
                    int tmpId = single_knn_table[idx[j].first][i][idx[j].second];
                    double tmpDis = distance_compare(row(dataset, i), row(dataset, tmpId));
                    if (tmpDis < minDis) {
                        minIdx = j;
                        minId = tmpId;
                        minDis = tmpDis;
                    }
                }
                
                knn_table_gt[i][cnt] = minId;
                if (idx[minIdx].second == k-1) {
                    idx.erase(idx.begin() + minIdx);
                }else{
                    idx[minIdx].second++;
                }
        
            }
            
        }
        single_knn_table.clear();
        
        for (int i = 0; i < data_sz; i++) {
            std::vector<dataPair> nn;
            for (int j = 0; j < k; j++) {
                nn.push_back(std::make_pair(distance_compare(row(dataset, i), row(dataset, knn_table_gt[i][j])), knn_table_gt[i][j]));
                for (int ii = 0; ii < k; ii++) {
                    nn.push_back(std::make_pair(distance_compare(row(dataset, i), row(dataset, knn_table_gt[j][ii])), knn_table_gt[j][ii]));
                }
            }
            std::partial_sort(nn.begin(), nn.begin()+k, nn.end());
            nn.resize(k);
            knn_table_gt[i].clear();
            for (int jj = 0; jj < k; jj++) {
                knn_table_gt[i].push_back(nn[jj].second);
            }
        }
    }
    
    void saveIndex(const char* filename){
        std::ofstream out(filename,std::ios::binary);
        std::vector<std::vector<int>>::iterator i;
        for(i = knn_table_gt.begin(); i!= knn_table_gt.end(); i++){
            std::vector<int>::iterator j;
            int tmpdim = 10;
            out.write((char*)&tmpdim, sizeof(int));
            for(j = i->begin(); j != i->end(); j++){
                int id = *j;
                out.write((char*)&id, sizeof(int));
            }
        }
        out.close();
    }
    
    
};




int main( )
{
    string dataFile = "/Users/laoreja/study/MachineLearning/fgraph/siftsmall/siftsmall_base.fvecs", queryFile = "/Users/laoreja/study/MachineLearning/fgraph/siftsmall/siftsmall_query.fvecs";
    float* data_load = NULL;
    float* query_load = NULL;
    size_t points_num, q_num;
    int dim, qdim;
    
    load_data((char* )dataFile.c_str(), data_load, points_num, dim);
    load_data((char* )queryFile.c_str(), query_load, q_num, qdim);
    assert(dim == qdim);
    
    matrix<float> dataset(points_num, dim);
    matrix<float> query(q_num, qdim);
    
    std::copy(data_load, data_load + points_num * dim, dataset.data().begin());
    std::copy(query_load, query_load + q_num * qdim, query.data().begin());
    
    matrix<double> data_d = (matrix<double>)dataset;
    matrix<double> query_d = (matrix<double>)query;
    
    
    fastKNNIndex fknn(data_d, 10, 15, 100);
    clock_t s, f;
    s = clock();
    fknn.kNNGraphConstruction();
    f = clock();
    cout << "Index building time : " << (f-s)*1.0/CLOCKS_PER_SEC << " seconds" << endl;
    fknn.saveIndex(string("/Users/laoreja/study/MachineLearning/fgraph/fknn_res").c_str());
    
    
    
//    testAghasher();
//    testAghSmallData();
//    testLanczos();
    
    return 0;
}

void testLanczos(){
    const int SIZE = 100;
    freopen("/Users/laoreja/study/MachineLearning/learnBoost/lanczos_test/A.txt", "r", stdin);
    double noUse;
    
    symmetric_matrix<double, upper> myCorrMat(SIZE);
    for (int i = 0; i < SIZE; i++) {
        for(int j = 0; j < i; j++){
            cin >> noUse;
        }
        for(int j = i; j < SIZE; j++){
            cin >> myCorrMat(i, j);
        }
    }
    
    
    vector<double> eigenvalues(SIZE);
    matrix<double> eigenvectors(SIZE, SIZE);
    int eigNum = Lanczos::symEigenRaw(myCorrMat, eigenvalues, eigenvectors);
    
    
    cout << "Got " << eigNum << " eigen values." << endl;
    for (int i= 0; i< eigenvalues.size(); i++ ) {
        cout << "eigenvalue: " << std::setprecision(10) << eigenvalues(i) << endl;
    }
    cout << "eigenvectors: " << endl << eigenvectors << endl;
    
    //    for(int i = 0; i < 20; i++){
    //        cout << std::setprecision(10) << column(eigenvectors, SIZE-1-i) << endl;
    //    }
}

void testAghasher(){
    const int dim = 784;
    const int ntrain = 69000;
    const int ntest = 1000;
    const int nanchors = 300;
    
    std::ifstream fin("/Users/laoreja/Downloads/PyAnchorGraphHasher-master/AllDataBin", std::ios::binary);
    
    matrix<double> traindata(ntrain, dim);
    matrix<double> testdata(ntest, dim);
    matrix<int> traingnd(ntrain, 1);
    matrix<int> testgnd(ntest, 1);
    matrix<double> anchors(nanchors, dim);
    
    uint8_t ui8;
    double d;
    loadMatrix(fin, traindata, ui8);
    loadMatrix(fin, testdata, ui8);
    loadMatrix(fin, traingnd, ui8);
    loadMatrix(fin, testgnd, ui8);
    loadMatrix(fin, anchors, d);
    
    fin.close();
    
    int precisionradius = 2;
    double sigma = -1.0;
    int nnanchors = 2;
    
    int numbitsArray[6] {12, 16, 24, 32, 48, 64};
    
    for (int numbitsIdx = 0; numbitsIdx < 6; numbitsIdx++) {
        int numbits = numbitsArray[numbitsIdx];
        
        matrix<int> trainY(ntrain, numbits);
        
        AnchorGraphHasher agh = AnchorGraphHasher::train(traindata, anchors, trainY, numbits, nnanchors, sigma);
        
        matrix<int> testY(ntest, numbits);
        agh.hash(testdata, testY);
        double precision = AnchorGraphHasher::test(trainY, testY, traingnd, testgnd, precisionradius);
        
        cout << "1-AGH: the Hamming radius " << precisionradius << " precision for " << numbits << " bits is " << precision << "." << endl;
        
    }
}

void testAghSmallData(){
    const int dim = 4;
    const int ntrain = 5;
    const int ntest = 3;
    const int nanchors = 3;
    
    
    matrix<double> traindata(ntrain, dim);
    matrix<double> testdata(ntest, dim);
    matrix<int> traingnd(ntrain, 1);
    matrix<int> testgnd(ntest, 1);
    matrix<double> anchors(nanchors, dim);
    
    for (int i = 0; i < ntrain; i++) {
        for (int j = 0; j < dim; j++) {
            traindata(i, j) = i * dim + j + 30;
        }
    }
    
    for (int i = 0; i < ntest; i++) {
        for (int j = 0; j < dim; j++) {
            testdata(i, j) = i * dim + j + 20;
        }
    }
    
    cout << traindata << endl;
    cout << testdata << endl;
    
    
    anchors = (matrix<double>)project(traindata, slice(0, 2, nanchors), slice(0, 1, dim));
    double sigma = -1.0;
    int nnanchors = 2;
    int numbits = 2;
    matrix<int> trainY(ntrain, numbits);
    
    AnchorGraphHasher agh = AnchorGraphHasher::train(traindata, anchors, trainY, numbits, nnanchors, sigma);
    
    matrix<int> testY(ntest, numbits);
    agh.hash(testdata, testY);
    
    for (int i = 0; i < traindata.size1(); i++) {
        traingnd(i, 0) = i/2;
    }
    
    testgnd(0,0) = 0;
    testgnd(1,0) = 0;
    testgnd(2,0) = 1;
    
    double precision = AnchorGraphHasher::test(trainY, testY, traingnd, testgnd, 2);
    
    cout << "1-AGH: the Hamming radius 2 precision for " << numbits << " bits is " << precision << "." << endl;
}

