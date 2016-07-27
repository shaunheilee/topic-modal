#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <map>
#include <cstdlib>
#include <math.h>

using namespace std;
typedef tuple<int, int, float> Token;
typedef vector<Token>   Tokens;
typedef tuple<float, float> BetaParam;
typedef vector<BetaParam>   BetaParams;
typedef vector<float>       tSamples;
typedef boost::unordered_map<string, int> umap;
typedef boost::unordered_map<int, string> imap;
using namespace Eigen;

#define OUT_BUF_SIZE    1024

class LDA{
    public:
        imap    m1;
        imap    m2;
        Tokens  tks;
        float   alpha, beta;
        int     K, niter, V;
        ArrayXXf dt;
        ArrayXXf tw;
        ArrayXf tpw;
        ArrayXi topics;
        BetaParams bps;
        vector<tSamples>    ts;

        LDA(float a, float b, int k, int n){
            alpha = a;
            beta = b;
            K = k;
            niter = n;
        }

        ~LDA(){
        }

        void update_beta_params(int cflag){
            // cal sample mean for each topic
            // cal biased sampe varience for each topic
            // using method of moments to estimate beta params
            float mean, var, * buf, tmp;
            int sz;
            for(int i = 0 ; i < ts.size(); i++){
                buf = ts[i].data();
                sz = ts[i].size();
                Map<ArrayXf> v(buf, ts[i].size());
                mean = v.mean();
                auto x = v - mean;
                var = (x * x).sum() / sz; // biased varience
                // using boost estimator
                tmp = mean * (1 - mean) / var;
                get<0>(bps[i]) = mean * (tmp - 1);
                get<1>(bps[i]) = (1 - mean) * (tmp - 1);
                if(cflag == 0)
                    ts[i].clear();
            }
        }

        void output(){
            ofstream doc_topic;
            ofstream word_topic, topic_time;
            doc_topic.open("doc_topic", ios::out);
            word_topic.open("word_topic", ios::out);
            topic_time.open("topic_time", ios::out);
            char buf[OUT_BUF_SIZE];
            int k;
            for(int i = 0 ; i < m1.size(); i++){
                snprintf(buf, OUT_BUF_SIZE, "%s", m1[i].c_str());
                k = strlen(buf);
                for(int j = 0; j < K; j++){
                    int c = dt(j, i);
                    sprintf(buf + k, "\t%d", c);
                    k = strlen(buf);
                }
                snprintf(buf + k, OUT_BUF_SIZE, "\n");
                doc_topic.write(buf, strlen(buf));
            }
            doc_topic.close();

            for(int i = 0; i < m2.size(); i++){
                snprintf(buf, OUT_BUF_SIZE, "%s", m2[i].c_str());
                k = strlen(buf);
                for(int j = 0; j < K; j++){
                    int c = tw(j, i);
                    sprintf(buf + k, "\t%d", c);
                    k = strlen(buf);
                }
                snprintf(buf + k, OUT_BUF_SIZE, "\n");
                word_topic.write(buf, strlen(buf));
            }
            word_topic.close();

            int tsz = 0;
            for(int i = 0; i < K; i++){
                snprintf(buf, OUT_BUF_SIZE, "topic:%d", i);
                k = strlen(buf);
                for(int j = 0; j < ts[i].size(); j++){
                    sprintf(buf + k, "\t%.4f", ts[i][j]);
                    k = strlen(buf);
                    if(k > OUT_BUF_SIZE - 5){
                        topic_time.write(buf, strlen(buf));
                        k = 0;
                    }
                }
                snprintf(buf + k, OUT_BUF_SIZE, "\n");
                topic_time.write(buf, strlen(buf));
            }
            topic_time.close();
        }

        void cal_beta_fun(vector<float>& bo){
            float t;
            for(int i = 0; i < bps.size(); i++){
                t = lgamma(get<0>(bps[i])) + lgamma(get<1>(bps[i])) - lgamma(get<0>(bps[i]) + get<1>(bps[i]));
                bo.push_back(t);
            }    
        }

        void gibbsample(){
            int di, wi, k, _k = 0;
            float t;
            // prepare K beta func
            vector<float> a, b, bo;
            for(int i = 0; i < bps.size(); i++){
                a.push_back(get<0>(bps[i]) - 1);
                b.push_back(get<1>(bps[i]) - 1);
            }
            cal_beta_fun(bo);

            Map<ArrayXf> alpha(a.data(), K);
            Map<ArrayXf> bt(b.data(), K);
            Map<ArrayXf> div(bo.data(), K);

            for(int i = 0 ; i < tks.size(); i++){
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                t = get<2>(tks[i]);
                k = topics(i);

                dt(k, di) -= 1;
                tw(k, wi) -= 1;
                tpw(k) -= 1;
                
                // evaluate beta[k][t]
                auto bp = (alpha * log(t) + bt * log(1 - t) - div).exp();
                auto v =  bp * (tw.col(wi) + beta) * (dt.col(di) + alpha) / (tpw + V * beta);
                auto s = v.sum();
                // select a new topic
                float r = (rand() % 10000) * s / 10000.0, _s = 0;
                _k = 0;
                for(; _k < K; _k++){
                    _s += v(_k);
                    if(r < _s)
                        break;
                }
                dt(_k, di) += 1;
                tw(_k, wi) += 1;
                tpw(_k) += 1;
                topics(i) = _k;
                ts[_k].push_back(t);
            } 
        }

        void train(){
            srand(123);
            int s1 = m1.size(); // doc
            int s2 = m2.size(); // word
            int s3 = tks.size(); // tokens
            V = s2;
            
            dt.resize(K, s1);
            dt.setZero();

            tw.resize(K, s2);
            tw.setZero();

            tpw.resize(K);
            tpw.setZero();

            topics.resize(s3);

            bps.resize(K); // K beta functions
            ts.resize(K); // K time samples' containers

            int di, wi, k;
            for(int i = 0 ; i < s3; i++){
                k = topics(i) = rand() % K;
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                dt(k, di) += 1;
                tw(k, wi) += 1;
                tpw(k) += 1;
                ts[k].push_back(get<2>(tks[i])); // TODO: vectorize ?
            }

            // init beta parameters
            update_beta_params(0);

            for(int i = 0 ; i < niter; i++){
                cerr<<"Iter: "<<i<<"\t...\t";
                gibbsample();
                update_beta_params(i == (niter - 1));
                cerr<<"done!"<<endl;
            }
        }
};

int load_tokens(string& fname, LDA& lda){
    Tokens& tks = lda.tks;
    imap& m1 = lda.m1;
    imap& m2 = lda.m2;

    fstream f(fname);
    string s;
    int c1 = 0, c2 = 0, pos, pos1, sz;
    char buf[64];
    const char * p;
    int idx[2];
    umap x;
    float t;
    while(getline(f, s)){
        pos = s.find("\t");
        pos1 = s.find("\t", pos + 1);
        sz = s.length();
        string t1 = s.substr(0, pos);
        string t2 = s.substr(pos + 1, pos1 - pos - 1);
        float t = atof(s.substr(pos1 + 1, sz - pos1).c_str());
        if(x.find(t1) == x.end()){
            x[t1] = c1;
            m1[c1] = t1;
            ++c1;
        }
        if(x.find(t2) == x.end()){
            x[t2] = c2;
            m2[c2] = t2;
            ++c2;
        }
        tks.push_back(Token(x[t1], x[t2], t));
    }
    return 0;
}

int main(int argc, char * argv[]){
    string st(argv[1]);
    float a = atof(argv[2]);
    float b = atof(argv[3]);
    int k = atoi(argv[4]);
    int n = atoi(argv[5]);
    LDA lda(a, b, k, n);
    load_tokens(st, lda);
    cerr<<"load done"<<endl;
    lda.train();
    cerr<<"saving..."<<endl;
    lda.output();
    return 0;
}
