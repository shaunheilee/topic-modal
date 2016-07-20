#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <map>
#include <cstdlib>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using namespace boost::math;
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

        void update_beta_params(){
            // cal sample mean for each topic
            // cal biased sampe varience for each topic
            // using method of moments to estimate beta params
            float mean, var, * buf;
            int sz;
            for(int i = 0 ; i < ts.size(); i++){
                buf = ts[i].data();
                sz = ts[i].size();
                Map<ArrayXf> v(buf, ts[i].size());
                mean = v.mean();
                auto x = v - mean;
                var = (x * x).sum() / sz; // biased varience
                // using boost estimator
                get<0>(bps[i]) = beta_distribution<>::find_alpha(mean, var);
                get<1>(bps[i]) = beta_distribution<>::find_beta(mean, var);
                ts[i].clear();
            }
        }

        void output(){
            ofstream doc_topic;
            ofstream word_topic;
            doc_topic.open("doc_topic", ios::out);
            word_topic.open("word_topic", ios::out);
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
        }

        void gibbsample(){
            int di, wi, k, _k = 0;
            float t;
            // prepare K beta func
            vector<beta_distribution<> > bfs;
            for(int i = 0; i < bps.size(); i++)
                bfs.push_back(beta_distribution<>(get<0>(bps[i]), get<1>(bps[i])));
            vector<float> _bp;
            _bp.resize(K);

            for(int i = 0 ; i < tks.size(); i++){
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                t = get<2>(tks[i]);
                k = topics(i);

                dt(k, di) -= 1;
                tw(k, wi) -= 1;
                tpw(k) -= 1;

                // evaluate beta[k][t]
                for(int j = 0; j < K; j++)
                    _bp[j] = pdf(bfs[j], t);
               
                Map<ArrayXf> bp(_bp.data(), K);

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
            update_beta_params();

            for(int i = 0 ; i < niter; i++){
                cerr<<"Iter: "<<i<<"\t...\t";
                gibbsample();
                update_beta_params();
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
        string t2 = s.substr(pos + 1, pos1 - pos);
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
    lda.train();
    lda.output();
    return 0;
}
