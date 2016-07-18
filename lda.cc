#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <string>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <map>
#include <cstdlib>

using namespace std;
typedef tuple<int, int> Token;
typedef vector<Token>   Tokens;
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
        LDA(float a, float b, int k, int n){
            alpha = a;
            beta = b;
            K = k;
            niter = n;
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
            for(int i = 0 ; i < tks.size(); i++){
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                k = topics(i);

                dt(k, di) -= 1;
                tw(k, wi) -= 1;
                tpw(k) -= 1;

                auto v =  (tw.col(wi) + beta) * (dt.col(di) + alpha) / (tpw + V * beta);
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

            int di, wi, k;
            for(int i = 0 ; i < s3; i++){
                k = topics(i) = rand() % K;
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                dt(k, di) += 1;
                tw(k, wi) += 1;
                tpw(k) += 1;
            }

            for(int i = 0 ; i < niter; i++){
                cerr<<"Iter: "<<i<<"\t...\t";
                gibbsample();
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
    int c1 = 0, c2 = 0, pos, sz;
    char buf[64];
    const char * p;
    int idx[2];
    umap x;
    while(getline(f, s)){
        pos = s.find("\t");
        sz = s.length();
        string t1 = s.substr(0, pos);
        string t2 = s.substr(pos + 1, sz - pos);
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
        tks.push_back(Token(x[t1], x[t2]));
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
