#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <boost/tokenizer.hpp>
#include <string>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <boost/algorithm/string.hpp>
#include <map>
#include <boost/lexical_cast.hpp>
#include <cstdlib>

using namespace std;
typedef tuple<int, int> Token;
typedef vector<Token>   Tokens;
typedef boost::unordered_map<string, int> umap;
typedef boost::unordered_map<int, string> imap;
using namespace Eigen;

#define get_prev(x) (((x)&0x7fff0000)>>16)
#define get_next(x) (((x)&0x0000ffff))
#define set_prev(x, i)  {(x)= ((x)&0x8000ffff)|(i<<16);}
#define set_next(x, i)  {(x)= ((x)&0xffff0000) + i;}
typedef struct _vc {
    int v, ptr;
}   VNode;

#define print(x, y, z) cerr<<(x)<<" "<<(y)<<" "<<z<<endl;
#define show_row(x, r, m) {cerr<<m<<endl;int p = get_next(x[r * (K+1) + K].ptr);while(p!=K){cerr<<p<<",";p=get_next(x[r*(K+1)+p].ptr);}cerr<<endl;}

class LDA{
    public:
        imap    m1;
        imap    m2;
        Tokens  tks;
        float   alpha, beta;
        int     K, niter, V;
        VNode * dt;
        VNode * tw;
        int * tpw;
        int * topics;
        LDA(float a, float b, int k, int n){
            alpha = a;
            beta = b;
            K = k;
            niter = n;
        }

        ~LDA(){
            free(dt);
            free(tw);
            free(tpw);
            free(topics);
            topics = NULL;
            tpw = NULL;
            tw = NULL;
            dt = NULL;
        }

        void output(){
            ofstream doc_topic;
            ofstream word_topic;
            doc_topic.open("doc_topic", ios::out);
            word_topic.open("word_topic", ios::out);
            int _K = K + 1;
            for(int i = 0 ; i < m1.size(); i++){
                doc_topic << m1[i];
                for(int j = 0; j < K; j++){
                    auto c = dt[i * _K + j].v;
                    doc_topic << "\t" << boost::lexical_cast<string>(c);
                }
                doc_topic << endl;
            }
            doc_topic.close();
            for(int i = 0; i < m2.size(); i++){
                word_topic << m2[i];
                for(int j = 0; j < K; j++){
                    auto c = tw[i * _K + j].v;
                    word_topic << "\t" << boost::lexical_cast<string>(c);
                }
                word_topic << endl;
            }
            word_topic.close();
        }

        void remove_from_chain(VNode * ar, int r, int c){
            int _K = K + 1;
            int prev = get_prev(ar[r * _K + c].ptr);
            int next = get_next(ar[r * _K + c].ptr);
            //prev next
            set_next(ar[r * _K + prev].ptr, next);
            set_prev(ar[r * _K + next].ptr, prev);
        }

        void add_to_chain(VNode * ar, int r, int c){
            int _K = K + 1;
            int v = ar[r * _K + c].v, tv;
            // K's prev is the last valid entry
            int prev = get_prev(ar[r * _K + K].ptr);
            set_next(ar[r * _K + prev].ptr, c);// prev's next is c
            set_prev(ar[r * _K + c].ptr, prev);// c's prev is prev
            set_prev(ar[r * _K + K].ptr, c);// next's prev is c
            set_next(ar[r * _K + c].ptr, K);// c's next is next
        }

        void gibbsample(int _it){
            int di, wi, k, _k = 0, _K = K + 1, p, preidx = -1;
            double e, f, g, s;
            float ab = alpha * beta, vb = V * beta, div, r;
            float * ea = new float[K], *fa = new float[K], *ga = new float[K];
            memset(ea, 0, sizeof(float) * K);
            memset(fa, 0, sizeof(float) * K);
            memset(ga, 0, sizeof(float) * K);
            e = 0;
            for(int i = 0; i < K; i++){
                ea[i] = ab / (tpw[i] + vb);
                e += ea[i];
            }

            for(int i = 0 ; i < tks.size(); i++){
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                k = topics[i];
                if(preidx != di){
                    f = 0;
                    p = get_next(dt[di * _K + K].ptr);
                    while(p != K){
                        fa[p] = beta * dt[di * _K + p].v / (tpw[p] + vb);
                        f += fa[p];
                        p = get_next(dt[di * _K + p].ptr);
                    }               
                }
                preidx = di;

                dt[di * _K + k].v -= 1;
                tw[wi * _K + k].v -= 1;
                tpw[k] -= 1;
                // keep non zero chain
                if(dt[di * _K + k].v == 0)
                    remove_from_chain(dt, di, k);
                if(tw[wi * _K + k].v == 0){
                    remove_from_chain(tw, wi, k);
                }
                // cal e f g
                // select a _k
                div = tpw[k] + vb;
                ea[k] = ea[k] - ab / (div + 1) + ab/ div;
                e = e - ab / (div + 1) + ab/ div;
                // f: dt non zero
                fa[k] = fa[k] + beta * dt[di * _K + k].v / div - beta * (dt[di * _K + k].v + 1) / (div + 1);
                f = f + beta * dt[di * _K + k].v / div - beta * (dt[di * _K + k].v + 1) / (div + 1);
                
                // g: tw non zero
                g = 0;
                p = get_next(tw[wi * _K + K].ptr);
                while(p != K){
                    ga[p] = tw[wi * _K + p].v * (alpha + dt[di * _K + p].v) / (tpw[p] + vb);
                    g += ga[p];
                    p = get_next(tw[wi * _K + p].ptr);
                }
                s = e + f + g;
                r = rand() % 100000 * s / 100000.0;
                double _s = 0;
                if(r < e){
                    for(_k = 0; _k < K; _k++){
                        _s += ea[_k];
                        if(r < _s)
                            break;
                    }
                    if(_k == K)
                        _k = K - 1;
                    //cout<<_s<<" "<<r<<endl;
                }
                else
                if(r < e + f){
                    _k = get_next(dt[di * _K + K].ptr);
                    while(_k != K){
                        _s += fa[_k];
                        if(r < _s + e)
                            break;
                        _k = get_next(dt[di * _K + _k].ptr);
                    } 
                }
                else{
                    _k = get_next(tw[wi * _K + K].ptr);
                    while(_k != K){
                        _s += ga[_k];
                        if(r < _s + e + f)
                            break;
                        _k = get_next(tw[wi * _K + _k].ptr);
                    }
                }
               
                if(_k == K){//ERROR
                    cerr<<"di:"<<di<<endl;
                    cerr<<"wi:"<<wi<<endl;
                    cerr<<"r,e,f,g,s:"<<r<<" "<<e<<" "<<f<<" "<<g<<" "<<s<<endl;
                    show_row(dt, di, "dt[di]");
                    show_row(tw, wi, "tw[wi]");
                    exit(-1);
                }

                dt[_k + di * _K].v += 1;
                tw[_k + wi * _K].v += 1;
                tpw[_k] += 1;
                topics[i] = _k;
                
                if(dt[di * _K + _k].v == 1)
                    add_to_chain(dt, di, _k);
                if(tw[wi * _K + _k].v == 1)
                    add_to_chain(tw, wi, _k);

                div = tpw[_k] + vb;
                ea[_k] = ea[_k] + ab / div - ab / (div - 1);
                e = e + ab / div - ab / (div - 1);
                fa[_k] = fa[_k] + beta * dt[di * _K + _k].v / div - beta * (dt[di * _K + _k].v - 1) / (div - 1);
                f = f + beta * dt[di * _K + _k].v / div - beta * (dt[di * _K + _k].v - 1) / (div - 1);
            }
            delete [] ea;
            delete [] fa;
            delete [] ga;
        }

        void train(){
            srand(123);
            int s1 = m1.size(); // doc
            int s2 = m2.size(); // word
            int s3 = tks.size(); // tokens
            V = s2;
            
            dt = (VNode *) malloc(s1 * (K+1) * sizeof(VNode));
            memset(dt, 0, s1 * (K+1) * sizeof(VNode));
            
            tw = (VNode *) malloc(s2 * (K+1) * sizeof(VNode));
            memset(tw, 0, s2 * (K+1) * sizeof(VNode));

            tpw = (int *) malloc(K * sizeof(int));
            memset(tpw, 0, K * sizeof(int));

            topics = (int *) malloc( s3 * sizeof(int));

            int di, wi, k;
            int _K = K + 1;
            for(int i = 0 ; i < s3; i++){
                k = topics[i] = rand() % K;
                di = get<0>(tks[i]);
                wi = get<1>(tks[i]);
                dt[di * _K + k].v += 1;
                tw[wi * _K + k].v += 1;
                tpw[k] += 1;
            }
            // init non zero chain for doc topic
            for(int i = 0 ; i < s1; i++){
                int prev = K, next = K, c;
                set_prev(dt[i * _K + K].ptr, K);
                set_next(dt[i * _K + K].ptr, K);
                for(int j = 0; j < K; ++j){
                    c = K - j - 1;
                    if(dt[i * _K + j].v !=0){
                        set_prev(dt[i * _K + j].ptr, prev);
                        if(prev == K)
                            set_next(dt[i * _K + K].ptr, j);
                        prev = j;
                    }
                    if(dt[i * _K + c].v !=0){
                        set_next(dt[i * _K + c].ptr, next);
                        if(next == K)
                            set_prev(dt[i * _K + K].ptr, c);
                        next = c;
                    }
                }
            }
            // init non zero chain for word topic
            for(int i = 0; i < s2; i++){
                int prev = K, next = K, c;
                set_prev(tw[i * _K + K].ptr, K);
                set_next(tw[i * _K + K].ptr, K);
                for(int j = 0; j < K; ++j){
                    c = K - j - 1;
                    if(tw[i * _K + j].v !=0){
                        set_prev(tw[i * _K + j].ptr, prev);
                        if(prev == K)
                            set_next(tw[i * _K + K].ptr, j);
                        prev = j;
                    }
                    if(tw[i * _K + c].v !=0){
                        set_next(tw[i * _K + c].ptr, next);
                        if(next == K)
                            set_prev(tw[i * _K + K].ptr, c);
                        next = c;
                    }
                }
            }

            for(int i = 0 ; i < niter; i++){
                cerr<<"Iter: "<<i<<"\t...\t";
                gibbsample(i);
                cerr<<"done!"<<endl;
            }
        }
};

int split(char * buf, int * idx, const char* sep){
    char *p = buf;
    int c = 0;
    idx[0] = 0;

    while(strsep(&p, sep)){
        ++c;
        idx[c] = p - buf;
    }
    return c;
}

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
    using boost::lexical_cast;
    float a = lexical_cast<float>(argv[2]);
    float b = lexical_cast<float>(argv[3]);
    int k = lexical_cast<int>(argv[4]);
    int n = lexical_cast<int>(argv[5]);
    LDA lda(a, b, k, n);
    load_tokens(st, lda);
    lda.train();
    lda.output();
    return 0;
}
