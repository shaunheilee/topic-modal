#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include <cstring>
#include <tuple>

using namespace std;
typedef map<string, int> umap;
typedef map<int, string> imap;
typedef vector<int> feature;
typedef tuple<int, float>   IntFloat;

bool cmp(IntFloat a, IntFloat b){
    return get<1>(a) > get<1>(b);
}

class AUC{
    public:
        float auc(vector<float>& y, vector<float>& p){
            int m = 0, n = 0, sz = y.size(); // positive numbers
            vector<IntFloat> ip;
            for(int i = 0; i < sz; i++){
                if(y[i] == 1.0)
                    ++m;
                ip.push_back(IntFloat(i, p[i]));
            }
            sort(ip.begin(), ip.end(), cmp);

            n = sz - m;
            float sr = 0;
            for(int i = 0; i < ip.size(); i++){
                if(y[get<0>(ip[i])] == 1.0)
                    sr += sz - i;
            }
            return (sr - 0.5 * m * (m + 1)) / (m * n);
        }
};

class samples{
    public:
        vector<float>   Y, tY;
        vector<feature> X, tX;
        int             feature_size;
        int             size;
        umap            idmap;
        vector<string>  ridmap;

        void load(char * fname, int test = 0){
            ifstream fp(fname);
            string line;
            char buf[102400];
            int sz = 0;
            if(test == 0){
                this->feature_size = 0;
                this->size = 0;
            }
            fprintf(stderr,"startloading\n");
            while(getline(fp, line)){
                line.copy(buf, line.size());
                buf[line.size()] = 0;
                char * p = strstr(buf, "\t");
                *p = 0;
                if(test == 0)
                    Y.push_back(atoi(buf));
                else
                    tY.push_back(atoi(buf));
                
                feature f;
                p = strstr(p+1, "\t");//feature len
                while(p != NULL){
                    char * tmp = p + 1;
                    p = strstr(p+1, "\t"); // feature idx
                    if(p != NULL)
                        *p = 0;
                    
                    string key(tmp);
                    if(idmap.find(key) == idmap.end()){
                        if(test == 1)
                            continue;
                        idmap[key] = sz;
                        ridmap.push_back(key);
                        ++sz;
                    }

                    int idx = idmap[key];
                    f.push_back(idx);
                    if(this->feature_size < idx && test == 0)
                        this->feature_size = idx;
                    
                    //p = strstr(p+1, "\t"); // feature val
                }
                if(test == 0)
                    X.push_back(f);
                else
                    tX.push_back(f);
                size++;
            }
            fp.close();
            fprintf(stderr,"done with loading\n");
            if(test == 0){
                ridmap.push_back(string("BIAS"));
                this->feature_size += 1;
            }
        }
};

void load_model(char * mn, vector<float>& w){
    string line;
    ifstream fp(mn);
    getline(fp, line);
    char buf[8096000];
    line.copy(buf, line.size());
    buf[line.size()] = 0;

    fprintf(stderr, "model loading:%s\n", buf);
    char * p = buf;
    char * tmp = NULL;
    do{
        tmp = strstr(p, "\t");
        if(tmp != NULL)
            *tmp = 0;
        float wi = atof(p);
        w.push_back(wi);
        if(tmp != NULL)
            p = tmp + 1;
    }while(tmp != NULL);
    
    fprintf(stderr, "model load done!\n");
    
    fp.close();
}

class LR{
    public:
        float predict_sigmoid(feature& f){
            float dp = w[w.size() - 1];
            for(int i = 0; i < f.size(); i++){
                dp += w[f[i]];
            }
            //fprintf(stderr, "dp:%.3f\n", dp);
            return 1.0 / ( 1.0 + exp( dp * -1.0));
        }

        void savemodel(char * fn, vector<string>& rid){
            FILE* fp = fopen(fn, "w");
            for(int i = 0 ; i < w.size(); i++)
                fprintf(fp, "%s\t%.4f\t\n", rid[i].c_str(), w[i]);
            fclose(fp);
        }

        float predict_linear(feature& f){
            float dp = w[w.size() - 1];
            for(int i = 0; i < f.size(); i++)
                dp += w[f[i]];

            return dp;
        }


        float predict(feature& f){
            if(tp == 0)
                return predict_linear(f);
            else
            if(tp == 1)
                return predict_sigmoid(f);
            return 0;
        }

        void train(samples& s , int niter){
            float preloss = 99999999;
            vector<float> tp;
            AUC r;
            for(int i = 0; i < niter; i++){
                OnePassTrain(s);
                float closs = loss(s);
                if(preloss < closs)
                    a = a * 0.99;
                //a = a * 0.99;
                //if(i % 10 == 9)
                tp.clear();
                for(int j = 0; j < s.tX.size(); j++){
                    tp.push_back(predict(s.tX[j]));
                }
                preloss = closs;
                fprintf(stderr, "iter %d loss:%.6f\t\t test auc:%.3f\n", i, closs, r.auc(s.tY, tp));
            }
        }

        float loss_logistic(samples& s){
            float los = 0;
            float regularization = 0;
            for(int i = 0 ; i < (w.size() - 1 )&& l > 0; i++)
                regularization += w[i] * w[i];
            regularization *= l;

            los += regularization / s.size;
            int cc = 0;
            for(int i = 0; i < s.size; i++){
                float y = s.Y[i];
                feature& f = s.X[i];
                float yest = predict(f);
                //fprintf(stderr, "yest:%.3f\n", yest);
                float ls = (y > 0.1 ? log(yest) : log(1 - yest));
                los += -1 * (ls < -9 ? -9 : ls);
                
                if(y > 0.5 && yest > 0.5)
                    cc += 1;
                else if(y < 0.5 && yest < 0.5)
                    cc += 1;
            }
            fprintf(stderr, "precision:%.3f\t", cc * 1.0 / s.size);
            return los / s.size;
        }

        float loss_linear(samples& s){
            float los = 0;
            float regularization = 0;
            for(int i = 0 ; i < (w.size() - 1 )&& l > 0; i++)
                regularization += w[i] * w[i];
            regularization *= l;

            los += regularization / s.size;
            
            for(int i = 0; i < s.size; i++){
                float y = s.Y[i];
                feature& f = s.X[i];
                float yest = predict(f);
                //fprintf(stderr, "yest:%.3f\n", yest);
                los += (y - yest) * (y - yest);
            }
            
            
            return sqrt(los / s.size);
        }


        float loss(samples& s){
            if(tp == 0)
                return loss_linear(s);
            else
            if(tp == 1)
                return loss_logistic(s);
            return 0;
        }

        void cal_g(samples& s, vector<float>& err, vector<float>& g){
            for(int i = 0; i < s.size; i++){
                float ei = err[i];
                feature& x = s.X[i];
                g[w.size() - 1] += ei;
                for(int j = 0; j < x.size(); j++){
                    g[x[j]] += ei;
                }
            }

            for(int i = 0; i < w.size(); i++)
                g[i] = g[i] / s.size + l * w[i] / s.size;
        }

        void OnePassTrain(samples& s){
            vector<float> err;
            for(int i = 0; i < s.size; i++){
                float y = s.Y[i];
                feature& f = s.X[i];
                float yest = predict(f);
                err.push_back(yest - y);    
            }
            // cal g
            vector<float> g(w.size());
            cal_g(s, err, g);

            //update w
            for(int i = 0; i < w.size(); i++){
                w[i] = w[i] - a * g[i];
            }
        }

        void loadmodel(vector<float>& md){
            w.erase(w.begin(), w.end());
            for(int i = 0 ; i < md.size(); i++)
                w.push_back(md[i]);
        }

        LR(int wsize, float alpha, float lambda, int type){
            w.resize(wsize + 1);
            fprintf(stderr,"wsize:%lu\n", w.size());
            for(int i = 0; i < wsize; i++){
                w[i] = rand() % 100 / 100000.0;            
            }
            a = alpha;
            l = lambda;
            tp = type;
        }
    private:
        vector<float>   w;
        float           a;  //learn_rate
        float           l;  //lambda for regularization
        int             tp; // linear:tp = 0 sigmoid: tp = 1
};

class FTRL{
    public:
        FTRL(float a, float b, float l1, float l2, int wsize){
            w.resize(wsize + 1);
            z.resize(wsize + 1);
            n.resize(wsize + 1);
            fa = a;
            fb = b;
            fl1 = l1;
            fl2 = l2;
        }

        float predict(feature & f, int up_w = 0){
            for(int i = 0 ; i < w.size() && up_w == 1; i++){
                if(fabs(z[i]) < fl1)
                    w[i] = 0;
                else{
                    int sgn = z[i] > 0 ? 1:-1;
                    float tmp = (sqrt(n[i]) + fb) / fa;
                    w[i] = (sgn * fl1 - z[i]) / (fl2 + tmp);
                }
            }

            float dp = 0;
            for(int i = 0 ; i < f.size(); i++)
                dp += w[f[i]];
            //return dp;
            return 1.0 / ( 1.0 + exp( dp * -1.0));
        }

        float update(feature & f, float y){
            float pt = predict(f, 1);
            float g = pt - y;
            for(int j = 0 ; j < f.size(); j++){
                int i = f[j];
                float gs = g * g;
                float dt = (sqrt(n[i] + gs) - sqrt(n[i])) / fa;
                z[i] = z[i] + g - dt * w[i];
                n[i] = n[i] + gs;
            }
            return pt;
        }

        float loss(samples& s){
            float los = 0;
            for(int i = 0; i < s.size; i++){
                float y = s.Y[i];
                feature& f = s.X[i];
                float yest = predict(f, 0);
                //fprintf(stderr, "yest:%.3f\n", yest);
                yest = yest > (1 - 10e-15) ? (1 - 10e-15) : (yest < 10e-15 ? 10e-15:yest);
                los += y == 1.0 ? -log(yest) : -log(1 - yest);
                /*
                if(y > 0.5 && yest > 0.5)
                    cc += 1;
                else if(y < 0.5 && yest < 0.5)
                    cc += 1;
                */
            }
            //fprintf(stderr, "precision:%.3f\t loss:%.3f\n", cc * 1.0 / s.size, los/s.size);
            return sqrt(los / s.size);
        }

        void train(samples& s, int niter){
            vector<float> p, tp;
            AUC r;
            for(int i = 0 ; i < niter; i++){
                float loss = 0;
                p.clear();
                for(int j = 0; j < s.X.size(); j++){
                    float opt = update(s.X[j], s.Y[j]);
                    p.push_back(opt);
                    loss += (opt - s.Y[j]) * (opt - s.Y[j]);
                    //fprintf(stderr, "pred:%.3f y:%.3f\n", opt, s.Y[j]);
                    if(j % 20 && 0){
                        tp.clear();
                        for(int j = 0; j < s.tX.size(); j++){
                            tp.push_back(predict(s.tX[j]));
                        }
                        fprintf(stderr,"\t\t\ttest auc:%.3f\n", r.auc(s.tY, tp));
                    }
                }
                tp.clear();
                for(int j = 0; j < s.tX.size(); j++){
                    tp.push_back(predict(s.tX[j]));
                }
                fprintf(stderr, "loss:%.3f\t\t train auc:%.3f \t\t test auc:%.3f\n", sqrt(loss), r.auc(s.Y, p), r.auc(s.tY, tp));
            } 
        }

        void savemodel(char * fn, vector<string>& rid){
            FILE* fp = fopen(fn, "w");
            for(int i = 0 ; i < w.size(); i++)
                fprintf(fp, "%s\t%.4f\t\n", rid[i].c_str(), w[i]);
            fclose(fp);
        }
    private:
        vector<float>   w;
        vector<float>   z;  // params of ftrl proximal
        vector<float>   n;  // params of ftrl proximal
        float           fa,fb;  // input params of ftrl
        float           fl1, fl2;   // input params of ftrl
};

int main(int argc, char * argv[])
{
    srand(time(0));
    samples s;
    s.load(argv[1]);
    s.load(argv[2], 1);
    float alpha, beta, l1, l2;
    alpha = atof(argv[3]);
    beta = atof(argv[4]);
    l1 = atof(argv[5]);
    l2 = atof(argv[6]);
    //LR lr(s.feature_size, 0.1, 0.02, 1);
    //LR lr(s.feature_size, alpha, beta, 1);
    //lr.train(s, 1500);
    //lr.savemodel("./lr.model", s.ridmap);
   
    
    FTRL f(alpha, beta, l1, l2, s.feature_size);
    f.train(s, 10);
    f.savemodel("./ftrl.model", s.ridmap);

    /*
    LR lr(s.feature_size, 0.001, 0.001, 0);
    lr.train(s, 100);
    float loss = lr.loss(s);
    fprintf(stderr, "loss:%.3f\n", loss);
    */
    return 0;
}
