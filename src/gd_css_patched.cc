// NOTE: Japanese runtime messages have been translated into English (string literals changed).
// Comments are preserved. If you want to keep the original Japanese output, use the *_en_comments.cc file instead.
#include <unordered_set>
#include <iostream>
#include <regex>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <map>
#include <set>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
using namespace std;

// ======= Parameter Objects for TryDecodeSmallErrors (argument reduction) =======
struct SM_StateRef {
  int &SyndromeIsSatisfied;
  std::vector<int> &SuspectJ;
  std::vector<int> &RUSS;
  std::vector<std::vector<int>> &USSHistory;
  std::vector<int> &EstmNoiseSynd;
  std::vector<int> &TrueNoiseSynd;
  std::vector<int> &TrueNoise;
  std::vector<int> &EstmNoise;
  std::vector<int> &Candidate_Covering_Normal_Rows;
};
struct SM_CodeRef {
  std::vector<std::vector<int>> &JatI;
  std::vector<std::vector<int>> &IatJ;
  std::vector<std::vector<int>> &Mat;
  std::vector<std::vector<int>> &MatValue;
  std::vector<int> &RowDeg;
};
struct SM_UtcBcRef {
  std::vector<std::vector<int>> &UTCBC_Rows_C_orthogonal_D;
  std::vector<std::vector<int>> &UTCBC_Cols_C_orthogonal_D;
  std::vector<std::vector<int>> &full_JatI_C;
  std::vector<std::vector<int>> &full_IatJ_C;
  std::vector<std::vector<int>> &full_JatI_D;
  std::vector<std::vector<int>> &full_IatJ_D;
};
struct SM_GFTablesRef {
  std::vector<std::vector<int>> &MULGF;
  std::vector<std::vector<int>> &ADDGF;
  std::vector<std::vector<int>> &DIVGF;
  std::vector<std::vector<int>> &BINGF;
};
// ==============================================================================
string EF_LOG;
vector<int> inv_ZP;
int L,P;
int DEBUG_transmission;
int M,N;
int itr;
int eS;
int eS_C;
int eS_D;
int transmission;
int GF,logGF;
FILE *f;

char *MatrixFilePrefix_C=(char *)malloc(500);
char *MatrixFilePrefix_D=(char *)malloc(500);

vector<int> TrueNoise_C;
vector<int> TrueNoise_D;
vector<vector<int>> ADDGF;
vector<vector<int>> MULGF;
vector<vector<int>> DIVGF;
vector<vector<int>> BINGF;
vector<vector<int>> FFTSQ;
vector<vector<int>> TBINGF;
vector<vector<int>> TFFTSQ;
vector<int> TrueNoiseSynd_C;
vector<int> EstmNoiseSynd_C;
vector<int> ColDeg_C;
vector<int> RowDeg_C;
vector<vector<int>> Mat_C;
vector<vector<int>> MatValue_C;
vector<vector<int>> NtoB_C;
vector<int> Interleaver_C;
vector<int> Puncture_C;
vector<int> EstmNoise_C;
vector<vector<double>> CNtoVNxxx_C;
vector<vector<double>> VNtoCNxxx_C;
vector<vector<double>> APP_C;
vector<int> TrueNoiseSynd_D;
vector<int> EstmNoiseSynd_D;
vector<int> ColDeg_D;
vector<int> RowDeg_D;
vector<vector<int>> Mat_D;
vector<vector<int>> MatValue_D;
vector<vector<int>> NtoB_D;
vector<int> Interleaver_D;
vector<int> Puncture_D;
vector<int> EstmNoise_D;
vector<vector<double>> VNtoChN_CD;
vector<vector<double>> VNtoChN_DC;
vector<vector<double>> CNtoVNxxx_D;
vector<vector<double>> VNtoCNxxx_D;
vector<vector<double>> APP_D;
vector<vector<double>> ChNtoVN_CD;
vector<vector<double>> ChNtoVN_DC;
int NumUSS_C,NumUSS_D;
const int HistoryLength=8;
vector<vector<int>> Updated_EstmNoise_History_C(HistoryLength);
vector<vector<int>> Updated_EstmNoise_History_D(HistoryLength);
vector<vector<int>> USSHistory_C(HistoryLength);
vector<vector<int>> USSHistory_D(HistoryLength);

vector<vector<int>> UTCBC_Rows_C_orthogonal_D;

vector<vector<int>> UTCBC_Rows_D_orthogonal_C;

vector<vector<int>> UTCBC_Cols_C_orthogonal_D;

vector<vector<int>> UTCBC_Cols_D_orthogonal_C;
vector<vector<int>> JatI_C;
vector<vector<int>> IatJ_C;
vector<vector<int>> JatI_D;
vector<vector<int>> IatJ_D;
vector<vector<int>> full_JatI_C;
vector<vector<int>> full_JatI_D;
vector<vector<int>> full_IatJ_C;
vector<vector<int>> full_IatJ_D;

int SyndromeIsSatisfied_C;
int NumEdge_C,P_C=0;
double Rate_C;

int SyndromeIsSatisfied_D;
int NumEdge_D,P_D=0;
double Rate_D;

int TeB=0,TeS=0,eB,TeF=0,TdS=0;
double f_m,pD;
int NbUndetectedErrors;
vector<int> IncorrectJ_C;
vector<int> IncorrectJ_D;

Eigen::MatrixXd ChFactorMatrix_CD;
Eigen::MatrixXd ChFactorMatrix_DC;
#include "Matrix.h"
Matrix G_Gamma;
Matrix G_Delta;
using Node = pair<char, int>;
using Loop = vector<Node>;
using LoopSet = vector<Loop>;
using IndexList = vector<vector<int>>;
using BitSet = unordered_set<int>;
using NodeList = vector<int>;
template <typename T>
// Function: contains
// Purpose: TODO - describe the function's responsibility succinctly.
bool contains(const vector<T>& S, const T& x) {
  // Loop: iterate over a range/collection.
  for (const T& elem : S) {
    // Conditional branch.
    if (elem == x) return true;
  }
  return false;
}
// Function: difference
// Purpose: TODO - describe the function's responsibility succinctly.
template <typename T>
vector<T> difference(const vector<T>& A, const vector<T>& B) {
  unordered_set<T> setB(B.begin(), B.end());
  vector<T> diff;
  for (const auto& x : A) {
    if (setB.find(x) == setB.end()) diff.push_back(x);
  }
  return diff;
}
// Function: find_missing_elements
// Purpose: TODO - describe the function's responsibility succinctly.
std::vector<int> find_missing_elements(const std::vector<int>& subset, const std::vector<int>& superset) {
  std::unordered_set<int> set_superset(superset.begin(), superset.end());
  std::vector<int> missing;
  // Loop: iterate over a range/collection.
  for (int j : subset) {
    // Conditional branch.
    if (!set_superset.count(j)) {
      missing.push_back(j);
    }
  }
  // Conditional branch.
  if (!missing.empty()) {
    printf("Missing elements:");
    // Loop: iterate over a range/collection.
    for (int j : missing) {
      printf(" %d", j);
    }
    printf("\n");
  }else{
  printf("All included!\n");
}
return missing;
}

vector<int> find_dangerous_checks(const NodeList& C0, const IndexList& JatI, const BitSet& S) {
  vector<int> danger;
  // Loop: iterate over a range/collection.
  for (int c : C0) {
    int count = 0;
    // Loop: iterate over a range/collection.
    for (int v : JatI[c]) {
      // Conditional branch.
      if (S.find(v) != S.end()) count++;
    }
    // Conditional branch.
    if (count == 1) {
      danger.push_back(c);
    }
  }
  return danger;
}
// Function: print_progress_bar
// Purpose: TODO - describe the function's responsibility succinctly.
void print_progress_bar(int current, int total, const string& label = "", int bar_width = 50) {
  float progress = float(current) / total;
  int pos = int(bar_width * progress);
  cout << "\r\033[32m[";
  // Loop: iterate over a range/collection.
  for (int j = 0; j < bar_width; ++j) {
    // Conditional branch.
    if (j < pos) cout << "=";
    else if (j == pos) cout << ">";
    else cout << " ";
  }
  cout << "]\033[0m ";
  cout << int(progress * 100) << "% ";
  // Conditional branch.
  if (!label.empty()) {
    cout << "(" << label << ")";
  }
  cout << flush;
}
// Function: VNtoChN_init
// Purpose: TODO - describe the function's responsibility succinctly.

void VNtoChN_init(Eigen::MatrixXd& f_VNtoChN_eigen,
double pD,int GF,int logGF,vector<vector<int>>& BINGF0,vector<vector<int>>& BINGF1){
  cout << "@@@ VNtoChN_init" << endl;
  f_VNtoChN_eigen=Eigen::MatrixXd(GF,GF);
  cout << "eigen done" << endl;
  // Loop: iterate over a range/collection.
  for(size_t d=0;d<GF;d++){

    // Loop: iterate over a range/collection.
    for(size_t e=0;e<GF;e++){
      f_VNtoChN_eigen(d,e)=1.0f;
      // Loop: iterate over a range/collection.
      for(size_t l=0;l<logGF;l++){
        // Conditional branch.
        if(BINGF0[d][l]==0 && BINGF1[e][l]==0){
          f_VNtoChN_eigen(d,e)*=1-pD;
        }else{
        f_VNtoChN_eigen(d,e)*=pD/3;
      }
    }
  }
}
cout << "*** VNtoChN_init" << endl;
}
// Function: normalize
// Purpose: TODO - describe the function's responsibility succinctly.

void normalize(vector<double>& input, int n){
  double sum=0;for(size_t i=0;i<n;i++){sum+=input[i];}
  // Conditional branch.
  if(sum==0){
    cout << "divided by zero" << endl;
    // Loop: iterate over a range/collection.
    for(size_t i=0;i<n;i++){input[i]=1.0f/n;}
    return;
  }else
  // Loop: iterate over a range/collection.
  for(size_t i=0;i<n;i++){input[i]/=sum;}
}
// Function: log2
// Purpose: TODO - describe the function's responsibility succinctly.

int log2(int x){
  return log((double)x)/log(2.0);
}
// Function: h2
// Purpose: TODO - describe the function's responsibility succinctly.

double h2(double x){ return -x*std::log2(x) - (1.0 - x)*std::log2(1.0 - x); }
// Function: Bin2GF
// Purpose: TODO - describe the function's responsibility succinctly.

int Bin2GF(vector<int>& U,int GF,int logGF,vector<vector<int>>& BINGF){
  // Loop: iterate over a range/collection.
  for(size_t k=0;k<GF;k++){
    bool mtc=true;
    // Loop: iterate over a range/collection.
    for(size_t j=0;j<logGF;j++){if(U[j]!=BINGF[k][j]){mtc=false;break;}}
    // Conditional branch.
    if(mtc)return(k);
  }
  return -1;
}
// Function: GF2GF
// Purpose: TODO - describe the function's responsibility succinctly.

int GF2GF(int g,int GF,int logGF,vector<vector<int>>& BINGF0,vector<vector<int>>& BINGF1){

  return Bin2GF(BINGF0[g],GF,logGF,BINGF1);
}

// Function: ComputeAPP
// Purpose: TODO - describe the function's responsibility succinctly.
void ComputeAPP(std::vector<std::vector<double>> &APP,
std::vector<std::vector<double>> &ChNtoVN,
const std::vector<std::vector<double>> &CNtoVNxxx,
const std::vector<std::vector<double>> &VNtoChN,
const std::vector<int> &Interleaver,
const std::vector<int> &ColDeg,
int N, int GF) {
  int numB = 0;
  // Loop: iterate over a range/collection.
  for (int n = 0; n < N; ++n) {
    // Loop: iterate over a range/collection.
    for (int g = 0; g < GF; ++g) {
      ChNtoVN[n][g] = 1.0;
      // Loop: iterate over a range/collection.
      for (int t = numB; t < numB + ColDeg[n]; ++t) {
        ChNtoVN[n][g] *= CNtoVNxxx[Interleaver[t]][g];
      }
      APP[n][g] = ChNtoVN[n][g] * VNtoChN[n][g];
    }
    numB += ColDeg[n];
  }
}

// Function: computeUnion
// Purpose: TODO - describe the function's responsibility succinctly.
std::vector<int> computeUnion(const std::vector<std::vector<int>>& Updated_EstmNoise_History) {

  std::set<int> unionSet;

  // Loop: iterate over a range/collection.
  for (const auto& vec : Updated_EstmNoise_History) {

    unionSet.insert(vec.begin(), vec.end());
  }

  std::vector<int> unionVector(unionSet.begin(), unionSet.end());

  std::sort(unionVector.begin(), unionVector.end());
  return unionVector;
}
// Function: Decision
// Purpose: TODO - describe the function's responsibility succinctly.

void Decision(std::vector<int> &Decision,
std::vector<int> &Updated_EstmNoise_History,
const std::vector<std::vector<double>> &APP,
int N, int GF) {
  Updated_EstmNoise_History.clear();
  // Loop: iterate over a range/collection.
  for (int n = 0; n < N; ++n) {
    int ind = 0;
    double max = APP[n][0];
    // Loop: iterate over a range/collection.
    for (int g = 1; g < GF; ++g) {
      // Conditional branch.
      if (APP[n][g] >= max) {
        max = APP[n][g];
        ind = g;
      }
    }
    // Conditional branch.
    if (Decision[n] != ind) {
      Updated_EstmNoise_History.push_back(n);
    }
    Decision[n] = ind;
  }
}
// Function: ChannelPass_zero
// Purpose: TODO - describe the function's responsibility succinctly.

void ChannelPass_zero(vector<vector<double>>& VNtoChN, int N,int GF,int logGF,double f_m,vector<vector<int>>& BINGF){
  vector<double> VNtoChN0(GF,1);
  // Loop: iterate over a range/collection.
  for(size_t d=0;d<GF;d++){
    // Loop: iterate over a range/collection.
    for(size_t l=0;l<logGF;l++){
      // Conditional branch.
      if(BINGF[d][l]==0)
      VNtoChN0[d]*=1-f_m;
      else
      VNtoChN0[d]*=f_m;
    }
  }
  // Loop: iterate over a range/collection.
  for(size_t n=0;n<N;n++){
    // Loop: iterate over a range/collection.
    for(size_t g=0;g<GF;g++){
      VNtoChN[n][g]=VNtoChN0[g];
    }
  }
}
// Function: ChannelPass
// Purpose: TODO - describe the function's responsibility succinctly.

void ChannelPass(vector<vector<double>>& VNtoChN, Eigen::MatrixXd& f_VNtoChN_eigen, vector<vector<double>>& ChNtoVN, int N, int GF){// Loop: iterate over a range/collection.
  for(size_t n=0;n<N;n++){
    vector<double> input(GF);
// Loop: iterate over a range/collection.
    for(size_t g=0;g<GF;g++){input[g]=ChNtoVN[n][g];}
    normalize(input,GF);
    Eigen::VectorXd vec(GF),res(GF);
    // Loop: iterate over a range/collection.
    for(size_t e=0;e<GF;e++){vec(e)=input[e];}
    res=f_VNtoChN_eigen.transpose()*vec;

    // Loop: iterate over a range/collection.
    for(size_t e=0;e<GF;e++){VNtoChN[n][e]=res(e);}
    normalize(VNtoChN[n],GF);
  }
}
// Function: DataPass
// Purpose: TODO - describe the function's responsibility succinctly.

void DataPass(vector<vector<double>>& VNtoCNxxx,vector<vector<double>>& CNtoVNxxx,vector<vector<double>>& VNtoChN,vector<int>& Interleaver,vector<int>& ColumnDegree,int N,int GF){

  int numB=0;
  // Loop: iterate over a range/collection.
  for(size_t n=0;n<N;n++){
    // Loop: iterate over a range/collection.
    for(size_t t=0;t<ColumnDegree[n];t++){
      int numBint=Interleaver[numB+t];
      // Loop: iterate over a range/collection.
      for(size_t g=0;g<GF;g++) VNtoCNxxx[numBint][g]=VNtoChN[n][g];
      // Loop: iterate over a range/collection.
      for(size_t tz=0;tz<ColumnDegree[n];tz++){
        // Conditional branch.
        if (tz!=t)
        // Loop: iterate over a range/collection.
        for(size_t g=0;g<GF;g++)
        VNtoCNxxx[numBint][g]*=CNtoVNxxx[Interleaver[numB+tz]][g];
      }
      normalize(VNtoCNxxx[numBint],GF);
    }
    numB+=ColumnDegree[n];
  }
}
// Function: CheckPass
// Purpose: TODO - describe the function's responsibility succinctly.

void CheckPass(vector<vector<double>>& CNtoVNxxx,vector<vector<double>>& VNtoCNxxx,vector<vector<int>>& MatValue,int M,vector<int>& RowDegree,vector<vector<int>>& MULGF,vector<vector<int>>& DIVGF,vector<vector<int>>& FFTSQ,int GF,vector<int>& TrueNoiseSynd){

  int           tz,t,k,m,g,numB;
  int           logGF;
  vector<double> TMP;
  double  Afft,Bfft,buff;
  TMP=vector<double>(GF,0);
  vector<double> F_TrueNoiseSynd(GF,0);
  numB=0;
  logGF=rint(log2(GF)/log2(2));
  // Loop: iterate over a range/collection.
  for(size_t m=0;m<M;m++){

    // Loop: iterate over a range/collection.
    for(g=0;g<GF;g++){F_TrueNoiseSynd[g]=0;}F_TrueNoiseSynd[TrueNoiseSynd[m]]=1;
    // Loop: iterate over a range/collection.
    for(k=0;k<logGF*GF/2;k++){
      Afft=F_TrueNoiseSynd[FFTSQ[k][0]];      Bfft=F_TrueNoiseSynd[FFTSQ[k][1]];
      F_TrueNoiseSynd[FFTSQ[k][0]]=Afft+Bfft; F_TrueNoiseSynd[FFTSQ[k][1]]=Afft-Bfft;
    }

    // Loop: iterate over a range/collection.
    for(t=0;t<RowDegree[m];t++){
      // Loop: iterate over a range/collection.
      for(g=0;g<GF;g++) TMP[g]=VNtoCNxxx[numB+t][DIVGF[g][MatValue[m][t]]];
      // Loop: iterate over a range/collection.
      for(g=0;g<GF;g++) VNtoCNxxx[numB+t][g]=TMP[g];
      // Loop: iterate over a range/collection.
      for(k=0;k<logGF*GF/2;k++){
        Afft=VNtoCNxxx[numB+t][FFTSQ[k][0]];
        Bfft=VNtoCNxxx[numB+t][FFTSQ[k][1]];
        VNtoCNxxx[numB+t][FFTSQ[k][0]]=Afft+Bfft;
        VNtoCNxxx[numB+t][FFTSQ[k][1]]=Afft-Bfft;
      }
    }

    // Loop: iterate over a range/collection.
    for(t=0;t<RowDegree[m];t++)	{
      // Loop: iterate over a range/collection.
      for(g=0;g<GF;g++){CNtoVNxxx[numB+t][g]=F_TrueNoiseSynd[g];}
      // Loop: iterate over a range/collection.
      for(tz=0;tz<RowDegree[m];tz++){if(tz!=t){for(g=0;g<GF;g++){CNtoVNxxx[numB+t][g]*=VNtoCNxxx[numB+tz][g];}}}
    }

    // Loop: iterate over a range/collection.
    for(t=0;t<RowDegree[m];t++){
      // Loop: iterate over a range/collection.
      for(k=0;k<(logGF*GF/2);k++){
        Afft=CNtoVNxxx[numB+t][FFTSQ[k][0]];
        Bfft=CNtoVNxxx[numB+t][FFTSQ[k][1]];
        CNtoVNxxx[numB+t][FFTSQ[k][0]]=0.5f*(Afft+Bfft);
        CNtoVNxxx[numB+t][FFTSQ[k][1]]=0.5f*(Afft-Bfft);
      }
      // Loop: iterate over a range/collection.
      for(g=0;g<GF;g++) TMP[g]=CNtoVNxxx[numB+t][MULGF[MatValue[m][t]][g]];
      // Loop: iterate over a range/collection.
      for(g=0;g<GF;g++) CNtoVNxxx[numB+t][g]=std::max(TMP[g], 0.0);
      normalize(CNtoVNxxx[numB+t],GF);
    }
    numB+=RowDegree[m];
  }
}
// Function: calcSyndrome
// Purpose: TODO - describe the function's responsibility succinctly.

void calcSyndrome(vector<int>& Synd, int M,vector<int>& EstmNoise,vector<vector<int>>& MatValue,vector<int>& RowDegree,vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF, vector<vector<int>>& Mat){
  // Loop: iterate over a range/collection.
  for(int k=0;k<M;k++){
    Synd[k]=0;
    // Loop: iterate over a range/collection.
    for(int l=0;l<RowDegree[k];l++){
      int n=Mat[k][l];
      int H=MatValue[k][l];
      int x=MULGF[H][EstmNoise[n]];
      int s=Synd[k];
      Synd[k]=ADDGF[s][x];
    }
  }
}
// Function: IsSyndromeSatisfied
// Purpose: TODO - describe the function's responsibility succinctly.

int IsSyndromeSatisfied(vector<int>& TrueNoiseSynd,vector<int>& EstmNoiseSynd,vector<int>& USS, int M,vector<int>& EstmNoise,vector<vector<int>>& MatValue,vector<int>& RowDegree,vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF, vector<vector<int>>& Mat){

  USS.clear();
  calcSyndrome(EstmNoiseSynd, M,EstmNoise,MatValue,RowDegree,ADDGF, MULGF,Mat);
  // Loop: iterate over a range/collection.
  for(size_t k=0;k<M;k++){
    // Conditional branch.
    if(EstmNoiseSynd[k]!=TrueNoiseSynd[k]){
      USS.push_back(k);
    }
  }
  // Loop: iterate over a range/collection.
  for(size_t k=0;k<M;k++){
    // Conditional branch.
    if(EstmNoiseSynd[k]!=TrueNoiseSynd[k]){
      return 0;
    }
  }
  return 1;
}
// Function: count_errors
// Purpose: TODO - describe the function's responsibility succinctly.

void count_errors(
int N, int M,
vector<int>& EstmNoise_C, vector<int>& EstmNoise_D,
vector<int>& TrueNoise_C, vector<int>& TrueNoise_D,
vector<int>& EstmNoiseSynd_C, vector<int>& TrueNoiseSynd_C,
vector<int>& EstmNoiseSynd_D, vector<int>& TrueNoiseSynd_D,
vector<int>& IncorrectJ_C,
vector<int>& IncorrectJ_D,
int& eS, int& eS_C, int& eS_D,
int& NumUSS_C, int& NumUSS_D
) {

  IncorrectJ_C.clear();
  IncorrectJ_D.clear();
  eS=0;
  eS_C=0;
  eS_D=0;
  // Loop: iterate over a range/collection.
  for(size_t k=0;k<N;k++){
    // Conditional branch.
    if ( EstmNoise_C[k]!=TrueNoise_C[k]){ eS_C++;  eS++;IncorrectJ_C.push_back(k);}
  }
  // Loop: iterate over a range/collection.
  for(size_t k=0;k<N;k++){
    // Conditional branch.
    if ( EstmNoise_D[k]!=TrueNoise_D[k]){ eS_D++;  eS++;IncorrectJ_D.push_back(k);}
  }
  NumUSS_C=0;for(int i=0;i<M;i++){if(EstmNoiseSynd_C[i]!=TrueNoiseSynd_C[i]){NumUSS_C++;}}
  NumUSS_D=0;for(int i=0;i<M;i++){if(EstmNoiseSynd_D[i]!=TrueNoiseSynd_D[i]){NumUSS_D++;}}
}
// Function: gaussianElimination
// Purpose: TODO - describe the function's responsibility succinctly.

void gaussianElimination(Matrix& H, vector<int>& Perm) {
  int m = H.rows, n = H.cols;
  Perm=vector<int>(n);
  // Loop: iterate over a range/collection.
  for(int j=0;j<n;j++){Perm[j]=j;}
  int lead = 0;
  int N=n;
  int M=m;

  // Loop: iterate over a range/collection.
  for(int n=0;n<N;n++){Perm[n]=n;}
  int i,j,k,l,pivot;
  int p,q;
  // Loop: iterate over a range/collection.
  for(k=0;k<M;k++){
    p=H[k][k];
    // Conditional branch.
    if(p==0){

      int j1,j2;
      // Loop: iterate over a range/collection.
      for(j=k;j<N;j++){
        // Conditional branch.
        if(H[k][j]!=0){
          j1=k;
          j2=j;
          break;
        }
      }

      // Conditional branch.
      if(j==N){
        H.removeRow(k);
        M=H.rows;
        k--;
        continue;
      }
      p=H[k][j];

      int t=Perm[j1];Perm[j1]=Perm[j2];Perm[j2]=t;
      H.swapColumns(j1,j2);

    }

    // Loop: iterate over a range/collection.
    for(j=0;j<N;j++) {
      H[k][j]=DIVGF[H[k][j]][p];
    }

    // Loop: iterate over a range/collection.
    for(i=k+1;i<M;i++) {
      int q=H[i][k];
      // Conditional branch.
      if(q){

        // Loop: iterate over a range/collection.
        for(j=0;j<N;j++) {
          H[i][j]=ADDGF[H[i][j]][MULGF[q][H[k][j]]];
        }

      }
    }

  }

  // Loop: iterate over a range/collection.
  for(k=1;k<M;k++) {
    // Loop: iterate over a range/collection.
    for(i=0;i<=k-1;i++){
      int q=H[i][k];
      // Conditional branch.
      if(q){

        // Loop: iterate over a range/collection.
        for(j=0;j<N;j++) {
          H[i][j]=ADDGF[H[i][j]][MULGF[q][H[k][j]]];
        }
      }
    }
  }
}

Matrix findGeneratorMatrix(Matrix H) {
  int m = H.rows;
  int n = H.cols;

  Matrix H0(m, n);
  // Loop: iterate over a range/collection.
  for (int i = 0; i < m; ++i) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < n; ++j) {
      H0[i][j] = H[i][j];
    }
  }

  vector<int> Perm(n);
  gaussianElimination(H, Perm);
  int rankH=H.rows;
  printf("rankH=%d, H.rows=%d\n",rankH,H.rows);

  int N=n;
  int M=m;
  int K=N-rankH;
  Matrix G(K,N);
  // Loop: iterate over a range/collection.
  for (int m=0;m<K;m++){for (int n=0;n<N;n++){G[m][n]=0;}}
  // Loop: iterate over a range/collection.
  for(int k=0;k<K;k++){
    G[k][k+rankH]=1;
    // Loop: iterate over a range/collection.
    for(int j=0;j<N-K;j++) {
      G[k][j]=H[j][rankH+k];
    }
  }

  Matrix TH=H;
  // Loop: iterate over a range/collection.
  for(int j1=0;j1<N;j1++){
    int j2 = Perm[j1];
    // Conditional branch.
    if(j1!=j2){

      // Loop: iterate over a range/collection.
      for(int i=0;i<rankH;i++) {
        H[i][j2]=TH[i][j1];
      }
    }
  }
  Matrix TG=G;
  // Loop: iterate over a range/collection.
  for(int j1=0;j1<N;j1++){
    int j2 = Perm[j1];
    // Conditional branch.
    if(j1!=j2){

      // Loop: iterate over a range/collection.
      for(int i=0;i<K;i++) {
        G[i][j2]=TG[i][j1];
      }
    }
  }

  return G;
}

tuple<vector<vector<int>>, unordered_map<int, int>, unordered_map<int, int>>
generateDenseMatrixA(const vector<int>& rows,
const vector<int>& cols,
const vector<vector<int>>& JatI,
vector<vector<int>>& MatValue)
{
  printf("@@@generateDenseMatrixA\n");

  vector<vector<int>> A(rows.size(), vector<int>(cols.size(), 0));

  unordered_map<int, int> colMapping;
  // Loop: iterate over a range/collection.
  for (size_t j = 0; j < cols.size(); j++) {
    colMapping[cols[j]] = j;
  }

  unordered_map<int, int> rowMapping;
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < rows.size(); i++) {
    rowMapping[rows[i]] = i;
  }

  // Loop: iterate over a range/collection.
  for (size_t r = 0; r < rows.size(); r++) {
    int sparseRow = rows[r];
    // Loop: iterate over a range/collection.
    for (int k = 0; k < JatI[sparseRow].size(); k++) {
      int col = JatI[sparseRow][k];

      // Conditional branch.
      if (colMapping.find(col) != colMapping.end()) {
        int c = colMapping[col];
        A[r][c] = MatValue[sparseRow][k];
      }
    }
  }
  printf("***generateDenseMatrixA\n");
  return make_tuple(A, colMapping, rowMapping);
}
// Function: computeRankGF
// Purpose: TODO - describe the function's responsibility succinctly.
int computeRankGF(
vector<vector<int>> A) {
  int n = A.size();
  // Conditional branch.
  if (n == 0) return 0;
  int m = A[0].size();
  int rank = 0;
  // Loop: iterate over a range/collection.
  for (int col = 0, row = 0; col < m && row < n; ++col) {

    int pivot = -1;
    // Loop: iterate over a range/collection.
    for (int i = row; i < n; ++i) {
      // Conditional branch.
      if (A[i][col] != 0) {
        pivot = i;
        break;
      }
    }
    // Conditional branch.
    if (pivot == -1) continue;

    // Conditional branch.
    if (pivot != row) std::swap(A[pivot], A[row]);

    int inv = DIVGF[1][A[row][col]];
    // Loop: iterate over a range/collection.
    for (int j = col; j < m; ++j) {
      A[row][j] = MULGF[inv][A[row][j]];
    }

    // Loop: iterate over a range/collection.
    for (int i = 0; i < n; ++i) {
      // Conditional branch.
      if (i != row && A[i][col] != 0) {
        int factor = A[i][col];
        // Loop: iterate over a range/collection.
        for (int j = col; j < m; ++j) {
          A[i][j] = ADDGF[A[i][j]][MULGF[factor][A[row][j]]];
        }
      }
    }
    ++rank;
    ++row;
  }
  return rank;
}
// Function: computeRankGF
// Purpose: TODO - describe the function's responsibility succinctly.
int computeRankGF(
vector<int>& rows,
vector<int>& cols,
vector<vector<int>>& JatI_C,
vector<vector<int>>& MatValue_C){
  printf("@@@computeRankGF\n");
  printf("rows<%d>:",rows.size());for(int i:rows){printf("%7d(%d)",i,i/P);}printf("\n");
  printf("cols<%d>:",cols.size());for(int i:cols){printf("%7d(%d)",i,i/P);}printf("\n");

  auto result = generateDenseMatrixA(rows, cols, JatI_C, MatValue_C);
  vector<vector<int>> A = std::get<0>(result);
  unordered_map<int, int> colMapping = std::get<1>(result);
  unordered_map<int, int> rowMapping = std::get<2>(result);

  printf("Matrix A(%dx%d)=\n", A.size(), A[0].size());
  // Loop: iterate over a range/collection.
  for (const auto& row : A) {
    // Loop: iterate over a range/collection.
    for (int val : row) {
      printf("%3x", val);
    }
    std::cout << std::endl;
  }
  return computeRankGF(A);
}
vector<vector<int>> enumerateAllSolutions(
vector<vector<int>> A, vector<int> b,
const vector<vector<int>>& ADDGF,
const vector<vector<int>>& MULGF,
const vector<vector<int>>& DIVGF,
int GFq
) {
  int n = A.size();
  // Conditional branch.
  if (n == 0) return {};
  int m = A[0].size();

  vector<vector<int>> aug(n, vector<int>(m + 1));
  // Loop: iterate over a range/collection.
  for (int i = 0; i < n; i++) {
    copy(A[i].begin(), A[i].end(), aug[i].begin());
    aug[i][m] = b[i];
  }
  vector<int> pivot_col(m, -1);
  int pivot_row = 0;
  // Loop: iterate over a range/collection.
  for (int col = 0; col < m && pivot_row < n; col++) {
    int pivot = -1;
    // Loop: iterate over a range/collection.
    for (int row = pivot_row; row < n; row++) {
      // Conditional branch.
      if (aug[row][col] != 0) {
        pivot = row;
        break;
      }
    }
    // Conditional branch.
    if (pivot == -1) continue;
    swap(aug[pivot_row], aug[pivot]);
    int invPivot = DIVGF[1][aug[pivot_row][col]];
    // Loop: iterate over a range/collection.
    for (int j = 0; j <= m; j++) {
      aug[pivot_row][j] = MULGF[aug[pivot_row][j]][invPivot];
    }
    // Loop: iterate over a range/collection.
    for (int row = 0; row < n; row++) {
      // Conditional branch.
      if (row == pivot_row) continue;
      int factor = aug[row][col];
      // Loop: iterate over a range/collection.
      for (int j = 0; j <= m; j++) {
        aug[row][j] = ADDGF[aug[row][j]][MULGF[factor][aug[pivot_row][j]]];
      }
    }
    pivot_col[col] = pivot_row;
    pivot_row++;
  }

  // Loop: iterate over a range/collection.
  for (int i = pivot_row; i < n; i++) {
    bool allZero = true;
    // Loop: iterate over a range/collection.
    for (int j = 0; j < m; j++) {
      // Conditional branch.
      if (aug[i][j] != 0) {
        allZero = false;
        break;
      }
    }
    // Conditional branch.
    if (allZero && aug[i][m] != 0) {
      cout << "No solution." << endl;
      return {};
    }
  }

  vector<int> free_vars;
  vector<int> pivot_vars;
  // Loop: iterate over a range/collection.
  for (int j = 0; j < m; j++) {
    // Conditional branch.
    if (pivot_col[j] == -1) free_vars.push_back(j);
    else pivot_vars.push_back(j);
  }
  int num_free = free_vars.size();
  vector<vector<int>> solutions;

  int total = 1;
  // Loop: iterate over a range/collection.
  for (int i = 0; i < num_free; i++) total *= GFq;
  // Loop: iterate over a range/collection.
  for (int idx = 0; idx < total; idx++) {
    vector<int> x(m, 0);
    int tmp = idx;
    // Loop: iterate over a range/collection.
    for (int k = 0; k < num_free; k++) {
      x[free_vars[k]] = tmp % GFq;
      tmp /= GFq;
    }

    // Loop: iterate over a range/collection.
    for (int i = pivot_row - 1; i >= 0; i--) {
      int pivot_j = -1;
      // Loop: iterate over a range/collection.
      for (int j = 0; j < m; j++) {
        // Conditional branch.
        if (aug[i][j] != 0) {
          pivot_j = j;
          break;
        }
      }
      // Conditional branch.
      if (pivot_j == -1) continue;
      int sum = 0;
      // Loop: iterate over a range/collection.
      for (int j = pivot_j + 1; j < m; j++) {
        sum = ADDGF[sum][MULGF[aug[i][j]][x[j]]];
      }
      x[pivot_j] = ADDGF[aug[i][m]][sum];
    }
    solutions.push_back(x);
  }
  cout << "Found " << solutions.size() << " solutions." << endl;
  return solutions;
}

pair<bool, vector<int>> solveLinearEquations(vector<vector<int>>& A, vector<int>& b,
vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF, vector<vector<int>>& DIVGF) {
  int n = A.size();
  // Conditional branch.
  if (n == 0) return {false, {}};
  int m = A[0].size();
  vector<int> x(m, 0);

  vector<vector<int>> aug(n, vector<int>(m + 1, 0));
  // Loop: iterate over a range/collection.
  for (int i = 0; i < n; i++) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < m; j++) {
      aug[i][j] = A[i][j];
    }
    aug[i][m] = b[i];
  }

  int pivot_row = 0;
  // Loop: iterate over a range/collection.
  for (int col = 0; col < m && pivot_row < n; col++) {

    int pivot = -1;
    // Loop: iterate over a range/collection.
    for (int row = pivot_row; row < n; row++) {
      // Conditional branch.
      if (aug[row][col] != 0) {
        pivot = row;
        break;
      }
    }
    // Conditional branch.
    if (pivot == -1) continue;

    swap(aug[pivot_row], aug[pivot]);

    int invPivot = DIVGF[1][aug[pivot_row][col]];
    // Loop: iterate over a range/collection.
    for (int j = 0; j <= m; j++) {
      aug[pivot_row][j] = MULGF[aug[pivot_row][j]][invPivot];
    }

    // Loop: iterate over a range/collection.
    for (int row = 0; row < n; row++) {
      // Conditional branch.
      if (row == pivot_row) continue;
      int factor = aug[row][col];
      // Loop: iterate over a range/collection.
      for (int j = 0; j <= m; j++) {
        aug[row][j] = ADDGF[aug[row][j]][MULGF[factor][aug[pivot_row][j]]];
      }
    }

    pivot_row++;
  }

  // Loop: iterate over a range/collection.
  for (int i = pivot_row - 1; i >= 0; i--) {

    int pivot_col = -1;
    // Loop: iterate over a range/collection.
    for (int j = 0; j < m; j++) {
      // Conditional branch.
      if (aug[i][j] != 0) {
        pivot_col = j;
        break;
      }
    }
    // Conditional branch.
    if (pivot_col == -1) continue;
    int sum = 0;
    // Loop: iterate over a range/collection.
    for (int j = pivot_col + 1; j < m; j++) {
      sum = ADDGF[sum][MULGF[aug[i][j]][x[j]]];
    }
    x[pivot_col] = ADDGF[aug[i][m]][sum];
  }

  printf("x=\n");for (int i = 0; i < x.size(); i++) {printf("%3x",x[i]);}printf("\n");

  printf("A=\n");
  // Loop: iterate over a range/collection.
  for (int i = 0; i < A.size(); i++) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < A[0].size(); j++) {
      printf("%3x",A[i][j]);
    }
    printf("\n");
  }

  vector<int> computed_b(n, 0);
  // Loop: iterate over a range/collection.
  for (int i = 0; i < n; i++) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < m; j++) {
      computed_b[i] = ADDGF[computed_b[i]][MULGF[A[i][j]][x[j]]];
    }
  }

  printf("computed_b=\n");
  // Loop: iterate over a range/collection.
  for (int i = 0; i < computed_b.size(); i++) {printf("%3x",computed_b[i]);}printf("\n");

  printf("b=\n");
  // Loop: iterate over a range/collection.
  for (int i = 0; i < b.size(); i++) {printf("%3x",b[i]);}printf("\n");

  bool correct = true;
  // Loop: iterate over a range/collection.
  for (int i = 0; i < n; i++) {
    // Conditional branch.
    if (computed_b[i] != b[i]) {
      correct = false;
      break;
    }
  }
  // Conditional branch.
  if (!correct) {
    cout << "Error: Computed solution does not satisfy Ax = b!" << endl;
    return {false, {}};
  }
  cout << "Solution verified successfully: Ax = b holds." << endl;
  return {true, x};
}
// Function: check_degenerate_decoding_success
// Purpose: TODO - describe the function's responsibility succinctly.

bool check_degenerate_decoding_success(vector<int>& EstmNoise_C, vector<int>& Noise_C, vector<vector<int>>& JatI_C, vector<vector<int>>& JatI_D, vector<vector<int>>& MatValue_D, int N, int M){
  printf("@check_degenerate_decoding_success\n");

  unordered_set<int> J_set;
  vector<int> J;

  // Loop: iterate over a range/collection.
  for (int j = 0; j < N; ++j) {
    // Conditional branch.
    if (EstmNoise_C[j] != Noise_C[j]) {
      J_set.insert(j);
      J.push_back(j);
    }
  }

  cout << "J=";
  // Loop: iterate over a range/collection.
  for (int j : J) { cout << j << " "; }
  cout << endl;

  // Conditional branch.
  if (J.empty()) return true;
  unordered_set<int> I_set;
  vector<int> I;

  // Loop: iterate over a range/collection.
  for (int i = 0; i < M; ++i) {
    bool allInJ = true;
    // Loop: iterate over a range/collection.
    for (int j : JatI_D[i]) {
      // Conditional branch.
      if (!J_set.count(j)) {
        allInJ = false;
        break;
      }
    }
    // Conditional branch.
    if (allInJ) {
      I_set.insert(i);
      I.push_back(i);
    }
  }

  cout << "I=";
  // Loop: iterate over a range/collection.
  for (int i : I) { cout << i << " "; }
  cout << endl;

  // Conditional branch.
  if (I.empty()) return false;

  cout << "I=";for(int i:I){cout << i << " ";}cout << endl;

  auto result = generateDenseMatrixA(I, J, JatI_D, MatValue_D);
  vector<vector<int>> A = std::get<0>(result);
  unordered_map<int, int> colMapping = std::get<1>(result);
  unordered_map<int, int> rowMapping = std::get<2>(result);

  vector<vector<int>> At(A[0].size(), vector<int>(A.size()));
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < A.size(); ++i) {
    // Loop: iterate over a range/collection.
    for (size_t j = 0; j < A[0].size(); ++j) {
      At[j][i] = A[i][j];
    }
  }

  std::cout << "Matrix At:" << std::endl;
  // Loop: iterate over a range/collection.
  for (const auto& row : At) {
    // Loop: iterate over a range/collection.
    for (int val : row) {
      std::cout << std::setw(5) << val << " ";
    }
    std::cout << std::endl;
  }
  vector<int> b(J.size(), 0);
  // Loop: iterate over a range/collection.
  for (size_t k = 0; k < J.size(); k++) {
    b[k] = ADDGF[Noise_C[J[k]]][EstmNoise_C[J[k]]];
  }

  cout << "b=";for(int val:b){cout << val << " ";}cout << endl;
  auto res = solveLinearEquations(At, b, ADDGF, MULGF, DIVGF);
  // Conditional branch.
  if(res.first){

    vector<int> x = res.second;

    std::cout << "Solution x:" << std::endl;
    // Loop: iterate over a range/collection.
    for (int val : x) {
      printf("%3x", val);
    }
    std::cout << std::endl;
  }
  return res.first;
}
// Function: addIfNotIncluded
// Purpose: TODO - describe the function's responsibility succinctly.
void addIfNotIncluded(std::vector<int>& v, int x) {

  // Conditional branch.
  if (std::find(v.begin(), v.end(), x) == v.end()) {
    v.push_back(x);
  }
}
vector<int> makeUnion(
const vector<int>& Candidate_Covering_Normal_Rows_D,
const vector<vector<int>>& UTCBC_Indices
) {
  set<int> union_set;
  // Loop: iterate over a range/collection.
  for (int i : Candidate_Covering_Normal_Rows_D) {
    // Loop: iterate over a range/collection.
    for (int v : UTCBC_Indices[i]) {
      union_set.insert(v);
    }
  }

  vector<int> sorted_union(union_set.begin(), union_set.end());
  sort(sorted_union.begin(), sorted_union.end());
  return sorted_union;
}
// Function: Find_Nonsingular_Cycle_of_Length_Larger_thatn_L
// Purpose: TODO - describe the function's responsibility succinctly.
bool Find_Nonsingular_Cycle_of_Length_Larger_thatn_L(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@Find_Nonsingular_Cycle_of_Length_Larger_thatn_L" << endl;
  cols.clear();
  rows.clear();
  set<int> rows_set,cols_set;

  sort(SuspectJ.begin(), SuspectJ.end());

  // Loop: iterate over a range/collection.
  for (int j : SuspectJ) {

    // Loop: iterate over a range/collection.
    for (int i : IatJ_C[j]) {
      int count_J = 0;

      vector<int> intersection;
      set_intersection(JatI_C[i].begin(), JatI_C[i].end(),
      SuspectJ.begin(), SuspectJ.end(),
      back_inserter(intersection));
      // Conditional branch.
      if(intersection.size() == 2) {
        rows_set.insert(i);
        // Loop: iterate over a range/collection.
        for(int j: intersection) {
          cols_set.insert(j);
        }
      }
    }
  }
  rows.assign(rows_set.begin(), rows_set.end());
  cols.assign(cols_set.begin(), cols_set.end());
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("rank=%d\n",rank);
  // Conditional branch.
  if(cols_set.size()>L && cols_set.size()==rows_set.size() && (cols_set.size()==rank)){
    printf("cols_set.size()=0\n");
    printf("***Find_Nonsingular_Cycle_of_Length_Larger_thatn_L end true\n");
    rows.assign(rows_set.begin(), rows_set.end());
    cols.assign(cols_set.begin(), cols_set.end());
    return true;
  }else{
  printf("***Find_Nonsingular_Cycle_of_Length_Larger_thatn_L end false\n");
  return false;
}
}
// Function: Find_Cycle_of_Length_L
// Purpose: TODO - describe the function's responsibility succinctly.
bool Find_Cycle_of_Length_L(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@Find_Cycle_of_Length_L" << endl;
  cols.clear();
  rows.clear();
  set<int> rows_set,cols_set;

  sort(SuspectJ.begin(), SuspectJ.end());

  // Loop: iterate over a range/collection.
  for (int j : SuspectJ) {

    // Loop: iterate over a range/collection.
    for (int i : IatJ_C[j]) {
      vector<int> intersection;
      set_intersection(JatI_C[i].begin(), JatI_C[i].end(),
      SuspectJ.begin(), SuspectJ.end(),
      back_inserter(intersection));
      // Conditional branch.
      if(intersection.size() == 2) {
        rows_set.insert(i);
        // Loop: iterate over a range/collection.
        for(int j: intersection) {
          cols_set.insert(j);
        }
      }
    }
  }
  rows.assign(rows_set.begin(), rows_set.end());
  cols.assign(cols_set.begin(), cols_set.end());
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("rank=%d\n",rank);
  // Conditional branch.
  if(cols_set.size()==L && rows_set.size()==L && (rank==L || rank==L-1)){
    printf("cols_set.size()=0\n");
    printf("***Find_Cycle_of_Length_L end true\n");
    rows.assign(rows_set.begin(), rows_set.end());
    cols.assign(cols_set.begin(), cols_set.end());
    return true;
  }else{
  printf("***Find_Cycle_of_Length_L end false\n");
  return false;
}
}
// Function: Find_Covering_Cycles_By_RUSS
// Purpose: TODO - describe the function's responsibility succinctly.

bool Find_Covering_Cycles_By_RUSS(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<int>& RUSS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C){
  cout << "@@@Find_Covering_Cycles_By_RUSS" << endl;
  cols.clear();
  rows.clear();
  rows=RUSS;
  set<int> rows_set,cols_set;

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    // Loop: iterate over a range/collection.
    for(int ii=0;ii<rows.size();ii++){
      // Conditional branch.
      if(i==ii){continue;}

      vector<int> intersection;
      set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
      JatI_C[rows[ii]].begin(), JatI_C[rows[ii]].end(),
      back_inserter(intersection));

      // Conditional branch.
      if(intersection.size()==1){

        // Loop: iterate over a range/collection.
        for(int j:intersection){
          cols_set.insert(j);
        }
      }
    }
  }
  cols.assign(cols_set.begin(), cols_set.end());

  vector<int> missingJ=difference(SuspectJ,cols);
  // Conditional branch.
  if (missingJ.empty()) {
    printf("YES. SuspectJ is covered by cols\n");
    printf("***Find_Covering_Cycles_By_RUSS end true \n");
    return true;
  }else{
  printf("NO. SuspectJ is NOT covered by cols\n");
  printf("***Find_Covering_Cycles_By_RUSS end false\n");
  cols.clear();
  rows.clear();
  return false;
}
}
// Function: Find_Rows_Coverintg_SuspectJ_From_RUSS
// Purpose: TODO - describe the function's responsibility succinctly.
bool Find_Rows_Coverintg_SuspectJ_From_RUSS(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<int>& RUSS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@@@Find_Rows_Coverintg_SuspectJ_From_RUSS" << endl;
  cols.clear();
  rows.clear();
  rows=RUSS;
  set<int> rows_set,cols_set;

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    vector<int> intersection;
    set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
    SuspectJ.begin(), SuspectJ.end(),
    back_inserter(intersection));

    // Loop: iterate over a range/collection.
    for(int j:intersection){
      cols_set.insert(j);
    }
  }
  cols.assign(cols_set.begin(), cols_set.end());

  vector<int> missingJ=difference(SuspectJ,cols);
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("cols.size()=%d rank(rows,cols)=%d\n",cols.size(),rank);
  // Conditional branch.
  if (cols.size()<= rank){
    printf("***Find_Rows_Coverintg_SuspectJ_From_RUSS end true\n");
    return true;
  }else{
  printf("***Find_Rows_Coverintg_SuspectJ_From_RUSS end false\n");
  cols.clear();
  rows.clear();
  return false;
}
}
// Function: Find_Unique_Solution_Noise_From_USS
// Purpose: TODO - describe the function's responsibility succinctly.
bool Find_Unique_Solution_Noise_From_USS(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<int>& USS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@@@Find_Unique_Solution_Noise_From_RUSS" << endl;
  cols.clear();
  rows.clear();
  rows=USS;
  set<int> rows_set,cols_set;

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    vector<int> intersection;
    set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
    SuspectJ.begin(), SuspectJ.end(),
    back_inserter(intersection));

    // Loop: iterate over a range/collection.
    for(int j:intersection){
      cols_set.insert(j);
    }
  }
  cols.assign(cols_set.begin(), cols_set.end());

  vector<int> missingJ=difference(SuspectJ,cols);
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("cols.size()=%d rank(rows,cols)=%d\n",cols.size(),rank);
  // Conditional branch.
  if (cols.size()<= rank){
    printf("***Find_Unique_Solution_Noise_From_RUSS end true\n");
    return true;
  }else{
  printf("***Find_Unique_Solution_Noise_From_RUSS end false\n");
  cols.clear();
  rows.clear();
  return false;
}
}
// Function: Rows_eq_USS_Cols_eq_Overlapping_USS
// Purpose: TODO - describe the function's responsibility succinctly.
bool Rows_eq_USS_Cols_eq_Overlapping_USS(vector<int>& cols, vector<int>& rows, vector<int>& USS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@@@Rows_eq_USS_Cols_eq_Overlapping_USS" << endl;
  cols.clear();
  rows.clear();
  rows=USS;
  set<int> rows_set,cols_set;

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    // Loop: iterate over a range/collection.
    for(int ii=0;ii<rows.size();ii++){
      // Conditional branch.
      if(i==ii){continue;}

      vector<int> intersection;
      set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
      JatI_C[rows[ii]].begin(), JatI_C[rows[ii]].end(),
      back_inserter(intersection));

      // Conditional branch.
      if(intersection.size()==1){

        // Loop: iterate over a range/collection.
        for(int j:intersection){
          cols_set.insert(j);
        }
      }
    }
  }
  // Conditional branch.
  if (cols_set.size() == 0) {
    printf("cols_set.size()=0\n");
    printf("***Rows_eq_USS_Cols_eq_Overlapping_USS end false\n");
    return false;
  }
  cols.assign(cols_set.begin(), cols_set.end());
  printf("cols=");for(int x: cols){printf("%7d(%d)",x,x/P);}printf("\n");
  printf("rows=");for(int x: rows){printf("%7d(%d)",x,x/P);}printf("\n");
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("cols.size()=%d rank(rows,cols)=%d\n",cols.size(),rank);
  // Conditional branch.
  if (cols.size()<= rank){
    printf("***Rows_eq_USS_Cols_eq_Overlapping_USS end true\n");
    return true;
  }else{
  printf("***Rows_eq_USS_Cols_eq_Overlapping_USS end false\n");
  return false;
}
}
// Function: Rows_eq_RUSS_Cols_eq_SuspectJ
// Purpose: TODO - describe the function's responsibility succinctly.
bool Rows_eq_RUSS_Cols_eq_SuspectJ(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<int>& RUSS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@@@Rows_eq_RUSS_Cols_eq_SuspectJ" << endl;
  cols.clear();
  rows.clear();
  rows=RUSS;
  cols=SuspectJ;
  printf("cols=");for(int x: cols){printf("%7d(%d)",x,x/P);}printf("\n");
  printf("rows=");for(int x: rows){printf("%7d(%d)",x,x/P);}printf("\n");
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("cols.size()=%d rank(rows,cols)=%d\n",cols.size(),rank);
  // Conditional branch.
  if (cols.size()<= rank){
    printf("***Rows_eq_RUSS_Cols_eq_SuspectJ end true\n");
    return true;
  }else{
  printf("***Rows_eq_RUSS_Cols_eq_SuspectJ end false\n");
  return false;
}
}
// Function: Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap
// Purpose: TODO - describe the function's responsibility succinctly.

bool Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ, vector<int>& RUSS, vector<vector<int>>& JatI_C, vector<vector<int>>& IatJ_C, vector<vector<int>>& MatValue_C){
  cout << "@@@Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap" << endl;
  cols.clear();
  rows.clear();
  rows=RUSS;
  set<int> rows_set,cols_set;

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    vector<int> intersection;
    set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
    SuspectJ.begin(), SuspectJ.end(),
    back_inserter(intersection));

    // Loop: iterate over a range/collection.
    for(int j:intersection){
      cols_set.insert(j);
    }
  }

  // Loop: iterate over a range/collection.
  for(int i=0;i<rows.size();i++){
    // Loop: iterate over a range/collection.
    for(int j=0;j<rows.size();j++){
      // Conditional branch.
      if(i==j){continue;}

      vector<int> intersection;
      set_intersection(JatI_C[rows[i]].begin(), JatI_C[rows[i]].end(),
      JatI_C[rows[j]].begin(), JatI_C[rows[j]].end(),
      back_inserter(intersection));

      // Conditional branch.
      if(intersection.size()==1){

        // Loop: iterate over a range/collection.
        for(int j:intersection){
          cols_set.insert(j);
        }
      }
    }
  }
  cols.assign(cols_set.begin(), cols_set.end());
  int rank=computeRankGF(rows,cols,JatI_C,MatValue_C);
  printf("cols.size()=%d rank(rows,cols)=%d\n",cols.size(),rank);
  // Conditional branch.
  if (cols.size()<= rank){
    printf("***Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap end true\n");
    return true;
  }else{
  printf("***Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap end false\n");
  cols.clear();
  rows.clear();
  return false;
}
}
// Function: Find_Normal_Rows_Covering_SuspectJ_By_UTCBC_Cols
// Purpose: TODO - describe the function's responsibility succinctly.

bool Find_Normal_Rows_Covering_SuspectJ_By_UTCBC_Cols(vector<int>& cols, vector<int>& rows, vector<int>& SuspectJ_C, vector<vector<int>>& UTCBC_Rows_C_orthogonal_D, vector<vector<int>>& UTCBC_Cols_C_orthogonal_D, vector<vector<int>>& full_IatJ_D){
  cout << "@findCoveringRowsByUTCBC_Cols" << endl;
  vector<int> Candidate_Covering_Normal_Rows_D;

  Candidate_Covering_Normal_Rows_D.clear();

  sort(SuspectJ_C.begin(), SuspectJ_C.end());

  // Loop: iterate over a range/collection.
  for (int j : SuspectJ_C) {

    // Loop: iterate over a range/collection.
    for (int i : full_IatJ_D[j]) {

      // Conditional branch.
      if(find(Candidate_Covering_Normal_Rows_D.begin(), Candidate_Covering_Normal_Rows_D.end(), i) != Candidate_Covering_Normal_Rows_D.end()){
        continue;
      }

      vector<int> intersection;
      set_intersection(UTCBC_Cols_C_orthogonal_D[i].begin(), UTCBC_Cols_C_orthogonal_D[i].end(),
      SuspectJ_C.begin(), SuspectJ_C.end(),
      back_inserter(intersection));
      printf("intersection=");for(int x: intersection){printf("%7d",x);}printf("\n");

      // Conditional branch.
      if (intersection.size() >= 2) {
        printf("%d is added\n",i);
        Candidate_Covering_Normal_Rows_D.push_back(i);
        // Conditional branch.
        if (Candidate_Covering_Normal_Rows_D.size() > 2) {
          printf("Candidate_Covering_Normal_Rows_D.size()>2\n");
          printf("***findCoveringRowsByUTCBC_Cols end false\n");
          return false;
        }
        set<int> rows_set,cols_set;
        // Loop: iterate over a range/collection.
        for (int i : Candidate_Covering_Normal_Rows_D) {
          // Loop: iterate over a range/collection.
          for (int j: UTCBC_Cols_C_orthogonal_D[i]) {
            cols_set.insert(j);
          }
          // Loop: iterate over a range/collection.
          for (int ii: UTCBC_Rows_C_orthogonal_D[i]) {
            rows_set.insert(ii);
          }
        }

        vector<int> UnionJ=makeUnion(Candidate_Covering_Normal_Rows_D, UTCBC_Cols_C_orthogonal_D);
        vector<int> missingJ=difference(SuspectJ_C,UnionJ);
        // Conditional branch.
        if (missingJ.empty()) {
          printf("YES. SuspectJ_C is covered by UnionJ\n");

          rows.assign(rows_set.begin(), rows_set.end());
          cols.assign(cols_set.begin(), cols_set.end());
          printf("***findCoveringRowsByUTCBC_Cols end true \n");
          return true;
        }
      }
    }
  }
  printf("***findCoveringRowsByUTCBC_Cols end false\n");
  return false;
}
// Function: printMatrix
// Purpose: TODO - describe the function's responsibility succinctly.

void printMatrix(const vector<vector<int>>& mat) {
  // Loop: iterate over a range/collection.
  for (const auto& row : mat) {
    // Loop: iterate over a range/collection.
    for (int val : row) {
      cout << setw(5) << val << " ";
    }
    cout << endl;
  }
  cout << "-------------------------" << endl;
}
// Function: isEqual
// Purpose: TODO - describe the function's responsibility succinctly.

bool isEqual(vector<int>& a, vector<int>& b, int size) {
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < size; i++) {
    // Conditional branch.
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}
// Function: computeDeterminantGF
// Purpose: TODO - describe the function's responsibility succinctly.

int computeDeterminantGF(
vector<vector<int>> A,
const vector<vector<int>>& ADDGF,
const vector<vector<int>>& MULGF,
const vector<vector<int>>& DIVGF,
int GF_minus_one
) {
  int N = A.size();
  int det = 1;
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; ++i) {
    // Conditional branch.
    if (A[i][i] == 0) {

      bool found = false;
      // Loop: iterate over a range/collection.
      for (int j = i + 1; j < N; ++j) {
        // Conditional branch.
        if (A[j][i] != 0) {
          std::swap(A[i], A[j]);
          det = MULGF[det][GF_minus_one];
          found = true;
          break;
        }
      }
      // Conditional branch.
      if (!found) return 0;
    }
    det = MULGF[det][A[i][i]];
    int inv = DIVGF[1][A[i][i]];
    // Loop: iterate over a range/collection.
    for (int j = i + 1; j < N; ++j) {
      // Conditional branch.
      if (A[j][i] != 0) {
        int factor = MULGF[A[j][i]][inv];
        // Loop: iterate over a range/collection.
        for (int k = i; k < N; ++k) {
          A[j][k] = ADDGF[A[j][k]][MULGF[factor][A[i][k]]];
        }
      }
    }
  }
  return det;
}
// Function: decode_small_errors_from_rows_cols
// Purpose: TODO - describe the function's responsibility succinctly.

void decode_small_errors_from_rows_cols(vector<int>& TrueNoiseSynd, vector<int>& TrueNoise, int M, int N, vector<int>& EstmNoise, vector<int>& RowDeg, vector<int>& rows, vector<int>& cols, vector<vector<int>>& IatJ_C, vector<vector<int>>& JatI_C, vector<vector<int>>& Mat, vector<vector<int>>& MatValue, vector<vector<int>>& MULGF, vector<vector<int>>& ADDGF, vector<vector<int>>& DIVGF, vector<vector<int>>& BINGF){
  printf("@decode_small_errors_from_rows_cols\n");
  vector<int> Set_J=cols;

  auto result = generateDenseMatrixA(rows, cols, JatI_C, MatValue);
  vector<vector<int>> A = std::get<0>(result);
  unordered_map<int, int> colMapping = std::get<1>(result);
  unordered_map<int, int> rowMapping = std::get<2>(result);

  std::cout << "Matrix A:" << std::endl;
  // Loop: iterate over a range/collection.
  for (const auto& row : A) {
    // Loop: iterate over a range/collection.
    for (int val : row) {
      printf("%3x", val);
    }
    std::cout << std::endl;
  }

  vector<int> Set_JO;
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {
    // Conditional branch.
    if (find(Set_J.begin(), Set_J.end(), i) == Set_J.end()) {
      Set_JO.push_back(i);
    }
  }

  // Conditional branch.
  if (A.size() == A[0].size()) {
    int det = computeDeterminantGF(A, ADDGF, MULGF, DIVGF, GF - 1);
    printf("det=%3x\n", det);
  }
  vector<int> sigmaU(M, 0);
  vector<int> sigmaU_J(M, 0);
  vector<int> sigmaU_JO(M, 0);
  vector<int> sigmaUH_JO(M, 0);
  vector<int> xiU(N, 0);
  vector<int> xiU_J(cols.size(), 0);
  vector<int> xiU_J_ZP(N, 0);
  vector<int> xiUH_J_ZP(N, 0);
  vector<int> xiU_JO_ZP(N, 0);
  vector<int> xiUH_JO_ZP(N, 0);
  vector<int> sigmaU_J_plus_sigmaU_JO(M, 0);
  vector<int> sigmaU_J_plus_sigmaUH_JO(M, 0);
  vector<int> xiUH_J(cols.size(), 0);
  printf("cols=");for(int i=0;i<cols.size();i++){printf("%7d",cols[i]);}printf("\n");
  printf("rows=");for(int i=0;i<rows.size();i++){printf("%7d",rows[i]);}printf("\n");

  // Loop: iterate over a range/collection.
  for (int i = 0; i < Set_J.size(); i++) {xiU_J[i]        = TrueNoise[cols[i]];}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < Set_J.size(); i++) {xiUH_J[i]       = EstmNoise[cols[i]];}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {xiU[i]        = TrueNoise[i];}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {xiU_J_ZP[i]  = TrueNoise[i];}
  // Loop: iterate over a range/collection.
  for (int n : Set_JO) {xiU_J_ZP[n]=0;}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {xiUH_J_ZP[i]  = EstmNoise[i];}
  // Loop: iterate over a range/collection.
  for (int n : Set_JO) {xiUH_J_ZP[n]=0;}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {xiU_JO_ZP[i]  = TrueNoise[i];}
  // Loop: iterate over a range/collection.
  for (int n : Set_J){xiU_JO_ZP[n]=0;}
  // Loop: iterate over a range/collection.
  for (int i = 0; i < N; i++) {xiUH_JO_ZP[i] = EstmNoise[i];}
  // Loop: iterate over a range/collection.
  for (int n : Set_J) {xiUH_JO_ZP[n]=0;}
  printf("-------------------------------------------------\n");
  printf("xiU_J=\n");for(int i=0;i<cols.size();i++){printf("%3x",xiU_J[i]);}printf("\n");
  printf("(xiU)_J=\n");for(int i=0;i<cols.size();i++){printf("%3x",xiU[cols[i]]);}printf("\n");
  printf("xiUH_J=\n");for(int i=0;i<cols.size();i++){printf("%3x",xiUH_J[i]);}printf("\n");
  printf("-------------------------------------------------\n");

  // Loop: iterate over a range/collection.
  for (int i = 0; i < M; i++) {sigmaU_JO[i] = TrueNoiseSynd[i];}

  calcSyndrome(sigmaU,    M,xiU,       MatValue,RowDeg,ADDGF, MULGF,Mat);
  calcSyndrome(sigmaU_J,  M,xiU_J_ZP,  MatValue,RowDeg,ADDGF, MULGF,Mat);
  calcSyndrome(sigmaU_JO, M,xiU_JO_ZP, MatValue,RowDeg,ADDGF, MULGF,Mat);
  calcSyndrome(sigmaUH_JO,M,xiUH_JO_ZP,MatValue,RowDeg,ADDGF, MULGF,Mat);

  printf("sigmaU_JO=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaU_JO[rows[i]]);}printf("\n");

  printf("sigmaUH_JO=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaUH_JO[rows[i]]);}printf("\n");
  // Conditional branch.
  if (isEqual(sigmaU_JO, sigmaUH_JO, M)) {
    std::cout << "sigmaU_JO == sigmaUH_JO" << std::endl;
  } else {
  std::cout << "sigmaU_JO != sigmaUH_JO" << std::endl;
}
printf("-------------------------------------------------\n");
// Loop: iterate over a range/collection.
for (int i = 0; i < M; i++) {
  sigmaU_J_plus_sigmaU_JO[i] = ADDGF[sigmaU_J[i]][sigmaU_JO[i]];
}
printf("sigmaU_J_plus_sigmaU_JO=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaU_J_plus_sigmaU_JO[rows[i]]);}printf("\n");
printf("sigmaU=\n"); for(int i=0;i<rows.size();i++){printf("%3x",sigmaU[rows[i]]);}printf("\n");
// Conditional branch.
if(isEqual(sigmaU_J_plus_sigmaU_JO, sigmaU, M)){
  std::cout << "sigmaU_J + sigmaU_JO == sigmaU" << std::endl;
}else{
std::cout << "sigmaU_J + sigmaU_JO != sigmaU" << std::endl;
}
printf("-------------------------------------------------\n");
printf("sigmaU_J=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaU_J[rows[i]]);}printf("\n");
printf("sigmaUH_JO=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaUH_JO[rows[i]]);}printf("\n");
// Loop: iterate over a range/collection.
for (int i = 0; i < M; i++) {
  sigmaU_J_plus_sigmaUH_JO[i] = ADDGF[sigmaU_J[i]][sigmaUH_JO[i]];
}

printf("(sigmaU_J_plus_sigmaUH_JO)_I=\n");
// Loop: iterate over a range/collection.
for (int i = 0; i < rows.size(); i++) {
  printf("%3x", sigmaU_J_plus_sigmaUH_JO[rows[i]]);
}
printf("\n");
printf("-------------------------------------------------\n");
printf("sigmaU=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaU[rows[i]]);}printf("\n");
printf("sigmaUH_JO=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaUH_JO[rows[i]]);}printf("\n");
vector<int> RHS_b(rows.size(), 0);

// Loop: iterate over a range/collection.
for (size_t i = 0; i < rows.size(); i++) {
  RHS_b[i] = ADDGF[sigmaU[rows[i]]][sigmaUH_JO[rows[i]]];
}
printf("(sigmaU-sigmaUH_JO)_I=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<RHS_b.size();i++){printf("%3x",RHS_b[i]);}printf("\n");
printf("-------------------------------------------------\n");
printf("sigmaU_J=\n");for(int i=0;i<rows.size();i++){printf("%3x",sigmaU_J[rows[i]]);}printf("\n");
printf("-------------------------------------------------\n");

printf("(sigmaU_JO)_I=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<rows.size();i++){printf("%3x",sigmaU_JO[rows[i]]);}printf("\n");
printf("xiU_J=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<cols.size();i++){printf("%3x",xiU_J[i]);}printf("\n");
printf("xiUH_J=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<cols.size();i++){printf("%3x",xiUH_J[i]);}printf("\n");
printf("-------------------------------------------------\n");

vector<int> LHS_b(rows.size(), 0);
// Loop: iterate over a range/collection.
for (size_t i = 0; i < rows.size(); i++) {
  // Loop: iterate over a range/collection.
  for (size_t j = 0; j < cols.size(); j++) {
    int t=MULGF[A[i][j]][xiU_J[j]];
    LHS_b[i] = ADDGF[LHS_b[i]][t];
  }
}

printf("LHS_b=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<LHS_b.size();i++){printf("%3x",LHS_b[i]);}printf("\n");
printf("-------------------------------------------------\n");

printf("RHS_b=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<RHS_b.size();i++){printf("%3x",RHS_b[i]);}printf("\n");
printf("xiU_J=\n");
// Loop: iterate over a range/collection.
for(int i=0;i<cols.size();i++){printf("%3x",xiU_J[i]);}printf("\n");

printf("-------------------------------------------------\n");
// Conditional branch.
if(isEqual(LHS_b, RHS_b, rows.size())){
  std::cout << "True noise gets verification successful: Ay = b holds." << std::endl;
}else{
std::cout << "Error: Ay != b. Verification failed!" << std::endl;
}
printf("-------------------------------------------------\n");

int weight_xiU=0;
// Loop: iterate over a range/collection.
for (int i = 0; i < cols.size(); i++) {
  // Loop: iterate over a range/collection.
  for(int l=0;l<logGF;l++){
    // Conditional branch.
    if((BINGF[xiU_J[i]][l])){
      weight_xiU++;
    }
  }
}
printf("weight_xiU=%d\n",weight_xiU);

auto res = solveLinearEquations(A, RHS_b, ADDGF, MULGF, DIVGF);
vector<int> solution = res.second;
// Conditional branch.
if(res.first){
  printf("A solution exists\n");

  printf("A=\n");
  // Loop: iterate over a range/collection.
  for (int i = 0; i < A.size(); i++) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < A[0].size(); j++) {
      printf("%3x",A[i][j]);
    }
    printf("\n");
  }

  printf("solution=\n");
  // Loop: iterate over a range/collection.
  for(int i=0;i<cols.size();i++){printf("%3x",res.second[i]);}printf("\n");

  int weight_solution=0;
  // Loop: iterate over a range/collection.
  for (int i = 0; i < solution.size(); i++) {
    // Loop: iterate over a range/collection.
    for(int l=0;l<logGF;l++){
      // Conditional branch.
      if((BINGF[solution[i]][l])){
        weight_solution++;
      }
    }
  }
  printf("weight_solution=%d\n",weight_solution);

  vector<int> x = res.second;
  // Loop: iterate over a range/collection.
  for(int k=0;k<cols.size();k++){
    int j=cols[k];
    EstmNoise[j]=x[k];
  }

  vector<int> verify_b(rows.size(), 0);
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < rows.size(); i++) {
    // Loop: iterate over a range/collection.
    for (size_t j = 0; j < cols.size(); j++) {
      int t=MULGF[A[i][j]][xiU_J[j]];
      verify_b[i] = ADDGF[verify_b[i]][t];
    }
  }

  printf("verify_b=\n");
  // Loop: iterate over a range/collection.
  for(int i=0;i<verify_b.size();i++){printf("%3x",verify_b[i]);}printf("\n");
  printf("***decode_small_errors_from_rows_cols %d\n", res.first);
}else{
printf("No solution exists\n");
printf("***decode_small_errors_from_rows_cols %d\n", res.first);
}
}
// Function: decode_small_errors
// Purpose: TODO - describe the function's responsibility succinctly.

void decode_small_errors(vector<int>& TrueNoiseSynd, vector<int>& TrueNoise_C, int M, int N, vector<int>& EstmNoise, vector<int>& RowDeg, vector<int>& Candidate_Covering_Normal_Rows_D, vector<vector<int>>& UTCBC_Rows_C_orthogonal_D, vector<vector<int>>& UTCBC_Cols_C_orthogonal_D, vector<vector<int>>& IatJ_C, vector<vector<int>>& JatI_C, vector<vector<int>>& Mat, vector<vector<int>>& MatValue, vector<vector<int>>& MULGF, vector<vector<int>>& ADDGF, vector<vector<int>>& DIVGF, vector<vector<int>>& BINGF){
  printf("@@@decode_small_errors\n");

  set<int> rows_set,cols_set;
  printf("Candidate_Covering_Normal_Rows_D(%d):",Candidate_Covering_Normal_Rows_D.size());for(int i:Candidate_Covering_Normal_Rows_D){printf("%5d ",i);}printf("\n");
  // Loop: iterate over a range/collection.
  for (int i : Candidate_Covering_Normal_Rows_D) {
    // Loop: iterate over a range/collection.
    for (int ii : UTCBC_Rows_C_orthogonal_D[i]) {
      rows_set.insert(ii);
    }
    // Loop: iterate over a range/collection.
    for (int jj : UTCBC_Cols_C_orthogonal_D[i]) {
      cols_set.insert(jj);
    }
  }

  vector<int> rows(rows_set.begin(), rows_set.end());
  vector<int> cols(cols_set.begin(), cols_set.end());
  decode_small_errors_from_rows_cols(TrueNoiseSynd, TrueNoise_C, M, N, EstmNoise, RowDeg, rows, cols, IatJ_C, JatI_C, Mat, MatValue, MULGF, ADDGF, DIVGF, BINGF);
  printf("***decode_small_errors end\n");
}
// Function: gcd
// Purpose: TODO - describe the function's responsibility succinctly.

int gcd(int a, int b){
  int c;
  // Loop: repeat while condition holds.
  while (b != 0)    {
    c = a % b;
    a = b;
    b = c;
  }
  return a;
}
// Function: inv
// Purpose: TODO - describe the function's responsibility succinctly.

int inv(int x, int P){
  int a=x/P;
  int b=x%P;
  int aa=inv_ZP[a];
  int bb=((P-b)*inv_ZP[a])%P;
  return aa*P+bb;
}
// Function: mult
// Purpose: TODO - describe the function's responsibility succinctly.

int mult(int x, int y, int P){

  int a=x/P;
  int b=x%P;
  int aa=y/P;
  int bb=y%P;
  int aaa=(a*aa)%P;
  int bbb=(a*bb+b)%P;
  return aaa*P+bbb;
}
// Function: mult_apm
// Purpose: TODO - describe the function's responsibility succinctly.

int mult_apm(int x,int y,int P){
  return mult(x,y,P);
}
// Function: print_apm
// Purpose: TODO - describe the function's responsibility succinctly.

void print_apm(int x,int P){
  // Conditional branch.
  if(x)
  cout << setw(log10(P)+1) << x/P << "X+" << setw(log10(P)+1) << x%P << " ";
  else
  cout << setw(log10(P)+1) << "" << "  " << setw(log10(P)+1) << "" << " ";
}
// Function: inv_apm
// Purpose: TODO - describe the function's responsibility succinctly.

int inv_apm(int x, int P){

  int a=x/P;
  int b=x%P;
  int aa=inv_ZP[a];
  int bb=((P-b)*inv_ZP[a])%P;
  // Conditional branch.
  if(aa<0 || bb<0){
    print_apm(aa*P+bb,P);exit(0);
  }
  return aa*P+bb;
}
// Function: extended_gcd
// Purpose: TODO - describe the function's responsibility succinctly.

int extended_gcd(int a, int b, int &x, int &y) {
  // Conditional branch.
  if (a == 0) {
    x = 0;
    y = 1;
    return b;
  }
  int x1, y1;
  int gcd = extended_gcd(b % a, a, x1, y1);
  x = y1 - (b / a) * x1;
  y = x1;
  return gcd;
}
// Function: mod_inverse
// Purpose: TODO - describe the function's responsibility succinctly.

int mod_inverse(int a, int P) {
  int x, y;
  int gcd = extended_gcd(a, P, x, y);
  // Conditional branch.
  if (gcd != 1) {
    throw invalid_argument("No modular inverse exists");
  } else {
  return (x % P + P) % P;
}
}
// Function: construct_inv_ZP
// Purpose: TODO - describe the function's responsibility succinctly.

void construct_inv_ZP(int P){

  cout << "constructing inv_ZP" << endl;
  inv_ZP= vector<int>(P);
  inv_ZP[1]=1;
  // Loop: iterate over a range/collection.
  for(int a=0;a<P;a++){
    // Conditional branch.
    if(gcd(a,P)==1){
      inv_ZP[a] = mod_inverse(a, P);

    }
  }
  cout << "Verify that z = x * invx equals 1 in Z_P." << endl;
  // Loop: iterate over a range/collection.
  for(unsigned int x=0;x<P;x++){

    // Conditional branch.
    if(gcd(x/P,P)==1){
      int invx=inv_apm(x,P);
      int z=mult(x,invx,P);
      // Conditional branch.
      if(z!=P){
        cout << "error" << endl;
        cout<< x << " " << invx << " " << z << endl;
        exit(0);
      }
    }
  }
  cout << "ok" << endl;
}
// Function: commute
// Purpose: TODO - describe the function's responsibility succinctly.

bool commute(int x, int y, int P){
  int a=x/P;
  int b=x%P;
  int c=y/P;
  int d=y%P;
  return ((c*b+d)%P==(a*d+b)%P);
}
// Function: print_commute_matrix_ff_gg
// Purpose: TODO - describe the function's responsibility succinctly.

void print_commute_matrix_ff_gg(vector<int>& ff,vector<int>& gg,int P){
  cout << "@print_commute_matrix_ff_gg" << endl;
  cout << "commute matrix ff,ff" << endl;
  // Loop: iterate over a range/collection.
  for(int i=0;i<ff.size();i++){
    // Loop: iterate over a range/collection.
    for(int j=0;j<ff.size();j++){
      cout << commute(ff[i],ff[j],P);
    }
    cout << endl;
  }
  cout << "commute matrix ff,gg" << endl;
  // Loop: iterate over a range/collection.
  for(int i=0;i<ff.size();i++){
    // Loop: iterate over a range/collection.
    for(int j=0;j<gg.size();j++){
      cout << commute(ff[i],gg[j],P);
    }
    cout << endl;
  }
  cout << "commute matrix gg,gg" << endl;
  // Loop: iterate over a range/collection.
  for(int i=0;i<gg.size();i++){
    // Loop: iterate over a range/collection.
    for(int j=0;j<gg.size();j++){
      cout << commute(gg[i],gg[j],P);
    }
    cout << endl;
  }
}
// Function: print_ff_gg
// Purpose: TODO - describe the function's responsibility succinctly.

void print_ff_gg(vector<int> ff,vector<int> gg,int P){
  cout << "ff=";for(int i=0;i<ff.size();i++){cout << setw(5) << ff[i]/P << "X+" << setw(5) << ff[i]%P << " ";}cout << endl;
  cout << "gg=";for(int i=0;i<gg.size();i++){cout << setw(5) << gg[i]/P << "X+" << setw(5) << gg[i]%P << " ";}cout << endl;
}
// Function: print_Hc_pair
// Purpose: TODO - describe the function's responsibility succinctly.

void print_Hc_pair(vector<vector<int>> HcA,vector<vector<int>>  HcB, int P){

  cout << "HcA=" << endl;
  // Loop: iterate over a range/collection.
  for(int j=0;j<HcA.size();j++){
    // Loop: iterate over a range/collection.
    for(int l=0;l<HcA[0].size();l++){
      print_apm(HcA[j][l],P);

    }
    cout << endl;
  }
  cout << "HcB=" << endl;
  // Loop: iterate over a range/collection.
  for(int j=0;j<HcB.size();j++){
    // Loop: iterate over a range/collection.
    for(int l=0;l<HcB[0].size();l++){
      print_apm(HcB[j][l],P);

    }
    cout << endl;
  }
}
// Function: construct_HcA_HcB_from_ff_gg
// Purpose: TODO - describe the function's responsibility succinctly.

void construct_HcA_HcB_from_ff_gg(vector<vector<int>> &HcA,vector<vector<int>> &HcB,vector<int> ff,vector<int> gg, int J, int L, int P){
  int L2=L/2;

  HcA=vector<vector<int>>(J);
  HcB=vector<vector<int>>(J);
  // Loop: iterate over a range/collection.
  for(int j=0;j<J;j++){HcA[j]=vector<int>(L,0);}
  // Loop: iterate over a range/collection.
  for(int j=0;j<J;j++){HcB[j]=vector<int>(L,0);}
  // Loop: iterate over a range/collection.
  for(int i=0;i<J;i++){
    // Loop: iterate over a range/collection.
    for(int l=0;l<L/2;l++){
      int j=i;
      HcA[i][l]    =ff[(l-j+L/2)%(L/2)];
      HcA[i][l+L/2]=gg[(l-j+L/2)%(L/2)];
    }
  }
  // Loop: iterate over a range/collection.
  for(int i=0;i<J;i++){
    // Loop: iterate over a range/collection.
    for(int l=0;l<L/2;l++){
      int k=i;
      HcB[i][l]    =inv(gg[(k-l+L/2)%(L/2)],P);
      HcB[i][l+L/2]=inv(ff[(k-l+L/2)%(L/2)],P);
    }
  }

  print_Hc_pair(HcA,HcB,P);
  cout << "Verify orthogonality of HcA and HcB" << endl;
  // Loop: iterate over a range/collection.
  for(int k=0;k<J;k++){
    // Loop: iterate over a range/collection.
    for(int j=0;j<J;j++){
      // Loop: iterate over a range/collection.
      for(int i=0;i<L/2;i++){
        cout << setw(3) << mult(ff[(i-j+L2)%L2],gg[(k-i+L2)%L2],P) << " ";
      }
      cout << "|";
      // Loop: iterate over a range/collection.
      for(int i=0;i<L/2;i++){
        cout << setw(3) << mult(gg[(i-j+L2)%L2],ff[(k-i+L2)%L2],P) << " ";
      }
      cout << endl;
    }
  }
}
// Function: make_JatI_IatJ
// Purpose: TODO - describe the function's responsibility succinctly.

void make_JatI_IatJ(
vector<vector<int>> Hc,
vector<vector<int>>& JatI,
vector<vector<int>>& IatJ,
int J,int L, int P
){
  JatI = vector<vector<int>>(P*Hc.size());
  IatJ = vector<vector<int>>(P*Hc[0].size());

  // Loop: iterate over a range/collection.
  for(int j=0;j<Hc.size();j++){

    // Loop: iterate over a range/collection.
    for(int l=0;l<Hc[0].size();l++){
      int a=Hc[j][l]/P;
      int b=Hc[j][l]%P;
      // Loop: iterate over a range/collection.
      for(int y=0;y<P;y++){
        int x=(a*y+b)%P;
        JatI[j*P+x].push_back(l*P+y);
        IatJ[l*P+y].push_back(j*P+x);
      }
    }
  }
}
// Function: make_full_JatI_IatJ
// Purpose: TODO - describe the function's responsibility succinctly.

void make_full_JatI_IatJ(
vector<vector<int>>& full_JatI_A,
vector<vector<int>>& full_IatJ_A,
vector<vector<int>>& full_JatI_B,
vector<vector<int>>& full_IatJ_B,
vector<int>& ff,
vector<int>& gg,
int P){
  int L2=ff.size();
  int L=L2*2;
  vector<vector<int>> HcA, HcB;
  construct_HcA_HcB_from_ff_gg(HcA,HcB,ff,gg, L2, L,P);

  make_JatI_IatJ(HcA, full_JatI_A, full_IatJ_A, L2, L, P);
  make_JatI_IatJ(HcB, full_JatI_B, full_IatJ_B, L2, L, P);

}
// Function: load_matrix
// Purpose: TODO - describe the function's responsibility succinctly.

void load_matrix(const char* MatrixFilePrefix_X, int& N_X, int& M_X, int P_X, int L, vector<vector<int>>& Mat_X, vector<vector<int>>& MatValue_X, vector<int>& ColDeg_X, vector<int>& RowDeg_X, vector<vector<int>>& JatI_X, vector<vector<int>>& IatJ_X, vector<vector<int>>& full_JatI_X, vector<vector<int>>& full_IatJ_X) {
  char FileName[500];
  FILE* f;

  ColDeg_X = vector<int>(N_X);
  RowDeg_X = vector<int>(M_X);
  strcpy(FileName, MatrixFilePrefix_X);
  strcat(FileName, "_row");
  f = fopen(FileName, "r");
  // Loop: iterate over a range/collection.
  for (size_t m = 0; m < M_X; m++) {
    fscanf(f, "%d", &RowDeg_X[m]);
  }
  fclose(f);

  Mat_X = vector<vector<int>>(M_X);
  // Loop: iterate over a range/collection.
  for (size_t m = 0; m < M_X; m++) {
    Mat_X[m] = vector<int>(RowDeg_X[m]);
  }

  strcpy(FileName, MatrixFilePrefix_X);
  f = fopen(FileName, "r");
  // Loop: iterate over a range/collection.
  for (size_t m = 0; m < M_X; m++) {
    // Loop: iterate over a range/collection.
    for (size_t k = 0; k < RowDeg_X[m]; k++) {
      fscanf(f, "%d", &Mat_X[m][k]);
    }
  }
  fclose(f);

  JatI_X = vector<vector<int>>(M_X);
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < M_X; i++) {
    // Loop: iterate over a range/collection.
    for (size_t k = 0; k < RowDeg_X[i]; k++) {
      JatI_X[i].push_back(Mat_X[i][k]);
    }
  }
  IatJ_X = vector<vector<int>>(N_X);
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < M_X; i++) {
    // Loop: iterate over a range/collection.
    for (size_t k = 0; k < JatI_X[i].size(); k++) {
      int j = JatI_X[i][k];
      IatJ_X[j].push_back(i);
    }
  }
  int P=P_X;
  printf("P=%d L=%d\n",P,L);
  vector<int> ff(L/2);
  // Loop: iterate over a range/collection.
  for(int l=0;l<L/2;l++){
    vector<int> a(L/2);
    vector<int> b(L/2);
    int j0=0;
    int i0=IatJ_X[P*l+j0][0];

    b[l]=i0;
    int j1=1;
    int i1=IatJ_X[P*l+j1][0];

    a[l]=(i1-b[l]+P)%P;
    ff[l]=a[l]*P+b[l];
  }
  vector<int> gg(L/2);
  // Loop: iterate over a range/collection.
  for(int l=0;l<L/2;l++){
    vector<int> a(L/2);
    vector<int> b(L/2);
    int j0=0;
    int i0=IatJ_X[P*(l+L/2)+j0][0];

    b[l]=i0;
    int j1=1;
    int i1=IatJ_X[P*(l+L/2)+j1][0];

    a[l]=(i1-b[l]+P)%P;
    gg[l]=a[l]*P+b[l];
  }
  print_ff_gg(ff,gg,P);

  strcpy(FileName, MatrixFilePrefix_X);
  strcat(FileName, "_blockcycleminusone");
  f = fopen(FileName, "r");
  // Conditional branch.
  if (f == NULL) {
    printf("Error: File %s not found.\n", FileName);
  }else{
  full_JatI_X = vector<vector<int>>(M_X+P_X);
  full_IatJ_X = vector<vector<int>>(N_X);
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < M_X; i++) {
    // Loop: iterate over a range/collection.
    for(size_t j:JatI_X[i]){
      full_JatI_X[i].push_back(j);
      full_IatJ_X[j].push_back(i);
    }
  }
  // Loop: iterate over a range/collection.
  for (size_t i = M_X; i < M_X+P_X; i++) {
    // Loop: iterate over a range/collection.
    for (size_t k = 0; k < L; k++) {
      int j;
      fscanf(f, "%d", &j);
      full_JatI_X[i].push_back(j);
    }
  }
  // Loop: iterate over a range/collection.
  for(int i=0;i<M_X+P_X;i++){
    std::sort(full_JatI_X[i].begin(), full_JatI_X[i].end());
  }
  // Loop: iterate over a range/collection.
  for(int j=0;j<N_X;j++){
    std::sort(full_IatJ_X[j].begin(), full_IatJ_X[j].end());
  }
  printf("Success: File %s loaded.\n", FileName);
  fclose(f);
}
printf("000\n");

// Loop: iterate over a range/collection.
for (size_t m = 0; m < M_X; m++) {
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < RowDeg_X[m]; i++) {
    ColDeg_X[Mat_X[m][i]]++;
  }
}
printf("001\n");

MatValue_X = vector<vector<int>>(M_X);
// Loop: iterate over a range/collection.
for (size_t m = 0; m < M_X; m++) {
  MatValue_X[m] = vector<int>(RowDeg_X[m]);
}
printf("002\n");
strcpy(FileName, MatrixFilePrefix_X);
strcat(FileName, "_value");
f = fopen(FileName, "r");
// Loop: iterate over a range/collection.
for (size_t m = 0; m < M_X; m++) {
  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < RowDeg_X[m]; i++) {
    fscanf(f, "%d", &MatValue_X[m][i]);
  }
}
fclose(f);
printf("003\n");
}
// Function: initializeUTCBC_Rows
// Purpose: TODO - describe the function's responsibility succinctly.
void initializeUTCBC_Rows(int M, int P,
vector<vector<int>>& UTCBC_Rows_C_orthogonal_D,
vector<vector<int>>& UTCBC_Rows_D_orthogonal_C,
vector<vector<int>>& full_JatI_C, vector<vector<int>>& IatJ_D,
vector<vector<int>>& full_JatI_D, vector<vector<int>>& IatJ_C
) {
  printf("@@@initializeUTCBC_Rows\n");

  UTCBC_Rows_C_orthogonal_D = vector<vector<int>>(M+P);
  // Loop: iterate over a range/collection.
  for (int iD = 0; iD < M+P; iD++) {
    // Loop: iterate over a range/collection.
    for (int j : full_JatI_D[iD]) {
      // Loop: iterate over a range/collection.
      for (int iC : IatJ_C[j]) {
        addIfNotIncluded(UTCBC_Rows_C_orthogonal_D[iD], iC);
      }
    }
    std::sort(UTCBC_Rows_C_orthogonal_D[iD].begin(), UTCBC_Rows_C_orthogonal_D[iD].end());
  }

  UTCBC_Rows_D_orthogonal_C = vector<vector<int>>(M+P);
  // Loop: iterate over a range/collection.
  for (int iC = 0; iC < M+P; iC++) {
    // Loop: iterate over a range/collection.
    for (int j : full_JatI_C[iC]) {
      // Loop: iterate over a range/collection.
      for (int iD : IatJ_D[j]) {
        addIfNotIncluded(UTCBC_Rows_D_orthogonal_C[iC], iD);
      }
    }
    std::sort(UTCBC_Rows_D_orthogonal_C[iC].begin(), UTCBC_Rows_D_orthogonal_C[iC].end());
  }
  printf("***initializeUTCBC_Rows***\n");
}
// Function: initializeUTCBC_Cols
// Purpose: TODO - describe the function's responsibility succinctly.
void initializeUTCBC_Cols(int M, int P,
vector<vector<int>>& UTCBC_Cols_C_orthogonal_D,
vector<vector<int>>& UTCBC_Cols_D_orthogonal_C,
vector<vector<int>>& full_JatI_C,
vector<vector<int>>& full_JatI_D
) {
  printf("@@@initializeUTCBC_Cols\n");

  UTCBC_Cols_D_orthogonal_C = vector<vector<int>>(M+P);
  // Loop: iterate over a range/collection.
  for (int iC = 0; iC < M+P; iC++){
    UTCBC_Cols_D_orthogonal_C[iC]=full_JatI_C[iC];
  }

  UTCBC_Cols_C_orthogonal_D = vector<vector<int>>(M+P);
  // Loop: iterate over a range/collection.
  for (int iD = 0; iD < M+P; iD++) {
    UTCBC_Cols_C_orthogonal_D[iD]=full_JatI_D[iD];
  }
  printf("***initializeUTCBC_Cols***\n");
}
// Function: initialize_decoding_arrays
// Purpose: TODO - describe the function's responsibility succinctly.

void initialize_decoding_arrays(int N, int logGF, int NumEdge_C, int GF,
vector<int>& TrueNoise_C,
vector<vector<double>>& CNtoVNxxx_C,
vector<vector<double>>& VNtoCNxxx_C,
vector<vector<double>>& ChNtoVN_CD,
vector<vector<double>>& APP_C,
vector<int>& EstmNoise_C,
vector<vector<double>>& VNtoChN_CD) {
  cout << "@@@ initialize_decoding_arrays" << endl;
  TrueNoise_C.resize(N);
  CNtoVNxxx_C.resize(NumEdge_C);
  // Loop: iterate over a range/collection.
  for(size_t l=0;l<NumEdge_C;l++) { CNtoVNxxx_C[l].resize(GF); }
  VNtoCNxxx_C.resize(NumEdge_C);
  // Loop: iterate over a range/collection.
  for(size_t l=0;l<NumEdge_C;l++) { VNtoCNxxx_C[l].resize(GF); }
  ChNtoVN_CD.resize(N);
  // Loop: iterate over a range/collection.
  for(size_t n=0;n<N;n++) { ChNtoVN_CD[n].resize(GF); }
  APP_C.resize(N);
  // Loop: iterate over a range/collection.
  for(size_t n=0;n<N;n++) { APP_C[n].resize(GF); }
  EstmNoise_C.resize(N);
  VNtoChN_CD.resize(N);
  // Loop: iterate over a range/collection.
  for (size_t n = 0; n < N; n++) {
    VNtoChN_CD[n].resize(GF);
  }
  cout << "*** initialize_decoding_arrays" << endl;
}
// Function: initialize_interleaver
// Purpose: TODO - describe the function's responsibility succinctly.

void initialize_interleaver(int N, int M, int*& ColDeg_C, int*& RowDeg_C, int**& Mat_C, int**& NtoB_C, vector<int>& Interleaver_C, int& NumEdge_C) {
  cout << "@@@ initialize_interleaver" << endl;
  vector<int> ind(N, 0);
  NumEdge_C = 0;
  // Loop: iterate over a range/collection.
  for(size_t m = 0; m < M; m++) { NumEdge_C += RowDeg_C[m]; }
  NtoB_C = (int**)calloc(N, sizeof(int*));
  // Loop: iterate over a range/collection.
  for(size_t n = 0; n < N; n++) { NtoB_C[n] = (int*)calloc(ColDeg_C[n], sizeof(int)); }
  Interleaver_C.resize(NumEdge_C);
  int e = 0;
  // Loop: iterate over a range/collection.
  for(size_t m = 0; m < M; m++) {
    // Loop: iterate over a range/collection.
    for(size_t k = 0; k < RowDeg_C[m]; k++) {
      int n = Mat_C[m][k];
      NtoB_C[n][ind[n]++] = e++;
    }
  }
  e = 0;
  // Loop: iterate over a range/collection.
  for(size_t n = 0; n < N; n++) {
    // Loop: iterate over a range/collection.
    for(size_t k = 0; k < ColDeg_C[n]; k++) {
      Interleaver_C[e++] = NtoB_C[n][k];
    }
  }
  // removed unsafe free on vector memory
  cout << "*** initialize_interleaver" << endl;
}
// Function: initialize_interleaver
// Purpose: TODO - describe the function's responsibility succinctly.

void initialize_interleaver(int N, int M, vector<int>& ColDeg_C, vector<int>& RowDeg_C, vector<vector<int>>& Mat_C, vector<vector<int>>& NtoB_C, vector<int>& Interleaver_C, int& NumEdge_C) {
  cout << "@@@ initialize_interleaver" << endl;
  vector<int> ind(N, 0);
  NumEdge_C = 0;
  // Loop: iterate over a range/collection.
  for(size_t m = 0; m < M; m++) { NumEdge_C += RowDeg_C[m]; }
  NtoB_C = vector<vector<int>>(N);
  // Loop: iterate over a range/collection.
  for(size_t n = 0; n < N; n++) { NtoB_C[n] = vector<int>(ColDeg_C[n]); }
  Interleaver_C.resize(NumEdge_C);
  int e = 0;
  // Loop: iterate over a range/collection.
  for(size_t m = 0; m < M; m++) {
    // Loop: iterate over a range/collection.
    for(size_t k = 0; k < RowDeg_C[m]; k++) {
      int n = Mat_C[m][k];
      NtoB_C[n][ind[n]++] = e++;
    }
  }
  e = 0;
  // Loop: iterate over a range/collection.
  for(size_t n = 0; n < N; n++) {
    // Loop: iterate over a range/collection.
    for(size_t k = 0; k < ColDeg_C[n]; k++) {
      Interleaver_C[e++] = NtoB_C[n][k];
    }
  }
  cout << "*** initialize_interleaver" << endl;
}
// Function: initialize_syndrome_and_channel
// Purpose: TODO - describe the function's responsibility succinctly.

void initialize_syndrome_and_channel(int M, vector<int>& TrueNoiseSynd_C, vector<int>& EstmNoiseSynd_C) {
  cout << "@@@ initialize_syndrome_and_channel" << endl;
  TrueNoiseSynd_C = vector<int>(M, 0);
  EstmNoiseSynd_C = vector<int>(M, 0);
  cout << "*** initialize_syndrome_and_channel" << endl;
}
// Function: load_GF_tables
// Purpose: TODO - describe the function's responsibility succinctly.

void load_GF_tables(
int GF, int logGF,
vector<vector<int>>& BINGF, vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF, vector<vector<int>>& DIVGF, vector<vector<int>>& FFTSQ) {
  char FileName[100], name[10];
  FILE* f;
  BINGF = vector<vector<int>>(GF, vector<int>(logGF));
  ADDGF = vector<vector<int>>(GF, vector<int>(GF));
  MULGF = vector<vector<int>>(GF, vector<int>(GF));
  DIVGF = vector<vector<int>>(GF, vector<int>(GF));
  FFTSQ = vector<vector<int>>(logGF * GF / 2, vector<int>(2));
  // Loop: iterate over a range/collection.
  for (size_t k = 0; k < GF; k++) {
    BINGF[k] = vector<int>(logGF);
    ADDGF[k] = vector<int>(GF);
    MULGF[k] = vector<int>(GF);
    DIVGF[k] = vector<int>(GF);
  }
  // Loop: iterate over a range/collection.
  for (size_t k = 0; k < logGF * GF / 2; k++) {
    FFTSQ[k] = vector<int>(2);
  }
  sprintf(name, "%d", GF);
  std::vector<std::pair<const char*, vector<vector<int>>*>> files = {
    {"data/Tables/BINGF", &BINGF},
    {"data/Tables/ADDGF", &ADDGF},
    {"data/Tables/MULGF", &MULGF},
    {"data/Tables/DIVGF", &DIVGF}
  };
  // Loop: iterate over a range/collection.
  for (auto& file : files) {
    strcpy(FileName, file.first);
    strcat(FileName, name);
    f = fopen(FileName, "r");
    // Loop: iterate over a range/collection.
    for (size_t k = 0; k < GF; k++) {
      // Loop: iterate over a range/collection.
      for (size_t n = 0; n < ((file.first == "data/Tables/BINGF") ? logGF : GF); n++) {
        fscanf(f, "%d", &(*file.second)[k][n]);
      }
    }
    fclose(f);
  }
  strcpy(FileName, "data/Tables/TENSORFFT");
  strcat(FileName, name);
  f = fopen(FileName, "r");
  // Loop: iterate over a range/collection.
  for (size_t k = 0; k < logGF * GF / 2; k++) {
    fscanf(f, "%d", &FFTSQ[k][0]);
    fscanf(f, "%d", &FFTSQ[k][1]);
  }
  fclose(f);
  cout << "GF table loaded" << endl;
}
// Function: load_transpose_GF_tables
// Purpose: TODO - describe the function's responsibility succinctly.

void load_transpose_GF_tables(
int GF, int logGF,
vector<vector<int>>& BINGF, vector<vector<int>>& FFTSQ,
vector<vector<int>>& TBINGF, vector<vector<int>>& TFFTSQ)
{

  TBINGF = vector<vector<int>>(GF, vector<int>(logGF));
  TFFTSQ = vector<vector<int>>(logGF * GF / 2, vector<int>(2));

  vector<vector<vector<int>>> C(GF, vector<vector<int>>(logGF, vector<int>(logGF, 0)));
  vector<vector<vector<int>>> TC(GF, vector<vector<int>>(logGF, vector<int>(logGF, 0)));
  // Loop: iterate over a range/collection.
  for (size_t A = 0; A < GF; A++) {
    // Loop: iterate over a range/collection.
    for (size_t e = 1; e <= logGF; e++) {
      int Ae = MULGF[A][e];
      // Loop: iterate over a range/collection.
      for (size_t j = 0; j < logGF; j++) {
        C[A][j][e - 1] = BINGF[Ae][j];
      }
    }
  }
  // Loop: iterate over a range/collection.
  for (size_t A = 0; A < GF; A++) {
    // Loop: iterate over a range/collection.
    for (size_t i = 0; i < logGF; i++) {
      // Loop: iterate over a range/collection.
      for (size_t j = 0; j < logGF; j++) {
        TC[A][i][j] = C[A][j][i];
      }
    }
  }

  // Loop: iterate over a range/collection.
  for (size_t g = 0; g < GF; g++) {
    // Loop: iterate over a range/collection.
    for (size_t l = 0; l < logGF; l++) {
      TBINGF[g][l] = TC[g][l][0];
    }
  }

  // Loop: iterate over a range/collection.
  for (size_t i = 0; i < logGF * GF / 2; i++) {
    TFFTSQ[i][0] = GF2GF(FFTSQ[i][0], GF, logGF, BINGF, TBINGF);
    TFFTSQ[i][1] = GF2GF(FFTSQ[i][1], GF, logGF, BINGF, TBINGF);
  }
  cout << "GF transpose table loaded" << endl;
}
#include <ctime>
// Function: simulateTransmissionErrors
// Purpose: TODO - describe the function's responsibility succinctly.

void simulateTransmissionErrors(int N,  int logGF, int GF, double pD,
vector<int>& TrueNoise_C, vector<int>& TrueNoise_D,
vector<vector<int>>& BINGF, vector<vector<int>>& TBINGF, int &num_X, int &num_Z) {
  num_X = 0;
  num_Z = 0;
  // Loop: iterate over a range/collection.
  for (size_t n = 0; n < N; n++) {
    vector<int> ZBIN_C(logGF,0);
    vector<int> ZBIN_D(logGF,0);
    // Loop: iterate over a range/collection.
    for (size_t i = 0; i < logGF; i++) {
      // Conditional branch.
      if (drand48() < 1.0 - pD) {
        ZBIN_C[i] = 0;
        ZBIN_D[i] = 0;
      } else {
      int r = lrand48() % 3;
      // Conditional branch.
      if (r == 0) {
        num_X++;
        ZBIN_C[i] = 1;
        ZBIN_D[i] = 0;
      } else if (r == 1) {
      num_Z++;
      ZBIN_C[i] = 0;
      ZBIN_D[i] = 1;
    } else if (r == 2) {
    num_X++;
    num_Z++;
    ZBIN_C[i] = 1;
    ZBIN_D[i] = 1;
  }
}
}
std::vector<int> s_C(logGF);
std::vector<int> s_D(logGF);
// Loop: iterate over a range/collection.
for (size_t i = 0; i < logGF; i++) {
  s_C[i] = ZBIN_C[i];
  s_D[i] = ZBIN_D[i];
}
TrueNoise_C[n] = Bin2GF(s_C, GF, logGF, BINGF);
TrueNoise_D[n] = Bin2GF(s_D, GF, logGF, TBINGF);
}
}
// Function: extractValueFromFilename
// Purpose: TODO - describe the function's responsibility succinctly.

int extractValueFromFilename(const std::string& filename, const std::string& pattern_char) {
  int value = 0;
  std::regex pattern("_"+pattern_char+"(\\d+)");
  std::smatch match;

  // Conditional branch.
  if (std::regex_search(filename, match, pattern)) {
    value = std::stoi(match[1].str());
  }
  return value;
}
// Function: DecodeIteration
// Purpose: TODO - describe the function's responsibility succinctly.

void DecodeIteration(
int &SyndromeIsSatisfied,
vector<vector<double>>& VNtoCNxxx, vector<vector<double>>& CNtoVNxxx,
vector<vector<double>>& VNtoChN, vector<vector<double>>& ChNtoVN,
vector<vector<double>>& APP,
vector<int>& Interleaver, vector<int>& ColDeg,
int N, int M, int GF, int logGF,
vector<vector<int>>& MatValue, vector<int>& RowDeg,
vector<int> &TrueNoiseSynd, vector<int>& Synd,
vector<int> &USS,
vector<int> &UpdatedDecision,
vector<int> &EstmNoise,
vector<vector<int>> Mat,
vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF,
vector<vector<int>>& DIVGF, vector<vector<int>>& FFTSQ){

  DataPass(VNtoCNxxx, CNtoVNxxx, VNtoChN, Interleaver, ColDeg, N, GF);
  CheckPass(CNtoVNxxx, VNtoCNxxx, MatValue, M, RowDeg, MULGF, DIVGF, FFTSQ, GF, TrueNoiseSynd);
  ComputeAPP(VNtoChN, ChNtoVN, CNtoVNxxx, VNtoChN, Interleaver, ColDeg, N, GF);

  Decision(EstmNoise, UpdatedDecision, VNtoChN, N, GF);
  SyndromeIsSatisfied = IsSyndromeSatisfied(TrueNoiseSynd, Synd, USS, M, EstmNoise, MatValue, RowDeg, ADDGF, MULGF, Mat);
  // Conditional branch.
  if (SyndromeIsSatisfied) {
    // Loop: iterate over a range/collection.
    for (int n = 0; n < N; n++) {
      // Loop: iterate over a range/collection.
      for (int g = 0; g < GF; g++) { ChNtoVN[n][g] = 0; }
      int g_hat = EstmNoise[n];
      ChNtoVN[n][g_hat] = 1;
    }
  }
}
// Function: load_size
// Purpose: TODO - describe the function's responsibility succinctly.

bool load_size(const char* baseFileName, int& M, int& N, int& GF,  int& logGF) {
  char FileName[256];
  sprintf(FileName, "%s_size", baseFileName);
  FILE* f = fopen(FileName, "r");
  // Conditional branch.
  if (!f) {
    perror("Failed to open size file");
    return false;
  }
  fscanf(f, "%d", &M);
  fscanf(f, "%d", &N);
  fscanf(f, "%d", &GF);
  fclose(f);
  logGF = rint(log2(GF));
  return true;
}
// Function: check_code_parameters_equal
// Purpose: TODO - describe the function's responsibility succinctly.

void check_code_parameters_equal(const char* MatrixFilePrefix_C, const char* MatrixFilePrefix_D, int &M, int &N, int &GF, int &logGF){
  int M_C, N_C,  GF_C, logGF_C;
  int M_D, N_D,  GF_D, logGF_D;

  bool okC = load_size(MatrixFilePrefix_C, M_C, N_C, GF_C,  logGF_C);

  bool okD = load_size(MatrixFilePrefix_D, M_D, N_D, GF_D,  logGF_D);
  // Conditional branch.
  if (!okC || !okD) {
    cerr << "Failed to read the size file." << endl;
    exit(1);
  }

  assert(GF_C == GF_D);
  assert(logGF_C == logGF_D);
  assert(N_C == N_D);
  assert(M_C == M_D);
  N=N_C;
  M=M_C;
  GF=GF_C;
  logGF=logGF_C;
}
// Function: extract_ff_gg
// Purpose: TODO - describe the function's responsibility succinctly.
void extract_ff_gg(vector<int>& ff,vector<int>& gg,
vector<vector<int>>& JatI_X,
vector<vector<int>>& IatJ_X,
vector<vector<int>>& JatI_Z,
vector<vector<int>>& IatJ_Z,
int L, int P){
  printf("P=%d L=%d\n",P,L);
  ff=vector<int>(L/2);
  // Loop: iterate over a range/collection.
  for(int l=0;l<L/2;l++){
    vector<int> a(L/2);
    vector<int> b(L/2);
    int j0=0;
    int i0=IatJ_X[P*l+j0][0];

    b[l]=i0;
    int j1=1;
    int i1=IatJ_X[P*l+j1][0];

    a[l]=(i1-b[l]+P)%P;
    ff[l]=a[l]*P+b[l];
  }
  gg=vector<int>(L/2);
  // Loop: iterate over a range/collection.
  for(int l=0;l<L/2;l++){
    vector<int> a(L/2);
    vector<int> b(L/2);
    int j0=0;
    int i0=IatJ_X[P*(l+L/2)+j0][0];

    b[l]=i0;
    int j1=1;
    int i1=IatJ_X[P*(l+L/2)+j1][0];

    a[l]=(i1-b[l]+P)%P;
    gg[l]=a[l]*P+b[l];
  }
  print_ff_gg(ff,gg,P);
  print_commute_matrix_ff_gg(ff,gg,P);
}
// Function: printDecodingDebugInfo
// Purpose: TODO - describe the function's responsibility succinctly.
void printDecodingDebugInfo(
bool SyndromeIsSatisfied_C,
vector<int>& IncorrectJ_C,
vector<int>& SuspectJ_C,
vector<int>& USS_C,
vector<int>& RUSS_C,
vector<vector<int>>& JatI_C,
vector<vector<int>>& MatValue_C,
int itr,
int HistoryLength,
int P
) {
  printf("@@@printDecodingDebugInfo\n");
  printf("MatValue_C[0][0]=%3x\n",MatValue_C[0][0]);
  // Conditional branch.
  if (!SyndromeIsSatisfied_C) {
    printf("IncorrectJ_C<%d>=",IncorrectJ_C.size());for (int j : IncorrectJ_C) {printf("%5d(%d) ", j, j / P);}  printf("\n");
    printf("SuspectJ_C<%d>=",SuspectJ_C.size());  for (int j : SuspectJ_C) {  printf("%5d(%d) ", j, j / P);}  printf("\n");

    // Loop: iterate over a range/collection.
    for (int j : IncorrectJ_C) {
      // Conditional branch.
      if (find(SuspectJ_C.begin(), SuspectJ_C.end(), j) == SuspectJ_C.end()) {
        printf("  Error: %d is not in SuspectJ_C\n", j);
      }
    }
    printf("USS_C<%d>=",USS_C.size());
    // Loop: iterate over a range/collection.
    for (int i : USS_C) {printf("%5d(%d) ", i, i / P);}printf("\n");
    // Loop: iterate over a range/collection.
    for (int i : USS_C) {
      printf("%5d(%d):", i, i / P);
      // Loop: iterate over a range/collection.
      for (int j : JatI_C[i]) {
        printf("%s%5d(%d) ",
        find(SuspectJ_C.begin(), SuspectJ_C.end(), j) != SuspectJ_C.end() ? "*" : " ",
        j, j / P);
      }
      printf("\n");
    }
    printf("RUSS_C<%d>=",RUSS_C.size());
    // Loop: iterate over a range/collection.
    for (int i : RUSS_C) {
      printf("%5d(%d) ", i, i / P);
    }
    printf("\n");
    // Loop: iterate over a range/collection.
    for (int i : RUSS_C) {
      printf("%5d(%d):", i, i / P);
      // Loop: iterate over a range/collection.
      for (int j : JatI_C[i]) {
        printf("%s%5d(%d) ",
        find(SuspectJ_C.begin(), SuspectJ_C.end(), j) != SuspectJ_C.end() ? "*" : " ",
        j, j / P);
      }
      printf("\n");
    }
    int rank=computeRankGF( RUSS_C, SuspectJ_C,JatI_C, MatValue_C);
    printf("rank=%d\n", rank);
  }
  printf("***printDecodingDebugInfo\n");
}
// Function: TryDecodeSmallErrors
// Purpose: TODO - describe the function's responsibility succinctly.
void TryDecodeSmallErrors(
int& SyndromeIsSatisfied_C,
vector<int>& SuspectJ_C,
vector<int>& RowDeg_C,
vector<int>& RUSS_C,
vector<vector<int>>& UTCBC_Rows_C_orthogonal_D,
vector<vector<int>>& UTCBC_Cols_C_orthogonal_D,
vector<vector<int>>& USSHistory_C,
int itr,
int HistoryLength,
vector<vector<int>>& JatI_C,
vector<vector<int>>& IatJ_C,
vector<vector<int>>& full_JatI_C,
vector<vector<int>>& full_IatJ_C,
vector<vector<int>>& full_JatI_D,
vector<vector<int>>& full_IatJ_D,
vector<int>& Candidate_Covering_Normal_Rows_C,
vector<int>& EstmNoiseSynd_C,
vector<int>& TrueNoiseSynd_C,
vector<int>& TrueNoise_C,
int M,
int N,
vector<int>& EstmNoise_C,
vector<vector<int>>& Mat_C,
vector<vector<int>>& MatValue_C,
vector<vector<int>>& MULGF,
vector<vector<int>>& ADDGF,
vector<vector<int>>& DIVGF,
vector<vector<int>>& BINGF,
int L) {
  printf("@@@TryDecodeSmallErrors\n");
  printf("MatValue_C[0][0]=%3x\n",MatValue_C[0][0]);
  EF_LOG.clear();
  vector<int> USS_C=USSHistory_C[itr % HistoryLength];
  // Conditional branch.
  if (!SyndromeIsSatisfied_C ){
    vector<int> cols_C;
    vector<int> rows_C;
    // Loop: repeat while condition holds.
    while(1){
      // Conditional branch.
      if (RUSS_C.size() <= 2*L && SuspectJ_C.size()<=2*L) {
        EF_LOG = "Find_Normal_Rows_Covering_SuspectJ_By_UTCBC_Cols";
        printf("#########case: Find_Normal_Rows_Covering_SuspectJ_By_UTCBC_Cols#####################\n");
        // Conditional branch.
        if(Find_Normal_Rows_Covering_SuspectJ_By_UTCBC_Cols(cols_C, rows_C, SuspectJ_C, UTCBC_Rows_C_orthogonal_D, UTCBC_Cols_C_orthogonal_D, full_IatJ_D)){break;}
      }
      // Conditional branch.
      if (SuspectJ_C.size()<=L && RUSS_C.size()<=L) {
        EF_LOG = "Rows_eq_RUSS_Cols_eq_SuspectJ";
        printf("###########case: Rows_eq_RUSS_Cols_eq_SuspectJ#####################\n");
        // Conditional branch.
        if(Rows_eq_RUSS_Cols_eq_SuspectJ(cols_C, rows_C, SuspectJ_C, RUSS_C, JatI_C, IatJ_C, MatValue_C)) {break;}
      }
      // Conditional branch.
      if (true){
        EF_LOG = "Find_Cycle_of_Length_L";
        printf("###########case: Find_Cycle_of_Length_L#####################\n");
        // Conditional branch.
        if (Find_Cycle_of_Length_L(cols_C, rows_C, SuspectJ_C, JatI_C, IatJ_C, MatValue_C)) {break;}
      }
      // Conditional branch.
      if (true){
        EF_LOG = "Find_Nonsingular_Cycle_of_Length_Larger_thatn_L";
        printf("###########case: Find_Nonsingular_Cycle_of_Length_Larger_thatn_L#####################\n");
        // Conditional branch.
        if (Find_Nonsingular_Cycle_of_Length_Larger_thatn_L(cols_C, rows_C, SuspectJ_C, JatI_C, IatJ_C, MatValue_C)) {break;}
      }

      // Conditional branch.
      if (SuspectJ_C.size()<=2*L && RUSS_C.size() % 2 == 0 && (RUSS_C.size() == L || RUSS_C.size()  == L + 1)) {
        EF_LOG = "Find_Covering_Cycles_By_RUSS";
        printf("########case: Find_Covering_Cycles_By_RUSS#####################\n");
        // Conditional branch.
        if (Find_Covering_Cycles_By_RUSS(cols_C, rows_C, SuspectJ_C, RUSS_C, JatI_C, IatJ_C)) {break;}
      }
      // Conditional branch.
      if (SuspectJ_C.size()<=2*L && USS_C.size() < L) {
        EF_LOG = "Find_Rows_Coverintg_SuspectJ_From_RUSS";
        printf("########case: Find_Rows_Coverintg_SuspectJ_From_RUSS#####################\n");
        // Conditional branch.
        if (Find_Rows_Coverintg_SuspectJ_From_RUSS(cols_C, rows_C, SuspectJ_C, RUSS_C, JatI_C, IatJ_C, MatValue_C)) {break;}
      }
      // Conditional branch.
      if(SuspectJ_C.size()<=2*L && USS_C.size() < L) {
        EF_LOG = "Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap";
        printf("########case: Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap#####################\n");
        // Conditional branch.
        if (Find_Unique_Solution_Noise_From_RUSS_Plus_Overlap(cols_C, rows_C, SuspectJ_C, RUSS_C, JatI_C, IatJ_C, MatValue_C)) {break;}
      }
      cols_C.clear();
      rows_C.clear();
      break;
    }
    printf("cols_C=");for(int j : cols_C) {printf("%5d(%d) ", j, j / P);}printf("\n");
    printf("rows_C=");for(int i : rows_C) {printf("%5d(%d) ", i, i / P);}printf("\n");
    // Conditional branch.
    if (!cols_C.empty() && !rows_C.empty()) {
      // Loop: iterate over a range/collection.
      for (int i : rows_C) {
        printf("%5d(%d) ", i, i / P);
        // Loop: iterate over a range/collection.
        for (int j : JatI_C[i]) {
          printf("%s%5d(%d):",
          find(cols_C.begin(), cols_C.end(), j) != cols_C.end() ? "*" : " ",
          j, j / P);
        }
        printf("\n");
      }
      decode_small_errors_from_rows_cols(TrueNoiseSynd_C, TrueNoise_C, M, N, EstmNoise_C, RowDeg_C, rows_C, cols_C, IatJ_C, JatI_C, Mat_C, MatValue_C, MULGF, ADDGF, DIVGF, BINGF);
    }
  }
  printf("***TryDecodeSmallErrors\n");
}
// Function: check_orthogonality
// Purpose: TODO - describe the function's responsibility succinctly.
void check_orthogonality(vector<vector<int>>& MatValue_C, vector<vector<int>>& MatValue_D, vector<vector<int>>& JatI_C, vector<vector<int>>& JatI_D, int M, vector<vector<int>>& ADDGF, vector<vector<int>>& MULGF) {
  printf("@@@ check_orthogonality\n");
  // Loop: iterate over a range/collection.
  for (int i = 0; i < M ; i++) {
    // Loop: iterate over a range/collection.
    for (int j = 0; j < M ; j++) {
      // Conditional branch.
      if (i == j) continue;
      int dot_product = 0;
      // Loop: iterate over a range/collection.
      for (int k=0;k< JatI_C[i].size();k++) {
        int jc = JatI_C[i][k];
        int value_C = MatValue_C[i][k];

        int l = -1;
        // Loop: iterate over a range/collection.
        for (int c = 0; c < JatI_D[j].size(); c++) {
          // Conditional branch.
          if (JatI_D[j][c] == jc) {
            l = c;
            break;
          }
        }
        // Conditional branch.
        if (l == -1) continue;
        int value_D = MatValue_D[j][l];

        int prod=MULGF[value_C][value_D];
        dot_product = ADDGF[dot_product][prod];

      }

      // Conditional branch.
      if (dot_product != 0) {
        printf("Orthogonality check failed for rows %d and %d: dot product = %d\n", i, j, dot_product);
        exit(1);
      }
    }
  }
  printf("Orthogonality check passed for all rows.\n");
  printf("*** check_orthogonality\n");
}
// Function: main
// Purpose: TODO - describe the function's responsibility succinctly.


// ======= Wrapper: TryDecodeSmallErrorsRef (short signature) ====================
static inline void TryDecodeSmallErrorsRef(
    SM_StateRef s,
    const SM_CodeRef& code,
    const SM_UtcBcRef& bc,
    const SM_GFTablesRef& gf,
    int M, int N, int L, int itr, int HistoryLength) {

  TryDecodeSmallErrors(
    s.SyndromeIsSatisfied,
    s.SuspectJ,
    code.RowDeg,
    s.RUSS,
    bc.UTCBC_Rows_C_orthogonal_D,
    bc.UTCBC_Cols_C_orthogonal_D,
    s.USSHistory,
    itr,
    HistoryLength,
    code.JatI,
    code.IatJ,
    bc.full_JatI_C,
    bc.full_IatJ_C,
    bc.full_JatI_D,
    bc.full_IatJ_D,
    s.Candidate_Covering_Normal_Rows,
    s.EstmNoiseSynd,
    s.TrueNoiseSynd,
    s.TrueNoise,
    M,
    N,
    s.EstmNoise,
    code.Mat,
    code.MatValue,
    gf.MULGF,
    gf.ADDGF,
    gf.DIVGF,
    gf.BINGF,
    L
  );
}
// ==============================================================================

int main(int argc, char * argv[]){

  char *FileName  =(char *)malloc(500);
  char *FileResult=(char *)malloc(500);
  char *name      =(char *)malloc(500);
  int max_num_iteration;
  int max_num_error;
  printf("argc=%d\n",argc);
  // Conditional branch.
  if(argc!=8){cout << "usage: gd_css max_iter filename_C filename_D logfile f_m  DEBUG_transmission seed" << endl; exit(0);}
  max_num_iteration=atoi(argv[1]);
  strcpy(MatrixFilePrefix_C,argv[2]);
  strcpy(MatrixFilePrefix_D,argv[3]);
  strcpy(FileResult,argv[4]);
  f_m=atof(argv[5]);
  DEBUG_transmission=atoi(argv[6]);
  unsigned seed = (unsigned) atoi(argv[7]);
  srand48(seed);
  const int USS_error_floor_threshold = 100;
  const int USS_stagnation_check_interval = 50;
  check_code_parameters_equal(MatrixFilePrefix_C, MatrixFilePrefix_D, M, N, GF, logGF);

  P = extractValueFromFilename(MatrixFilePrefix_C, std::string(1, 'P'));
  L = extractValueFromFilename(MatrixFilePrefix_C, std::string(1, 'L'));
  printf("P=%d,L=%d\n",P,L);
  construct_inv_ZP(P);
  load_GF_tables(GF, logGF, BINGF, ADDGF, MULGF, DIVGF, FFTSQ);
  load_transpose_GF_tables(GF, logGF, BINGF,  FFTSQ, TBINGF,  TFFTSQ);

  pD=3.0*f_m/2.0;
  VNtoChN_init(ChFactorMatrix_CD,pD,GF,logGF, BINGF,TBINGF);
  printf("VNtoChN_init -> done\n");
  VNtoChN_init(ChFactorMatrix_DC,pD,GF,logGF,TBINGF, BINGF);
  printf("VNtoChN_init <- done\n");

  load_matrix(MatrixFilePrefix_C, N, M, P, L, Mat_C, MatValue_C, ColDeg_C, RowDeg_C, JatI_C, IatJ_C, full_JatI_C, full_IatJ_C);
  printf("C loaded\n");

  load_matrix(MatrixFilePrefix_D, N, M, P, L, Mat_D, MatValue_D, ColDeg_D, RowDeg_D, JatI_D, IatJ_D, full_JatI_D, full_IatJ_D);
  printf("D loaded\n");
  check_orthogonality(MatValue_C, MatValue_D, JatI_C, JatI_D, M, ADDGF, MULGF);
  vector<int> ff(L/2),gg(L/2);
  extract_ff_gg(ff,gg,JatI_C,IatJ_C,JatI_D,IatJ_D,L,P);
  make_full_JatI_IatJ(full_JatI_C,full_IatJ_C,full_JatI_D,full_IatJ_D,ff,gg,P);

  initializeUTCBC_Rows(M, P,
  UTCBC_Rows_C_orthogonal_D,
  UTCBC_Rows_D_orthogonal_C,
  full_JatI_C, IatJ_D,
  full_JatI_D, IatJ_C
  );

  initializeUTCBC_Cols(M, P,
  UTCBC_Cols_C_orthogonal_D,
  UTCBC_Cols_D_orthogonal_C,
  full_JatI_C,
  full_JatI_D
  );

  initialize_interleaver(N, M, ColDeg_C, RowDeg_C, Mat_C, NtoB_C, Interleaver_C, NumEdge_C);
  initialize_interleaver(N, M, ColDeg_D, RowDeg_D, Mat_D, NtoB_D, Interleaver_D, NumEdge_D);

  initialize_decoding_arrays(N, logGF, NumEdge_C, GF,
  TrueNoise_C,
  CNtoVNxxx_C, VNtoCNxxx_C, ChNtoVN_CD, APP_C,
  EstmNoise_C, VNtoChN_CD);
  initialize_decoding_arrays(N, logGF, NumEdge_D, GF,
  TrueNoise_D,
  CNtoVNxxx_D, VNtoCNxxx_D, ChNtoVN_DC, APP_D,
  EstmNoise_D, VNtoChN_DC);
  initialize_syndrome_and_channel(M, TrueNoiseSynd_C, EstmNoiseSynd_C);
  initialize_syndrome_and_channel(M, TrueNoiseSynd_D, EstmNoiseSynd_D);

  NbUndetectedErrors=0;
  vector<int> fail_transmission;
  vector<int> degenerate_success_transmission;
  transmission=0;

  // Conditional branch.
  if(DEBUG_transmission){
    cout << "DEBUG MODE" << endl;
    // Loop: iterate over a range/collection.
    for(transmission=0;transmission<DEBUG_transmission-1;transmission++){
      // Loop: iterate over a range/collection.
      for(size_t k=0;k<N;k++){
        // Loop: iterate over a range/collection.
        for(size_t q=0;q<logGF;q++){
          // Conditional branch.
          if(drand48()<1.0-pD){}
          else{int r=lrand48()%3;}
        }
      }
    }
  }
  // Loop: repeat while condition holds.
  while(1) {

    transmission++;
    cout << "transmission=" << transmission << endl;cout.flush();

    int num_X=0,num_Z=0;
    simulateTransmissionErrors(N, logGF, GF, pD, TrueNoise_C, TrueNoise_D, BINGF, TBINGF, num_X, num_Z);
    cout << "fm_X=" << dec << num_X << "/" << (N*logGF) << "=" << (double)num_X/(N*logGF) << endl;
    cout << "fm_Z=" << dec << num_Z << "/" << (N*logGF) << "=" << (double)num_Z/(N*logGF) << endl;

    calcSyndrome(TrueNoiseSynd_C, M,TrueNoise_C,MatValue_C,RowDeg_C,ADDGF, MULGF,Mat_C);

    calcSyndrome(TrueNoiseSynd_D, M,TrueNoise_D,MatValue_D,RowDeg_D,ADDGF, MULGF,Mat_D);

    // Loop: iterate over a range/collection.
    for(size_t l=0;l<NumEdge_C;l++) { for(size_t g=0;g<GF;g++) CNtoVNxxx_C[l][g]=1.0/GF; }
    // Loop: iterate over a range/collection.
    for(size_t l=0;l<N;l++) { for(size_t g=0;g<GF;g++){ VNtoChN_CD[l][g]=1.0/GF;ChNtoVN_CD[l][g]=1.0; }}

    // Loop: iterate over a range/collection.
    for(size_t l=0;l<NumEdge_D;l++) { for(size_t g=0;g<GF;g++) CNtoVNxxx_D[l][g]=1.0/GF; }
    // Loop: iterate over a range/collection.
    for(size_t l=0;l<N;l++) { for(size_t g=0;g<GF;g++){ VNtoChN_DC[l][g]=1.0/GF;ChNtoVN_DC[l][g]=1.0; }}

    cout << "Decoding Iteration" << endl;
    itr=0;
    SyndromeIsSatisfied_C=0;
    SyndromeIsSatisfied_D=0;
    vector<int> USS_history;
    vector<int> Candidate_Covering_Normal_Rows_C;
    vector<int> Candidate_Covering_Normal_Rows_D;
    vector<int> Candidate_Covering_Cycle_Rows_C;
    vector<int> Candidate_Covering_Cycle_Rows_D;
    bool stagnated = false;
    EF_LOG.clear();
    do{
      // Conditional branch.
      if(itr==0){
        ChannelPass_zero(VNtoChN_DC,N,GF,logGF,f_m,  BINGF);
        ChannelPass_zero(VNtoChN_CD,N,GF,logGF,f_m, TBINGF);
      }else {
      // Conditional branch.
        ChannelPass(VNtoChN_CD, ChFactorMatrix_CD, ChNtoVN_CD, N, GF);
        ChannelPass(VNtoChN_DC, ChFactorMatrix_DC, ChNtoVN_DC, N, GF);
    }

    DecodeIteration(SyndromeIsSatisfied_C, VNtoCNxxx_C, CNtoVNxxx_C,
                    VNtoChN_DC, ChNtoVN_CD, APP_C, Interleaver_C, ColDeg_C,
    N, M, GF, logGF, MatValue_C, RowDeg_C,
    TrueNoiseSynd_C, EstmNoiseSynd_C,
    USSHistory_C[itr%HistoryLength],
    Updated_EstmNoise_History_C[itr%HistoryLength],
    EstmNoise_C, Mat_C, ADDGF, MULGF, DIVGF, FFTSQ);
    DecodeIteration(SyndromeIsSatisfied_D, VNtoCNxxx_D, CNtoVNxxx_D,
                    VNtoChN_CD, ChNtoVN_DC, APP_D, Interleaver_D, ColDeg_D,
    N, M, GF, logGF, MatValue_D, RowDeg_D,
    TrueNoiseSynd_D, EstmNoiseSynd_D,
    USSHistory_D[itr%HistoryLength],
    Updated_EstmNoise_History_D[itr%HistoryLength],
    EstmNoise_D, Mat_D, ADDGF, MULGF, DIVGF, TFFTSQ);
    // Conditional branch.
    if(itr==0){
      Updated_EstmNoise_History_C[0].clear();
      Updated_EstmNoise_History_D[0].clear();
    }

    count_errors(N,M,EstmNoise_C,EstmNoise_D,TrueNoise_C,TrueNoise_D,EstmNoiseSynd_C,TrueNoiseSynd_C,EstmNoiseSynd_D,TrueNoiseSynd_D,IncorrectJ_C,IncorrectJ_D,eS,eS_C,eS_D,NumUSS_C,NumUSS_D);
    vector<int> SuspectJ_C;
    vector<int> RUSS_C;
    vector<int> SuspectJ_D;
    vector<int> RUSS_D;
    SuspectJ_C=computeUnion(Updated_EstmNoise_History_C);
    RUSS_C=computeUnion(USSHistory_C);
    SuspectJ_D=computeUnion(Updated_EstmNoise_History_D);
    RUSS_D=computeUnion(USSHistory_D);

    // Conditional branch.
    if(stagnated){
      printf("########################################################################################################################################\n");
      printDecodingDebugInfo(SyndromeIsSatisfied_C,
      IncorrectJ_C,
      SuspectJ_C,
      USSHistory_C[itr%HistoryLength],
      RUSS_C,
      JatI_C,
      MatValue_C,
      itr,
      HistoryLength,
      P);
      {
  SM_StateRef sC{SyndromeIsSatisfied_C, SuspectJ_C, RUSS_C, USSHistory_C,
                 EstmNoiseSynd_C, TrueNoiseSynd_C, TrueNoise_C, EstmNoise_C,
                 Candidate_Covering_Normal_Rows_C};
  SM_CodeRef codeC{JatI_C, IatJ_C, Mat_C, MatValue_C, RowDeg_C};
  SM_UtcBcRef utcbcC{UTCBC_Rows_C_orthogonal_D, UTCBC_Cols_C_orthogonal_D,
                     full_JatI_C, full_IatJ_C, full_JatI_D, full_IatJ_D};
  SM_GFTablesRef gfC{MULGF, ADDGF, DIVGF, BINGF};
  TryDecodeSmallErrorsRef(sC, codeC, utcbcC, gfC, M, N, L, itr, HistoryLength);
}
      printf("---------------------------------------------------------------------------------------\n");
      printDecodingDebugInfo(SyndromeIsSatisfied_D,
      IncorrectJ_D,
      SuspectJ_D,
      USSHistory_D[itr%HistoryLength],
      RUSS_D,
      JatI_D,
      MatValue_D,
      itr,
      HistoryLength,
      P);
      {
  SM_StateRef sD{SyndromeIsSatisfied_D, SuspectJ_D, RUSS_D, USSHistory_D,
                 EstmNoiseSynd_D, TrueNoiseSynd_D, TrueNoise_D, EstmNoise_D,
                 Candidate_Covering_Normal_Rows_D};
  SM_CodeRef codeD{JatI_D, IatJ_D, Mat_D, MatValue_D, RowDeg_D};
  SM_UtcBcRef utcbcD{UTCBC_Rows_D_orthogonal_C, UTCBC_Cols_D_orthogonal_C,
                     full_JatI_D, full_IatJ_D, full_JatI_C, full_IatJ_C};
  SM_GFTablesRef gfD{MULGF, ADDGF, DIVGF, BINGF};
  TryDecodeSmallErrorsRef(sD, codeD, utcbcD, gfD, M, N, L, itr, HistoryLength);
}
    }

    SyndromeIsSatisfied_C=IsSyndromeSatisfied(TrueNoiseSynd_C,EstmNoiseSynd_C,USSHistory_C[itr%HistoryLength], M,EstmNoise_C,MatValue_C,RowDeg_C,ADDGF, MULGF, Mat_C);
    SyndromeIsSatisfied_D=IsSyndromeSatisfied(TrueNoiseSynd_D,EstmNoiseSynd_D,USSHistory_D[itr%HistoryLength], M,EstmNoise_D,MatValue_D,RowDeg_D,ADDGF, MULGF, Mat_D);

    SuspectJ_C=computeUnion(Updated_EstmNoise_History_C);
    RUSS_C=computeUnion(USSHistory_C);
    SuspectJ_D=computeUnion(Updated_EstmNoise_History_D);
    RUSS_D=computeUnion(USSHistory_D);

    count_errors(N,M,EstmNoise_C,EstmNoise_D,TrueNoise_C,TrueNoise_D,EstmNoiseSynd_C,TrueNoiseSynd_C,EstmNoiseSynd_D,TrueNoiseSynd_D,IncorrectJ_C,IncorrectJ_D,eS,eS_C,eS_D,NumUSS_C,NumUSS_D);

    printf("#itr=%3d (%d,%d) | eS=%5d | eS_C=%5d SuspectJ_C=%5d USS_C=%5d RUSS_C=%5d | eS_D=%5d SuspectJ_D=%5d USS_D=%5d RUSS_D=%5d |%d %d %d\n",
    itr,
    SyndromeIsSatisfied_C, SyndromeIsSatisfied_D,
    eS,
    eS_C,SuspectJ_C.size(), NumUSS_C,RUSS_C.size(),
    eS_D,SuspectJ_D.size(), NumUSS_D,RUSS_D.size(),
    stagnated,transmission,seed);

    // Conditional branch.
    if(SyndromeIsSatisfied_C && SyndromeIsSatisfied_D){
      // Conditional branch.
      if(eS){
        cout << "error in C:";for(size_t k=0;k<N;k++){if ( EstmNoise_C[k]!=TrueNoise_C[k]){ printf("%d ",k);}}cout << endl;
        cout << "error in D:";for(size_t k=0;k<N;k++){if ( EstmNoise_D[k]!=TrueNoise_D[k]){ printf("%d ",k);}}cout << endl;
      }
      break;
    }

    // Conditional branch.
    if (USS_history.size() >= USS_stagnation_check_interval ) {
      // Conditional branch.
      if (NumUSS_C+NumUSS_D >= USS_history[USS_history.size() - USS_stagnation_check_interval]) {
        printf("USS has not decreased compared to %d iterations ago: %d >= %d\n",
        USS_stagnation_check_interval, NumUSS_C+NumUSS_D, USS_history[USS_history.size() - USS_stagnation_check_interval]);
        stagnated = true;
      }
    }

    USS_history.push_back(NumUSS_C+NumUSS_D);
    // Conditional branch.
    if (stagnated  && NumUSS_C+NumUSS_D > USS_error_floor_threshold) {
      break;
    }
    itr++;
  }while(itr<max_num_iteration);
  int decoding_success=false;
  int degenerate_success=false;

  // Conditional branch.
  if(eS==0){
    decoding_success=true;
  }else if(SyndromeIsSatisfied_C && SyndromeIsSatisfied_D){
  bool fC,fD;
  fC=check_degenerate_decoding_success(EstmNoise_C, TrueNoise_C, JatI_C, JatI_D, MatValue_D, N, M);
  fD=check_degenerate_decoding_success(EstmNoise_D, TrueNoise_D, JatI_D, JatI_C, MatValue_C, N, M);
  decoding_success=fC&fD;
}

printf("decoding_success=%d\n",decoding_success);
// Conditional branch.
if(!decoding_success){
  count_errors(N,M,EstmNoise_C,EstmNoise_D,TrueNoise_C,TrueNoise_D,EstmNoiseSynd_C,TrueNoiseSynd_C,EstmNoiseSynd_D,TrueNoiseSynd_D,IncorrectJ_C,IncorrectJ_D,eS,eS_C,eS_D,NumUSS_C,NumUSS_D);
  TeF++;
  TeS+=eS;
  fail_transmission.push_back(transmission);
}else{
// Conditional branch.
if(eS!=0){
  TdS++;
  degenerate_success_transmission.push_back(transmission);
}
}
printf("#transmission=%d #f_m=%f TeF=%3d TeS=%5d FER=%f SER=%f itr=%3d eS=%5d TdS=%5d seed=%d\n",
transmission,f_m,
TeF,TeS,
(double)TeF/transmission,(double)TeS/(transmission*(N+N)),
itr,eS,TdS,seed);

// Conditional branch.
if(!EF_LOG.empty()){
  printf("EF_LOG=%s\n",EF_LOG.c_str());
  FILE *f;
  sprintf(name,"EF_LOG_%s_TRANS%d_EPS%f_SEED%d",FileResult,transmission,f_m,seed);
  f=fopen(name,"w");
  // Conditional branch.
  if (f == NULL) {
    perror("Failed to open file");
    return 1;
  }
  fprintf(f,"%s %d %f %d %d %d \n",
  EF_LOG.c_str(),itr,f_m,transmission,itr,seed);
  fclose(f);
}

// Conditional branch.
if(transmission%1==0){
  sprintf(FileName,"LOG_%s_ITR%d_GF%d_N%d_M%d_R%f_EPS%f_SEED%d",FileResult,max_num_iteration,GF,N,M,Rate_C,f_m,seed);
  // Conditional branch.
  if(DEBUG_transmission==0){
    f=fopen(FileName,"w");
    // Conditional branch.
    if (f == NULL) {
      perror("Failed to open file");
      return 1;
    }
    fprintf(f,"%d %f %d %d %d %d %d %f %f %d  %d %d  %d ",
    transmission,f_m,
    N,logGF,
    TeF,TeS,NbUndetectedErrors,
    (double)TeF/transmission,(double)TeS/(transmission*N),
    itr,eS,TdS,seed);
    fprintf(f,"|");
    printf(  "|");
    // Loop: iterate over a range/collection.
    for(int t:fail_transmission){fprintf(f," %d ",t);}fprintf(f,"|");
    // Loop: iterate over a range/collection.
    for(int t:fail_transmission){ printf(  " %d ",t);} printf(  "|");
    // Loop: iterate over a range/collection.
    for(int t:degenerate_success_transmission){fprintf(f," %d ",t);}fprintf(f,"\n");
    // Loop: iterate over a range/collection.
    for(int t:degenerate_success_transmission){ printf(  " %d ",t);} printf(  "\n");
    fclose(f);
  }
}
// Conditional branch.
if(DEBUG_transmission){
  break;
}
}

return 0;
}
