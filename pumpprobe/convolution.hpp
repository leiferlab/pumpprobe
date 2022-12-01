double _convolution1(double *A, double *B, int N, double delta, int k);
void convolution(double *A, double *B, int M, double delta, double *out, int k=8);
long double _convolution1_128(long double *A, long double *B, int N, double delta, int k);
void convolution(double *A, double *B, int M, double delta, double *out, bool add, int k=8);
void convolution_add(double *A, double *B, int M, double delta, double *out, int k=8);
