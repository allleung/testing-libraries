#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <vector>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>


#include <armadillo>

using namespace Eigen;
using namespace std;

extern "C" {
//Matrix multipication
	void dgemm_(char *TRANSA, char *TRANSB, int *M, int *N, int *K, double *ALPHA, double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);

//Matrix*vector multiplication
	void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

// LU decomoposition of a general matrix
	void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
	void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
//eigenvalues
	 void dsyev_(char* JOBZ, char* UPLO, int *N, double *A, int *LDA, double *W, double *WORK, int *LWORK, int *INFO);
}


double gsl_tester (int matrix_dim);
double eigen_tester (int matrix_dim);
double armadillo_tester (int matrix_dim);
double blas_tester (int matrix_dim);
double tester_tester(double (*function) (int), int matrix_dim);

int main ()
{

/*
blas_tester(1000);
blas_tester(2000);
blas_tester(5000);
//blas_tester(10000);
*/
gsl_tester(1000);
gsl_tester(2000);
gsl_tester(5000);
gsl_tester(10000);

/*
eigen_tester(1000);
eigen_tester(2000);
eigen_tester(5000);
//eigen_tester(10000);

armadillo_tester(1000);
armadillo_tester(2000);
armadillo_tester(5000);
//armadillo_tester(10000);


//tester_tester(&blas_tester, 1000);
*/

/*
double results1 = tester_tester(&gsl_tester, 1000);	//printing onto file
double results2 = tester_tester(&gsl_tester, 2000);
double results5 = tester_tester(&gsl_tester, 5000);
double results10 = tester_tester(&gsl_tester, 10000);


double results1 = gsl_tester(1000);
double results2 = gsl_tester(2000);
double results5 = gsl_tester(3000);
double results10 = gsl_tester(4000);

ofstream gsl_output("output.txt");

gsl_output << "# size" << '\t' << "times" << endl
<< "1000" <<'\t'<< results1 << endl
<< "2000" <<'\t'<< results2 << endl
<< "5000" <<'\t'<< results5 << endl
<< "10000" <<'\t'<< results10 << endl;

ofstream gsl_output("output.txt");
*/


return 0;
}

double tester_tester(double (*function)(int), int matrix_dim)	//runs tester 10 times
{
	double total = 0;
	double trial = 0;

	for(int i = 0; i < 10; i++)
	{
		double trial = (*function)(matrix_dim);
		total = trial + total;
	}


	cout << "AVERAGE OVER 10 TRIALS: " << total/10 << endl;
	return total/10;
}

double gsl_tester(int matrix_dim)//GSL TESTER
{
	gsl_matrix * m = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix m
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			gsl_matrix_set (m, i, j, rand() % 100);
		}
	}

	gsl_matrix * n = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix n
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			gsl_matrix_set (n, i, j, rand() % 100);
		}
	}

	gsl_matrix * p = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix p
	
	gsl_vector * v = gsl_vector_alloc(matrix_dim); //make vector v
	for (int k = 0; k < matrix_dim; k++)
	{
		gsl_vector_set(v, k, rand() % 100);
	}

	gsl_vector * w = gsl_vector_alloc(matrix_dim); //make vector w

	gsl_permutation * perm = gsl_permutation_alloc(matrix_dim); //initializing for LU decomp
	int s;

	gsl_eigen_symmv_workspace *ws = gsl_eigen_symmv_alloc(matrix_dim); //initializing for eigen decomp
	

	clock_t clock_start = clock(); //START

						//OPERATIONS HERE

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, m, n, 1, p); //m*m
	//gsl_blas_dgemv (CblasNoTrans, 1, m, v, 1, w);	//m*v
	//gsl_linalg_LU_decomp (m, perm, &s);	//invert
	//gsl_linalg_LU_invert(m, perm, p);
	//gsl_eigen_symmv(m, v, p, ws);	//eigen decomp

	clock_t clock_end = clock(); //END


	double clock_duration = (clock_end - clock_start);
	double clock_duration_seconds = clock_duration / (double) CLOCKS_PER_SEC;

	cout << "For matrix_dim: " << matrix_dim << ": ";
	printf("gsl_tester time (s): %-10.5f \n", clock_duration_seconds);
	 
	gsl_matrix_free (m);
	gsl_matrix_free (n);
	gsl_vector_free (v);
	
	return clock_duration_seconds;

}

double eigen_tester(int matrix_dim)	//EIGEN TESTER
{

	MatrixXd m(matrix_dim, matrix_dim);
	MatrixXd n(matrix_dim, matrix_dim);
	VectorXd v(matrix_dim);
	MatrixXd p;


	for (int i = 0; i < matrix_dim; i++)	//make matrix m
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			m(i, j) =  rand() % 100;
		}
	}

	for (int i = 0; i < matrix_dim; i++)	//make matrix n
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			n(i, j) =  rand() % 100;
		}
	}

	for (int k = 0; k < matrix_dim; k++)	//make vector v
	{
		v(k) = rand() % 100;
	}


	clock_t clock_start = clock(); //START
						//OPERATIONS HERE
	//p = m*n;
	//p = m*v;
	//p = m.inverse();
	EigenSolver<MatrixXd> eigensolver(m);

	clock_t clock_end = clock(); //END

	double clock_duration = (clock_end - clock_start);
	double clock_duration_seconds = clock_duration / (double) CLOCKS_PER_SEC;

	cout << "For matrix_dim: " << matrix_dim << ": ";
	printf("eigen_tester time (s): %-10.5f \n", clock_duration_seconds);
	return clock_duration_seconds;
}



double armadillo_tester (int matrix_dim) //ARMADILLO TESTER
{
	arma::mat m(matrix_dim, matrix_dim); //make matrix m
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			m(i, j) = rand() % 100;
		}
	}

	arma::mat n(matrix_dim, matrix_dim); //make matrix n
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			n(i, j) = rand() % 100;
		}
	}

	arma::mat p(matrix_dim, matrix_dim); //make matrix p
	
	arma::vec v(matrix_dim);
	for (int k = 0; k < matrix_dim; k++)
	{
		arma::vec v(matrix_dim);
	}

	arma::mat q = m.t() * m; //initializing for eigen decomp
	arma::vec eigval;
	arma::mat eigvec;
	
	clock_t clock_start = clock(); //START

						//OPERATIONS HERE
	//p = m * n;
	//p = m * v;
	//p = arma::inv(m);
	arma::eig_sym(eigval, eigvec, q);
	
	clock_t clock_end = clock(); //END


	double clock_duration = (clock_end - clock_start);
	double clock_duration_seconds = clock_duration / (double) CLOCKS_PER_SEC;

	cout << "For matrix_dim: " << matrix_dim << ": ";
	printf("armadillo_tester time (s): %-10.5f \n", clock_duration_seconds);
	 	
	return clock_duration_seconds;

}

double blas_tester(int matrix_dim)//TESTS LAPACK and OPENBLAS depending on linked library
{
	gsl_matrix * m = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix m
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			gsl_matrix_set (m, i, j, rand() % 100);
		}
	}

	gsl_matrix * n = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix n
	  
	for (int i = 0; i < matrix_dim; i++)
	{
		for (int j = 0; j < matrix_dim; j++)
		{
			gsl_matrix_set (n, i, j, rand() % 100);
		}
	}

	gsl_matrix * p = gsl_matrix_alloc (matrix_dim, matrix_dim); //make matrix p
	
	gsl_vector * v = gsl_vector_alloc(matrix_dim); //make vector v
	for (int k = 0; k < matrix_dim; k++)
	{
		gsl_vector_set(v, k, rand() % 100);
	}

	gsl_vector * w = gsl_vector_alloc(matrix_dim); //make vector w
	gsl_vector * x = gsl_vector_alloc(matrix_dim*3-1); //make vector x
	vector<int> y(matrix_dim);

	gsl_permutation * perm = gsl_permutation_alloc(matrix_dim); //initializing for LU decomp
	int s;

	gsl_eigen_symmv_workspace *ws = gsl_eigen_symmv_alloc(matrix_dim); //initializing for eigen decomp
	

	char TRANS = 'N';	//initializing for m*v, v*v, LU decomp
	int M = matrix_dim;
	int N = matrix_dim;
	int K = matrix_dim;
	double ALPHA = 1.0;
	int LDA = matrix_dim;
	int LDB = matrix_dim;
	double BETA = 0.0;
	int LDC = matrix_dim;
	int LWORK = 3*N-1;
	int INCX = 1;
	int INCY = 1;
	int INFO;
	char JOBZ = 'N';
	char UPLO = 'L';

	clock_t clock_start = clock(); //START

						//OPERATIONS HERE
	
	//dgemm_(&TRANS, &TRANS, &M, &N, &K, &ALPHA, m->data, &LDA, n->data, &LDB, &BETA, p->data, &LDC);//m*m
	//dgemv_(&TRANS, &M, &N, &ALPHA, m->data, &LDA, v->data, &INCX, &BETA, v->data, &INCY);//m*v
	dgetrf_(&M, &N, m->data, &LDA, & *y.begin(), &INFO);	//LU decomp
	//dgetri_(&N, m->data, &LDA, & *y.begin(), w->data, &LWORK, &INFO); //Inverse
	dsyev_(&JOBZ, &UPLO, &N, m->data, &LDA, w->data, x->data, &LWORK, &INFO);//Eigen (both eigenvalues and vectors)
	
	clock_t clock_end = clock(); //END


	double clock_duration = (clock_end - clock_start);
	double clock_duration_seconds = clock_duration / (double) CLOCKS_PER_SEC;
	
	cout << "For matrix_dim: " << matrix_dim << ": ";
	printf("blas_tester time (s): %-10.5f \n", clock_duration_seconds);
	
	return clock_duration_seconds;

}


























