/**
 * @file video_textures.h 
 * @author Ryan Cabeen <cabeen@gmail.com>
 * @version 1.0
 */

#ifndef VIDEO_TEXTURES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "randomlib.h"

#define DEFAULT_ALPHA .999
#define DEFAULT_THRESH 1.0e-8
#define DEFAULT_P 10
#define DEFAULT_SIGMA 0.05
#define DEFAULT_SAMPLES 1000
#define TRUE 1
#define FALSE 0

/********
 * ALGO *
 ********/

struct Elem
{
    double key;
    int id;
};

void swap(Elem* elems, int a, int b);
int partition(Elem* elems, int left, int right, int pivot);
void quicksort(Elem* elems, int left, int right);
void quicksort(Elem* elems, int n);
void reverse(Elem* elems, int n);

/*******
 * MAT *
 *******/

struct Mat;
Mat* create_mat(int n, int m, double v);
void free_mat(Mat* m);
double get_mat(Mat* m, int i, int j);
void set_mat(Mat* m, int i, int j, double v);
void set_mat_sym(Mat* m, int i, int j, double v);
void copy_mat(Mat* in, Mat* out);
void write_mat(Mat* m, const char* fn);
Mat* read_mat(const char* fn);

/*******
 * CDF *
 *******/

struct Cdf
{
    int n;
    int* ids;
    double* vals;
};

struct Cdf;
Cdf* create_cdf(int n);
Cdf** cdf_map(Mat* m);
int sample(Cdf* cdf);
void free_cdf(Cdf* cdf);

/*******
 * SEQ *
 *******/

struct Seq
{
    int n;
    int* frames;
};

Seq* create_seq(int n);
void free_seq(Seq* seq);

/*******
 * VID *
 *******/

struct Vid;
Vid* read_vid(const char* fn); 
void resize_vid(Vid* vid, int w, int h);
Vid* process_vid(Vid* vid, int w, int h, int bw);
int width_vid(Vid* vid);
int height_vid(Vid* vid);
int frames_vid(Vid* vid);
void write_vid(Vid* vid, const char* fn, Seq* seq);
void free_vid(Vid* vid);

/********
 * MATH *
 ********/

void ssd(Vid* vid, Mat* out);
void diag_filt(Mat* in, Mat* out);
void deadend_filt(Mat* in, Mat* out, double alpha, int p, double thresh);
void prob_map(Mat* in, Mat* out, double factor);

/********
 * UTIL *
 ********/

int test(Mat* m);

#endif
