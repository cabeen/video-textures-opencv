/**
 * @file video_textures.cpp
 * @author Ryan Cabeen <cabeen@gmail.com>
 * @version 1.0
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "video_textures.h"

/********
 * ALGO *
 ********/

void swap(Elem* elems, int a, int b)
{
    Elem s = elems[a];
    elems[a] = elems[b];
    elems[b] = s;
}

int partition(Elem* elems, int left, int right, int pivot)
{
   swap(elems, pivot, right);

   int i = left - 1;
   int j = right;

    while (true)
    {
        while (elems[++i].key < elems[right].key)
        {
            if (i == right) break;
        }

        while (elems[right].key < elems[j--].key)
        {
            if (j == left) break;
        }

        if (i >= j) break;

        swap(elems, i, j);
    }

    swap(elems, i, right);

    return i;
}

void quicksort(Elem* elems, int left, int right)
{
    if (right <= left)
        return;

    int pivot = (left + right) / 2;
    int j = partition(elems, left, right, pivot);
    int leftsize = (j - 1) - right;
    int rightsize = right - (j + 1);

    if(leftsize < rightsize)
    {
        quicksort(elems, left, j - 1);
        quicksort(elems, j + 1, right);
    }
    else
    {
        quicksort(elems, j + 1, right);
        quicksort(elems, left, j - 1);
    }
}

void quicksort(Elem* elems, int n)
{
    quicksort(elems, 0, n - 1);
}

void reverse(Elem* elems, int n)
{
    int h = floor(n / 2.0);
    for (int i = 0; i < h; i++)
    {
        swap(elems, i, n - 1 - i);
    }
}

/*******
 * MAT *
 *******/

struct Mat
{
    int n;
    int m;
    double *d;
};

Mat* create_mat(int n, int m, double v)
{
    Mat* out = (Mat*) malloc(sizeof(Mat));
    out->n = n;
    out->m = m;
    out->d = (double*) malloc(m * n * sizeof(double));
    for (int i = 0; i < out->m; i++)
        out->d[i] = v;

    return out;
}

void free_mat(Mat* m)
{
    free(m->d);
    free(m);
}

double get_mat(Mat* m, int i, int j)
{
    int idx = j + i * m->m;
    return m->d[idx];
}

void set_mat(Mat* m, int i, int j, double v)
{
    int idx = j + i * m->m;
    m->d[idx] = v;
}

void set_mat_sym(Mat* m, int i, int j, double v)
{
    int idx1 = j + i * m->m;
    int idx2 = i + j * m->m;
    m->d[idx1] = v;
    m->d[idx2] = v;
}

void copy_mat(Mat* in, Mat* out)
{
    for (int i = 0; i < in->n; i++)
        for (int j = 0; j < in->m; j++)
            set_mat(out, i, j, get_mat(in, i, j));
}

void write_mat(Mat* m, const char* fn)
{
    printf("Writing matrix to: %s\n", fn);
    FILE *fp = fopen(fn, "w");
    for (int i = 0; i < m->n; i++)
    {
        for (int j = 0; j < m->m; j++)
        {
            fprintf(fp, "%f ", get_mat(m, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

Mat* read_mat(const char* fn)
{
    return NULL;
}

/*******
 * CDF *
 *******/

Cdf* create_cdf(int n)
{
    Cdf* cdf = (Cdf*) malloc(sizeof(Cdf));
    cdf->n = n;
    cdf->ids = (int*) malloc(n * sizeof(int));
    cdf->vals = (double*) malloc(n * sizeof(double));

    return cdf;
}

void free_cdf(Cdf* cdf)
{
    free(cdf->ids);
    free(cdf->vals);
    free(cdf);
}

static void ncumsum(Elem* elems, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += elems[i].key;

    for (int i = 0; i < n; i++)
        elems[i].key /= sum;

    for (int i = 1; i < n; i++)
        elems[i].key += elems[i-1].key;
}

Cdf** cdf_map(Mat* mat)
{
    printf("Creating CDF\n");
    int n = mat->n;
    int m = mat->m;
    Cdf** cdfs = (Cdf**) malloc(n * sizeof(Cdf*));
    Elem* elems = (Elem*) malloc(n * sizeof(Elem));
    for (int i = 0; i < n; i++)
    {
        cdfs[i] = create_cdf(m);
        for (int j = 0; j < m; j++)
        {
            double v = get_mat(mat, i, j);
            elems[j].key = v;
            elems[j].id = j;
        }

        quicksort(elems, m);
        reverse(elems, m);
        ncumsum(elems, m);

        for (int j = 0; j < m; j++)
        {
            cdfs[i]->vals[j] = elems[j].key;
            cdfs[i]->ids[j] = elems[j].id;
        }
    }

    free(elems);

    return cdfs;
}

int sample(Cdf* cdf)
{
    double r = RandomUniform(); 
    for (int j = 0; j < cdf->n; j++)
        if (r < cdf->vals[j])
            return cdf->ids[j];
}

/*******
 * SEQ *
 *******/

Seq* create_seq(int n)
{
    Seq* seq = (Seq*) malloc(sizeof(Seq));
    seq->frames = (int*) malloc(n * sizeof(int));
    seq->n = n;
    return seq;
}

void free_seq(Seq* seq)
{
    free(seq->frames);
    free(seq);
}

/*******
 * VID *
*******/

struct Vid
{
    IplImage** imgs;
    int nf;
    int fps;
    int w;
    int h;
};

void free_vid(Vid* v)
{
    cvReleaseImage(v->imgs);
    free(v);
}

Vid* read_vid(const char* fn)
{
    printf("Reading video from %s\n", fn);
    CvCapture* capture = cvCaptureFromAVI(fn);

    cvQueryFrame(capture);
    int h = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    int w = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    int fps = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
    int nf = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
    CvSize s = cvSize(w, h);

    printf("...Header dimensions = (%d, %d)\n", w, h);
    printf("...Header FPS = %d\n", fps);
    printf("...Header Frames = %d\n", nf);

    printf("...Reading frames\n");
    Vid* vid = (Vid*) malloc(sizeof(Vid));
    vid->imgs = (IplImage**) malloc(nf * sizeof(IplImage*));
    int idx = 0;
    while (cvGrabFrame(capture) && idx < nf)
    {
        IplImage* frame = cvRetrieveFrame(capture);
        IplImage *out = cvCreateImage(s, IPL_DEPTH_8U, frame->nChannels);
        cvCopy(frame, out);
        vid->imgs[idx++] = out;
    }

    vid->nf = idx;
    vid->fps = fps;
    vid->w = w;
    vid->h = h;
    printf("...Frames read = %d\n", vid->nf);

    cvReleaseCapture(&capture);

    return vid;
}

int width_vid(Vid* vid)
{
    return vid->w;
}

int height_vid(Vid* vid)
{
    return vid->h;
}

int frames_vid(Vid* vid)
{
    return vid->nf;
}

void resize_vid(Vid* vid, int w, int h)
{
    printf("Resizing video\n");

    int nf = vid->nf;
    CvSize s = cvSize(w, h);
    CvSize os = cvSize(vid->w, vid->h);

    printf("...Input dimensions = (%d, %d)\n", vid->w, vid->h);
    printf("...Output dimensions = (%d, %d)\n", w, h);

    vid->w = w;
    vid->h = h;
    int nc = vid->imgs[0]->nChannels;
    IplImage *buffer = cvCreateImage(os, IPL_DEPTH_8U, nc);

    for (int i = 0; i < nf; i++) 
    {
        IplImage* img = vid->imgs[i];
        cvCopy(img, buffer);
        IplImage *nimg = cvCreateImage(s, IPL_DEPTH_8U, nc);
        cvResize(buffer, nimg);
        cvReleaseImage(&img);
        vid->imgs[i] = nimg;
    }

    cvReleaseImage(&buffer);
}

static void normalize(IplImage* img)
{
    int w = img->width;
    int h = img->height;
    int c = img->nChannels;
    int n = w * h;
    double min = -1;
    double max = -1;
    CvScalar mean = cvScalar(0, 0, 0, 0);

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            CvScalar sv = cvGet2D(img, i, j);
            for (int k = 0; k < c; k++)
            {
                double v = sv.val[k];
                if (min == -1 || v < min) min = v;
                if (max == -1 || v > max) max = v;
                mean.val[k] += v / (double) n;
            }
        }
    }

    CvScalar var = cvScalar(0, 0, 0, 0);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            CvScalar sv = cvGet2D(img, i, j);
            for (int k = 0; k < c; k++)
            {
                double v = sv.val[k];
                double dv = v - mean.val[k];
                var.val[k] += dv / (double) n; 
            }
        }
    }

    for (int k = 0; k < c; k++)
        if (abs(var.val[k]) < 1e-4)
            var.val[k] = 1.0;

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            CvScalar sv = cvGet2D(img, i, j);
            for (int k = 0; k < c; k++)
            {
                sv.val[k] = (sv.val[k] - mean.val[k]) / var.val[k];
            }
            cvSet2D(img, i, j, sv);
        }
    }
}

Vid* process_vid(Vid* in, int w, int h, int gray)
{
    printf("Processing video\n");

    int nf = in->nf;
    int resize = (h >= 1 || w >= 1);
    int nc = gray ? gray : in->imgs[0]->nChannels;
    int d = in->imgs[0]->depth;
    int ow = in->w;
    int oh = in->h;
    w = w ? w : ow;
    h = h ? h : oh;
    CvSize s = cvSize(w, h);
    CvSize os = cvSize(ow, oh);

    printf("...Input dimensions = (%d, %d)\n", ow, oh);
    printf("...Output dimensions = (%d, %d)\n", w, h);
    printf("...Grayscale = %d\n", gray);
    printf("...Depth code = %d\n", d);
    printf("...Number of channels = %d\n", nc);

    Vid* out = (Vid*) malloc(sizeof(Vid));
    out->imgs = (IplImage**) malloc(nf * sizeof(IplImage*));
    out->nf = nf;
    out->fps = in->fps;
    out->w = w;
    out->h = h;

    IplImage *buffer = cvCreateImage(os, d, nc);
    IplImage *buffer2 = cvCreateImage(s, d, nc);
    for (int i = 0; i < nf; i++) 
    {
        IplImage* img = in->imgs[i];

        if (gray)
            cvCvtColor(img, buffer, CV_RGB2GRAY);
        else
            cvCopy(img, buffer);

        cvResize(buffer, buffer2);

        IplImage *nimg = cvCreateImage(s, IPL_DEPTH_32F, nc);
        cvConvertScale(buffer2, nimg, 1.0 / 255.0);
        normalize(nimg);
        out->imgs[i] = nimg;
    }

    cvReleaseImage(&buffer);
    cvReleaseImage(&buffer2);

    return out;
}

void write_vid(Vid* vid, const char* fn, Seq* seq)
{
    printf("Writing video to %s\n", fn);
    int f = CV_FOURCC('X', 'V', 'I', 'D');
    CvSize s = cvSize(vid->w, vid->h);
    int c = vid->imgs[0]->nChannels > 1;
    CvVideoWriter *w = cvCreateVideoWriter(fn, f, vid->fps, s, c);

    int nf = seq ? seq->n : vid->nf; 
    for (int i = 0; i < nf; i++)
    {
        int idx = seq ? seq->frames[i] : i;
        cvWriteFrame(w, vid->imgs[idx]);
    }

    cvReleaseVideoWriter(&w);
}

/********
 * MATH *
 ********/

static double ssd(IplImage* src, IplImage* dest)
{
    int w = src->width;
    int h = src->height;
    int c = src->nChannels;
    double ssd = 0;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            CvScalar sv = cvGet2D(src, i, j);
            CvScalar dv = cvGet2D(dest, i, j);
            for (int k = 0; k < c; k++)
            {
                double delv = sv.val[k] - dv.val[k];
                ssd += delv * delv;
            }
        }
    }

    ssd /= h * w;

    return ssd;
}

void ssd(Vid* vid, Mat* out)
{
    int nf = vid->nf;
    IplImage** imgs = vid->imgs;

    printf("Computing SSDs\n");
    for (int j = 0; j < nf; j++)
    {
        for (int i = j + 1; i < nf; i++)
        {
            double v = ssd(imgs[i], imgs[j]);
            set_mat(out, i, j, v);
            set_mat(out, j, i, v);
        }
        
        set_mat(out, j, j, 0.0);
    }
}

void diag_filt(Mat* in, Mat* out)
{
    printf("Applying diagonal filter\n");
    int n = in->n;
    int m = in->m;
    int d = 1;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double c = 0;
            double v = 0;
            for (int k = -d; k < d; k++)
            {
                int di = i + k; 
                int dj = j + k;
                if (di >= 0 && di < n && dj >= 0 && dj < m)
                {
                    v += get_mat(in, i + k, j + k); 
                    c++;
                }
            }
            set_mat(out, i, j, v / c);
        }
    }
}

void deadend_filt(Mat* in, Mat* out, double alpha, int p, double thresh)
{
    printf("Applying dead-end filter\n");
    printf("...Alpha = %f\n", alpha);
    printf("...p = %d\n", p);
    printf("...Threshold = %f\n", thresh);
    int n = in->n;
    int m = in->m;
    double* ms = (double*) malloc(n * sizeof(double));

    // Initialize
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
            set_mat(out, i, j, pow(get_mat(in, i, j), p));
    }

    // Iterate
    double cmax = -1;
    int iters = 0;
    while(cmax == -1 || cmax > thresh)
    {
        for (int i = 0; i < n; i++)
        {
            double dmin = get_mat(out, i, 0);
            for (int j = 0; j < m; j++)
            {
                if (j != i)
                    dmin = fmin(dmin, get_mat(out, i, j));
            }
            ms[i] = dmin;
        }

        for (int i = n - 1; i >= 0; i--)
        {
            cmax = 0;
            for (int j = 0; j < m; j++)
            {
                double ta = pow(get_mat(in, i, j), p);
                double tb = alpha * ms[j];
                double next = ta + tb;
                double prev = get_mat(out, i, j);
                double diff = abs(next - prev);
                cmax = fmax(cmax, diff);
                set_mat(out, i, j, next);
            }
        }
        iters++;
    }
    printf("...Corrected for dead-ends after %d iterations\n", iters);
}

void prob_map(Mat* in, Mat* out, double factor)
{
    printf("Computing probability map\n");
    double vmean = 0;
    double vmin = -1;
    double vmax = -1;
    for (int j = 0; j < in->m; j++)
    {
        for (int i = 0; i < in->n; i++)
        {
            if (i == j) continue;

            double v = get_mat(in, i, j);
            vmean += v;

            if (vmin == -1)
                vmin = vmax = v;

            vmin = fmin(vmin, v);
            vmax = fmax(vmax, v);
        }
    }

    vmean /= in->m * in->n;
    printf("...Mean non-zero ssd = %f\n", vmean);
    printf("...Min non-zero ssd = %f\n", vmin);
    printf("...Max non-zero ssd = %f\n", vmax);

    double sigma = factor * vmean;
    printf("...Sigma = %f\n", sigma);

    // Compute probabilities
    for (int i = 0; i < in->n - 1; i++)
    {
        for (int j = 0; j < in->m; j++)
        {
            double v = get_mat(in, i + 1, j);
            double p = exp(-v / sigma);
            set_mat(out, i, j, p);
        }
    }

    // Avoid transitions to the last frame
    for (int j = 0; j < in->m; j++)
        set_mat(out, in->n - 1, j, 0);

    for (int i = 0; i < in->n; i++)
        set_mat(out, i, in->m - 1, 0);

    // Normalize
    for (int i = 0; i < in->n; i++)
    {
        double sum = 0;
        for (int j = 0; j < in->m; j++)
            sum += get_mat(out, i, j);

        for (int j = 0; j < in->m; j++)
            set_mat(out, i, j, get_mat(out, i, j) / sum);
    }
}

/********
 * UTIL *
 ********/

int test(Mat* m)
{
    int n = m->n;
    int j0 = m->n / 2;

    Elem* elems = (Elem*) malloc(n * sizeof(Elem));
    for (int i = 0; i < n; i++)
    {
        double v = get_mat(m, i, j0);
        elems[i].key = v;
        elems[i].id = i;
    }

    printf("index map:\n");
    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i <= j; i++)
        {
            int ii = fmax(i, j);
            int jj = fmin(i, j);
            int idx = jj + ii * (ii + 1) / 2;
            printf("(i, j) = (%d, %d), idx = %d\n", i, j, idx);
        }
    }

    printf("prob map:\n");
    for (int j = 0; j < 10; j++)
    {
        for (int i = 0; i <= j; i++)
        {
            printf("%.2f ", get_mat(m, i, j));
        }
        printf("\n");
    }

    printf("original distribution:\n");
    for (int i = 0; i < n; i++)
        printf("key = %f, id = %d:\n", elems[i].key, elems[i].id);

    quicksort(elems, 0, n-1);
    printf("sorted distribution:\n");
    for (int i = 0; i < n; i++)
        printf("key = %f, id = %d:\n", elems[i].key, elems[i].id);

    reverse(elems, n);
    printf("reverse-sorted distribution:\n");
    for (int i = 0; i < n; i++)
        printf("key = %f, id = %d:\n", elems[i].key, elems[i].id);

    ncumsum(elems, n);
    printf("cumulative distribution:\n");
    for (int i = 0; i < n; i++)
        printf("key = %f, id = %d:\n", elems[i].key, elems[i].id);

    free(elems);
}
