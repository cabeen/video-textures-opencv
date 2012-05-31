/**
 * @file main.cpp
 * @author Ryan Cabeen <cabeen@gmail.com>
 * @version 1.0
 */

#include "video_textures.h" 

static int create_texture(float sigma, int nsamples, 
                          const char* infn, const char* outfn)
{
    printf("Started\n");

    Vid* full_vid = read_vid(infn);
    int w = width_vid(full_vid);
    int h = height_vid(full_vid);
    double s = 1024.0 / (double) w; 
    s = fmin(1.0, s);
    w = (int) (s * (double) w);
    h = (int) (s * (double) h);
    resize_vid(full_vid, w, h);

    s = 640.0 / (double) w;
    s = fmin(1.0, s);
    w = (int) (s * (double) w);
    h = (int) (s * (double) h);
    Vid* vid = process_vid(full_vid, w, h, TRUE);

    int nf = frames_vid(vid);
    Mat* buffer = create_mat(nf, nf, 0.0);
    Mat* buffer2 = create_mat(nf, nf, 0.0);

    ssd(vid, buffer);
    free_vid(vid);
    write_mat(buffer, "ssd.txt");

    diag_filt(buffer, buffer2);
    write_mat(buffer2, "filt1.txt");

    diag_filt(buffer2, buffer);
    write_mat(buffer, "filt2.txt");

    copy_mat(buffer, buffer2);

    prob_map(buffer2, buffer, sigma);
    write_mat(buffer, "prob.txt");

    printf("Sampling markov chain\n");
    Cdf** cdfs = cdf_map(buffer);
    Seq* seq = create_seq(nsamples);
    seq->frames[0] = RandomInt(0, nf - 1);
    for (int i = 0; i < nsamples; i++)
        seq->frames[i+1] = sample(cdfs[seq->frames[i]]);

    write_vid(full_vid, outfn, seq);
    printf("Finished\n");

    free_mat(buffer);
    free_mat(buffer2);
    free_vid(full_vid);

    for (int i = 0; i < nf; i++)
        free(cdfs[i]);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("Name:\n");
        printf("\tvideo_textures - create a video texture\n");
        printf("Usage:\n");
        printf("\tvideo_textures sigma samples input output\n");
        printf("Description:\n");
        printf("\t  This program lets the user compute a video texture\n");
        printf("\t  Author: Ryan Cabeen, cabeen@gmail.com\n");
        return 0;
    }

    create_texture(atof(argv[argc - 4]), atoi(argv[argc - 3]),
                   argv[argc - 2], argv[argc - 1]);

    return 0;
}
