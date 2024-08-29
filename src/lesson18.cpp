#include "omp.h"
#include <cstdio>

#define NPOINTS 1000
#define MXITR 1000
struct d_complex {
    double r;
    double i;
};

struct d_complex c;
int numoutside = 0;

void testpoint()
{
    struct d_complex z;
    int iter;
    double temp;

    z = c;
    for(iter=0; iter<MXITR; iter++) {
        temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i=z.r*z.i*2+c.i;
        z.r=temp;
        if((z.r*z.r+z.i*z.i) > 4.0) {
            numoutside++;
            break;
        }
    }
}

void MandelBrotArea()
{
    int i, j;
    double area, error, eps = 1.0e-5;
//#pragma omp parallel for default(shared) private(c, eps)
//#pragma omp parallel for default(none) private(c, eps)
#pragma omp parallel for default(shared) private(c, eps) reduction(+:numoutside)
    for (i = 0; i < NPOINTS; i++) {
        for (j = 0; j < NPOINTS; j++) {
            c.r = -2.0 + 2.5*(double)(i) / (double)NPOINTS + eps;
            c.i = 1.125*(double)(j) / (double)NPOINTS + eps;
            testpoint();
        }
    }

    area = 2.0*2.5*1.125 * (double)(NPOINTS*NPOINTS-numoutside) / (double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;
    printf("error: %d\n", error);
}

int main(void) {
    MandelBrotArea();
    return 0;
}