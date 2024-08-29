#include <stdio.h>
#include <omp.h>
#include <vector>

void calc_pi_serial() // 串行版本
{
    long num_steps = 0x20000000;

    double step = 1.0 / (double)num_steps;
    double sum = 0.0;
    double start = omp_get_wtime( );
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    double pi = sum * step;

    printf("pi: %.16g in %.16g secs\n", pi, omp_get_wtime() - start);
    // will print "pi: 3.141592653589428 in 5.664520263002487 secs"
}

void calc_pi()
{
    omp_set_num_threads(0x20);
    long num_steps = 0x1000000;
    double step = 1.0 / (double)num_steps;
    double start = omp_get_wtime();
    double pi = 0.0;
    #pragma omp parallel
    {
        double sum = 0.0;
        int ID = omp_get_thread_num();
        for(int i = ID+1; i <=num_steps; i+=0x20)
            sum += 4.0 / (1.0 + ((i - 0.5) * step) * ((i - 0.5) * step));
        pi += sum * step;
    };
    printf("pi: %.16g in %.16g secs\n", pi, omp_get_wtime() - start);
}

//int main()
//{
//    calc_pi_serial();
//    calc_pi();
//    return 0;
//}