#include <stdio.h>
#include <omp.h>

void hello()
{
#pragma omp parallel  // 申请默认数量的线程
    {
        int ID = omp_get_thread_num();// 获取线程编号
        printf("hello(%d)", ID);
        printf(" world(%d) \n", ID);
    }
}
