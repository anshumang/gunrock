// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * ddp_ms.cu
 *
 * @brief Multi Scan using dummy data profiler
 */

//#pragma once

#include <string>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/ddp/ddp_data.cuh>
#include <gunrock/util/scan/multi_scan.cuh>
#include <gunrock/util/multithread_utils.cuh>

using namespace gunrock;
using namespace gunrock::util::scan;
using namespace gunrock::util::ddp;

template <typename T>
T div_ceiling (const T a, const T b)
{
    return (a%b)==0 ? a/b : a/b+1;
}

template <typename SizeT>
__global__ void Step2_5(
    SizeT* Length,
    SizeT* Offset,
    SizeT  Num_Rows)
{
    Offset[0]=0;
    for (int i=0;i<Num_Rows;i++)
        Offset[i+1]=Offset[i]+Length[i];
}

template <
    typename VertexId,
    typename SizeT,
    bool     EXCLUSIVE ,
    SizeT    BLOCK_SIZE,
    SizeT    BLOCK_N   >
__host__ void Dummy_Scan_with_Keys(
    Variable_Single<SizeT   >* Num_Elements,
    Variable_Single<SizeT   >* Num_Rows,
    Variable_Single<SizeT   >* Num_Associate,
    Variable_Array <VertexId>* Keys,
    Variable_Array <VertexId>* Result,
    Variable_Array <int     >* Splict,
    Variable_Array <VertexId>* Convertion,
    Variable_Array <SizeT   >* Length,
    Variable_Array2<VertexId>* Associate_in,
    Variable_Array2<VertexId>* Associate_out)
{
    if (Num_Elements->value <= 0) return;
    SizeT *History_Size = new SizeT [20];
    SizeT **d_Buffer    = new SizeT*[20];
    SizeT Current_Size  = Num_Elements->value;
    int   Current_Level = 0;
    dim3  Block_Size,Grid_Size;
    //char  message[512];
    SizeT *d_Offset;

    //sprintf(message,"%d,unknown",Num_Rows->value+1);
    //Variable_Array<SizeT> Offset1(std::string(message));
    for (int i=0;i<20;i++) d_Buffer[i]=NULL;
    //d_Buffer    [0] = Result->d_values;
    //util::cpu_mt::PrintGPUArray<int>("splict",Splict->d_values,Num_Elements->value);
    History_Size[0] = Current_Size;
    History_Size[1] = div_ceiling(Current_Size,BLOCK_SIZE);
    util::GRError(cudaMalloc(&(d_Buffer[0]),(long long) (1) * sizeof(SizeT) * History_Size[0]) ,
                 "cudaMalloc d_Buffer[0] failed", __FILE__, __LINE__);
    //printf(" 1,%lld ", (long long) (1) * sizeof(SizeT) * History_Size[1] * Num_Rows->value); fflush(stdout);

    util::GRError(cudaMalloc(&(d_Buffer[1]), (long long) (1) * sizeof(SizeT) * History_Size[1] * Num_Rows->value),
                 "cudaMalloc d_Buffer[1] failed", __FILE__, __LINE__);
    util::GRError(cudaMalloc(&(d_Offset), sizeof(SizeT) * Num_Rows->value+1), 
                 "cudaMalloc d_Offset failed", __FILE__, __LINE__);

    while (Current_Size>1 || Current_Level==0)
    {
        if (Current_Level == 0)
        {
            Step0b <VertexId, SizeT, BLOCK_N>
                <<<dim3(div_ceiling(History_Size[1],SizeT(32)),32,1),
                   dim3(BLOCK_SIZE/2,1,1), sizeof(SizeT) * BLOCK_SIZE>>> (
               History_Size[0],
               Num_Rows->value,
               History_Size[1],
               Keys  ->d_values,
               Splict->d_values,
               d_Buffer[0],
               d_Buffer[1]);
            util::GRError("Step0b failed", __FILE__, __LINE__);
        } else {
            Step1 <SizeT, BLOCK_N>
                <<<dim3(History_Size[Current_Level+1], Num_Rows->value, 1),
                   dim3(BLOCK_SIZE/2, 1, 1), sizeof(SizeT) * BLOCK_SIZE>>> (
                Current_Size,
                d_Buffer[Current_Level],
                d_Buffer[Current_Level+1]);
            util::GRError("Step1 failed", __FILE__, __LINE__);
        }

        Current_Level++;
        Current_Size = History_Size[Current_Level];
        if (Current_Size > 1)
        {
            History_Size[Current_Level +1]=div_ceiling(Current_Size, BLOCK_SIZE);
            util::GRError(cudaMalloc(&(d_Buffer[Current_Level+1]), 
                         sizeof(SizeT) * History_Size[Current_Level+1] * Num_Rows->value),
                         "cudaMalloc d_Buffer failed", __FILE__, __LINE__);
            //printf(" %d,%ld ",Current_Level+1, sizeof(SizeT) * History_Size[Current_Level+1] *Num_Rows->value);fflush(stdout);
        } 
    } //while (Current_Size>1 || Current_Level==0)

    util::GRError(cudaMemcpy(Length->d_values, d_Buffer[Current_Level], sizeof(SizeT) * Num_Rows->value, cudaMemcpyDeviceToDevice), "cudaMemcpy d_Length failed", __FILE__, __LINE__);
    //util::cpu_mt::PrintGPUArray<SizeT>("Length",Length->d_values,Num_Rows->value);
    //util::cpu_mt::PrintGPUArray<SizeT>("Level-1",d_Buffer[Current_Level-1],History_Size[Current_Level-1] * Num_Rows->value);
    
    Current_Level--;
    while (Current_Level > 1)
    {
        Step2 <SizeT> 
            <<< dim3(History_Size[Current_Level], Num_Rows->value, 1),
                dim3(BLOCK_SIZE, 1, 1)>>> (
            History_Size[Current_Level-1],
            d_Buffer[Current_Level],
            d_Buffer[Current_Level -1]);
        util::GRError("Step2 failed", __FILE__, __LINE__);
        Current_Level --;
    } // while Current_Level > 1

    Step2_5<SizeT> <<<dim3(1),dim3(1)>>> (Length->d_values, d_Offset, Num_Rows->value);
    
    Step3b <VertexId, SizeT, EXCLUSIVE>
        <<< dim3(div_ceiling(History_Size[1], SizeT(32)), 32, 1),
            dim3(BLOCK_SIZE, 1, 1)>>> (
        Num_Elements ->value,
        History_Size[1],
        Num_Associate->value,
        Keys       ->d_values,
        Splict     ->d_values,
        Convertion ->d_values,
        d_Offset,
        d_Buffer[1],
        d_Buffer[0],
        Result     ->d_values,
        Associate_in->d_values,
        Associate_out->d_values);
    util::GRError("Step3b failed", __FILE__, __LINE__);

    for (int i=0;i<20;i++)
    if (d_Buffer[i]!=NULL)
    {
        //printf(" %d ",i);fflush(stdout);
        util::GRError(cudaFree(d_Buffer[i]),
                     "cudaFree d_Buffer failed", __FILE__, __LINE__);
        d_Buffer[i]=NULL;
    }
    util::GRError(cudaFree(d_Offset),
                 "cudaFree d_Offset failed", __FILE__, __LINE__);
    delete[] d_Buffer;d_Buffer = NULL;
    delete[] History_Size;History_Size=NULL;
}

typedef int _VertexId;
typedef int _SizeT;

int main(int argc, char** argv)
{
    util::CommandLineArgs args(argc,argv);
    char buffer[512];

    DeviceInit(args);
    Variable_Single<_SizeT   > Iteration    ("sweep,linear,1,1,8");
    Variable_Single<_SizeT   > Num_Elements ("sweep,log,1,2,134217728");    
    Variable_Single<_SizeT   > Num_Rows     ("sweep,linear,1,1,4");        
    Variable_Single<_SizeT   > Num_Associate("sweep,linear,1,1,4");
    Variable_Single<_SizeT   > Block_N      ("sweep,linear,6,1,10");        
    //printf("0");fflush(stdout);
    Variable_Array <_SizeT   > Length       ("unknown,4");
    Variable_Array2<_VertexId> Associate_in ("random,4,134217728,0,134217728");
    Variable_Array2<_VertexId> Associate_out("unknown,4,134217728");
    //printf("A");fflush(stdout);
 
    for (Iteration    .reset();!Iteration    .ended;Iteration    .next())
    for (Num_Elements .reset();!Num_Elements .ended;Num_Elements .next())
    {
        sprintf(buffer,"random,%d,0,%d",Num_Elements.value,Num_Elements.value);    
        Variable_Array <_VertexId> Convertion   ;Convertion.Init(std::string(buffer));
        Variable_Array <_VertexId> Keys         ;Keys.Init(std::string(buffer));
        sprintf(buffer,"unknown,%d",Num_Elements.value);
        Variable_Array <_VertexId> Result       ;Result.Init(std::string(buffer));          
        for (Num_Rows     .reset();!Num_Rows     .ended;Num_Rows     .next())
        {
            sprintf(buffer,"random,%d,0,%d",Num_Elements.value,Num_Rows.value);
            Variable_Array <int      > Splict   ;Splict.Init(std::string(buffer));
            for (Num_Associate.reset();!Num_Associate.ended;Num_Associate.next())
            for (Block_N      .reset();!Block_N      .ended;Block_N      .next())
            {
                printf("%d\t%d\t%d\t%d\t%d\t",Iteration.value,Num_Elements.value,Num_Rows.value,Num_Associate.value,Block_N.value);
                fflush(stdout);
                util::GpuTimer gpu_timer;
                gpu_timer.Start();
                if (Block_N.value==6)
                    Dummy_Scan_with_Keys <_VertexId, _SizeT, true, 64, 6> (
                        &Num_Elements,
                        &Num_Rows,
                        &Num_Associate,
                        &Keys,
                        &Result,
                        &Splict,
                        &Convertion,
                        &Length,
                        &Associate_in,
                        &Associate_out);
                else if (Block_N.value==7) 
                    Dummy_Scan_with_Keys <_VertexId, _SizeT, true, 128, 7> (
                        &Num_Elements,
                        &Num_Rows,
                        &Num_Associate,
                        &Keys,
                        &Result,
                        &Splict,
                        &Convertion,
                        &Length,
                        &Associate_in,
                        &Associate_out);
                else if (Block_N.value==8)
                    Dummy_Scan_with_Keys <_VertexId, _SizeT, true, 256, 8> (
                        &Num_Elements,
                        &Num_Rows,
                        &Num_Associate,
                        &Keys,
                        &Result,
                        &Splict,
                        &Convertion,
                        &Length,
                        &Associate_in,
                        &Associate_out);
                else if (Block_N.value==9)
                    Dummy_Scan_with_Keys <_VertexId, _SizeT, true, 512, 9> (
                        &Num_Elements,
                        &Num_Rows,
                        &Num_Associate,
                        &Keys,
                        &Result,
                        &Splict,
                        &Convertion,
                        &Length,
                        &Associate_in,
                        &Associate_out);
                else if (Block_N.value==10)
                    Dummy_Scan_with_Keys <_VertexId, _SizeT, true, 1024, 10> (
                        &Num_Elements,
                        &Num_Rows,
                        &Num_Associate,
                        &Keys,
                        &Result,
                        &Splict,
                        &Convertion,
                        &Length,
                        &Associate_in,
                        &Associate_out);
                gpu_timer.Stop();
                float elapased = gpu_timer.ElapsedMillis();
                printf("%f\n",elapased);
            }
        }
    }
}
