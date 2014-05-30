// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * multithread_utils.cuh
 *
 * @brief utilities for cpu multithreading
 */

#pragma once
#include <typeinfo>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <gunrock/util/multithreading.cuh>

namespace gunrock {
namespace util {
namespace cpu_mt {

    void sleep_millisecs(float millisecs)
    {
#ifdef _WIN32
        Sleep(DWORD(millisecs));
#else
        usleep(useconds_t(millisecs*1000));
#endif
    }

#ifdef _WIN32
    struct CPUBarrier
    {
        int* marker;
        bool reseted;
        int waken, releaseCount, count;
        CRITICAL_SECTION criticalSection;
        HANDLE barrierEvent;
    };
#else
    struct CPUBarrier
    {
        int* marker;
        bool reseted;
        int waken, releaseCount, count;
        pthread_mutex_t mutex,mutex1;
        pthread_cond_t conditionVariable;
    };
#endif

#ifdef __cplusplus
extern "C" {
#endif 
    CPUBarrier CreateBarrier(int releaseCount);

    void IncrementnWaitBarrier(CPUBarrier *barrier, int thread_num);

    void DestoryBarrier(CPUBarrier *barrier);

    //template <typename _SizeT, typename _Value>
    //void PrintArray   (const char* const name, const int gpu, const _Value* const array, const _SizeT limit = 40);
    
    //template <typename _SizeT, typename _Value>
    //void PrintGPUArray(const char* const name, const int gpu, const _Value* const array, const _SizeT limit = 40);
#ifdef __cplusplus
} //extern "C"
#endif

#ifdef _WIN32

    CPUBarrier CreateBarrier(int releaseCount)
    {
        CPUBarrier barrier;
        
        InitializeCriticalSection(&baiier.criticalSection);
        barrier.barrierEvent = CreateEvent(NULL, TRUE, FALSE, TEXT("BarrierEvent"));
        barrier.count = 0;
        barrier.waken = 0;
        barrier.releaseCount = releaseCount;
        barrier.marker = new int[releaseCount];
        barrier.reseted = false;
        memset(barrier.marker, 0, sizeof(int)*releaseCount);
        return barrier;
    }

    void IncrementnWaitBarrier(CPUBarrier *barrier, int thread_num)
    {
        bool ExcEvent=false;
        EnterCriticalSection(&barrier->criticalSection);
        if (barrier->marker[thread_num] == 0)
        {
            barrier->marker[thread_num] = 1;
            barrier->count ++;
        }
        if (barrier->count == barrier->releaseCount)
        {
            barrier->count = 0;
            memset(barrier->marker, 0, sizeof(int)*barrier->releaseCount);
            ExcEvent=true;
            barrier->reseted = false;
            barrier->waken  = 0;
        }
        LeaveCriticalSection(&barrier->criticalSection);

        if (ExcEvent) 
        {
            SetEvent(barrier->barrierEvent);
            while (barrier->waken < releaseCount-1) Sleep(1);
            ResetEvent(barrier->barrierEvent);
            barrier->reseted = true;
        } else {
            WaitForSingleObject(barrier->barrierEvent, INFINITE);
            EnterCriticalSection(&barrier->criticalSection);
            barrier->waken++;
            LeaveCriticalSection(&barrier->criticalSection);
            while (!barrier->reseted) Sleep(1);
        }
    }

    void DestoryBarrier(CPUBarrier *barrier)
    {
        delete[] barrier->marker; barrier->marker=NULL;
    }
#else

    CPUBarrier CreateBarrier(int releaseCount)
    {
        CPUBarrier barrier;
        
        barrier.count = 0;
        barrier.waken = 0;
        barrier.releaseCount = releaseCount;
        barrier.marker = new int[releaseCount];
        barrier.reseted = false;
        memset(barrier.marker, 0, sizeof(int)*releaseCount);
        
        pthread_mutex_init(&barrier.mutex, 0 );
        pthread_mutex_init(&barrier.mutex1,0);
        pthread_cond_init(&barrier.conditionVariable, 0);
        return barrier;
    }

    void ReleaseBarrier(CPUBarrier *barrier)
    {
        pthread_mutex_lock(&barrier->mutex);
        pthread_cond_signal(&barrier->conditionVariable);
        pthread_mutex_unlock(&barrier->mutex);
    }

    void IncrementnWaitBarrier(CPUBarrier *barrier, int thread_num)
    {
        bool ExcEvent=false;
        pthread_mutex_lock(&barrier->mutex1);
        if (barrier->marker[thread_num] ==0)
        {
            barrier->count++;
            barrier->marker[thread_num] = 1;
            //printf("%d: counted\n", thread_num);fflush(stdout);
        }
        if (barrier->count == barrier->releaseCount)
        {
            barrier->count = 0;
            memset(barrier->marker, 0, sizeof(int)* barrier->releaseCount);
            ExcEvent=true;
            barrier->reseted = false;
            barrier->waken   = 0;
        }
        pthread_mutex_unlock(&barrier->mutex1);

        if (ExcEvent) {
            //printf("%d: full\n", thread_num);fflush(stdout);
            pthread_mutex_lock(&barrier->mutex);
            pthread_cond_signal(&barrier->conditionVariable);
            pthread_mutex_unlock(&barrier->mutex);
            /*printf("%d: waiting\n", thread_num);fflush(stdout);
            while (true)
            {
                bool all_done=false;
                pthread_mutex_lock(&barrier->mutex1);
                if (barrier->waken == barrier->releaseCount -1)
                {
                    all_done=true;
                    barrier->reseted = true;
                }
                pthread_mutex_unlock(&barrier->mutex1);
                if (all_done) break;
                usleep(10);
            }*/
            //printf("%d: past\n", thread_num); fflush(stdout);
        } else {
            //printf("%d: waiting1\n", thread_num);fflush(stdout);
            pthread_mutex_lock(&barrier->mutex);
            //while (barrier->count !=0)
                pthread_cond_wait(&barrier->conditionVariable, &barrier->mutex);
            pthread_mutex_unlock(&barrier->mutex);
            //printf("%d: waken\n", thread_num);fflush(stdout);
            /*pthread_mutex_lock(&barrier->mutex1);
            barrier->waken++;
            pthread_mutex_unlock(&barrier->mutex1);
            printf("%d: waiting2\n", thread_num);fflush(stdout);
            while (true)
            {
                bool all_done=false; 
                pthread_mutex_lock(&barrier->mutex1);
                if (barrier->reseted) all_done=true;
                pthread_mutex_unlock(&barrier->mutex1); 
                if (all_done) break;           
                usleep(10);
            }*/
            //printf("%d: past\n", thread_num);fflush(stdout);
        }
    }

    void DestoryBarrier(CPUBarrier *barrier)
    {
        pthread_mutex_destroy(&barrier->mutex);
        pthread_mutex_destroy(&barrier->mutex1);
        pthread_cond_destroy(&barrier->conditionVariable);
        delete[] barrier->marker; barrier->marker=NULL;
    }
#endif //_WIN32

    void PrintMessage (const char* const message, const int gpu=-1, const int iteration=-1)
    {
        if (gpu!=-1 && iteration!=-1) printf("%d\t %d\t %s\n",gpu,iteration,message);
        else if (gpu!=-1) printf("%d\t \t %s\n",gpu,message);
        else if (iteration!=-1) printf("\t %d\t %s\n",iteration,message);
        else printf("\t \t %s\n",message);
        fflush(stdout);
    }

    /*template < typename T >
    struct is_same
    {
          enum { value = true };
          //typedef is_same<T,T> type;
    };


    template < typename T1, typename T2 >
    struct is_same
    {
          enum { value = false }; // is_same represents a bool.
          //typedef is_same<T1,T2> type; // to qualify as a metafunction.
    };*/

    template <typename _Value>
    void PrintValue(char* buffer, _Value val, char* prebuffer = NULL)
    {
        if (prebuffer != NULL) sprintf(buffer,"%s", prebuffer);
        else sprintf(buffer,"");
        if      (typeid(_Value) == typeid(int   ) || typeid(_Value) == typeid(unsigned int  ) ||
                 typeid(_Value) == typeid(short ) || typeid(_Value) == typeid(unsigned short ))
            sprintf(buffer,"%s%d"  ,buffer,val);
        else if (typeid(_Value) == typeid(long  ) || typeid(_Value) == typeid(unsigned long  ))
            sprintf(buffer,"%s%ld" ,buffer,val);
        else if (typeid(_Value) == typeid(long long) || typeid(_Value) == typeid(unsigned long long)) 
            sprintf(buffer,"%s%lld", buffer, val);
        else if (typeid(_Value) == typeid(float ))// || typeid(_Value) == typeid(unsigned float ))  
            sprintf(buffer,"%s%f"  ,buffer,val);
        else if (typeid(_Value) == typeid(double))// || typeid(_Value) == typeid(unsigned double))  
            sprintf(buffer,"%s%lf" ,buffer,val);
        else if (typeid(_Value) == typeid(bool  ))
            sprintf(buffer,val?"%strue":"%sfalse",buffer);
    }

    template <typename _SizeT, typename _Value>
    void PrintCPUArray(const char* const name, const _Value* const array, const _SizeT limit, const int gpu=-1, const int iteration=-1)
    {
        char *buffer = new char[1024 * 128];
        
        sprintf(buffer, "%s = ", name);

        for (_SizeT i=0;i<limit;i++) 
        {
            if (i!=0) sprintf(buffer,"%s, ",buffer);
            PrintValue(buffer,array[i],buffer);     
        }
        PrintMessage(buffer,gpu,iteration);
        delete []buffer;buffer=NULL;
    }

    template <typename _SizeT, typename _Value>
    void PrintGPUArray(const char* const name, const _Value* const array, const _SizeT limit, const int gpu=-1, const int iteration=-1)
    {
        if (limit==0) return;
        _Value* h_array = new _Value[limit];
        util::GRError(cudaMemcpy(h_array,array,sizeof(_Value) * limit, cudaMemcpyDeviceToHost), "cuaMemcpy failed", __FILE__, __LINE__);
        PrintCPUArray<_SizeT,_Value>(name,h_array,limit,gpu,iteration);
        delete[] h_array;h_array=NULL;
    }
} //namespace cpu_mt
} //namespace util
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

