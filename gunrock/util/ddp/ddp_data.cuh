// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * multi_scan.cuh
 *
 * @brief Multi Scan that splict and scan on array
 */

#pragma once

#include <string>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace util {
namespace ddp {

int break_string(std::string str,std::string* &strs)
{
    int counter=0;

    for (unsigned int i=0;i<str.length();i++)
    if (str[i]==',') counter++;

    strs=new std::string[counter+1];
    counter=0;strs[0]="";
    for (unsigned int i=0;i<str.length();i++)
    {
        if (str[i]==',') 
        {
            counter++;strs[counter]="";
        } else strs[counter]=strs[counter]+str[i];
    }
    return counter+1; 
}

template <typename DataType>
DataType get_value(std::string str)
{
    DataType x;
    std::istringstream str_stream(str);
    str_stream >> x;
    return x;
}

template <typename DataType>
DataType div_remainder(DataType a, DataType b)
{
    return (a-((long long)(a*1.0/b+1.0e-6))*b);
}

template <typename DataType>
DataType get_rand(DataType minimum,DataType maximum)
{
    DataType x=minimum+div_remainder(DataType(rand()),(maximum-minimum));
    if (maximum==0 || maximum-minimum<1 ) return minimum;
    if (x>maximum-1) x=maximum-1;
    if (x<0) x=0-x;
    //printf("%ld,%ld,%ld ",minimum,maximum,x);
    return x;
}

enum Variable_Method
{
    random,
    linear,
    fixed,
    sweep,
    _log,
    unknown
}; //enum Variable_Method

template <typename DataType>
struct Variable_Single
{
public:
    DataType value;
    DataType minimum,maximum,step;
    Variable_Method method,sweep_method;
    bool ended;
    
    Variable_Single()
    {
        minimum = 0;
        maximum = 0;
        step    = 0;
        value   = 0;
        method  = unknown;
        ended   = false;
    }

    Variable_Single(std::string settings)
    {
        Init(settings);
    }

    void Init(std::string settings)
    {
        std::string* strs;
        int counter = break_string(settings,strs);

        if (strs[0]=="random")
        {
            method  = random;
            minimum = get_value<DataType>(strs[1]);
            maximum = get_value<DataType>(strs[2]);
            value   = get_rand <DataType>(minimum,maximum);
        } else if (strs[0]=="fixed")
        {
            method  = fixed;
            value   = get_value<DataType>(strs[1]);
        } else if (strs[0]=="sweep")
        {
            method  = sweep;
            minimum = get_value<DataType>(strs[2]);
            maximum = get_value<DataType>(strs[4]);
            step    = get_value<DataType>(strs[3]);
            if (strs[1]=="log") sweep_method=_log; else sweep_method=linear;
            value   = minimum;
        } else {
            method  = unknown;
        }
        delete[] strs;strs=NULL;
        ended=false;
    }

    void reset()
    {
        if (method == random)
            value = get_rand<DataType>(minimum,maximum);
        else if (method == sweep)
        {
            value = minimum;
            ended = false;
        } 
    }

    void next()
    {
        if (method == random)
            value = get_rand<DataType>(minimum,maximum);
        else if (method == sweep)
        {
            if (sweep_method == _log)
            {
                if (value*step > maximum) ended=true;
                else value=step*value;
            } else if (sweep_method == linear)
            {
                if (value+step > maximum) ended=true;
                else value+=step;
            }
        }
    }

    bool end()
    {
        return ended;
    }
    //DataType get_value()
    //{
    //    return value;
    //}
}; //struct Variable_Single

template <typename DataType>
struct Variable_Array
{
    int      size;
    bool     dirty,has_cpu,has_gpu;
    DataType minimum,maximum,step;
    DataType *d_values,*h_values;
    Variable_Method method;

    Variable_Array()
    {
        size     = 0;
        dirty    = true;
        minimum  = 0;
        maximum  = 0;
        step     = 0;
        method   = unknown;
        d_values = NULL;
        h_values = NULL;
        has_cpu  = true;
        has_gpu  = true; 
    }

    Variable_Array(std::string settings)
    {
        d_values = NULL;
        h_values = NULL;
        Init(settings);
    }

    void Init(std::string settings)
    {
        std::string* strs;
        int counter = break_string(settings,strs);
        int base_counter;
       
        //printf("%s ",settings.c_str());
        size=get_value<DataType>(strs[1]);
        h_values=new DataType[size];
        if (strs[0]=="random")
        {
            method   = random;
            minimum  = get_value<DataType>(strs[2]);
            maximum  = get_value<DataType>(strs[3]);
            for (int i=0;i<size;i++)
            {
                //printf(" %d,",x);
                h_values[i]=get_rand<DataType>(minimum,maximum);
                //printf("%d,%d ",i,h_values[i]);
            }
            base_counter = 4;
            //printf("random %d %d\n",minimum,maximum);
        } else if (strs[0]=="fixed")
        {
            method   = fixed;
            for (int i=0;i<size;i++)
            {
                h_values[i]=get_value<DataType>(strs[i+2]);
            }
            base_counter = size+2;
        } else if (strs[0]=="linear")
        {
            method   = fixed;
            minimum  = get_value<DataType>(strs[2]);
            step     = get_value<DataType>(strs[3]);
            h_values[0]=minimum;
            for (int i=1;i<size;i++)
                h_values[i]=h_values[i-1]+step;
            base_counter = 4;
        } else if (strs[0]=="log")
        {
            method   = fixed;
            minimum  = get_value<DataType>(strs[2]);
            step     = get_value<DataType>(strs[3]);
            h_values[0]=minimum;
            for (int i=1;i<size;i++)
                h_values[i]=h_values[i-1]*step;
            base_counter = 4;
        } else {
            method = unknown;
            memset(h_values,0,sizeof(DataType)*size);
            base_counter = 2;
        }

        if (counter > base_counter)
        {
            if (strs[base_counter]=="cpu only")
            { has_cpu = true ; has_gpu = false; }
            else if (strs[base_counter]=="gpu only")
            { has_cpu = false; has_gpu = true ; }
            else { has_cpu = true; has_gpu = true; }
        } else {
            has_cpu = true;has_gpu = true;
        }
        //printf(has_cpu?"has_cpu\n":"no_cpu\n");
        
        if (has_gpu) h2d();
        if (!has_cpu)
        {
            delete[] h_values;
            h_values=NULL;
        }
        dirty=false;
        delete[] strs;strs=NULL;
    }

    ~Variable_Array()
    {
        release();
    }

    void release()
    {
        if (d_values!=NULL) 
        {
            //printf("freeing:%p,%d ",d_values,size);
            util::GRError(cudaFree(d_values),
                         "cudaFree d_values failed", __FILE__, __LINE__);
            d_values=NULL;
        }
        if (h_values!=NULL) 
        {
            delete[] h_values;
            h_values=NULL;
        }
    }

    void h2d()
    {
        if (d_values == NULL)
        {
            util::GRError(cudaMalloc((void**)&(d_values),sizeof(DataType)*size),
                         "cudaMalloc d_values failed", __FILE__, __LINE__);
            //printf("%p,%d:created ",d_values,size);fflush(stdout);
            has_gpu=true;
        }
        util::GRError(cudaMemcpy(d_values,h_values,sizeof(DataType)*size, cudaMemcpyHostToDevice),
                     "cudaMemcpy d_values failed", __FILE__, __LINE__);
        //util::cpu_mt::PrintCPUArray<DataType>("cpu value",h_values,size);
        //util::cpu_mt::PrintGPUArray<DataType>("gpu value",d_values,size);
    }

    void d2h()
    {
        if (h_values == NULL)
        {
            h_values = new DataType[size];
            has_cpu = true;
        }
        util::GRError(cudaMemcpy(h_values,d_values,sizeof(DataType)*size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy h_values failed", __FILE__, __LINE__);
    }
}; //struct Variable_Array

template <typename DataType>
struct Variable_Array2
{
    int size1;
    int size2;
    DataType **h_values;
    DataType **p_values;
    DataType **d_values;
    DataType minimum,maximum;
    Variable_Method method;
    bool has_cpu,has_gpu;

    Variable_Array2()
    {
        size1=0;
        size2=0;
        h_values=NULL;
        d_values=NULL;
        p_values=NULL;
        method=unknown;
        has_cpu =false;
        has_gpu =false;
    }

    Variable_Array2(std::string settings)
    {
        h_values=NULL;
        d_values=NULL;
        p_values=NULL;
        Init(settings);
    }

    void Init(std::string settings)
    {
        std::string *strs;
        int counter = break_string(settings,strs);
        int base_counter;
        size1    = get_value<DataType>(strs[1]);
        size2    = get_value<DataType>(strs[2]);
        h_values = new DataType*[size1];

        for (int i=0;i<size1;i++)
           h_values[i]=new DataType[size2];
        if (strs[0]=="random")
        {
            minimum  = get_value<DataType>(strs[3]);
            maximum  = get_value<DataType>(strs[4]);
            method=random;
            for (int i=0;i<size1;i++)
            for (int j=0;j<size2;j++)
                h_values[i][j]=get_rand<DataType>(minimum,maximum);
            base_counter = 5;
        } else {
            method=unknown;
            for (int i=0;i<size1;i++)
                memset(h_values[i],0,sizeof(DataType)*size2);
            base_counter = 3;
        }
        
        if (counter > base_counter)
        {
            if (strs[base_counter]=="cpu only")
            { has_cpu = true ; has_gpu = false; }
            else if (strs[base_counter]=="gpu only")
            { has_cpu = false; has_gpu = true ; }
            else { has_cpu = true; has_gpu = true; }
        } else {
            has_cpu = true;has_gpu = true;
        }

        if (has_gpu) h2d();
        if (!has_cpu)
        {
             for (int i=0;i<size1;i++)
            {
                delete[] h_values[i];h_values[i]=NULL;
            }
            delete[] h_values;h_values=NULL;
        }
        delete[] strs;strs=NULL;
    }

    void d2h()
    {
        if (h_values == NULL)
        {
            h_values = new DataType*[size1];
            for (int i=0;i<size1;i++)
                h_values[i]=new DataType[size2];
            has_cpu=true;
        }
        for (int i=0;i<size1;i++)
        {
            util::GRError(cudaMemcpy(h_values[i],p_values[i], sizeof(DataType)*size2, cudaMemcpyDeviceToHost), "cudaMemcpy h_values failed", __FILE__, __LINE__);
        }
    }

    void h2d()
    {
        if (p_values == NULL)
        {
            p_values = new DataType*[size1];
            util::GRError(cudaMalloc(&(d_values), sizeof(DataType*)*size1),
                         "cudaMalloc d_values failed.", __FILE__, __LINE__);
            for (int i=0;i<size1;i++)
            {
               util::GRError(cudaMalloc(&(p_values[i]),sizeof(DataType)*size2),
                            "cudaMalloc p_values failed.", __FILE__, __LINE__);   
            }    
            util::GRError(cudaMemcpy(d_values,p_values, sizeof(DataType*)*size1, cudaMemcpyHostToDevice), "cudaMemcpy d_values failed", __FILE__, __LINE__);
            has_gpu=true;
        }
        for (int i=0;i<size1;i++)
               util::GRError(cudaMemcpy(p_values[i],h_values[i],sizeof(DataType)*size2,cudaMemcpyHostToDevice), "cudaMemcpy p_values failed", __FILE__, __LINE__);
    }

    ~Variable_Array2()
    {
        release();
    }
    
    void release()
    {
        if (p_values!=NULL)
        {
            for (int i=0;i<size1;i++)
                util::GRError(cudaFree(p_values[i]), 
                             "cudaFree p_values failed", __FILE__, __LINE__);
            util::GRError(cudaFree(d_values),
                         "cudaFree d_values failed", __FILE__, __LINE__);
            delete[] p_values; p_values=NULL;
            d_values=NULL;
        }
        if (h_values!=NULL)
        {
            for (int i=0;i<size1;i++)
            {
                delete[] h_values[i];h_values[i]=NULL;
            }
            delete[] h_values;h_values=NULL;
        }
    }
}; //struct Variable_Array2

} //namespace ddp
} //namespace util
} //namespace gunrock
