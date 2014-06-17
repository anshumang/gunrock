// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rp_partitioner.cuh
 *
 * @brief Implementation of random partitioner
 */

#pragma once

#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>

namespace gunrock {
namespace app {
namespace rp {

    template <typename SizeT>
    struct sort_node
    {
    public:
        SizeT posit;
        int   value;
        
        bool operator==(const sort_node& node) const
        {
            return (node.value == value);
        }

        bool operator<(const sort_node& node) const
        {
            return (node.value < value);
        }
        
        sort_node & operator=(const sort_node &rhs)
        {
            this->posit=rhs.posit;
            this->value=rhs.value;
            return *this;
        }
    };

    template <typename SizeT>
    bool compare_sort_node(sort_node<SizeT> A, sort_node<SizeT> B)
    {
        return (A.value<B.value);
    }


template <
    typename VertexId,
    typename SizeT,
    typename Value>
struct RandomPartitioner : PartitionerBase<VertexId,SizeT,Value>
{
    typedef Csr<VertexId,Value,SizeT> GraphT;

    // Members
    float *weitage;

    // Methods
    RandomPartitioner()
    {
        weitage=NULL;
    }

    RandomPartitioner(const GraphT &graph,
                      int   num_gpus,
                      float *weitage = NULL)
    {
        Init2(graph,num_gpus,weitage);
    }

    void Init2(
        const GraphT &graph,
        int num_gpus,
        float *weitage)
    {
        //printf("Init2 begin.\n"); fflush(stdout);
        //if (weitage!=NULL) util::cpu_mt::PrintCPUArray("weitage0",weitage,num_gpus);
        this->Init(graph,num_gpus);
        //printf("Init returned.\n"); fflush(stdout);
        this->weitage=new float[num_gpus+1];
        if (weitage==NULL)
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=1.0f/num_gpus;
        else {
            float sum=0;
            for (int gpu=0;gpu<num_gpus;gpu++) sum+=weitage[gpu];
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=weitage[gpu]/sum; 
        }
        for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu+1]+=this->weitage[gpu];
        //util::cpu_mt::PrintCPUArray("weitage1", this->weitage, num_gpus);
        //printf("Init2 end.\n"); fflush(stdout);
    }

    ~RandomPartitioner()
    {
        //printf("~RandomPartitioner begin\n");fflush(stdout);
        if (weitage!=NULL)
        {
            delete[] weitage;weitage=NULL;
        }
        //printf("~RandomPartitioner end\n"); fflush(stdout);
    }

    //template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
    cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        SizeT**    &in_offsets,
        SizeT**    &out_offsets)
    {
        cudaError_t retval = cudaSuccess;
        int*        tpartition_table=this->partition_tables[0];
        time_t      t =1402580570;//= time(NULL);
        SizeT       nodes  = this->graph->nodes;
        //int*        tValue = new int  [nodes];
        //SizeT*      tPosit = new SizeT[nodes];
        sort_node<SizeT> *sort_list = new sort_node<SizeT>[nodes];

        printf("Partition begin. seed=%ld\n", t);fflush(stdout);
        //util::cpu_mt::PrintCPUArray("weitage", weitage, this->num_gpus);

        srand(t);
        /*for (SizeT node=0;node<this->graph->nodes;node++)
        {
            float x=rand()*1.0f/RAND_MAX;
            tpartition_table[node]=this->num_gpus;
            printf("%f ",x);
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            if (x<=weitage[gpu])
            {
                tpartition_table[node]=gpu;
                break;
            }
            if (tpartition_table[node]==this->num_gpus) tpartition_table[node]=(rand() % this->num_gpus);
        }
        printf("\n");*/

        for (SizeT node=0;node<nodes;node++)
        {
            sort_list[node].value=rand();
            sort_list[node].posit=node;
        }
        std::vector<sort_node<SizeT> > sort_vector(sort_list, sort_list+nodes);
        //printf("1");fflush(stdout);
        std::sort(sort_vector.begin(),sort_vector.end());//,compare_sort_node<SizeT>);
        //printf("2");fflush(stdout);
        /*for (SizeT i=0;i<nodes-1;i++)
        for (SizeT j=i+1;j<nodes;j++)
        if (sort_list[j]< sort_list[i]) {
            sort_node<SizeT> temp=sort_list[i];sort_list[i]=sort_list[j];sort_list[j]=temp;
            //int   tempi=sort_list[i].value;sort_list[i].value=sort_list[j].value;sort_list[j].value=tempi;
            //SizeT tempp=sort_list[i].posit;sort_list[i].posit=sort_list[j].posit;sort_list[j].posit=tempp;
        }*/
        //util::cpu_mt::PrintCPUArray("tPosit", tPosit, nodes);
        //util::cpu_mt::PrintCPUArray("tValue", tValue, nodes);
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        for (SizeT pos= gpu==0?0:weitage[gpu-1]*nodes; pos<weitage[gpu]*nodes; pos++)
        {
            //printf("pos = %d, tPosit = %d, gpu = %d\n", pos, tPosit[pos], gpu);
            //tpartition_table[tPosit[pos]]=gpu;
            tpartition_table[sort_vector[pos].posit]=gpu;
        }

        //delete[] tValue;tValue=NULL;
        //delete[] tPosit;tPosit=NULL;
        delete[] sort_list;sort_list=NULL;
        //printf("3");fflush(stdout);
        retval = this->MakeSubGraph
                 //<LOAD_EDGE_VALUES, LOAD_NODE_VALUES>
                 ();
        //printf("4");fflush(stdout);
        sub_graphs        = this->sub_graphs;
        partition_tables  = this->partition_tables;
        convertion_tables = this->convertion_tables;
        original_vertexes = this->original_vertexes;
        in_offsets        = this->in_offsets;
        out_offsets       = this->out_offsets;
        //printf("%p, %p, %d\n", partition_tables, partition_tables[1], partition_tables[1][0]);fflush(stdout);
        //printf("Partition end.\n");fflush(stdout);
        return retval;
    }
};

} //namespace rp
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
