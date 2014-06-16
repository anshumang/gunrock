// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <gunrock/app/rp/rp_partitioner.cuh>
#include <gunrock/util/ddp/ddp_data.cuh>
#include <gunrock/util/multithread_utils.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::util::ddp;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

template <typename T>
T div_ceiling(const T a, const T b)
{
    return (a%b)==0? a/b: a/b+1;
}

template <
    typename SizeT,
    typename VertexId,
    typename Value>
struct DBFS_ThreadSlice
{
    int                               num_gpus;
    int*                              gpu_idx;
    int                               thread_num;
    CUTThread                         thread_Id;
    Csr<VertexId,Value,SizeT>*        sub_graph;
    volatile int**                    dones;
             int**                    d_dones;
    cudaError_t*                      retvals;
    int*                              iterations;
    int                               edge_map_grid_size;
    int                               vertex_map_grid_size;
    int                               num_associate;
    int*                              partition_table;
    VertexId*                         convertion_table;
    SizeT                             init_size;
    VertexId                          src;
    SizeT**                           in_offsets;
    SizeT**                           out_offsets;
    SizeT**                           in_lengths;
    cudaEvent_t*                      throttle_events;
    unsigned long long*               total_queued;
    unsigned long long*               total_runtimes;
    unsigned long long*               total_lifetimes;
    Variable_Array <VertexId>*        keys_in;
    Variable_Array2<VertexId>*        associate_in;
    util::cpu_mt::CPUBarrier*         cpu_barrier;
    util::CtaWorkProgressLifetime*    work_progress;
    util::KernelRuntimeStatsLifetime* edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime* vertex_map_kernel_stats;
};

template <
    bool     INSTRUMENT,
    typename EdgeMapPolicy,
    typename VertexMapPolicy,
    typename BFSProblem>
static CUT_THREADPROC DBFS_Thread(
    void * thread_data_)
{
    typedef typename BFSProblem::SizeT    SizeT;
    typedef typename BFSProblem::VertexId VertexId;
    typedef typename BFSProblem::Value    Value;
    typedef typename BFSProblem::DataSlice DataSlice;
    typedef BFSFunctor<
        VertexId,
        SizeT,
        VertexId,
        BFSProblem> BfsFunctor;

    DBFS_ThreadSlice<SizeT,VertexId,Value>* 
                        thread_data     = (DBFS_ThreadSlice<SizeT,VertexId,Value>*) thread_data_;
    Csr<VertexId, Value, SizeT>*
                        sub_graph       =   thread_data->sub_graph;
    int                 thread_num      =   thread_data->thread_num;
    util::cpu_mt::PrintMessage("thread_started",thread_num);
    int*                gpu_idx         =   thread_data->gpu_idx;
    int                 num_gpus        =   thread_data->num_gpus;
    volatile int**      dones           =   thread_data->dones;
    int*                d_done          =  (thread_data->d_dones                [thread_num]);
    int              edge_map_grid_size =   thread_data->edge_map_grid_size;
    int            vertex_map_grid_size =   thread_data->vertex_map_grid_size;
    int                 num_associate   =   thread_data->num_associate;
    SizeT**             in_offsets      =   thread_data->in_offsets;
    SizeT**             out_offsets     =   thread_data->out_offsets;
    SizeT**             in_lengths      =   thread_data->in_lengths;
    SizeT*              out_offset      = new SizeT[num_gpus+1];
    char*               buffer          = new char [1024];
    VertexId            queue_index     = 0;
    int                 selector        = 0;
    SizeT               num_elements    =   thread_data->init_size;
    SizeT               queue_length    = 0;
    bool                queue_reset     = true;
    int*                iteration       = &(thread_data->iterations             [thread_num]);
    cudaError_t*        retvals         =   thread_data->retvals;
    cudaError_t*        retval          = &(thread_data->retvals                [thread_num]);
    cudaEvent_t*        throttle_event  = &(thread_data->throttle_events        [thread_num]);
    unsigned long long* total_queued    = &(thread_data->total_queued           [thread_num]);
    unsigned long long* total_runtimes  = &(thread_data->total_runtimes         [thread_num]);
    unsigned long long* total_lifetimes = &(thread_data->total_lifetimes        [thread_num]);
    Variable_Array <VertexId>* keys_in  =   thread_data->keys_in;
    Variable_Array2<VertexId>* 
                        associate_in    =   thread_data->associate_in;
    util::cpu_mt::CPUBarrier* 
                        cpu_barrier     =   thread_data->cpu_barrier;
    util::CtaWorkProgressLifetime*
                        work_progress   = &(thread_data->work_progress          [thread_num]);
    util::KernelRuntimeStatsLifetime*
                  edge_map_kernel_stats = &(thread_data->  edge_map_kernel_stats[thread_num]);
    util::KernelRuntimeStatsLifetime*
                vertex_map_kernel_stats = &(thread_data->vertex_map_kernel_stats[thread_num]);
    util::scan::MultiScan<VertexId,SizeT,true,256,8>*
                        scaner          = new util::scan::MultiScan<VertexId,SizeT,true,256,8>;
    VertexId*           h_cur_queue     = new VertexId[sub_graph->edges];
    SizeT          frontier_elements[2] = {sub_graph->edges*3,sub_graph->edges*3};
    DataSlice           h_data_slice;
    DataSlice*          d_data_slice    = NULL;
    unsigned char*      d_visited_mask  = NULL;
  
    //sub_graph->DisplayGraph("sub_graph"); 
    util::cpu_mt::PrintMessage("assigment finished",thread_num); 
    do {
        if (retval[0] = util::GRError(cudaSetDevice(gpu_idx[thread_num]), 
                          "DBFSThread cudaSetDevice failed.", __FILE__, __LINE__)) break;
        //sprintf(buffer,"unknown,1");
        //Variable_Array <typename BFSProblem::DataSlice> data_slice;data_slice.Init(std::string(buffer));
        if (retval[0] = util::GRError(cudaMalloc((void**)&(d_data_slice),sizeof(DataSlice)), 
                         "cudaMalloc d_data_slice failed", __FILE__, __LINE__)) break;
        sprintf(buffer,"unknown,%d,%d,gpu_only",num_associate,sub_graph->nodes);
        Variable_Array2<VertexId> associate_org   ;associate_org   .Init(std::string(buffer));
        sprintf(buffer,"unknown,%d,%d,gpu_only",num_associate,out_offsets[thread_num][num_gpus]-out_offsets[thread_num][1]);
        Variable_Array2<VertexId> associate_out   ;associate_out   .Init(std::string(buffer));
        sprintf(buffer,"unknown,%d",num_gpus+1);
        Variable_Array <SizeT   > out_length      ;out_length      .Init(std::string(buffer));
        sprintf(buffer,"unknown,2,%d,gpu only",frontier_elements[0]);
        Variable_Array2<VertexId> keys            ;keys            .Init(std::string(buffer));
        Variable_Array2<Value   > values          ;values          .Init(std::string(buffer));
        sprintf(buffer,"unknown,%d,gpu only",sub_graph->nodes);
        Variable_Array <SizeT   > column_indices  ;column_indices  .Init(std::string(buffer));
        Variable_Array <int     > partition_table ;partition_table .Init(std::string(buffer));
        Variable_Array <VertexId> convertion_table;convertion_table.Init(std::string(buffer));
        column_indices  .h_values=sub_graph  ->column_indices  ;column_indices  .has_cpu=true;
        column_indices  .h2d();
        partition_table .h_values=thread_data->partition_table ;partition_table .has_cpu=true;
        partition_table .h2d();
        convertion_table.h_values=thread_data->convertion_table;convertion_table.has_cpu=true;
        convertion_table.h2d();
        
        h_data_slice.d_labels    = associate_org.p_values[0];
        if (num_associate>1) h_data_slice.d_preds= associate_org.p_values[1];
        else h_data_slice.d_preds=NULL;
        h_data_slice.d_visited_mask = NULL;
        if (retval[0] = util::GRError(cudaMemcpy(d_data_slice,&h_data_slice,sizeof(DataSlice), cudaMemcpyHostToDevice), "cudaMemcpy data_slice failed", __FILE__, __LINE__)) break;
        util::MemsetKernel<<<128,128>>>(associate_org.p_values[0],-1, sub_graph->nodes);
        if (num_associate>1) util::MemsetKernel<<<128,128>>>(associate_org.p_values[1],-2,sub_graph->nodes);
        if (num_elements!=0)
        {
            //printf("selector = %d\n",selector);
            //util::cpu_mt::PrintGPUArray<VertexId>("0:key",keys.p_values[selector],1,thread_num,0);
            if (retval [0] = util::GRError(cudaMemcpy(keys.p_values[selector],&(thread_data->src),sizeof(VertexId),cudaMemcpyHostToDevice),"cudaMemcpy src failed", __FILE__, __LINE__)) break;
            Value temp=0;
            if (retval [0] = util::GRError(cudaMemcpy(associate_org.p_values[0]+thread_data->src,&temp,sizeof(Value), cudaMemcpyHostToDevice), "cudaMemcpy src failed", __FILE__, __LINE__)) break;
        }
        util::cpu_mt::PrintMessage("init finished",thread_num);
        util::cpu_mt::PrintCPUArray<SizeT>("frontier_elements",frontier_elements,2,thread_num);

        while (!All_Done(dones,retvals,num_gpus))
        {
            //util::cpu_mt::PrintMessage("iteration begin",thread_num,iteration[0]);
            if (retval[0] = work_progress->SetQueueLength(queue_index+1,0)) break;
            if (retval[0] = work_progress->GetQueueLength(queue_index,queue_length)) break;
            //util::cpu_mt::PrintGPUArray<int     >("0: d_done",d_done,1,thread_num,iteration[0]); 
            //util::cpu_mt::PrintGPUArray<VertexId>("0: in_keys",keys.p_values[selector  ],queue_reset?num_elements:queue_length,thread_num,iteration[0]);
            //util::cpu_mt::PrintGPUArray<Value   >("0:out_vals",values.p_values[selector^1],frontier_elements[selector^1],thread_num,iteration[0]);
            //util::cpu_mt::PrintGPUArray<VertexId>("0:out_keys",keys.p_values[selector^1],frontier_elements[selector^1],thread_num,iteration[0]);
            //util::cpu_mt::PrintGPUArray<VertexId>("0:labels  ",h_data_slice.d_labels,sub_graph->nodes,thread_num,iteration[0]);
            //util::cpu_mt::PrintGPUArray<VertexId>("0:edges   ",column_indices.d_values,sub_graph->edges,thread_num,iteration[0]);

            gunrock::oprtr::edge_map_forward::Kernel
                <EdgeMapPolicy, BFSProblem, BfsFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>> (
                queue_reset,
                queue_index,
                1,
                iteration[0],
                num_elements,
                d_done,
                keys  .p_values[selector],
                values.p_values[selector^1],
                keys  .p_values[selector^1],
                column_indices.d_values,
                d_data_slice,
                work_progress[0],
                frontier_elements[selector],
                frontier_elements[selector^1],
                edge_map_kernel_stats[0]);
            if (retval[0] = util::GRError(cudaThreadSynchronize(), "edge_map_forward failed", __FILE__, __LINE__)) break;
            //util::cpu_mt::PrintMessage("edge_map_forward finished",thread_num,iteration[0]);

            if (queue_reset) queue_reset = false;
            cudaEventQuery(throttle_event[0]);
            queue_index++;
            selector ^=1;
            //util::cpu_mt::PrintMessage("1",thread_num,iteration[0]);
            if (retval[0] = work_progress->SetQueueLength(queue_index+1,0)) break;
            //util::cpu_mt::PrintMessage("2",thread_num,iteration[0]);
            if (retval[0] = edge_map_kernel_stats->Accumulate(
                edge_map_grid_size,
                total_runtimes[0],
                total_lifetimes[0])) break;
            if (All_Done(dones,retvals,num_gpus))
                {util::cpu_mt::PrintMessage("done",thread_num);break;}
            //util::cpu_mt::PrintMessage("vertex_map begin",thread_num,iteration[0]);

            gunrock::oprtr::vertex_map::Kernel
                <VertexMapPolicy, BFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>> (
                iteration[0]+1,
                queue_reset,
                queue_index,
                1,
                num_elements,
                d_done,
                keys  .p_values[selector],
                values.p_values[selector],
                keys  .p_values[selector^1],
                d_data_slice,
                d_visited_mask,
                work_progress[0],
                frontier_elements[selector],
                frontier_elements[selector^1],
                vertex_map_kernel_stats[0]);
            if (retval[0] = util::GRError(cudaThreadSynchronize(),"vertex_map failed", __FILE__, __LINE__)) break;
            //util::cpu_mt::PrintMessage("vertex_map finished",thread_num,iteration[0]);

            cudaEventQuery(throttle_event[0]);
            queue_index++;
            selector ^=1;
            if (retval[0] = work_progress->GetQueueLength(queue_index, queue_length)) break;
            total_queued[0]+=queue_length;
            if (retval[0] = work_progress->SetQueueLength(queue_index+1,0)) break;
            if (retval[0] = vertex_map_kernel_stats->Accumulate(
                vertex_map_grid_size,
                total_runtimes[0],
                total_lifetimes[0])) break;
            if (All_Done(dones,retvals,num_gpus)) 
                {util::cpu_mt::PrintMessage("done",thread_num);break;}

            in_lengths[thread_num][0]=0;
            if (retval[0] = work_progress->GetQueueLength(queue_index, num_elements)) break;
            if (num_elements > 0)
            {
                dones[thread_num][0]=-1;
                scaner->Scan_with_Keys(
                    num_elements,
                    num_gpus,
                    num_associate,
                    keys            .p_values[selector],
                    keys            .p_values[selector^1],
                    partition_table .d_values,
                    convertion_table.d_values,
                    out_length      .d_values,
                    associate_org   .d_values,
                    associate_out   .d_values);
                out_length.d2h();
                out_offset[0]=0;
                for (int i=0;i<num_gpus;i++)
                    out_offset[i+1]=out_offset[i]+out_length.h_values[i];
                util::cpu_mt::PrintMessage("multi_scan finished",thread_num,iteration[0]);
                util::cpu_mt::PrintCPUArray("out_length",out_length.h_values,num_gpus,thread_num,iteration[0]);

                queue_index++;
                selector ^=1;
                if (iteration[0]!=0)
                {
                    if (All_Done(dones,retvals,num_gpus))
                        {util::cpu_mt::PrintMessage("done",thread_num);break;}
                    util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),thread_num);
                }
                
                for (int peer=0;peer<num_gpus;peer++)
                {
                    if (peer == thread_num) continue;
                    int peer_ = peer<thread_num? peer+1    : peer;
                    int gpu_  = peer<thread_num? thread_num: thread_num+1;
                    in_lengths[peer][gpu_]=out_length.h_values[peer_];
                    if (out_length.h_values[peer_] == 0) continue;
                    dones[peer][0]=-1;
                    
                    //printf("%d->%d: offset=%d length=%d\n",thread_num,peer,in_offsets[peer][gpu_],out_length.h_values[peer_]);fflush(stdout);
                    //util::cpu_mt::PrintGPUArray<VertexId>("outting",keys.p_values[selector]+out_offset[peer_],out_length.h_values[peer_],thread_num,iteration[0]);

                    if (retval[0] = util::GRError(cudaMemcpy(
                        keys_in[peer].d_values  + in_offsets [peer      ][gpu_ ],
                        keys.p_values[selector] + out_offset [peer_],
                        sizeof(VertexId) * out_length.h_values[peer_], cudaMemcpyDefault),
                        "cudaMemcpy keys failed", __FILE__, __LINE__)) break;
                    //printf("%d->%d: keys moved\n",thread_num,peer);fflush(stdout);
                    for (int i=0;i<num_associate;i++)
                    {
                        if (retval[0] = util::GRError(cudaMemcpy(
                            associate_in[peer].p_values[i] + in_offsets[peer][gpu_],
                            associate_out     .p_values[i] + out_offset[peer_]-out_offset[1],
                            sizeof(VertexId) * out_length.h_values[peer_], cudaMemcpyDefault),
                            "cudaMemcpy associate failed", __FILE__, __LINE__)) break;
                        //util::cpu_mt::PrintGPUArray<VertexId>("outting1",associate_out.p_values[i]+(out_offset[peer_]-out_offset[1]),out_length.h_values[peer_],thread_num,iteration[0]);
                    }
                    if (retval[0]) break;
                } // for peer
                if (retval[0]) break;
                util::cpu_mt::PrintMessage("data movement finished",thread_num,iteration[0]);
            } else {
                if (iteration[0]!=0)
                {
                    if (All_Done(dones,retvals,num_gpus))
                        {util::cpu_mt::PrintMessage("done",thread_num);break;}
                    util::cpu_mt::PrintMessage("waiting1",thread_num,iteration[0]);
                    util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]), thread_num);
                    util::cpu_mt::PrintMessage("past1",thread_num,iteration[0]);
                }
                for (int peer=0;peer<num_gpus;peer++)
                {
                   if (peer == thread_num) continue;
                   int gpu_ = peer<thread_num? thread_num: thread_num+1;
                   in_lengths[peer][gpu_]=0;
                }
                out_length.h_values[0]=0;
                util::cpu_mt::PrintMessage("data movement skiped",thread_num,iteration[0]);
            } // if (num_elements > 0)

            if (All_Done(dones,retvals,num_gpus))
                {util::cpu_mt::PrintMessage("done",thread_num);break;}
            util::cpu_mt::PrintMessage("waiting0",thread_num,iteration[0]);
            util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[0]),thread_num);
            util::cpu_mt::PrintMessage("past0",thread_num,iteration[0]);

            util::cpu_mt::PrintCPUArray<SizeT   >("in_leng",in_lengths[thread_num],num_gpus,thread_num,iteration[0]);
            /*if (in_lengths[thread_num][1]!=0)
            {
                util::cpu_mt::PrintGPUArray<VertexId>("keys_in",keys_in[thread_num].d_values,in_lengths[thread_num][1],thread_num,iteration[0]);
                util::cpu_mt::PrintGPUArray<Value   >("vals_in",associate_in[thread_num].p_values[0],in_lengths[thread_num][1],thread_num,iteration[0]);
            }*/
            SizeT total_length=out_length.h_values[0];
            for (int peer=0;peer<num_gpus;peer++)
            {
                if (peer==thread_num) continue;
                int peer_ = peer<thread_num? peer+1:peer;
                if (in_lengths[thread_num][peer_]==0) continue;
                //printf("in_length = %d, in_offsets = %d, num_associate = %d\n",
                //    in_lengths[thread_num][peer_], in_offsets[thread_num][peer_],num_associate);fflush(stdout);
                Expand_Incoming <VertexId, SizeT, BFSProblem::MARK_PREDECESSORS>
                    <<<div_ceiling<int>(in_lengths[thread_num][peer_],256), 256>>> (
                    in_lengths    [thread_num][peer_],
                    num_associate,
                    in_offsets    [thread_num][peer_],
                    keys_in       [thread_num].d_values,
                    keys.p_values [selector] + total_length,
                    associate_in  [thread_num].d_values,
                    associate_org .d_values);
                if (retval[0] = util::GRError(cudaDeviceSynchronize(),"Expand_Incoming failed", __FILE__, __LINE__)) break;
                total_length+=in_lengths[thread_num][peer_];
            }
            if (retval[0]) break;
            if (retval[0] = work_progress->SetQueueLength(queue_index,total_length)) break;
            if (total_length !=0)
            {
                dones[thread_num][0]=-1;
            } else {
                dones[thread_num][0]=0;
            }
            util::cpu_mt::PrintMessage("iteration finished",thread_num,iteration[0]);
            iteration[0]++;
        } //while (!All_Done)

        //util::cpu_mt::PrintMessage("outted from main loop",thread_num);
        /*bool overflowed = false;
        if (retval[0] = work_progress -> CheckOverflow<SizeT>(overflowed)) break;
        if (overflowed) {
            retval[0] = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizeing factor", __FILE__, __LINE__);
            break;
        }*/
        h_data_slice    .d_labels=NULL;
        h_data_slice    .d_preds =NULL;
        column_indices  .h_values=NULL;column_indices  .has_cpu=false;
        partition_table .h_values=NULL;partition_table .has_cpu=false;
        convertion_table.h_values=NULL;convertion_table.has_cpu=false;
        convertion_table.release();//util::cpu_mt::PrintMessage("convertion_table released",thread_num);
        partition_table .release();//util::cpu_mt::PrintMessage("partition_table  released",thread_num);
        out_length      .release();//util::cpu_mt::PrintMessage("out_length released"      ,thread_num);
        associate_out   .release();//util::cpu_mt::PrintMessage("associate_out released"   ,thread_num);
        associate_org   .release();//util::cpu_mt::PrintMessage("associate_org released"   ,thread_num);
        keys            .release();//util::cpu_mt::PrintMessage("keys released"            ,thread_num);
        values          .release();//util::cpu_mt::PrintMessage("values released"          ,thread_num);
        column_indices  .release();//util::cpu_mt::PrintMessage("column_indices released"  ,thread_num);    
        //util::cpu_mt::PrintMessage("quiting while",thread_num);
    } while (0);

    for (int peer=0;peer<num_gpus;peer++)
    {
        if (peer==thread_num) continue;
        int gpu_=thread_num<peer?thread_num+1:thread_num;
        in_lengths[peer][gpu_]=0;
    }
    util::cpu_mt::ReleaseBarrier(&(cpu_barrier[0]));
    util::cpu_mt::ReleaseBarrier(&(cpu_barrier[1]));
    delete[] h_cur_queue;h_cur_queue= NULL;
    delete[] out_offset; out_offset = NULL;
    delete[] buffer;     buffer     = NULL;
    delete   scaner;     scaner     = NULL;
    util::cpu_mt::PrintMessage("thread_end",thread_num);
    CUT_THREADEND;
}

template <
    typename SizeT,
    typename VertexId,
    typename Value>
void Dummy_BFS(
    Csr<VertexId,Value,SizeT> &graph,
    VertexId src,
    int  num_gpus,
    int* gpu_idx,
    int  max_grid_size = 0)
{
    typedef BFSProblem<
        VertexId,
        SizeT,
        Value,
        false,
        false,
        false> BfsProblem;
    typedef gunrock::oprtr::vertex_map::KernelPolicy<
        BfsProblem,                         // Problem data type
        300,                                // CUDA_ARCH
        true,                               // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
        VertexMapPolicy;

    typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
        BfsProblem,                         // Problem data type
        300,                                // CUDA_ARCH
        true,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        6,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7>                                  // LOG_SCHEDULE_GRANULARITY
        EdgeMapPolicy;


    Csr<VertexId,Value,SizeT> *sub_graphs = NULL;
    int**             partition_tables  = NULL;
    SizeT**           convertion_tables = NULL;
    SizeT**           in_offsets        = NULL;
    SizeT**           out_offsets       = NULL;
    util::cpu_mt::CPUBarrier cpu_barrier[2];
    DBFS_ThreadSlice<SizeT,VertexId,Value>* 
                      thread_slices     = new DBFS_ThreadSlice<SizeT,VertexId,Value>[num_gpus];
    CUTThread*        thread_Ids        = new CUTThread       [num_gpus];
    cudaError_t*      retvals           = new cudaError_t     [num_gpus];
    volatile int**    dones             = new volatile int*   [num_gpus];
    int**             d_dones           = new          int*   [num_gpus];
    int*              iterations        = new int             [num_gpus];
    int               num_associate     = 1;
    SizeT**           in_lengths        = new SizeT*            [num_gpus];
    for (int gpu=0;gpu<num_gpus;gpu++) in_lengths[gpu]=new SizeT[num_gpus];
    cudaEvent_t*      throttle_events   = new cudaEvent_t       [num_gpus];
    unsigned long long* total_queued    = new unsigned long long[num_gpus];
    unsigned long long* total_runtimes  = new unsigned long long[num_gpus];
    unsigned long long* total_lifetimes = new unsigned long long[num_gpus];
    Variable_Array <SizeT   >* row_offsets = new Variable_Array<SizeT>[num_gpus];
    Variable_Array <VertexId>* keys_in  = new Variable_Array <VertexId>[num_gpus];
    Variable_Array2<VertexId>* associate_in 
                                        = new Variable_Array2<VertexId>[num_gpus];
    util::CtaWorkProgressLifetime* work_progress
                                        = new util::CtaWorkProgressLifetime   [num_gpus];
    util::KernelRuntimeStatsLifetime* edge_map_kernel_stats
                                        = new util::KernelRuntimeStatsLifetime[num_gpus];
    util::KernelRuntimeStatsLifetime* vertex_map_kernel_stats
                                        = new util::KernelRuntimeStatsLifetime[num_gpus];
    char buffer[512];

    app::rp::RandomPartitioner<VertexId,SizeT,Value> partitioner(graph,num_gpus);
    partitioner.Partition(
        sub_graphs,
        partition_tables,
        convertion_tables,
        in_offsets,
        out_offsets);

    cpu_barrier[0] = util::cpu_mt::CreateBarrier(num_gpus);
    cpu_barrier[1] = util::cpu_mt::CreateBarrier(num_gpus);

    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        retvals[gpu]=cudaSuccess;
        if (util::GRError(cudaSetDevice(gpu_idx[gpu]),"cudaSetDevice failed", __FILE__, __LINE__)) return;
        CudaProperties cuda_props;
        int edge_map_grid_size   = max_grid_size<=0? cuda_props.device_props.multiProcessorCount* EdgeMapPolicy  ::CTA_OCCUPANCY : max_grid_size;
        int vertex_map_grid_size = max_grid_size<=0? cuda_props.device_props.multiProcessorCount* VertexMapPolicy::CTA_OCCUPANCY : max_grid_size;
        sprintf(buffer,"unknown,%d,gpu_only",sub_graphs[gpu].nodes+1);
        row_offsets[gpu].Init(std::string(buffer));
        row_offsets[gpu].h_values=sub_graphs[gpu].row_offsets;
        row_offsets[gpu].has_cpu =true;
        row_offsets[gpu].h2d();
        sprintf(buffer,"unknown,%d,gpu_only",in_offsets[gpu][num_gpus]);
        keys_in[gpu].Init(std::string(buffer));
        sprintf(buffer,"unknown,%d,%d,gpu_only",num_associate,in_offsets[gpu][num_gpus]);
        associate_in[gpu].Init(std::string(buffer));
        int flags = cudaHostAllocMapped;
        work_progress[gpu].Setup();
        if (util::GRError(cudaHostAlloc((void**)&(dones[gpu]), sizeof(int), flags),
                         "cudaHostAlloc done failed", __FILE__, __LINE__)) break;
        if (util::GRError(cudaHostGetDevicePointer((void**)&(d_dones[gpu]), (void*) dones[gpu],0),
                         "cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;
        if (util::GRError(cudaEventCreateWithFlags(&(throttle_events[gpu]), cudaEventDisableTiming),
                         "cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
        if (edge_map_kernel_stats[gpu].Setup(edge_map_grid_size)) break;
        if (vertex_map_kernel_stats[gpu].Setup(vertex_map_grid_size)) break;
        
        iterations     [gpu]                    = 0;
        total_runtimes [gpu]                    = 0;
        total_lifetimes[gpu]                    = 0;
        total_queued   [gpu]                    = 0;
        dones          [gpu][0]                 = -1;
        cudaChannelFormatDesc row_offset_desc = cudaCreateChannelDesc<SizeT>();
        if (util::GRError(cudaBindTexture(
            0,
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
            row_offsets[gpu].d_values,
            row_offset_desc,
            (sub_graphs[gpu].nodes+1)*sizeof(SizeT)),
            "cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

        if (gpu == partition_tables[0][src])
        {
            thread_slices[gpu].init_size        = 1;
            thread_slices[gpu].src              = convertion_tables[0][src];
        } else {
            thread_slices[gpu].init_size        = 0;
            thread_slices[gpu].src              = 0;
        }
        thread_slices[gpu].num_gpus             = num_gpus;
        thread_slices[gpu].gpu_idx              = gpu_idx;
        thread_slices[gpu].thread_num           = gpu;
        thread_slices[gpu].sub_graph            = &(sub_graphs[gpu]);
        thread_slices[gpu].dones                = dones;
        thread_slices[gpu].d_dones              = d_dones;
        thread_slices[gpu].retvals              = retvals;
        thread_slices[gpu].iterations           = iterations;
        thread_slices[gpu].edge_map_grid_size   = edge_map_grid_size;
        thread_slices[gpu].vertex_map_grid_size = vertex_map_grid_size;
        thread_slices[gpu].num_associate        = num_associate;
        thread_slices[gpu].partition_table      = partition_tables[gpu+1];
        thread_slices[gpu].convertion_table     = convertion_tables[gpu+1];
        thread_slices[gpu].in_offsets           = in_offsets;
        thread_slices[gpu].out_offsets          = out_offsets;
        thread_slices[gpu].in_lengths           = in_lengths;
        thread_slices[gpu].throttle_events      = throttle_events;
        thread_slices[gpu].total_queued         = total_queued;
        thread_slices[gpu].total_runtimes       = total_runtimes;
        thread_slices[gpu].total_lifetimes      = total_lifetimes;
        thread_slices[gpu].keys_in              = keys_in;
        thread_slices[gpu].associate_in         = associate_in;
        thread_slices[gpu].cpu_barrier          = cpu_barrier;
        thread_slices[gpu].work_progress        = work_progress;
        thread_slices[gpu].edge_map_kernel_stats= edge_map_kernel_stats;
        thread_slices[gpu].vertex_map_kernel_stats=vertex_map_kernel_stats;
    }
    
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        //printf("%d: thread_starting\n",gpu);fflush(stdout);
        thread_slices[gpu].thread_Id            = cutStartThread((CUT_THREADROUTINE)&(DBFS_Thread<true,EdgeMapPolicy,VertexMapPolicy,BfsProblem>),(void*)&(thread_slices[gpu]));
        thread_Ids   [gpu]                      = thread_slices[gpu].thread_Id;
    }

    cutWaitForThreads(thread_Ids,num_gpus);
    //printf("Cleanning begin\n");fflush(stdout);
    util::cpu_mt::DestoryBarrier(cpu_barrier);
    util::cpu_mt::DestoryBarrier(cpu_barrier+1);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        if (util::GRError(cudaSetDevice(gpu_idx[gpu]),"cudaSetDevice failed",__FILE__, __LINE__)) break;
        if (retvals[gpu]!=cudaSuccess) break;
        row_offsets[gpu].h_values=NULL;row_offsets[gpu].has_cpu=false;
        row_offsets[gpu].release();
        keys_in[gpu].release();
        associate_in[gpu].release();
        if (util::GRError(cudaFreeHost((void*)(dones[gpu])), "cudaFreeHost done failed", __FILE__, __LINE__)) break;
        if (util::GRError(cudaEventDestroy(throttle_events[gpu]), "cudaEventDestory throttle_events failed", __FILE__, __LINE__)) break;
        thread_slices[gpu].gpu_idx           = NULL;
        thread_slices[gpu].sub_graph         = NULL;
        thread_slices[gpu].dones             = NULL;
        thread_slices[gpu].d_dones           = NULL;
        thread_slices[gpu].retvals           = NULL;
        thread_slices[gpu].iterations        = NULL;
        thread_slices[gpu].partition_table   = NULL;
        thread_slices[gpu].convertion_table  = NULL;
        thread_slices[gpu].in_offsets        = NULL;
        thread_slices[gpu].out_offsets       = NULL;
        thread_slices[gpu].in_lengths        = NULL;
        thread_slices[gpu].throttle_events   = NULL;
        thread_slices[gpu].total_queued      = NULL;
        thread_slices[gpu].total_runtimes    = NULL;
        thread_slices[gpu].total_lifetimes   = NULL;
        thread_slices[gpu].keys_in           = NULL;
        thread_slices[gpu].associate_in      = NULL;
        thread_slices[gpu].cpu_barrier       = NULL;
        thread_slices[gpu].work_progress     = NULL;
        thread_slices[gpu].edge_map_kernel_stats = NULL;
        thread_slices[gpu].vertex_map_kernel_stats = NULL;
        //printf("%d: cleaned\n",gpu);fflush(stdout);
    }
    delete[] dones          ;dones           = NULL;
    delete[] throttle_events;throttle_events = NULL;
    delete[] thread_Ids     ;thread_Ids      = NULL;
    delete[] thread_slices  ;thread_slices   = NULL;
}

typedef int _SizeT;
typedef int _VertexId;
typedef int _Value;

int main()
{
    _SizeT max_nodes   = 131072;
    _SizeT maxa_degree = 8;
    int    num_gpus    = 2;
    int    gpu_idx[2]  = {0,1};

    srand(time(NULL));
    Variable_Single<_SizeT> Num_Nodes;
    Variable_Single<_SizeT> Avg_Degree;
    char buffer[512];
    
    sprintf(buffer,"sweep,log,131072,2,%d",max_nodes);
    Num_Nodes .Init(std::string(buffer));
    sprintf(buffer,"sweep,log,8,2,%d",maxa_degree);
    Avg_Degree.Init(std::string(buffer));
    for (Num_Nodes .reset();!Num_Nodes .ended;Num_Nodes .next())
    for (Avg_Degree.reset();!Avg_Degree.ended;Avg_Degree.next())
    {
        Variable_Array <_SizeT   > Row_Offset;
        Variable_Array <_VertexId> Column_Index;
        Variable_Single<_VertexId> Source_Id;
        Csr<_VertexId,_Value,_SizeT> Graph;

        sprintf(buffer,"random,0,%d",Num_Nodes.value);
        Source_Id.Init(std::string(buffer));
        sprintf(buffer,"random,%d,0,%d,cpu only",Num_Nodes.value+1,Avg_Degree.value*2);
        Row_Offset.Init(std::string(buffer));
        //util::cpu_mt::PrintCPUArray("row_offsets",Row_Offset.h_values,Num_Nodes.value+1);
        Row_Offset.h_values[0]=0;
        for (_SizeT i=0;i<Num_Nodes.value;i++)
            Row_Offset.h_values[i+1]+=Row_Offset.h_values[i];
        sprintf(buffer,"random,%d,0,%d,cpu only",Row_Offset.h_values[Num_Nodes.value],Num_Nodes.value-1);
        Column_Index.Init(std::string(buffer));
        Graph.nodes          = Num_Nodes .value;
        Graph.edges          = Row_Offset.h_values[Num_Nodes.value];
        Graph.row_offsets    = Row_Offset.h_values;
        Graph.column_indices = Column_Index.h_values;
        //util::cpu_mt::PrintCPUArray("row_offsets ",Graph.row_offsets,Graph.nodes+1);
        //util::cpu_mt::PrintCPUArray("column_index",Graph.column_indices,Graph.edges);
        //Graph.DisplayGraph("org");
        printf("%d \t%d \t%d \t%d \t",Num_Nodes.value,Avg_Degree.value,Row_Offset.h_values[Num_Nodes.value],Source_Id.value);fflush(stdout);
        util::CpuTimer cpu_timer;

        cpu_timer.Start();
        Dummy_BFS <_SizeT,_VertexId,_Value>
            (Graph,Source_Id.value,num_gpus,gpu_idx);
        cpu_timer.Stop();
        printf("%f\n",cpu_timer.ElapsedMillis());fflush(stdout);
        Graph.row_offsets    = NULL;
        Graph.column_indices = NULL;      
    }
}
