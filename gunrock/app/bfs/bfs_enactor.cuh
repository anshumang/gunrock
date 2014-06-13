// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/scan/multi_scan.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>


namespace gunrock {
namespace app {
namespace bfs {

    template <typename BFSProblem, bool INSTRUMENT> class BFSEnactor;

    class ThreadSlice
    {
    public:
        int           thread_num;
        int           init_size;
        int           max_grid_size;
        int           edge_map_grid_size;
        int           vertex_map_grid_size;
        CUTThread     thread_Id;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        void*         problem;
        void*         enactor;

        ThreadSlice()
        {
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }

        virtual ~ThreadSlice()
        {
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }
    };
 
    template <typename VertexId, typename SizeT, bool MARK_PREDECESSORS>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        const SizeT            num_associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              VertexId**       associate_in,
              VertexId**       associate_out)
    {  
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT x2=incoming_offset+x;
        VertexId key=keys_in[x2];
        /*__shared__ VertexId* a_out_0;
        a_out_0=associate_out[0];

        keys_out[x]=key;
        //if (num_associates <1) return;
        VertexId t=associate_in[0][incoming_offset+x];
        if (t >= a_out_0[key] && a_out_0[key]>=0) 
        // TODO:: change to atomic version
        {
            keys_out[x]=-1;
            return;
        }
        a_out_0[key]=t;
        for (SizeT i=1;i<num_associates;i++)
        {   
            associate_out[i][key]=associate_in[i][incoming_offset+x];   
        }*/

        /*if (MARK_PREDECESSORS)
        {
           if (atomicCAS(associate_out[1]+key, -2, associate_in[1][x2])== -2) 
           {
               for (SizeT i=2;i<num_associates;i++)
                   associate_out[i][key]=associate_in[i][x2];
               associate_out[0][key]=associate_in[0][x2];
           } else {
               keys_out[x]=-1;
               return;
           }
        } else */{
           VertexId t=associate_in[0][x2];
           if (atomicCAS(associate_out[0]+key, -1, t)== -1)
           {
           } else {
               if (atomicMin(associate_out[0]+key, t)<t)
               {
                   keys_out[x]=-1;
                   return;
               }
           }
           for (SizeT i=1;i<num_associates;i++)
               associate_out[i][key]=associate_in[i][x2];
        }
        keys_out[x]=key;
    }   

    bool All_Done(volatile int **dones, cudaError_t *retvals,int num_gpus)
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
        if (retvals[gpu]!=cudaSuccess) 
        {
            printf("(CUDA error %d @ GPU %d: %s\n", retvals[gpu], gpu, cudaGetErrorString(retvals[gpu])); fflush(stdout); 
            return true;
        }

        for (int gpu=0;gpu<num_gpus;gpu++)
        if (dones[gpu][0]!=0) 
        {
            return false;
        }
        return true;
    }

   template<
        bool     INSTRUMENT,
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
   static CUT_THREADPROC BFSThread(
        void * thread_data_)
    {
        ThreadSlice
            * thread_data = (ThreadSlice *) thread_data_;
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;
        typedef typename BFSProblem::DataSlice  DataSlice;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            BFSProblem> BfsFunctor;
        
        SizeT*       out_offset;
        char*         message              = new char [1024];
        BFSProblem*  problem               =   (BFSProblem*) thread_data->problem;
        BFSEnactor<BFSProblem, INSTRUMENT>*  
                     enactor               =   (BFSEnactor<BFSProblem, INSTRUMENT>*)thread_data->enactor;
        int          thread_num            =   thread_data->thread_num;
        util::cpu_mt::CPUBarrier* cpu_barrier =   thread_data->cpu_barrier;
        int          gpu                   =   problem    ->gpu_idx        [thread_num];
        int          num_gpus              =   problem    ->num_gpus;
        int          edge_map_grid_size    =   thread_data->edge_map_grid_size;
        int          vertex_map_grid_size  =   thread_data->vertex_map_grid_size;
        cudaError_t* retval                = &(enactor    ->retvals        [thread_num]);
        volatile int* d_done               =  (enactor    ->d_dones        [thread_num]);
        volatile int** dones               =   enactor    ->dones;
        int*         iteration             = &(enactor    ->iterations     [thread_num]);
        cudaEvent_t* throttle_event        = &(enactor    ->throttle_events[thread_num]);
        cudaError_t* retvals               =   enactor    ->retvals.GetPointer();
        bool         DEBUG                 =   enactor    ->DEBUG;
        unsigned long long* total_queued   = &(enactor    ->total_queued   [thread_num]);
        unsigned long long* total_runtimes = &(enactor    ->total_runtimes [thread_num]);
        unsigned long long* total_lifetimes= &(enactor    ->total_lifetimes[thread_num]);
        typename BFSProblem::GraphSlice*
                     graph_slice           =   problem    ->graph_slices   [thread_num];
        //typename BFSProblem::DataSlice*
        //             data_slice            =   problem    ->data_slices    [thread_num];
        util::Array1D<SizeT,DataSlice>* 
                     data_slice            = &(problem    ->data_slices    [thread_num]);
        //typename BFSProblem::DataSlice*
        //             d_data_slice          =   problem    ->d_data_slices  [thread_num];
        util::CtaWorkProgressLifetime*
                     work_progress         = &(enactor    ->work_progress  [thread_num]);
        util::KernelRuntimeStatsLifetime*
                       edge_map_kernel_stats = &(enactor->  edge_map_kernel_stats[thread_num]);
        util::KernelRuntimeStatsLifetime*
                     vertex_map_kernel_stats = &(enactor->vertex_map_kernel_stats[thread_num]);
        util::scan::MultiScan<VertexId,SizeT,true,256,8> *Scaner = NULL;
        bool break_clean=true;

        if (num_gpus >1) 
        {
            Scaner=new util::scan::MultiScan<VertexId,SizeT,true,256,8>;
            out_offset=new SizeT[num_gpus+1];
        }
        do {
            if ( retval[0] = util::GRError(cudaSetDevice(gpu), "BFSThread cudaSetDevice failed." , __FILE__, __LINE__)) break; 
            VertexId queue_index    = 0;        // Work queue index
            int      selector       = 0;
            SizeT    num_elements   = thread_data->init_size; //?
            bool     queue_reset    = true; 
            // Step through BFS iterations
                        
            //VertexId *h_cur_queue = new VertexId[graph_slice->edges];
            while (!All_Done(dones,retvals,num_gpus)) {
                if (retval[0] = work_progress->SetQueueLength(queue_index+1, 0)) break;
                if (DEBUG) {
                    SizeT _queue_length;
                    if (queue_reset) _queue_length = num_elements;
                    else if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    //printf("%d\t%d\tQueue_Length = %d\n",thread_num,iteration[0],_queue_length);fflush(stdout);
                    //if (retval[0] = work_progress->GetQueueLength(queue_index+1, _queue_length)) break;
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys0",graph_slice->frontier_queues.d_keys[selector],_queue_length,thread_num,iteration[0]);
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe0",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                }

                // Edge Map
                //printf("%d\t%d\tEdge map begin.\n",thread_num,iteration[0]);fflush(stdout);
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, BFSProblem, BfsFunctor>
                    <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration[0],
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[selector^1],          // d_pred_out_queue
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    //graph_slice->d_column_indices,
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    data_slice->GetPointer(util::DEVICE),//d_data_slice,
                    work_progress[0],
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    edge_map_kernel_stats[0]);
               
                if (DEBUG && (retval[0] = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event[0]);                                 // give host memory mapped visibility to GPU updates 
                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;
               
                queue_index++;
                selector ^= 1;
                if (retval[0] = work_progress->SetQueueLength(queue_index+1, 0)) break;
                if (DEBUG) {
                    SizeT _queue_length;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    //printf("%d\t%d\tQueue_Length = %d\n",thread_num,iteration[0],_queue_length);fflush(stdout);
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys1",graph_slice->frontier_queues.d_keys[selector],_queue_length,thread_num,iteration[0]);
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe1",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                }

                if (INSTRUMENT) {
                    if (retval[0] = edge_map_kernel_stats->Accumulate(
                        edge_map_grid_size,
                        total_runtimes[0],
                        total_lifetimes[0])) break;
                }

                // Throttle
                if (iteration[0] & 1) {
                    if (retval[0] = util::GRError(cudaEventRecord(throttle_event[0]),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval[0] = util::GRError(cudaEventSynchronize(throttle_event[0]),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (All_Done(dones,retvals,num_gpus)) break;

                // Vertex Map
                //printf("%d\t%d\tVertex map begin.\n",thread_num,iteration[0]);fflush(stdout);
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, BFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    iteration[0]+1,
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    //d_data_slice,
                    data_slice->GetPointer(util::DEVICE),
                    //data_slice->d_visited_mask,
                    data_slice[0]->visited_mask.GetPointer(util::DEVICE),
                    work_progress[0],
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    vertex_map_kernel_stats[0]);

                if (DEBUG && (retval[0] = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event[0]); // give host memory mapped visibility to GPU updates

                queue_index++;
                selector ^= 1;

                if (INSTRUMENT || DEBUG) {
                    SizeT _queue_length;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, _queue_length)) break;
                    //printf("%d\t%d\tQueue_Length = %d\n",thread_num,iteration[0],_queue_length);fflush(stdout);
                    total_queued[0] += _queue_length;
                    if (DEBUG) 
                    {
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys2",graph_slice->frontier_queues.d_keys[selector],_queue_length,thread_num,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe2",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                    }
                    if (INSTRUMENT) {
                        if (retval[0] = vertex_map_kernel_stats->Accumulate(
                            vertex_map_grid_size,
                            total_runtimes[0],
                            total_lifetimes[0])) break;
                    }
                }
                // Check if done
                if (All_Done(dones,retvals,num_gpus)) break;

                //Use multi_scan to splict the workload into multi_gpus
                if (num_gpus >1)
                {
                    // Split the frontier into multiple frontiers for multiple GPUs, local remains infront
                    SizeT n;
                    if (retval[0] = work_progress->GetQueueLength(queue_index, n)) break;
                    if (n >0)
                    {
                        dones[thread_num][0]=-1;
                        //printf("%d\t%d\tScan map begin.\n",thread_num,iteration[0]);fflush(stdout);
                        //int* _partition_table = graph_slice->partition_table.GetPointer(util::DEVICE);
                        //SizeT* _convertion_table = graph_slice->convertion_table.GetPointer(util::DEVICE);
                        Scaner->Scan_with_Keys(n,
                                  num_gpus,
                                  data_slice[0]->num_associate,
                                  graph_slice  ->frontier_queues  .d_keys[selector],
                                  graph_slice  ->frontier_queues  .d_keys[selector^1],
                                  //graph_slice->d_partition_table,
                                  graph_slice  ->partition_table  .GetPointer(util::DEVICE),
                                  //graph_slice->d_convertion_table,
                                  graph_slice  ->convertion_table .GetPointer(util::DEVICE),
                                  //data_slice ->d_out_length,
                                  data_slice[0]->out_length       .GetPointer(util::DEVICE),
                                  //data_slice ->d_associate_org,
                                  data_slice[0]->associate_orgs   .GetPointer(util::DEVICE),
                                  //data_slice ->d_associate_out);
                                  data_slice[0]->associate_outs   .GetPointer(util::DEVICE));
                        /*if (retval[0] = util::GRError(cudaMemcpy(
                                  data_slice->out_length,
                                  data_slice->d_out_length,
                                  sizeof(SizeT)*num_gpus,
                                  cudaMemcpyDeviceToHost),
                                  "BFSEnactor cudaMemcpy h_Length failed", __FILE__, __LINE__)) break;*/
                        if (retval[0] = data_slice[0]->out_length.Move(util::DEVICE,util::HOST)) break;
                        out_offset[0]=0;
                        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice[0]->out_length[i];
 
                        queue_index++;
                        selector ^= 1;
                       
                        if (iteration[0]!=0)
                        {  //CPU global barrier
                            util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),thread_num);
                            if (All_Done(dones,retvals,num_gpus)) {break_clean=false;break;}
                        }

                        //util::cpu_mt::PrintCPUArray<SizeT,SizeT>("out_length",data_slice[0]->out_length.GetPointer(util::HOST),num_gpus,thread_num,iteration[0]);
                        //util::cpu_mt::PrintCPUArray<SizeT,SizeT>("out_offset",out_offset,num_gpus+1,thread_num,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,SizeT>("keys3",graph_slice->frontier_queues.d_keys[selector],out_offset[num_gpus],thread_num,iteration[0]);
                        //Move data
                        //printf("%d\t%d\tMove begin.\n",thread_num,iteration[0]);fflush(stdout);
                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int peer_ = peer<thread_num? peer+1     : peer;
                            int gpu_  = peer<thread_num? thread_num : thread_num+1;
                            if (peer==thread_num) continue;
                            problem->data_slices[peer]->in_length[gpu_]=data_slice[0]->out_length[peer_];
                            if (data_slice[0]->out_length[peer_] == 0) continue;
                            dones[peer][0]=-1;
                            if (retval [0] = util::GRError(cudaMemcpy(
                                  problem->data_slices[peer]->keys_in.GetPointer(util::DEVICE) + problem->graph_slices[peer]->in_offset[gpu_],
                                  graph_slice->frontier_queues.d_keys[selector] + out_offset[peer_],
                                  sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                  "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;

                            for (int i=0;i<data_slice[0]->num_associate;i++)
                            {
                                if (retval [0] = util::GRError(cudaMemcpy(
                                    //problem->data_slices[peer]->h_associate_in[i] + problem->graph_slices[peer]->in_offset[gpu_],
                                    problem->data_slices[peer]->associate_ins[i] + problem->graph_slices[peer]->in_offset[gpu_],
                                    //data_slice->h_associate_out[i] + (out_offset[peer_] - out_offset[1]),
                                    data_slice[0]->associate_outs[i] + (out_offset[peer_] - out_offset[1]), 
                                    sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                    "cudaMemcpyPeer associate_out failed", __FILE__, __LINE__)) break;
                            }
                            if (retval [0]) break;
                        }
                        if (retval [0]) break;
                    }  else {
                        if (iteration[0]!=0)
                        {  //CPU global barrier
                            util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),thread_num);
                            if (All_Done(dones,retvals,num_gpus)) {break_clean=false;break;}
                        }

                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int gpu_ = peer<thread_num? thread_num: thread_num+1;
                            if (peer == thread_num) continue;
                            problem->data_slices[peer]->in_length[gpu_]=0;
                        }
                        data_slice[0]->out_length[0]=0;
                    }

                    //CPU global barrier
                    util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[1]),thread_num);
                    if (All_Done(dones,retvals,num_gpus)) {break_clean=false;break;}
                    SizeT total_length=data_slice[0]->out_length[0];
                    //printf("%d\t%d\tExpand begin.\n",thread_num,iteration[0]);fflush(stdout);
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (peer==thread_num) continue;
                        int peer_ = peer<thread_num ? peer+1: peer ;
                        if (data_slice[0]->in_length[peer_] ==0) continue;
                        int grid_size = data_slice[0]->in_length[peer_] / 256;
                        if ((data_slice[0]->in_length[peer_] % 256)!=0) grid_size++;
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys_in",data_slice[0]->keys_in.GetPointer(util::DEVICE)+graph_slice->in_offset[peer_],data_slice[0]->in_length[peer_],thread_num,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_in",data_slice[0]->associate_ins[0]+graph_slice->in_offset[peer_],data_slice[0]->in_length[peer_],thread_num,iteration[0]);
                        Expand_Incoming <VertexId, SizeT, BFSProblem::MARK_PREDECESSORS>
                            <<<grid_size,256>>> (
                            data_slice[0]  ->in_length[peer_],
                            data_slice[0]  ->num_associate,
                            graph_slice    ->in_offset[peer_],
                            //data_slice  ->d_keys_in,
                            data_slice[0]  ->keys_in.GetPointer(util::DEVICE),
                            graph_slice    ->frontier_queues.d_keys[selector]+total_length,
                            //data_slice  ->d_associate_in,
                            data_slice[0]  ->associate_ins.GetPointer(util::DEVICE),
                            //data_slice  ->d_associate_org);
                            data_slice[0]  ->associate_orgs.GetPointer(util::DEVICE));
                        if (retval[0] = util::GRError("Expand_Incoming failed", __FILE__, __LINE__)) break;
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_orgs",data_slice[0]->associate_orgs[0],graph_slice->nodes,thread_num,iteration[0]);
                        total_length+=data_slice[0]->in_length[peer_];
                    }
                    if (retval[0]) break;
                    if (retval[0] = work_progress->SetQueueLength(queue_index,total_length)) break;
                    //printf("%d\t%d\ttotal_length = %d\n",thread_num,iteration[0],total_length);fflush(stdout);
                    if (total_length !=0) 
                    {
                        dones[thread_num][0]=-1;
                    } else {
                        dones[thread_num][0]=0;
                    }
                }
                //printf("%d\t%d\tIteration end.\n", thread_num, iteration[0]);fflush(stdout);
                iteration[0]++;
            }

            //printf("%d\t%d\tLoop quited.\n", thread_num, iteration[0]);fflush(stdout);
            //delete[] h_cur_queue;h_cur_queue=NULL;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval[0] = work_progress->CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval[0] = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
        } while(0);

        if (num_gpus >1)
        {
            if (break_clean) util::cpu_mt::ReleaseBarrier(&(cpu_barrier[1]));
            //util::cpu_mt::ReleaseBarrier(&(cpu_barrier[0]));
            //util::cpu_mt::ReleaseBarrier(&(cpu_barrier[1]));
            delete Scaner; Scaner=NULL;
            delete[] out_offset; out_offset=NULL;
        }
        delete[] message;message=NULL;
        CUT_THREADEND; 
    }

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename BFSProblem, bool INSTRUMENT>
class BFSEnactor : public EnactorBase
{
    typedef typename BFSProblem::SizeT      SizeT;
    typedef typename BFSProblem::VertexId   VertexId;
    // Members
public:

    /**
     * CTA duty kernel stats
     */
    //util::KernelRuntimeStatsLifetime *edge_map_kernel_stats;
    util::Array1D<SizeT, util::KernelRuntimeStatsLifetime> edge_map_kernel_stats;// ("edge_map_kernel_stats");
    //util::KernelRuntimeStatsLifetime *vertex_map_kernel_stats;
    util::Array1D<SizeT, util::KernelRuntimeStatsLifetime> vertex_map_kernel_stats;//("vertex_map_kernel_stats");
    //util::CtaWorkProgressLifetime    *work_progress;
    util::Array1D<SizeT, util::CtaWorkProgressLifetime   > work_progress;//("work_progress");

    //unsigned long long *total_runtimes;              // Total working time by each CTA
    util::Array1D<SizeT, unsigned long long> total_runtimes ;//("total_runtimes" );
    //unsigned long long *total_lifetimes;             // Total life time of each CTA
    util::Array1D<SizeT, unsigned long long> total_lifetimes;//("total_lifetimes");
    //unsigned long long *total_queued;
    util::Array1D<SizeT, unsigned long long> total_queued   ;//("total_queued"   );

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        **dones;
    int                 **d_dones;
    //cudaEvent_t         *throttle_events;
    util::Array1D<SizeT, cudaEvent_t> throttle_events;//("throttle_events");
    //cudaError_t         *retvals;
    util::Array1D<SizeT, cudaError_t> retvals        ;//("retvals");
    int                 num_gpus;
    //int                 *gpu_idx;
    util::Array1D<SizeT, int>         gpu_idx        ;//("gpu_idx");

    /**
     * Current iteration, also used to get the final search depth of the BFS search
     */
    //int                 *iterations;
    util::Array1D<SizeT, int>         iterations     ;//("iterations");
    
   // Methods
public:
    /**
     * @brief Prepare the enactor for BFS kernel call. Must be called prior to each BFS search.
     *
     * @param[in] problem BFS Problem object which holds the graph data and BFS problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename ProblemData>
    cudaError_t Setup(
        BFSProblem *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        //typedef typename ProblemData::SizeT         SizeT;
        //typedef typename ProblemData::VertexId      VertexId;
        cudaError_t retval = cudaSuccess;
        this->num_gpus = problem->num_gpus;
        //this->gpu_idx  = problem->gpu_idx;
        this->gpu_idx.SetPointer(problem->gpu_idx,num_gpus);
        do {
            dones           = new volatile int*      [num_gpus];
            d_dones         = new          int*      [num_gpus];
            //throttle_events = new cudaEvent_t        [num_gpus];
            throttle_events.Init(num_gpus); 
            //retvals         = new cudaError_t        [num_gpus];
            retvals        .Init(num_gpus); 
            //total_runtimes  = new unsigned long long [num_gpus];
            total_runtimes .Init(num_gpus); 
            //total_lifetimes = new unsigned long long [num_gpus];
            total_lifetimes.Init(num_gpus); 
            //total_queued    = new unsigned long long [num_gpus];
            total_queued   .Init(num_gpus); 
            //iterations      = new          int       [num_gpus];
            iterations     .Init(num_gpus); 
            //edge_map_kernel_stats   = new util::KernelRuntimeStatsLifetime[num_gpus];
            edge_map_kernel_stats  .Init(num_gpus); 
            //vertex_map_kernel_stats = new util::KernelRuntimeStatsLifetime[num_gpus];
            vertex_map_kernel_stats.Init(num_gpus); 
            //work_progress           = new util::CtaWorkProgressLifetime   [num_gpus];
            work_progress          .Init(num_gpus); 

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]), "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__)) break;
                int flags = cudaHostAllocMapped;
                work_progress[gpu].Setup();

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&(dones[gpu]), sizeof(int) * 1, flags),
                    "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&(d_dones[gpu]), (void*) dones[gpu], 0),
                    "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;
                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&(throttle_events[gpu]), cudaEventDisableTiming),
                    "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;

                //initialize runtime stats
                if (retval =   edge_map_kernel_stats[gpu].Setup(  edge_map_grid_size)) break;
                if (retval = vertex_map_kernel_stats[gpu].Setup(vertex_map_grid_size)) break;

                //Reset statistics
                iterations      [gpu]    = 0;
                total_runtimes  [gpu]    = 0;
                total_lifetimes [gpu]    = 0;
                total_queued    [gpu]    = 0;
                dones           [gpu][0] = -1;
                retvals         [gpu]    = cudaSuccess;

                //graph slice
                //typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[gpu];

                // Bind row-offsets and column_indices texture
                cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    problem->graph_slices[gpu]->row_offsets.GetPointer(util::GPU),
                    row_offsets_desc,
                    (problem->graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            } // for gpu
            if (retval) break;
        } while (0);
        return retval;
    }

    public:

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG)
    {
        //edge_map_kernel_stats   = NULL;
        //vertex_map_kernel_stats = NULL;
        //work_progress           = NULL;
        //total_runtimes          = NULL;
        //total_lifetimes         = NULL;
        //total_queued            = NULL;
        dones                   = NULL;
        d_dones                 = NULL;
        //throttle_events         = NULL;
        //retvals                 = NULL;
        //iterations              = NULL;
       // gpu_idx                 = NULL;
        num_gpus                = 0;
        edge_map_kernel_stats   .SetName("edge_map_kernel_stats"  );
        vertex_map_kernel_stats .SetName("vertex_map_kernel_stats");
        work_progress           .SetName("work_progress"          );
        total_runtimes          .SetName("total_runtimes"         );
        total_lifetimes         .SetName("total_lifetimes"        );
        total_queued            .SetName("total_queued"           );
        throttle_events         .SetName("throttle_events"        );
        retvals                 .SetName("retvals"                );
        iterations              .SetName("iterations"             );
        gpu_idx                 .SetName("gpu_idx"                );
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        if (All_Done(dones,retvals.GetPointer(),num_gpus)) {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                if (num_gpus !=1)
                    util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);

                util::GRError(cudaFreeHost((void*)(dones[gpu])),
                    "BFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

                util::GRError(cudaEventDestroy(throttle_events[gpu]),
                    "BFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
            }
            delete[] dones;          dones           = NULL; 
            //delete[] throttle_events;throttle_events = NULL; 
            //delete[] retvals;        retvals         = NULL; 
            //delete[] iterations;     iterations      = NULL; 
            //delete[] total_runtimes; total_runtimes  = NULL; 
            //delete[] total_lifetimes;total_lifetimes = NULL; 
            //delete[] total_queued;   total_queued    = NULL; 
            //delete[] work_progress;  work_progress   = NULL; 
            //delete[]   edge_map_kernel_stats;  edge_map_kernel_stats = NULL; 
            //delete[] vertex_map_kernel_stats;vertex_map_kernel_stats = NULL; 
            //gpu_idx = NULL;
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    //template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (num_gpus!=1) 
                util::GRError(cudaSetDevice(gpu_idx[gpu]),
                    "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);
            cudaThreadSynchronize();

            total_queued += this->total_queued[gpu];
            if (this->iterations[gpu] > search_depth) search_depth = this->iterations[gpu];
            total_lifetimes += this->total_lifetimes[gpu];
            total_runtimes += this->total_runtimes[gpu];
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */
    
   /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy>//,
        //typename BFSProblem>
    cudaError_t EnactBFS(
    BFSProblem                          *problem,
    //typename BFSProblem::VertexId       src,
    VertexId                            src,
    int                                 max_grid_size = 0)
    {
        util::cpu_mt::CPUBarrier cpu_barrier[2];
        ThreadSlice *thread_slices;
        CUTThread   *thread_Ids;
        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("BFS edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("BFS vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
            }

            // Lazy initialization
            //if (retval = Setup<BFSProblem>(problem, edge_map_grid_size, vertex_map_grid_size)) break;
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;
            thread_slices  = new ThreadSlice [num_gpus];
            thread_Ids     = new CUTThread   [num_gpus];
        
            cpu_barrier[0] = util::cpu_mt::CreateBarrier(num_gpus);
            cpu_barrier[1] = util::cpu_mt::CreateBarrier(num_gpus);

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                thread_slices[gpu].thread_num           = gpu;
                thread_slices[gpu].problem              = (void*)problem;
                thread_slices[gpu].enactor              = (void*)this;
                thread_slices[gpu].cpu_barrier          = cpu_barrier;
                thread_slices[gpu].max_grid_size        = max_grid_size;
                thread_slices[gpu].edge_map_grid_size   = edge_map_grid_size;
                thread_slices[gpu].vertex_map_grid_size = vertex_map_grid_size;
                if ((num_gpus == 1) || (gpu==problem->partition_tables[0][src])) 
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                thread_slices[gpu].thread_Id = cutStartThread((CUT_THREADROUTINE)&(BFSThread<INSTRUMENT,EdgeMapPolicy, VertexMapPolicy, BFSProblem>),(void*)&(thread_slices[gpu]));
                thread_Ids[gpu]=thread_slices[gpu].thread_Id;
            }

            cutWaitForThreads(thread_Ids,num_gpus);
            util::cpu_mt::DestoryBarrier(cpu_barrier);
            util::cpu_mt::DestoryBarrier(cpu_barrier+1);

            for (int gpu=0;gpu<num_gpus;gpu++)
            if (this->retvals[gpu]!=cudaSuccess) {retval=this->retvals[gpu];break;}
        } while (0);
        if (retval) return retval;
        if (DEBUG) {printf("\nGPU BFS Done.\n");fflush(stdout);}
        for (int gpu=0; gpu<num_gpus;gpu++)
        {
            thread_slices[gpu].problem = NULL;
            thread_slices[gpu].enactor = NULL;
            thread_slices[gpu].cpu_barrier = NULL;
        }
        delete[] thread_Ids;   thread_Ids    = NULL;
        delete[] thread_slices;thread_slices = NULL; 
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename BFSProblem>
    cudaError_t Enact(
        BFSProblem                      *problem,
        //typename BFSProblem::VertexId    src,
        VertexId                        src,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
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
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                EdgeMapPolicy;

                //return EnactBFS<EdgeMapPolicy, VertexMapPolicy, BFSProblem>(
                return EnactBFS<EdgeMapPolicy, VertexMapPolicy>(
                problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
