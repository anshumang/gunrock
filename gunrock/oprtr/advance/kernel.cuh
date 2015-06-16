#pragma once
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/util/test_utils.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/enactor_base.cuh>

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

#include <sys/time.h>

#include "EvqueueManager.h"

extern EvqueueManager *evqm;

namespace gunrock {
namespace oprtr {

struct timeval start, end;
unsigned int h_yield_point;
int h_elapsed;
unsigned int *d_yield_point_ret;
int *d_elapsed_ret;

namespace advance {
/**
 * @brief Advance operator kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for advance operator.
 * @tparam ProblemData Problem data type for advance operator.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam Op Operation for gather reduce. mgpu::plus<int> by default.
 *
 * @param[in] d_done                    Pointer of volatile int to the flag to set when we detect incoming frontier is empty
 * @param[in] enactor_stats             EnactorStats object to store enactor related variables and stast
 * @param[in] frontier_attribute        FrontierAttribute object to store frontier attribute while doing the advance operation
 * @param[in] data_slice                Device pointer to the problem object's data_slice member
 * @param[in] backward_index_queue      If backward mode is activated, this is used to store the vertex index. (deprecated)
 * @param[in] backward_frontier_map_in  If backward mode is activated, this is used to store input frontier bitmap
 * @param[in] backward_frontier_map_out If backward mode is activated, this is used to store output frontier bitmap
 * @param[in] partitioned_scanned_edges If load balanced mode is activated, this is used to store the scanned edge number for neighbor lists in current frontier
 * @param[in] d_in_key_queue            Device pointer of input key array to the incoming frontier queue
 * @param[in] d_out_key_queue           Device pointer of output key array to the outgoing frontier queue
 * @param[in] d_in_value_queue          Device pointer of input value array to the incoming frontier queue
 * @param[in] d_out_value_queue         Device pointer of output value array to the outgoing frontier queue
 * @param[in] d_row_offsets             Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices          Device pointer of VertexId to the column indices queue
 * @param[in] d_column_offsets          Device pointer of SizeT to the row offsets queue for inverse graph
 * @param[in] d_row_indices             Device pointer of VertexId to the column indices queue for inverse graph
 * @param[in] max_in_queue              Maximum number of elements we can place into the incoming frontier
 * @param[in] max_out_queue             Maximum number of elements we can place into the outgoing frontier
 * @param[in] work_progress             queueing counters to record work progress
 * @param[in] context                   CudaContext pointer for moderngpu APIs
 * @param[in] ADVANCE_TYPE              enumerator of advance type: V2V, V2E, E2V, or E2E
 * @param[in] inverse_graph             whether this iteration of advance operation is in the opposite direction to the previous iteration (false by default)
 * @param[in] REDUCE_OP                 enumerator of available reduce operations: plus, multiplies, bit_or, bit_and, bit_xor, maximum, minimum. none by default.
 * @param[in] REDUCE_TYPE               enumerator of available reduce types: EMPTY(do not do reduce) VERTEX(extract value from |V| array) EDGE(extract value from |E| array)
 * @param[in] d_value_to_reduce         array to store values to reduce
 * @param[out] d_reduce_frontier        neighbor list values for nodes in the output frontier
 * @param[out] d_reduced_value          array to store reduced values
 */

//TODO: Reduce by neighbor list now only supports LB advance mode.
//TODO: Add a switch to enable advance+filter (like in BFS), pissibly moving idempotent ops from filter to advance?

template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(
            volatile int                            *d_done,
            gunrock::app::EnactorStats              &enactor_stats,
            gunrock::app::FrontierAttribute         &frontier_attribute,
            typename ProblemData::DataSlice         *data_slice,
            typename ProblemData::VertexId          *backward_index_queue,
            bool                                    *backward_frontier_map_in,
            bool                                    *backward_frontier_map_out,
            unsigned int                            *partitioned_scanned_edges,
            typename KernelPolicy::VertexId         *d_in_key_queue,
            typename KernelPolicy::VertexId         *d_out_key_queue,
            typename KernelPolicy::VertexId         *d_in_value_queue,
            typename KernelPolicy::VertexId         *d_out_value_queue,
            typename KernelPolicy::SizeT            *d_row_offsets,
            typename KernelPolicy::VertexId         *d_column_indices,
            typename KernelPolicy::SizeT            *d_column_offsets,
            typename KernelPolicy::VertexId         *d_row_indices,
            typename KernelPolicy::SizeT            max_in,
            typename KernelPolicy::SizeT            max_out,
            util::CtaWorkProgress                   work_progress,
            CudaContext                             &context,
            TYPE                                    ADVANCE_TYPE,
            bool                                    inverse_graph = false,
            REDUCE_OP                               R_OP = gunrock::oprtr::advance::NONE,
            REDUCE_TYPE                             R_TYPE = gunrock::oprtr::advance::EMPTY,
            typename KernelPolicy::Value            *d_value_to_reduce = NULL,
            typename KernelPolicy::Value            *d_reduce_frontier = NULL,
            typename KernelPolicy::Value            *d_reduced_value = NULL)

{
    switch (KernelPolicy::ADVANCE_MODE)
    {
        case TWC_FORWARD:
        {
            // Load Thread Warp CTA Forward Kernel
            gunrock::oprtr::edge_map_forward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_FORWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    d_done,
                    d_in_key_queue,              // d_in_queue
                    d_out_value_queue,          // d_pred_out_queue
                    d_out_key_queue,            // d_out_queue
                    d_column_indices,
                    d_row_indices,
                    data_slice,
                    work_progress,
                    max_in,                   // max_in_queue
                    max_out,                 // max_out_queue
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    inverse_graph);
            break;
        }
        case LB_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // Load Load Balanced Kernel
            // Get Rowoffsets
            // Use scan to compute edge_offsets for each vertex in the frontier
            // Use sorted sort to compute partition bound for each work-chunk
            // load edge-expand-partitioned kernel
            //util::DisplayDeviceResults(d_in_key_queue, frontier_attribute.queue_length);
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned_backward::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_column_offsets,
                                        d_row_indices,
                                        d_in_key_queue,
                                        partitioned_scanned_edges,
                                        frontier_attribute.queue_length+1,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeExc>((int*)partitioned_scanned_edges, frontier_attribute.queue_length+1, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)partitioned_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];
            //printf("input queue:%d, output_queue:%d\n", frontier_attribute.queue_length, output_queue_len);

            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_in,
                        backward_frontier_map_out,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_out,
                        backward_frontier_map_in,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            }
            break;
        }
        case TWC_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_in,
                            backward_frontier_map_out,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_out,
                            backward_frontier_map_in,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            }
            break;
        }
        case LB:
        {
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename ProblemData::Value         Value;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_row_offsets,
                                        d_column_indices,
                                        d_in_key_queue,
                                        partitioned_scanned_edges,
                                        frontier_attribute.queue_length+1,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeExc>((int*)partitioned_scanned_edges, frontier_attribute.queue_length+1, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)partitioned_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];

            if (output_queue_len < LBPOLICY::LIGHT_EDGE_THRESHOLD)
            {
                gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_row_offsets,
                        d_column_indices,
                        d_row_indices,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        d_out_key_queue,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph,
                        R_TYPE,
                        R_OP,
                        d_value_to_reduce,
                        d_reduce_frontier);
            }
            else
            {
                unsigned int split_val = (output_queue_len + KernelPolicy::LOAD_BALANCED::BLOCKS - 1) / KernelPolicy::LOAD_BALANCED::BLOCKS;
                int num_block = KernelPolicy::LOAD_BALANCED::BLOCKS;
                int nb = (num_block + 1 + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
                gunrock::oprtr::edge_map_partitioned::MarkPartitionSizes<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                    <<<nb, KernelPolicy::LOAD_BALANCED::THREADS>>>(
                            enactor_stats.d_node_locks,
                            split_val,
                            num_block+1,
                            output_queue_len);
                //util::MemsetIdxKernel<<<128, 128>>>(enactor_stats.d_node_locks, KernelPolicy::LOAD_BALANCED::BLOCKS, split_val);

                SortedSearch<MgpuBoundsLower>(
                        enactor_stats.d_node_locks,
                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                        &partitioned_scanned_edges[1],
                        frontier_attribute.queue_length,
                        enactor_stats.d_node_locks_out,
                        context);
                cudaDeviceSynchronize();
                gettimeofday(&start, NULL);
                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        frontier_attribute.queue_reset,
                                        frontier_attribute.queue_index,
                                        enactor_stats.iteration,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_row_indices,
                                        &partitioned_scanned_edges[1],
                                        enactor_stats.d_node_locks_out,
                                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                                        d_done,
                                        d_in_key_queue,
                                        d_out_key_queue,
                                        data_slice,
                                        frontier_attribute.queue_length,
                                        output_queue_len,
                                        split_val,
                                        max_in,
                                        max_out,
                                        work_progress,
                                        enactor_stats.advance_kernel_stats,
                                        ADVANCE_TYPE,
                                        inverse_graph,
                                        R_TYPE,
                                        R_OP,
                                        d_value_to_reduce,
                                        d_reduce_frontier);
		cudaDeviceSynchronize();
		gettimeofday(&end, NULL);
		std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
                unsigned long grid[3], block[3];
                grid[0] = num_block; grid[1] = 1; grid[2] = 1;
                block[0] = KernelPolicy::LOAD_BALANCED::THREADS; block[1] = 1; block[2] = 1;
                KernelIdentifier kid("RelaxPartitionedEdges2", grid, block);
                //EvqueueLaunch(kid);
		unsigned int launch_ctr = 0;
		while(h_yield_point < grid[0]*grid[1]-1)
		{
                gettimeofday(&start, NULL);
                cudaDeviceSynchronize();
                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2instrumented<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        frontier_attribute.queue_reset,
                                        frontier_attribute.queue_index,
                                        enactor_stats.iteration,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_row_indices,
                                        &partitioned_scanned_edges[1],
                                        enactor_stats.d_node_locks_out,
                                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                                        d_done,
                                        d_in_key_queue,
                                        d_out_key_queue,
                                        data_slice,
                                        frontier_attribute.queue_length,
                                        output_queue_len,
                                        split_val,
                                        max_in,
                                        max_out,
                                        work_progress,
                                        enactor_stats.advance_kernel_stats,
                                        ADVANCE_TYPE,
                                        inverse_graph,
                                        R_TYPE,
                                        R_OP,
                                        d_value_to_reduce,
                                        d_reduce_frontier);
                cudaDeviceSynchronize();
		gettimeofday(&end, NULL);
		launch_ctr++;
		std::cout << (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec) << std::endl;
		cudaMemcpy(&h_yield_point, d_yield_point_ret, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_elapsed, d_elapsed_ret, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << launch_ctr << " => " << h_yield_point << " " << h_elapsed << std::endl;
                }

                //util::DisplayDeviceResults(d_out_key_queue, output_queue_len);
            }

            // TODO: switch REDUCE_OP for different reduce operators
            // Do segreduction using d_scanned_edges and d_reduce_frontier
            if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce && d_reduce_frontier) {
              switch (R_OP) {
                case gunrock::oprtr::advance::PLUS: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MULTIPLIES: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MAXIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MINIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context);
                      break;
                }
                default:
                    //default operator is plus
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
              }
            }
            break;
        }
    }
}


} //advance
} //oprtr
} //gunrock/
