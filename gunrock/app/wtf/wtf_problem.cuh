// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * wtf_problem.cuh
 *
 * @brief GPU Storage management Structure for Who-To-Follow framework
 * (combines Personalized PageRank and SALSA/Personalized-SALSA)
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>

namespace gunrock {
namespace app {
namespace wtf {

/**
 * @brief WTF Problem structure stores device-side vectors for doing PageRank on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing WTF value.
 */
template <
    typename    _VertexId,                       
    typename    _SizeT,                          
    typename    _Value>
struct WTFProblem : ProblemBase<_VertexId, _SizeT, false> // USE_DOUBLE_BUFFER = false
{

    typedef _VertexId 			VertexId;
	typedef _SizeT			    SizeT;
	typedef _Value              Value;

    static const bool MARK_PREDECESSORS     = true;
    static const bool ENABLE_IDEMPOTENCE    = false;

    //Helper structures

    /**
     * @brief Data slice structure which contains WTF problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        Value   *d_rank_curr;           /**< Used for ping-pong page rank value */
        Value   *d_rank_next;           /**< Used for ping-pong page rank value */       
        Value   *d_refscore_curr;
        Value   *d_refscore_next;
        SizeT   *d_out_degrees;             /**< Used for keeping out-degree for each vertex */
        SizeT    *d_in_degrees;
        Value   *d_threshold;               /**< Used for recording accumulated error */
        Value   *d_delta;
        Value   *d_alpha;
        VertexId   *d_src_node;
        SizeT   *d_labels;
        VertexId *d_node_ids;
        bool        *d_cot_map;     /**< Input frontier bitmap */
    };

    // Members
    
    // Number of GPUs to be sliced over
    int                 num_gpus;

    // Size of the graph
    SizeT               nodes;
    SizeT               edges;

    // Selector, which d_rank array stores the final page rank?
    SizeT               selector;

    // Set of data slices (one for each GPU)
    DataSlice           **data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    DataSlice           **d_data_slices;

    // Device indices for each data slice
    int                 *gpu_idx;

    // Methods

    /**
     * @brief WTFProblem default constructor
     */

    WTFProblem():
    nodes(0),
    edges(0),
    num_gpus(0) {}

    /**
     * @brief WTFProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    WTFProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               const Csr<VertexId, Value, SizeT> &graph,
               int         num_gpus) :
        num_gpus(num_gpus)
    {
        Init(
            stream_from_host,
            graph,
            num_gpus);
    }

    /**
     * @brief WTFProblem default destructor
     */
    ~WTFProblem()
    {
        for (int i = 0; i < num_gpus; ++i)
        {
            if (util::GRError(cudaSetDevice(gpu_idx[i]),
                "~WTFProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            if (data_slices[i]->d_rank_curr)      util::GRError(cudaFree(data_slices[i]->d_rank_curr), "GpuSlice cudaFree d_rank[0] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_rank_next)      util::GRError(cudaFree(data_slices[i]->d_rank_next), "GpuSlice cudaFree d_rank[1] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_refscore_curr)      util::GRError(cudaFree(data_slices[i]->d_refscore_curr), "GpuSlice cudaFree d_refscore[0] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_refscore_next)      util::GRError(cudaFree(data_slices[i]->d_refscore_next), "GpuSlice cudaFree d_refscore[1] failed", __FILE__, __LINE__);
            if (data_slices[i]->d_out_degrees)      util::GRError(cudaFree(data_slices[i]->d_out_degrees), "GpuSlice cudaFree d_out_degrees failed", __FILE__, __LINE__);
            if (data_slices[i]->d_in_degrees)      util::GRError(cudaFree(data_slices[i]->d_in_degrees), "GpuSlice cudaFree d_in_degrees failed", __FILE__, __LINE__);
            if (data_slices[i]->d_threshold)      util::GRError(cudaFree(data_slices[i]->d_threshold), "GpuSlice cudaFree d_threshold failed", __FILE__, __LINE__);
            if (data_slices[i]->d_delta)      util::GRError(cudaFree(data_slices[i]->d_delta), "GpuSlice cudaFree d_delta failed", __FILE__, __LINE__);
            if (data_slices[i]->d_alpha)      util::GRError(cudaFree(data_slices[i]->d_alpha), "GpuSlice cudaFree d_alpha failed", __FILE__, __LINE__);
            if (data_slices[i]->d_cot_map)      util::GRError(cudaFree(data_slices[i]->d_cot_map), "GpuSlice cudaFree d_cot_map failed", __FILE__, __LINE__);
            if (data_slices[i]->d_node_ids)      util::GRError(cudaFree(data_slices[i]->d_node_ids), "GpuSlice cudaFree d_node_ids failed", __FILE__, __LINE__);
            if (data_slices[i]->d_src_node)      util::GRError(cudaFree(data_slices[i]->d_src_node), "GpuSlice cudaFree d_src_node failed", __FILE__, __LINE__);
            if (d_data_slices[i])                 util::GRError(cudaFree(d_data_slices[i]), "GpuSlice cudaFree data_slices failed", __FILE__, __LINE__);
        }
        if (d_data_slices)  delete[] d_data_slices;
        if (data_slices) delete[] data_slices;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_rank host-side vector to store page rank values.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(Value *h_rank, VertexId *h_node_id)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(gpu_idx[0]),
                            "WTFProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                if (retval = util::GRError(cudaMemcpy(
                                h_rank,
                                data_slices[0]->d_refscore_curr,
                                sizeof(Value) * nodes,
                                cudaMemcpyDeviceToHost),
                            "WTFProblem cudaMemcpy d_rank_curr failed", __FILE__, __LINE__)) break;
        
                if (retval = util::GRError(cudaMemcpy(
                                h_node_id,
                                data_slices[0]->d_node_ids,
                                sizeof(VertexId) * nodes,
                                cudaMemcpyDeviceToHost),
                            "WTFProblem cudaMemcpy d_node_id failed", __FILE__, __LINE__)) break;
            } else {
                // TODO: multi-GPU extract result
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief WTFProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool        stream_from_host,       // Only meaningful for single-GPU
            const Csr<VertexId, Value, SizeT> &graph,
            int         _num_gpus)
    {
        num_gpus = _num_gpus;
        nodes = graph.nodes;
        edges = graph.edges;
        VertexId *h_row_offsets = graph.row_offsets;
        VertexId *h_column_indices = graph.column_indices;
            ProblemBase<VertexId, SizeT, false>::Init(stream_from_host,
                    nodes,
                    edges,
                    h_row_offsets,
                    h_column_indices,
                    NULL,
                    NULL,
                    num_gpus);

        // No data in DataSlice needs to be copied from host

        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new DataSlice*[num_gpus];
        d_data_slices = new DataSlice*[num_gpus];

        do {
            if (num_gpus <= 1) {
                gpu_idx = (int*)malloc(sizeof(int));
                // Create a single data slice for the currently-set gpu
                int gpu;
                if (retval = util::GRError(cudaGetDevice(&gpu), "WTFProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
                gpu_idx[0] = gpu;

                data_slices[0] = new DataSlice;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_data_slices[0],
                                sizeof(DataSlice)),
                            "WTFProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;

                // Create SoA on device
                Value    *d_rank1;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank1,
                        nodes * sizeof(Value)),
                    "WTFProblem cudaMalloc d_rank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_rank_curr = d_rank1;

                Value    *d_rank2;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank2,
                        nodes * sizeof(Value)),
                    "WTFProblem cudaMalloc d_rank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_rank_next = d_rank2;

                Value    *d_rank3;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank3,
                        nodes * sizeof(Value)),
                    "WTFProblem cudaMalloc d_rank3 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_refscore_curr = d_rank3;

                Value    *d_rank4;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_rank4,
                        nodes * sizeof(Value)),
                    "WTFProblem cudaMalloc d_rank4 failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_refscore_next = d_rank4;
 
                SizeT   *d_out_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_out_degrees,
                        nodes * sizeof(SizeT)),
                    "WTFProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_out_degrees = d_out_degrees;

                SizeT   *d_in_degrees;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_in_degrees,
                        nodes * sizeof(SizeT)),
                    "WTFProblem cudaMalloc d_in_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_in_degrees = d_in_degrees;

                Value   *d_threshold;
                    if (retval = util::GRError(cudaMalloc(
                        (void**)&d_threshold,
                        1 * sizeof(Value)),
                    "WTFProblem cudaMalloc d_threshold failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_threshold = d_threshold;

                Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_delta,
                        1 * sizeof(Value)),
                    "WTFProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_delta = d_delta;

                Value    *d_alpha;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_alpha,
                        1 * sizeof(Value)),
                    "WTFProblem cudaMalloc d_alpha failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_alpha = d_alpha;

                VertexId    *d_node_ids;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_node_ids,
                        nodes * sizeof(VertexId)),
                    "WTFProblem cudaMalloc d_node_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_node_ids = d_node_ids;

                SizeT    *d_src_node;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_src_node,
                        1 * sizeof(SizeT)),
                    "WTFProblem cudaMalloc d_src_node failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_src_node = d_src_node;

                bool    *d_cot_map;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_cot_map,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_cot_map failed", __FILE__, __LINE__)) return retval;
                data_slices[0]->d_cot_map = d_cot_map;

                data_slices[0]->d_labels = NULL;

            }
            //TODO: add multi-GPU allocation code
        } while (0);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for WTF problem type. Must be called prior to each WTF iteration.
     *
     *  @param[in] src Source node for one WTF computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId src,
            Value    delta,
            Value    alpha,
            Value    threshold,
            FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
            double  queue_sizing=1.0)
    {
        typedef ProblemBase<VertexId, SizeT, false> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing); // Default queue sizing is 1.0

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output page ranks if necessary
            if (!data_slices[gpu]->d_rank_curr) {
                Value    *d_rank1;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank1,
                                nodes * sizeof(Value)),
                            "WTFProblem cudaMalloc d_rank1 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_rank_curr = d_rank1;
            }

            if (!data_slices[gpu]->d_rank_next) {
                Value    *d_rank2;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank2,
                                nodes * sizeof(Value)),
                            "WTFProblem cudaMalloc d_rank2 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_rank_next = d_rank2;
            } 

            if (!data_slices[gpu]->d_refscore_curr) {
                Value    *d_rank3;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank3,
                                nodes * sizeof(Value)),
                            "WTFProblem cudaMalloc d_rank3 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_refscore_curr = d_rank3;
            }

            if (!data_slices[gpu]->d_refscore_next) {
                Value    *d_rank4;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_rank4,
                                nodes * sizeof(Value)),
                            "WTFProblem cudaMalloc d_rank4 failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_refscore_next = d_rank4;
            }

            if (!data_slices[gpu]->d_delta) {
                Value    *d_delta;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_delta,
                                1 * sizeof(Value)),
                            "WTFProblem cudaMalloc d_delta failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_delta = d_delta;
            }

            if (!data_slices[gpu]->d_alpha) {
                Value    *d_alpha;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_alpha,
                                1 * sizeof(Value)),
                            "WTFProblem cudaMalloc d_alpha failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_alpha = d_alpha;
            }

            if (!data_slices[gpu]->d_node_ids) {
                VertexId    *d_node_ids;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_node_ids,
                                nodes * sizeof(VertexId)),
                            "WTFProblem cudaMalloc d_node_ids failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_node_ids = d_node_ids;
            }

            if (!data_slices[gpu]->d_threshold) {
                Value    *d_threshold;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_threshold,
                                1 * sizeof(Value)),
                            "WTFProblem cudaMalloc d_threshold failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_threshold = d_threshold;
            }

            if (!data_slices[gpu]->d_src_node) {
                SizeT    *d_src_node;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_src_node,
                                1 * sizeof(SizeT)),
                            "WTFProblem cudaMalloc d_src_node failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_src_node = d_src_node;
            }

            // Allocate d_out_degrees if necessary
            if (!data_slices[gpu]->d_out_degrees) {
                VertexId    *d_out_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_out_degrees,
                                nodes * sizeof(VertexId)),
                            "WTFProblem cudaMalloc d_out_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_out_degrees = d_out_degrees;
            }

            // Allocate d_in_degrees if necessary
            if (!data_slices[gpu]->d_in_degrees) {
                SizeT    *d_in_degrees;
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_in_degrees,
                                nodes * sizeof(SizeT)),
                            "WTFProblem cudaMalloc d_in_degrees failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_in_degrees = d_in_degrees;
            }

            if (!data_slices[gpu]->d_cot_map) {
                bool    *d_cot_map;
                if (retval = util::GRError(cudaMalloc(
                        (void**)&d_cot_map,
                        nodes * sizeof(bool)),
                    "DOBFSProblem cudaMalloc d_cot_map failed", __FILE__, __LINE__)) return retval;
                data_slices[gpu]->d_cot_map = d_cot_map;
            }

            data_slices[gpu]->d_labels = NULL;

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_next, (Value)0.0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_rank_curr, (Value)1.0/nodes, nodes);

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_refscore_curr, (Value)0.0, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_refscore_next, (Value)0.0, nodes);
            
            // Compute degrees
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_out_degrees, 0, nodes);
            util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[gpu]->d_out_degrees, BaseProblem::graph_slices[0]->d_row_offsets, &BaseProblem::graph_slices[0]->d_row_offsets[1], -1, nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_in_degrees, 0, nodes);
            //util::MemsetMadVectorKernel<<<128, 128>>>(data_slices[gpu]->d_in_degrees, BaseProblem::graph_slices[0]->d_column_offsets, &BaseProblem::graph_slices[0]->d_column_offsets[1], -1, nodes);
 
            util::MemsetIdxKernel<<<128, 128>>>(data_slices[gpu]->d_node_ids, nodes);

            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_cot_map, false, nodes);

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_delta,
                            (Value*)&delta,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                        "WTFProblem cudaMemcpy d_delta failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_alpha,
                            (Value*)&alpha,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                        "WTFProblem cudaMemcpy d_alpha failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_threshold,
                            (Value*)&threshold,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                        "WTFProblem cudaMemcpy d_threshold failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->d_src_node,
                            (VertexId*)&src,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "WTFProblem cudaMemcpy d_src_node failed", __FILE__, __LINE__)) return retval;

            if (retval = util::GRError(cudaMemcpy(
                            d_data_slices[gpu],
                            data_slices[gpu],
                            sizeof(DataSlice),
                            cudaMemcpyHostToDevice),
                        "WTFProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }
        
        // Fillin the initial input_queue for WTF problem, this needs to be modified
        // in multi-GPU scene

        // Put every vertex in there
        util::MemsetIdxKernel<<<128, 128>>>(BaseProblem::graph_slices[0]->frontier_queues.d_keys[0], nodes);

        return retval;
    }

    /** @} */

};

} //namespace wtf
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
