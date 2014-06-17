// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_problem.cuh
 *
 * @brief GPU Storage management Structure for BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Breadth-First Search Problem structure stores device-side vectors for doing BFS computing on the GPU.
 *
 * @tparam _VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam _SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam _Value               Type of float or double to use for computing BC value.
 * @tparam _MARK_PREDECESSORS   Boolean type parameter which defines whether to mark predecessor value for each node.
 * @tparam _USE_DOUBLE_BUFFER   Boolean type parameter which defines whether to use double buffer.
 */
template <
    typename    VertexId,                       
    typename    SizeT,                          
    typename    Value,                          
    bool        _MARK_PREDECESSORS,
    bool        _ENABLE_IDEMPOTENCE,             
    bool        _USE_DOUBLE_BUFFER>
struct BFSProblem : ProblemBase<VertexId, SizeT, Value,
                                _USE_DOUBLE_BUFFER>
{

    static const bool MARK_PREDECESSORS     = _MARK_PREDECESSORS;
    static const bool ENABLE_IDEMPOTENCE    = _ENABLE_IDEMPOTENCE; 

    //Helper structures

    /**
     * @brief Data slice structure which contains BFS problem specific data.
     */
    struct DataSlice
    {
        // device storage arrays
        //VertexId        *d_labels;              /**< Used for source distance */
        util::Array1D<SizeT,VertexId > labels        ;
        //VertexId        *d_preds;               /**< Used for predecessor */
        util::Array1D<SizeT,VertexId > preds         ;
        //unsigned char   *d_visited_mask;        /**< used for bitmask for visited nodes */
        util::Array1D<SizeT,unsigned char > visited_mask  ;
        int             num_associate,gpu_idx;
        //VertexId        **d_associate_in;
        //VertexId        **h_associate_in;
        util::Array1D<SizeT,VertexId > *associate_in[2];
        util::Array1D<SizeT,VertexId*> associate_ins[2];
        //VertexId        **d_associate_out;
        //VertexId        **h_associate_out;
        util::Array1D<SizeT,VertexId > *associate_out;
        util::Array1D<SizeT,VertexId*> associate_outs;
        //VertexId        **d_associate_org;
        //VertexId        **h_associate_org;
        //util::Array1D<SizeT,VertexId > *associate_org;
        util::Array1D<SizeT,VertexId*> associate_orgs;
        //SizeT           * d_out_length;
        //SizeT           * out_length;
        util::Array1D<SizeT,SizeT    > out_length    ;
        //SizeT           * in_length;
        util::Array1D<SizeT,SizeT    > in_length[2]  ;
        //VertexId        * d_keys_in;
        util::Array1D<SizeT,VertexId > keys_in  [2]  ;

        DataSlice()
        {
            //d_labels        = NULL;
            //d_preds         = NULL;
            num_associate   = 0;
            gpu_idx         = 0;
            associate_in[0] = NULL;
            associate_in[1] = NULL;
            associate_out   = NULL;
            //associate_org   = NULL;
            labels          .SetName("labels"          );
            preds           .SetName("preds"           );
            visited_mask    .SetName("visited_mask"    );
            associate_ins[0].SetName("associate_ins[0]");
            associate_ins[1].SetName("associate_ins[1]");
            associate_outs  .SetName("associate_outs"  );
            associate_orgs  .SetName("associate_orgs"  );
            out_length      .SetName("out_length"      );
            in_length    [0].SetName("in_length[0]"    );
            in_length    [1].SetName("in_length[1]"    );
            keys_in      [0].SetName("keys_in[0]"      );
            keys_in      [1].SetName("keys_in[1]"      );
            //d_associate_in  = NULL;
            //h_associate_in  = NULL;
            //d_associate_out = NULL;
            //h_associate_out = NULL;
            //d_associate_org = NULL;
            //h_associate_org = NULL;
            //d_out_length    = NULL;
            //out_length      = NULL;
            //in_length       = NULL;
            //d_keys_in       = NULL;
            //d_visited_mask  = NULL;
        }

        ~DataSlice()
        {
            printf("~DataSlice begin.\n"); fflush(stdout);
            util::GRError(cudaSetDevice(gpu_idx),
                "~DataSlice cudaSetDevice failed", __FILE__, __LINE__);
            //if (d_labels) util::GRError(cudaFree(d_labels), "~DataSlice cudaFree d_labels failed", __FILE__, __LINE__);
            //if (d_preds ) util::GRError(cudaFree(d_preds ), "~DataSlice cudaFree d_preds failed" , __FILE__, __LINE__);
            //d_labels = NULL; d_preds = NULL;
            labels        .Release();
            preds         .Release();
            keys_in    [0].Release();
            keys_in    [1].Release();
            visited_mask  .Release();
            in_length  [0].Release();
            in_length  [1].Release();
            out_length    .Release();
            associate_orgs.Release();
 
            if (associate_in != NULL)
            {
                for (int i=0;i<num_associate;i++)
                {
                    associate_in[0][i].Release();
                    associate_in[1][i].Release();
                }
                //util::GRError(cudaFree(h_associate_in[i]), "~DataSlice cudaFree h_associate_in failed", __FILE__, __LINE__);
                //util::GRError(cudaFree(d_associate_in   ), "~DataSlice cudaFree d_associate_in failed", __FILE__, __LINE__);
                //util::GRError(cudaFree(d_keys_in        ), "_DataSlice cudaFree d_keys_in failed",      __FILE__, __LINE__);
                delete[] associate_in[0];
                delete[] associate_in[1];
                associate_in[0]=NULL;
                associate_in[1]=NULL;
                associate_ins[0].Release();
                associate_ins[1].Release();
                //delete[] h_associate_in;
                //delete[] in_length;
                //h_associate_in = NULL;
                //d_associate_in = NULL;
                //in_length      = NULL;
                //d_keys_in      = NULL;
            }

            if (associate_out != NULL)
            {
                for (int i=0;i<num_associate;i++)
                    associate_out[i].Release();
                //util::GRError(cudaFree(h_associate_out[i]), "~DataSlice cudaFree h_associate_out failed", __FILE__, __LINE__);
                //util::GRError(cudaFree(d_associate_out   ), "~DataSlice cudaFree d_associate_out failed", __FILE__, __LINE__);
                //util::GRError(cudaFree(d_out_length      ), "~DataSlice cudaFree d_out_length failed",    __FILE__, __LINE__);
                //delete[] h_associate_out;
                delete[] associate_out;
                associate_out=NULL;
                associate_outs.Release();
                //delete[] out_length;
                //h_associate_out = NULL;
                //d_associate_out = NULL;
                //d_out_length    = NULL;
                //out_length      = NULL;
            }

            //if (associate_org != NULL)
            //{
                //util::GRError(cudaFree(d_associate_org), "~DataSlice cudaFree d_associate_org failed", __FILE__, __LINE__);
                //delete[] h_associate_org;
                //d_associate_org = NULL;
                //h_associate_org = NULL;
            //}
            printf("~DataSlice end.\n"); fflush(stdout);
        }

        cudaError_t Init(
            int   num_gpus,
            int   gpu_idx,
            int   num_associate,
            SizeT num_nodes,
            SizeT num_in_nodes,
            SizeT num_out_nodes)
        {
            cudaError_t retval = cudaSuccess;
            this->gpu_idx       = gpu_idx;
            this->num_associate = num_associate;
            if (retval = util::GRError(cudaSetDevice(gpu_idx), "DataSlice cudaSetDevice failed", __FILE__, __LINE__)) return retval;
            // Create SoA on device
            if (retval = labels.Allocate(num_nodes,util::DEVICE)) return retval;
            //if (retval = util::GRError(cudaMalloc((void**)&(d_labels),num_nodes * sizeof(VertexId)),
            //                "DataSlice cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;

            if (_MARK_PREDECESSORS) 
            {
                if (retval = preds.Allocate(num_nodes,util::DEVICE)) return retval;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_preds),num_nodes * sizeof(VertexId)),
                //                "DataSlice cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
            }

            if (num_associate != 0)
            {
                //h_associate_org = new VertexId*[num_associate];
                //h_associate_org[0] = d_labels;
                if (retval = associate_orgs.Allocate(num_associate, util::HOST | util::DEVICE)) return retval;
                associate_orgs[0]=labels.GetPointer(util::DEVICE);
                if (_MARK_PREDECESSORS) 
                    //h_associate_org[1] = d_preds;
                    associate_orgs[1]=preds.GetPointer(util::DEVICE);
                if (retval = associate_orgs.Move(util::HOST,util::DEVICE)) return retval;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_associate_org), num_associate * sizeof(VertexId*)),
                //                "DataSlice cudaMalloc d_associate_org failed", __FILE__, __LINE__)) return retval;
                //if (retval = util::GRError(cudaMemcpy(d_associate_org, h_associate_org, 
                //                num_associate * sizeof(VertexId*), cudaMemcpyHostToDevice),
                //                "DataSlice cudaMemcpy d_associate_org failed", __FILE__, __LINE__)) return retval;
            }

            if (retval = in_length[0].Allocate(num_gpus,util::HOST)) return retval;
            if (retval = in_length[1].Allocate(num_gpus,util::HOST)) return retval;
            // Create incoming buffer on device
            if (num_in_nodes > 0)
            for (int t=0;t<2;t++) {
                //h_associate_in = new VertexId*[num_associate];
                associate_in[t]=new util::Array1D<SizeT,VertexId>[num_associate];
                associate_ins[t].SetName("associate_ins");
                if (retval = associate_ins[t].Allocate(num_associate, util::DEVICE | util::HOST)) return retval;
                for (int i=0;i<num_associate;i++)
                {
                    associate_in[t][i].SetName("associate_ins[]");
                    if (retval = associate_in[t][i].Allocate(num_in_nodes,util::DEVICE)) return retval;
                    associate_ins[t][i]=associate_in[t][i].GetPointer(util::DEVICE);
                    //if (retval = util::GRError(cudaMalloc((void**)&(h_associate_in[i]),num_in_nodes * sizeof(VertexId)),
                    //                "DataSlice cudamalloc h_associate_in failed", __FILE__, __LINE__)) break;
                }
                if (retval = associate_ins[t].Move(util::HOST,util::DEVICE)) return retval;
                if (retval = keys_in[t].Allocate(num_in_nodes,util::DEVICE)) return retval;
                //if (retval) return retval;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_associate_in),num_associate * sizeof(VertexId*)),
                //                "DataSlice cuaaMalloc d_associate_in failed", __FILE__, __LINE__)) return retval;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_keys_in     ), num_in_nodes * sizeof(VertexId)),
                //                "DataSlice cudaMalloc d_key_in failed",       __FILE__, __LINE__)) return retval;
                //if (retval = util::GRError(cudaMemcpy(d_associate_in, h_associate_in,
                //                num_associate * sizeof(VertexId*),cudaMemcpyHostToDevice),
                //                "DataSlice cudaMemcpy d_associate_in failed", __FILE__, __LINE__)) return retval;
                //in_length  = new SizeT[num_gpus];
                //memset(in_length,0,sizeof(SizeT)*num_gpus);
            }

             // Create outgoing buffer on device
            if (num_out_nodes > 0)
            {
                //h_associate_out = new VertexId*[num_associate];
                associate_out=new util::Array1D<SizeT,VertexId>[num_associate];
                associate_outs.SetName("associate_outs");
                if (retval = associate_outs.Allocate(num_associate, util::HOST | util::DEVICE)) return retval;
                for (int i=0;i<num_associate;i++)
                {
                    associate_out[i].SetName("associate_out[]");
                    if (retval = associate_out[i].Allocate(num_out_nodes, util::DEVICE)) return retval;
                    associate_outs[i]=associate_out[i].GetPointer(util::DEVICE);
                    //if (retval = util::GRError(cudaMalloc((void**)&(h_associate_out[i]),num_out_nodes * sizeof(VertexId)),
                    //                 "DataSlice cudamalloc h_associate_out failed", __FILE__, __LINE__)) break;
                }
                if (retval = associate_outs.Move(util::HOST, util::DEVICE)) return retval;
                if (retval = out_length.Allocate(num_gpus,util::HOST | util::DEVICE)) return retval;
                //if (retval) return retval;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_associate_out),num_associate * sizeof(VertexId*)),
                //                "DataSlice cuaaMalloc d_associate_out failed", __FILE__, __LINE__)) return retval;
                //if (retval = util::GRError(cudaMemcpy(d_associate_out, h_associate_out,
                //                num_associate * sizeof(VertexId*),cudaMemcpyHostToDevice),
                //                "DataSlice cudaMemcpy d_associate_out failed", __FILE__, __LINE__)) return retval;
                //out_length = new SizeT[num_gpus];
                //memset(out_length, 0, sizeof(SizeT)*num_gpus);
                //if (retval = util::GRError(cudaMalloc((void**)&(d_out_length), num_gpus * sizeof(SizeT)),
                //                "DataSlice cuaMalloc d_out_length failed",     __FILE__, __LINE__)) return retval;
            }
            return retval;
        } // Init
    }; // DataSlice

    // Members
    
    // Set of data slices (one for each GPU)
    //DataSlice           **data_slices;
    util::Array1D<SizeT,DataSlice> *data_slices;
   
    // Nasty method for putting struct on device
    // while keeping the SoA structure
    //DataSlice           **d_data_slices;

    // Methods

    /**
     * @brief BFSProblem default constructor
     */
    BFSProblem()
    {
        data_slices      = NULL;
        //d_data_slices    = NULL;
    }

    /**
     * @brief BFSProblem constructor
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on.
     * @param[in] num_gpus Number of the GPUs used.
     */
    BFSProblem(bool        stream_from_host,       // Only meaningful for single-GPU
               std::string partition_method,
               const Csr<VertexId, Value, SizeT> &graph,
               int         num_gpus,
               int*        gpu_idx)
    {
        Init(
            stream_from_host,
            partition_method,
            graph,
            num_gpus,
            gpu_idx);
    }

    /**
     * @brief BFSProblem default destructor
     */
    ~BFSProblem()
    {
        printf("~BFSProblem() begin.\n");fflush(stdout);
        if (data_slices==NULL) return;
        for (int i = 0; i < this->num_gpus; ++i)
        {
            //delete data_slices[i]; data_slices[i]=NULL;
            if (util::GRError(cudaSetDevice(this->gpu_idx[i]),
                "~BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;
            data_slices[i].Release();
            //if (d_data_slices[i]) util::GRError(cudaFree(d_data_slices[i]), "~BFSProblem cudaFree data_slices failed", __FILE__, __LINE__);
        }
        //if (d_data_slices) delete[] d_data_slices; d_data_slices=NULL;
        if (data_slices  ) delete[] data_slices; data_slices=NULL;
        printf("~BFSProblem() end.\n");fflush(stdout);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Copy result labels and/or predecessors computed on the GPU back to host-side vectors.
     *
     * @param[out] h_labels host-side vector to store computed node labels (distances from the source).
     * @param[out] h_preds host-side vector to store predecessor vertex ids.
     *
     *\return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Extract(VertexId *h_labels, VertexId *h_preds)
    {
        cudaError_t retval = cudaSuccess;

        do {
            if (this->num_gpus == 1) {

                // Set device
                if (util::GRError(cudaSetDevice(this->gpu_idx[0]),
                            "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

                data_slices[0]->labels.SetPointer(h_labels);
                if (retval = data_slices[0]->labels.Move(util::DEVICE,util::HOST)) return retval;
                //if (retval = util::GRError(cudaMemcpy(
                //                h_labels,
                //                data_slices[0]->d_labels,
                //                sizeof(VertexId) * this->nodes,
                //                cudaMemcpyDeviceToHost),
                //            "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

                if (_MARK_PREDECESSORS) {
                    data_slices[0]->preds.SetPointer(h_preds);
                    if (retval = data_slices[0]->preds.Move(util::DEVICE,util::HOST)) return retval;
                    //if (retval = util::GRError(cudaMemcpy(
                    //                h_preds,
                    //                data_slices[0]->d_preds,
                    //                sizeof(VertexId) * this->nodes,
                    //                cudaMemcpyDeviceToHost),
                    //            "BFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                }

            } else {
                VertexId **th_labels=new VertexId*[this->num_gpus];
                VertexId **th_preds =new VertexId*[this->num_gpus];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {
                    //th_labels[gpu]=new VertexId[this->out_offsets[gpu][1]];
                    //th_preds [gpu]=new VertexId[this->out_offsets[gpu][1]];

                    if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]),
                                "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
                    if (retval = data_slices[gpu]->labels.Move(util::DEVICE,util::HOST)) return retval;
                    th_labels[gpu]=data_slices[gpu]->labels.GetPointer(util::HOST);
                    //if (retval = util::GRError(cudaMemcpy(
                    //                th_labels[gpu],
                    //                data_slices[gpu]->d_labels,
                    //                sizeof(VertexId) * this->out_offsets[gpu][1],
                    //                cudaMemcpyDeviceToHost),
                    //            "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
                    if (_MARK_PREDECESSORS) {
                        if (retval = data_slices[gpu]->preds.Move(util::DEVICE,util::HOST)) return retval;
                        th_preds[gpu]=data_slices[gpu]->preds.GetPointer(util::HOST);
                        //if (retval = util::GRError(cudaMemcpy(
                        //                th_preds[gpu],
                        //                data_slices[gpu]->d_preds,
                        //                sizeof(VertexId) * this->out_offsets[gpu][1],
                        //                cudaMemcpyDeviceToHost),
                        //             "BFSProblem cudaMemcpy d_preds failed", __FILE__, __LINE__)) break;
                    }
                } //end for(gpu)

                for (VertexId node=0;node<this->nodes;node++)
                    h_labels[node]=th_labels[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                if (_MARK_PREDECESSORS)
                    for (VertexId node=0;node<this->nodes;node++)
                        h_preds[node]=th_preds[this->partition_tables[0][node]][this->convertion_tables[0][node]];
                for (int gpu=0;gpu<this->num_gpus;gpu++)
                {
                    if (retval = data_slices[gpu]->labels.Release(util::HOST)) return retval;
                    if (retval = data_slices[gpu]->preds.Release(util::HOST)) return retval;
                    //delete[] th_labels[gpu];th_labels[gpu]=NULL;
                    //delete[] th_preds [gpu];th_preds [gpu]=NULL;
                }
                delete[] th_labels;th_labels=NULL;
                delete[] th_preds ;th_preds =NULL;
            } //end if (data_slices.size() ==1)
        } while(0);

        return retval;
    }

    /**
     * @brief BFSProblem initialization
     *
     * @param[in] stream_from_host Whether to stream data from host.
     * @param[in] graph Reference to the CSR graph object we process on. @see Csr
     * @param[in] _num_gpus Number of the GPUs used.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
            bool            stream_from_host,       // Only meaningful for single-GPU
            std::string     partition_method,
            Csr<VertexId, Value, SizeT> &graph,
            int             num_gpus,
            int*            gpu_idx)
    {
        printf("ProblemBase Init start.\n");fflush(stdout);
        ProblemBase<VertexId, SizeT,Value,_USE_DOUBLE_BUFFER>::Init(
            stream_from_host,
            partition_method,
            graph,
            num_gpus,
            gpu_idx);
        // No data in DataSlice needs to be copied from host
        printf("ProblemBase Init end.\n");fflush(stdout);

        //graph.DisplayGraph("Original");
        /**
         * Allocate output labels/preds
         */
        cudaError_t retval = cudaSuccess;
        data_slices = new util::Array1D<SizeT,DataSlice>[this->num_gpus];
        //util::cpu_mt::PrintCPUArray<SizeT,int>("Partition",this->partition_tables[0],graph.nodes);
        //util::cpu_mt::PrintCPUArray<SizeT,VertexId>("Convertion",this->convertion_tables[0],graph.nodes);
        //data_slices   = new DataSlice*[this->num_gpus];
        //d_data_slices = new DataSlice*[this->num_gpus];

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                printf("GPU %d\n",gpu);fflush(stdout);
                //this->sub_graphs[gpu].DisplayGraph("subgraph");
                data_slices[gpu].SetName("data_slices[]");
                if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;
                if (retval = data_slices[gpu].Allocate(1,util::DEVICE | util::HOST)) return retval;
                //data_slices[gpu] = new DataSlice;
                //if (retval = util::GRError(cudaMalloc((void**)&(d_data_slices[gpu]),sizeof(DataSlice)),
                //               "BFSProblem cudaMalloc d_data_slices failed", __FILE__, __LINE__)) return retval;
                //printf("0");fflush(stdout);
                DataSlice* _data_slice = data_slices[gpu].GetPointer(util::HOST);
                /*printf("1");fflush(stdout);
                SizeT      _nodes      = this->sub_graphs[gpu].nodes;
                printf("2");fflush(stdout);
                int        _num_gpus   = this->num_gpus;
                printf("3");fflush(stdout);
                int        _gpu_idx    = this->gpu_idx[gpu];
                printf("4");fflush(stdout);
                SizeT      _in_offset  = this->graph_slices[gpu]->in_offset[this->num_gpus];
                printf("5");fflush(stdout);
                SizeT      _out_offset = this->graph_slices[gpu]->out_offset[this->num_gpus]-this->graph_slices[gpu]->out_offset[1];
                printf("pointer = %p, _num_gpus = %d, _gpu_idx = %d, _nodes = %d, _in_offset = %d, _out_offset = %d\n",_data_slice, _num_gpus, _gpu_idx, _nodes, _in_offset, _out_offset);fflush(stdout);*/
                if (this->num_gpus > 1)
                {
                    if (_MARK_PREDECESSORS)
                        //data_slices[gpu]->Init(this->num_gpus, this->gpu_idx[gpu], 2, 
                        _data_slice->Init(this->num_gpus,this->gpu_idx[gpu], 2,
                            this->sub_graphs[gpu].nodes,
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus]-this->graph_slices[gpu]->out_offset[1]);
                    else   //data_slices[gpu]->Init(this->num_gpus, this->gpu_idx[gpu], 1, 
                          _data_slice->Init(this->num_gpus, this->gpu_idx[gpu], 1,
                            this->sub_graphs[gpu].nodes,
                            this->graph_slices[gpu]->in_offset[this->num_gpus],
                            this->graph_slices[gpu]->out_offset[this->num_gpus]-this->graph_slices[gpu]->out_offset[1]);
                } else {
                    //data_slices[gpu]->Init(this->num_gpus,this->gpu_idx[gpu], 0,
                    _data_slice->Init(this->num_gpus, this->gpu_idx[gpu], 0,
                        this->sub_graphs[gpu].nodes, 0, 0);
                }
            } //end for(gpu)
            //} // end if (num_gpus)
        } while (0);
        printf("problemInit finish.\n");fflush(stdout);

        return retval;
    }

    /**
     *  @brief Performs any initialization work needed for BFS problem type. Must be called prior to each BFS run.
     *
     *  @param[in] src Source node for one BFS computing pass.
     *  @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     *  @param[in] queue_sizing Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively).
     * 
     *  \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Reset(
            VertexId    src,
            FrontierType frontier_type,             // The frontier type (i.e., edge/vertex/mixed)
            double queue_sizing)                    // Size scaling factor for work queue allocation (e.g., 1.0 creates n-element and m-element vertex and edge frontiers, respectively). 0.0 is unspecified.
    {
        typedef ProblemBase<VertexId, SizeT,Value,
                                _USE_DOUBLE_BUFFER> BaseProblem;
        //load ProblemBase Reset
        BaseProblem::Reset(frontier_type, queue_sizing);

        cudaError_t retval = cudaSuccess;

        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            // Set device
            if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]),
                        "BSFProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

            // Allocate output labels if necessary
            if (data_slices[gpu]->labels.GetPointer(util::DEVICE)==NULL)
                if (retval = data_slices[gpu]->labels.Allocate(this->sub_graphs[gpu].nodes,util::DEVICE)) return retval;
            //if (!data_slices[gpu]->d_labels) {
            //    VertexId    *d_labels;
            //    if (retval = util::GRError(cudaMalloc(
            //                    (void**)&d_labels,
            //                    this->sub_graphs[gpu].nodes * sizeof(VertexId)),
            //                "BFSProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) return retval;
            //    data_slices[gpu]->d_labels = d_labels;
            //}

            //util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_labels, -1, this->sub_graphs[gpu].nodes);
            util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->labels.GetPointer(util::DEVICE), -1, this->sub_graphs[gpu].nodes);

            // Allocate preds if necessary
            if (_MARK_PREDECESSORS)
            {
                if (data_slices[gpu].GetPointer(util::HOST)->preds.GetPointer(util::DEVICE)==NULL)
                    if (retval = data_slices[gpu]->preds.Allocate(this->sub_graphs[gpu].nodes, util::DEVICE)) return retval;
                util::MemsetKernel<<<128,128>>>(data_slices[gpu]->preds.GetPointer(util::DEVICE), -2, this->sub_graphs[gpu].nodes);
            }
            //if (_MARK_PREDECESSORS && !data_slices[gpu]->d_preds) {
                //VertexId    *d_preds;
                //if (retval = util::GRError(cudaMalloc(
                //                (void**)&d_preds,
                //                this->sub_graphs[gpu].nodes * sizeof(VertexId)),
                //            "BFSProblem cudaMalloc d_preds failed", __FILE__, __LINE__)) return retval;
                //data_slices[gpu]->d_preds = d_preds;
            //}
            
            //if (_MARK_PREDECESSORS)
            //    util::MemsetKernel<<<128, 128>>>(data_slices[gpu]->d_preds, -2, this->sub_graphs[gpu].nodes);

            if (retval = data_slices[gpu].Move(util::HOST, util::DEVICE)) return retval;  
            //if (retval = util::GRError(cudaMemcpy(
            //                d_data_slices[gpu],
            //                data_slices[gpu],
            //                sizeof(DataSlice),
            //                cudaMemcpyHostToDevice),
            //            "BFSProblem cudaMemcpy data_slices to d_data_slices failed", __FILE__, __LINE__)) return retval;

        }

        // Fillin the initial input_queue for BFS problem
        int gpu;
        VertexId tsrc;
        if (this->num_gpus <= 1) 
        {
           gpu=0;tsrc=src;
        } else {
            gpu = this->partition_tables[0][src];
            tsrc= this->convertion_tables[0][src];
        }
        if (retval = util::GRError(cudaSetDevice(this->gpu_idx[gpu]), "BFSProblem cudaSetDevice failed", __FILE__, __LINE__)) return retval;

        /*if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[gpu]->frontier_queues.d_keys[0],
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;*/
        if (retval = util::GRError(cudaMemcpy(
                        BaseProblem::graph_slices[gpu]->frontier_queues.keys[0].GetPointer(util::DEVICE),
                        &tsrc,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                     "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        VertexId src_label = 0; 
        if (retval = util::GRError(cudaMemcpy(
                        data_slices[gpu]->labels.GetPointer(util::DEVICE)+tsrc,//data_slices[gpu]->d_labels+tsrc,
                        &src_label,
                        sizeof(VertexId),
                        cudaMemcpyHostToDevice),
                    "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        
        if (_MARK_PREDECESSORS) {
            VertexId src_pred = -1; 
            if (retval = util::GRError(cudaMemcpy(
                            data_slices[gpu]->preds.GetPointer(util::DEVICE)+tsrc,//data_slices[gpu]->d_preds+tsrc,
                            &src_pred,
                            sizeof(VertexId),
                            cudaMemcpyHostToDevice),
                        "BFSProblem cudaMemcpy frontier_queues failed", __FILE__, __LINE__)) return retval;
        }

        return retval;
    } // reset
    /** @} */

}; //bfs_problem

} //namespace bfs
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
