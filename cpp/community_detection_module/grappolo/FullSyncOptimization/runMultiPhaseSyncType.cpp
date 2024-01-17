// ***********************************************************************
//
//            Grappolo: A C++ library for graph clustering
//               Mahantesh Halappanavar (hala@pnnl.gov)
//               Pacific Northwest National Laboratory
//
// ***********************************************************************
//
//       Copyright (2014) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************

#include "mg_procedure.h"
#include "defs.h"
#include "sync_comm.h"

using namespace std;
//WARNING: This will overwrite the original graph data structure to
//         minimize memory footprint
// Return: C_orig will hold the cluster ids for vertices in the original graph
//         Assume C_orig is initialized appropriately
//WARNING: Graph G will be destroyed at the end of this routine
void runMultiPhaseSyncType(graph *G, mgp_graph *mg_graph, long *C_orig, int syncType, long minGraphSize,
                           double threshold, double C_threshold, int numThreads, int threadsOpt)
{
    double totTimeClustering=0, totTimeBuildingPhase=0, totTimeColoring=0, tmpTime=0;
    int tmpItr=0, totItr = 0;
    long NV = G->numVertices;


    /* Step 1: Find communities */
    double prevMod = -1;
    double currMod = -1;
    long phase = 1;
    int freedom = 0;

    graph *Gnew; //To build new hierarchical graphs
    long numClusters;
    long *C = (long *) malloc (NV * sizeof(long));
    assert(C != 0);
#pragma omp parallel for
    for (long i=0; i<NV; i++) {
        C[i] = -1;
    }

    bool nonET = false; //Make sure that at least one phase with lower threshold runs
    while(1){
        prevMod = currMod;
        //Compute clusters
        if(nonET == false) {
            switch (syncType){
                case 2:
                    currMod = parallelLouvainMethodFullSync(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
                case 4:
                    currMod = parallelLouvainMethodFullSyncEarly(G, C, numThreads, currMod, C_threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
                case 3:
                    currMod = parallelLouvianMethodEarlyTerminate(G, C, numThreads, currMod, C_threshold, &tmpTime, &tmpItr); break;
                default:
                    currMod = parallelLouvainMethodFullSync(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
            }
        } else {
            switch (syncType){
                case 2:
                    currMod = parallelLouvainMethodFullSync(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
                case 4:
                    currMod = parallelLouvainMethodFullSyncEarly(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
                case 3:
                    currMod = parallelLouvianMethodEarlyTerminate(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr);
                    break;
                default:
                    currMod = parallelLouvainMethodFullSync(G, C, numThreads, currMod, threshold, &tmpTime, &tmpItr,syncType, freedom);
                    break;
            }
            nonET = true;
        }

        totTimeClustering += tmpTime;
        totItr += tmpItr;

        //Renumber the clusters contiguiously
        numClusters = renumberClustersContiguously(C, G->numVertices);

        //Keep track of clusters in C_orig
        if(phase == 1) {
#pragma omp parallel for
            for (long i=0; i<NV; i++) {
                C_orig[i] = C[i]; //After the first phase
            }
        } else {
#pragma omp parallel for
            for (long i=0; i<NV; i++) {
                assert(C_orig[i] < G->numVertices);
                if (C_orig[i] >=0)
                    C_orig[i] = C[C_orig[i]]; //Each cluster in a previous phase becomes a vertex
            }
        }

        //Break if too many phases or iterations
        if((phase > 200)||(totItr > 10000)) {
            break;
        }

        //Check for modularity gain and build the graph for next phase
        //In case coloring is used, make sure the non-coloring routine is run at least once
        if( (currMod - prevMod) > threshold ) {
            Gnew = (graph *) malloc (sizeof(graph)); assert(Gnew != 0);
            tmpTime =  buildNextLevelGraphOpt(G, mg_graph, Gnew, C, numClusters, numThreads);
            totTimeBuildingPhase += tmpTime;
            //Free up the previous graph
            free(G->edgeListPtrs);
            free(G->edgeList);
            free(G);
            G = Gnew; //Swap the pointers
            G->edgeListPtrs = Gnew->edgeListPtrs;
            G->edgeList = Gnew->edgeList;

            //Free up the previous cluster & create new one of a different size
            free(C);
            C = (long *) malloc (numClusters * sizeof(long)); assert(C != 0);

#pragma omp parallel for
            for (long i=0; i<numClusters; i++) {
                C[i] = -1;
            }
            phase++; //Increment phase number
        } else { //To force another phase with coloring again
            if ( ((syncType == 3)||(syncType == 4))&&(nonET == false) ) {
                nonET = true; //Run at least one loop of ET routine with smaller threshold
            }
            else {
                break; //Modularity gain is not enough. Exit.
            }
        }

    } //End of while(1)


    //Clean up:
    free(C);
    if(G != 0) {
        free(G->edgeListPtrs);
        free(G->edgeList);
        free(G);
    }
}//End of runMultiPhaseLouvainAlgorithm()
