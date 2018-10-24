#include <algorithm>
#include <assert.h>
#include <limits.h>
#include <map>
#include <math.h>
#include <random>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

//#include <cudpp_config.h>
#include <cudpp.h>
#include <cudpp_hash.h>
#define CUDPP_APP_COMMON_IMPL

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"

#include "crc64.c"

struct {
        uint32_t index;
        uint32_t value;
} insideEntry;
const unsigned kBlockSize = 64;
const unsigned kGridSize = 16384;
const unsigned BigPrimeDevisor = 4294967291u;
const unsigned EmptyFlag = 0xFFFFFFFFu;
const unsigned EmptyIndex = 0xFFFFFFFFu;
const unsigned RetrieveNULL = 0xFFFFFFFFu;

#define BUILD_SUCCESS 0
#define BUILD_FAILURE 1

class scalaGPUHash
{

      public:
        uint32_t NumOfHashFunctions;
        uint32_t NumOfEntry;
        float spaceUsage;
        uint32_t stashSize;
        uint2 *hashConstants;
        uint2 hashStashConstants;
        uint32_t tableSize;
        uint32_t fullTableSize;
	uint32_t maxIterations;
	uint32_t maxEvict;
        uint64_t *keys;
        uint32_t *values;
        uint32_t *deviceIndex;
        uint32_t *deviceFlag;
        uint64_t *deviceKeys;
        uint32_t *deviceValues;

        uint2 *deviceHashConstants;
        uint2 *deviceHashStashConstants;

        uint32_t *buildFailures;
        uint32_t buildStatus = 0;
        scalaGPUHash(float space_usage_, uint32_t NumOfHashFunctions_,
                     uint32_t NumOfEntry_, uint64_t *keys_, uint32_t *values_)
        {
                spaceUsage = space_usage_;
                NumOfHashFunctions = NumOfHashFunctions_;
                NumOfEntry = NumOfEntry_;
                keys = keys_;
                values = values_;
                init();
        }

        // void CUrandom(int N, uint32_t *hosdtData, int range);
        void generateHashFunction();
        inline void resetIndex();
        inline void resetFlag();
        unsigned ComputeMaxEvict();
        int init();
        int hashBuild();
        int hashRetrieve(uint64_t *query, uint32_t NumOfQuery,
                         uint32_t *Retrieve);
        int hashRelease();
        void hashRehash();
        void hashClear();
};

__device__ uint32_t hashFunction(uint64_t key, uint2 constants,
                                                 uint32_t tableSize)
{
        uint64_t hashKey;
        hashKey =
            (((uint64_t)constants.x ^ key + (uint64_t)constants.y) % (uint64_t)BigPrimeDevisor) % (uint64_t)tableSize;
	//if (hashKey > tableSize){
	//	printf("in hashfunction hashKey is too bigger,%llu\t%u\n",hashKey,tableSize);
	//}
        return (uint32_t)hashKey;
}

__device__ uint32_t determineNextLocation(uint64_t *key, uint32_t index,
                                                 uint32_t NumHashFunctions,
                                                 uint32_t preLocation,
                                                 uint2 *constants,
                                                 uint32_t tableSize)
{
        uint64_t thisKey = key[index];
        uint32_t nextLocation = hashFunction(thisKey,constants[0],tableSize);
        for (int i = NumHashFunctions-2; i >= 0 ; --i) {
                //nextLocation = (uint32_t)hashFunction(thisKey, constants[i], tableSize);
		//if (nextLocation > tableSize){
		//	printf("thisKey %llu\t nextlocation %u\t constants.x constants.y %u\t%u\t%u\n",thisKey,nextLocation,constants[i].x,constants[i].y,i);
			//return 0xFFFFFFFF;
		//	}
		nextLocation = (preLocation == hashFunction(thisKey,constants[i],tableSize) ? hashFunction(thisKey,constants[i+1],tableSize) : nextLocation );
                //if (preLocation == nextLocation and i != NumHashFunctions-1) {
                 //       nextLocation=hashFunction(thisKey, constants[i + 1],
                  //                          tableSize);
		//	break;
                //}
		//if (preLocation == nextLocation and i== NumHashFunctions-1){
		//	nextLocation=hashFunction(thisKey,constants[NumHashFunctions-1],tableSize);
        	}
	
	return nextLocation;
}
__global__ void hashInsert(uint64_t *key, uint32_t *index, uint32_t *flag,
                           uint32_t NumOfEntry, uint32_t NumHashFunctions,
                           uint2 *constants, uint32_t tableSize,
                           uint32_t stashSize, uint2 *stashConstants,
                           uint32_t *buildFailures, uint32_t maxEvict)
{

        // uint32_t expectFlag = 0;
        uint32_t targetFlag = 1;
        uint32_t location=0;
        uint32_t globaltid = threadIdx.x + blockIdx.x * blockDim.x;
        if (globaltid >= NumOfEntry) {
                return;
        }
        uint64_t thisKey = key[globaltid];
	uint32_t thisIndex = globaltid;
        // calculate all locations first
	// location is index,store index in flag
        location = hashFunction(thisKey, constants[0], tableSize);
	        // for (uint32_t i = 0; i < NumHashFunctions; ++i) {
        //       location[i] = hashFunction(thisKey, constants[i], tableSize);
        //}
        for (uint32_t i = 0; i < maxEvict; ++i) {
		

                // targetFlag = atomicCAS(&flag[location[i]], 1, 0);
                //thisIndex = atomicCAS(&flag[location], EmptyFlag,thisIndex )
		thisIndex = atomicExch(&flag[location],thisIndex);
                if (thisIndex == EmptyFlag) {
                        //index[globaltid] = location;
                        break;
                }
                location =
                    determineNextLocation(key, thisIndex, NumHashFunctions,
                                          location, constants, tableSize);
		if (location > tableSize){
			printf("threadid %d\tthisKey %llu\t location %u\t constants.x constants.y %u\t%u\t%u\n",globaltid,thisKey,location,constants[0].x,constants[0].y,i);
			
			break;
			}
        }
        // stash
        if (thisIndex != EmptyFlag and location==tableSize+1) {
                uint32_t slot =
                    hashFunction(key[thisIndex], stashConstants[0], stashSize);
                // targetFlag = atomicCAS(&flag[slot + tableSize], 1, 0);
                thisIndex=
                    atomicCAS(&flag[slot + tableSize], EmptyFlag, thisIndex);
                if (thisIndex== EmptyFlag) {
                        //index[globaltid] = slot + tableSize;
                }
        }
        if (thisIndex != EmptyFlag) {
                index[globaltid] = EmptyIndex;
                atomicAdd(buildFailures, 1);
        }
#ifdef DEBUG
        printf("%d insert failed\n", globaltid);
#endif
}

dim3 ComputeGridDim(unsigned n)
{
        dim3 grid((n + kBlockSize - 1) / kBlockSize);
        return grid;
}

inline void scalaGPUHash::resetIndex()
{
        cudaMemset(deviceIndex, 0xFF, sizeof(uint32_t) * NumOfEntry);
}

inline void scalaGPUHash::resetFlag()
{
        cudaMemset(deviceFlag, 0xFF, sizeof(uint32_t) * fullTableSize);
}

unsigned scalaGPUHash::ComputeMaxEvict() {
	float lg_input_size = (float)(log((float)NumOfEntry)/(spaceUsage));
	float load_factor = float(NumOfEntry)/tableSize;
	float ln_load_factor = (float)(log(load_factor))/(log(2.71828183));
	
	unsigned maxEvict= (unsigned )(4.0*ceil(-1.0/(0.02855 + 1.1594722 *ln_load_factor)* lg_input_size));
	return 7*log(NumOfEntry);

}

__global__ void reArrangeIndex(uint32_t *deviceIndex, uint32_t *deviceFlag,
                               uint32_t fullTableSize)
{
        uint32_t globaltid = threadIdx.x + blockDim.x * blockIdx.x;
        if (globaltid >= fullTableSize) {
                return;
        }
	uint32_t thisIndex = deviceFlag[globaltid];
        //deviceFlag[deviceIndex[globaltid]] = globaltid;
        deviceIndex[thisIndex] = globaltid;
}

__global__ void retrieve(uint64_t *keys, uint32_t *flags, uint32_t *values,
                         uint32_t NumOfEntry, uint32_t NumOfHashFunctions,
                         uint2 *constants, uint2 *stashConstants,
                         uint32_t tableSize, uint32_t stashSize,
                         uint64_t *query, uint32_t *retrieve,
                         uint32_t NumOfQuery)
{
        uint32_t globaltid = threadIdx.x + blockDim.x * blockIdx.x;
        uint32_t queryStatus = BUILD_FAILURE;
        if (globaltid >= NumOfQuery) {
                return;
        }

        uint64_t thisKey = query[globaltid];

        for (uint32_t i = 0; i < NumOfHashFunctions; ++i) {
                uint32_t location =
                    hashFunction(thisKey, constants[i], tableSize);
                if (keys[flags[location]] == thisKey) {
                        retrieve[globaltid] = values[flags[location]];
                        queryStatus = BUILD_SUCCESS;
                        break;
                }
        }
        if (queryStatus == BUILD_FAILURE) {
                uint32_t location =
                    hashFunction(thisKey, stashConstants[0], stashSize) +
                    tableSize;
                if (keys[flags[location]] == thisKey) {
                        retrieve[globaltid] = values[flags[location]];
                        queryStatus = BUILD_SUCCESS;
                }
        }
        if (queryStatus == BUILD_FAILURE) {
                retrieve[globaltid] = RetrieveNULL;
        }
}

int scalaGPUHash::init()
{
        tableSize = ceil(NumOfEntry * spaceUsage);
	stashSize = 101;
        fullTableSize = tableSize + stashSize;
        hashConstants = new uint2[NumOfHashFunctions];
	maxIterations = 10;
	maxEvict = ComputeMaxEvict();
        cudaMalloc((void **)&deviceIndex, sizeof(uint32_t) * NumOfEntry);
        cudaMalloc((void **)&deviceKeys, sizeof(uint64_t) * NumOfEntry);
        cudaMalloc((void **)&deviceValues, sizeof(uint32_t) * NumOfEntry);

        cudaMalloc((void **)&buildFailures, sizeof(uint32_t) * 1);

        cudaMalloc((void **)&deviceFlag, sizeof(uint32_t) * fullTableSize);

        cudaMalloc((void **)&deviceHashConstants,
                   sizeof(uint2) * NumOfHashFunctions);
        cudaMalloc((void **)&deviceHashStashConstants, sizeof(uint2) * 1);

        cudaMemcpy(deviceKeys, keys, NumOfEntry * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(deviceValues, values, NumOfEntry * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
}

/*
void CUrandom(int N, uint32_t *hostData, int range)
{
        curandGenerator_t gen;
        float *devData;
        float *tmp_hostData;

        tmp_hostData = new float[N];
        CUDA_SAFE_CALL(cudaMalloc((void **)&devData, sizeof(float) * N));

        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, devData, N);
        cudaMemcpy(tmp_hostData, devData, N * sizeof(float),
                   cudaMemcpyDeviceToHost);
        curandDestroyGenerator(gen);
        cudaFree(devData);
        for (int i = 0; i < N; ++i) {
                hostData[i] = ceil(tmp_hostData[i] * range);
        }
        delete[] tmp_hostData;
}
*/

void CUrandom(int N, uint32_t *hostData, int range)
{
        srand((int)time(NULL));
        for (uint32_t i = 0; i < N; i++) {
                hostData[i] = random();
        }
}
void scalaGPUHash::generateHashFunction()
{
        uint32_t A[(NumOfHashFunctions + 1) * 2];
        CUrandom((NumOfHashFunctions + 1) * 2, A, 0xFFFFFFFF);
        for (int i = 0; i < NumOfHashFunctions; ++i) {
                hashConstants[i].x = A[i * 2] % BigPrimeDevisor;
                hashConstants[i].y = A[i * 2 + 1] % BigPrimeDevisor;
        }
        hashStashConstants.x = A[NumOfHashFunctions * 2] % BigPrimeDevisor;
        hashStashConstants.y = A[NumOfHashFunctions * 2 + 1] % BigPrimeDevisor;

        cudaMemcpy(deviceHashConstants, hashConstants,
                   sizeof(uint2) * NumOfHashFunctions, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceHashStashConstants, &hashStashConstants, sizeof(uint2),
                   cudaMemcpyHostToDevice);
}

int scalaGPUHash::hashBuild()
{
        generateHashFunction();
        resetIndex();
        resetFlag();
        dim3 grid = ComputeGridDim(NumOfEntry);
	//maxEvict= ComputeMaxEvict();
        for (uint32_t i = 0; i < maxIterations; ++i) {
                hashInsert<<<grid, kBlockSize>>>(
                    deviceKeys, deviceIndex, deviceFlag, NumOfEntry,
                    NumOfHashFunctions, deviceHashConstants, tableSize,
                    stashSize, deviceHashStashConstants, buildFailures,maxEvict);
                cudaDeviceSynchronize();
                cudaMemcpy(&buildStatus, buildFailures, sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
                if (buildStatus == BUILD_SUCCESS) {
                        //cudaMemset(deviceFlag, 0x00,
                         //          sizeof(uint32_t) * fullTableSize);
                        //reArrangeIndex<<<grid, kBlockSize>>>(
                         //   deviceIndex, deviceFlag, NumOfEntry);

                        return 1;
                } else {
                        hashRehash();
                }
        }
        if (buildStatus != 0) {
                printf("Build Failure!!!\n");
		hashRelease();
                exit(EXIT_FAILURE);
                return 0;
        }
}

void scalaGPUHash::hashRehash()
{
        if (buildStatus != BUILD_SUCCESS) {
                // refresh random constants
                generateHashFunction();
                // reset index
                resetIndex();
                // reset deviceFlag
                resetFlag();
                // reset buildFailures to zero
		cudaMemset(buildFailures,0x00,sizeof(uint32_t));
                //buildFailures = 0;
        }
}

void scalaGPUHash::hashClear()
{
        // after build successful
        if (buildStatus != 0) {
                return;
        }
        // compress unusefull data for retrieve
        // only deviceKeys,deviceFlags,deviceValues,hashConstants and
        // hashStashConstants will be used
        cudaFree(deviceIndex);
        cudaFree(buildFailures);
}

int scalaGPUHash::hashRelease()
{
        // release all data
        // if build not successful
        if (buildStatus != 0) {
                cudaFree(deviceIndex);
                cudaFree(buildFailures);
        }
        cudaFree(deviceFlag);
        cudaFree(deviceKeys);
        cudaFree(deviceValues);

        cudaFree(deviceHashConstants);
        cudaFree(deviceHashStashConstants);

        return 0;
}

int scalaGPUHash::hashRetrieve(uint64_t *query, uint32_t NumOfQuery,
                               uint32_t *Retrieve)
{
        uint64_t *deviceQuery;
        uint32_t *deviceRetrieve;

        cudaMalloc((void **)&deviceQuery, sizeof(uint64_t) * NumOfQuery);
        cudaMalloc((void **)&deviceRetrieve, sizeof(uint32_t) * NumOfQuery);

        cudaMemcpy(deviceQuery, query, sizeof(uint64_t) * NumOfQuery,
                   cudaMemcpyHostToDevice);

        retrieve<<<ComputeGridDim(NumOfQuery), kBlockSize>>>(
            deviceKeys, deviceFlag, deviceValues, NumOfEntry,
            NumOfHashFunctions, deviceHashConstants, deviceHashStashConstants,
            tableSize, stashSize, deviceQuery, deviceRetrieve, NumOfQuery);
        cudaMemcpy(Retrieve, deviceRetrieve, sizeof(uint32_t) * NumOfQuery,
                   cudaMemcpyDeviceToHost);

        cudaFree(deviceQuery);
        cudaFree(deviceRetrieve);
        return 0;
}

int HashTest()
{
        float space_usage;
        uint32_t NumOfHashFucntions;
        uint32_t NumOfEntry;
        uint64_t *keys;
        uint32_t *values;
        uint32_t *retrieve;

        space_usage = 1.25f;
        NumOfHashFucntions = 5;
        NumOfEntry = 65536 * 2048 + 97;
        // CUrandom(NumOfEntry,values,0x7FFFFFFF-53);

        keys = new uint64_t[NumOfEntry];
        values = new uint32_t[NumOfEntry];
        retrieve = new uint32_t[NumOfEntry];

        for (uint32_t i = 0; i < NumOfEntry; ++i) {
                values[i] = i;
                keys[i] = crc64(0ULL, (unsigned char *)&values[i], 4);
        }

        scalaGPUHash A(space_usage, NumOfHashFucntions, NumOfEntry, keys,
                       values);
        A.hashBuild();
        A.hashClear();
        A.hashRetrieve(keys, NumOfEntry, retrieve);

        float ErrorSum = 0.0f;
        for (uint32_t i = 0; i < NumOfEntry; ++i) {
                uint32_t Error =
                    (values[i] - retrieve[i]) * (values[i] - retrieve[i]);
                ErrorSum += Error;
                if (values[i] != retrieve[i]) {
                        printf("the %d th query is not equal,%llu\t%d\t%d\n", i,
                               keys[i], values[i], retrieve[i]);
                }
        }
        printf("Total Sum = %f\n", ErrorSum);
        A.hashRelease();
        return 0;
}

int main() { HashTest(); }
