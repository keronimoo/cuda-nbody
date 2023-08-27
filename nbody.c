#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z, vx, vy, vz;
} Particle;

void randomizeParticles(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__
void calculateForces(Particle *particles, float dt, int n) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n; i += stride) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        particles[i].vx += dt * Fx;
        particles[i].vy += dt * Fy;
        particles[i].vz += dt * Fz;
    }
}

__global__
void integratePositions(Particle *particles, float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

int main(const int argc, const char **argv) {
    int numParticles = 2 << 11;
    if (argc > 1) numParticles = 2 << atoi(argv[1]);

    const char *initializedFile;
    const char *solutionFile;

    if (numParticles == 2 << 11) {
        initializedFile = "09-nbody/files/initialized_4096";
        solutionFile = "09-nbody/files/solution_4096";
    } else {
        initializedFile = "09-nbody/files/initialized_65536";
        solutionFile = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initializedFile = argv[2];
    if (argc > 3) solutionFile = argv[3];

    const float timeStep = 0.01f;
    const int numIterations = 10;

    int particleBytes = numParticles * sizeof(Particle);
    float *particleData;

    cudaMallocManaged(&particleData, particleBytes);

    Particle *particles = (Particle *)particleData;

    read_values_from_file(initializedFile, particleData, particleBytes);

    double totalTime = 0.0;

    int deviceId;
    int numSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);

    cudaMemPrefetchAsync(particleData, particleBytes, deviceId);

    for (int iter = 0; iter < numIterations; iter++) {
        StartTimer();

        int numThreads = 128;
        int numBlocks = 32 * numSMs;

        calculateForces<<<numBlocks, numThreads>>>(particles, timeStep, numParticles);
        cudaDeviceSynchronize();

        integratePositions<<<numBlocks, numThreads>>>(particles, timeStep, numParticles);
        cudaDeviceSynchronize();

        const double elapsedSeconds = GetTimer() / 1000.0;
        totalTime += elapsedSeconds;
    }

    double avgTime = totalTime / (double)(numIterations);
    float billionsOfOpsPerSecond = 1e-9 * numParticles * numParticles / avgTime;
    write_values_to_file(solutionFile, particleData, particleBytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
    cudaMemPrefetchAsync(particleData, particleBytes, cudaCpuDeviceId);

    cudaFree(particleData);

    return 0;
}
