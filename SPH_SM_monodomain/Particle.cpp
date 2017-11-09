#include <Particle.h>

void Particles::init_particles(std::vector<m3Vector> positions, float Stand_Density, int Number_Particles)
{
    /// Allocate host storagem
    pos = new m3Vector[Number_Particles]();
    vel = new m3Vector[Number_Particles]();
    predicted_vel = new m3Vector[Number_Particles]();
    inter_vel = new m3Vector[Number_Particles]();
    corrected_vel = new m3Vector[Number_Particles]();
    acc = new m3Vector[Number_Particles]();
    mass = new float[Number_Particles]();

    mOriginalPos = new m3Vector[Number_Particles]();
    mGoalPos = new m3Vector[Number_Particles]();
    mFixed = new bool[Number_Particles]();

    dens = new float[Number_Particles]();
    pres = new float[Number_Particles]();

    Vm = new m3Real[Number_Particles]();
    Inter_Vm = new m3Real[Number_Particles]();
    Iion = new m3Real[Number_Particles]();
    stim = new m3Real[Number_Particles]();
    w = new m3Real[Number_Particles]();

    for(int i = 0; i < Number_Particles; i++)
    {
        pos[i] = positions[i];
        mOriginalPos[i] = positions[i];
        mGoalPos[i] = positions[i];
        mFixed[i] = false;
        dens[i] = Stand_Density;
        mass[i] = 0.2f;
    }

    unsigned int memSize = sizeof(m3Vector)*Number_Particles;

    checkCudaErrors(cudaMalloc((void**)&pos_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedPos_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedVel_d, memSize));
    // checkCudaErrors(cudaMalloc((void**)&predicted_vel_d, memSize));
    // checkCudaErrors(cudaMalloc((void**)&sorted_pred_vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&inter_vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sorted_int_vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&corrected_vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sorted_corr_vel_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&acc_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&sortedAcc_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&mass_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sortedMass_d, sizeof(m3Real)*Number_Particles));
    // checkCudaErrors(cudaMalloc((void**)&mOriginalPos_d, memSize));
    // checkCudaErrors(cudaMalloc((void**)&sorted_mOriginalPos_d, memSize));
    // checkCudaErrors(cudaMalloc((void**)&mGoalPos_d, memSize));
    // checkCudaErrors(cudaMalloc((void**)&sorted_mGoalPos_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&mFixed_d, sizeof(bool) * Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_mFixed_d, memSize));
    checkCudaErrors(cudaMalloc((void**)&dens_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_dens_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&pres_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_pres_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&Vm_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_Vm_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&Inter_Vm_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_Inter_Vm_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&Iion_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_Iion_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&stim_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_stim_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&w_d, sizeof(m3Real)*Number_Particles));
    checkCudaErrors(cudaMalloc((void**)&sorted_w_d, sizeof(m3Real)*Number_Particles));

    checkCudaErrors(cudaMemcpy(pos_d, pos, memSize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(mOriginalPos_d, pos, memSize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(mGoalPos_d, pos, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mFixed_d, mFixed, sizeof(bool) * Number_Particles, cudaMemcpyHostToDevice));
}