#pragma once
#ifndef _SPH_SM_M_KERNEL_H_
#define _SPH_SM_M_KERNEL_H_

#include <m3Vector.h>
#include <m3Bounds.h>
#include <m3Real.h>
#include <m3Matrix.h>
#include <m9Matrix.h>

#include <Particle.h>

#include <vector>
#include <map>
#include <chrono>

#include <helper_cuda.h>

#define PI 3.141592f
#define INF 1E-12f

// extern "C"
// {
// Kernel function
__device__ m3Real kernel = 0.035f;

__device__ m3Real K = 0.8f;
__device__ m3Real Stand_Density = 5000.0f;

__device__ m3Real voltage_constant = 50;
__device__ m3Real max_pressure = 100000;
__device__ m3Real max_voltage = 500;

__device__ m3Real velocity_mixing = 1.0f;

__device__ m3Real Wall_Hit = -0.05f;
__device__ m3Real mu = 100.0f;

__device__ m3Real Cm = 1.0;
__device__ m3Real Beta = 140;
__device__ m3Real sigma = 1.0f;
__device__ m3Real stim_strength = 500000.0f;

//membrane model parameters
__device__ m3Real FH_Vt=-75.0;
__device__ m3Real FH_Vp=15.0;
__device__ m3Real FH_Vr=-85.0;
__device__ m3Real C1 = 0.175;
__device__ m3Real C2 = 0.03;
__device__ m3Real C3 = 0.011;
__device__ m3Real C4 = 0.55;

/// For density computation
__device__ m3Real Poly6(m3Real Poly6_constant, m3Real r2)
{
	m3Real pow_value = 0.035f * 0.035f - r2;
	if(r2 >= 0 && r2 <= 0.035f*0.035f)
		return Poly6_constant * pow_value * pow_value * pow_value;
	else 
		return 0.0f;
}

/// For force of pressure computation
__device__ float Spiky(m3Real Spiky_constant, float r)
{
	if(r >= 0 && r <= kernel)
		return -Spiky_constant * (kernel - r) * (kernel - r) ;
	else
		return 0.0f;
}

/// For viscosity computation
__device__ float Visco(m3Real Spiky_constant, float r)
{
	if(r >= 0 && r <= kernel )
		return Spiky_constant * (kernel - r);
	else
		return 0;
}

__device__ m3Real B_spline(m3Real B_spline_constant, m3Real r)
{
	m3Real q = r / kernel;

	if (q >= 0 && q < 1)
		return B_spline_constant * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (0.25*pow(2 - q, 3));
	else
		return 0;
}

__device__ m3Real B_spline_1(m3Real B_spline_constant, m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3.0f * q + 2.25f * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (-0.75 * pow(2 - q, 2));
	else
		return 0;
}

__device__ m3Real B_spline_2(m3Real B_spline_constant, m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3 + 4.5 * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (1.5 * (2 - q));
	else
		return 0;
}

__device__ m3Vector Calculate_Cell_Position(m3Vector pos, m3Real Cell_Size)
{
	m3Vector cellpos = pos / Cell_Size;
	cellpos.x = (int)cellpos.x;
	cellpos.y = (int)cellpos.y;
	cellpos.z = (int)cellpos.z;
	return cellpos;
}

__device__ int Calculate_Cell_Hash(m3Vector pos, m3Vector Grid_Size)
{
	if((pos.x < 0)||(pos.x >= Grid_Size.x)||(pos.y < 0)||(pos.y >= Grid_Size.y)||
	(pos.z < 0)||(pos.z >= Grid_Size.z))
		return -1;

	// pos.x = pos.x & (Grid_Size.x - 1); // Size must be power of 2
	// pos.y = pos.y & (Grid_Size.y - 1); // Size must be power of 2
	// pos.z = pos.z & (Grid_Size.z - 1); // Size must be power of 2

	return  pos.x + Grid_Size.x * (pos.y + Grid_Size.y * pos.z);;
}

__global__
void calcHashD(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, m3Real Cell_Size, m3Vector Grid_Size, int numberParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= numberParticles) return;

	m3Vector p = pos[index];
	int hash = Calculate_Cell_Hash(Calculate_Cell_Position(p, Cell_Size), Grid_Size);

	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

__global__ 
void reorderDataAndFindCellStartD(
	m3Vector *sortedPos_d, m3Vector *pos_d,
	m3Vector *sortedVel_d, m3Vector *vel_d,
	m3Vector *sorted_corr_vel_d, m3Vector *corrected_vel_d,
	m3Vector *sortedAcc_d, m3Vector *acc_d,
	m3Real *sortedMass_d, m3Real *mass_d,
	bool *sorted_mFixed_d, bool *mFixed_d,
	m3Real *sorted_dens_d, m3Real *dens_d,
	m3Real *sorted_pres_d, m3Real *pres_d,
	m3Real *sorted_Vm_d, m3Real *Vm_d,
	m3Real *sorted_Inter_Vm_d, m3Real *Inter_Vm_d,
	m3Real *sorted_Iion_d, m3Real *Iion_d,
	m3Real *sorted_stim_d, m3Real *stim_d,
	m3Real *sorted_w_d, m3Real *w_d,
	uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex, uint numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;

	// handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

	__syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
            // cellEnd[hash] = index;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        
		sortedPos_d[index] = pos_d[sortedIndex];
		sortedVel_d[index] = vel_d[sortedIndex];
		// sorted_pred_vel_d[index] = predicted_vel_d[sortedIndex];
		// sorted_int_vel_d[index] = inter_vel_d[sortedIndex];
		sorted_corr_vel_d[index] = corrected_vel_d[sortedIndex];
		sortedAcc_d[index] = acc_d[sortedIndex];
		sortedMass_d[index] = mass_d[sortedIndex];
		// sorted_mOriginalPos_d[index] = mOriginalPos_d[sortedIndex];
		// sorted_mGoalPos_d[index] = mGoalPos_d[sortedIndex];
		sorted_mFixed_d[index] = mFixed_d[sortedIndex];
		sorted_dens_d[index] = dens_d[sortedIndex];
		sorted_pres_d[index] = pres_d[sortedIndex];

		sorted_Vm_d[index] = Vm_d[sortedIndex];
		sorted_Inter_Vm_d[index] = Inter_Vm_d[sortedIndex];
		sorted_Iion_d[index] = Iion_d[sortedIndex];
		sorted_stim_d[index] = stim_d[sortedIndex];
		sorted_w_d[index] = w_d[sortedIndex];
	}
}

__global__ void Compute_Density_SingPressureD(
	m3Vector *sortedPos_d,
	m3Real *sorted_dens_d,
	m3Real *sorted_pres_d,
	m3Real *sortedMass_d,
	m3Real *sorted_Vm_d,
	uint *m_dGridParticleIndex, uint *m_dCellStart, uint *m_dCellEnd, int m_numParticles, int m_numGridCells, m3Real Cell_Size, m3Vector Grid_Size, m3Real Poly6_constant)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos, NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		sorted_dens_d[index] = 0.f;
		sorted_pres_d[index] = 0.f;
		
		CellPos = Calculate_Cell_Position(sortedPos_d[index], Cell_Size);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos, Grid_Size);
			
			uint startIndex = m_dCellStart[hash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = m_dCellEnd[hash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = sortedPos_d[index] - sortedPos_d[j];
						
						m3Real dis2 = Distance.x * Distance.x + Distance.y * Distance.y + Distance.z * Distance.z;
						// p->dens += np->mass * B_spline(Distance.magnitude());

						sorted_dens_d[index] += sortedMass_d[j] * Poly6(Poly6_constant, dis2);
					}
				}
			}
			/// Calculates the density, Eq.3

		}

		// sorted_dens_d[index] += sortedMass_d[index] * Poly6(Poly6_constant, 0.0f);

		/// Calculates the pressure, Eq.12
		sorted_pres_d[index] = K * (sorted_dens_d[index]  - Stand_Density);
		
		sorted_pres_d[index] -= sorted_Vm_d[index] * voltage_constant;

		if(sorted_pres_d[index] < -max_pressure)
			sorted_pres_d[index] = -max_pressure;
		else if(sorted_pres_d[index] > max_pressure)
			sorted_pres_d[index] = max_pressure;
	// }
}

__global__ void Compute_ForceD(Particles *particles, uint *m_dGridParticleIndex, uint *m_dCellStart, uint *m_dCellEnd, int m_numParticles, int m_numGridCells, m3Real Cell_Size, m3Vector Grid_Size, m3Real Spiky_constant, m3Real B_spline_constant, m3Real Time_Delta)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		// p = &Particles[k];

		particles->sortedAcc_d[index] = m3Vector(0.0f, 0.0f, 0.0f);
		particles->sorted_Inter_Vm_d[index] = 0.0f;

		CellPos = Calculate_Cell_Position(particles->sortedPos_d[index], Cell_Size);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos, Grid_Size);

			uint startIndex = m_dCellStart[hash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = m_dCellEnd[hash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = particles->sortedPos_d[index] - particles->sortedPos_d[j];
						float dis2 = (float)Distance.magnitudeSquared();

						if(dis2 > INF)
						{
							float dis = sqrt(dis2);

							/// Calculates the force of pressure, Eq.10
							float Volume = particles->sortedMass_d[j] / particles->sorted_dens_d[j];
							// float Force_pressure = Volume * (p->pres+np->pres)/2 * B_spline_1(dis);
							float Force_pressure = Volume * (particles->sorted_pres_d[index] + particles->sorted_pres_d[j])/2 * Spiky(Spiky_constant, dis);

							particles->sortedAcc_d[index] -= Distance * Force_pressure / dis;

							/// Calculates the relative velocity (vj - vi), and then multiplies it to the mu, volume, and viscosity kernel. Eq.14
							// m3Vector RelativeVel = np->corrected_vel - p->corrected_vel;

							m3Vector RelativeVel = particles->sorted_int_vel_d[j] - particles->sorted_int_vel_d[index];
							float Force_viscosity = Volume * mu * Visco(Spiky_constant, dis);
							particles->sortedAcc_d[index] += RelativeVel * Force_viscosity;

							/// Calculates the intermediate voltage needed for the monodomain model
							particles->sorted_Inter_Vm_d[index] += (particles->sorted_Vm_d[j] - particles->sorted_Vm_d[index]) * Volume * B_spline_2(B_spline_constant, dis);
						}
					}
				}
			}
		}

		/// Sum of the forces that make up the fluid, Eq.8

		particles->sortedAcc_d[index] = particles->sortedAcc_d[index] / particles->sorted_dens_d[index];

		/// Adding the currents, and time integration for the intermediate voltage
		particles->sorted_Inter_Vm_d[index] += (sigma / (Beta*Cm)) + particles->sorted_Inter_Vm_d[index] - ((particles->sorted_Iion_d[index]- particles->sorted_stim_d[index] * Time_Delta / particles->sortedMass_d[index]) / Cm);

		uint originalIndex = m_dGridParticleIndex[index];

		particles->pos_d[originalIndex] = particles->sortedPos_d[index];
		particles->vel_d[originalIndex] = particles->sortedVel_d[index];
		particles->predicted_vel_d[originalIndex] = particles->sorted_pred_vel_d[index];
		particles->inter_vel_d[originalIndex] = particles->sorted_int_vel_d[index];
		particles->corrected_vel_d[originalIndex] = particles->sorted_corr_vel_d[index];
		particles->acc_d[originalIndex] = particles->sortedAcc_d[index];
		particles->mass_d[originalIndex] = particles->sortedMass_d[index];
		particles->mOriginalPos_d[originalIndex] = particles->sorted_mOriginalPos_d[index];
		particles->mGoalPos_d[originalIndex] = particles->sorted_mGoalPos_d[index];
		particles->mFixed_d[originalIndex] = particles->sorted_mFixed_d[index];
		particles->dens_d[originalIndex] = particles->sorted_dens_d[index];
		particles->pres_d[originalIndex] = particles->sorted_pres_d[index];
		particles->Vm_d[originalIndex] = particles->sorted_Vm_d[index];
		particles->Inter_Vm_d[originalIndex] = particles->sorted_Inter_Vm_d[index];
		particles->Iion_d[originalIndex] = particles->sorted_Iion_d[index];
		particles->stim_d[originalIndex] = particles->sorted_stim_d[index];
		particles->w_d[originalIndex] = particles->sorted_w_d[index];
	// }
}

__global__ void calculate_cell_modelD(m3Real *Iion_d, m3Real *w_d, m3Real *mass_d, m3Real *Vm_d, m3Real Time_Delta)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Real denom = (FH_Vp - FH_Vr);
	m3Real asd = (FH_Vt - FH_Vr)/denom;
	m3Real u = 0.0;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		// p = &Particles[k];

		u = (Vm_d[index] - FH_Vr) / denom;

		Iion_d[index] += Time_Delta * (C1*u*(u - asd)*(u - 1.0) + C2* w_d[index]) / mass_d[index];
		
		w_d[index] += Time_Delta * C3*(u - C4*w_d[index]) / mass_d[index];
	// }
}


/// Time integration as in 2016 - Fluid simulation by the SPH Method, a survey.
/// Eq.13 and Eq.14
__global__ void Update_PropertiesD(Particles *particles, m3Bounds bounds, m3Vector World_Size, m3Real Time_Delta)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	// for(int i=0; i < Number_Particles; i++)
	// {
		// p = &Particles[i];

		if(!particles->mFixed_d[index])
		{
			particles->vel_d[index] = particles->inter_vel_d[index] + (particles->acc_d[index]*Time_Delta/particles->mass_d[index]);
			particles->pos_d[index] = particles->pos_d[index] + (particles->vel_d[index]*Time_Delta);
		}

		particles->Vm_d[index] += particles->Inter_Vm_d[index] * Time_Delta / particles->mass_d[index];
		if(particles->Vm_d[index] > max_voltage)
			particles->Vm_d[index] = max_voltage;
		else if(particles->Vm_d[index] < -max_voltage)
			particles->Vm_d[index] = -max_voltage;

		if(particles->pos_d[index].x < 0.0f)
		{
			particles->vel_d[index].x = particles->vel_d[index].x * Wall_Hit;
			particles->pos_d[index].x = 0.0f;
		}
		if(particles->pos_d[index].x >= World_Size.x)
		{
			particles->vel_d[index].x = particles->vel_d[index].x * Wall_Hit;
			particles->pos_d[index].x = World_Size.x - 0.0001f;
		}
		if(particles->pos_d[index].y < 0.0f)
		{
			particles->vel_d[index].y = particles->vel_d[index].y * Wall_Hit;
			particles->pos_d[index].y = 0.0f;
		}
		if(particles->pos_d[index].y >= World_Size.y)
		{
			particles->vel_d[index].y = particles->vel_d[index].y * Wall_Hit;
			particles->pos_d[index].y = World_Size.y - 0.0001f;
		}
		if(particles->pos_d[index].z < 0.0f)
		{
			particles->vel_d[index].z = particles->vel_d[index].z * Wall_Hit;
			particles->pos_d[index].z = 0.0f;
		}
		if(particles->pos_d[index].z >= World_Size.z)
		{
			particles->vel_d[index].z = particles->vel_d[index].z * Wall_Hit;
			particles->pos_d[index].z = World_Size.z - 0.0001f;
		}

		bounds.clamp(particles->pos_d[index]);
	// }
}

__global__ void calculate_intermediate_velocityD(
	m3Vector *sortedPos_d,
	m3Vector *sorted_corr_vel_d,
	m3Real *sortedMass_d,
	m3Real *sorted_dens_d,
	m3Vector *sorted_int_vel_d,
	uint *m_dGridParticleIndex, uint *m_dCellStart, uint *m_dCellEnd, int m_numParticles, int m_numGridCells, m3Real Cell_Size, m3Vector Grid_Size, m3Real Poly6_constant)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		// p = &Particles[k];
		CellPos = Calculate_Cell_Position(sortedPos_d[index], Cell_Size);
		m3Vector partial_velocity(0.0f, 0.0f, 0.0f);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos, Grid_Size);
			
			uint startIndex = m_dCellStart[hash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = m_dCellEnd[hash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = sortedPos_d[index] - sortedPos_d[j];
						m3Real dis2 = Distance.x * Distance.x + Distance.y * Distance.y + Distance.z * Distance.z;
						// partial_velocity += (sorted_corr_vel_d[j] - sorted_corr_vel_d[index]) * Poly6(Poly6_constant, dis2) * (sortedMass_d[j] / sorted_dens_d[j]);
						partial_velocity += sorted_corr_vel_d[0];
					}
				}
			}
		}
		sorted_int_vel_d[index] = sorted_corr_vel_d[index] + partial_velocity * velocity_mixing;
	// }
}
// }
#endif