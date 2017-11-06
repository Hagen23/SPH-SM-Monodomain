#ifndef __SPH_SM_monodomain_CPP__
#define __SPH_SM_monodomain_CPP__

#include <SPH_SM_monodomain.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

SPH_SM_monodomain::SPH_SM_monodomain()
{
	kernel = 0.035f;

	Max_Number_Paticles = 50000;
	total_time_steps = 0;
	Number_Particles = 0;

	particles = new Particles();
	
	Cm = 1.0;
    Beta = 140;
	isStimOn = false;
	sigma = 1.0f;
	stim_strength = 500000.0f;
	
	World_Size = m3Vector(1.5f, 1.5f, 1.5f);

	Cell_Size = 0.04;
	Grid_Size = World_Size / Cell_Size;
	Grid_Size.x = (int)Grid_Size.x;
	Grid_Size.y = (int)Grid_Size.y;
	Grid_Size.z = (int)Grid_Size.z;

	Number_Cells = (int)Grid_Size.x * (int)Grid_Size.y * (int)Grid_Size.z;

	Gravity.set(0.0f, -9.8f, 0.0f);
	K = 0.8f;
	Stand_Density = 5000.0f;
	max_vel = m3Vector(3.0f, 3.0f, 3.0f);
	velocity_mixing = 1.0f;

	/// Time step is calculated as in 2016 - Divergence-Free SPH for Incompressible and Viscous Fluids.
	/// Then we adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition [6] ∆t ≤ 0.4 * d / (||vmax||)
	Time_Delta = 0.4 * kernel / sqrt(max_vel.magnitudeSquared());
	Wall_Hit = -0.05f;
	mu = 100.0f;

	Poly6_constant = 315.0f/(64.0f * m3Pi * pow(kernel, 9));
	Spiky_constant = 45.0f/(m3Pi * pow(kernel, 6));

	B_spline_constant = 1.0f / (m3Pi*kernel*kernel*kernel);

	// SM initializations
	bounds.min.zero();
	bounds.max.set(1.5f, 1.5f, 1.5f);

	alpha = 0.5f;
	beta = 0.2f;

	quadraticMatch = false;
	volumeConservation = true;
	allowFlip = true;

	cout<<"SPHSystem"<<endl;
	cout<<"Grid_Size_X : " << Grid_Size.x << endl;
	cout<<"Grid_Size_Y : " << Grid_Size.y << endl;
	cout<<"Grid_Size_Z : " << Grid_Size.y << endl;
	cout<<"Alpha :" << alpha << " Beta :" << beta << endl;
	cout<<"Volume conservation :" << volumeConservation << " Quadratic match : "<<quadraticMatch << endl;
	cout<<"Cell Number : "<<Number_Cells<<endl;
	cout<<"Time Delta : "<<Time_Delta<<endl;
}

SPH_SM_monodomain::~SPH_SM_monodomain()
{
	delete[] particles;
}

void SPH_SM_monodomain::add_viscosity(float value)
{
	mu += (mu + value) >= 0 ? value : 0;
	cout << "Viscosity: " << mu  << endl;
}

void SPH_SM_monodomain::Init_Fluid(vector<m3Vector> positions)
{
	Number_Particles = positions.size();

	particles->init_particles(positions, Stand_Density, Number_Particles);
}

// void SPH_SM_monodomain::Init_Particle(m3Vector pos, m3Vector vel)
// {
// 	if(Number_Particles + 1 > Max_Number_Paticles)
// 		return;

// 	Particles *p = &(Particles[Number_Particles]);
	
// 	p->pos = pos;
// 	p->mOriginalPos = pos;
// 	p->mGoalPos = pos;
// 	p->mFixed = false;

// 	p->vel = vel;
// 	p->acc = m3Vector(0.0f, 0.0f, 0.0f);
// 	p->dens = Stand_Density;
// 	p->mass = 0.2f;

// 	p->Inter_Vm = 0.0f;
// 	p->Vm = 0.0f;
// 	p->Iion = 0.0f;
// 	p->stim = 0.0f;
// 	p->w = 0.0f;

// 	Number_Particles++;
// }

__device__ m3Vector SPH_SM_monodomain::Calculate_Cell_Position(m3Vector pos)
{
	m3Vector cellpos = pos / Cell_Size;
	cellpos.x = (int)cellpos.x;
	cellpos.y = (int)cellpos.y;
	cellpos.z = (int)cellpos.z;
	return cellpos;
}

__device__ int SPH_SM_monodomain::Calculate_Cell_Hash(m3Vector pos)
{
	pos.x = pos.x & (Grid_Size.x - 1); // Size must be power of 2
	pos.y = pos.y & (Grid_Size.y - 1); // Size must be power of 2
	pos.z = pos.z & (Grid_Size.z - 1); // Size must be power of 2

	return  pos.x + Grid_Size.x * (pos.y + Grid_Size.y * pos.z);;
}

/// For density computation
__device__ float SPH_SM_monodomain::Poly6(float r2)
{
	return (r2 >= 0 && r2 <= kernel*kernel) ? Poly6_constant * pow(kernel * kernel - r2, 3) : 0;
}

/// For force of pressure computation
__device__ float SPH_SM_monodomain::Spiky(float r)
{
	return (r >= 0 && r <= kernel ) ? -Spiky_constant * (kernel - r) * (kernel - r) : 0;
}

/// For viscosity computation
__device__ float SPH_SM_monodomain::Visco(float r)
{
	return (r >= 0 && r <= kernel ) ? Spiky_constant * (kernel - r) : 0;
}

__device__ m3Real SPH_SM_monodomain::B_spline(m3Real r)
{
	m3Real q = r / kernel;

	if (q >= 0 && q < 1)
		return B_spline_constant * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (0.25*pow(2 - q, 3));
	else
		return 0;
}

__device__ m3Real SPH_SM_monodomain::B_spline_1(m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3.0f * q + 2.25f * q * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (-0.75 * pow(2 - q, 2));
	else
		return 0;
}

__device__ m3Real SPH_SM_monodomain::B_spline_2(m3Real r)
{
	m3Real q = r / kernel;
	if (q >= 0 && q < 1)
		return B_spline_constant * (-3 + 4.5 * q);
	else if (q >= 1 && q < 2)
		return B_spline_constant * (1.5 * (2 - q));
	else
		return 0;
}

__global__ void SPH_SM_monodomain::calcHashD(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= numberParticles) return;

	volatile m3Vector p = pos[index];
	int hash = Calculate_Cell_Hash(Calculate_Cell_Position(p));

	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

void SPH_SM_monodomain::calcHash(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numberParticles, 256, numBlocks, numThreads);

	calcHashD<<< numBlocks, numThreads >>>(gridParticleHash, gridParticleIndex, pos, numberParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

void SPH_SM_monodomain::sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, int numberParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash), thrust::device_ptr<uint>(dGridParticleHash + numParticles, thrust::device_ptr<uint>(dGridParticleIndex));
}

__global__ void SPH_SM_monodomain::reorderDataAndFindCellStartD(Particles *p, uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex, uint numParticles)
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
        
		p->sortedPos_d[index] = p->pos_d[sortedIndex];
		p->sortedVel_d[index] = p->vel_d[sortedIndex];
		p->sorted_pred_vel_d[index] = p->predicted_vel_d[sortedIndex];
		p->sorted_int_vel_d[index] = p->inter_vel_d[sortedIndex];
		p->sorted_corr_vel_d[index] = p->corrected_vel_d[sortedIndex];
		p->sortedAcc_d[index] = p->acc_d[sortedIndex];
		p->sortedMass_d[index] = p->mass_d[sortedIndex];
		p->sorted_mOriginalPos_d[index] = p->mOriginalPos_d[sortedIndex];
		p->sorted_mGoalPos_d[index] = p->mGoalPos_d[sortedIndex];
		p->sorted_mFixed_d[index] = p->mFixed_d[sortedIndex];
		p->sorted_dens_d[index] = p->dens_d[sortedIndex];
		p->sorted_pres_d[index] = p->pres_d[sortedIndex];
		p->sorted_Vm_d[index] = p->Vm_d[sortedIndex];
		p->sorted_Inter_Vm_d[index] = p->Inter_Vm_d[sortedIndex];
		p->sorted_Iion_d[index] = p->Iion_d[sortedIndex];
		p->sorted_stim_d[index] = p->stim_d[sortedIndex];
		p->sorted_w_d[index] = p->w_d[sortedIndex];
	}
}

void SPH_SM_monodomain::reorderDataAndFindCellStart(Particles *p, uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex uint numParticles, uint numCells)
{
	uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

	// set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(p, cellStart, cellEnd, gridParticleHash, gridParticleIndex,     numParticles);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void SPH_SM_monodomain::Find_neighbors()
{
	int hash;
	Particle *p;

	for(int i = 0; i < Number_Cells; i++)
		Cells[i].contained_particles.clear();

	for(int i = 0; i < Number_Particles; i ++)
	{
		p = &Particles[i];
		hash = Calculate_Cell_Hash(Calculate_Cell_Position(p->pos));
		Cells[hash].contained_particles.push_back(p);
	}
}

void SPH_SM_monodomain::apply_external_forces(Particles *particles, m3Vector* forcesArray = NULL, int* indexArray = NULL, int size = 0)
{
	//// External forces
	for (int i = 0; i < size; i++)
	{
		int j = indexArray[i];
		if (Particles[j].mFixed) continue;
		Particles[j].predicted_vel += (forcesArray[i] * Time_Delta) / Particles[j].mass;
	}

	//// Gravity
	for (int i = 0; i < Number_Particles; i++)
	{
		if (particles->mFixed_d[i]) continue;
		particles->predicted_vel_d[i] = particles->vel_d[i] + (Gravity * Time_Delta) / particles->mass_d[i];
		Particles[i].mGoalPos = Particles[i].mOriginalPos;
	}
}

void SPH_SM_monodomain::projectPositions()
{
	if (Number_Particles <= 1) return;
	int i, j, k;

	// center of mass
	m3Vector cm, originalCm;
	cm.zero(); originalCm.zero();
	float mass = 0.0f;

	for (i = 0; i < Number_Particles; i++)
	{
		m3Real m = particles->mass[index];
		if (particles->mFixed[index]) m *= 100.0f;
		mass += m;
		cm += particles->pos[i] * m;
		originalCm += particles->mOriginalPos[i] * m;
	}

	cm /= mass;
	originalCm /= mass;

	m3Vector p, q;

	m3Matrix Apq, Aqq;

	Apq.zero();
	Aqq.zero();

	for (i = 0; i < Number_Particles; i++)
	{
		p = particles->pos[index] - cm; 
		q = particles->mOriginalPos[index] - originalCm;
		m3Real m = particles->mass[i];

		Apq.r00 += m * p.x * q.x;
		Apq.r01 += m * p.x * q.y;
		Apq.r02 += m * p.x * q.z;

		Apq.r10 += m * p.y * q.x;
		Apq.r11 += m * p.y * q.y;
		Apq.r12 += m * p.y * q.z;

		Apq.r20 += m * p.z * q.x;
		Apq.r21 += m * p.z * q.y;
		Apq.r22 += m * p.z * q.z;

		Aqq.r00 += m * q.x * q.x;
		Aqq.r01 += m * q.x * q.y;
		Aqq.r02 += m * q.x * q.z;

		Aqq.r10 += m * q.y * q.x;
		Aqq.r11 += m * q.y * q.y;
		Aqq.r12 += m * q.y * q.z;

		Aqq.r20 += m * q.z * q.x;
		Aqq.r21 += m * q.z * q.y;
		Aqq.r22 += m * q.z * q.z;
	}

	if (!allowFlip && Apq.determinant() < 0.0f)
	{  	// prevent from flipping
		Apq.r01 = -Apq.r01;
		Apq.r11 = -Apq.r11;
		Apq.r22 = -Apq.r22;
	}

	m3Matrix R, S;
	m3Matrix::polarDecomposition(Apq, R, S);

	if (!quadraticMatch)
	{	// --------- linear match

		m3Matrix A = Aqq;
		A.invert();
		A.multiply(Apq, A);

		if (volumeConservation)
		{
			m3Real det = A.determinant();
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;
				A *= det;
			}
		}

		m3Matrix T = R * (1.0f - beta) + A * beta;

		for (i = 0; i < Number_Particles; i++)
		{
			if (particles->mFixed[index]) continue;
			q = particles->mOriginalPos[index] - originalCm;
			particles->mGoalPos[index] = T.multiply(q) + cm;
		}
	}
	else
	{	// -------------- quadratic match---------------------

		m3Real A9pq[3][9];

		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 9; j++)
			A9pq[i][j] = 0.0f;

		m9Matrix A9qq;
		A9qq.zero();

		for (int i = 0; i < Number_Particles; i++)
		{
			p = particles->pos[index] - cm;
			q = particles->mOriginalPos[index] - originalCm;

			m3Real q9[9];
			q9[0] = q.x; q9[1] = q.y; q9[2] = q.z; q9[3] = q.x*q.x; q9[4] = q.y*q.y; q9[5] = q.z*q.z;
			q9[6] = q.x*q.y; q9[7] = q.y*q.z; q9[8] = q.z*q.x;

			m3Real m = Particles[i].mass;
			A9pq[0][0] += m * p.x * q9[0];
			A9pq[0][1] += m * p.x * q9[1];
			A9pq[0][2] += m * p.x * q9[2];
			A9pq[0][3] += m * p.x * q9[3];
			A9pq[0][4] += m * p.x * q9[4];
			A9pq[0][5] += m * p.x * q9[5];
			A9pq[0][6] += m * p.x * q9[6];
			A9pq[0][7] += m * p.x * q9[7];
			A9pq[0][8] += m * p.x * q9[8];

			A9pq[1][0] += m * p.y * q9[0];
			A9pq[1][1] += m * p.y * q9[1];
			A9pq[1][2] += m * p.y * q9[2];
			A9pq[1][3] += m * p.y * q9[3];
			A9pq[1][4] += m * p.y * q9[4];
			A9pq[1][5] += m * p.y * q9[5];
			A9pq[1][6] += m * p.y * q9[6];
			A9pq[1][7] += m * p.y * q9[7];
			A9pq[1][8] += m * p.y * q9[8];

			A9pq[2][0] += m * p.z * q9[0];
			A9pq[2][1] += m * p.z * q9[1];
			A9pq[2][2] += m * p.z * q9[2];
			A9pq[2][3] += m * p.z * q9[3];
			A9pq[2][4] += m * p.z * q9[4];
			A9pq[2][5] += m * p.z * q9[5];
			A9pq[2][6] += m * p.z * q9[6];
			A9pq[2][7] += m * p.z * q9[7];
			A9pq[2][8] += m * p.z * q9[8];

			for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				A9qq(j, k) += m * q9[j] * q9[k];
		}

		A9qq.invert();

		m3Real A9[3][9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				A9[i][j] = 0.0f;
				for (k = 0; k < 9; k++)
					A9[i][j] += A9pq[i][k] * A9qq(k, j);

				A9[i][j] *= beta;
				if (j < 3)
					A9[i][j] += (1.0f - beta) * R(i, j);
			}
		}

		m3Real det =
			A9[0][0] * (A9[1][1] * A9[2][2] - A9[2][1] * A9[1][2]) -
			A9[0][1] * (A9[1][0] * A9[2][2] - A9[2][0] * A9[1][2]) +
			A9[0][2] * (A9[1][0] * A9[2][1] - A9[1][1] * A9[2][0]);

		if (!allowFlip && det < 0.0f) {         		// prevent from flipping
			A9[0][1] = -A9[0][1];
			A9[1][1] = -A9[1][1];
			A9[2][2] = -A9[2][2];
		}

		if (volumeConservation)
		{
			if (det != 0.0f)
			{
				det = 1.0f / sqrt(fabs(det));
				if (det > 2.0f) det = 2.0f;

				for (int i = 0; i < 3; i++)
				for (int j = 0; j < 9; j++)
					A9[i][j] *= det;
			}
		}

		for (int i = 0; i < Number_Particles; i++)
		{
			if (particles->mFixed[i]) continue;
			q = particles->mOriginalPos[i] - originalCm;

			particles->mGoalPos[i].x = A9[0][0] * q.x + A9[0][1] * q.y + A9[0][2] * q.z + A9[0][3] * q.x*q.x + A9[0][4] * q.y*q.y +
				A9[0][5] * q.z*q.z + A9[0][6] * q.x*q.y + A9[0][7] * q.y*q.z + A9[0][8] * q.z*q.x;

			particles->mGoalPos[i].y = A9[1][0] * q.x + A9[1][1] * q.y + A9[1][2] * q.z + A9[1][3] * q.x*q.x + A9[1][4] * q.y*q.y +
				A9[1][5] * q.z*q.z + A9[1][6] * q.x*q.y + A9[1][7] * q.y*q.z + A9[1][8] * q.z*q.x;

			particles->mGoalPos[i].z = A9[2][0] * q.x + A9[2][1] * q.y + A9[2][2] * q.z + A9[2][3] * q.x*q.x + A9[2][4] * q.y*q.y +
				A9[2][5] * q.z*q.z + A9[2][6] * q.x*q.y + A9[2][7] * q.y*q.z + A9[2][8] * q.z*q.x;

			particles->mGoalPos[i] += cm;
		}
	}
}

__device__ cell_Density_SingPressure()
{

}

__global__ void SPH_SM_monodomain::Compute_Density_SingPressureD(Particles *particles, uint *m_dGridParticleIndex, uint *m_dCellStart, uint *m_dCellEnd, uint *m_numParticles, uint *m_numGridCells)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos, NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		particles->dens_d[index] = 0.f;
		particles->pres_d[index] = 0.f;
		
		CellPos = Calculate_Cell_Position(particles->sortedPos_d[index]);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);
			
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = particles->sortedPos_d[index] - particles->sortedPos_d[j];
						
						float dis2 = (float)Distance.magnitudeSquared();
						// p->dens += np->mass * B_spline(Distance.magnitude());

						particles->sorted_dens_d[index] += particles->sortedMass_d[j] * Poly6(dis2);
					}
				}
			}
			/// Calculates the density, Eq.3

		}

		particles->sorted_dens_d[index] += particles->sortedMass_d[index] * Poly6(0.0f);

		/// Calculates the pressure, Eq.12
		particles->sorted_pres_d[index] = K * (particles->sorted_dens_d[index]  - Stand_Density);
		
		particles->sorted_pres_d[index] -= particles->sorted_Vm_d[index] * voltage_constant;

		if(particles->sorted_pres_d[index] < -max_pressure)
			particles->sorted_pres_d[index] = -max_pressure;
		else if(particles->sorted_pres_d[index] > max_pressure)
			particles->sorted_pres_d[index] = max_pressure;
	// }
}

__global__ void SPH_SM_monodomain::Compute_ForceD(Particles *particles, uint *m_dGridParticleIndex, uint *m_dCellStart, uint *m_dCellEnd, uint *m_numParticles, uint *m_numGridCells)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		p = &Particles[k];

		p->acc = m3Vector(0.0f, 0.0f, 0.0f);
		p->Inter_Vm = 0.0f;

		CellPos = Calculate_Cell_Position(p->pos);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);

			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

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
							float Force_pressure = Volume * (particles->sorted_pres_d[index] + particles->sorted_pres_d[j])/2 * Spiky(dis);

							particles->sortedAcc_d[index] -= Distance * Force_pressure / dis;

							/// Calculates the relative velocity (vj - vi), and then multiplies it to the mu, volume, and viscosity kernel. Eq.14
							// m3Vector RelativeVel = np->corrected_vel - p->corrected_vel;

							m3Vector RelativeVel = particles->sorted_int_vel_d[j] - particles->sorted_int_vel_d[index];
							float Force_viscosity = Volume * mu * Visco(dis);
							particles->sortedAcc_d[index] += RelativeVel * Force_viscosity;

							/// Calculates the intermediate voltage needed for the monodomain model
							particles->sorted_int_vel_d[index] += (particles->sorted_Vm_d[j] - particles->sorted_Vm_d[index]) * Volume * B_spline_2(dis);
						}
					}
				}
			}
		}
		/// Sum of the forces that make up the fluid, Eq.8

		particles->sortedAcc_d[index] = particles->sortedAcc_d[index] / particles->sorted_dens_d[index];

		/// Adding the currents, and time integration for the intermediate voltage
		particles->sorted_Inter_Vm_d[index] += (sigma / (Beta*Cm)) + particles->sorted_Inter_Vm_d[index] - ((particles->sorted_Iion_d[index]- particles->sorted_stim_d[index] * Time_Delta / particles->sortedMass_d[index]) / Cm);

		uint originalIndex = gridParticleIndex[index];
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
		particles->pres_d[originalIndex] = particles->sortedpres_d[index];
		particles->Vm_d[originalIndex] = particles->sorted_Vm_d[index];
		particles->Inter_Vm_d[originalIndex] = particles->sorted_Inter_Vm_d[index];
		particles->Iion_d[originalIndex] = particles->sorted_Iion_d[index];
		particles->stim_d[originalIndex] = particles->sorted_stim_d[index];
		particles->w_d[originalIndex] = particles->sorted_w_d[index];
	// }
}

__global__ void SPH_SM_monodomain::calculate_cell_model(Particles *particles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Real denom = (FH_Vp - FH_Vr);
	m3Real asd = (FH_Vt - FH_Vr)/denom;
	m3Real u = 0.0;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		p = &Particles[k];

		u = (particles->Vm_d[index] - FH_Vr) / denom;

		particles->Iion_d[index] += Time_Delta * (C1*u*(u - asd)*(u - 1.0) + C2* particles->w[index]) / particles->mass_d[index];
		
		particles->w_d[index] += Time_Delta * C3*(u - C4*particles->w_d[index]) / particles->mass_d[index];
	// }
}


/// Time integration as in 2016 - Fluid simulation by the SPH Method, a survey.
/// Eq.13 and Eq.14
__global__ void SPH_SM_monodomain::Update_Properties(Particles *particles)
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

		bounds.clamp(particls->pos_d[index]);
	}
}

void SPH_SM_monodomain::calculate_corrected_velocity(Particles *particles)
{

}

__global__ void SPH_SM_monodomain::calculate_corrected_velocityD(Particles *particles)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	/// Computes predicted velocity from forces except viscoelastic and pressure
	apply_external_forces();

	/// Calculates corrected velocity
	projectPositions();

	m3Real time_delta_1 = 1.0f / Time_Delta;

	// for (int i = 0; i < Number_Particles; i++)
	// {
		particles->corrected_vel_d[i] = particles->predicted_vel_d[i] + (particles->mGoalPos_d[i] - particles->pos_d[i]) * time_delta_1 * alpha;
	// }
}

void SPH_SM_monodomain::calculate_intermediate_velocity()
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	m3Vector CellPos;
	m3Vector NeighborPos;
	int hash;

	// for(int k = 0; k < Number_Particles; k++)
	// {
		p = &Particles[k];
		CellPos = Calculate_Cell_Position(particles->sortedPos_d[index]);
		m3Vector partial_velocity(0.0f, 0.0f, 0.0f);

		for(int k = -1; k <= 1; k++)
		for(int j = -1; j <= 1; j++)
		for(int i = -1; i <= 1; i++)
		{
			NeighborPos = CellPos + m3Vector(i, j, k);
			hash = Calculate_Cell_Hash(NeighborPos);
			uint startIndex = cellStart[gridHash];

			if (startIndex != 0xffffffff)
			{
				uint endIndex = cellEnd[gridHash];

				for(uint j = startIndex; j < endIndex; j++)
				{
					if(j != index)
					{
						m3Vector Distance;
						Distance = particles->sortedPos_d[index] - particles->sortedPos_d[j];
						float dis2 = (float)Distance.magnitudeSquared();
						partial_velocity += (particles->sorted_corr_vel_d[j] - particles->corrected_vel_d[index]) * Poly6(dis2) * (particles->mass_d[j] / particles->dens_d[j]);
					}
				}
			}
		}
		particles->sorted_int_vel_d[index] = particles->sorted_corr_vel_d[index] + partial_velocity * velocity_mixing;
	// }
}


void SPH_SM_monodomain::set_stim(Particles *particles, m3Vector center, m3Real radius, m3Real stim_strength)
{
	Particle *p;
	isStimOn = true;
	for(int k = 0; k < Number_Particles; k++)
	{
		m3Vector position = particles->pos[k];
		if (((position.x-center.x)*(position.x-center.x)+(position.y-center.y)*(position.y-center.y)+(position.z-center.z)*(position.z-center.z)) <= radius)
		{
			particles->stim[k] = stim_strength;
		}
	}
}

void SPH_SM_monodomain::turnOnStim_Cube(std::vector<m3Vector> positions)
{
	m3Vector cm;
	Particle *p;

	for(m3Vector pos : positions)
	{
		cm += pos;
		if( (pos.x >= 0.3 && pos.x <= 0.33) || (pos.x > 0.6839 && pos.z <= 0.7f) )
			set_stim(pos, 0.001f, stim_strength);
	}
	cm /= Number_Particles;
	// set_stim(m3Vector(0.3,0.0,0.7), 0.001f, stim_strength);
	// set_stim(cm, 0.001f, stim_strength);
	
	for(int k = 0; k < Number_Particles; k++)
	{
		p = &Particles[k];
		m3Vector position = p->pos;
		if ((position.y == 0.0f && position.x == 0.3f) || (position.y == 0.0f && position.x >= 0.68399f))
			p->mFixed = true;
	}

	cout<<"Particles stimulated."<<endl;
}

void SPH_SM_monodomain::turnOnStim_Mesh(Particles *p, std::vector<m3Vector> positions)
{
	m3Vector cm;

	for(m3Vector pos : positions)
	{
		if( (pos.x >= 0.3 && pos.x <= 0.36) || (pos.x >= 0.5 && pos.x <= 0.56) || (pos.x > 1.26 && pos.x <= 1.29f) )
			set_stim(pos, 0.001f, stim_strength);
	}
	for(int k = 0; k < Number_Particles; k++)
	{
		if ((p->pos[k].x >= 0.3 && p->pos[k].x <= 0.36) || (p->pos[k].x >= 1.27 && p->pos[k].x <= 1.29f))
			p->mFixed[k] = true;
	}
}

void SPH_SM_monodomain::turnOffStim(Particles *p)
{
	isStimOn = false;
	for(int k = 0; k < Number_Particles; k++)
	{
		if(p->stim[k] > 0.0f)
		{
			p->stim[k] = 0.0f;
		}
	}
}

void SPH_SM_monodomain::print_report(double avg_fps, double avg_step_d)
{
	cout << "Avg FPS ; Avg Step Duration ; Time Steps ; Find neighbors ; Corrected Velocity ; Intermediate Velocity ; Density-Pressure ; Cell model ; Compute Force ; Update Properties ; K ; Alpha ; Beta ; Mu ; sigma ; Stim strength ; FH_VT ; FH_VP ; FH_VR ; C1 ; C2 ; C3 ; C4" << endl;
	
	cout << avg_fps << ";" << avg_step_d << ";" << total_time_steps << ";" << d_find_neighbors.count() / total_time_steps << ";" << d_corrected_velocity.count() / total_time_steps << ";" << d_intermediate_velocity.count() / total_time_steps << ";" << d_Density_SingPressure.count() / total_time_steps << ";" << d_cell_model.count() / total_time_steps << ";" << d_compute_Force.count() / total_time_steps << ";" << d_Update_Properties.count() / total_time_steps << ";";

	cout << K << ";" << alpha << ";" << beta  << ";" << mu << ";" << sigma << ";" << stim_strength << ";" << FH_Vt << ";" << FH_Vp << ";" << FH_Vr << ";" << C1 << ";" << C2 << ";" << C3 << ";" << C4 << endl;
}

void SPH_SM_monodomain::compute_SPH_SM_monodomain()
{
	calcHash(dGridParticleHash, dGridParticleIndex, p.pos_d, Number_Particles);

	sortParticles(dGridParticleHash, dGridParticleIndex, Number_Particles);

	reorderDataAndFindCellStart(particles, cellStart, cellEnd, gridParticleHash, gridParticleIndex, numParticles, numCells);
	// tpoint tstart = std::chrono::system_clock::now();
	// Find_neighbors();
	// d_find_neighbors += std::chrono::system_clock::now() - tstart;

	// tstart = std::chrono::system_clock::now();
	// calculate_corrected_velocity();
	// d_corrected_velocity += std::chrono::system_clock::now() - tstart;
	
	// tstart = std::chrono::system_clock::now();
	// calculate_intermediate_velocity();
	// d_intermediate_velocity += std::chrono::system_clock::now() - tstart;
	
	// tstart = std::chrono::system_clock::now();
	// Compute_Density_SingPressure();
	// d_Density_SingPressure += std::chrono::system_clock::now() - tstart;
	
	// tstart = std::chrono::system_clock::now();
	// calculate_cell_model();
	// d_cell_model += std::chrono::system_clock::now() - tstart;

	// tstart = std::chrono::system_clock::now();
	// Compute_Force();
	// d_compute_Force += std::chrono::system_clock::now() - tstart;

	// tstart = std::chrono::system_clock::now();
	// Update_Properties();
	// d_Update_Properties += std::chrono::system_clock::now() - tstart;

	total_time_steps++;
}

void SPH_SM_monodomain::Animation()
{
	compute_SPH_SM_monodomain();
}


#endif