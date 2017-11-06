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


/// Calculates the cell position for each particle
void SPH_SM_monodomain::calcHash(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles)
{
	uint numThreads, numBlocks;
	computeGridSize(numberParticles, 256, numBlocks, numThreads);

	calcHashD<<< numBlocks, numThreads >>>(gridParticleHash, gridParticleIndex, pos, Cell_Size, Grid_Size, numberParticles);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

/// Sorts the hashes and indices
void SPH_SM_monodomain::sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numberParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash), 
		thrust::device_ptr<uint>(dGridParticleHash + numberParticles), 
		thrust::device_ptr<uint>(dGridParticleIndex));
}

/// Reorders the particles based on the hashes and indices. Also finds the start and end cells.
void SPH_SM_monodomain::reorderDataAndFindCellStart(Particles *p, uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex, uint numParticles, uint numCells)
{
	uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

	// set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(p, cellStart, cellEnd, gridParticleHash, gridParticleIndex,     numParticles);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

// void SPH_SM_monodomain::Find_neighbors()
// {
// 	int hash;
// 	Particle *p;

// 	for(int i = 0; i < Number_Cells; i++)
// 		Cells[i].contained_particles.clear();

// 	for(int i = 0; i < Number_Particles; i ++)
// 	{
// 		p = &Particles[i];
// 		hash = Calculate_Cell_Hash(Calculate_Cell_Position(pp->pos, Cell_Size));
// 		Cells[hash].contained_particles.push_back(p);
// 	}
// }

void SPH_SM_monodomain::apply_external_forces(m3Vector* forcesArray = NULL, int* indexArray = NULL, int size = 0)
{
	//// External forces
	// for (int i = 0; i < size; i++)
	// {
	// 	int j = indexArray[i];
	// 	if (particles->mFixed[i]) continue;
	// 	Particles[j].predicted_vel += (forcesArray[i] * Time_Delta) / Particles[j].mass;
	// }

	//// Gravity
	for (int i = 0; i < Number_Particles; i++)
	{
		if (particles->mFixed_d[i]) continue;
		particles->predicted_vel_d[i] = particles->vel_d[i] + (Gravity * Time_Delta) / particles->mass_d[i];
		particles->mGoalPos[i] = particles->mOriginalPos[i];
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
		m3Real m = particles->mass[i];
		if (particles->mFixed[i]) m *= 100.0f;
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
		p = particles->pos[i] - cm; 
		q = particles->mOriginalPos[i] - originalCm;
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
			if (particles->mFixed[i]) continue;
			q = particles->mOriginalPos[i] - originalCm;
			particles->mGoalPos[i] = T.multiply(q) + cm;
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
			p = particles->pos[i] - cm;
			q = particles->mOriginalPos[i] - originalCm;

			m3Real q9[9];
			q9[0] = q.x; q9[1] = q.y; q9[2] = q.z; q9[3] = q.x*q.x; q9[4] = q.y*q.y; q9[5] = q.z*q.z;
			q9[6] = q.x*q.y; q9[7] = q.y*q.z; q9[8] = q.z*q.x;

			m3Real m = particles->mass[i];
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

void SPH_SM_monodomain::Compute_Density_SingPressure()
{

}

void SPH_SM_monodomain::calculate_corrected_velocity(Particles *particles)
{
	// uint i = blockIdx.x * blockDim.x + threadIdx.x;
	/// Computes predicted velocity from forces except viscoelastic and pressure
	apply_external_forces();

	/// Calculates corrected velocity
	projectPositions();

	m3Real time_delta_1 = 1.0f / Time_Delta;

	for (int i = 0; i < Number_Particles; i++)
	{
		particles->corrected_vel_d[i] = particles->predicted_vel_d[i] + (particles->mGoalPos_d[i] - particles->pos_d[i]) * time_delta_1 * alpha;
	}
}

void SPH_SM_monodomain::set_stim(m3Vector center, m3Real radius, m3Real stim_strength)
{
	// Particle *p;
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
	// Particle *p;

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
		// p = &Particles[k];
		m3Vector position = particles->pos[k];
		if ((position.y == 0.0f && position.x == 0.3f) || (position.y == 0.0f && position.x >= 0.68399f))
			particles->mFixed[k] = true;
	}

	cout<<"Particles stimulated."<<endl;
}

void SPH_SM_monodomain::turnOnStim_Mesh(std::vector<m3Vector> positions)
{
	m3Vector cm;

	for(m3Vector pos : positions)
	{
		if( (pos.x >= 0.3 && pos.x <= 0.36) || (pos.x >= 0.5 && pos.x <= 0.56) || (pos.x > 1.26 && pos.x <= 1.29f) )
			set_stim(pos, 0.001f, stim_strength);
	}
	for(int k = 0; k < Number_Particles; k++)
	{
		if ((particles->pos[k].x >= 0.3 && particles->pos[k].x <= 0.36) || (particles->pos[k].x >= 1.27 && particles->pos[k].x <= 1.29f))
			particles->mFixed[k] = true;
	}
}

void SPH_SM_monodomain::turnOffStim()
{
	isStimOn = false;
	for(int k = 0; k < Number_Particles; k++)
	{
		if(particles->stim[k] > 0.0f)
		{
			particles->stim[k] = 0.0f;
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
	calculate_corrected_velocity();

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