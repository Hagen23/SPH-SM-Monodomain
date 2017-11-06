#pragma once
#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#include <m3Vector.h>
#include <vector>
#include <helper_cuda.h>
#include <cuda_runtime.h>

class Particles
{
public:

	/// Host data

	m3Vector 	*pos;			// Position of the particle
	m3Vector 	*vel;			// Velocity of the particle
	m3Vector	*predicted_vel;	// Predicted velocity: calculated with all forces but viscoelastic and pressure ones
	m3Vector	*inter_vel;		// Intermediate velocity of the particle
	m3Vector	*corrected_vel;	// Corrected velocity using SM
	m3Vector 	*acc;			// Acceleration of the particle
	float		*mass;

	m3Vector 	*mOriginalPos;	// Original positions of the mesh points
	m3Vector 	*mGoalPos;		// Goal positions
	bool		*mFixed;			// Whether de particle is fixed in place

	float 		*dens;			// density
	float 		*pres;			// pressure

	m3Real    	*Vm;     		// Voltage
	m3Real	  	*Inter_Vm;		// Intermediate Voltage for time integration
    m3Real    	*Iion;			// Ionic current
    m3Real    	*stim;   		// Stimulation
    m3Real   	*w;      		// Recovery variable

	/// GPU Data

	m3Vector	*pos_d, *sortedPos_d;
	m3Vector 	*vel_d, *sortedVel_d;					// Velocity of the particle
	m3Vector	*predicted_vel_d, *sorted_pred_vel_d;	// Predicted velocity: calculated with all forces but viscoelastic and pressure ones
	m3Vector	*inter_vel_d, *sorted_int_vel_d;		// Intermediate velocity of the particle
	m3Vector	*corrected_vel_d, *sorted_corr_vel_d;	// Corrected velocity using SM
	m3Vector 	*acc_d, *sortedAcc_d;					// Acceleration of the particle
	float		*mass_d, *sortedMass_d;

	m3Vector 	*mOriginalPos_d, *sorted_mOriginalPos_d;	// Original positions of the mesh points
	m3Vector 	*mGoalPos_d, *sorted_mGoalPos_d;			// Goal positions
	bool		*mFixed_d, *sorted_mFixed_d;				// Whether de particle is fixed in place

	float 		*dens_d, *sorted_dens_d;			// density
	float 		*pres_d, *sorted_pres_d;			// pressure

	m3Real    	*Vm_d, *sorted_Vm_d;	     		// Voltage
	m3Real	  	*Inter_Vm_d, *sorted_Inter_Vm_d;	// Intermediate Voltage for time integration
	m3Real    	*Iion_d, *sorted_Iion_d;			// Ionic current
	m3Real    	*stim_d, *sorted_stim_d;	   		// Stimulation
	m3Real   	*w_d,*sorted_w_d;		      		// Recovery variable

	Particles(){}

	void init_particles(std::vector<m3Vector> positions, float Stand_Density, int Number_Particles)
	{
		/// Allocate host storage
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
		checkCudaErrors(cudaMalloc((void**)&predicted_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_pred_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&inter_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_int_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&corrected_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_corr_vel_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&acc_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sortedAcc_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&mass_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sortedMass_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&mOriginalPos_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_mOriginalPos_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&mGoalPos_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_mGoalPos_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&mFixed_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_mFixed_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&dens_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_dens_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&pres_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_pres_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&Vm_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_Vm_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&Inter_Vm_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_Inter_Vm_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&Iion_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_Iion_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&stim_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_stim_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&w_d, memSize));
		checkCudaErrors(cudaMalloc((void**)&sorted_w_d, memSize));

		checkCudaErrors(cudaMemcpy(pos_d, pos, memSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mOriginalPos_d, pos, memSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mGoalPos_d, pos, memSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mFixed_d, pos, memSize, cudaMemcpyHostToDevice));
	}
};

#endif
