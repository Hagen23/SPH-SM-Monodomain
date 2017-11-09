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
	m3Real		*mass_d, *sortedMass_d;

	m3Vector 	*mOriginalPos_d, *sorted_mOriginalPos_d;	// Original positions of the mesh points
	m3Vector 	*mGoalPos_d, *sorted_mGoalPos_d;			// Goal positions
	bool		*mFixed_d, *sorted_mFixed_d;				// Whether de particle is fixed in place

	m3Real 		*dens_d, *sorted_dens_d;			// density
	m3Real 		*pres_d, *sorted_pres_d;			// pressure

	m3Real    	*Vm_d, *sorted_Vm_d;	     		// Voltage
	m3Real	  	*Inter_Vm_d, *sorted_Inter_Vm_d;	// Intermediate Voltage for time integration
	m3Real    	*Iion_d, *sorted_Iion_d;			// Ionic current
	m3Real    	*stim_d, *sorted_stim_d;	   		// Stimulation
	m3Real   	*w_d,*sorted_w_d;		      		// Recovery variable

	Particles(){}

	void init_particles(std::vector<m3Vector> positions, float Stand_Density, int Number_Particles);
};

#endif
