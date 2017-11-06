extern "C"
{
	/// Calculates the cell position for each particle
	void calcHash(uint *gridParticleHash, uint * gridParticleIndex, m3Vector *pos, int numberParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numberParticles, 256, numBlocks, numThreads);

		calcHashD<<< numBlocks, numThreads >>>(gridParticleHash, gridParticleIndex, pos, Cell_Size, Grid_Size, numberParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	/// Sorts the hashes and indices
	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numberParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash), 
			thrust::device_ptr<uint>(dGridParticleHash + numberParticles), 
			thrust::device_ptr<uint>(dGridParticleIndex));
	}

	/// Reorders the particles based on the hashes and indices. Also finds the start and end cells.
	void reorderDataAndFindCellStart(Particles *p, uint *cellStart, uint *cellEnd, uint *gridParticleHash, uint *gridParticleIndex, uint numParticles, uint numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

		uint smemSize = sizeof(uint)*(numThreads+1);
		reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(p, cellStart, cellEnd, gridParticleHash, gridParticleIndex,     numParticles);

		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
	}

	void calculate_cell_model()
	{
		uint numThreads, numBlocks;
		computeGridSize(Number_Particles, 256, numBlocks, numThreads);

		calculate_cell_modelD<<<numBlocks, numThreads>>>(particles, Time_Delta);

		getLastCudaError("Kernel execution failed: calculate_cell_model");
	}

	void calculate_intermediate_velocity()
	{
		uint numThreads, numBlocks;
		computeGridSize(Number_Particles, 256, numBlocks, numThreads);

		calculate_intermediate_velocityD<<<numBlocks, numThreads>>>(particles, dGridParticleIndex, dCellStart, dCellEnd, Number_Particles, Number_Cells, Cell_Size, Grid_Size, Poly6_constant);

		getLastCudaError("Kernel execution failed: calculate_intermediate_velocityD");
	}

	void compute_SPH_SM_monodomain()
	{
		calculate_corrected_velocity();

		unsigned int memSize = sizeof(m3Vector)*Number_Particles;

		checkCudaErrors(cudaMemcpy(particles->pos_d, particles->pos, memSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(particles->corrected_vel_d, particles->corrected_vel, memSize, cudaMemcpyHostToDevice));

		calculate_cell_model();

		calcHash(dGridParticleHash, dGridParticleIndex, particles->pos_d, Number_Particles);

		sortParticles(dGridParticleHash, dGridParticleIndex, Number_Particles);

		reorderDataAndFindCellStart(particles, dCellStart, dCellEnd, dGridParticleHash, dGridParticleIndex, Number_Particles, Number_Cells);

		calculate_intermediate_velocity();
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

	void Animation()
	{
		compute_SPH_SM_monodomain();
	}
}