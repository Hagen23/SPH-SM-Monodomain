#ifndef M2_BOUNDS_H
#define M2_BOUNDS_H

//---------------------------------------------------------------------------

#include "math3d.cuh"

//---------------------------------------------------------------------------
class m3Bounds
	//---------------------------------------------------------------------------
{
public:
	__host__ __device__ inline m3Bounds() { setEmpty(); }
	__host__ __device__ inline m3Bounds(const m3Vector &min0, const m3Vector &max0) { min = min0; max = max0; }

	__host__ __device__ inline void set(const m3Vector &min0, const m3Vector &max0) { min = min0; max = max0; }

	__host__ __device__ inline void setEmpty() {
		set(m3Vector(m3RealMax, m3RealMax, m3RealMax),
			m3Vector(m3RealMin, m3RealMin, m3RealMin));
	}
	
	__host__ __device__ inline void setInfinite() {
		set(m3Vector(m3RealMin, m3RealMin, m3RealMin),
			m3Vector(m3RealMax, m3RealMax, m3RealMax));
	}

	__host__ __device__ inline bool isEmpty() const {
		if (min.x > max.x) return true;
		if (min.y > max.y) return true;
		if (min.z > max.z) return true;
		return false;
	}

	bool operator == (const m3Bounds &b) const {
		return (min == b.min) && (max == b.max);
	}

	__host__ __device__ void combine(const m3Bounds &b) {
		min.minimum(b.min);
		max.maximum(b.max);
	}

	void operator += (const m3Bounds &b) {
		combine(b);
	}

	m3Bounds operator + (const m3Bounds &b) const {
		m3Bounds r = *this;
		r.combine(b);
		return r;
	}

	__host__ __device__ bool intersects(const m3Bounds &b) const {
		if ((b.min.x > max.x) || (min.x > b.max.x)) return false;
		if ((b.min.y > max.y) || (min.y > b.max.y)) return false;
		return true;
	}

	__host__ __device__ void intersect(const m3Bounds &b) {
		min.maximum(b.min);
		max.minimum(b.max);
	}

	__host__ __device__ void include(const m3Vector &v) {
		max.maximum(v);
		min.minimum(v);
	}

	__host__ __device__ bool contain(const m3Vector &v) const {
		return
			min.x <= v.x && v.x <= max.x &&
			min.y <= v.y && v.y <= max.y;
	}

	void operator += (const m3Vector &v) {
		include(v);
	}

	__host__ __device__ void getCenter(m3Vector &v) const {
		v = min + max; v *= 0.5f;
	}

	__host__ __device__ void clamp(m3Vector &pos) const {
		if (isEmpty()) return;
		pos.maximum(min);
		pos.minimum(max);
	}

	__host__ __device__ void clamp(m3Vector &pos, m3Real offset) const {
		if (isEmpty()) return;
		if (pos.x < min.x + offset) pos.x = min.x + offset;
		if (pos.x > max.x - offset) pos.x = max.x - offset;
		if (pos.y < min.y + offset) pos.y = min.y + offset;
		if (pos.y > max.y - offset) pos.y = max.y - offset;
	}
	//--------------------------------------------
	m3Vector min, max;
};


#endif