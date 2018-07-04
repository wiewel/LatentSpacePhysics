/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2017 Moritz Becher, Steffen Wiewel
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * metric plugin
 *
 ******************************************************************************/

#include "grid.h"
#include "levelset.h"
#include "kernel.h"
#include "commonkernels.h"

#include <math.h>

using namespace std;

namespace Manta
{
    //----------------------------------------------------------------------------------------------------
    inline bool isSurfacePoint(const LevelsetGrid& _LS, const Vec3i& _Coord, Real& _fValue)
    {
        _fValue = _LS.get(_Coord);
        return (0.0 >= _fValue) && (_fValue >= -1.0);
    }

    //----------------------------------------------------------------------------------------------------
    /// Calculates the surface position by linear interpolation of surface neighbor coordinates
    inline Real getDirectionalNeighbourLevelset(const LevelsetGrid& _LS1, const LevelsetGrid& _LS2, const Vec3i& _Coord)
    {
        auto ClampVec3 = [](const Vec3& _In, const Vec3& _Max) -> Vec3
        {
            Vec3 New = _In;
            Real Min = 0.0;
            New.x = min(max(New.x, Min), _Max.x - Real(1.0));
            New.y = min(max(New.y, Min), _Max.y - Real(1.0));
            New.z = min(max(New.z, Min), _Max.z - Real(1.0));
            return New;
        };

        const Vec3 CenterCoord = toVec3(_Coord);
        const Vec3i LSBounds = _LS1.getSize();

        /// Calculate Direction
        Vec3 vDirection = getGradient(_LS1, _Coord.x, _Coord.y, _Coord.z);
        vDirection *= 0.5;

        /// Step in negative direction (LS)
        Real LS2_Value = _LS2.getInterpolated(ClampVec3(CenterCoord + vDirection, toVec3(LSBounds)));
        Real LS1_Value = _LS1.getInterpolated(ClampVec3(CenterCoord + vDirection, toVec3(LSBounds)));

        /// Return distance between both values
        return abs(LS2_Value - LS1_Value);
    }

    //----------------------------------------------------------------------------------------------------
    // levelset distance kernel
    KERNEL(reduce = +) returns(int surfacePoints = 0) returns(Real surfaceDistances = 0)
    void knLevelsetDistance(const LevelsetGrid& _LS1, const LevelsetGrid& _LS2)
    {
        Vec3i CenterCoord = Vec3i(i, j, k);
        Real fValueLS1;
        if(isSurfacePoint(_LS1, CenterCoord, fValueLS1))
        {
            Real fMeanNeighbourLS = getDirectionalNeighbourLevelset(_LS1, _LS2, CenterCoord);

            surfaceDistances += fMeanNeighbourLS;
            surfacePoints++;
        }
    }

    //----------------------------------------------------------------------------------------------------
    // levelset distance
    PYTHON()
    Real levelsetDistance(const LevelsetGrid& _LS1, const LevelsetGrid& _LS2, bool _bidirectional=true)
    {
        knLevelsetDistance levelsetDistance12 (_LS1, _LS2);
        /// build mean
        Real fLevelsetDistance = levelsetDistance12.surfaceDistances;
        if(levelsetDistance12.surfacePoints > 0u)
            fLevelsetDistance /= levelsetDistance12.surfacePoints;

        if(_bidirectional)
        {
            knLevelsetDistance levelsetDistance21(_LS2, _LS1);
            /// build mean
            Real fLevelsetDistance21 = levelsetDistance21.surfaceDistances;
            if(levelsetDistance21.surfacePoints > 0u)
                fLevelsetDistance21 /= levelsetDistance21.surfacePoints;

            ///// build mean over end result
            //fLevelsetDistance += fLevelsetDistance21;
            //fLevelsetDistance *= 0.5f;

            /// take max (closer to hausdorff distance)
            fLevelsetDistance = max<Real>(fLevelsetDistance, fLevelsetDistance21);
        }

        return fLevelsetDistance;
    }

    //----------------------------------------------------------------------------------------------------
    // levelset distance to grid
    KERNEL()
    void knLevelsetDistanceToGrid(LevelsetGrid& _Target, const LevelsetGrid& _LS1, const LevelsetGrid& _LS2, bool _bidirectional)
    {
        Vec3i CenterCoord = Vec3i(i, j, k);
        Real fValueLS;
        Real fLSDistance = 0.0f;
        if (isSurfacePoint(_LS1, CenterCoord, fValueLS))
        {
            fLSDistance = getDirectionalNeighbourLevelset(_LS1, _LS2, CenterCoord);
        }

        if (_bidirectional)
        {
            if (isSurfacePoint(_LS2, CenterCoord, fValueLS))
            {
                fLSDistance = max<Real>(fLSDistance, getDirectionalNeighbourLevelset(_LS2, _LS1, CenterCoord));
            }
        }

        _Target(CenterCoord) = fLSDistance;
    }

    //----------------------------------------------------------------------------------------------------
    PYTHON()
    void levelsetDistanceToGrid(LevelsetGrid& _Target, const LevelsetGrid& _LS1, const LevelsetGrid& _LS2, bool _bidirectional = true)
    {
        knLevelsetDistanceToGrid levelsetToGrid(_Target, _LS1, _LS2, _bidirectional);
    }


	//----------------------------------------------------------------------------------------------------
	// mean squared error
	KERNEL(reduce = +) returns(Real squared_dist = 0)
	Real knSquaredDistance(const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		squared_dist += square(_LS1(CenterCoord) - _LS2(CenterCoord));
	}
	KERNEL(reduce = +) returns(Real squared_dist = 0)
	Real knSquaredDistanceVec3(const Grid<Vec3>& _G1, const Grid<Vec3>& _G2)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		Vec3 g1 = _G1(CenterCoord);
		Vec3 g2 = _G2(CenterCoord);
		squared_dist += square(g1.x - g2.x) + square(g1.y - g2.y) + square(g1.z - g2.z);
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	Real meanSquaredError(const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		assertMsg(_LS1.getSize() == _LS2.getSize(), "LS1 and LS2 must be of equal size.");
		Real squaredError = knSquaredDistance(_LS1, _LS2);
		squaredError /= static_cast<Real>(_LS1.getSizeX() * _LS1.getSizeY() * _LS1.getSizeZ());
		return squaredError;
	}
	PYTHON()
	Real meanSquaredErrorVec3(const Grid<Vec3>& _G1, const Grid<Vec3>& _G2)
	{
		assertMsg(_G1.getSize() == _G2.getSize(), "_G1 and _G2 must be of equal size.");
		Real squaredError = knSquaredDistanceVec3(_G1, _G2);
		squaredError /= static_cast<Real>(_G1.getSizeX() * _G1.getSizeY() * _G1.getSizeZ());
		return squaredError;
	}

	//----------------------------------------------------------------------------------------------------
	// mean absolute error
	KERNEL(reduce = +) returns(Real dist = 0)
	Real knAbsoluteDistance(const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		dist += abs(_LS1(CenterCoord) - _LS2(CenterCoord));
	}
	KERNEL(reduce = +) returns(Real dist = 0)
	Real knAbsoluteDistanceVec3(const Grid<Vec3>& _G1, const Grid<Vec3>& _G2)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		Vec3 g1 = _G1(CenterCoord);
		Vec3 g2 = _G2(CenterCoord);
		dist += abs(g1.x-g2.x) + abs(g1.y - g2.y) + abs(g1.z - g2.z);
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	Real meanAbsoluteError(const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		assertMsg(_LS1.getSize() == _LS2.getSize(), "LS1 and LS2 must be of equal size.");
		Real absError = knAbsoluteDistance(_LS1, _LS2);
		absError /= static_cast<Real>(_LS1.getSizeX() * _LS1.getSizeY() * _LS1.getSizeZ());
		return absError;
	}
	PYTHON()
	Real meanAbsoluteErrorVec3(const Grid<Vec3>& _G1, const Grid<Vec3>& _G2)
	{
		assertMsg(_G1.getSize() == _G2.getSize(), "_G1 and _G2 must be of equal size.");
		Real absError = knAbsoluteDistanceVec3(_G1, _G2);
		absError /= static_cast<Real>(_G1.getSizeX() * _G1.getSizeY() * _G1.getSizeZ());
		return absError;
	}

	//----------------------------------------------------------------------------------------------------
	// mean absolute error to grid
	KERNEL()
	void knAbsoluteErrorToGrid(Grid<Real>& _Target, const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		_Target(CenterCoord) = abs(_LS1(CenterCoord) - _LS2(CenterCoord));
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	void absoluteErrorToGrid(Grid<Real>& _Target, const Grid<Real>& _LS1, const Grid<Real>& _LS2)
	{
		knAbsoluteErrorToGrid levelsetToGrid(_Target, _LS1, _LS2);
	}


	//----------------------------------------------------------------------------------------------------
	// standard deviation
	KERNEL(reduce = +) returns(Real squared_dev = 0)
	Real knSquaredDeviation(const Grid<Real>& _LS, const Real _fMean)
	{
		Vec3i CenterCoord = Vec3i(i, j, k);
		squared_dev += square(abs(_LS(CenterCoord) - _fMean));
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	Real standardDeviation(const Grid<Real>& _LS, const Real _fMean)
	{
		Real fSquaredDev = knSquaredDeviation(_LS, _fMean);
		fSquaredDev /= static_cast<Real>(_LS.getSizeX() * _LS.getSizeY() * _LS.getSizeZ());
		if(fSquaredDev > 0.0f)
			fSquaredDev = sqrtf(fSquaredDev);
		return fSquaredDev;
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	void getGradientGrid(const Grid<Real>& _Grid, Grid<Vec3>& _GradGrid)
	{
		GradientOp(_GradGrid, _Grid);
	}

	//----------------------------------------------------------------------------------------------------
	KERNEL(bnd=1, reduce = +) returns(Real fluid_cell_cnt = 0)
	Real knFluidCellCount(const FlagGrid &flags)
	{
		if (flags.isFluid(i, j, k))
			++fluid_cell_cnt;
	}
	//----------------------------------------------------------------------------------------------------
	// KERNEL(bnd=1)
	// void knDivergence(Grid<Real>& div, const MACGrid& grid)
	// {
	// 	Vec3 del = Vec3(grid(i + 1, j, k).x, grid(i, j + 1, k).y, 0.) - grid(i, j, k);
	// 	if (grid.is3D()) del[2] += grid(i, j, k + 1).z;
	// 	else             del[2] = 0.;
	// 	div(i, j, k) = del.x + del.y + del.z;
	// }
	KERNEL(bnd=1)
	void knDivergence(Grid<Real> &div, const MACGrid &grid, const FlagGrid &flags)
	{
		if (!flags.isFluid(i, j, k))
			return; // only for fluid cells

		Vec3 del = Vec3(grid(i + 1, j, k).x, grid(i, j + 1, k).y, 0.) - grid(i, j, k);
		if (grid.is3D())	del[2] += grid(i, j, k + 1).z;
		else				del[2] = 0.;
		div(i, j, k) = del.x + del.y + del.z;
	}

	// the kernel is broken
	//KERNEL(idx, reduce = +) returns(Real result = 0.0)
	//Real knGridTotalSum(const Grid<Real>& a)
	//{
	//	result += fabs(a[idx]);
	//}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	Real getDivergence(Grid<Real> &DivGrid, const MACGrid &_Vel, const FlagGrid &_Flags)
	{
		DivGrid.setConst(0.0);
		knDivergence(DivGrid, _Vel, _Flags);
		Real fNumberOfCells = knFluidCellCount(_Flags);
		Real fDiv = 0.0;
		FOR_IJK(DivGrid)
		{
			fDiv += fabs(DivGrid(i, j, k));
		}
		//Real fDiv = knGridTotalSum(DivGrid);
		return fDiv / fNumberOfCells;
	}

	//----------------------------------------------------------------------------------------------------
	PYTHON()
	void getDivergenceGrid(Grid<Real> &DivGrid, const MACGrid &_Vel, const FlagGrid &_Flags)
	{
		DivGrid.setConst(0.0);
		knDivergence(DivGrid, _Vel, _Flags);
	}

	//----------------------------------------------------------------------------------------------------
	// calculates the PSNR for the noisy field K and the ground truth I
	// _fMinValue: describes the minimal value that can be reached in the ground truth grid _I
	// _fMaxValue: describes the maximal value that can be reached in the ground truth grid _I
	// the returned value ranges from 0.0 to infinity, higher is better
	PYTHON()
	Real peakSignalToNoiseRatio(const Grid<Real>& _I, const Grid<Real>& _K, const Real _fMinValue, const Real _fMaxValue)
	{
		/// special case: MSE is zero -> PSNR should return infinity
		Real meanSqrError = meanSquaredError(_I, _K);
		Real maxI = _fMaxValue - _fMinValue;
		/// Sanity checks
		if(meanSqrError == 0.0)
		{
			return std::numeric_limits<Real>::infinity();
		}
		if(maxI == 0.0)
		{
			return 0.0;
		}
		/// PSNR calculation
		return 20.0 * std::log10(maxI) - 10.0 * std::log10(meanSqrError);
	}
	//----------------------------------------------------------------------------------------------------
	PYTHON()
	Real peakSignalToNoiseRatioVec3(const Grid<Vec3>& _I, const Grid<Vec3>& _K, const Real _fMinValue, const Real _fMaxValue)
	{
		/// special case: MSE is zero -> PSNR should return infinity
		Real meanSqrError = meanSquaredErrorVec3(_I, _K);
		Real maxI = _fMaxValue - _fMinValue;
		/// Sanity checks
		if(meanSqrError == 0.0)
		{
			return std::numeric_limits<Real>::infinity();
		}
		if(maxI == 0.0)
		{
			return 0.0;
		}
		/// PSNR calculation
		return 20.0 * std::log10(maxI) - 10.0 * std::log10(meanSqrError);
	}
} // namespace Manta
