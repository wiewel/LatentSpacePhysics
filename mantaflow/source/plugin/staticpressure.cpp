/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2018 Moritz Becher, Steffen Wiewel, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Plugins for pressure separation
 *
 ******************************************************************************/

#include "grid.h"
#include "levelset.h"
#include "kernel.h"

#include <math.h>

using namespace std;

namespace Manta
{
    
    //----------------------------------------------------------------------------------------------------
    void extrapolateLsSimple (Grid<Real>& phi, int distance = 4, bool inside=false );

    //----------------------------------------------------------------------------------------------------
    KERNEL()
    void knResizeLevelset(Grid<Real>& levelset, Real distance = 4.0f)
    {
        levelset(i,j,k) = levelset(i,j,k) + distance;
    }
    //----------------------------------------------------------------------------------------------------
    void SmoothLevelset(LevelsetGrid& _Levelset, const FlagGrid& _Flags, Real _Distance)
    {
        FlagGrid f = _Flags;
        f.updateFromLevelset(_Levelset);
        _Levelset.reinitMarching(f, _Distance * 1.5f);

        knResizeLevelset(_Levelset, _Distance);
        f.updateFromLevelset(_Levelset);
        _Levelset.reinitMarching(f, _Distance * 1.5f);
        //extrapolateLsSimple(_Levelset, 32);
        knResizeLevelset(_Levelset, -_Distance);
        
        f.updateFromLevelset(_Levelset);
        _Levelset.reinitMarching(f, _Distance * 1.5f);
        //extrapolateLsSimple(_Levelset, 32);
    }
    //----------------------------------------------------------------------------------------------------
    void ClampCoordinate(Vec3i& _Coord, const Vec3i& _Size)
    {
        if (_Coord.x < 0)
            _Coord.x = 0;
        if (_Coord.y < 0)
            _Coord.y = 0;
        if (_Coord.z < 0)
            _Coord.z = 0;
        if (_Coord.x >= _Size.x)
            _Coord.x = _Size.x - 1;
        if( _Coord.y >= _Size.y)
            _Coord.y = _Size.y - 1;
        if( _Coord.z >= _Size.z)
            _Coord.z = _Size.z - 1;
    }
    //----------------------------------------------------------------------------------------------------
    Real SafeSample(const Grid<Real>& _Grid, int i, int j, int k)
    {
        Vec3i Coord = Vec3i(i, j, k);
        ClampCoordinate(Coord, _Grid.getSize());
        return _Grid(Coord);
    }
    //----------------------------------------------------------------------------------------------------
    Real GaussValue(Real _Distance, Real _Sigma)
    {
        Real TwoSqSigma = 2.0 * _Sigma * _Sigma;
        return (1.0 / sqrtf(3.141599265359 * TwoSqSigma)) * exp(- _Distance * _Distance / TwoSqSigma);
    }
    //----------------------------------------------------------------------------------------------------
    KERNEL()
    void knBlurDir(const Grid<Real>& _InGrid, Grid<Real>& _OutGrid, unsigned int _Radius, Real _Sigma, unsigned int _Axis)
    {
        Real GaussFactor = GaussValue(0.0, _Sigma);
        //Real GaussFactorSum = GaussFactor;
        Real Blur = GaussFactor * _InGrid(i,j,k);

        for(unsigned int Offset = 1; Offset < _Radius; ++Offset)
        {
            Real Left = 0.0;
            Real Right = 0.0;
            switch(_Axis)
            {
                case 0:
                    Left = SafeSample(_InGrid, i - Offset, j, k);
                    Right = SafeSample(_InGrid, i + Offset, j, k);
                    break;
                case 1:
                    Left = SafeSample(_InGrid, i, j - Offset, k);
                    Right = SafeSample(_InGrid, i, j + Offset, k);
                    break;
                case 2:
                    Left = SafeSample(_InGrid, i, j, k - Offset);
                    Right = SafeSample(_InGrid, i, j, k + Offset);
                    break;
                default:
                    return;
            }
            Real GaussFactor = GaussValue(static_cast<Real>(Offset), _Sigma);
            //GaussFactorSum += GaussFactor;
            Blur += GaussFactor * (Left + Right);
        }
        _OutGrid(i,j,k) = Blur; // / GaussFactorSum;
    }
    //----------------------------------------------------------------------------------------------------
    // http://dev.theomader.com/gaussian-kernel-calculator/
    void Blur(Grid<Real>& _Grid, unsigned int _Radius = 4, Real _Sigma = 1.0)
    {
        // Radius of 1 or smaller is not valid
        if (_Radius < 2u)
            return;
        Grid<Real> Tmp(_Grid);
        knBlurDir(Tmp, _Grid, _Radius, _Sigma, 0);
        Grid<Real> VerticallyBlured(_Grid);
        knBlurDir(VerticallyBlured, _Grid, _Radius, _Sigma, 2);
    }

    //----------------------------------------------------------------------------------------------------
    void ComputeStaticPressure(const Grid<Real>& _Levelset, const Vec3& _vGravity, Grid<Real>& _StaticPressure)
    {
        const Real fGravity = -_vGravity.y;
        for(int k = 0; k < _Levelset.getSizeZ(); ++k)
        {
            for(int i = 0; i < _Levelset.getSizeX(); ++i)
            {
                const int iSizeY = _Levelset.getSizeY();
                int iSurface = 0;
                Real fDistance = 0.0f;

                do
                {
                    fDistance = max<Real>(0.0, -_Levelset(i, iSurface, k));
                    iSurface += floorf(fDistance);
                }
                while(fDistance >= 1.0 && iSurface < iSizeY);
                
                for(int j = iSurface; j >= 0; --j)
                {
                    Real fHeight = fDistance + static_cast<float>(iSurface - j);
                    fHeight = min<Real>(iSizeY, fHeight);
                    // gravity factor
                    const Real fGravityFactor = _StaticPressure.getParent()->getDt() / _StaticPressure.getDx(); /// 0.0981 / ( 1 / 64 ) = 6,2784 <- only makes sense for 64 fields
                    // Static Pressure = Density * Gravity * Height
                    _StaticPressure(i,j,k) = fGravityFactor * fGravity * fHeight;
                }
            }
        }
    }
    
    //----------------------------------------------------------------------------------------------------
    KERNEL()
    void knClearNonfluid(Grid<Real>& _StaticPressure, const FlagGrid& _Flags)
    {
        // In the walls pressure is zero
        if(_Flags.isFluid(i,j,k) == false)
        {
            _StaticPressure(i,j,k) = 0.0f;
        }
    }

    //----------------------------------------------------------------------------------------------------
    KERNEL()
    void knComputeDynamicPressure(const Grid<Real>& _TotalPressure, const Grid<Real>& _StaticPressure, Grid<Real>& _DynamicPressure)
    {
        // Total Pressure = Static Pressure + Dynamic Pressure
        _DynamicPressure(i,j,k) = _TotalPressure(i,j,k) - _StaticPressure(i,j,k);
    }

    //----------------------------------------------------------------------------------------------------
    KERNEL(idx)
    void knCombinePressure(const Grid<Real>& _DynamicPressure, const Grid<Real>& _StaticPressure, Grid<Real>& _TotalPressure)
    {
        _TotalPressure(idx) = _StaticPressure(idx) + _DynamicPressure(idx);
    }

    //----------------------------------------------------------------------------------------------------
    PYTHON()
    void separatePressure(const LevelsetGrid& levelset, const Grid<Real>& total_pressure, const FlagGrid& flags, const Vec3& gravity, Grid<Real>& dynamic_pressure, Grid<Real>& static_pressure, int blur_radius = 4)
    {
        // smooth the levelset to get smoother static pressure
        LevelsetGrid ls = levelset;
        SmoothLevelset(ls, flags, blur_radius);
        
        ComputeStaticPressure(ls, gravity, static_pressure);
        //Blur(static_pressure, 2u, 1.0);
        // clear static pressure in nonfluid cells
        //knClearNonfluid(static_pressure, flags);
        knComputeDynamicPressure(total_pressure, static_pressure, dynamic_pressure);
    }

    //----------------------------------------------------------------------------------------------------
    // simply add the static and dynamic pressure
    PYTHON()
    void combinePressure(const Grid<Real>& dynamic_pressure, const Grid<Real>& static_pressure, Grid<Real>& total_pressure)
    {
        knCombinePressure(dynamic_pressure, static_pressure, total_pressure);
    }

    //----------------------------------------------------------------------------------------------------
    PYTHON()
    void computeStaticPressure(const LevelsetGrid& levelset, const FlagGrid& flags, const Vec3& gravity, Grid<Real>& static_pressure, int blur_radius = 4)
    {
        // smooth the levelset to get smoother static pressure
        LevelsetGrid ls = levelset;
        if(blur_radius > 0)
        {
            SmoothLevelset(ls, flags, blur_radius);
        }
        
        ComputeStaticPressure(ls, gravity, static_pressure);
        //Blur(static_pressure, 2u, 1.0);
        // clear static pressure in nonfluid cells
        //knClearNonfluid(static_pressure, flags);
    }
    //----------------------------------------------------------------------------------------------------
    /// TODO: currently collides with blurRealGrid implementation in initplugins.cpp
    // PYTHON()
    // void blurRealGrid(Grid<Real>& grid, const FlagGrid& flags, int blur_radius = 4, Real _fSigma = 1.0f)
    // {
    //     Grid<Real> Tmp(grid);
    //     knBlurDir(Tmp, grid, blur_radius, _fSigma, 0);
    //     Grid<Real> VerticallyBlured(grid);
    //     knBlurDir(VerticallyBlured, grid, blur_radius, _fSigma, 2);
    //     Grid<Real> HorizontallyBlured(grid);
    //     knBlurDir(HorizontallyBlured, grid, blur_radius, _fSigma, 1);
        
    //     knClearNonfluid(grid, flags);
    // }

	//----------------------------------------------------------------------------------------------------
	template<class S>
	void simpleBlurFunc(Grid<S>& a, int iter = 1, Real strength = 1.)
	{
		Grid<S> tmp(a.getParent());
		for (int numIt = 0; numIt<iter; numIt++) {
			FOR_IJK_BND(a, 1) {
				tmp(i, j, k) = a(i, j, k) + a(i + 1, j, k) + a(i - 1, j, k) + a(i, j + 1, k) + a(i, j - 1, k);
				if (a.is3D()) {
					tmp(i, j, k) += a(i, j, k + 1) + a(i, j, k - 1);
					tmp(i, j, k) *= 1. / 7.;
				}
				else {
					tmp(i, j, k) *= 1. / 5.;
				}
				tmp(i, j, k) = tmp(i, j, k)*strength + a(i, j, k)*(1. - strength);
			}
			a.swap(tmp);
		}
	}
	PYTHON() void simpleBlur(Grid<Real>& a, int iter = 1) { simpleBlurFunc<Real>(a, iter); }
	PYTHON() void simpleBlurVec(Grid<Vec3>& a, int iter = 1) { simpleBlurFunc<Vec3>(a, iter); }

    //----------------------------------------------------------------------------------------------------
    KERNEL(idx)
    void knZeroBand(Grid<Real>& _Grid, const Real _Threshold)
    {
        if(fabs(_Grid(idx)) < _Threshold)
        {
            _Grid(idx) = 0.0f;
        }
    }

    //----------------------------------------------------------------------------------------------------
    PYTHON()
    void zeroBand(Grid<Real>& grid, const Real threshold)
    {
        knZeroBand(grid, threshold);
    }

    //----------------------------------------------------------------------------------------------------
    PYTHON()
    void clearNonfluid(Grid<Real>& grid, const FlagGrid& flags)
    {
        knClearNonfluid(grid, flags);
    }
	
	//----------------------------------------------------------------------------------------------------
	KERNEL(ijk)
	void knSmoothenSurface(Grid<Real>& _Grid, const LevelsetGrid& _Levelset, const Real _SurfaceWidth)
	{
		if (_Levelset(i,j,k) > 0.0f || _Levelset(i,j,k) <= -_SurfaceWidth)
		{
			return;
		}

		Vec3i Low = Vec3i(i-1, j-1, k-1);
		ClampCoordinate(Low, _Grid.getSize());
		Vec3i High = Vec3i(i + 1, j + 1, k + 1);
		ClampCoordinate(High, _Grid.getSize());

		Real fAccumulator = _Grid(i, j, k);
		fAccumulator += _Grid(Low.x, j, k);
		fAccumulator += _Grid(High.x, j, k);
		fAccumulator += _Grid(i, Low.y, k);
		fAccumulator += _Grid(i, High.y, k);
		fAccumulator += _Grid(i, j, Low.z);
		fAccumulator += _Grid(i, j, High.z);

		_Grid(i, j, k) = fAccumulator / 7.0;
	}
	
	//----------------------------------------------------------------------------------------------------
	PYTHON()
	void smoothenSurface(Grid<Real>& grid, const LevelsetGrid& levelset, const int iterations = 1, const Real surface_width = 2.0f)
	{
		for (int i = 0; i < iterations; ++i)
		{
			knSmoothenSurface(grid, levelset, surface_width);
		}
	}
}
