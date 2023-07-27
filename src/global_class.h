#pragma once
#include <ctime>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "global_setup.h"

void getInput(std::string filename, real_t &Re, real_t &U, real_t &rho0, int &NX, int &NY);

// D2Q9
#define Q 9
const real_t w0 = 4.0 / 9.0;
const real_t w1 = 1.0 / 9.0;
const real_t w2 = 1.0 / 36.0;

class BaseSolver;

class BaseSolver
{
protected:
    size_t NX, NY;
    real_t Re, U, rho0, nu, tau, omega, O_rhoba, O_Fd, O_cd;
    D_Data1d ex, ey, w;
    D_IData1d iex, iey;
    D_Data2d rho, ux, uy, ux0, uy0, stream_func, vorticity;
    H_Data2d h_rho, h_ux, h_uy, h_stream_func, h_vorticity;
    D_Data3d f, fb;

public:
    BaseSolver(size_t nx, size_t ny, real_t Re_, real_t U_, real_t rho_);
    void Initialize();
    void Evolution();
    void Collision();
    void Propagation();
    void Boundary();
    void Update();
    real_t Error();
    void StreamFunction();
    void VorticityFunction();
    void Cal_cd_Fd();
    void Output(const int m, std::string Path);
};