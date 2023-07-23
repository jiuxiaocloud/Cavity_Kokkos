#pragma once
#include "global_setup.h"

class BaseSolver;

class BaseSolver
{
public:
    BaseSolver();
    void AllocateMemory();
    void Init();
    void Evolution();
};