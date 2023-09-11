#include "global_class.h"

int main(int argc, char *argv[])
{
    int NX, NY, mx = 1;
    real_t Re_, U_, rho_;
    std::string Ini = std::string(IniFile);
    getInput(Ini + "/input.txt", Re_, U_, rho_, NX, NY);

    if (argc == 2)
        sscanf(argv[1], "%d", &mx);
    else if (argc == 3)
        sscanf(argv[1], "%d", &mx), sscanf(argv[2], "%d", &NX), NY = NX;
    else if (argc == 4)
        sscanf(argv[1], "%d", &mx), sscanf(argv[2], "%d", &NX), sscanf(argv[3], "%d", &NY);
    // ##Begin# MPI ###########################################################################//
    int rank = 0, nranks = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    // ##Begin#Kokkos##########################################################################//
    Kokkos::initialize(argc, argv);
    Kokkos::DefaultExecutionSpace().print_configuration(std::cout);
    {
        BaseSolver Bs(NX, NY, Re_, U_, rho_, mx); // Ini
        Timer timer;
        for (int tn = 0;; ++tn) // tn < 21
        {
            if (0 == rank && 0 == (tn % 2000))
                timer.OutTime(), std::cout << "The " << std::setprecision(10) << tn << "th computation todo, time consumption: " << timer.time / 1000.0 << "s.\n";
            Bs.Evolution();
            if (tn > 1 && 0 == (tn % 10000))
            {
                real_t L2Error = Bs.Error();
                if (0 == rank) //<< std::setprecision(6) << std::setiosflags(std::ios_base::scientific)
                    std::cout << "The " << std::setprecision(10) << tn << "th computation L2Error is " << L2Error << "\n";
                Bs.Output(tn, Ini);

                if (L2Error < 1e-10 || std::isnan(L2Error))
                {
                    Bs.Output(tn, Ini);
                    break;
                }
            }
        }
    }
    Kokkos::finalize();
    // ##End##Kokkos##########################################################################//
    MPI_Finalize();
    // ##End## MPI ###########################################################################//
    return 0;
}