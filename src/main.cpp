#include "global_class.h"

int main(int argc, char *argv[])
{
    int NX, NY;
    real_t Re_, U_, rho_;
    std::string Ini = std::string(IniFile);
    getInput(Ini + "/input.txt", Re_, U_, rho_, NX, NY);
    if (argc == 2)
        sscanf(argv[1], "%d", &NX), NY = NX;
    else if (argc == 3)
        sscanf(argv[1], "%d", &NX), sscanf(argv[2], "%d", &NY);
    else if (argc > 3)
        std::cout << "Too much argcs appended to executable.\n";
    // ##Begin#Kokkos##########################################################################//
    Kokkos::initialize(argc, argv);
    Kokkos::DefaultExecutionSpace().print_configuration(std::cout);
    {
        BaseSolver Bs(NX, NY, Re_, U_, rho_); // Ini
        for (int tn = 0;; ++tn)               // tn < 21
        {
            std::cout << "The " << tn << "th computation todo.\n";
            if (tn % 10 == 0)
                Bs.Output(tn, Ini);
            Bs.Evolution();

            if (tn > 1 && tn % 10000 == 0)
            {
                real_t L2Error = Bs.Error();

                std::cout << "The " << tn << "th computation,\t"
                          << std::setprecision(6) << std::setiosflags(std::ios_base::scientific)
                          << "the L2error is " << L2Error << "\n";

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
    return 0;
}