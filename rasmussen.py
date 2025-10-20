import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import periodictable
from dataclasses import dataclass

# For plotting nuclear potential and Q value run with "draw" argument (python3 rasmussen.py draw) and set last parameter in the input file to 1


# global variables
root_finding_precision = 1e-4   # for finding r_min and r_max
integral_points = 10000         # intervals for integration
# range for plots in r (fm)
r_min_plot = 8
r_max_plot = 60


@dataclass
class Input_Values:
    neutron_number: int
    mass_number: int
    #proton_number: int
    transition_energy: float
    transition_energy_unc: float
    half_life: float
    half_life_unc: float
    alpha_branching: float
    alpha_branching_unc: float
    alpha_relative_intensity: float
    alpha_relative_intensity_unc: float
    angular_momentum: int
    draw: int = 0

@dataclass
class Output_Values:
    neutron_number: int
    mass_number: int
    proton_number: int = 0
    r_min: float = 0.0
    r_max: float = 0.0
    transition_energy: float = 0.0
    Q_value: float = 0.0
    half_life: float = 0.0
    partial_half_life: float = 0.0
    alpha_absolute_intensity: float = 0.0
    penetration_factor: float = 0.0
    #penetration_factor_unc_sym: float = 0.0
    #penetration_factor_unc_plus: float = 0.0
    #penetration_factor_unc_minus: float = 0.0
    reduced_width: float = 0.0
    reduced_width_unc_sym: float = 0.0
    reduced_width_unc_plus: float = 0.0
    reduced_width_unc_minus: float = 0.0
    angular_momentum: int = 0


def read_inputs(input_file):
    inputs = []
    with open(input_file, "r") as file:
        for line in file:
            #print(repr(line))
            line = line.strip()
            #print(repr(line))
            parts = line.split()
            if len(parts) != 12:
                raise ValueError(f"Expected 12 values per line, got {len(parts)}: {line.strip()}")

            input_line = Input_Values(
                neutron_number = int(parts[0]),
                mass_number = int(parts[1]),
                #proton_number = int(parts[1]) - int(parts[0]),
                transition_energy = float(parts[2]),
                transition_energy_unc = float(parts[3]),
                half_life = float(parts[4]),
                half_life_unc = float(parts[5]),
                alpha_branching = float(parts[6]),
                alpha_branching_unc = float(parts[7]),
                alpha_relative_intensity = float(parts[8]),
                alpha_relative_intensity_unc = float(parts[9]),
                angular_momentum = int(parts[10]),
                draw = int(parts[11])
            )
            inputs.append(input_line)
    print(f"Loaded coefficients from file {input_file}:")
    print("N\tA\tE (MeV)\tdE (MeV)\tT1/2(s)\tdT1/2(s)\tb_a(%)\tdb_a(%)\tIrel(%)\tdIrel(%)\tl\tDraw?")
    #i=1
    for line in inputs:
        print(f"{line.neutron_number}\t{line.mass_number}\t{line.transition_energy}\t{line.transition_energy_unc}\t{line.half_life}\t{line.half_life_unc}\t{line.alpha_branching}\t{line.alpha_branching_unc}\t{line.alpha_relative_intensity}\t{line.alpha_relative_intensity_unc}\t{line.angular_momentum}\t{line.draw}")
        #print(*vars(line).values())
        

    return inputs

def write_output(outputs, output_file):
    print("Output:")
    print("N\tZ\tA\tE (MeV)\tQ (MeV)\tT1/2(s)\tI_a (%)\tPart. T1/2 (s)\tP\tRed. width (keV)\tUnc. sym\tUnc. plus\tUnc. minus\tl\tr_min (fm)\tr_max (fm)")
    with open(output_file, "w") as file:
            file.write("Reduced alpha-decay widths\n")
            file.write("N\tZ\tA\tE (MeV)\tQ (MeV)\tT1/2(s)\tI_a (%)\tPart. T1/2 (s)\tP\tRed. width (keV)\tUnc. sym\tUnc. plus\tUnc. minus\tl\tr_min (fm)\tr_max (fm)\n")
            for output in outputs:
                file.write(f"{output.neutron_number}\t{output.proton_number}\t{output.mass_number}\t{output.transition_energy}\t{output.Q_value:.3f}\t{output.half_life}\t{output.alpha_absolute_intensity:.4f}\t{output.partial_half_life:.1f}\t{output.penetration_factor:.3e}\t{output.reduced_width:.4f}\t{output.reduced_width_unc_sym:.4f}\t{output.reduced_width_unc_plus:.4f}\t{output.reduced_width_unc_minus:.4f}\t{output.angular_momentum}\t{output.r_min:.3f}\t{output.r_max:.3f}\n")
                print(f"{output.neutron_number}\t{output.proton_number}\t{output.mass_number}\t{output.transition_energy}\t{output.Q_value:.3f}\t{output.half_life}\t{output.alpha_absolute_intensity:.4f}\t{output.partial_half_life:.1f}\t{output.penetration_factor:.3e}\t{output.reduced_width:.4f}\t{output.reduced_width_unc_sym:.4f}\t{output.reduced_width_unc_plus:.4f}\t{output.reduced_width_unc_minus:.4f}\t{output.angular_momentum}\t{output.r_min:.3f}\t{output.r_max:.3f}")
    print("Results are written in file", output_file)

def tunneling(decay, energies, output_line):
    penetration_factor = []
    Qvalues = []
    #r_min = []
    #r_max = []
    proton_number_mother = decay.mass_number - decay.neutron_number
    proton_number_daughter = proton_number_mother - 2
    mass_number_daughter = decay.mass_number - 4
    reduced_mass_A = mass_number_daughter * 4 / decay.mass_number
    Qvalue_correction = (65.3*np.power(proton_number_mother,1.4)-80.0*np.power(proton_number_mother,0.4))*1.0e-6
    for energy in energies: #calculate Q values from decay energies and add correction
        Qvalue = energy*decay.mass_number/(decay.mass_number-4) + Qvalue_correction
        Qvalues.append(Qvalue)
    r_min = find_rmin(proton_number_daughter, mass_number_daughter, decay.angular_momentum, Qvalues)
    r_max = find_rmax(proton_number_daughter, mass_number_daughter, decay.angular_momentum, Qvalues, r_min)
    #print(r_min)
    #print(r_max)
    output_line.proton_number = proton_number_mother
    output_line.Q_value = Qvalues[0]
    output_line.r_min = r_min[0]
    output_line.r_max = r_max[0]
    for Qvalue, rmin, rmax in zip(Qvalues, r_min, r_max):
        r = np.linspace(rmin, rmax, integral_points)
        Gamow_factor_potential = np.sqrt(np.clip(total_potential(proton_number_daughter, mass_number_daughter, decay.angular_momentum, r) - Qvalue, 0, None)) #If r_mim or r_max give negative result, np.clip(array, lower_bound, upper_bound) makes it 0
        Gamow_factor_integral = np.trapz(Gamow_factor_potential, r)
        #print("Integral:", Gamow_factor_integral)
        Gamow_factor = Gamow_factor_integral * np.sqrt(2*reduced_mass_A*931.4941) / 197.32698 # mass in MeV and hbar*c
        #print("Gamow factor:", Gamow_factor)
        p = np.exp(-2*Gamow_factor)
        penetration_factor.append(p)
        
    if "draw" in sys.argv:
        if decay.draw == 1:
            draw_potential(proton_number_daughter, mass_number_daughter, decay.angular_momentum, Qvalues, decay.transition_energy, decay.mass_number, proton_number_mother)    
    return penetration_factor

def potential_nuclear(mass_number_daughter, r): 
    return -1100*np.exp(-(r-1.17*np.power(mass_number_daughter,1.0/3))/0.574) # in MeV

def potential_coulomb(proton_number_daughter, r):
    return 2.87993035*proton_number_daughter/r

def potential_centrifugal(mass_number_daughter, momentum, r):
    return (mass_number_daughter+4)*momentum*(momentum+1)*5.2252016/(mass_number_daughter*np.power(r,2))

def total_potential(proton_number_daughter, mass_number_daughter, momentum, r):
    return potential_nuclear(mass_number_daughter, r) + potential_coulomb(proton_number_daughter, r) + potential_centrifugal(mass_number_daughter, momentum, r)

def find_rmin(proton_number_daughter, mass_number_daughter, momentum, Qvalues): #finds lower integration bounds for three Q values, central, upper, lower 
    r_min = []
    for Qvalue in Qvalues:
        for r in range(1,50):
                if total_potential(proton_number_daughter, mass_number_daughter, momentum, r) - Qvalue >= 0: 
                    r_min1 = r-1
                    r_min2 = r
                    while (r_min2 - r_min1) > root_finding_precision: #stop when r_min is found with a set precision
                        if total_potential(proton_number_daughter, mass_number_daughter, momentum, (r_min1+r_min2)/2.0) - Qvalue >= 0:
                            r_min2 = (r_min1+r_min2)/2.0
                        else:
                            r_min1 = (r_min1+r_min2)/2.0
                    r_min.append((r_min1+r_min2)/2.0)
                    #print(r_min)
                    break
                #else:
                    #print(r, "negative")
    return r_min

def find_rmax(proton_number_daughter, mass_number_daughter, momentum, Qvalues, r_min): # finds upper integration bounds, starts from r_min
    r_max = []
    for Qvalue, rmin in zip(Qvalues, r_min):
        for r in range(math.ceil(rmin),100):
                if total_potential(proton_number_daughter, mass_number_daughter, momentum, r) - Qvalue <= 0: 
                    r_max1 = r-1
                    r_max2 = r
                    while (r_max2 - r_max1) > root_finding_precision:
                        #print(r_max1, r_max2)
                        if total_potential(proton_number_daughter, mass_number_daughter, momentum, (r_max1+r_max2)/2.0) - Qvalue <= 0:
                            r_max2 = (r_max1+r_max2)/2.0
                        else:
                            r_max1 = (r_max1+r_max2)/2.0
                    r_max.append((r_max1+r_max2)/2.0)
                    #print(r_min)
                    break
                #else:
                    #print(r, "positive")
    return r_max

def draw_potential(drawn_proton_number_daughter, drawn_mass_number_daughter, drawn_momentum, drawn_Qvalues, drawn_energy, drawn_mass, drawn_element):
    points = 500
    r_plot = np.linspace(r_min_plot, r_max_plot, points)
    #drawn_proton_number_mother = drawn_decay.mass_number - drawn_decay.neutron_number
    #drawn_proton_number_daughter = drawn_proton_number_mother - 2
    #drawn_mass_number_daughter = drawn_decay.mass_number - 4
    nuclear = potential_nuclear(drawn_mass_number_daughter, r_plot)
    coulomb = potential_coulomb(drawn_proton_number_daughter, r_plot)
    centrifugal = potential_centrifugal(drawn_mass_number_daughter, drawn_momentum, r_plot)
    Q_mean = np.full_like(r_plot, drawn_Qvalues[0])
    #Q_upper = np.full_like(r_plot, drawn_Qvalues[1])
    #Q_lower = np.full_like(r_plot, drawn_Qvalues[2])
    potential_sum = nuclear + coulomb + centrifugal

    plt.figure(figsize=(8,6))

    plt.plot(r_plot, nuclear, label="Nuclear", color="blue", linestyle=":")
    plt.plot(r_plot, coulomb, label="Coulomb", color="green", linestyle=":")
    plt.plot(r_plot, centrifugal, label="Centrifugal", color="purple", linestyle=":")
    plt.plot(r_plot, potential_sum, label="Total potential", color="black", linewidth=2)
    plt.plot(r_plot, Q_mean, label="Q value", color="red")
    #plt.plot(r_plot, Q_upper, label="Q upper bound", color="red", linestyle="--")
    #plt.plot(r_plot, Q_lower, label="Q lower bound", color="red", linestyle="--")

    plt.xlabel("r [fm]")
    plt.ylabel("Energy [MeV]")
    title = f"Barrier penetration for {drawn_energy} MeV decay of {drawn_mass}{periodictable.elements[drawn_element].symbol}, l={drawn_momentum}"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    #plt.show()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    input_file = "rasmussen_input.txt"   
    #input_file = "input.txt" 
    output_file = "rasmussen_output.txt" 
    figures_drawn = 0
    input = read_inputs(input_file)
    output = []
    
    for decay in input:
        #print(decay.transition_energy)
        output_line = Output_Values(neutron_number=decay.neutron_number, mass_number=decay.mass_number, angular_momentum=decay.angular_momentum, half_life=decay.half_life, transition_energy=decay.transition_energy)
        energies = [decay.transition_energy, decay.transition_energy+decay.transition_energy_unc, decay.transition_energy-decay.transition_energy_unc]
        #energies = [central value, upper_bound, lower_bound]
        #print(energies)
        penetration_factor = tunneling(decay, energies, output_line) #penetration_factor[central_value, upper_bound, lower_bound]
        #print("Penetration_factor:", penetration_factor)
        penetration_factor_unc_plus = (penetration_factor[1] - penetration_factor[0])
        penetration_factor_unc_minus = (penetration_factor[0] - penetration_factor[2])
        penetration_factor_unc_sym = (penetration_factor_unc_minus + penetration_factor_unc_plus)/2
        #print("Uncertainty:", penetration_factor_unc_sym, penetration_factor_unc_plus, penetration_factor_unc_minus)
        alpha_absolute_intensity = decay.alpha_branching/100 * decay.alpha_relative_intensity
        alpha_absolute_intensity_unc = alpha_absolute_intensity * np.sqrt(decay.alpha_branching_unc*decay.alpha_branching_unc/decay.alpha_branching/decay.alpha_branching + decay.alpha_relative_intensity_unc*decay.alpha_relative_intensity_unc/decay.alpha_relative_intensity/decay.alpha_relative_intensity)
        #print("Absolute alpha intensity:", alpha_absolute_intensity, alpha_absolute_intensity_unc)
        partial_half_life = decay.half_life/alpha_absolute_intensity*100
        partial_half_life_unc = partial_half_life * np.sqrt(decay.half_life_unc*decay.half_life_unc/decay.half_life/decay.half_life + alpha_absolute_intensity_unc*alpha_absolute_intensity_unc/alpha_absolute_intensity/alpha_absolute_intensity)
        #print("Partial half-life:", partial_half_life, partial_half_life_unc)
        reduced_width = np.log(2)*4.135668e-21/partial_half_life/penetration_factor[0]*1000
        reduced_width_unc_sym = reduced_width * np.sqrt(partial_half_life_unc*partial_half_life_unc/partial_half_life/partial_half_life + penetration_factor_unc_sym*penetration_factor_unc_sym/penetration_factor[0]/penetration_factor[0])
        reduced_width_unc_plus = reduced_width * np.sqrt(partial_half_life_unc*partial_half_life_unc/partial_half_life/partial_half_life + penetration_factor_unc_minus*penetration_factor_unc_minus/penetration_factor[0]/penetration_factor[0])
        reduced_width_unc_minus = reduced_width * np.sqrt(partial_half_life_unc*partial_half_life_unc/partial_half_life/partial_half_life + penetration_factor_unc_plus*penetration_factor_unc_plus/penetration_factor[0]/penetration_factor[0])
        #print("Reduced width:", reduced_width, reduced_width_unc_sym, reduced_width_unc_plus, reduced_width_unc_minus)
        output_line.alpha_absolute_intensity = alpha_absolute_intensity
        output_line.partial_half_life = partial_half_life
        output_line.penetration_factor = penetration_factor[0]
        output_line.reduced_width = reduced_width
        output_line.reduced_width_unc_sym = reduced_width_unc_sym
        output_line.reduced_width_unc_plus = reduced_width_unc_plus
        output_line.reduced_width_unc_minus = reduced_width_unc_minus
        #print(output_line)
        output.append(output_line)
        figures_drawn = figures_drawn + decay.draw

    write_output(output, output_file)

        
        
    
    if "draw" in sys.argv:
        if figures_drawn > 0:
            print("Close figure(s) to exit.")
            plt.show() # this keeps plots open
        else:
            print("No decays were plotted, set last input parameter to 1.")

if __name__ == "__main__":
    main()