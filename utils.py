"""
Functions for processing data from the computational model described in the paper:
"A Stackelberg game scheme for pricing and task offloading based on idle node-assisted edge computational model"

Key formulas and references:
- Local execution time: Eq. (3) and (1)
- Local energy consumption: Eq. (4) and (2)
- MEC execution time: Eq. (5)
- MEC energy consumption: Eq. (6)
- ID execution time: Eq. (7)
- ID energy consumption: Eq. (8)
"""


def calculate_task_execution_time_on_local(C, d, F, x):
    """
    Calculate partial task execution time on local device (Eq. 3)
    Input:
        C (float): CPU cycles per bit (C)
        d (float): Task data size (d_i)
        F (float): Local device computing power (F_i)
        x (float): Task offloading ratio (x_i)
    Output:
        float: Local execution time for (1-x) portion of the task
    """
    return (1 - x) * C * d / F


def calculate_full_task_execution_time_on_local(C, d, F):
    """
    Calculate full task execution time when executed entirely locally (Eq. 1)
    Input:
        C (float): CPU cycles per bit (C)
        d (float): Task data size (d_i)
        F (float): Local device computing power (F_i)
    Output:
        float: Total local execution time
    """
    return calculate_task_execution_time_on_local(C, d, F, 0)


def calculate_task_execution_energy_consumption_on_local(C, d, theta_exe, x):
    """
    Calculate energy consumption for partial local execution (Eq. 4)
    Input:
        C (float): CPU cycles per bit (C)
        d (float): Task data size (d_i)
        theta_exe (float): Energy cost per CPU cycle
        x (float): Task offloading ratio (x_i)
    Output:
        float: Energy consumption for (1-x) portion executed locally
    """
    b = C * d  # Total CPU cycles required (b_i)
    return b * (1 - x) * theta_exe


def calculate_full_task_execution_energy_consumption_on_local(C, d, theta_exe):
    """
    Calculate full energy consumption for local task execution (Eq. 2)
    Input:
        C (float): CPU cycles per bit (C)
        d (float): Task data size (d_i)
        theta_exe (float): Energy cost per CPU cycle
    Output:
        float: Total energy consumption for full local execution
    """
    return calculate_task_execution_energy_consumption_on_local(C, d, theta_exe, 0)


def calculate_partial_task_execution_time_on_mec(x, d, omega, C, f):
    """
    Calculate MEC execution time for (1-ω) portion of offloaded task (Eq. 5)
    Input:
        x (float): Task offloading ratio (x_i)
        d (float): Task data size (d_i)
        omega (float): ID offloading ratio (w_i)
        C (float): CPU cycles per bit (C)
        f (float): MEC allocated computing power (f_i)
    Output:
        float: Execution time at MEC server
    """
    I = x * d  # Offloaded data size (I_i)
    return C * (1 - omega) * I / f


def calculate_partial_task_execution_energy_consumption_on_mec(
    x, d, omega, C, theta_exe
):
    """
    Calculate MEC energy consumption for (1-ω) portion (Eq. 6)
    Input:
        x (float): Task offloading ratio (x_i)
        d (float): Task data size (d_i)
        omega (float): ID offloading ratio (w_i)
        C (float): CPU cycles per bit (C)
        theta_exe (float): Energy cost per CPU cycle
    Output:
        float: Energy consumption at MEC server
    """
    I = x * d  # Offloaded data size (I_i)
    return C * (1 - omega) * I * theta_exe


def calculate_partial_task_execution_time_on_id(C, omega, x, d, f_id):
    """
    Calculate ID execution time for ω portion of offloaded task (Eq. 7)

    Input:
        C (float): CPU cycles per bit (C)
        omega (float): ID offloading ratio (w_i)
        x (float): Task offloading ratio (x_i)
        d (float): Task data size (d_i)
        f_id (float): ID allocated computing power (f_i^ID)

    Output:
        float: Execution time at ID node
    """
    I = x * d  # Offloaded data size (I_i)
    return C * omega * I / f_id


def calculate_partial_task_execution_energy_consumption_on_ID(
    C, omega, x, d, theta_exe
):
    """
    Calculate ID energy consumption for ω portion (Eq. 8)
    Input:
        C (float): CPU cycles per bit (C)
        omega (float): ID offloading ratio (w_i)
        x (float): Task offloading ratio (x_i)
        d (float): Task data size (d_i)
        theta_exe (float): Energy cost per CPU cycle
    Output:
        float: Energy consumption at ID node
    """
    I = x * d  # Offloaded data size (I_i)
    return C * omega * I * theta_exe


"""
Communication model functions based on the paper:
"A Stackelberg game scheme for pricing and task offloading based on idle node-assisted edge computational model"

Key formulas and references:
- SINR calculation: Eq. (9)-(10)
- Transmission rate: Eq. (11)-(12)
- Transmission time: Eq. (13), (15)
- Transmission energy: Eq. (14), (16)
"""


def calculate_sinr_eu_to_bs(P_tran_i_bs, G_i_bs, sigma_sq, interference_sum):
    """
    Calculate SINR for EU to BS transmission (Eq. 9)
    Input:
        P_tran_i_bs(float): EU transmission power (P_i^tran)
        G_i_bs (float): Channel gain between EU i and BS
        sigma_sq (float): Noise power (σ²)
        interference_sum (float): Sum of interference from other EUs
    Output:
        float: Signal-to-Interference-plus-Noise Ratio (γ_i,BS)
    """
    return (P_tran_i_bs * G_i_bs) / (sigma_sq + interference_sum)


def calculate_sinr_bs_to_id(P_tran_bs_j, G_bs_j, sigma_sq):
    """
    Calculate SINR for BS to ID transmission (Eq. 10)
    Input:
        P_tran_bs_j (float): BS transmission power to ID j (P_BS,j^tran)
        G_bs_j (float): Channel gain between BS and ID j
        sigma_sq (float): Noise power (σ²)
    Output:
        float: Signal-to-Interference-plus-Noise Ratio (γ_BS,j)
    """
    return (P_tran_bs_j * G_bs_j) / sigma_sq


def calculate_transmission_rate(B_m, sinr):
    """
    Calculate transmission rate using Shannon formula (Eq. 11-12)
    Input:
        B_m (float): Channel bandwidth (B_m)
        sinr (float): SINR value
    Output:
        float: Transmission rate (R)
    """
    from math import log2

    return B_m * log2(1 + sinr)


def calculate_eu_to_bs_transmission_time(x, d, R_i_bs):
    """
    Calculate EU to BS transmission time (Eq. 13)
    Input:
        x (float): Task offloading ratio
        d (float): Task data size (d_i)
        R_i_bs (float): Transmission rate from EU i to BS
    Output:
        float: Transmission time (t_i^tran)
    """
    return (x * d) / R_i_bs


def calculate_bs_to_id_transmission_time(omega, x, d, R_bs_j):
    """
    Calculate BS to ID transmission time (Eq. 15)
    Input:
        omega (float): ID offloading ratio (w_i)
        x (float): Task offloading ratio
        d (float): Task data size (d_i)
        R_bs_j (float): Transmission rate from BS to ID j
    Output:
        float: Transmission time (t_BS,j^tran)
    """
    return (omega * x * d) / R_bs_j


def calculate_eu_transmission_energy(P_tran_i_bs, t_tran_i_bs, theta_tran):
    """
    Calculate EU transmission energy consumption (Eq. 14)
    Input:
        P_tran_i_bs (float): EU transmission power (P_i^tran)
        t_tran_i_bs (float): Transmission time
        theta_tran (float): Energy cost per transmission unit
    Output:
        float: Transmission energy (c_i^tran)
    """
    return P_tran_i_bs * t_tran_i_bs * theta_tran


def calculate_bs_transmission_energy(P_tran_bs_j, t_tran_bs_j, theta_tran):
    """
    Calculate BS transmission energy consumption (Eq. 16)
    Input:
        P_tran_bs_j (float): BS transmission power to ID j
        t_tran_bs_j (float): Transmission time
        theta_tran (float): Energy cost per transmission unit
    Output:
        float: Transmission energy (c_BS,j^tran)
    """
    return P_tran_bs_j * t_tran_bs_j * theta_tran


def calculate_offloading_delay(t_tran_i_bs, t_mec, t_id, t_tran_bs_j):
    """
    Calculate total offloading delay (Eq. 17)
    Input:
        t_tran_i_bs (float): EU to BS transmission time
        t_mec (float): task execution time
        t_id (float): ID execution time
        t_tran_bs_j (float): BS to ID transmission time
    Output:
        float: Total offloading delay (t_off)
    """
    return t_tran_i_bs + max(t_mec, t_id + t_tran_bs_j)


def calculate_total_task_time(t_local, t_off):
    """
    Calculate total task completion time (Eq. 18)
    Input:
        t_local (float): Local execution time
        t_off (float): Offloading delay
    Output:
        float: Total task completion time (t_i^tot)
    """
    return max(t_local, t_off)
