"""
LAMMPS interface for running simulations and reading output.
"""

import subprocess
import numpy as np
import os
import time
from typing import Tuple, Optional, Dict


class LAMMPSInterface:
    """
    Interface for running LAMMPS simulations and reading RDF output.

    Parameters
    ----------
    lammps_command : str, optional
        LAMMPS executable command (default: 'lmp')
    use_gpu : bool, optional
        Use GPU/Kokkos acceleration (default: False)
    gpu_id : int, optional
        GPU device ID (default: 0)
    kokkos_args : list, optional
        Additional Kokkos arguments (default: ['-k', 'on', 'g', '1', '-sf', 'kk'])
    """

    def __init__(
        self,
        lammps_command: str = 'lmp',
        use_gpu: bool = False,
        gpu_id: int = 0,
        kokkos_args: Optional[list] = None
    ):
        self.lammps_command = lammps_command
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        if kokkos_args is None:
            self.kokkos_args = ['-k', 'on', 'g', '1', '-sf', 'kk']
        else:
            self.kokkos_args = kokkos_args

    def run(
        self,
        input_file: str,
        log_file: str = 'lammps.log',
        verbose: bool = True
    ) -> int:
        """
        Run a LAMMPS simulation.

        Parameters
        ----------
        input_file : str
            LAMMPS input script file
        log_file : str, optional
            LAMMPS log file (default: 'lammps.log')
        verbose : bool, optional
            Print progress information (default: True)

        Returns
        -------
        returncode : int
            Return code from LAMMPS (0 = success)

        Raises
        ------
        RuntimeError
            If LAMMPS execution fails
        """
        if self.use_gpu:
            cmd = [self.lammps_command] + self.kokkos_args + \
                  ['-in', input_file, '-log', log_file]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            if verbose:
                print(f"Running LAMMPS with GPU {self.gpu_id}: {input_file}")
        else:
            cmd = [self.lammps_command, '-in', input_file, '-log', log_file]
            env = None
            if verbose:
                print(f"Running LAMMPS (CPU): {input_file}")

        if verbose:
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Log: {log_file}")

        start_time = time.time()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            # Monitor progress if verbose
            if verbose:
                self._monitor_progress(process, log_file, start_time)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"\nLAMMPS STDERR:\n{stderr}")
                raise RuntimeError(
                    f"LAMMPS failed with return code {process.returncode}"
                )

            elapsed = time.time() - start_time
            if verbose:
                print(f"\nLAMMPS completed ({elapsed:.1f}s)")

            return process.returncode

        except FileNotFoundError:
            raise RuntimeError(
                f"LAMMPS executable '{self.lammps_command}' not found. "
                "Make sure LAMMPS is installed and in your PATH."
            )

    def _monitor_progress(self, process, log_file: str, start_time: float):
        """Monitor LAMMPS log file for progress updates."""
        last_size = 0
        header_printed = False

        while process.poll() is None:
            time.sleep(0.5)

            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        last_size = f.tell()

                        for line in new_content.split('\n'):
                            # Print phase transitions
                            if any(keyword in line for keyword in
                                   ["Minimization", "Equilibration", "Production"]):
                                print(f"\n  {line.strip()}")

                            # Print thermo header
                            if line.strip().startswith("Step ") and not header_printed:
                                print(f"  {line.strip()}")
                                header_printed = True

                            # Print thermo output
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    step = int(parts[0])
                                    temp = float(parts[1])
                                    elapsed = time.time() - start_time
                                    print(f"  Step {step:>6} | T={temp:>7.2f}K | "
                                          f"elapsed: {elapsed:>6.1f}s", flush=True, end='\r')
                                except (ValueError, IndexError):
                                    pass
                except Exception:
                    pass

    def read_rdf(
        self,
        filename: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Read LAMMPS RDF output from 'fix ave/time' command.

        Expected format (from LAMMPS 'fix ave/time'):
        timestep n_bins
        bin  r  g_total  coord_total  [g_11  coord_11  g_12  coord_12  ...]
        ...

        Parameters
        ----------
        filename : str
            RDF output file from LAMMPS

        Returns
        -------
        r : np.ndarray
            Distance array (Å)
        g_total : np.ndarray
            Total g(r)
        g_partial : dict or None
            Partial g(r) functions {'GaGa': array, 'GaIn': array, 'InIn': array}
            Returns None if partial RDFs are not available

        Raises
        ------
        FileNotFoundError
            If RDF file does not exist
        ValueError
            If file format is invalid
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"RDF file not found: {filename}")

        all_blocks = []
        current_block = []
        last_timestep = None

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.split()

                # Timestep header: "timestep n_bins"
                if len(parts) == 2:
                    try:
                        timestep = int(parts[0])
                        n_bins = int(parts[1])
                        if current_block:
                            all_blocks.append((last_timestep, np.array(current_block)))
                        current_block = []
                        last_timestep = timestep
                        continue
                    except ValueError:
                        pass

                # Data row
                if len(parts) >= 3:
                    try:
                        current_block.append([float(x) for x in parts])
                    except ValueError:
                        continue

        # Add last block
        if current_block:
            all_blocks.append((last_timestep, np.array(current_block)))

        if len(all_blocks) == 0:
            raise ValueError(f"No valid RDF data found in {filename}")

        # Use last timestep (most recent data)
        last_timestep, data = all_blocks[-1]
        print(f"  Read RDF from timestep {last_timestep} ({len(data)} bins)")

        # Extract columns
        # Column 0: bin index
        # Column 1: r
        # Column 2: g_total
        # Column 3: coord_total
        # Column 4+: optional partial g(r) and coordination numbers
        r = data[:, 1]
        g_total = data[:, 2]

        # Check for partial g(r)
        g_partial = None
        if data.shape[1] >= 8:
            g_partial = {}
            g_partial['GaGa'] = data[:, 4]  # g_11 (type 1-1)
            g_partial['GaIn'] = data[:, 6]  # g_12 (type 1-2)
            if data.shape[1] >= 10:
                g_partial['InIn'] = data[:, 8]  # g_22 (type 2-2)

        return r, g_total, g_partial


def run_lammps_simulation(
    input_file: str,
    log_file: str = 'lammps.log',
    use_gpu: bool = False,
    gpu_id: int = 0,
    verbose: bool = True
) -> int:
    """
    Convenience function to run a LAMMPS simulation.

    Parameters
    ----------
    input_file : str
        LAMMPS input script
    log_file : str, optional
        Log file path (default: 'lammps.log')
    use_gpu : bool, optional
        Use GPU acceleration (default: False)
    gpu_id : int, optional
        GPU device ID (default: 0)
    verbose : bool, optional
        Print progress (default: True)

    Returns
    -------
    returncode : int
        LAMMPS return code (0 = success)
    """
    interface = LAMMPSInterface(use_gpu=use_gpu, gpu_id=gpu_id)
    return interface.run(input_file, log_file, verbose)
