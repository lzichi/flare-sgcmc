from ase.calculators.calculator import Calculator, all_changes
import ase, os, numpy as np, sys
from typing import Union, Optional, Callable, Any, List
from flare.bffs.sgp._C_flare import Structure, SparseGP
from flare.bffs.sgp.sparse_gp import optimize_hyperparameters
from flare.bffs.sgp.calculator import sort_variances
import logging
import time
from ase import Atoms
import numpy as np
from lammps import lammps
from ase.io import read

def transform_stress(stress: List[List[float]]) -> List[List[float]]:
    return -np.array(
        [
            stress[(0, 0)],
            stress[(0, 1)],
            stress[(0, 2)],
            stress[(1, 1)],
            stress[(1, 2)],
            stress[(2, 2)],
        ]
    )


class FlareOTF(Calculator):

    implemented_properties = ['energy', 'forces', 'stress']
    
    """
    FLARE on-the-fly training with an ASE Calculator. Based on LMPOTF.

    Parameters
    ----------
    sparse_gp
        The :cpp:class:`SparseGP` object to train.
    descriptors
        A list of descriptor objects, or a single descriptor (most common), e.g. :cpp:class:`B2`.
    rcut
        The interaction cut-off radius.
    type2number
        FLARE index of ASE atomic numbers e.g. box of C, ase type 6 FLARE type 0 type2number = [0]
    dftcalc
        An ASE calculator, e.g. Espresso.
    energy_correction
        Per-type corrections to the DFT potential energy.
    dft_call_threshold
        Uncertainty threshold for whether to call DFT.
    dft_add_threshold
        Uncertainty threshold for whether to add an atom to the training set.
    dft_xyz_fname
        Name of the file in which to save the DFT results.
        Should contain '*', which will be replaced with the current step.
    std_xyz_fname
        Name of the file in which to save ASE Atoms with per-atom uncertainties as charges.
        Should contain '*', which will be replaced with the current step.
    model_fname
        Name of the saved model, must correspond to `pair_coeff`.
    hyperparameter_optimization
        Boolean function that determines whether to run hyperparameter optimization, as a function of this LMPOTF
        object, the LAMMPS instance and the current step.
    opt_bounds
        Bounds for the hyperparameter optimization.
    opt_method
        Algorithm for the hyperparameter optimization.
    opt_iterations
        Max number of iterations for the hyperparameter optimization.
    post_dft_callback
        A function that is called after every DFT call. Receives this LMPOTF object and the current step.
    wandb
        The wandb object, which should already be initialized.
    log_fname
        An output file to which logging info is written.
    """

    def __init__(
        self,
        sparse_gp: SparseGP,
        descriptors: List,
        rcut: float,
        type2number,
        type2numberase,
        dftcalc: object,
        energy_correction: Union[(float, List[float])] = 0.0,
        force_training=True,
        energy_training=True,
        stress_training=True,
        dft_call_threshold: float = 0.005,
        dft_add_threshold: float = 0.0025,
        dft_xyz_fname: Optional[str] = None,
        std_xyz_fname: Optional[str] = None,
        model_fname: str = "otf.flare",
        hyperparameter_optimization: Callable[
            (["FlareOTF", object, int], bool)
        ] = lambda flareotf,  step: False,
        opt_bounds: Optional[List[float]] = None,
        opt_method: Optional[str] = "L-BFGS-B",
        opt_iterations: Optional[int] = 50,
        wandb: object = None,
        log_fname: str = "otf.log",
    ) -> object:
        """

        """
        # ASE Calculator
        Calculator.__init__(self)
        self.results = {}

        self.sparse_gp = sparse_gp
        self.descriptors = np.atleast_1d(descriptors)
        self.rcut = rcut
        self.type2number = np.atleast_1d(type2number)
        self.type2numberase = type2numberase
        self.ntypes = len(self.type2number)
        self.energy_correction = np.atleast_1d(energy_correction)
        assert len(self.energy_correction) == self.ntypes
        self.dftcalc = dftcalc
        self.dft_call_threshold = dft_call_threshold
        self.dft_add_threshold = dft_add_threshold
        self.force_training = force_training
        self.energy_training = energy_training
        self.stress_training = stress_training
        self.dft_calls = 0
        self.last_dft_call = -100
        self.dft_xyz_fname = dft_xyz_fname
        self.std_xyz_fname = std_xyz_fname
        self.model_fname = model_fname
        self.hyperparameter_optimization = hyperparameter_optimization
        self.opt_bounds = opt_bounds
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.wandb = wandb
        logging.basicConfig(
            filename=log_fname, level=(logging.DEBUG), format="%(asctime)s: %(message)s"
        )
        self.logger = logging.getLogger("flareotf")

        self.time_dft = 0.0
        self.time_hyp_opt = 0.0
        self.time_training = 0.0
        self.time_predict_uncertainties = 0.0
        self.time_prediction = 0.0
        self.t0 = time.time()
        
        self.dft_call = 0
        self.call = 0

    def save(self, fname):
        self.sparse_gp.write_mapping_coefficients(fname, "FlareOTF", 0)

    def main_step(
            self,
            cell,
            x,
            types_ase,
            types_flare,
            step,
            structure,
            natoms
        ):
        E, F, S = None, None, None
        if self.dft_calls == 0:
            self.logger.info("Initial step, calling DFT")
            E, F, S = self.run_dft(cell, x, types_ase, types_flare, step, structure)
            t0 = time.time()
            self.sparse_gp.add_training_structure(structure)
            self.sparse_gp.add_random_environments(structure, [int(natoms/4)])
            self.sparse_gp.update_matrices_QR()
            self.time_training += time.time() - t0
            self.save(self.model_fname)

        else:
            self.logger.info(f"Step {step}")
            sigma = self.sparse_gp.hyperparameters[0]
            t0 = time.time()
            variances = sort_variances(structure, self.sparse_gp.compute_cluster_uncertainties(structure)[0])
            self.time_predict_uncertainties += time.time() - t0
            stds = np.sqrt(np.abs(variances)) / sigma
            if self.std_xyz_fname is not None:
                frame = ase.Atoms(
                    positions=x,
                    numbers=types_ase,
                    cell=cell,
                    pbc=True,
                )
                frame.set_array("charges", stds)
                
                ase.io.write(self.std_xyz_fname.replace("*", str(step)), frame, format="extxyz")
            wandb_log = {"max_uncertainty": np.amax(stds)}
            self.logger.info(f"Max uncertainty: {np.amax(stds)}")
            call_dft = np.any(stds > self.dft_call_threshold)
            if call_dft:
                t0 = time.time()
                self.sparse_gp.predict_DTC(structure)
                self.time_prediction += time.time() - t0
                predE = structure.mean_efs[0]
                predF = structure.mean_efs[1:-6].reshape((-1, 3))
                predS = structure.mean_efs[-6:]
                Fstd = np.sqrt(np.abs(structure.variance_efs[1:-6])).reshape(
                    (-1, 3)
                )
                Estd = np.sqrt(np.abs(structure.variance_efs[0]))
                Sstd = np.sqrt(np.abs(structure.variance_efs[-6:]))
                wandb_log["max_F_uncertainty"] = np.amax(Fstd)
                self.logger.info(f"Max force uncertainty: {np.amax(Fstd)}")
                self.logger.info(f"DFT call #{self.dft_calls}")
                E, F, S = self.run_dft(cell, x, types, step, structure)
                atoms_to_be_added = np.arange(natoms)[stds > self.dft_add_threshold]
                t0 = time.time()
                self.sparse_gp.add_training_structure(structure)
                self.sparse_gp.add_specific_environments(
                    structure, atoms_to_be_added
                )
                self.sparse_gp.update_matrices_QR()
                self.time_training += time.time() - t0
                if self.hyperparameter_optimization(self, step):
                    self.logger.info("Optimizing hyperparameters!")
                    self.sparse_gp.compute_likelihood_stable()
                    likelihood_before = self.sparse_gp.log_marginal_likelihood
                    t0 = time.time()
                    optimize_hyperparameters(
                        (self.sparse_gp),
                        bounds=(self.opt_bounds),
                        method=(self.opt_method),
                        max_iterations=(self.opt_iterations),
                    )
                    self.time_hyp_opt += time.time() - t0
                    likelihood_after = self.sparse_gp.log_marginal_likelihood
                    self.logger.info(
                        f"Likelihood before/after: {likelihood_before:.2e} {likelihood_after:.2e}"
                    )
                    self.logger.info(
                        f"Likelihood gradient: {self.sparse_gp.likelihood_gradient}"
                    )
                    self.logger.info(
                        f"Hyperparameters: {self.sparse_gp.hyperparameters}"
                    )
                self.save(self.model_fname)
        return E, F, S
            
    def calculate(
            self, 
            atoms: Atoms = None,
            properties=None,
            system_changes=all_changes
        ) -> None:
        """
        """
        Calculator.calculate(self, atoms)
        self.call += 1

        try:
            natoms = len(atoms)
            x = atoms.get_positions()
            cell = atoms.get_cell()
            types = atoms.numbers
            step = self.call
            types_ase = types
            types_flare = np.vectorize(self.type2numberase.get)(types)
            structure = Structure(cell, types_flare, x, self.rcut, self.descriptors) # TODO: why subtract by 1
            
            E, F, S = self.main_step(cell, x, types_ase, types_flare, step, structure, natoms)
            if(E is None):
                # no call to DFT
                self.sparse_gp.predict_DTC(structure)
                E = structure.mean_efs[0]
                F = structure.mean_efs[1:-6].reshape((-1, 3))
                S = structure.mean_efs[-6:]

            # Store results in ASE format
            self.results = {
                "energy": E,
                "forces": F,
                "stress": S,
            }

        except Exception as err:
            try:
                self.logger.exception("LMPOTF ERROR")
                raise err
            finally:
                err = None
                del err

    def step(
            self,
            lmpptr,
            evflag=0
        ):
        try:
            lmp = lammps(ptr=lmpptr)
            natoms = lmp.get_natoms()
            x = lmp.gather_atoms("x", 1, 3)
            x = np.ctypeslib.as_array(x, shape=(natoms, 3)).reshape(natoms, 3)
            step = int(lmp.get_thermo("step"))
            boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
            cell = np.diag(np.array(boxhi) - np.array(boxlo))
            cell[(1, 0)] = xy
            cell[(2, 0)] = xz
            cell[(2, 1)] = yz
            types = lmp.gather_atoms("type", 0, 1)
            types = np.ctypeslib.as_array(types, shape=natoms)
            types_flare = types - 1
            types_ase = self.type2number[types - 1]
            structure = Structure(cell, types_flare, x, self.rcut, self.descriptors)
            E, F, S = self.main_step(cell, x, types_ase, types_flare, step, structure, natoms)

            if(E is not None and self.dft_calls != 1):
                # called DFT
                # for the first step need to not call lammps or 
                # Exception: ERROR: Pair_coeff command without a pair style (src/input.cpp:1745)
                # Last input line: pair_coeff * * NiH2O.otf.flare
                lmp.command(f"pair_coeff * * {self.model_fname}")
        
        except Exception as err:
            try:
                self.logger.exception("LMPOTF ERROR")
                raise err
            finally:
                err = None
                del err


    def run_dft(self, cell, x, types_ase, types_flare, step, structure):
        t0 = time.time()
        frame = ase.Atoms(
            positions=x,
            numbers=types_ase,
            cell=cell,
            calculator=(self.dftcalc),
            pbc=True,
        )
        E = frame.get_potential_energy()
        E -= np.sum(self.energy_correction[types_flare])
        F = frame.get_forces()
        S = frame.get_stress(voigt=False)
        if self.dft_xyz_fname is not None:
            ase.io.write(self.dft_xyz_fname.replace("*", str(step)), frame, format="extxyz")
        if self.force_training:
            structure.forces = F.reshape(-1)
        if self.energy_training:
            structure.energy = np.array([E])
        if self.stress_training:
            structure.stresses = transform_stress(S)
        self.dft_calls += 1
        self.last_dft_call = step

        self.time_dft += time.time() - t0
        return (E, F, S)
    
    def offline_train(
            self,
            input_frames: str,
            typeMapping
        ):

        atoms_frames = read(input_frames, ":")
        atoms = atoms_frames[0]
        # treat first frame like first DFT call
        self.logger.info(f"[offline training] Frame 0")
        natoms = len(atoms)
        x = atoms.get_positions()
        cell = atoms.get_cell()
        types = atoms.numbers

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress(voigt=False)

        structure = Structure(cell, np.vectorize(typeMapping.get)(types), x, self.rcut, self.descriptors) 
        
        structure.forces = F.reshape(-1)
        structure.energy = np.array([E])
        structure.stresses = transform_stress(S)
        
        self.sparse_gp.add_training_structure(structure)
        self.sparse_gp.add_random_environments(structure, [int(natoms/4)])
        self.sparse_gp.update_matrices_QR()

        # go through all remaining frames
        for idx, atoms in enumerate(atoms_frames[1:]):
            natoms = len(atoms)
            x = atoms.get_positions()
            cell = atoms.get_cell()
            types = atoms.numbers
            structure = Structure(cell, np.vectorize(typeMapping.get)(types), x, self.rcut, self.descriptors) 

            self.logger.info(f"[offline training] Frame {idx + 1}")
            E = atoms.get_potential_energy()
            F = atoms.get_forces()
            S = atoms.get_stress(voigt=False)

            structure.forces = F.reshape(-1)
            structure.energy = np.array([E])
            structure.stresses = transform_stress(S)

            sigma = self.sparse_gp.hyperparameters[0]
            variances = sort_variances(structure, self.sparse_gp.compute_cluster_uncertainties(structure)[0])
            stds = np.sqrt(np.abs(variances)) / sigma
            atoms_to_be_added = np.arange(natoms)[stds > self.dft_add_threshold]

            t0 = time.time()
            self.sparse_gp.add_training_structure(structure)
            self.sparse_gp.add_specific_environments(
                        structure, atoms_to_be_added
                    )
            self.sparse_gp.update_matrices_QR()
            self.time_training += time.time() - t0

        # save the model
        self.save(self.model_fname)

