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
        dftcalc: object,
        energy_correction: Union[(float, List[float])] = 0.0,
        force_training=True,
        energy_training=True,
        stress_training=True,
        dft_call_threshold: float = 0.005,
        dft_add_threshold: float = 0.0025,
        dft_xyz_fname: Optional[str] = None,
        std_xyz_fname: Optional[str] = None,
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
        self.type2number = type2number
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
        self.hyperparameter_optimization = hyperparameter_optimization
        self.opt_bounds = opt_bounds
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.wandb = wandb
        logging.basicConfig(
            filename=log_fname, level=(logging.DEBUG), format="%(asctime)s: %(message)s"
        )
        self.logger = logging.getLogger("lmpotf")

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
            structure = Structure(cell, np.vectorize(self.type2number.get)(types), x, self.rcut, self.descriptors) # TODO: why subtract by 1
            
            if self.dft_calls == 0:
                self.logger.info("Initial step, calling DFT")
                E, F, S = self.run_dft(cell, x, types, step, structure)
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
                        numbers=types,
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
                    wandb_log["Fmae"] = np.mean(np.abs(F - predF))
                    wandb_log["Emae"] = np.abs(pe - predE) / natoms
                    wandb_log["n_added"] = len(atoms_to_be_added)
                    for qty in ("n_added", "Fmae", "Emae"):
                        self.logger.info(f"{qty}: {wandb_log[qty]}")

                else:
                    # no call to DFT
                    self.sparse_gp.predict_DTC(structure)
                    E = structure.mean_efs[0]
                    F = structure.mean_efs[1:-6].reshape((-1, 3))
                    S = structure.mean_efs[-6:]

                if self.wandb is not None:
                    wandb_log["uncertainties"] = self.wandb.Histogram(stds)
                    wandb_log["time_dft"] = self.time_dft
                    wandb_log["time_training"] = self.time_training
                    wandb_log["time_prediction"] = self.time_prediction
                    wandb_log["time_predict_uncertainties"] = self.time_predict_uncertainties
                    wandb_log["time_hyp_opt"] = self.time_hyp_opt
                    wandb_log["time_lammps"] = self.time_lammps
                    if call_dft:
                        wandb_log["Funcertainties"] = self.wandb.Histogram(Fstd.ravel())
                        wandb_log["Ferror"] = self.wandb.Histogram(
                            np.abs(F - predF).ravel()
                        )
                        wandb_log["logrelFerror"] = self.wandb.Histogram(
                            np.log10(np.abs(F - predF)/np.abs(F)).ravel()
                        )
                    self.wandb.log(wandb_log, step=step)
                    self.t0 = time.time()

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

    def run_dft(self, cell, x, types, step, structure):
        t0 = time.time()
        frame = ase.Atoms(
            positions=x,
            numbers=types,
            cell=cell,
            calculator=(self.dftcalc),
            pbc=True,
        )
        E = frame.get_potential_energy()
        E -= np.sum(self.energy_correction[np.vectorize(self.type2number.get)(types)])
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

