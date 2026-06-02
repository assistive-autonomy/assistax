from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import Dict, List
import yaml 

def scan_completed_sweeps(base_dir: Path) -> List[Dict]:
    """
    Scan a directory for completed sweep configurations.
    
    Looks for directories containing both hparams.npy (config) and returns.npy (completion marker).
    
    Returns list of hparam dicts that have completed.
    """
    base_path = Path(base_dir)
    completed = []
    
    if not base_path.exists():
        return completed
    
    for hparams_file in base_path.rglob("hparams.npy"):
        results_file = hparams_file.parent / "returns.npy"
        if results_file.exists():
            hparams = np.load(hparams_file, allow_pickle=True).item()
            completed.append(hparams)
    
    return completed


def config_already_run(config: Dict, completed: list[Dict], expected_sweep: Dict) -> bool:
    """
    Check if a configuration has already been completed.
    
    Uses lr/ent_coef/clip_eps arrays as fingerprints for the seed.
    """
    
    def extract_from_config(cfg):
        return {
            "num_steps": cfg.get("NUM_STEPS"),
            "num_envs": cfg.get("NUM_ENVS"),
            "update_epochs": cfg.get("UPDATE_EPOCHS"),
            "num_minibatches": cfg.get("NUM_MINIBATCHES"),
            "lr_array": expected_sweep["lr"]["val"],
            "ent_coef_array": expected_sweep["ent_coef"]["val"],
            "clip_eps_array": expected_sweep["clip_eps"]["val"],
        }
    
    def extract_from_hparams(hparams):
        return {
            "num_steps": hparams.get("num_steps"),
            "num_envs": hparams.get("num_envs"),
            "update_epochs": hparams.get("update_epochs"),
            "num_minibatches": hparams.get("num_minibatches"),
            "lr_array": hparams.get("lr"),
            "ent_coef_array": hparams.get("ent_coef"),
            "clip_eps_array": hparams.get("clip_eps"),
        }
    
    current = extract_from_config(config)
    
    for c in completed:
        saved = extract_from_hparams(c)
        
        match = True
        for k in current:
            if current[k] is None or saved[k] is None:
                continue
            
            if k.endswith("_array"):
                if not jnp.allclose(current[k], saved[k]):
                    match = False
                    break
            else:
                if current[k] != saved[k]:
                    match = False
                    break
        
        if match:
            return True
    
    return False

def config_already_run_sac(config: Dict, completed: list[Dict], expected_sweep: Dict) -> bool:
    """
    Check if ISAC/MASAC configuration has already been completed.
    
    Uses p_lr/q_lr/alpha_lr/tau arrays as fingerprints for the seed.
    """
    
    def extract_from_config(cfg):
        return {
            "num_updates": cfg.get("NUM_UPDATES"),
            "total_timesteps": cfg.get("TOTAL_TIMESTEPS"),
            "num_envs": cfg.get("NUM_ENVS"),
            "num_sac_updates": cfg.get("NUM_SAC_UPDATES"),
            "batch_size": cfg.get("BATCH_SIZE"),
            "buffer_size": cfg.get("BUFFER_SIZE"),
            "rollout_length": cfg.get("ROLLOUT_LENGTH"),
            "explore_steps": cfg.get("EXPLORE_STEPS"),
            "p_lr_array": expected_sweep["p_lr"]["val"],
            "q_lr_array": expected_sweep["q_lr"]["val"],
            "alpha_lr_array": expected_sweep["alpha_lr"]["val"],
            "tau_array": expected_sweep["tau"]["val"],
        }
    
    def extract_from_hparams(hparams):
        return {
            "num_updates": hparams.get("num_updates"),
            "total_timesteps": hparams.get("total_timesteps"),
            "num_envs": hparams.get("num_envs"),
            "num_sac_updates": hparams.get("num_sac_updates"),
            "batch_size": hparams.get("batch_size"),
            "buffer_size": hparams.get("buffer_size"),
            "rollout_length": hparams.get("rollout_length"),
            "explore_steps": hparams.get("explore_steps"),
            "p_lr_array": hparams.get("p_lr"),
            "q_lr_array": hparams.get("q_lr"),
            "alpha_lr_array": hparams.get("alpha_lr"),
            "tau_array": hparams.get("tau"),
        }
    
    current = extract_from_config(config)
    
    for c in completed:
        saved = extract_from_hparams(c)
        
        match = True
        for k in current:
            if current[k] is None or saved[k] is None:
                continue
            
            if k.endswith("_array"):
                if not jnp.allclose(current[k], saved[k]):
                    match = False
                    break
            else:
                if current[k] != saved[k]:
                    match = False
                    break
        
        if match:
            return True
    
    return False