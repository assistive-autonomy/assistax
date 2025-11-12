import jax
from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp
import wandb
import os 


def upload_eval_data_to_wandb(eval_data, config, suffix=""):
    """Upload evaluation data to wandb as artifacts (compact: NaNs removed from storage)."""
    if not eval_data or not config.get("UPLOAD_EVAL_DATA", True):
        return

    print("Uploading evaluation data to wandb (compact)…")
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        for key, data in eval_data.items():
            if isinstance(data, (np.ndarray, jnp.ndarray)):
                file_path = os.path.join(temp_dir, f"{key}.npz")  # use .npz (compressed bundle)
                _save_compact_npz(file_path, data)
                print(f"Saved {key} compactly with shape {np.asarray(data).shape}")
            else:
                # For non-arrays, fall back to a small npy (or skip)
                file_path = os.path.join(temp_dir, f"{key}.npy")
                np.save(file_path, np.array(data, dtype=object))
                print(f"Saved non-array {key} as object npy")

        artifact = wandb.Artifact("evaluation_data" + suffix, type="dataset")
        artifact.add_dir(temp_dir)
        wandb.log_artifact(artifact)

    print("Evaluation data uploaded to wandb successfully!")

def save_checkpoint_callback(train_states_and_step):
    """Saves the train_state to a file using Orbax."""
    train_states, update_step = train_states_and_step
    params_to_save = train_states.params
    
    # Create a directory for the checkpoint
    ckpt_dir = os.path.join(wandb.run.dir, "checkpoints", f"update_{int(update_step)}")
    
    # Set up the Orbax checkpointer
    checkpointer = ocp.StandardCheckpointer()
    
    save_args = orbax_utils.save_args_from_target(params_to_save)
    # Save the train_states Pytree
    checkpointer.save(ckpt_dir, params_to_save, save_args=save_args, force=True)
    print(f"--- Saved checkpoint at step {update_step} to {ckpt_dir} ---")
