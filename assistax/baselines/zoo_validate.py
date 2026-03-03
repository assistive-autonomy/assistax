"""Validate zoo agent parameter dimensions against observation dimensions.

Loads each zoo agent's safetensors params and config, then checks whether
the first Dense layer kernel input dimension matches the saved OBS_DIM and
the expected base environment observation dimension (with/without the +7
preference observation expansion).

Usage:
    # Validate all feeding agents
    python zoo_validate.py --zoo-path /pvc/assistax/zoo --scenario feeding

    # Validate with explicit expected base obs dim
    python zoo_validate.py --zoo-path /pvc/assistax/zoo --scenario feeding --expected-obs-dim 42
"""

import argparse
import os.path as osp
from typing import Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import OmegaConf
from safetensors import safe_open


PREF_OBS_DIM = 7


def get_first_dense_kernel_shape(params_path: str) -> Optional[Tuple[int, ...]]:
    """Return the shape of the first Dense_0 kernel from a safetensors file."""
    with safe_open(params_path, framework="numpy") as f:
        keys = list(f.keys())
        # Primary key (all current architectures use flat Dense_0)
        primary = "params/Dense_0/kernel"
        if primary in keys:
            return f.get_tensor(primary).shape
        # Fallback: any key containing Dense_0/kernel
        for k in sorted(keys):
            if "Dense_0/kernel" in k:
                return f.get_tensor(k).shape
    return None


def get_base_env_obs_dim(scenario: str) -> Optional[int]:
    """Get the base observation dim for a scenario via assistax.make."""
    try:
        from assistax.envs.registration import make
        from assistax.wrappers.baselines import get_space_dim

        env = make(scenario)
        for agent, space in env.observation_spaces.items():
            if agent != "global":
                return int(get_space_dim(space))
    except Exception as e:
        print(f"  [warn] Could not get base env obs dim via assistax.make: {e}")
    return None


def validate_agents(
    zoo_path: str, scenario: str, expected_obs_dim: Optional[int] = None
) -> Tuple[List[Dict], Optional[int]]:
    """Validate parameter dims for each agent in *scenario*.

    Returns:
        (results_list, base_obs_dim_used)
    """
    index_path = osp.join(zoo_path, "index.csv")
    index = pd.read_csv(index_path)
    scenario_agents = index.query(f'scenario == "{scenario}"')

    if expected_obs_dim is not None:
        base_obs_dim = expected_obs_dim
    else:
        base_obs_dim = get_base_env_obs_dim(scenario)

    results: List[Dict] = []
    for _, row in scenario_agents.iterrows():
        agent_uuid = str(row["agent_uuid"])
        algorithm = str(row.get("algorithm", "unknown"))
        is_rnn = row.get("is_rnn", False)
        scenario_agent_id = str(row.get("scenario_agent_id", "unknown"))

        config_path = osp.join(zoo_path, "config", f"{agent_uuid}.yaml")
        if not osp.exists(config_path):
            results.append(_result(
                agent_uuid, algorithm, is_rnn, scenario_agent_id,
                status="MISSING_CONFIG",
            ))
            continue

        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        config_obs_dim = cfg.get("OBS_DIM")
        has_pref = cfg.get("ENV_KWARGS", {}).get("preference_rewards") is not None

        params_path = osp.join(zoo_path, "params", f"{agent_uuid}.safetensors")
        if not osp.exists(params_path):
            results.append(_result(
                agent_uuid, algorithm, is_rnn, scenario_agent_id,
                status="MISSING_PARAMS",
                config_obs_dim=config_obs_dim, has_pref_config=has_pref,
            ))
            continue

        kernel_shape = get_first_dense_kernel_shape(params_path)
        if kernel_shape is None:
            results.append(_result(
                agent_uuid, algorithm, is_rnn, scenario_agent_id,
                status="NO_DENSE_KERNEL",
                config_obs_dim=config_obs_dim, has_pref_config=has_pref,
            ))
            continue

        kernel_input_dim = kernel_shape[0]

        issues: List[str] = []
        if config_obs_dim is not None and kernel_input_dim != config_obs_dim:
            issues.append(
                f"kernel_dim({kernel_input_dim}) != config_OBS_DIM({config_obs_dim})"
            )

        if base_obs_dim is not None and has_pref:
            expected_with_pref = base_obs_dim + PREF_OBS_DIM
            if kernel_input_dim != expected_with_pref:
                issues.append(
                    f"has pref_config but kernel_dim({kernel_input_dim}) "
                    f"!= base+7({expected_with_pref})"
                )

        status = "MISMATCH: " + "; ".join(issues) if issues else "OK"
        results.append(_result(
            agent_uuid, algorithm, is_rnn, scenario_agent_id,
            status=status, config_obs_dim=config_obs_dim,
            kernel_input_dim=kernel_input_dim, has_pref_config=has_pref,
        ))

    return results, base_obs_dim


def _result(
    agent_uuid: str,
    algorithm: str,
    is_rnn: bool,
    scenario_agent_id: str,
    status: str,
    config_obs_dim: Optional[int] = None,
    kernel_input_dim: Optional[int] = None,
    has_pref_config: Optional[bool] = None,
) -> Dict:
    """Build a single result dict."""
    return {
        "agent_uuid": agent_uuid,
        "algorithm": algorithm,
        "is_rnn": is_rnn,
        "scenario_agent_id": scenario_agent_id,
        "status": status,
        "config_obs_dim": config_obs_dim,
        "kernel_input_dim": kernel_input_dim,
        "has_pref_config": has_pref_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate zoo agent parameter dimensions against observation dimensions."
    )
    parser.add_argument("--zoo-path", required=True, help="Path to the zoo directory")
    parser.add_argument(
        "--scenario", default="feeding",
        help="Scenario/env name to filter (default: feeding)",
    )
    parser.add_argument(
        "--expected-obs-dim", type=int, default=None,
        help="Expected base observation dim (skips assistax.make lookup if provided)",
    )
    args = parser.parse_args()

    print(f"Zoo path         : {args.zoo_path}")
    print(f"Scenario         : {args.scenario}")
    print(f"Expected obs dim : {args.expected_obs_dim or '(auto-detect)'}\n")

    results, base_obs_dim = validate_agents(
        args.zoo_path, args.scenario, args.expected_obs_dim
    )

    if not results:
        print(f"No agents found for scenario '{args.scenario}'.")
        return

    if base_obs_dim is not None:
        print(f"Base obs dim     : {base_obs_dim}")
        print(f"With pref (+7)   : {base_obs_dim + PREF_OBS_DIM}\n")
    else:
        print("Base obs dim     : unknown (provide --expected-obs-dim)\n")

    ok_count = 0
    for r in results:
        is_ok = r["status"] == "OK"
        ok_count += int(is_ok)
        marker = "OK" if is_ok else "!!"
        print(
            f"  [{marker}] {r['agent_uuid'][:12]}...  "
            f"alg={r['algorithm']:<6} rnn={str(r['is_rnn']):<5} "
            f"agent={r['scenario_agent_id']:<6} "
            f"config_obs={str(r['config_obs_dim']):<4} "
            f"kernel_in={str(r['kernel_input_dim']):<4} "
            f"pref={r['has_pref_config']}"
        )
        if not is_ok:
            print(f"       -> {r['status']}")

    issue_count = len(results) - ok_count
    print(f"\nSummary: {ok_count} OK, {issue_count} issue(s) out of {len(results)} agents")


if __name__ == "__main__":
    main()
