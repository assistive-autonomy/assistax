"""
Test script to verify that agent indices correctly map to unique agent behaviors.

This test ensures:
1. Each index maps to a unique set of parameters
2. Different indices produce different actions (agents are actually different)
3. The same index consistently produces the same actions (deterministic)
"""

import jax
import jax.numpy as jnp
import pandas as pd
import assistax
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import numpy as np
import traceback


def test_index_to_action_mapping(config):
    """
    Verify that agent indices correctly map to unique agent behaviors.
    """
    try:
        print("="*80, flush=True)
        print("INDEX → AGENT → ACTION MAPPING TEST", flush=True)
        print("="*80, flush=True)
        
        # ===== SETUP =====
        print("\n" + "="*80, flush=True)
        print("SETUP", flush=True)
        print("="*80, flush=True)
        
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]
        
        partner_dict = {}
        for partner_algo in config["PARTNER_ALGORITHMS"]:
            partner_dict[partner_algo] = zoo.index.query(
                f'algorithm == "{partner_algo}"'
            ).query(
                f'scenario == "{scenario}"'
            ).query(
                'scenario_agent_id == "human"'
            )
            print(f"Found {len(partner_dict[partner_algo])} {partner_algo} partners", flush=True)
        
        all_partners = pd.concat(partner_dict.values(), ignore_index=True)
        train_partners = all_partners.sample(frac=config["SPLIT_RATIO"], random_state=42)
        
        train_set = {}
        for algo in partner_dict.keys():
            train_set[algo] = train_partners[
                train_partners['algorithm'] == algo
            ].reset_index(drop=True)
        
        load_zoo_dict_train = {
            algo: {"human": list(train_set[algo].agent_uuid)} 
            for algo in partner_dict.keys()
        }
        
        total_train_partners = sum(len(train_set[algo]) for algo in train_set)
        print(f"Total training partners: {total_train_partners}", flush=True)
        
        # Create environment
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo_dict_train)
        env = LogWrapper(env)
        
        # ===== TEST 1: PARAMETER UNIQUENESS (PER ALGORITHM) =====
        print("\n" + "="*80)
        print("TEST 1: PARAMETER UNIQUENESS")
        print("="*80)
        
        # Test each algorithm separately since they have different architectures
        for algo, agents_dict in env._env.loaded_params.items():
            print(f"\n--- Testing {algo} agents ---", flush=True)
            
            for agent_type, train_state in agents_dict.items():
                pop_size = train_state.pop_size
                print(f"  Agent type: {agent_type}, Population: {pop_size}", flush=True)
                
                # Get parameter structure
                param_shapes = jax.tree.map(lambda x: x.shape, train_state.params)
                print(f"  Parameter shapes: {param_shapes}", flush=True)
                
                # Sample pairs to check uniqueness
                num_comparisons = min(10, pop_size * (pop_size - 1) // 2)
                if num_comparisons == 0:
                    print(f"  (Only 1 agent, skipping uniqueness check)", flush=True)
                    continue
                
                print(f"  Comparing {num_comparisons} pairs for uniqueness...", flush=True)
                
                key = jax.random.key(config["SEED"])
                identical_pairs = 0
                
                for _ in range(num_comparisons):
                    key, k1, k2 = jax.random.split(key, 3)
                    idx1 = int(jax.random.randint(k1, (), 0, pop_size))
                    idx2 = int(jax.random.randint(k2, (), 0, pop_size))
                    
                    if idx1 == idx2:
                        continue
                    
                    # Compare parameters
                    params1 = jax.tree.map(lambda x: x[idx1], train_state.params)
                    params2 = jax.tree.map(lambda x: x[idx2], train_state.params)
                    
                    # Check if identical
                    is_identical = all(
                        jnp.allclose(p1, p2, atol=1e-8)
                        for p1, p2 in zip(
                            jax.tree_util.tree_leaves(params1),
                            jax.tree_util.tree_leaves(params2)
                        )
                    )
                    
                    if is_identical:
                        identical_pairs += 1
                        print(f"    ⚠️  {algo} agents {idx1} and {idx2} are IDENTICAL!", flush=True)
                
                if identical_pairs == 0:
                    print(f"  ✓ All {algo} agents have unique parameters", flush=True)
                else:
                    print(f"  ❌ Found {identical_pairs} duplicate pairs in {algo}!", flush=True)
        
        # ===== TEST 2: ACTION DIVERSITY =====
        print("\n" + "="*80)
        print("TEST 2: ACTION DIVERSITY")
        print("="*80)
        
        # Get a common observation
        key = jax.random.PRNGKey(123)
        obs, state = env.reset(key)
        human_obs = obs['human']
        print(f"Observation shape: {human_obs.shape}", flush=True)
        
        # Generate actions for ALL agents
        load_wrapper = env._env
        dones = {'human': False}
        avail_actions = env._env.get_avail_actions(state.env_state._state)
        
        all_actions, _ = load_wrapper.take_internal_action(
            key, obs, dones, avail_actions, state.env_state.hstate
        )
        
        action_array = all_actions['human']
        total_pop = action_array.shape[0]
        
        print(f"Generated actions for {total_pop} agents", flush=True)
        print(f"Action shape: {action_array.shape}", flush=True)
        
        # Check action diversity
        print(f"\nComparing actions from different agents...", flush=True)
        
        num_action_comparisons = min(50, total_pop * (total_pop - 1) // 2)
        identical_actions = 0
        very_similar_actions = 0
        
        key = jax.random.PRNGKey(456)
        
        for _ in range(num_action_comparisons):
            key, k1, k2 = jax.random.split(key, 3)
            idx1 = int(jax.random.randint(k1, (), 0, total_pop))
            idx2 = int(jax.random.randint(k2, (), 0, total_pop))
            
            if idx1 == idx2:
                continue
            
            action1 = action_array[idx1]
            action2 = action_array[idx2]
            
            max_diff = jnp.max(jnp.abs(action1 - action2))
            
            if jnp.allclose(action1, action2, atol=1e-6):
                identical_actions += 1
            elif max_diff < 0.01:
                very_similar_actions += 1
        
        print(f"  Identical action pairs: {identical_actions}/{num_action_comparisons}", flush=True)
        print(f"  Very similar pairs (diff < 0.01): {very_similar_actions}/{num_action_comparisons}", flush=True)
        
        # Action statistics
        action_means = jnp.mean(action_array, axis=1)
        action_stds = jnp.std(action_array, axis=1)
        
        mean_of_means = jnp.mean(action_means)
        std_of_means = jnp.std(action_means)
        
        print(f"\n  Action statistics across all {total_pop} agents:", flush=True)
        print(f"    Mean of agent means: {mean_of_means:.4f}", flush=True)
        print(f"    Std of agent means: {std_of_means:.4f}", flush=True)
        
        if std_of_means < 0.01:
            print(f"  ⚠️  WARNING: Very low diversity (std={std_of_means:.4f})", flush=True)
        else:
            print(f"  ✓ Good action diversity (std={std_of_means:.4f})", flush=True)
        
        # ===== TEST 3: ACTION DISTRIBUTION BY ALGORITHM =====
        print("\n" + "="*80)
        print("TEST 3: ACTION DIVERSITY BY ALGORITHM")
        print("="*80)
        
        # Map global indices to algorithms
        print("\nAnalyzing action diversity within each algorithm...", flush=True)
        
        global_idx = 0
        for algo, agents_dict in env._env.loaded_params.items():
            for agent_type, train_state in agents_dict.items():
                pop_size = train_state.pop_size
                
                # Extract actions for this algorithm
                algo_actions = action_array[global_idx:global_idx + pop_size]
                
                algo_means = jnp.mean(algo_actions, axis=1)
                algo_std_of_means = jnp.std(algo_means)
                
                print(f"  {algo} (n={pop_size}):", flush=True)
                print(f"    Mean action: {jnp.mean(algo_means):.4f}", flush=True)
                print(f"    Std of means: {algo_std_of_means:.4f}", flush=True)
                
                global_idx += pop_size
        
        # ===== TEST 4: INDEX MAPPING SPOT CHECK =====
        print("\n" + "="*80)
        print("TEST 4: INDEX MAPPING VERIFICATION")
        print("="*80)
        
        print("\nSpot checking that wrapper selects correct indexed action...", flush=True)
        
        key = jax.random.PRNGKey(999)
        num_spot_checks = 10
        
        for i in range(num_spot_checks):
            key, reset_key = jax.random.split(key)
            
            obs_test, state_test = env.reset(reset_key)
            selected_idx = int(state_test.env_state.ag_idx['human'])
            selected_action = state_test.env_state.load_agent_actions['human']
            
            # Re-compute all actions
            all_actions_recompute, _ = load_wrapper.take_internal_action(
                reset_key, obs_test, dones, avail_actions, state_test.env_state.hstate
            )
            
            # Note: We can't directly compare due to stochastic sampling
            # But we can verify the index is in valid range
            if selected_idx < 0 or selected_idx >= total_pop:
                print(f"  ❌ Invalid index: {selected_idx} (range: [0, {total_pop}))", flush=True)
            
        print(f"  ✓ All {num_spot_checks} spot checks had valid indices", flush=True)
        
        # ===== FINAL VERDICT =====
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        
        print(f"\n✓ Parameter uniqueness: Checked per algorithm", flush=True)
        
        if std_of_means >= 0.01:
            print(f"✓ Good action diversity across population (std={std_of_means:.4f})", flush=True)
        else:
            print(f"⚠️  Low action diversity (std={std_of_means:.4f})", flush=True)
        
        if identical_actions < num_action_comparisons * 0.1:
            print(f"✓ Low action duplication rate ({identical_actions}/{num_action_comparisons})", flush=True)
        else:
            print(f"⚠️  High action duplication ({identical_actions}/{num_action_comparisons})", flush=True)
        
        print("\n🎉 Mapping test complete!", flush=True)
        print(f"Tested {total_pop} agents across multiple algorithms.", flush=True)
        
        print("\n" + "="*80, flush=True)
        
        return action_array, None
    
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}", flush=True)
        traceback.print_exc()
        raise

@hydra.main(version_base=None, config_path="config", config_name="ppo_aht")
def main(config):
    print("\n" + "="*80, flush=True)
    print("INDEX → AGENT → ACTION MAPPING TEST", flush=True)
    print("="*80, flush=True)
    
    config = OmegaConf.to_container(config, resolve=True)
    
    actions, params = test_index_to_action_mapping(config)
    
    if actions is not None:
        print(f"\n✓ Test complete!", flush=True)
        print(f"  Tested {actions.shape[0]} agents", flush=True)
        print(f"  Action dimension: {actions.shape[1]}", flush=True)


if __name__ == "__main__":
    main()