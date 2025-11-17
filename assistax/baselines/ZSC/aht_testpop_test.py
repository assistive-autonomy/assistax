"""
Test script for verifying agent sampling with TEST population.
"""

import jax
import jax.numpy as jnp
import pandas as pd
from scipy.stats import chisquare
import assistax
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import traceback
import sys
import numpy as np


def test_agent_sampling_test_set(config):
    """
    Test agent sampling with the TEST population (unseen during training).
    """
    try:
        print("="*80, flush=True)
        print("AGENT SAMPLING TEST - TEST POPULATION", flush=True)
        print("="*80, flush=True)
        
        # ===== SETUP =====
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]
        
        # Load partners
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
        
        # Create train/test split
        all_partners = pd.concat(partner_dict.values(), ignore_index=True)
        print(f"Total partners available: {len(all_partners)}", flush=True)
        
        train_partners = all_partners.sample(frac=config["SPLIT_RATIO"], random_state=42)
        test_partners = all_partners.drop(train_partners.index)
        
        # Split into algorithm-specific dictionaries
        train_set = {}
        test_set = {}
        
        for algo in partner_dict.keys():
            train_set[algo] = train_partners[
                train_partners['algorithm'] == algo
            ].reset_index(drop=True)
            test_set[algo] = test_partners[
                test_partners['algorithm'] == algo
            ].reset_index(drop=True)
            print(f"  {algo}: {len(train_set[algo])} train, {len(test_set[algo])} test", flush=True)
        
        # Create zoo loading dictionary for TEST set
        load_zoo_dict_test = {
            algo: {"human": list(test_set[algo].agent_uuid)} 
            for algo in partner_dict.keys()
        }
        
        total_train_partners = sum(len(train_set[algo]) for algo in train_set)
        total_test_partners = sum(len(test_set[algo]) for algo in test_set)
        
        print(f"\nTotal TRAIN partners: {total_train_partners}", flush=True)
        print(f"Total TEST partners: {total_test_partners}", flush=True)
        print("\n⚠️  NOTE: Testing with TEST population (unseen agents)", flush=True)
        
        # ===== CREATE WRAPPED ENVIRONMENT WITH TEST AGENTS =====
        print("\n" + "="*80, flush=True)
        print("ENVIRONMENT SETUP (TEST POPULATION)", flush=True)
        print("="*80, flush=True)
        
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo_dict_test)  # ← TEST SET!
        env = LogWrapper(env)
        
        # Check wrapper configuration
        total_pop = sum([
            train_state.pop_size 
            for agents_dict in env._env.loaded_params.values() 
            for train_state in agents_dict.values()
        ])
        print(f"Loaded population size: {total_pop} agents", flush=True)
        print(f"(Should equal total test partners: {total_test_partners})", flush=True)
        
        if total_pop != total_test_partners:
            print("⚠️  WARNING: Population size mismatch!", flush=True)
        
        # ===== RUN SAMPLING TEST =====
        print("\n" + "="*80)
        print("RUNNING SAMPLING TEST")
        print("="*80)
        
        # Run enough episodes to see all agents multiple times
        num_episodes = min(10000, total_pop * 6)
        print(f"Running {num_episodes} episodes...", flush=True)
        print(f"(This is {num_episodes/total_pop:.1f}x the population size)", flush=True)
        
        indices_history = []
        key = jax.random.PRNGKey(config["SEED"])
        
        mid_episode_changes = 0
        
        for ep in range(num_episodes):
            if ep % 200 == 0:
                print(f"  Progress: {ep}/{num_episodes} episodes...", flush=True)
            
            key, reset_key, step_key = jax.random.split(key, 3)
            
            obs, state = env.reset(reset_key)
            human_idx_reset = int(state.env_state.ag_idx['human'])
            indices_history.append(human_idx_reset)
            
            dummy_actions = {
                agent: jnp.zeros(env.action_space(agent).shape) 
                for agent in env.agents
            }
            obs, state, rewards, dones, infos = env.step(step_key, state, dummy_actions)
            
            human_idx_after_step = int(state.env_state.ag_idx['human'])
            if human_idx_reset != human_idx_after_step:
                mid_episode_changes += 1
        
        print(f"✓ Completed {num_episodes} episodes", flush=True)
        
        # ===== ANALYZE RESULTS =====
        print("\n" + "="*80)
        print("ANALYSIS - TEST POPULATION")
        print("="*80)
        
        # Test 1: Index stability
        print("\n1. Index Stability Within Episodes:", flush=True)
        if mid_episode_changes > 0:
            print(f"   ❌ FAIL: Index changed {mid_episode_changes} times", flush=True)
        else:
            print(f"   ✓ PASS: Index stayed constant within episodes", flush=True)
        
        # Test 2: Coverage
        print("\n2. Agent Coverage:", flush=True)
        unique_agents = set(indices_history)
        num_unique = len(unique_agents)
        print(f"   Unique agents sampled: {num_unique}/{total_pop}", flush=True)
        
        expected_unique = int(total_pop * (1 - (1 - 1/total_pop)**num_episodes))
        print(f"   Expected unique: ~{expected_unique}", flush=True)
        
        missing_agents = set(range(total_pop)) - unique_agents
        if missing_agents:
            print(f"   ⚠️  {len(missing_agents)} agents never sampled", flush=True)
            if len(missing_agents) <= 10:
                print(f"   Missing indices: {sorted(list(missing_agents))}", flush=True)
        else:
            print(f"   ✓ All agents sampled at least once", flush=True)
        
        # Test 3: Distribution
        print("\n3. Sampling Distribution:", flush=True)
        dist = pd.Series(indices_history).value_counts().sort_index()
        
        print(f"   Agents sampled: {len(dist)}/{total_pop}", flush=True)
        print(f"   Min appearances: {dist.min()}", flush=True)
        print(f"   Max appearances: {dist.max()}", flush=True)
        print(f"   Mean appearances: {dist.mean():.2f}", flush=True)
        print(f"   Std appearances: {dist.std():.2f}", flush=True)
        print(f"   Expected mean: {num_episodes/total_pop:.2f}", flush=True)
        
        # Test 4: Statistical uniformity
        print("\n4. Uniformity Test (chi-square):", flush=True)
        if len(unique_agents) == total_pop and total_pop >= 5:
            full_dist = np.array([dist.get(i, 0) for i in range(total_pop)])
            
            try:
                chi2, p_value = chisquare(full_dist)
                
                print(f"   chi2 statistic: {chi2:.2f}", flush=True)
                print(f"   p-value: {p_value:.4f}", flush=True)
                
                if p_value < 0.01:
                    print(f"   ❌ HIGHLY non-uniform (p < 0.01)!", flush=True)
                elif p_value < 0.05:
                    print(f"   ⚠️  Marginally non-uniform (p < 0.05)", flush=True)
                    print(f"   (This can happen by chance ~5% of the time)", flush=True)
                else:
                    print(f"   ✓ PASS: Distribution is uniform", flush=True)
            except Exception as e:
                print(f"   ⚠️  Chi-square test error: {e}", flush=True)
        else:
            print(f"   ⚠️  Skipping (not all agents sampled)", flush=True)
        
        # Test 5: Consecutive repeats
        print("\n5. Consecutive Repeats:", flush=True)
        consecutive_same = sum(
            1 for i in range(len(indices_history)-1) 
            if indices_history[i] == indices_history[i+1]
        )
        expected_consecutive = (num_episodes - 1) / total_pop
        
        print(f"   Observed: {consecutive_same}", flush=True)
        print(f"   Expected: ~{expected_consecutive:.1f}", flush=True)
        
        if consecutive_same > 3 * expected_consecutive and total_pop > 10:
            print(f"   ⚠️  WARNING: Too many consecutive repeats!", flush=True)
        else:
            print(f"   ✓ Within expected range", flush=True)
        
        # ===== COMPARE WITH TRAINING POPULATION =====
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"Train population: {total_train_partners} agents", flush=True)
        print(f"Test population:  {total_pop} agents", flush=True)
        
        if total_pop == total_train_partners:
            print("✓ Populations are same size (balanced split)", flush=True)
        else:
            print(f"⚠️  Population size difference: {abs(total_pop - total_train_partners)}", flush=True)
        
        # ===== FINAL VERDICT =====
        print("\n" + "="*80)
        print("FINAL VERDICT - TEST POPULATION")
        print("="*80)
        
        all_passed = True
        
        if mid_episode_changes > 0:
            print("❌ Indices changing mid-episode", flush=True)
            all_passed = False
        else:
            print("✓ Indices stable within episodes", flush=True)
        
        coverage_ratio = num_unique / total_pop
        if coverage_ratio >= 0.95:
            print(f"✓ Excellent coverage ({coverage_ratio:.1%})", flush=True)
        elif coverage_ratio >= 0.80:
            print(f"✓ Good coverage ({coverage_ratio:.1%})", flush=True)
        else:
            print(f"❌ Poor coverage ({coverage_ratio:.1%})", flush=True)
            all_passed = False
        
        # For p-value, be lenient since we expect ~5% false positives
        if len(unique_agents) == total_pop and total_pop >= 5:
            if p_value >= 0.05:
                print("✓ Uniform distribution", flush=True)
            elif p_value >= 0.01:
                print("⚠️  Marginally non-uniform (likely random noise)", flush=True)
            else:
                print("❌ Highly non-uniform distribution", flush=True)
                all_passed = False
        
        if all_passed:
            print("\n🎉 TEST POPULATION SAMPLING WORKS CORRECTLY!", flush=True)
        else:
            print("\n⚠️  ISSUES DETECTED IN TEST POPULATION!", flush=True)
        
        print("\n" + "="*80, flush=True)
        
        return indices_history, env
    
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}", flush=True)
        traceback.print_exc()
        raise


@hydra.main(version_base=None, config_path="config", config_name="ppo_aht")
def main(config):
    print("\n" + "="*80, flush=True)
    print("TESTING TEST POPULATION SAMPLING", flush=True)
    print("="*80, flush=True)
    
    config = OmegaConf.to_container(config, resolve=True)
    
    print(f"\nConfiguration:", flush=True)
    print(f"  Environment: {config['ENV_NAME']}", flush=True)
    print(f"  Zoo Path: {config['ZOO_PATH']}", flush=True)
    print(f"  Partner Algorithms: {config['PARTNER_ALGORITHMS']}", flush=True)
    print(f"  Split Ratio: {config['SPLIT_RATIO']:.0%}/{1-config['SPLIT_RATIO']:.0%}", flush=True)
    
    indices, env = test_agent_sampling_test_set(config)
    
    if indices:
        print(f"\nFirst 30 sampled indices:", flush=True)
        print(f"{indices[:30]}", flush=True)
        
        np.save("test_population_sampling_indices.npy", indices)
        print(f"\nSaved to 'test_population_sampling_indices.npy'", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print("TEST COMPLETE", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    main()