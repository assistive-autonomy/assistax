"""
Test script for verifying agent sampling in LoadAgentWrapper.

Usage:
    python test_agent_sampling.py
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


def test_agent_sampling(config):
    """
    Test that agents are sampled correctly across episodes.
    This mimics the actual training setup from ippo_aht.py
    """
    try:
        print("="*80, flush=True)
        print("AGENT SAMPLING TEST", flush=True)
        print("="*80, flush=True)
        
        # ===== SETUP (mimicking ippo_aht.py) =====
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]
        
        # Load partners from multiple algorithms
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
        
        # Check if we have any partners
        total_found = sum(len(df) for df in partner_dict.values())
        if total_found == 0:
            print("ERROR: No partners found in zoo!", flush=True)
            print(f"Available zoo entries:\n{zoo.index.head()}", flush=True)
            return None, None
        
        # Create train/test split (same as training script)
        all_partners = pd.concat(partner_dict.values(), ignore_index=True)
        print(f"Total partners: {len(all_partners)}", flush=True)
        
        train_partners = all_partners.sample(frac=config["SPLIT_RATIO"], random_state=42)
        test_partners = all_partners.drop(train_partners.index)
        
        # Split back into algorithm-specific dictionaries
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
        
        # Create zoo loading dictionary (SAME AS TRAINING)
        load_zoo_dict_train = {
            algo: {"human": list(train_set[algo].agent_uuid)} 
            for algo in partner_dict.keys()
        }
        
        print(f"\nTraining partners: {sum(len(train_set[algo]) for algo in train_set)}", flush=True)
        
        # ===== CREATE WRAPPED ENIRONMENT (mimicking make_train) =====
        print("\nCreating wrapped environment...", flush=True)
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        
        # This is what happens inside make_train/make_evaluation
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo_dict_train)
        env = LogWrapper(env)
        
        # Check wrapper configuration
        total_pop = sum([
            train_state.pop_size 
            for agents_dict in env._env.loaded_params.values() 
            for train_state in agents_dict.values()
        ])
        print(f"Wrapper total population: {total_pop}", flush=True)
        
        if total_pop == 0:
            print("ERROR: No agents loaded!", flush=True)
            return None, None
        
        # ===== RUN SAMPLING TEST =====
        print("\n" + "="*80)
        print("SAMPLING VERIFICATION")
        print("="*80)
        
        num_episodes = max(1000, total_pop * 6)  # At least 4x population size
        indices_history = []
        key = jax.random.PRNGKey(config["SEED"])
        
        print(f"Running {num_episodes} episodes...", flush=True)
        
        mid_episode_changes = 0
        
        for ep in range(num_episodes):
            if ep % 50 == 0:
                print(f"  Episode {ep}/{num_episodes}...", flush=True)
            
            key, reset_key, step_key = jax.random.split(key, 3)
            
            # Reset (this calls LoadAgentWrapper.reset)
            obs, state = env.reset(reset_key)
            
            # Extract the human agent index
            # Note: state.env_state is the LoadAgentState from LoadAgentWrapper
            human_idx_reset = int(state.env_state.ag_idx['human'])
            indices_history.append(human_idx_reset)
            
            # Take a step
            dummy_actions = {
                agent: jnp.zeros(env.action_space(agent).shape) 
                for agent in env.agents
            }
            obs, state, rewards, dones, infos = env.step(step_key, state, dummy_actions)
            
            # Check if index stayed the same
            human_idx_after_step = int(state.env_state.ag_idx['human'])
            if human_idx_reset != human_idx_after_step:
                mid_episode_changes += 1
                print(f"⚠️  Episode {ep}: Index changed! {human_idx_reset} -> {human_idx_after_step}", 
                      flush=True)
        
        # ===== ANALYZE RESULTS =====
        print(f"\n✓ Completed {num_episodes} episodes", flush=True)
        
        if mid_episode_changes > 0:
            print(f"❌ FAIL: Index changed mid-episode {mid_episode_changes} times!", flush=True)
        else:
            print(f"✓ PASS: Index stayed constant within episodes", flush=True)
        
        # Check diversity
        unique_agents = len(set(indices_history))
        print(f"\nUnique agents encountered: {unique_agents}/{total_pop}", flush=True)
        
        expected_unique = int(total_pop * (1 - (1 - 1/total_pop)**num_episodes))
        print(f"Expected unique agents: ~{expected_unique}", flush=True)
        
        # Distribution analysis
        print("\nAgent sampling distribution:", flush=True)
        dist = pd.Series(indices_history).value_counts().sort_index()
        
        print(f"  Agents sampled: {len(dist)}/{total_pop}", flush=True)
        print(f"  Min appearances: {dist.min()}", flush=True)
        print(f"  Max appearances: {dist.max()}", flush=True)
        print(f"  Mean appearances: {dist.mean():.2f}", flush=True)
        print(f"  Std appearances: {dist.std():.2f}", flush=True)
        
        # Show full distribution if small
        if total_pop <= 20:
            print("\nFull distribution:", flush=True)
            for idx in range(total_pop):
                count = dist.get(idx, 0)
                bar = "█" * int(count / (num_episodes / 40))
                print(f"  Agent {idx:2d}: {count:3d} times {bar}", flush=True)
        
        # Chi-square test
        if len(dist) >= 5:
            expected_freq = num_episodes / total_pop
            chi2, p_value = chisquare(dist.values, f_exp=[expected_freq]*len(dist))
            
            print(f"\nUniformity test (chi-square):", flush=True)
            print(f"  chi2 statistic: {chi2:.2f}", flush=True)
            print(f"  p-value: {p_value:.4f}", flush=True)
            
            if p_value < 0.01:
                print(f"  ❌ HIGHLY non-uniform (p < 0.01)!", flush=True)
            elif p_value < 0.05:
                print(f"  ⚠️  Significantly non-uniform (p < 0.05)", flush=True)
            else:
                print(f"  ✓ PASS: Distribution appears uniform", flush=True)
        
        # Consecutive repeats
        consecutive_same = sum(
            1 for i in range(len(indices_history)-1) 
            if indices_history[i] == indices_history[i+1]
        )
        expected_consecutive = (num_episodes - 1) / total_pop
        
        print(f"\nConsecutive same agent:", flush=True)
        print(f"  Observed: {consecutive_same}", flush=True)
        print(f"  Expected: ~{expected_consecutive:.1f}", flush=True)
        
        if consecutive_same > 3 * expected_consecutive and total_pop > 5:
            print(f"  ⚠️  WARNING: Too many consecutive repeats!", flush=True)
            print(f"  This suggests the same agent is being reused!", flush=True)
        
        # ===== FINAL VERDICT =====
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        
        all_passed = True
        
        # Test 1: No mid-episode changes
        if mid_episode_changes > 0:
            print("❌ FAIL: Indices changing mid-episode", flush=True)
            all_passed = False
        else:
            print("✓ PASS: Indices stable within episodes", flush=True)
        
        # Test 2: Good diversity
        if unique_agents >= 0.7 * expected_unique:
            print("✓ PASS: Good diversity in agent sampling", flush=True)
        else:
            print("❌ FAIL: Poor diversity in agent sampling", flush=True)
            all_passed = False
        
        # Test 3: Uniformity
        if len(dist) >= 5:
            if p_value >= 0.05:
                print("✓ PASS: Uniform sampling distribution", flush=True)
            else:
                print("❌ FAIL: Non-uniform sampling distribution", flush=True)
                all_passed = False
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED!", flush=True)
            print("Agent sampling is working correctly.", flush=True)
        else:
            print("\n⚠️  SOME TESTS FAILED!", flush=True)
            print("There may be issues with agent sampling logic.", flush=True)
        
        print("\n" + "="*80, flush=True)
        
        return indices_history, env
    
    except Exception as e:
        print(f"\n{'='*80}", flush=True)
        print(f"ERROR OCCURRED:", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"{type(e).__name__}: {str(e)}", flush=True)
        print("\nFull traceback:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


@hydra.main(version_base=None, config_path="config", config_name="ppo_aht")
def main(config):
    """Main function with Hydra config."""
    print("\n" + "="*80, flush=True)
    print("STARTING AGENT SAMPLING TEST", flush=True)
    print("="*80, flush=True)
    
    config = OmegaConf.to_container(config, resolve=True)
    
    print(f"\nConfiguration:", flush=True)
    print(f"  ENV_NAME: {config['ENV_NAME']}", flush=True)
    print(f"  ZOO_PATH: {config['ZOO_PATH']}", flush=True)
    print(f"  PARTNER_ALGORITHMS: {config['PARTNER_ALGORITHMS']}", flush=True)
    print(f"  SPLIT_RATIO: {config['SPLIT_RATIO']}", flush=True)
    
    indices, env = test_agent_sampling(config)
    
    if indices is not None:
        print(f"\n{'='*80}", flush=True)
        print("SAMPLE OUTPUT", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"First 30 sampled indices:", flush=True)
        print(f"{indices[:30]}", flush=True)
        
        # Check for obvious patterns
        if len(set(indices[:10])) == 1:
            print("\n⚠️  WARNING: First 10 episodes all used the SAME agent!", flush=True)
            print("This is a clear bug - agents should vary between episodes.", flush=True)
        
        # Save for later analysis
        import numpy as np
        np.save("test_indices.npy", indices)
        print(f"\nSaved indices to test_indices.npy", flush=True)
    else:
        print("\n❌ Test failed to run properly!", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print("TEST COMPLETE", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    print("Script started...", flush=True)
    try:
        main()
    except Exception as e:
        print(f"\nScript failed with error: {e}", flush=True)
        traceback.print_exc()
    print("Script finished.", flush=True)