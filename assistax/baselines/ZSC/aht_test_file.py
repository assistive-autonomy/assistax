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


def test_agent_sampling(config):
    """
    Test that agents are sampled correctly across episodes.
    """
    print("="*80)
    print("AGENT SAMPLING TEST")
    print("="*80)
    
    # Setup
    zoo = ZooManager(config["ZOO_PATH"])
    scenario = config["ENV_NAME"]
    
    # Load train partners (same as in your training script)
    partner_dict = {}
    for partner_algo in config["PARTNER_ALGORITHMS"]:
        partner_dict[partner_algo] = zoo.index.query(
            f'algorithm == "{partner_algo}"'
        ).query(
            f'scenario == "{scenario}"'
        ).query(
            'scenario_agent_id == "human"'
        )
        print(f"Found {len(partner_dict[partner_algo])} {partner_algo} partners")
    
    # Create train/test split
    all_partners = pd.concat(partner_dict.values(), ignore_index=True)
    train_partners = all_partners.sample(frac=config["SPLIT_RATIO"], random_state=42)
    
    # Split back into algorithm-specific dictionaries
    train_set = {}
    for algo in partner_dict.keys():
        train_set[algo] = train_partners[
            train_partners['algorithm'] == algo
        ].reset_index(drop=True)
    
    # Create zoo loading dictionary
    load_zoo_dict_train = {
        algo: {"human": list(train_set[algo].agent_uuid)} 
        for algo in partner_dict.keys()
    }
    
    print(f"\nTotal training partners: {sum(len(train_set[algo]) for algo in train_set)}")
    for algo in train_set.keys():
        print(f"  {algo}: {len(train_set[algo])} partners")
    
    # Create environment with loaded agents
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    wrapper = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo_dict_train)
    
    # Check wrapper configuration
    total_pop = sum([
        train_state.pop_size 
        for agents_dict in wrapper.loaded_params.values() 
        for train_state in agents_dict.values()
    ])
    print(f"\nWrapper total population: {total_pop}")
    
    # ===== RUN SAMPLING TEST =====
    print("\n" + "="*80)
    print("SAMPLING VERIFICATION")
    print("="*80)
    
    num_episodes = 200
    indices_history = []
    key = jax.random.PRNGKey(42)
    
    print(f"Running {num_episodes} episodes...")
    
    # Track if index changes mid-episode (should NOT happen)
    mid_episode_changes = 0
    
    for ep in range(num_episodes):
        key, reset_key, step_key = jax.random.split(key, 3)
        
        # Reset episode
        obs, state = wrapper.reset(reset_key)
        
        # Extract the human agent index at reset
        human_idx_reset = int(state.ag_idx['human'])
        indices_history.append(human_idx_reset)
        
        # Take a step to verify index doesn't change mid-episode
        dummy_actions = {
            agent: jnp.zeros(env.action_space(agent).shape) 
            for agent in wrapper.agents
        }
        obs, state, rewards, dones, infos = wrapper.step(step_key, state, dummy_actions)
        
        # Check if index stayed the same
        human_idx_after_step = int(state.ag_idx['human'])
        if human_idx_reset != human_idx_after_step:
            mid_episode_changes += 1
            print(f"⚠️  Episode {ep}: Index changed mid-episode! "
                  f"{human_idx_reset} -> {human_idx_after_step}")
    
    # ===== ANALYZE RESULTS =====
    print(f"\n✓ Completed {num_episodes} episodes")
    
    if mid_episode_changes > 0:
        print(f"❌ FAIL: Index changed mid-episode {mid_episode_changes} times!")
    else:
        print(f"✓ PASS: Index stayed constant within episodes")
    
    # Check unique agents encountered
    unique_agents = len(set(indices_history))
    print(f"\nUnique agents encountered: {unique_agents}/{total_pop}")
    
    # Expected unique agents based on probability
    expected_unique = int(total_pop * (1 - (1 - 1/total_pop)**num_episodes))
    print(f"Expected unique agents: ~{expected_unique}")
    
    if unique_agents < 0.8 * expected_unique:
        print(f"⚠️  WARNING: Fewer unique agents than expected!")
    
    # Distribution analysis
    print("\nAgent sampling distribution:")
    dist = pd.Series(indices_history).value_counts().sort_index()
    
    print(f"  Min appearances: {dist.min()}")
    print(f"  Max appearances: {dist.max()}")
    print(f"  Mean appearances: {dist.mean():.2f}")
    print(f"  Std appearances: {dist.std():.2f}")
    
    # Chi-square test for uniformity
    if len(dist) >= 5:  # Need at least 5 categories for chi-square
        expected_freq = num_episodes / total_pop
        chi2, p_value = chisquare(dist.values, f_exp=[expected_freq]*len(dist))
        
        print(f"\nUniformity test (chi-square):")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ⚠️  WARNING: Distribution is significantly non-uniform!")
            print(f"  This suggests biased sampling.")
        else:
            print(f"  ✓ PASS: Distribution appears uniform")
    
    # Check for unexpected consecutive repeats
    consecutive_same = sum(
        1 for i in range(len(indices_history)-1) 
        if indices_history[i] == indices_history[i+1]
    )
    expected_consecutive = (num_episodes - 1) / total_pop
    
    print(f"\nConsecutive same agent:")
    print(f"  Observed: {consecutive_same}")
    print(f"  Expected: ~{expected_consecutive:.1f}")
    
    if consecutive_same > 2 * expected_consecutive:
        print(f"  ⚠️  WARNING: Too many consecutive repeats!")
    
    # ===== FINAL VERDICT =====
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    all_passed = True
    
    if mid_episode_changes > 0:
        print("❌ FAIL: Indices changing mid-episode")
        all_passed = False
    else:
        print("✓ PASS: Indices stable within episodes")
    
    if unique_agents >= 0.8 * expected_unique:
        print("✓ PASS: Good diversity in agent sampling")
    else:
        print("❌ FAIL: Poor diversity in agent sampling")
        all_passed = False
    
    if len(dist) >= 5 and p_value >= 0.05:
        print("✓ PASS: Uniform sampling distribution")
    elif len(dist) >= 5:
        print("❌ FAIL: Non-uniform sampling distribution")
        all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Agent sampling is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED! Check the issues above.")
    
    return indices_history, wrapper


if __name__ == "__main__":
    # Your typical config (adjust paths as needed)
    config = {
        "ENV_NAME": "MABrax-ant_4x2",
        "ENV_KWARGS": {},
        "ZOO_PATH": "zoo_assistax/MABrax-ant_4x2",
        "PARTNER_ALGORITHMS": ["IPPO", "MAPPO"],
        "SPLIT_RATIO": 0.5,
    }
    
    indices, wrapper = test_agent_sampling(config)
    
    # Optional: Print first 20 indices to visually inspect
    print(f"\nFirst 20 sampled indices: {indices[:20]}")
```

**To run this test:**

1. Save as `test_agent_sampling.py`
2. Update the `config` dictionary at the bottom with your actual paths
3. Run: `python test_agent_sampling.py`

**What this test checks:**

1. ✅ **Index stability**: Verifies that the agent index doesn't change within an episode
2. ✅ **Diversity**: Checks that you're actually encountering different agents across episodes
3. ✅ **Uniformity**: Statistical test that all agents are sampled with equal probability
4. ✅ **No bias**: Checks for unexpected patterns (like too many consecutive repeats)

**Expected output for working code:**
```
✓ PASS: Index stayed constant within episodes
Unique agents encountered: 45/50
✓ PASS: Good diversity in agent sampling
✓ PASS: Uniform sampling distribution
🎉 ALL TESTS PASSED!