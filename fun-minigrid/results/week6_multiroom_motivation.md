# Week 6 MultiRoom Motivation

## Goal

Week 6 extends the DoorKey-6x6 two-stage comparison to MultiRoom in order to test whether Manager recurrent memory becomes useful in a longer-horizon, multi-door navigation setting.

## Starting Point

The Week 5 DoorKey-6x6 two-stage V2 experiment showed that both Vanilla FuN and AblationManager can reach strong deterministic performance when trained with enough budget and low-entropy fine-tuning.

- Vanilla FuN mean sample success: 1.000
- Vanilla FuN mean argmax success: 0.943
- AblationManager mean sample success: 1.000
- AblationManager mean argmax success: 0.943

This means DoorKey-6x6 alone is not difficult enough to strongly support the claim that recurrent Manager memory is necessary. AblationManager, which removes recurrent Manager state and generates goals from the current state embedding only, performs about as well as Vanilla FuN under the final protocol.

## Why MultiRoom

DoorKey-6x6 has a short horizon and a single key-door structure. Once exploration and deterministic execution are stabilized, both models can solve it reliably.

MultiRoom requires the agent to pass through multiple rooms and doors in sequence. This creates a longer temporal dependency and a larger exploration space. In this setting, past trajectory information may become more useful for high-level goal generation.

## Research Question

Does Manager recurrent memory give Vanilla FuN a meaningful advantage over AblationManager in MultiRoom, where the agent must navigate a longer path through multiple rooms and doors?

## Expected Outcomes

If Vanilla FuN outperforms AblationManager in MultiRoom, this would support the hypothesis that recurrent Manager memory helps in longer-horizon environments.

If AblationManager matches or outperforms Vanilla FuN, this would strengthen the conclusion that the current FuN implementation does not rely strongly on recurrent Manager memory, at least under the tested MiniGrid settings.

If both models fail, the result should be interpreted as a limitation of the current REINFORCE-style FuN training setup under sparse reward, not as evidence for or against recurrent memory.

## Initial Scope

The first target environment is:

- `MiniGrid-MultiRoom-N2-S4-v0`

The initial goal is not to complete a full three-seed benchmark. The first milestone is to verify environment compatibility, collect random and untrained baselines, then run seed 1 only if the smoke tests pass.
