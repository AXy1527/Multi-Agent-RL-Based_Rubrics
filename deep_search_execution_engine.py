import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from omegaconf import DictConfig
from verl import DataProto

from pettingllms.multi_agent_env.deep_search.deep_search_env import (
    DeepSearchEnv,
    DeepSearchEnvBatch,
    AgentRole,
)
from pettingllms.multi_agent_env.deep_search.agents import (
    QAgent,
    RAgent,
    AAgent,
    RMAgent,
)
from pettingllms.trainer.async_generate import llm_async_generate

logger = logging.getLogger(__name__)


@dataclass
class DeepSearchAgentInstance:
    """Instance of an agent for a specific rollout"""
    agent: Any  # QAgent, RAgent, or AAgent
    rollout_idx: int
    env_idx: int


class DeepSearchExecutionEngine:
    """
    Execution engine for Deep Search multi-agent system.

    Integrates with the PettingLLMs training framework to handle:
    - Dynamic turn order (Q -> R -> Q -> R -> ... -> A)
    - RM-Agent scoring after episode completion
    - Reward computation and trajectory collection
    """

    def __init__(
            self,
            config: DictConfig,
            ppo_trainer_config_dict: Dict[str, Any],
            tokenizer_dict: Dict[str, Any],
            processor_dict: Dict[str, Any],
            server_address_dict: Dict[str, List[str]],
            agent_policy_mapping: Dict[str, str],
            lora_differ_mode: bool = False,
            agent_lora_mapping: Dict[str, str] = None,
    ):
        """
        Initialize the Deep Search execution engine.

        Args:
            config: Global configuration
            ppo_trainer_config_dict: PPO trainer configs per model
            tokenizer_dict: Tokenizers per model
            processor_dict: Processors per model (for multi-modal)
            server_address_dict: Server addresses per model
            agent_policy_mapping: Maps agent name to policy/model name
            lora_differ_mode: Whether using LoRA differentiation
            agent_lora_mapping: Maps agent name to LoRA ID
        """
        self.config = config
        self.ppo_trainer_config_dict = ppo_trainer_config_dict
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict
        self.server_address_dict = server_address_dict
        self.agent_policy_mapping = agent_policy_mapping
        self.lora_differ_mode = lora_differ_mode
        self.agent_lora_mapping = agent_lora_mapping or {}

        # Environment configuration
        self.env_config = config.env
        self.max_rounds = getattr(self.env_config, 'max_rounds', 10)
        self.max_turns = getattr(self.env_config, 'max_turns', 10)

        # Agent turn order from config
        self.turn_order = list(config.multi_agent_interaction.turn_order)

        # RM-Agent configuration
        self.rm_config = getattr(config.multi_agent_interaction, 'rm_agent', None)
        self.rm_agent = None
        if self.rm_config and self.rm_config.get('enabled', False):
            self.rm_agent = RMAgent(
                llm_api_url=self.rm_config.get('api_url'),
                llm_api_key=self.rm_config.get('api_key'),
                llm_model=self.rm_config.get('model'),
            )

        # Runtime state
        self.env_list: List[DeepSearchEnv] = []
        self.env_idx_list: List[int] = []
        self.rollout_idx_list: List[int] = []
        self.agent_instances: Dict[int, Dict[str, DeepSearchAgentInstance]] = {}

        # Success tracking
        self.success_rollout_idx_list_dict: Dict[str, List[int]] = {}
        self.success_ave_turn_dict: Dict[str, float] = {}

    def init_agents_and_envs(self, mode: str = "train", step_idx: int = 0):
        """
        Initialize agents and environments for a training/validation step.

        Args:
            mode: "train" or "validate"
            step_idx: Current training step index
        """
        # Clear previous state
        self.env_list = []
        self.agent_instances = {}
        self.success_rollout_idx_list_dict = {agent: [] for agent in self.turn_order}
        self.success_ave_turn_dict = {agent: 0. 0
        for agent in self.turn_order}

        # Determine batch size
        if mode == "train":
            batch_size = self.config.training.train_batch_size
            sample_num = self.config.training.train_sample_num
        else:
            batch_size = self.config.training.train_batch_size
            sample_num = self.config.training.validate_sample_num

        # Create environment indices
        self.env_idx_list = list(range(batch_size))
        self.rollout_idx_list = list(range(batch_size * sample_num))

        # Initialize environment batch
        env_batch = DeepSearchEnvBatch(
            env_idx_list=self.env_idx_list,
            env_indices=self.env_idx_list,
            rollout_idx_list=self.rollout_idx_list,
            samples=sample_num,
            max_turns=self.max_turns,
            config=self.env_config,
            mode=mode,
        )
        self.env_list = env_batch.env_list

        # Initialize agent instances for each environment
        for env in self.env_list:
            rollout_idx = env.rollout_idx
            self.agent_instances[rollout_idx] = {
                "q_agent": DeepSearchAgentInstance(
                    agent=QAgent(rollout_idx=rollout_idx),
                    rollout_idx=rollout_idx,
                    env_idx=env.env_idx,
                ),
                "r_agent": DeepSearchAgentInstance(
                    agent=RAgent(rollout_idx=rollout_idx),
                    rollout_idx=rollout_idx,
                    env_idx=env.env_idx,
                ),
                "a_agent": DeepSearchAgentInstance(
                    agent=AAgent(rollout_idx=rollout_idx),
                    rollout_idx=rollout_idx,
                    env_idx=env.env_idx,
                ),
            }

        logger.info(f"Initialized {len(self.env_list)} environments with {len(self.agent_instances)} agent sets")

    async def generate_multiple_rollouts_concurrent(
            self,
            env_idx_list: List[int],
            rollout_mode: str = "train",
    ) -> Dict[str, DataProto]:
        """
        Generate rollouts for all environments concurrently.

        This method orchestrates the Q -> R -> ...  -> A flow for all environments.

        Args:
            env_idx_list: List of environment indices
            rollout_mode: "train" or "validate"

        Returns:
            Dict mapping model names to DataProto batches
        """
        # Collect all trajectories per model
        trajectories_per_model: Dict[str, List[Dict]] = {
            model_name: [] for model_name in self.ppo_trainer_config_dict.keys()
        }

        # Run all environments concurrently
        tasks = [
            self._run_single_rollout(env, rollout_mode)
            for env in self.env_list
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for env, result in zip(self.env_list, results):
            if isinstance(result, Exception):
                logger.error(f"Rollout failed for env {env.rollout_idx}: {result}")
                continue

            # result is Dict[str, List[Dict]] - trajectories per agent
            for agent_name, agent_trajectories in result.items():
                policy_name = self.agent_policy_mapping.get(agent_name)
                if policy_name and policy_name in trajectories_per_model:
                    trajectories_per_model[policy_name].extend(agent_trajectories)

        # Convert trajectories to DataProto
        output_per_model = {}
        for model_name, trajectories in trajectories_per_model.items():
            if trajectories:
                output_per_model[model_name] = self._trajectories_to_dataproto(
                    trajectories, model_name
                )
            else:
                output_per_model[model_name] = DataProto.from_dict({})

        return output_per_model

    async def _run_single_rollout(
            self,
            env: DeepSearchEnv,
            mode: str,
    ) -> Dict[str, List[Dict]]:
        """
        Run a single rollout for one environment.

        Follows the pattern: Q -> R -> Q -> R -> ... -> Q(generate_answer) -> A

        Args:
            env: The environment instance
            mode: "train" or "validate"

        Returns:
            Dict mapping agent names to their trajectory data
        """
        rollout_idx = env.rollout_idx
        agents = self.agent_instances[rollout_idx]

        trajectories: Dict[str, List[Dict]] = {
            "q_agent": [],
            "r_agent": [],
            "a_agent": [],
        }

        turn_idx = 0

        while not env.is_done():
            current_agent_role = env.get_current_agent()

            if current_agent_role is None:
                break

            agent_name = current_agent_role.value
            agent_instance = agents.get(agent_name)

            if agent_instance is None:
                logger.error(f"Unknown agent: {agent_name}")
                break

            agent = agent_instance.agent

            # Update agent from environment
            agent.update_from_env(turn_idx, env)

            # Get prompt
            prompt = agent.current_prompt

            # Generate response via LLM
            policy_name = self.agent_policy_mapping.get(agent_name)
            response = await self._call_llm(
                rollout_idx=rollout_idx,
                turn_idx=turn_idx,
                agent_name=agent_name,
                prompt=prompt,
                policy_name=policy_name,
                mode=mode,
            )

            # Update agent from model response
            agent.update_from_model(response)

            # Execute agent step (updates environment)
            await agent.step(env)

            # Collect trajectory data
            trajectory_entry = {
                "rollout_idx": rollout_idx,
                "env_idx": env.env_idx,
                "turn_idx": turn_idx,
                "agent_name": agent_name,
                "prompt": prompt.get("text", "") if isinstance(prompt, dict) else str(prompt),
                "response": response,
                "reward": agent.agent_reward if hasattr(agent, 'agent_reward') else 0. 0,
            }
            trajectories[agent_name].append(trajectory_entry)

            turn_idx += 1

            # Safety check for max turns
            if turn_idx >= self.max_turns * 2:  # Q-R pairs
                break

        # After episode completion, compute RM scores
        if self.rm_agent and env.is_done():
            await self._compute_rm_scores(env)

            # Update rewards with RM scores
            self._apply_rm_scores_to_trajectories(env, trajectories)

        # Track success
        if env.state.a_agent_is_correct:
            for agent_name in self.turn_order:
                self.success_rollout_idx_list_dict[agent_name].append(rollout_idx)
                self.success_ave_turn_dict[agent_name] += turn_idx

        return trajectories

    async def _call_llm(
            self,
            rollout_idx: int,
            turn_idx: int,
            agent_name: str,
            prompt: Dict[str, Any],
            policy_name: str,
            mode: str,
    ) -> str:
        """
        Call LLM to generate response.

        Args:
            rollout_idx: Rollout index
            turn_idx: Turn index
            agent_name: Agent name
            prompt: Prompt dict with 'text' and optionally 'system'
            policy_name: Policy/model name
            mode: "train" or "validate"

        Returns:
            Generated response string
        """
        # Get server address for this policy
        server_addresses = self.server_address_dict.get(policy_name, [])
        if not server_addresses:
            raise ValueError(f"No server addresses for policy: {policy_name}")

        # Round-robin server selection
        address = server_addresses[rollout_idx % len(server_addresses)]

        # Get tokenizer
        tokenizer = self.tokenizer_dict.get(policy_name)

        # Get LoRA ID if in lora_differ_mode
        lora_id = None
        if self.lora_differ_mode:
            lora_id = self.agent_lora_mapping.get(agent_name)

        # Get agent config
        agent_config = None
        if hasattr(self.config, 'agent_policy_configs'):
            agent_configs = self.config.agent_policy_configs.agent_configs
            for cfg_key, cfg in agent_configs.items():
                if cfg.name == agent_name:
                    agent_config = cfg
                    break

        # Get PPO trainer config
        ppo_config = self.ppo_trainer_config_dict.get(policy_name, {})

        # Create DataProto for prompt
        prompt_text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
        system_prompt = prompt.get("system", "") if isinstance(prompt, dict) else ""

        # Combine system and user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt_text}"
        else:
            full_prompt = prompt_text

        # Tokenize
        if tokenizer:
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
            prompt_dpr = DataProto.from_dict({
                "input_ids": input_ids,
                "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
            })
        else:
            prompt_dpr = DataProto.from_dict({})

        # Call async generate
        result = await llm_async_generate(
            rollout_idx=rollout_idx,
            turn_idx=turn_idx,
            agent_idx=self.turn_order.index(agent_name) if agent_name in self.turn_order else 0,
            prompt_dpr=prompt_dpr,
            ppo_trainer_config=ppo_config,
            address=address,
            model_name=policy_name,
            tokenizer=tokenizer,
            mode=mode,
            lora_id=lora_id,
            agent_config=agent_config,
        )

        # Decode response
        if tokenizer and hasattr(result, 'batch') and 'responses' in result.batch:
            response = tokenizer.decode(result.batch['responses'][0], skip_special_tokens=True)
        elif hasattr(result, 'non_tensor_batch') and 'response_text' in result.non_tensor_batch:
            response = result.non_tensor_batch['response_text'][0]
        else:
            response = ""

        return response

    async def _compute_rm_scores(self, env: DeepSearchEnv):
        """
        Compute RM-Agent scores for completed episode.

        Args:
            env: Completed environment
        """
        if not self.rm_agent:
            return

        try:
            await self.rm_agent.compute_all_rewards(env)
        except Exception as e:
            logger.error(f"RM scoring failed for env {env.rollout_idx}: {e}")

    def _apply_rm_scores_to_trajectories(
            self,
            env: DeepSearchEnv,
            trajectories: Dict[str, List[Dict]],
    ):
        """
        Apply RM scores to trajectory rewards.

        Args:
            env: Environment with computed RM scores
            trajectories: Trajectories to update
        """
        # Get scores from environment
        q_scores = env.state.q_agent_scores
        r_scores = env.state.r_agent_scores
        a_reward = env.state.a_agent_reward

        # Apply Q-Agent scores
        for i, traj in enumerate(trajectories.get("q_agent", [])):
            if i < len(q_scores):
                score = q_scores[i]
                if isinstance(score, dict):
                    traj["reward"] = score.get("_normalized_score", 0.0)

        # Apply R-Agent scores
        for i, traj in enumerate(trajectories.get("r_agent", [])):
            if i < len(r_scores):
                score = r_scores[i]
                if isinstance(score, dict):
                    # Average across all document scores
                    doc_scores = [
                        v.get("_present_ratio", 0.0)
                        for k, v in score.items()
                        if k.startswith("doc_") and isinstance(v, dict)
                    ]
                    traj["reward"] = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0

        # Apply A-Agent binary reward
        for traj in trajectories.get("a_agent", []):
            traj["reward"] = a_reward

    def _trajectories_to_dataproto(
            self,
            trajectories: List[Dict],
            model_name: str,
    ) -> DataProto:
        """
        Convert trajectory list to DataProto format.

        Args:
            trajectories: List of trajectory dicts
            model_name: Model name for tokenization

        Returns:
            DataProto object
        """
        import torch
        import numpy as np

        tokenizer = self.tokenizer_dict.get(model_name)

        prompts = []
        responses = []
        rewards = []
        env_indices = []
        turn_indices = []
        agent_indices = []
        agent_names = []
        rollout_indices = []

        for traj in trajectories:
            prompt_text = traj.get("prompt", "")
            response_text = traj.get("response", "")

            if tokenizer:
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0]
                response_ids = tokenizer.encode(response_text, return_tensors="pt")[0]
            else:
                prompt_ids = torch.tensor([])
                response_ids = torch.tensor([])

            prompts.append(prompt_ids)
            responses.append(response_ids)
            rewards.append(traj.get("reward", 0.0))
            env_indices.append(traj.get("env_idx", 0))
            turn_indices.append(traj.get("turn_idx", 0))
            agent_names.append(traj.get("agent_name", ""))
            rollout_indices.append(traj.get("rollout_idx", 0))

            # Map agent name to index
            agent_name = traj.get("agent_name", "")
            agent_idx = self.turn_order.index(agent_name) if agent_name in self.turn_order else 0
            agent_indices.append(agent_idx)

        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
            },
            non_tensors={
                "reward": np.array(rewards),
                "env_idx": np.array(env_indices),
                "turn_idx": np.array(turn_indices),
                "agent_idx": np.array(agent_indices),
                "agent_name": np.array(agent_names, dtype=object),
                "rollout_idx": np.array(rollout_indices),
            },
        )