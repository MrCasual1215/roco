import os 
import json
import pickle 
import requests
import re
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
import openai
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from .llm_client import query_ollama_chat
from typing import List, Tuple, Dict, Union, Optional, Any

def _query_openai_compatible_chat(
    model: str,
    system_prompt: str,
    user_prompt: str = "",
    temperature: float = 0,
    max_tokens: int = 1000,
):
    """Query Ollama/OpenAI-compatible chat completions endpoint.

    Defaults are for local Ollama. Override with:
      OPENAI_BASE_URL=http://localhost:11434/v1/
      OPENAI_API_KEY=ollama
    """
    client = openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1/"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    )
    messages = [{"role": "system", "content": system_prompt}]
    # Some OpenAI-compatible local chat backends (notably Ollama/Llama chat
    # templates) may immediately emit an end-of-turn token when the request has
    # only a system message.  SingleThreadPrompter used to pass user_prompt="",
    # which caused empty responses such as completion_tokens=1 and no content.
    # Always include a real user turn to trigger assistant generation.
    messages.append({
        "role": "user",
        "content": user_prompt or (
            "Please follow the instructions above and output the next plan now. "
            "Use exactly the required EXECUTE/NAME/ACTION format."
        ),
    })

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = completion.choices[0].message.content or ""
    usage = completion.usage.model_dump() if completion.usage is not None else {}
    usage["finish_reason"] = completion.choices[0].finish_reason
    return response, usage

PATH_PLAN_INSTRUCTION="""
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""



def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan. 
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt 


def get_plan_prompt(env: MujocoSimEnv):
    return """
Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Your reasoning and final plan output are:
    """
    

class SingleThreadPrompter:
    """
    At each round, queries LLM once for each action plan, 
    query again with environment feedback if the action plan cannot be executed
    """
    def __init__(
        self, 
        env: MujocoSimEnv,
        parser: LLMResponseParser, 
        feedback_manager: FeedbackManager,
        comm_mode: str = "plan", # or chat
        use_waypoints: bool = False,
        use_history: bool = True,
        max_api_queries: int = 3,
        num_replans: int = 3,
        debug_mode: bool = False,   
        temperature: float = 0,
        max_tokens: int = 1000, 
        llm_source: str = "gpt-4",
    ):
        self.env = env 
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.debug_mode = debug_mode 
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = [] # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = [] # could inherit from previous round if the final plan failed to execute in env.
        self.unresolved_plan_feedbacks = [] # carry LLM parse/env feedback across runner steps if no plan was executable
        self.failed_action_blacklist = [] # exact invalid single-robot actions that should not be repeated
        self.failed_plan_blacklist = [] # exact invalid multi-robot plan patterns that should not be repeated
        self.response_history = [] # [response_t]
        

    def save_state(self, save_path, fname = 'prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
            unresolved_plan_feedbacks=self.unresolved_plan_feedbacks,
            failed_action_blacklist=self.failed_action_blacklist,
            failed_plan_blacklist=self.failed_plan_blacklist,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname = 'prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]
        self.unresolved_plan_feedbacks = state_dict.get("unresolved_plan_feedbacks", [])
        self.failed_action_blacklist = state_dict.get("failed_action_blacklist", [])
        self.failed_plan_blacklist = state_dict.get("failed_plan_blacklist", [])

    def _is_makesandwich_task(self) -> bool:
        """Use extra verifier only for the MakeSandwich task."""
        return self.env.__class__.__name__ == "MakeSandwichTask"

    def _is_pack_task(self) -> bool:
        """Use extra verifier only for the PackGrocery task."""
        return self.env.__class__.__name__ == "PackGroceryTask"

    def _is_sort_task(self) -> bool:
        """Use the public-code Sort pipeline only for SortOneBlockTask."""
        return self.env.__class__.__name__ == "SortOneBlockTask"

    def _add_unique_limited(self, items: List[str], item: str, limit: int = 20):
        item = item.strip()
        if not item or item in items:
            return
        items.append(item)
        del items[:-limit]

    def _plan_action_lines(self, llm_plan) -> List[str]:
        if llm_plan is None:
            return []
        lines = []
        for agent_name, action_str in llm_plan.action_strs.items():
            lines.append(f"NAME {agent_name} ACTION {action_str}")
        return lines

    def _remember_failed_actions(self, feedback: str, failed_llm_plan=None):
        """Build a prompt-level blacklist from failed env feedback.

        Keep this conservative: blacklist exact failed full-plan patterns for all
        failures, and blacklist individual actions only when feedback identifies
        a concrete illegal action (e.g. bad recipe order).
        """
        if not feedback or feedback == "None" or failed_llm_plan is None:
            return

        proposal = getattr(failed_llm_plan, "parsed_proposal", "").strip()
        if proposal:
            self._add_unique_limited(self.failed_plan_blacklist, proposal, limit=12)

        action_lines = self._plan_action_lines(failed_llm_plan)
        action_by_obj = []
        for line in action_lines:
            # Match the object being PUT/PICK so feedback can identify the bad action.
            put_m = re.search(r"\bACTION\s+PUT\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)", line)
            pick_m = re.search(r"\bACTION\s+PICK\s+([A-Za-z0-9_]+)", line)
            if put_m:
                action_by_obj.append(("PUT", put_m.group(1), line))
            if pick_m:
                action_by_obj.append(("PICK", pick_m.group(1), line))

        # Recipe-order feedback names the illegally PUT object:
        # "recipe says cheese must be put on tomato"
        bad_put_objs = set(re.findall(r"recipe says\s+([A-Za-z0-9_]+)\s+must be put on", feedback))
        for action_type, obj, line in action_by_obj:
            if action_type == "PUT" and obj in bad_put_objs:
                self._add_unique_limited(self.failed_action_blacklist, line)

        # Feedback for already-stacked objects names the illegal PICK object:
        # "Chad cannot PICK cucumber, it's already stacked"
        bad_pick_objs = set(re.findall(r"cannot PICK\s+([A-Za-z0-9_]+)", feedback))
        for action_type, obj, line in action_by_obj:
            if action_type == "PICK" and obj in bad_pick_objs:
                self._add_unique_limited(self.failed_action_blacklist, line)

    def _format_blacklist_prompt(self) -> str:
        if len(self.failed_action_blacklist) == 0 and len(self.failed_plan_blacklist) == 0:
            return ""

        prompt = """
[Failed Action Blacklist]
The following are known failed actions or failed plan patterns for the current unresolved situation.
Do NOT repeat any blacklisted individual action. If a robot is holding a future ingredient that it cannot legally PUT yet, make that robot WAIT.
Do NOT repeat any blacklisted whole-plan pattern exactly; change the invalid robot action, usually to WAIT, while preserving valid progress by the other robot.
"""
        if len(self.failed_action_blacklist) > 0:
            prompt += "Blacklisted individual actions:\n"
            for action in self.failed_action_blacklist[-12:]:
                prompt += f"- {action}\n"
        if len(self.failed_plan_blacklist) > 0:
            prompt += "Blacklisted whole-plan patterns:\n"
            for plan in self.failed_plan_blacklist[-6:]:
                one_line = " | ".join([ln.strip() for ln in plan.splitlines() if ln.strip()])
                prompt += f"- {one_line}\n"
        prompt += "\n"
        return prompt

    def compose_verifier_system_prompt(
        self,
        obs_desp: str,
        candidate_response: str,
        plan_feedbacks: List[str],
    ) -> str:
        """A MakeSandwich-only second-stage verifier/corrector."""
        task_desp = self.env.describe_task_context()
        action_desp = self.env.get_action_prompt()
        history_desp = self.compose_round_history() if self.use_history else ""
        feedback_prompt = ""
        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n" + "\n".join(plan_feedbacks) + "\n"

        return f"""
{task_desp}
{action_desp}
{history_desp}
{obs_desp}
{feedback_prompt}
{self._format_blacklist_prompt()}
[Candidate Plan To Verify]
{candidate_response}

[Sandwich Verifier Rules]
You are a strict verifier for MakeSandwich only.
Silently check the recipe order before final output:
1) Determine the current stack on cutting_board from the scene and history.
2) Determine the next required ingredient in the recipe.
3) A PUT is valid only when the PUT object is exactly the next required ingredient and the target is exactly its immediate predecessor/top of stack.
4) A robot may PUT an item only if that robot is currently holding that item. If the next required ingredient is still on the table, the valid progress action is to PICK it with a reachable empty-gripper robot.
5) If a robot holds a future ingredient whose predecessor is not yet stacked, that robot must WAIT; never PUT the future ingredient early.
6) Only one robot may PUT in one round.
7) Never repeat a blacklisted individual action or exact failed plan pattern.

If the candidate plan is valid, output it unchanged.
If it is invalid, output a corrected plan. Prefer replacing only the invalid robot action with WAIT while preserving any valid PICK/PUT that advances the next required ingredient.
Return ONLY the executable plan in the required EXECUTE/NAME/ACTION format.
""".strip()

    def compose_verifier_user_prompt(self) -> str:
        agent_names = list(self.robot_agent_names)
        required_lines = "\n".join(
            [f"NAME {agent_name} ACTION <one valid action>" for agent_name in agent_names]
        )
        return f"""
Verify and, if needed, correct the candidate MakeSandwich plan.
Return ONLY:
EXECUTE
{required_lines}
""".strip()

    def verify_sandwich_plan(
        self,
        obs_desp: str,
        candidate_response: str,
        plan_feedbacks: List[str],
        save_path: str,
        replan_idx: int,
    ) -> str:
        if not self._is_makesandwich_task() or self.debug_mode:
            return candidate_response

        system_prompt = self.compose_verifier_system_prompt(
            obs_desp=obs_desp,
            candidate_response=candidate_response,
            plan_feedbacks=plan_feedbacks,
        )
        user_prompt = self.compose_verifier_user_prompt()
        verifier_response, usage = self.query_once(system_prompt, user_prompt=user_prompt)

        timestamp = datetime.now().strftime("%m%d-%H%M")
        tosave = [
            {
                "sender": "VerifierSystemPrompt",
                "message": system_prompt,
            },
            {
                "sender": "VerifierUserPrompt",
                "message": user_prompt,
            },
            {
                "sender": "CandidatePlanner",
                "message": candidate_response,
            },
            {
                "sender": "Verifier",
                "message": verifier_response,
            },
            usage,
        ]
        if save_path:
            fname = f'{save_path}/replan{replan_idx}_verifier_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

        # Keep a malformed verifier from destroying an otherwise parseable plan;
        # the normal parser+feedback path will still reject invalid candidates.
        if verifier_response and "EXECUTE" in verifier_response:
            return verifier_response
        return candidate_response

    def compose_pack_verifier_system_prompt(
        self,
        obs_desp: str,
        candidate_response: str,
        plan_feedbacks: List[str],
    ) -> str:
        """A PackGrocery-only second-stage verifier/corrector."""
        task_desp = self.env.describe_task_context()
        action_desp = self.env.get_action_prompt()
        history_desp = self.compose_round_history() if self.use_history else ""
        feedback_prompt = ""
        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n" + "\n".join(plan_feedbacks) + "\n"

        return f"""
{task_desp}
{action_desp}
{history_desp}
{obs_desp}
{feedback_prompt}
{self._format_blacklist_prompt()}
[Candidate Plan To Verify]
{candidate_response}

[Pack Verifier Rules]
You are a strict verifier for PackGrocery only.
Silently check the candidate plan before final output:
1) Output format must be exactly EXECUTE followed by one NAME/ACTION line for every robot.
2) Only PICK and PLACE are valid actions. Never output WAIT for PackGrocery.
3) A robot with an empty gripper may PICK one grocery item that is still on the table. A robot holding an item must PLACE that held item into an empty bin slot.
4) Never PLACE into an occupied bin slot shown in the scene. Treat a slot as occupied if any grocery item is described as "inside slot <slot>" or is already very close to that slot. For example, if milk is inside bin_front_right, do not PLACE cereal into bin_front_right; if soda_can is inside bin_back_left, do not PLACE bread into bin_back_left.
5) Before choosing PLACE targets, infer the empty slots from the scene. Use only truly empty slots for newly placed objects. If the preferred distant pair uses an occupied slot, choose another empty pair instead.
6) If both robots PLACE in the same round, their target slots should be non-adjacent/distant when possible: the target slot XY positions should be at least 0.35 apart. Avoid close pairs such as bin_back_middle with bin_front_right. However, occupied-slot avoidance is more important: never choose an occupied slot just to satisfy distance.
7) If both robots PLACE in the same round, their middle waypoints must be high and separated until final descent. Alice should use a high left/front corridor with middle waypoints around x<=0.15, y<=0.52, z=0.60-0.68. Bob should use a high back/right corridor with middle waypoints around y=0.60-0.70, z=0.60-0.68.
8) Bob must avoid low/central middle waypoints such as (0.20,0.50,0.40), (0.20,0.54,0.62), or (0.35,0.50,0.62). Prefer (0.45,0.64,0.64) or (0.60,0.64,0.64).
9) Keep Alice/Bob same-index middle waypoint XY separation about 0.25 or more for simultaneous PLACE. Do not over-correct plans just because non-simultaneous middle waypoints are close.
10) Respect previous environment feedback and do not repeat blacklisted failed plans.

If the candidate plan is valid, output it unchanged.
If it is invalid, output a corrected plan that obeys all PackGrocery rules.
Return ONLY the executable plan in the required EXECUTE/NAME/ACTION/PATH format.
""".strip()

    def compose_pack_verifier_user_prompt(self) -> str:
        agent_names = list(self.robot_agent_names)
        required_lines = "\n".join(
            [f"NAME {agent_name} ACTION <one valid action with PATH>" for agent_name in agent_names]
        )
        return f"""
Verify and, if needed, correct the candidate PackGrocery plan.
Return ONLY:
EXECUTE
{required_lines}
""".strip()

    def verify_pack_plan(
        self,
        obs_desp: str,
        candidate_response: str,
        plan_feedbacks: List[str],
        save_path: str,
        replan_idx: int,
    ) -> str:
        if not self._is_pack_task() or self.debug_mode:
            return candidate_response

        system_prompt = self.compose_pack_verifier_system_prompt(
            obs_desp=obs_desp,
            candidate_response=candidate_response,
            plan_feedbacks=plan_feedbacks,
        )
        user_prompt = self.compose_pack_verifier_user_prompt()
        verifier_response, usage = self.query_once(system_prompt, user_prompt=user_prompt)

        timestamp = datetime.now().strftime("%m%d-%H%M")
        tosave = [
            {
                "sender": "VerifierSystemPrompt",
                "message": system_prompt,
            },
            {
                "sender": "VerifierUserPrompt",
                "message": user_prompt,
            },
            {
                "sender": "CandidatePlanner",
                "message": candidate_response,
            },
            {
                "sender": "Verifier",
                "message": verifier_response,
            },
            usage,
        ]
        if save_path:
            fname = f'{save_path}/replan{replan_idx}_pack_verifier_{timestamp}.json'
            json.dump(tosave, open(fname, 'w'))

        # Keep a malformed verifier from destroying an otherwise parseable plan;
        # the normal parser+feedback path will still reject invalid candidates.
        if verifier_response and "EXECUTE" in verifier_response:
            return verifier_response
        return candidate_response

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}"
        ret += f"== Current Round ==\n"
        return ret

    # ---- Sort-only public deployment pipeline helpers ----
    # These mirror the public Sort pipeline but are gated by _is_sort_task() so
    # Cabinet/Rope/Sweep/Sandwich/Pack keep their existing deployment flows.

    def _sort_format_legal_actions_prompt(self, obs: EnvState) -> str:
        if self._is_sort_task() and hasattr(self.env, "format_legal_actions_prompt"):
            return self.env.format_legal_actions_prompt(obs)
        return ""

    def _sort_get_legal_actions(self, obs: EnvState) -> Dict[str, List[str]]:
        if self._is_sort_task() and hasattr(self.env, "get_legal_actions"):
            return self.env.get_legal_actions(obs)
        return {}

    def _sort_get_recommended_response(self, obs: EnvState) -> Optional[str]:
        if not (self._is_sort_task() and hasattr(self.env, "get_recommended_plan")):
            return None
        plan = self.env.get_recommended_plan(obs)
        if not plan:
            return None
        lines = ["EXECUTE"]
        for agent_name in self.robot_agent_names:
            if agent_name not in plan:
                return None
            lines.append(f"NAME {agent_name} ACTION {plan[agent_name]}")
        return "\n".join(lines)

    def _response_from_actions(self, actions: Dict[str, str]) -> str:
        lines = ["EXECUTE"]
        for agent_name in self.robot_agent_names:
            lines.append(f"NAME {agent_name} ACTION {actions.get(agent_name, 'WAIT')}")
        return "\n".join(lines)

    def _extract_action_lines(self, response: str) -> Dict[str, str]:
        if not response or "EXECUTE" not in response:
            return {}
        execute_str = response.split("EXECUTE", 1)[1]
        actions = {}
        for raw_line in execute_str.splitlines():
            line = raw_line.strip()
            if not line or "NAME" not in line or "ACTION" not in line:
                continue
            agent_name = line.split("NAME", 1)[1].split("ACTION", 1)[0].strip()
            action = line.split("ACTION", 1)[1].strip()
            actions[agent_name] = action
        return actions

    def _sort_validate_against_legal_actions(
        self,
        obs: EnvState,
        response: str,
        legal_actions: Dict[str, List[str]],
        forbidden_actions: Dict[str, set],
    ) -> Tuple[bool, str]:
        """Public-code validation layer, enabled only for SortOneBlockTask."""
        actions = self._extract_action_lines(response)
        expected_agents = list(self.robot_agent_names)
        missing = [agent for agent in expected_agents if agent not in actions]
        extra = [agent for agent in actions if agent not in expected_agents]
        if missing or extra:
            return False, f"Plan must contain exactly one action for each robot. missing={missing}, extra={extra}"

        allowed_action_names = None
        if hasattr(self.env, "get_allowed_action_names"):
            allowed_action_names = set(self.env.get_allowed_action_names())

        picked_objects = []
        placed_targets = []
        active_actions = []
        for agent_name, action in actions.items():
            first_token = action.split()[0] if action.split() else ""
            if allowed_action_names is not None and first_token not in allowed_action_names:
                return False, (
                    f"Invalid action name for {agent_name}: '{first_token}'. "
                    f"Allowed action names: {sorted(allowed_action_names)}"
                )

            legal_for_agent = legal_actions.get(agent_name, []) if legal_actions else []
            if legal_actions and action not in legal_for_agent:
                return False, (
                    f"Illegal action for {agent_name}: '{action}'. "
                    f"Choose one of: {legal_for_agent}"
                )
            if action in forbidden_actions.get(agent_name, set()):
                return False, f"Action for {agent_name} repeats a failed action this round: '{action}'"
            if action != "WAIT":
                active_actions.append((agent_name, action))
            if "PICK" in action and "PLACE" in action:
                obj = action.split("PICK", 1)[1].split("PLACE", 1)[0].strip()
                target = action.split("PLACE", 1)[1].strip()
                picked_objects.append(obj)
                placed_targets.append(target)

        max_parallel_actions = None
        if hasattr(self.env, "get_max_parallel_actions"):
            try:
                max_parallel_actions = self.env.get_max_parallel_actions(obs)
            except TypeError:
                max_parallel_actions = self.env.get_max_parallel_actions()
        if max_parallel_actions is not None and len(active_actions) > max_parallel_actions:
            return False, (
                f"Too many non-WAIT actions: {len(active_actions)}. "
                f"This task allows at most {max_parallel_actions} non-WAIT action(s) per round. "
                f"Active actions: {active_actions}. Use WAIT for the other robots."
            )

        duplicate_objects = sorted({obj for obj in picked_objects if picked_objects.count(obj) > 1})
        if duplicate_objects:
            return False, f"Multiple robots cannot PICK the same object in one round: {duplicate_objects}"
        duplicate_targets = sorted({target for target in placed_targets if placed_targets.count(target) > 1})
        if duplicate_targets:
            return False, f"Multiple robots should not PLACE into the same target in one round: {duplicate_targets}"

        if hasattr(self.env, "verify_plan_semantics"):
            valid, reason = self.env.verify_plan_semantics(obs, actions)
            if not valid:
                return False, f"Task semantic verification failed: {reason}"
        return True, "OK"

    def _extract_agents_from_text(self, text: str) -> List[str]:
        failed_agents = []
        if not text:
            return failed_agents
        for agent_name in self.robot_agent_names:
            patterns = [
                f"Action for {agent_name}",
                f"Illegal action for {agent_name}",
                f"{agent_name}'s ACTION",
                f"Out of reach: {agent_name}",
                f"IK failed: on {agent_name}",
            ]
            if any(pattern in text for pattern in patterns):
                failed_agents.append(agent_name)
        return failed_agents

    def _ban_actions_from_response(
        self,
        response: str,
        forbidden_actions: Dict[str, set],
        agents: Optional[List[str]] = None,
    ) -> None:
        agents_to_ban = set(agents) if agents else None
        for agent_name, action in self._extract_action_lines(response).items():
            if agents_to_ban is not None and agent_name not in agents_to_ban:
                continue
            if action != "WAIT":
                forbidden_actions.setdefault(agent_name, set()).add(action)

    def _format_forbidden_actions(self, forbidden_actions: Dict[str, set]) -> str:
        if not any(forbidden_actions.values()):
            return ""
        lines = ["[Forbidden Actions This Replan Round]"]
        for agent_name in self.robot_agent_names:
            for action in sorted(forbidden_actions.get(agent_name, [])):
                lines.append(f"- {agent_name}: {action}")
        lines.append("Do not repeat forbidden actions; choose another listed legal action or WAIT.")
        return "\n".join(lines) + "\n"

    def _should_ban_individual_actions(self, feedback: str) -> bool:
        return "Collision detected" not in feedback

    def _make_feedback_more_actionable(self, feedback: str) -> str:
        if "Collision detected" not in feedback:
            return feedback
        return (
            feedback
            + "\nCollision feedback means the concurrent robot combination is unsafe. "
            + "Try fewer simultaneous actions, preferably one active robot and others WAIT."
        )

    def _try_parse_and_feedback(self, obs: EnvState, response: str):
        parse_succ, parsed_str, plans = self.parser.parse(obs, response)
        if not parse_succ:
            return False, f"Parsing failed: {parsed_str}", []
        for plan in plans:
            ready, feedback = self.feedback_manager.give_feedback(plan)
            if not ready:
                return False, feedback, []
        return True, "None", plans

    def _sort_partial_fallback_responses(
        self,
        obs: EnvState,
        legal_actions: Dict[str, List[str]],
        forbidden_actions: Dict[str, set],
    ) -> List[str]:
        responses = []
        seen = set()

        def add_if_valid(actions: Dict[str, str]) -> None:
            response = self._response_from_actions(actions)
            if response in seen:
                return
            seen.add(response)
            valid, _ = self._sort_validate_against_legal_actions(
                obs,
                response,
                legal_actions,
                forbidden_actions,
            )
            if valid:
                responses.append(response)

        fallback = self._sort_get_recommended_response(obs)
        if fallback:
            base_actions = self._extract_action_lines(fallback)
            active_agents = [
                agent for agent in self.robot_agent_names
                if base_actions.get(agent, "WAIT") != "WAIT"
            ]
            from itertools import combinations
            for size in range(len(active_agents), 0, -1):
                for combo in combinations(active_agents, size):
                    actions = {agent: "WAIT" for agent in self.robot_agent_names}
                    for agent in combo:
                        actions[agent] = base_actions[agent]
                    add_if_valid(actions)

        atomic_candidates = []
        for agent_name in self.robot_agent_names:
            for action in legal_actions.get(agent_name, []):
                if action == "WAIT" or action in forbidden_actions.get(agent_name, set()):
                    continue
                atomic_candidates.append((agent_name, action))

        def compatible(combo: Tuple[Tuple[str, str], ...]) -> bool:
            agents = [agent for agent, _ in combo]
            if len(set(agents)) != len(agents):
                return False
            picked = []
            targets = []
            for _, action in combo:
                if "PICK" in action and "PLACE" in action:
                    picked.append(action.split("PICK", 1)[1].split("PLACE", 1)[0].strip())
                    targets.append(action.split("PLACE", 1)[1].strip())
            return len(set(picked)) == len(picked) and len(set(targets)) == len(targets)

        from itertools import combinations
        max_parallel_actions = min(2, len(self.robot_agent_names))
        if hasattr(self.env, "get_max_parallel_actions"):
            try:
                max_parallel_actions = min(max_parallel_actions, self.env.get_max_parallel_actions(obs))
            except TypeError:
                max_parallel_actions = min(max_parallel_actions, self.env.get_max_parallel_actions())
        for size in range(max_parallel_actions, 0, -1):
            for combo in combinations(atomic_candidates, size):
                if not compatible(combo):
                    continue
                actions = {agent: "WAIT" for agent in self.robot_agent_names}
                for agent, action in combo:
                    actions[agent] = action
                add_if_valid(actions)
        return responses
        
    def compose_system_prompt(
        self,
        obs_desp: str,
        plan_feedbacks: List[str] = [],
        obs: Optional[EnvState] = None,
        forbidden_actions: Optional[Dict[str, set]] = None,
        ):
        
        task_desp = self.env.describe_task_context() # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n"

        if self._is_sort_task() and obs is not None and hasattr(self.env, "get_plan_state_prompt"):
            full_prompt += self.env.get_plan_state_prompt(obs) + "\n"
        if self._is_sort_task() and obs is not None:
            legal_actions_prompt = self._sort_format_legal_actions_prompt(obs)
            if legal_actions_prompt:
                full_prompt += legal_actions_prompt + "\n"
        if self._is_sort_task() and forbidden_actions is not None:
            full_prompt += self._format_forbidden_actions(forbidden_actions)
        
        if self.use_history:
            history_desp = self.compose_round_history() 
            full_prompt += history_desp + "\n" 
        
        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) 
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt

        full_prompt += self._format_blacklist_prompt()
        
        if self.comm_mode == "plan":
            if self._is_sort_task() and hasattr(self.env, "central_plan_prompt"):
                comm_prompt = self.env.central_plan_prompt()
            else:
                comm_prompt = get_plan_prompt(self.env)
        elif self.comm_mode == "chat":
            comm_prompt = get_chat_prompt(self.env) 
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt 

    def compose_user_prompt(self):
        agent_names = list(self.robot_agent_names)
        agent_list = ", ".join(agent_names)
        required_lines = "\n".join(
            [f"NAME {agent_name} ACTION <one valid action>" for agent_name in agent_names]
        )

        if self.comm_mode == "plan":
            return f"""
You are the centralized planner for these robots: {agent_list}.
Decide the best next action for every robot.

Return ONLY the executable plan. Do not include analysis, markdown, or extra text.
The response must have exactly this shape:
EXECUTE
{required_lines}

Replace each '<one valid action>' with one valid action selected from [Action Options].
Do not stop after EXECUTE; include exactly one NAME/ACTION line for each robot.
Your response is:
            """.strip()

        if self.comm_mode == "chat":
            return f"""
You are the coordinator for these robots: {agent_list}.
Use the instructions above to produce the final agreed plan.

Return ONLY the final executable plan in this exact shape:
EXECUTE
{required_lines}

Replace each '<one valid action>' with a valid action.
Your response is:
            """.strip()

        return "Follow the instructions above and output the next response now."

    def _prompt_sort_one_round(self, obs: EnvState, save_path: str = ""):
        """Sort-only prompt loop matching the public-code Sort deployment flow."""
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        ready_to_execute = False
        llm_plans = []
        forbidden_actions = {}
        legal_actions = self._sort_get_legal_actions(obs)
        for i in range(self.num_replans):
            system_prompt = self.compose_system_prompt(
                obs_desp,
                plan_feedbacks,
                obs=obs,
                forbidden_actions=forbidden_actions,
            )
            response, usage = self.query_once(
                system_prompt, user_prompt=""
            )
            candidate_response = response
            response_history.append(response)

            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": "",
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
            if save_path:
                fname = f'{save_path}/replan{i}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))

            curr_feedback = "None"
            valid_legal, legal_reason = self._sort_validate_against_legal_actions(
                obs,
                response,
                legal_actions,
                forbidden_actions,
            )
            if not valid_legal:
                curr_feedback = f"""
Action candidate validation failed! {legal_reason}
Previous response:
{response}
Choose exactly one action per robot from [Legal Actions]. Do not invent actions.
                """
                failed_agents = self._extract_agents_from_text(legal_reason)
                if "Too many non-WAIT actions" not in legal_reason:
                    self._ban_actions_from_response(
                        response,
                        forbidden_actions,
                        agents=(failed_agents or None),
                    )
                ready_to_execute = False
                parse_succ = False
                llm_plans = []
            else:
                parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response)
                if not parse_succ:
                    execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                    curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                    """
                    failed_agents = self._extract_agents_from_text(parsed_str)
                    self._ban_actions_from_response(
                        response,
                        forbidden_actions,
                        agents=(failed_agents or None),
                    )
                    ready_to_execute = False
                else:
                    ready_to_execute = True
                    for j, llm_plan in enumerate(llm_plans):
                        ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                        if not ready_to_execute:
                            curr_feedback = self._make_feedback_more_actionable(env_feedback)
                            if self._should_ban_individual_actions(env_feedback):
                                failed_agents = self._extract_agents_from_text(env_feedback)
                                self._ban_actions_from_response(
                                    response,
                                    forbidden_actions,
                                    agents=(failed_agents or None),
                                )
                            break

            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            if save_path:
                fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))

            if ready_to_execute:
                break

        if not ready_to_execute:
            fallback_attempts = []
            for fallback_response in self._sort_partial_fallback_responses(obs, legal_actions, forbidden_actions):
                fallback_ready, fallback_feedback, fallback_plans = self._try_parse_and_feedback(
                    obs,
                    fallback_response,
                )
                fallback_attempts.append(
                    {
                        "response": fallback_response,
                        "feedback": fallback_feedback,
                        "ready": fallback_ready,
                    }
                )
                if fallback_ready:
                    ready_to_execute = True
                    llm_plans = fallback_plans
                    response_history.append(fallback_response)
                    break
            if fallback_attempts and save_path:
                timestamp = datetime.now().strftime("%m%d-%H%M")
                json.dump(
                    [
                        {
                            "sender": "Planner",
                            "message": fallback_attempts[-1]["response"],
                        },
                        {
                            "sender": "Feedback",
                            "message": (
                                "Used deterministic partial fallback after LLM replans failed."
                                if ready_to_execute
                                else "Deterministic partial fallback attempted but no candidate passed feedback."
                            ),
                        },
                        {
                            "sender": "FallbackAttempts",
                            "message": json.dumps(fallback_attempts, indent=2),
                        },
                    ],
                    open(f"{save_path}/fallback_{timestamp}.json", "w"),
                )

        self.response_history = response_history
        if ready_to_execute:
            self.unresolved_plan_feedbacks = []
        else:
            self.unresolved_plan_feedbacks = plan_feedbacks[-3:]
        return ready_to_execute, llm_plans, plan_feedbacks, response_history

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        if self._is_sort_task():
            return self._prompt_sort_one_round(obs, save_path=save_path)

        # Start with feedback from a previous runner step that failed to produce
        # any executable plan.  Without this, the next step sees an unchanged
        # scene and repeats the same invalid proposal.
        plan_feedbacks = list(self.unresolved_plan_feedbacks)
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        ready_to_execute = False
        llm_plans = None
        for i in range(self.num_replans): 
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            user_prompt = self.compose_user_prompt()
            response, usage = self.query_once(
                system_prompt, user_prompt=user_prompt
                )
            candidate_response = response
            response = self.verify_sandwich_plan(
                obs_desp=obs_desp,
                candidate_response=candidate_response,
                plan_feedbacks=plan_feedbacks,
                save_path=save_path,
                replan_idx=i,
            )
            response = self.verify_pack_plan(
                obs_desp=obs_desp,
                candidate_response=response,
                plan_feedbacks=plan_feedbacks,
                save_path=save_path,
                replan_idx=i,
            )
            response_history.append(response)
            
            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": user_prompt,
                    },
                    {
                        "sender": "Planner",
                        "message": candidate_response,
                    },
                    {
                        "sender": "VerifiedPlanner" if response != candidate_response else "PlannerUsed",
                        "message": response,
                    },
                    usage,
                ]
            if save_path:
                fname = f'{save_path}/replan{i}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))  
            
            curr_feedback = "None"
            failed_llm_plan = None
            # try parsing 
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response) 
            if not parse_succ: 
                execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                """
                ready_to_execute = False  
            # give env. feedback 
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        failed_llm_plan = llm_plan
                        break
            
            if curr_feedback != "None":
                self._remember_failed_actions(curr_feedback, failed_llm_plan)
                plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            if save_path:
                fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
                json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute:
                plan_str = parsed_str
                break  
        self.response_history = response_history
        if ready_to_execute:
            self.unresolved_plan_feedbacks = []
        else:
            # Keep only the latest few non-success feedback messages; this gives
            # the next runner step continuity without making the prompt explode.
            self.unresolved_plan_feedbacks = plan_feedbacks[-3:]
        return ready_to_execute, llm_plans, plan_feedbacks, response_history


    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None   
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                if self._is_sort_task():
                    response, usage = query_ollama_chat(
                        model=self.llm_source,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    response, usage = _query_openai_compatible_chat(
                        model=self.llm_source,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as e:
                print(f"API error, try again: {e}")
                if self._is_sort_task():
                    response = ""
                    usage = {"error": str(e), "model": self.llm_source}
            continue
        return response, usage

    

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            self.unresolved_plan_feedbacks = []
            self.failed_action_blacklist = []
            self.failed_plan_blacklist = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n{obs_desp}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.unresolved_plan_feedbacks = []
        self.failed_action_blacklist = []
        self.failed_plan_blacklist = []
        self.response_history = []
