import os
import copy
import time
import cv2 
import random
import numpy as np  
from pydantic import dataclasses, validator 
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control 
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, UR5E_SUCTION_CONSTANTS, PANDA_CONSTANTS

SORT_ALL_OBJECTS=[
    "panel2",
    "panel4",
    "panel6",
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]
ONE_OBJ_EACH=[
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
] 
SORTING_BIN_NAMES=[
    "panel2",
    "panel4",
    "panel6",
]

SORT_TASK_CONTEXT=""" 
7 panels on the table, ordered left to right: panel1,...,panel7. They form a straight assembly line, panel1 is closed to panel2 and farthest from panel7.
There are 3 cubes, each robot must place their cube on the correct target, their (cube, target_panel) pairs: 
Alice: (blue_square, panel2), 
Bob: (pink_polygon, panel4), 
Chad: (yellow_trapezoid, panel6).
There are 3 robots, each with a limited reach range, this means they can only pick cubes from these panels, and can only place cubes on these panels. The (robot, [reachable_panels]) pairs: 
(Alice, [panel1, panel2, panel3])
(Bob, [panel3, panel4, panel5])
(Chad, [panel5, panel6, panel7])
"""
 
SORT_TASK_DIALOG_PROMPT=""

SORT_TASK_CHAT_PROMPT="""Robots discuss to find the best strategy. When each robot talk, it reasons about its own capability (e.g. I am Bob, I can only reach panel3), verifies other robots' claims (e.g. I am Alice, this trapezoid cube is indeed too far for Chad), then decide how to best achieve the task and help each other (e.g. therefore I will move it to panel3 so it's closer to Chad). 
Carefully analyze Environment Feedback, Scene Description and others' responses to coordinate together. They talk in order [Alice],[Bob],[Chad],[Alice] ..., then, after everyone agreed, propose **exactly** one ACTION per robot, then stop talking. 
Their entire chat history and the final plan are: 
"""

SORT_TASK_PLAN_PROMPT="""
Reason about the Sort Cubes task step-by-step. Carefully consider [Scene description], [Structured Task State], [Legal Actions], [Recommended Plan], and [Environment Feedback].

Important Sort-specific rules:
- Object target panels are fixed. Do not swap object goals between robots.
- Fixed goals: blue_square -> panel2, pink_polygon -> panel4, yellow_trapezoid -> panel6.
- Each robot has a limited reach range; choose only actions listed in [Legal Actions].
- Prefer actions that move an object closer to its fixed target. If direct final placement is impossible, use a reachable handoff panel.
- Alice/Bob usually hand off through panel3; Bob/Chad usually hand off through panel5, but these are preferences, not object goals.
- Avoid repeating an action that did not change the scene. If a cube is already on a shared handoff panel and the receiver can pick it, prefer the receiver's final placement over same-panel re-placement.
- For reliability, use at most two non-WAIT actions per round.
- Parallel actions are best when their panel corridors do not overlap. If unsure, use one active robot and others WAIT.
- Prefer [Recommended Plan] unless feedback says it failed, but all [Legal Actions] are valid candidates.
- To reduce collision risk, use WAIT for robots that do not need to move.

At each round, output exactly one ACTION per robot and strictly follow [Action Output Instruction].
Your final plan output is:
"""

SORTING_ACTION_SPACE="""
[Action Options]
1) PICK <object name> PLACE <location>
2) PLACE <object name> <location>: recovery action only if your gripper is already holding the object.
3) WAIT
Only PICK an object if your gripper is empty. If you are already holding a cube, use PLACE <object name> <location> to release it. Target <location> should be a panel.
[Action Output Instruction]
You must first output 'EXECUTE\n', then give **exactly** one action per robot, put each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PICK red_square PLACE panel3\nNAME Bob ACTION WAIT\nNAME Chad ACTION PICK green_trapezoid PLACE panel6\n'
"""

class SortOneBlockTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_sort.xml", 
        one_obj_each: bool = False,
        **kwargs,
    ):    
        self.robot_names = ["ur5e_robotiq", "panda", "ur5e_suction"] 

        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
            "ur5e_suction": "Chad",
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda",
            "Chad": "ur5e_suction",
        }
        self.robots = dict() 
        self.obj_to_panel = dict()
        self.cube_names = ONE_OBJ_EACH
        self.cube_to_bin = dict(
            blue_square="panel2",
            pink_polygon="panel4",
            yellow_trapezoid="panel6",
        )
        
        super(SortOneBlockTask, self).__init__(
            filepath=filepath, 
            task_objects=SORT_ALL_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=UR5E_ROBOTIQ_CONSTANTS,
                panda=PANDA_CONSTANTS,
                ur5e_suction=UR5E_SUCTION_CONSTANTS,
            ),
            **kwargs
        )

        self.panel_coords = dict()
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                self.panel_coords[geom.name] = geom.pos

        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_ROBOTIQ_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **PANDA_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["ur5e_suction"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_SUCTION_CONSTANTS,
        )

        self.align_threshold = 0.1
        self.bin_slot_pos = dict()
        for bin_name in SORTING_BIN_NAMES: 
            for slot in ["middle"]: # ["left", "middle", "right"]:
                self.bin_slot_pos[f"{bin_name}_{slot}"] = self.physics.named.data.site_xpos[f"{bin_name}_{slot}"]
        
        self.robot_to_bin = dict(
            ur5e_robotiq="panel2",
            panda="panel4",
            ur5e_suction="panel6",
        )
        self.bin_x_coords = {
            'panel2': -0.8,
            'panel4': 0,
            'panel6': 0.8,
        }         

        self.cube_targets = dict(
            Alice=("blue_square", "panel2"),
            Bob=("pink_polygon", "panel4"),
            Chad=("yellow_trapezoid", "panel6"),
        )
        self.reachable_panels = dict(
            Alice=["panel1", "panel2", "panel3"],
            Bob=["panel3", "panel4", "panel5"],
            Chad=["panel5", "panel6", "panel7"],
        )
        
    @property
    def use_preplace(self):
        return True

    def get_agent_prompt(self, obs, agent_name, include_response_instructions=True):
        cube_name, bin_name = self.cube_targets[agent_name]
        other_robots = ", ".join(
            [r for r in self.robots.keys() if r != agent_name]
        )
        robot_name = self.get_robot_name(agent_name)
        agent_state = self.describe_robot_state(obs, robot_name)
        agent_state = agent_state.replace(f"{agent_name}'s", "Your")
        
        cube_states = "\n".join(
            [self.describe_cube_state(obs, cube_name) for cube_name in self.cube_names]
        )
        reachable_panels = ", ".join(self.reachable_panels[agent_name])

        agent_prompt = f"""
7 panels on the table, ordered left to right: panel1,...,panel7. They form a straight assembly line, panel1 is closed to panel2 and farthest from panel7.
You are robot {agent_name} in front of {bin_name}. You are collaborating with {other_robots} to sort cubes into their target panels. The task is NOT done until all three cubes are sorted.
At current round: 
{cube_states}
Your goal is to place {cube_name} on {bin_name}, but you can only reach {reachable_panels}: this means you can only pick cubes from these panels, and can only place cubes on these panels.
{agent_state}
Never forget you are {agent_name}! Never forget you can only reach {reachable_panels}!
Think step-by-step about the task and others' response. Carefully check and correct them if they made a mistake. 
Improve your plans if given [Environment Feedback].
"""
        if include_response_instructions:
            agent_prompt += f"""
When you respond, tell others about your goal and all constraints. Respond very concisely but informatively, and do not repeat what others have said.
Discuss with others to come up with the best plan, e.g. if your cube is out of your reach, ask others for help, and you can do the same for them. 
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion, or 2) If everyone has made proposals and got approved, output EXECUTE and the final plan, must strictly follow [Action Output Instruction]!
In the plan, at least one robot should be acting, you can't all WAIT.
"""
# Example response #1:
# [Reasons] I am {agent_name}, I must put blue_square on panel2, but I can't reach blue_square for now. Since Chad needs yellow_trapezoid, I propose to help Chad move it closer. What does everyone think?
# [Proposal] PICK yellow_trapezoid PLACE panel3
# [Decision] PROCEED
# Example response #2:
# [Reasons] I am Chad, My previous proposal was approved and no need for update. I approve the latest proposals from Alice and Bob.
# [Proposal] WAIT 
# [Decision] 
# EXECUTE\nNAME Alice ACTION WAIT\nNAME Bob ACTION PICK blue_square PLACE panel3\nNAME Chad WAIT
        

        # if agent_name == "Alice":
        #     agent_prompt += f"You must put blue_square in panel2" #you can only reach panel2, panel1, panel3. But you can't reach panel5, panel7, or other bins."
        # elif agent_name == "Bob":
        #     agent_prompt += "You must put pink_polygon in panel4" # you can only reach panel4, panel3, panel5. But you can't reach panel1, panel7, or other bins."
        # elif agent_name == "Chad":
        #     agent_prompt += "You must put yellow_trapezoid in panel6" #you can only reach panel6, panel5, panel7. But you can't reach panel1, panel3, or other bins."
 
        return agent_prompt

        
    def get_action_prompt(self) -> str:
        return SORTING_ACTION_SPACE

    def get_plan_state_prompt(self, obs: EnvState) -> str:
        """Compact structured state used by the centralized planner."""
        lines = ["[Structured Task State]"]
        for cube_name, target_panel in self.cube_to_bin.items():
            current_panel = self.get_cube_panel(obs, cube_name)
            status = "done" if self.is_cube_done(obs, cube_name) else "not_done"
            reachable_by = [
                agent_name
                for agent_name in self.robots.keys()
                if self.can_agent_reach_cube(obs, agent_name, cube_name)
            ]
            lines.append(
                f"- {cube_name}: current={current_panel}, target={target_panel}, "
                f"status={status}, reachable_by={reachable_by or ['none']}"
            )
        return "\n".join(lines) + "\n"

    def is_cube_done(self, obs: EnvState, cube_name: str) -> bool:
        correct_panel = self.cube_to_bin[cube_name]
        cube_state = obs.objects[cube_name]
        bin_pos = self.bin_slot_pos[f"{correct_panel}_middle"]
        return (
            np.linalg.norm(bin_pos[:2] - cube_state.xpos[:2]) <= self.align_threshold
            or correct_panel in cube_state.contacts
        )

    def can_agent_reach_cube(self, obs: EnvState, agent_name: str, cube_name: str) -> bool:
        robot_name = self.robot_name_map_inv[agent_name]
        cube_state = obs.objects[cube_name]
        top_site = cube_state.sites[f"{cube_name}_top"]
        return self.check_reach_range(robot_name, top_site.xpos)

    def can_agent_place_panel(self, agent_name: str, panel_name: str) -> bool:
        target_pos = self.get_target_pos(agent_name, panel_name)
        if target_pos is None:
            return False
        robot_name = self.robot_name_map_inv[agent_name]
        return self.check_reach_range(robot_name, target_pos)

    def _agent_holding_cube(self, obs: EnvState, agent_name: str) -> Optional[str]:
        robot_name = self.robot_name_map_inv[agent_name]
        contacts = getattr(obs, robot_name).contacts
        held_cubes = [c for c in contacts if c in self.cube_names]
        return held_cubes[0] if len(held_cubes) > 0 else None

    def _panel_occupancy(self, obs: EnvState) -> Dict[str, List[str]]:
        occupancy: Dict[str, List[str]] = {}
        for cube_name in self.cube_names:
            if self.is_cube_done(obs, cube_name):
                continue
            panel_name = self.get_cube_panel(obs, cube_name)
            occupancy.setdefault(panel_name, []).append(cube_name)
        return occupancy

    def get_allowed_action_names(self) -> Set[str]:
        return {"PICK", "PLACE", "WAIT"}

    def get_max_parallel_actions(self, obs: Optional[EnvState] = None) -> int:
        """Allow limited safe parallelism for Sort.

        Bob is the middle robot and shares panel3/panel5 handoff regions with
        both neighbors, so Bob is kept serial.  Alice and Chad can sometimes
        act at opposite ends without intersecting workspaces; the semantic
        verifier below checks that compatibility.
        """
        return 2

    def _parse_sort_action(self, action: str) -> Tuple[Optional[str], Optional[str]]:
        if action.startswith("PICK ") and "PLACE" in action:
            cube_name = action.split("PICK", 1)[1].split("PLACE", 1)[0].strip()
            target_panel = action.split("PLACE", 1)[1].strip()
            return cube_name, target_panel
        if action.startswith("PLACE "):
            parts = action.split("PLACE", 1)[1].strip().split()
            if len(parts) < 2:
                return None, None
            return parts[0], parts[1]
        return None, None

    def _panel_index(self, panel_name: Optional[str]) -> Optional[int]:
        if panel_name is None or not panel_name.startswith("panel"):
            return None
        try:
            return int(panel_name.replace("panel", ""))
        except ValueError:
            return None

    def _sort_action_footprint(self, obs: EnvState, action: str) -> Set[int]:
        """Approximate panel corridor touched by a high-level action."""
        cube_name, target_panel = self._parse_sort_action(action)
        if cube_name is None or target_panel is None:
            return set()
        start_panel = self.get_cube_panel(obs, cube_name)
        start_idx = self._panel_index(start_panel)
        target_idx = self._panel_index(target_panel)
        if start_idx is None or target_idx is None:
            return set()
        low, high = sorted([start_idx, target_idx])
        return set(range(low, high + 1))

    def _sort_parallel_compatible(self, obs: EnvState, active: List[Tuple[str, str]]) -> Tuple[bool, str]:
        if len(active) <= 1:
            return True, "OK"
        if len(active) > 2:
            return False, "Sort allows at most two active robots."

        parsed = [(agent, *self._parse_sort_action(action), action) for agent, action in active]
        cubes = [cube for _, cube, _, _ in parsed]
        targets = [target for _, _, target, _ in parsed]
        if len(set(cubes)) != len(cubes):
            return False, f"Parallel actions cannot move the same cube: {cubes}"
        if len(set(targets)) != len(targets):
            return False, f"Parallel actions cannot place into the same target: {targets}"

        footprints = {
            agent: self._sort_action_footprint(obs, action)
            for agent, action in active
        }
        active_agents = list(footprints.keys())
        for i, agent_i in enumerate(active_agents):
            for agent_j in active_agents[i + 1:]:
                if not footprints[agent_i].isdisjoint(footprints[agent_j]):
                    return False, f"Parallel action footprints overlap: {footprints}"

        return True, "OK"

    def _sort_plan_risk(self, obs: EnvState, active: List[Tuple[str, str]]) -> Tuple[int, List[str]]:
        """Soft risk model for ranking, not a broad hard constraint.

        The older Sort policy encoded many route/parallel restrictions as hard
        legality checks.  That helped some seeds but reduced generalization.
        Here we keep only obvious collision/duplicate conflicts as hard checks
        and use this score to prefer safer plans while still leaving alternative
        legal actions available to the LLM/verifier.
        """
        risk = 0
        notes: List[str] = []
        if len(active) <= 1:
            return risk, notes
        footprints = {
            agent: self._sort_action_footprint(obs, action)
            for agent, action in active
        }
        agents = [agent for agent, _ in active]
        if "Bob" in agents:
            risk += 25
            notes.append("Bob is the middle robot; parallel Bob motion is possible but riskier.")
        shared_panels = {3, 5}
        touched_shared = {
            agent: fp.intersection(shared_panels)
            for agent, fp in footprints.items()
        }
        if sum(1 for touched in touched_shared.values() if touched) >= 2:
            risk += 35
            notes.append("Multiple robots touch shared handoff panels in the same round.")
        return risk, notes

    def verify_plan_semantics(self, obs: EnvState, actions: Dict[str, str]) -> Tuple[bool, str]:
        """Task-specific hard validation hook.

        This intentionally checks only hard constraints.  Route preference,
        Bob-serial preference, and handoff-panel occupancy are handled by
        scoring/recommendation, not by rejecting otherwise physical candidates.
        """
        if set(actions.keys()) != set(self.robots.keys()):
            return False, f"Expected actions for {list(self.robots.keys())}, got {list(actions.keys())}"

        all_done = all(self.is_cube_done(obs, cube_name) for cube_name in self.cube_names)
        active = [(agent, action) for agent, action in actions.items() if action != "WAIT"]
        if not all_done and len(active) == 0:
            return False, "All robots WAIT while cubes remain unsorted."
        compatible, reason = self._sort_parallel_compatible(obs, active)
        if not compatible:
            return False, reason

        legal_actions = self.get_legal_actions(obs)
        for agent_name, action in actions.items():
            if action not in legal_actions.get(agent_name, []):
                return False, f"{agent_name} action '{action}' is not in current legal actions."
            if action == "WAIT":
                continue
            held_cube = self._agent_holding_cube(obs, agent_name)
            if not (action.startswith("PICK ") or action.startswith("PLACE ")):
                return False, f"{agent_name} action must be PICK <cube> PLACE <panel>, PLACE <cube> <panel>, or WAIT."
            cube_name, target_panel = self._parse_sort_action(action)
            if cube_name not in self.cube_names:
                return False, f"Unknown cube {cube_name}."
            if action.startswith("PICK ") and held_cube is not None:
                return False, f"{agent_name} is already holding {held_cube} and cannot PICK."
            if action.startswith("PLACE ") and held_cube != cube_name:
                return False, f"{agent_name} holds {held_cube}, but tried to PLACE {cube_name}."
            if self.is_cube_done(obs, cube_name) and not action.startswith("PLACE "):
                return False, f"{cube_name} is already done and should not be moved."
            if not self.can_agent_place_panel(agent_name, target_panel):
                return False, f"{agent_name} cannot place to {target_panel}."
        return True, "OK"

    def _sort_route_targets(self, obs: EnvState, agent_name: str, cube_name: str) -> List[str]:
        """Directed handoff policy for sorting.

        The old free-form prompt allowed moving objects back to arbitrary shared
        panels.  That can create states where no robot can pick the object
        again.  These route targets only move cubes toward their fixed goals:
        Alice <-> Bob handoff is panel3, Bob <-> Chad handoff is panel5.
        """
        if self.is_cube_done(obs, cube_name):
            return []

        if cube_name == "blue_square":
            # Final owner is Alice. From the right, move left through panel5 -> panel3 -> panel2.
            route = {
                "Alice": ["panel2"],
                "Bob": ["panel3"],
                "Chad": ["panel5"],
            }
        elif cube_name == "pink_polygon":
            # Final owner is Bob. Use panel3 or panel5 only as directed handoff panels.
            route = {
                "Alice": ["panel3"],
                "Bob": ["panel4"],
                "Chad": ["panel5"],
            }
        elif cube_name == "yellow_trapezoid":
            # Final owner is Chad. Prefer Bob->panel5; Alice->panel3 is last resort only.
            bob_or_chad_can_pick = any(
                self.can_agent_reach_cube(obs, helper, cube_name)
                for helper in ["Bob", "Chad"]
            )
            route = {
                "Alice": [] if bob_or_chad_can_pick else ["panel3"],
                "Bob": ["panel5"],
                "Chad": ["panel6"],
            }
        else:
            route = {}
        return route.get(agent_name, [])

    def get_legal_actions(self, obs: EnvState) -> Dict[str, List[str]]:
        """Return broad physically-plausible high-level action candidates.

        Legal actions are no longer the same as the preferred route.  A legal
        PICK/PLACE must satisfy current gripper state and reachability; the
        recommender below scores whether it is good progress toward the goal.
        This keeps code2's LLM/verifier pipeline flexible while still preventing
        clearly impossible actions.
        """
        legal_actions = {agent_name: ["WAIT"] for agent_name in self.robots.keys()}
        for agent_name in self.robots.keys():
            held_cube = self._agent_holding_cube(obs, agent_name)
            if held_cube is not None:
                for panel_name in self.reachable_panels[agent_name]:
                    if self.can_agent_place_panel(agent_name, panel_name):
                        legal_actions[agent_name].append(f"PLACE {held_cube} {panel_name}")
                continue
            for cube_name in self.cube_names:
                if self.is_cube_done(obs, cube_name):
                    continue
                if not self.can_agent_reach_cube(obs, agent_name, cube_name):
                    continue
                current_panel = self.get_cube_panel(obs, cube_name)
                for panel_name in self.reachable_panels[agent_name]:
                    if panel_name == current_panel:
                        continue
                    if not self.can_agent_place_panel(agent_name, panel_name):
                        continue
                    legal_actions[agent_name].append(f"PICK {cube_name} PLACE {panel_name}")
        return legal_actions

    def _action_progress(self, obs: EnvState, cube_name: str, target_panel: str) -> int:
        current_panel = self.get_cube_panel(obs, cube_name)
        current_idx = self._panel_index(current_panel)
        target_idx = self._panel_index(self.cube_to_bin[cube_name])
        next_idx = self._panel_index(target_panel)
        if current_idx is None or target_idx is None or next_idx is None:
            return 0
        return abs(current_idx - target_idx) - abs(next_idx - target_idx)

    def _score_sort_action(self, obs: EnvState, agent_name: str, action: str) -> int:
        if action == "WAIT":
            return 0
        cube_name, target_panel = self._parse_sort_action(action)
        if cube_name is None or target_panel is None:
            return -1000

        score = 0
        final_panel = self.cube_to_bin[cube_name]
        progress = self._action_progress(obs, cube_name, target_panel)
        current_panel = self.get_cube_panel(obs, cube_name)

        if target_panel == final_panel:
            score += 120
        score += 35 * progress

        preferred_targets = self._sort_route_targets(obs, agent_name, cube_name)
        if target_panel in preferred_targets:
            score += 25

        if target_panel == current_panel:
            # Same-panel replacement is a recovery tool, not a normal routing
            # action.  Keep it available only through explicit PLACE recovery or
            # if legal generation later adds it, but do not recommend it by
            # default over true progress.
            score -= 60

        occupancy = self._panel_occupancy(obs)
        if target_panel in occupancy and cube_name not in occupancy[target_panel]:
            score -= 45

        if action.startswith("PLACE "):
            score += 30

        # Small deterministic tie-breakers that prefer clearing outer objects
        # without hard-coding route legality.
        priority = {
            "yellow_trapezoid": 3,
            "blue_square": 2,
            "pink_polygon": 1,
        }.get(cube_name, 0)
        score += priority
        return score

    def get_recommended_plan(self, obs: EnvState) -> Dict[str, str]:
        """Score legal candidates and recommend the best low-risk combination."""
        legal_actions = self.get_legal_actions(obs)
        plan = {agent_name: "WAIT" for agent_name in self.robots.keys()}

        candidates = []
        for agent_name, actions in legal_actions.items():
            for action in actions:
                if action == "WAIT":
                    continue
                cube_name, target_panel = self._parse_sort_action(action)
                score = self._score_sort_action(obs, agent_name, action)
                candidates.append((score, agent_name, cube_name, target_panel, action))

        from itertools import combinations
        best_combo: Tuple[Tuple[int, str, str, str, str], ...] = ()
        best_score = -10**9
        max_parallel = min(self.get_max_parallel_actions(obs), len(candidates))
        for size in range(1, max_parallel + 1):
            for combo in combinations(candidates, size):
                agents = [agent for _, agent, _, _, _ in combo]
                cubes = [cube for _, _, cube, _, _ in combo]
                targets = [target for _, _, _, target, _ in combo]
                if len(set(agents)) != len(agents):
                    continue
                if len(set(cubes)) != len(cubes):
                    continue
                if len(set(targets)) != len(targets):
                    continue
                active = [(agent, action) for _, agent, _, _, action in combo]
                compatible, _ = self._sort_parallel_compatible(obs, active)
                if not compatible:
                    continue
                risk, _ = self._sort_plan_risk(obs, active)
                score = sum(s for s, _, _, _, _ in combo) - risk + 5 * size
                if score > best_score:
                    best_score = score
                    best_combo = combo

        for _, agent_name, _, _, action in best_combo:
            plan[agent_name] = action
        return plan

    def format_legal_actions_prompt(self, obs: EnvState) -> str:
        legal_actions = self.get_legal_actions(obs)
        recommended = self.get_recommended_plan(obs)
        lines = [
            "[Legal Actions]",
            "You must choose exactly one listed action for each robot. Do not invent actions.",
            f"For this task, choose at most {self.get_max_parallel_actions(obs)} non-WAIT actions per round.",
            "Parallel actions are allowed when their panel corridors do not overlap; Bob-parallel motions are riskier but not always forbidden.",
            "Actions marked recommended are high-scoring soft preferences, not the only valid actions.",
        ]
        for agent_name, actions in legal_actions.items():
            lines.append(f"{agent_name}:")
            for action in actions:
                prefix = " (recommended)" if recommended.get(agent_name) == action else ""
                lines.append(f"- {action}{prefix}")
        lines.append("[Recommended Plan]")
        lines.append("EXECUTE")
        for agent_name in self.robots.keys():
            lines.append(f"NAME {agent_name} ACTION {recommended[agent_name]}")
        return "\n".join(lines) + "\n"

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]
    
    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs
    
    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.4, -0.1), y=(0.1, 1.3), z=(0.16, 1))
        
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-0.7, 0.7), y=(-0.21, 1.3), z=(0.16, 1))
        
        elif robot_name == "ur5e_suction" or robot_name == self.robot_name_map["ur5e_suction"]:
            return dict(x=(0.2, 1.5), y=(0.1, 1.3), z=(0.16, 1))
        
        else:
            raise NotImplementedError
    
    def check_reach_range(self, robot_name, point: Tuple[float, float, float]) -> bool: 
        reach_range = self.get_robot_reach_range(robot_name)
        for i, axis in enumerate(["x", "y", "z"]):
            if point[i] < reach_range[axis][0] or point[i] > reach_range[axis][1]:
                return False
        return True

    def sample_initial_scene(self):
        # find the pre-defined panel positions in the xml
        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                tosample_panels.append(
                    (geom.name, geom.pos, geom.size)
                )
        assert len(tosample_panels) >= 3, "Not enough panel positions to sample from"
        
        far_panels = dict()
        far_panels['square'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] > 0.15] 
        far_panels['polygon'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.7 or tup[1][0] > 0.9]
        far_panels['trapezoid'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.15]

        # sample the panel positions
        occupied_idxs = []
        for i, name in enumerate(ONE_OBJ_EACH):
            try:
                qpos_slice = self.physics.named.data.qpos._convert_key(f"{name}_joint")
            except KeyError:
                print('Skipping object: ', name, ' because its _joint does not exist in the xml file')
                continue
            assert int(qpos_slice.stop - qpos_slice.start) == 7, "object qpos must be 7-dim"
            start = qpos_slice.start
            stop = qpos_slice.stop
            shape = name.split('_')[1]
            
            idx = self.random_state.choice(far_panels[shape]) 
            # remove this index from the list of available panels
            for shape, idxs in far_panels.items():
                if idx in idxs:
                    idxs.remove(idx)

            panel_name, panel_pos, panel_size = tosample_panels[idx]
            self.obj_to_panel[name] = panel_name
            # sample a random position within the panel
            # new_pos = self.random_state.uniform(
            #     low=panel_pos - panel_size / 2 + 0.001,
            #     high=panel_pos + panel_size / 2 - 0.001, 
            # )
            new_pos = panel_pos.copy()
 
            new_quat = Quaternion(
                axis=[0,0,1], 
                angle=self.random_state.uniform(low=0, high=2*np.pi)
                ) 
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            old_pos = self.physics.named.data.qpos[start:stop]
            new_pos[2] = old_pos[2]
            self.physics.named.data.qpos[start:stop] = np.concatenate([new_pos, new_quat])
            
             
        self.physics.forward()
        self.physics.step(100)
    
    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs
    
    def describe_robot_state(self, obs: EnvState, robot_name: str):
        agent_name = self.get_agent_name(robot_name)
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos

        dist_to_panels = [(name, np.linalg.norm(robot_state.ee_xpos[:2] - pos[:2])) for name, pos in self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0]
        robot_desp = "" # f"{agent_name}'s gripper is closest to {closest_panel}, "

        if len(robot_state.contacts) == 0:
            obj = "empty"
        else:
            obj = "holding " + ",".join([c for c in robot_state.contacts])
        # robot_desp += f"{agent_name}'s gripper is at ({x:.2f} {y:.2f} {z:.2f}), holding {obj},"
        robot_desp += f"{agent_name}'s gripper is {obj},"
        
        reachables = []
        not_reachables = []
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name]
            top_site = block_state.sites[f'{block_name}_top']
            if self.check_reach_range(robot_name, top_site.xpos):
                reachables.append(block_name)
            else:
                not_reachables.append(block_name)

        if len(reachables) > 0:
            robot_desp += f" can reach cubes: "
            for obj in reachables:
                robot_desp += f"{obj}, "
        if len(not_reachables) > 0:
            robot_desp += f"can't reach cubes: "
            for obj in not_reachables:
                robot_desp += f"{obj}, "
        
        return robot_desp
    
    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        ret = []
        cube_ids = [self.physics.model.body(cube).id for cube in self.cube_names ]
       
        table_id = self.physics.model.body("table").id 
        bin_ids = [self.physics.model.body(bin_name).id for bin_name in SORTING_BIN_NAMES]
        
        for link_id in self.robots["Alice"].all_link_body_ids + self.robots["Bob"].all_link_body_ids + self.robots["Chad"].all_link_body_ids:
            for cube_id in cube_ids:
                ret.append((link_id, cube_id))
            for bin_id in bin_ids:
                ret.append((link_id, bin_id))
            ret.append((link_id, table_id))

        for cube_id in cube_ids:
            ret.append((cube_id, table_id))
            for cube_id2 in cube_ids:
                if cube_id != cube_id2:
                    ret.append((cube_id, cube_id2))
            for bin_id in bin_ids:
                ret.append((cube_id, bin_id)) 

        return ret
    
    def get_cube_panel(self, obs, cube_name: str):
        cube_state = obs.objects[cube_name]
        dist_to_panels = [(name, np.linalg.norm(cube_state.xpos[:2] - pos[:2])) for name, pos in self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0] 
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in obs.objects[cube_name].contacts:
                closest_panel = pname
                break
        return closest_panel

    def describe_cube_state(self, obs: EnvState, cube_name: str):
        cube_state = obs.objects[cube_name]
        top_site = cube_state.sites[f'{cube_name}_top']
        x, y, z = top_site.xpos
        cube_desp = ""
        for slot_name, pos in self.bin_slot_pos.items():
            if np.linalg.norm(pos[:2] - cube_state.xpos[:2]) < self.align_threshold:
                slot_name = "_".join(slot_name.split("_")[:-1])
                cube_desp = f"{cube_name} is in {slot_name}"
                break 
        if len(cube_desp) == 0:
            closest_panel = self.get_cube_panel(obs, cube_name)
            cube_desp = f"{cube_name} is on {closest_panel}"
        return cube_desp

    def describe_obs(self, obs: EnvState):
        """ For each cube, just describe whether it's on a bin, or between which two bins, no output numerical coordinates """
        object_desp = "[Scene description]\n"  
        for cube_name in ONE_OBJ_EACH:
            object_desp += self.describe_cube_state(obs, cube_name)+"\n" 
        
        robot_desp = ""
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name)+"\n"
        robot_desp = robot_desp[:-2]+".\n"
        full_desp = object_desp + robot_desp
        return full_desp  
    
    def get_reward_done(self, obs):
        reward = 1
        done = True
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name] 
            correct_bin = self.cube_to_bin[block_name]
            bin_pos = self.bin_slot_pos[f"{correct_bin}_middle"]
            if np.linalg.norm(bin_pos[:2] - block_state.xpos[:2]) > self.align_threshold and (correct_bin not in obs.objects[block_name].contacts):
                reward = 0
                done = False
                break
        return reward, done
 

    def describe_task_context(self):
        return SORT_TASK_CONTEXT 
    
    def get_grasp_site(self, obj_name: str = "pink_polygon") -> str:
        return f"{obj_name}_top"
    
    def get_target_pos(self, agent_name, target_name, target_type: str = 'site') -> Optional[np.ndarray]:
        """ useful for parsing place targets """
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]
        if 'panel' in target_name:
            try:
                ret = self.physics.data.geom(target_name).xpos.copy()
            except KeyError:
                return None  

            if target_name == 'panel3':
                # panel3 is the Alice<->Bob handoff panel.  The original
                # agent-dependent offset lets Alice place on the far side of
                # panel3; in observed rollouts Bob then passed reach checks but
                # failed IK when trying to pick the object.  Use a single shared
                # handoff point that Bob can pick and Alice can also reach.
                ret[0] -= 0.12
                ret[1] -= 0.1
            if target_name == 'panel5':
                # panel5 is the Bob<->Chad handoff panel.  Use Bob's original
                # placement side as the shared handoff point; it was observed
                # to be pickable by Chad and remains placeable by Bob.
                ret[0] += 0.12
                ret[1] -= 0.1

            ret[2] = 0.5
        elif target_name in self.cube_names:
            sname = f"{target_name}_top"
            ret = self.physics.data.site(sname).xpos.copy() 
        else:
            ret = None
        # print(f"Agent: {agent_name} target site for {target_name} is {ret}")
 
        return ret 

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix! 
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in self.cube_names]
        contacts["panda"] = [c for c in contacts["panda"] if c in self.cube_names]
        contacts["ur5e_suction"] = [c for c in contacts["ur5e_suction"] if c in self.cube_names]
        return contacts
    
    def get_task_feedback(self, llm_plan, pose_dict):
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            if action_str == "WAIT":
                continue
            if not (action_str.startswith("PICK ") or action_str.startswith("PLACE ")):
                feedback += f"{agent_name}'s ACTION must be PICK <cube> PLACE <panel>, PLACE <cube> <panel>, or WAIT.\n"
                continue
            obj, target = self._parse_sort_action(action_str)
            if obj not in self.cube_names:
                feedback += f"{agent_name}'s ACTION uses unknown cube {obj}.\n"
                continue
            if target is None or not str(target).startswith("panel"):
                feedback += f"{agent_name}'s ACTION must place {obj} onto a panel target.\n"
        if all(['WAIT' in action_str for action_str in llm_plan.action_strs.values()]):
            feedback += f"You can't all WAIT. The task is not complete, at least one robot should be acting."
        return feedback 
    
    def get_object_joint_name(self, obj_name):
        return f"{obj_name}_joint"

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return SORT_TASK_CHAT_PROMPT
        
    def central_plan_prompt(self):
        return SORT_TASK_PLAN_PROMPT
    
    def dialog_mode_prompt(self):
        return SORT_TASK_DIALOG_PROMPT
 
    def get_waypoint_feedback(
        self, 
        waypoint_paths: Dict[str, List],
        display = False,
        save_img = False,
        img_path = 'test.jpg',
        ):
        """
        Give feedback to the robots about the waypoints they are going to visit.
        """
        bad_waypoints = defaultdict(list)
        for robot_name, path in waypoint_paths.items(): 
            for waypoint in path:
                if not self.check_reach_range(robot_name, waypoint):
                    bad_waypoints[robot_name].append(waypoint)
        summ = ""
        for name, waypoints in bad_waypoints.items():
            summ += f"{name}: {waypoints} \n"
        if display:
            print(summ)
            self.render_point_cloud = True 
            obs = self.get_obs() 
            path_ls = list(waypoint_paths.values())
            visualize_voxel_scene(obs.scene, path_pts=path_ls, path_colors=[], save_img=save_img, img_path=img_path)
        if summ == "":
            summ = "Reachability feedback: sucess."
        else:
            summ = "Reachability feedback: failed. These steps are beyond the robot's reach: \n" + summ
        return summ 
            

        


if __name__ == "__main__":
    env = SortOneBlockTask(np_seed=0, render_point_cloud=0)
    obs = env.reset()
    # obs.scene.show()
    print(env.get_action_prompt())
    print(env.get_agent_prompt(obs, "Alice"))
    breakpoint()
    img=env.physics.render(camera_id="teaser", height=480, width=600)
