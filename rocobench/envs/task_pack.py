import os
import copy
import time
import cv2 
import re
import random
import numpy as np  
from pydantic import dataclasses, validator 
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control 
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, PANDA_CONSTANTS

PACK_TASK_OBJECTS=[
    "bin",
    "table_top",
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_ITEM_NAMES=[
    "apple",
    "banana",
    "milk",
    "soda_can",
    "bread",
    "cereal",
]
PACK_BIN_SITE_NAMES=[
    "bin_front_left",
    "bin_front_right",
    "bin_front_middle",
    "bin_back_left",
    "bin_back_right", 
    "bin_back_middle",
]
 
PACK_TASK_CONTEXT="""[Task Description]
Two robots, Alice and Bob, each stands at a different side of the table, and together pack all the grocery items on the table into a bin.
They choose objects closest to their grippers. At each round, they are given [Scene description], [Environment feedback], and must reason about the task. Each robot does **exactly** one ACTION and PATH per round, their PATHs must avoid collision.
"""

PACK_ACTION_SPACE="""
[Action Options]
1) PICK <obj> PATH <path>: only PICK if your gripper is empty;
2) PLACE <obj> bin PATH <path>: only if you have already PICKed the object, you can PLACE it into an empty bin slot, do NOT PLACE if another object is already in a slot!

Each <path> must contain exactly four <coord>s that smoothly interpolate between start and goal, coordinates must be evenly distanced from each other.
The robot PATHs must efficiently reach target while avoiding collision avoid collision (e.g. move above the objects' heights).
The PATHs must do top-down pick or place: 
- move directly atop an object by height 0.2 before PICK: e.g. Alice's gripper is at (0, 0, 0.3), banana is at (-0.25, 0.39, 0.29): NAME Alice ACTION PICK banana PATH [(0, 0.1, 0.3),(0, 0.2, 0.49),(-0.1, 0.25, 0.49),(-0.25, 0.39, 0.49)]
- lift an object vertically up before moving it to PLACE: e.g. Bob's gripper is at (0.9, 0, 0.2), bin_front_left is at (0.35, 0.35, 0.43): NAME Bob ACTION PLACE apple bin_front_left PATH [(0.90,0.00,0.55), (0.70,0.62,0.62), (0.52,0.60,0.62), (0.35,0.35,0.55)]

[Packing PLACE path rule]
When one or both robots PLACE held grocery items into the bin, use high and separated corridors instead of crossing through the table center.
- Use z about 0.55-0.65 for the middle two PLACE waypoints, then descend only near the final bin slot.
- If Alice and Bob both PLACE in the same round, choose non-adjacent/distant empty bin slots. The two target slot XY positions should be at least 0.35 apart. Avoid same-column, same-row-neighbor, and close diagonal pairs such as bin_back_middle with bin_front_right; instead prefer opposite-side pairs such as Alice->bin_back_left and Bob->bin_front_right when both are empty.
- If Alice and Bob both PLACE in the same round, do NOT route both robots through the central low area around x=0.0-0.35, y=0.45-0.58, z<0.55. Keep same-index middle waypoints reasonably separated in XY (about 0.25 or more) until the final descent.
- Alice should use a high left/front corridor: middle waypoints around x<=0.15, y<=0.52, z=0.60-0.68, then approach her final slot only at the last waypoint.
- Bob should use a high back/right corridor: middle waypoints around y=0.60-0.70 and z=0.60-0.68 before approaching the bin. Use waypoints such as (0.45,0.64,0.64) or (0.60,0.64,0.64), not low/central waypoints.
- In particular, Bob must avoid low or central middle waypoints near (0.20, 0.50, 0.40), (0.20,0.54,0.62), or (0.35,0.50,0.62); use high separated waypoints such as (0.45,0.64,0.64) or (0.60,0.64,0.64) instead.

[Action Output Instruction]
First output 'EXECUTE\n', then give exactly one ACTION per robot, each on a new line.
Example: 'EXECUTE\nNAME Alice ACTION PICK apple PATH <path>\nNAME Bob ACTION PLACE banana bin_back_middle PATH <path>\n'
"""

PACK_CHAT_PROMPT="""Robots discuss to find the best strategy and path. When each robot talk, it first reflects on the task status and its own capability. 
Carefully consider [Environment Feedback]. Coordinate with others to plan and improve paths following the instructions. They talk in order [Alice],[Bob],[Alice],..., then, after they agreed, plan exactly one ACTION per robot, output an EXECUTE to summarize the plan and stop talking.
Their discussion and the final plan: """

class PackGroceryTask(MujocoSimEnv):
    def __init__( 
        self,
        filepath: str = "rocobench/envs/task_pack.xml",
        one_obj_each: bool = False,
        **kwargs,
    ):    
        self.robot_names = ["ur5e_robotiq", "panda"] 
        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob", 
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda", 
        }
        self.robots = dict()  

        robotiq_config = UR5E_ROBOTIQ_CONSTANTS.copy()  
        panda_config = PANDA_CONSTANTS.copy() 

        self.item_names = PACK_ITEM_NAMES

        super(PackGroceryTask, self).__init__(
            filepath=filepath,  
            task_objects=PACK_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=robotiq_config,
                panda=panda_config, 
            ),
            **kwargs
        ) 
        
        self.bin_slot_xposes = dict()
        for sname in PACK_BIN_SITE_NAMES:
            self.bin_slot_xposes[sname] = self.physics.data.site(sname).xpos.copy()

        self.robots[
            self.robot_name_map["ur5e_robotiq"]
            ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **robotiq_config,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **panda_config,
        )
         
        self.align_threshold = 0.06
    
    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]: 
        ret = None 
        robot_name = self.robot_name_map_inv[agent_name]

        if target_name in self.item_names:
            sname = f"{target_name}_top"  
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xpos.copy() 
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass

        return ret

    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if target_name in self.item_names:
            sname = f"{target_name}_top" 
        elif target_name in self.bin_slot_xposes.keys():
            sname = target_name
        else:
            return None 
        try:
            ret = self.physics.data.site(sname).xmat.copy().reshape(3, 3)
            ret = mat_to_quat(ret)
            if any([name in sname for name in ['apple', 'soda_can', 'milk']]):
                # change quat
                if agent_name == "Bob":
                    ret = np.array([1, 0, 0, 1])
                else:
                    ret = np.array([1, 0, 0, 0])
            if 'bin_' in target_name and agent_name == "Bob":
                ret = np.array([1, 0, 0, 1])
        except KeyError:
            print(f"KeyError: {sname} not in model sites")
            pass
        return ret 
    
    @property 
    def use_prepick(self):
        return False  

    @property
    def use_preplace(self):
        return False
    
    @property
    def waypoint_std_threshold(self):
        return 0.19

    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        
        bin_id = self.physics.model.body("bin").id
        bin_bottom_id = self.physics.model.body("bin_inside").id
        table_id = self.physics.model.body("table").id

        ret = [(table_id, bin_bottom_id)]
        all_body_ids = []
        for obj_name in self.item_names:
            body_ids = self.get_all_body_ids(obj_name)
            for body_id in body_ids:
                ret.append((body_id, bin_bottom_id))
                # ret.append((body_id, bin_id)) this makes direct path less likely
                ret.append((body_id, table_id))
                all_body_ids.append(body_id)

        ee_link_ids = self.robots["Alice"].ee_link_body_ids + self.robots["Bob"].ee_link_body_ids
        ee_link_ids = [_id for _id in ee_link_ids if _id != "panda_hand"]

        return ret 

    def get_graspable_objects(self):
        graspables = self.item_names.copy()
        return dict(
            Alice=graspables,
            Bob=graspables, 
        )

    def get_grasp_site(self, obj_name: str = "apple") -> Optional[str]:
        if obj_name in self.item_names:
            return f"{obj_name}_top"
        else:
            return None

    def get_object_joint_name(self, obj_name: str) -> str:
        return f"{obj_name}_joint"

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]
    
    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name] 

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.3, 1.6), y=(-0.4, 1.5), z=(0, 1))
        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-1.3, 1.6), y=(0, 1.5), z=(0, 1))
        else:
            raise NotImplementedError
    
    def sample_initial_scene(self): 
        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'grid' in geom.name:
                low = geom.pos - geom.size
                high = geom.pos + geom.size
                tosample_panels.append(
                    (low, high)
                )
        assert len(tosample_panels) >= len(self.item_names), "Not enough grid positions to sample from"
        panel_idxs = self.random_state.choice(
            len(tosample_panels), 
            len(self.item_names),
            replace=False
            )
        for _idx, item_name in zip(panel_idxs, self.item_names):
            low, high = tosample_panels[_idx]
            new_pos = self.random_state.uniform(low, high) 
            new_pos[2] = self.physics.data.body(item_name).xpos[2] # height stays same!
            new_quat = Quaternion(
                axis=[0,0,1], 
                angle=self.random_state.uniform(low=0, high=2*np.pi)
                ) 
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            self.reset_body_pose(
                body_name=item_name,
                pos=new_pos,
                quat=new_quat,
            )  
            self.reset_qpos(
                jnt_name=f"{item_name}_joint",
                pos=new_pos,
                quat=new_quat,
            )
          
        self.physics.forward()
        self.physics.step(50)
    
    def get_obs(self) -> EnvState:
        contacts = self.get_contact()
        allow_objs = self.item_names + ["bin", "table"]
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in allow_objs]
        contacts["panda"] = [c for c in contacts["panda"] if c in allow_objs]

        obj_states = self.get_object_states(contact_dict=contacts)
        agent_states = dict()
        for agent_name, agent_constants in self.agent_configs.items():
            agent_state = self.get_agent_state(
                agent_constants, contact_dict=contacts
            ) 
            agent_states[agent_name] = agent_state
        kwargs = dict(
            objects=obj_states,
        )
        kwargs.update(agent_states)
        if self.render_point_cloud:
            point_cloud = self.get_point_cloud()
            kwargs['scene'] = point_cloud # NOTE: should include bboxes! 
        obs = EnvState(**kwargs)
         
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation" 
        return obs
    
    def get_reward_done(self, obs): 
        all_packed = True
        reward = 1
        for food in self.item_names:
            bin_coord = self.physics.data.body("bin").xpos[:2]
            dist = np.linalg.norm(obs.objects[food].xpos[:2] - bin_coord)
            if 'bin_inside' not in obs.objects[food].contacts and dist > self.align_threshold:
                all_packed = False 
                reward = 0
                break 
        return reward, all_packed

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix! 
        robotiq_link_names = self.agent_configs["ur5e_robotiq"]['all_link_names'] + ['ur5e_robotiq']
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c not in robotiq_link_names] 

        panda_link_names = self.agent_configs["panda"]['all_link_names'] + ["panda_right_finger", "panda_left_finger", "panda"]
        contacts["panda"] = [c for c in contacts['panda'] if c not in panda_link_names] 
        contacts["panda"].append("broom")

        return contacts

    def central_plan_prompt(self, chat_history: List[str] = []):
        return PACK_PLAN_PROMPT 

    def get_action_prompt(self) -> str:
        return PACK_ACTION_SPACE

    def describe_object(self, obs, name):
        x,y,z = self.physics.data.site(f"{name}_top").xpos
        z += 0.05 # further avoid collision
        contacts = obs.objects[name].contacts 
        object_desp = f"{name}: ({x:.2f}, {y:.2f}, {z:.2f}), "
        if 'bin_inside' in contacts:
            dist_to_slot = [
                (
                    slot_name, np.linalg.norm(np.array([x,y]) - slot_xpos[:2])
                ) for slot_name, slot_xpos in self.bin_slot_xposes.items()

            ]
            slot_name = min(dist_to_slot, key=lambda x: x[1])[0]
            object_desp += f"inside slot {slot_name}"
        else:
            object_desp += f"on table"
        return object_desp

    def describe_robot_state(self, obs, robot_name):
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos
        contacts = robot_state.contacts 
        contacts = [c for c in contacts if c in self.item_names]
        obj = contacts[0] if len(contacts) > 0 else "nothing"
        agent_name = self.robot_name_map[robot_name]
        robot_desp = f"{agent_name}'s gripper: ({x:.2f}, {y:.2f}, {z:.2f}), holding {obj}" 
        return robot_desp
    
    def describe_obs(self, obs: EnvState):
        full_desp =  "[Scene description]\n" 
        table_height = self.physics.data.body("table_top").xpos[2] + 0.15
        full_desp += f"robots must move lower than 0.8 but higher than table height {table_height:.2f}\n"
        for name in self.item_names:
            full_desp += self.describe_object(obs, name) + "\n"

        for slot_name, slot_xpos in self.bin_slot_xposes.items():
            x, y, z = slot_xpos
            full_desp += f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
 
        for robot_name, agent_name in self.robot_name_map.items():
            full_desp += self.describe_robot_state(obs, robot_name) + "\n"
            
        return full_desp 
    
    def describe_task_context(self):
        return PACK_TASK_CONTEXT

    def get_occupied_bin_slots(
        self,
        exclude_items: Optional[Set[str]] = None,
        xy_threshold: float = 0.14,
    ) -> Dict[str, List[str]]:
        """Return bin slots that already contain or are very close to items.

        The text observation labels an object as inside the nearest slot when it
        contacts the bin, but late-stage packing can still fail if a new object
        is placed close to an already packed item.  Use XY distance to slot
        centers as an additional guard and ignore items currently being placed.
        """
        exclude_items = exclude_items or set()
        occupied = {slot_name: [] for slot_name in self.bin_slot_xposes}

        try:
            obs = self.get_obs()
        except Exception:
            obs = None

        for item_name in self.item_names:
            if item_name in exclude_items:
                continue

            try:
                item_xy = self.physics.data.site(f"{item_name}_top").xpos[:2]
            except KeyError:
                continue

            nearest_slot = None
            nearest_dist = float("inf")
            for slot_name, slot_xpos in self.bin_slot_xposes.items():
                dist = float(np.linalg.norm(item_xy - slot_xpos[:2]))
                if dist < nearest_dist:
                    nearest_slot = slot_name
                    nearest_dist = dist

            item_contacts = []
            if obs is not None:
                try:
                    item_contacts = obs.objects[item_name].contacts
                except Exception:
                    item_contacts = []

            if nearest_slot is not None and (
                nearest_dist < xy_threshold or "bin_inside" in item_contacts
            ):
                occupied[nearest_slot].append(item_name)

        return {slot: items for slot, items in occupied.items() if len(items) > 0}
    
    def get_agent_prompt(self, obs, agent_name):        
        robot_name = self.get_robot_name(agent_name)
        other_robot = "Alice" if agent_name == "Bob" else "Bob"
        object_desp = "\n".join([self.describe_object(obs, name) for name in self.item_names])

        table_height = self.physics.data.body("table_top").xpos[2] + 0.15 
        robot_desp = self.describe_robot_state(obs, robot_name).replace(f"{agent_name}'s", "Your")
        slot_desp = "\n".join(
            [
                f"{slot_name}: ({x:.2f}, {y:.2f}, {z:.2f})" for slot_name, (x,y,z) in self.bin_slot_xposes.items()
            ]
            )

        agent_prompt = f"""
You are {agent_name}, you and robot {other_robot} each stands at a different side of the table, and together you must put all the grocery items into a bin.
Locations of slots in the bin:
{slot_desp}
At current round:
You see the following objects:
{object_desp}
{robot_desp}
Your gripper must move higher than these objects and higher than table height {table_height:.2f}, but move lower than 0.8.
For PLACE in the packing task, first lift to a high carrying corridor. If both robots PLACE, choose non-adjacent/distant empty bin slots whose XY positions are at least 0.35 apart; avoid close pairs such as bin_back_middle with bin_front_right, and prefer opposite-side pairs such as Alice->bin_back_left and Bob->bin_front_right when possible. Keep corridors separated until final descent: Alice should use a high left/front corridor with middle waypoints around x<=0.15, y<=0.52, z=0.60-0.68, while Bob should use a high back/right corridor with middle waypoints around y=0.60-0.70 and z=0.60-0.68. Bob must avoid low/central middle waypoints near (0.20, 0.50, 0.40), (0.20,0.54,0.62), or (0.35,0.50,0.62). Same-index middle waypoints should stay about 0.25 or more apart in XY.
Never forget you are {agent_name}!
Think step-by-step about the task and {other_robot}'s response. Carefully check and correct {other_robot} if they made a mistake. 
Discuss with {other_robot} to come up with the best plan and smooth, collision-free paths. 
Improve your paths if given [Environment Feedback], choose a different object or target slot if needed.

When you respond, tell {other_robot} about your status. Respond very concisely but informatively, and do not repeat what others have said.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction] and [Path Plan Instruction].
"""
        return agent_prompt
    
    def get_task_feedback(self, llm_plan, pose_dict): 
        feedback = ""
        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' not in action_str and 'PLACE' not in action_str:
                feedback += f"{agent_name}'s ACTION is invalid, can only PICK or PLACE"

        placing_items = set()
        place_slots = {}
        for agent_name, action_str in llm_plan.action_strs.items():
            match = re.search(r"\bPLACE\s+(\S+)\s+(bin_\S+)\s+PATH\b", action_str)
            if match:
                placing_items.add(match.group(1))
                place_slots[agent_name] = match.group(2)

        # Reject PLACE targets that are already occupied by previously packed
        # items.  This catches failures such as run_28, where bread/cereal were
        # repeatedly planned into slots already containing soda_can/milk.
        if len(place_slots) > 0:
            occupied_slots = self.get_occupied_bin_slots(exclude_items=placing_items)
            for agent_name, slot_name in place_slots.items():
                blocking_items = occupied_slots.get(slot_name, [])
                if len(blocking_items) > 0:
                    empty_slots = [
                        slot for slot in PACK_BIN_SITE_NAMES
                        if slot not in occupied_slots and slot not in place_slots.values()
                    ]
                    empty_slot_msg = (
                        f" Empty slots appear to be: {', '.join(empty_slots)}."
                        if len(empty_slots) > 0
                        else ""
                    )
                    feedback += (
                        f"{agent_name} cannot PLACE into {slot_name}: that slot "
                        f"is already occupied or too close to packed item(s) "
                        f"{', '.join(blocking_items)}. Choose a truly empty bin "
                        f"slot away from existing packed items.{empty_slot_msg} "
                    )

        # Packing-specific path sanity checks.  The generic waypoint collision
        # checks only test each discrete waypoint; a low path through the table
        # center can still pass those checks but make the downstream BiRRT spend
        # a long time searching around narrow passages.  Reject those plans early
        # and ask the LLM to use the high separated corridors described above.
        if all('PLACE' in action_str for action_str in llm_plan.action_strs.values()):
            # Do not let simultaneous PLACE actions target adjacent/nearby slots:
            # in practice this frequently puts both grippers into the same narrow
            # bin region during the final descent (e.g. milk->bin_back_middle and
            # soda_can->bin_front_right in run_24).
            if len(place_slots) == len(llm_plan.action_strs):
                alice_slot = place_slots.get("Alice")
                bob_slot = place_slots.get("Bob")
                if (
                    alice_slot in self.bin_slot_xposes
                    and bob_slot in self.bin_slot_xposes
                ):
                    slot_dist = np.linalg.norm(
                        self.bin_slot_xposes[alice_slot][:2]
                        - self.bin_slot_xposes[bob_slot][:2]
                    )
                    if slot_dist < 0.35:
                        feedback += (
                            f"Simultaneous PLACE target slots are too close: "
                            f"Alice->{alice_slot} and Bob->{bob_slot} are only "
                            f"{slot_dist:.2f} apart in XY. Choose non-adjacent, "
                            "distant empty slots with XY separation at least 0.35. "
                            "Avoid close pairs such as bin_back_middle with "
                            "bin_front_right; prefer opposite-side pairs such as "
                            "Alice->bin_back_left and Bob->bin_front_right when "
                            "both slots are empty. "
                        )

            alice_waypoints = llm_plan.ee_waypoint_poses.get("Alice", [])
            bob_waypoints = llm_plan.ee_waypoint_poses.get("Bob", [])
            alice_mid_positions = np.array(
                [pose.position for pose in alice_waypoints[1:-1]]
            ) if len(alice_waypoints) > 2 else np.empty((0, 3))
            bob_mid_positions = np.array(
                [pose.position for pose in bob_waypoints[1:-1]]
            ) if len(bob_waypoints) > 2 else np.empty((0, 3))

            if len(alice_mid_positions) > 0:
                ax = alice_mid_positions[:, 0]
                ay = alice_mid_positions[:, 1]
                az = alice_mid_positions[:, 2]
                if np.any(az < 0.58):
                    feedback += (
                        "Alice's PLACE path is not lifted high enough for "
                        "simultaneous PLACE. Alice should use middle waypoints "
                        "at z about 0.60-0.68 before the final slot approach. "
                    )
                if np.any((ax > 0.15) | (ay > 0.52)):
                    feedback += (
                        "Alice's simultaneous PLACE middle waypoints are too "
                        "central/right or too far back. Alice should stay in a "
                        "high left/front corridor with middle waypoints around "
                        "x<=0.15, y<=0.52, z=0.60-0.68, and only approach the "
                        "target slot at the last waypoint. "
                    )

            if len(bob_mid_positions) > 0:
                x = bob_mid_positions[:, 0]
                y = bob_mid_positions[:, 1]
                z = bob_mid_positions[:, 2]
                low_middle_mask = (
                    (x >= 0.0) & (x <= 0.35) &
                    (y >= 0.45) & (y <= 0.58) &
                    (z < 0.55)
                )
                if np.any(low_middle_mask):
                    feedback += (
                        "Bob's PLACE path uses a low middle-table waypoint, "
                        "which often makes RRT planning hang. For simultaneous "
                        "PLACE actions, Bob must route through a high back/right "
                        "corridor: use middle waypoints with y around 0.58-0.68 "
                        "and z around 0.60-0.65, e.g. avoid (0.20,0.50,0.40) "
                        "and use (0.25,0.62,0.62) or (0.45,0.62,0.62). "
                    )

                if np.max(z) < 0.58:
                    feedback += (
                        "Bob's PLACE path is not lifted high enough. During "
                        "simultaneous PLACE, Bob should lift the held object to "
                        "z about 0.60-0.65 before moving toward the bin. "
                    )
                if np.any(y < 0.60):
                    feedback += (
                        "Bob's simultaneous PLACE middle waypoints are too "
                        "central/front. Bob should stay in a high back/right "
                        "corridor with middle waypoints around y=0.60-0.70 and "
                        "z=0.60-0.68 before the final descent, e.g. "
                        "(0.45,0.64,0.64) or (0.60,0.64,0.64). "
                    )

            if len(alice_mid_positions) > 0 and len(bob_mid_positions) > 0:
                # Only compare same-time middle waypoints.  Comparing every
                # Alice waypoint against every Bob waypoint is overly
                # conservative and rejects otherwise usable plans when the
                # two robots pass through nearby areas at different times.
                num_pairs = min(len(alice_mid_positions), len(bob_mid_positions))
                same_step_xy_dists = np.linalg.norm(
                    alice_mid_positions[:num_pairs, :2] - bob_mid_positions[:num_pairs, :2],
                    axis=-1,
                )
                min_xy_dist = float(np.min(same_step_xy_dists))
                if min_xy_dist < 0.25:
                    feedback += (
                        f"Alice and Bob's simultaneous PLACE corridors are too "
                        f"close at the same waypoint step before final descent: "
                        f"minimum same-step middle-waypoint XY "
                        f"separation is {min_xy_dist:.2f}. Keep Alice's and "
                        "Bob's same-step middle waypoints about 0.25 or more "
                        "apart in XY; Alice "
                        "should use the high left/front corridor and Bob the high "
                        "back/right corridor until the final descent. "
                    )
        return feedback
 
 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image 
    env = PackGroceryTask()
    obs = env.reset()
    print(env.describe_obs(obs))
    print(env.get_agent_prompt(obs, "Alice"))
    print(env.get_agent_prompt(obs, "Bob"))
    breakpoint()
    print(obs.ur5e_robotiq.ee_xquat)
    img=env.physics.render(camera_id="teaser", height=480, width=600)
    im = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
    breakpoint()
