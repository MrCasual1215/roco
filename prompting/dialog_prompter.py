import os 
import time
import json
import pickle 
import openai
import requests
import numpy as np
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any
from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState 
from .feedback import FeedbackManager
from .parser import LLMResponseParser

def _query_openai_compatible_chat(
    model: str,
    system_prompt: str,
    user_prompt: str = "",
    temperature: float = 0,
    max_tokens: int = 512,
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
    # Keep a non-empty user turn for local OpenAI-compatible chat backends.
    # Some Ollama/Llama chat templates return an empty assistant message when
    # prompted with only a system message.
    messages.append({
        "role": "user",
        "content": user_prompt or (
            "Please follow the instructions above and output the next response now."
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
[Path Plan Instruction]
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

class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent 
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512, 
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3, 
        max_calls_per_round: int = 10,
        use_history: bool = True,  
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round 
        self.temperature = temperature
        self.llm_source = llm_source

    def compose_system_prompt(
        self, 
        obs: EnvState, 
        agent_name: str,
        chat_history: List = [], # chat from previous replan rounds
        current_chat: List = [],  # chat from current round, this comes AFTER env feedback 
        feedback_history: List = []
    ) -> str:
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt(obs, agent_name)
        
        round_history = self.get_round_history() if self.use_history else ""

        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"

        chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""
            
        system_prompt = f"{action_desp}\n{round_history}\n{execute_feedback}{agent_prompt}\n{chat_history}\n" 
        
        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)
        
        if len(current_chat) > 0:
            system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"

        return system_prompt 

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def compose_output_verifier_system_prompt(
        self,
        obs: EnvState,
        candidate_response: str,
        feedback_history: List[str] = [],
    ) -> str:
        """Compose a second-stage verifier prompt for dialog-mode output.

        Dialog mode often lets the last speaker summarize other agents' proposals.
        This verifier is a lightweight final guard: it must return only an
        executable plan, correcting format mistakes and obvious task/reachability
        mistakes before the normal parser + environment feedback run.
        """
        task_desp = self.env.describe_task_context()
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        obs_desp = self.env.describe_obs(obs)
        round_history = self.get_round_history() if self.use_history else ""
        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute; do not repeat the same failure:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"

        feedback_prompt = ""
        useful_feedback = [f for f in feedback_history if f and f != "None"]
        if len(useful_feedback) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(useful_feedback) + "\n"

        extra_sort_rules = ""
        if self.env.__class__.__name__ == "SortOneBlockTask":
            extra_sort_rules = """
[Extra SortOneBlock Verification Rules]
- Current cube locations in [Scene description] are the only source of truth.
- Robot panel reachability is fixed: Alice can only use panel1/panel2/panel3; Bob panel3/panel4/panel5; Chad panel5/panel6/panel7.
- All robot actions in one EXECUTE plan are simultaneous. A robot may PICK an object only if that object is already on a panel the robot can reach at the start of this round.
- Valid PLACE panels are only each cube's target panel or handoff panels panel3/panel5:
  blue_square -> panel2, panel3, or panel5
  pink_polygon -> panel4, panel3, or panel5
  yellow_trapezoid -> panel6, panel3, or panel5
- Never PLACE any cube on panel1 or panel7.
"""

        return f"""
{task_desp}
{action_desp}
{round_history}
{execute_feedback}
{obs_desp}
{feedback_prompt}
[Candidate Dialog Output To Verify]
{candidate_response}

[Verifier Rules]
You are a strict output verifier/corrector.
Silently check the candidate against the current scene, action format, task constraints, robot reachability, and previous feedback.
If the candidate is valid, output it unchanged.
If it is invalid, output a corrected plan. Prefer changing only invalid robot actions to WAIT while preserving valid actions that make progress. If all actions would be WAIT, choose one valid progress action from the current scene.
Return ONLY the executable plan in the required EXECUTE/NAME/ACTION format; no explanations, markdown, or chat.
{extra_sort_rules}
""".strip()

    def compose_output_verifier_user_prompt(self) -> str:
        required_lines = "\n".join(
            [f"NAME {agent_name} ACTION <one valid action>" for agent_name in self.robot_agent_names]
        )
        return f"""
Verify and, if needed, correct the candidate dialog output.
Return ONLY:
EXECUTE
{required_lines}
""".strip()

    def verify_output_plan(
        self,
        obs: EnvState,
        candidate_response: str,
        feedback_history: List[str],
        save_path: str,
        replan_idx: int,
    ) -> str:
        if self.debug_mode or candidate_response is None:
            return candidate_response

        system_prompt = self.compose_output_verifier_system_prompt(
            obs=obs,
            candidate_response=candidate_response,
            feedback_history=feedback_history,
        )
        user_prompt = self.compose_output_verifier_user_prompt()
        verifier_response, usage = self.query_once(
            system_prompt,
            user_prompt=user_prompt,
            max_query=3,
        )

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
                "sender": "CandidateDialog",
                "message": candidate_response,
            },
            {
                "sender": "Verifier",
                "message": verifier_response,
            },
            usage,
        ]
        fname = f'{save_path}/replan{replan_idx}_output_verifier_{timestamp}.json'
        json.dump(tosave, open(fname, 'w'))

        # Keep a malformed verifier from destroying an otherwise parseable plan;
        # parser + environment feedback still provide the hard validation layer.
        if verifier_response and "EXECUTE" in verifier_response:
            return verifier_response
        return candidate_response
    
    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        chat_history = [] 
        for i in range(self.num_replans):
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            chat_history += agent_responses
            candidate_response = final_response
            final_response = self.verify_output_plan(
                obs=obs,
                candidate_response=candidate_response,
                feedback_history=plan_feedbacks,
                save_path=save_path,
                replan_idx=i,
            )
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response) 

            curr_feedback = "None"
            if not parse_succ:  
                curr_feedback = f"""
This previous response from [{final_agent}] failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False  
            
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute: 
                break  
            else:
                print(curr_feedback)
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history
   
    def prompt_one_dialog_round(
        self, 
        obs, 
        chat_history, 
        feedback_history, 
        replan_idx=0,
        save_path='data/',
        ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached
        """
        
        agent_responses = []
        usages = []
        dialog_done = False 
        num_responses = {agent_name: 0 for agent_name in self.robot_agent_names}
        n_calls = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs, 
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,   
                    ) 
                
                agent_prompt = f"You are {agent_name}, your response is:"
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = f"""
You are {agent_name}, this is the last call, you must end your response by incoporating all previous discussions and output the best plan via EXECUTE. 
Your response is:
                    """
                response, usage = self.query_once(
                    system_prompt, 
                    user_prompt=agent_prompt, 
                    max_query=3,
                    )
                
                tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response,
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))  

                num_responses[agent_name] += 1
                # strip all the repeated \n and blank spaces in response: 
                pruned_response = response.strip()
                # pruned_response = pruned_response.replace("\n", " ")
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                    )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break
 
                if self.debug_mode:
                    dialog_done = True
                    break
            
            if dialog_done:
                break
 
        # response = "\n".join(response.split("EXECUTE")[1:])
        # print(response)  
        return agent_name, response, agent_responses

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None   
        print('======= system prompt ======= \n ', system_prompt)
        print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: 
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
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
                time.sleep(2)
            continue
        # breakpoint()
        return response, usage
    
    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
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
        self.latest_chat_history = []
