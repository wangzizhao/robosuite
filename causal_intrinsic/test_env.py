from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

import random
import pickle

if __name__ == "__main__":
    render = True

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    options["env_name"] = "Causal1"
    options["robots"] = "UR5e"
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=4,
        horizon=250,
    )
    obs = env.reset()
    for k, v in obs.items():
        if k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cubeA_pos"]:
            print(k)
            print(v)
    print()

    done = False
    if render:
        env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    print(low)
    print(high)

    # for record
    obses = {key: [] for key in obs}
    actions = []
    dones = []
    next_obses = {key: [] for key in obs}

    # objs = [env.cubeA, env.cubeB, env.cubeC, env.cubeD, env.cubeE, env.ballA, env.ballB, env.cylinderA, env.cylinderB]
    objs = [env.cubeA, env.cubeC, env.cubeE]
    # objs = [env.cubeA]
    object_names = [obj.name for obj in objs]
    for e in range(500):
        obj_to_pick_name = random.choice(object_names)
        obj_to_pick = getattr(env, obj_to_pick_name)
        goal = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(1.0, 1.2)])

        print("\nGoing to pick up", "red" if "A" in obj_to_pick_name else "green", obj_to_pick_name)
        print("Then lift to position", goal)
        start_grasp = False
        picked = False
        n_steps = 0
        while not done:
            eef_pos = obs["robot0_eef_pos"]
            object_pos = obs[obj_to_pick_name + "_pos"]
            action = np.zeros_like(low)
            action[-1] = -1
            if not picked:
                placement = object_pos - eef_pos
                xy_place, z_place = placement[:2], placement[-1]

                if not start_grasp:
                    if np.linalg.norm(xy_place) >= 0.02:
                        action[:2] = xy_place
                    elif np.abs(z_place) >= 0.02:
                        action[2] = z_place
                    elif np.linalg.norm(placement) >= 0.01:
                        action[:3] = placement
                    else:
                        start_grasp = True

                if start_grasp:
                    action[-1] = 1

                    grasped = int(env._check_grasp(
                            gripper=env.robots[0].gripper,
                            object_geoms=[g for g in obj_to_pick.contact_geoms])
                        )
                    print("try to grasp, success:", bool(grasped))
                    if grasped:
                        picked = True

                action[:3] = np.clip(action, low, high)[:3]
            else:
                if np.linalg.norm(goal - eef_pos) >= 0.02:
                    action[:3] = goal - eef_pos
                    action[-1] = 1
                    action = np.clip(action, low, high)
                elif object_pos[2] >= 0.85:
                    pass
                else:
                    done = True

            # action = np.random.uniform(low, high)

            next_obs, reward, terminate, _ = env.step(action)
            if render:
                env.render()

            if terminate:
                done = True

            action = np.concatenate([next_obs["robot0_eef_pos"] - obs["robot0_eef_pos"],
                                     next_obs["robot0_gripper_qpos"] - obs["robot0_gripper_qpos"]])
            actions.append(action.copy())
            for key in obs.keys():
                obses[key].append(obs[key].copy())
                next_obses[key].append(next_obs[key].copy())
            dones.append(done)

            obs = next_obs

            n_steps += 1
            if n_steps % 10 == 0:
                print(n_steps, "steps")

        obs = env.reset()
        done = False

        if (e + 1) % 10 == 0:
            data = {"obs": obses,
                    "action": actions,
                    "done": dones,
                    "next_obs": next_obses}
            with open("data.p", "wb") as f:
                pickle.dump(data, f)

    data = {"obs": obses,
            "action": actions,
            "done": dones,
            "next_obs": next_obses}
    with open("data.p", "wb") as f:
        pickle.dump(data, f)

